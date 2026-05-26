#!/usr/bin/env python3
"""Standalone runner for the video clipping agent pipeline.

Stages:
    1a  Preprocess         — AWS Transcribe JSON → sentences + language
    1b  Category Detect    — transcript sample → category, audience, weights
    1c  Scene Detect       — video frames + category → scene map (skipped if
                             frames unavailable or category doesn't benefit)
    2   Segment            — sentences → candidate windows (scene-boundary-aware)
    3a  Sequential         — Agent #1 selects windows + Agent #3 reviews
    3b  Non-Sequential     — Agent #2 assembles sentences + Agent #4 reviews
    4   Final Metadata     — Agent #5 generates titles + platform tags
    5   Ranking            — sort and return

Changes from v3:
    FIX 1 — used_window_ids is now a SINGLE shared set across the entire
             sequential pipeline (initial pick + all review iterations).
             Previously two separate sets — reviewer had no knowledge of
             what the selector already picked.

    FIX 2 — Used windows are physically REMOVED from the window list before
             every LLM call. Previously only mentioned in prompt text, which
             the LLM ignored when the banned window scored highest.
             Result: LLM literally cannot pick a used window — it is not there.

    FIX 3 — Dynamic TARGET_SCORE. After the first clip is scored, the target
             is calibrated to what this video can actually achieve.
             Formula: max(MIN_QUALITY_FLOOR, best_score_seen * 0.92).
             Prevents infinite regeneration on videos that can't reach 90.

    FIX 4 — Post-review quality filter. Clips below MIN_QUALITY_FLOOR (75)
             are dropped from the final output entirely.
             Previously sub-75 clips (e.g. nonseq_4 at 58) appeared in result.json.

    FIX 5 — Token cost. Regeneration calls receive only unused windows,
             not the full 177-window list. For a 27-min video this drops
             regeneration calls from ~101,500 tokens to ~5,000 tokens.

    FIX 6 — Duplicate clip guard. After review, any clip whose time range
             already exists in the output is dropped before writing result.json.

Quick start:
    pip install anthropic
    export CLAUDE_API_KEY="your-key"
    python lfts.py

CLI:
    python lfts.py --transcript /path/to/transcript.json \\
                   --video /path/to/video.mp4 \\
                   --output result.json \\
                   --clip-mode both
"""

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Set

from anthropic import Anthropic


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSCRIPT_PATH = os.path.join(_SCRIPT_DIR, "transcript.json")
DEFAULT_VIDEO_PATH      = os.path.join(_SCRIPT_DIR, "input.mp4")
DEFAULT_OUTPUT_PATH     = os.path.join(_SCRIPT_DIR, "result.json")


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("clipping_agent")


# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE_API_KEY    = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL      = "claude-haiku-4-5-20251001"
CLAUDE_MAX_TOKENS = 8192

SEQUENTIAL_MIN_DURATION     = 25
SEQUENTIAL_MAX_DURATION     = 55
NON_SEQUENTIAL_MIN_DURATION = 25
NON_SEQUENTIAL_MAX_DURATION = 90

# FIX 3: dynamic target — this is now a FLOOR, not the fixed target
# actual target is calibrated per-video after first clip scored
MIN_QUALITY_FLOOR        = 75   # absolute minimum — below this is dropped
INITIAL_TARGET_SCORE     = 90   # starting assumption before calibration
MAX_ITERATIONS_PER_SHORT = 3

CATEGORY_SAMPLE_SENTENCES = 60
FRAME_INTERVAL = 10
MAX_FRAMES     = 60

FILLER_ENDINGS = {
    "great.", "okay.", "ok.", "right.", "all right.", "alright.",
    "yeah.", "yes.", "good.", "sure.", "nice.", "cool.", "perfect.",
    "wonderful.", "excellent.", "fantastic.", "awesome.",
}

SKIP_SCENE_DETECTION_CATEGORIES = {
    "podcast", "audio only", "static presentation",
    "animated explainer", "screencast", "audio recording",
}

DEFAULT_DIMENSION_WEIGHTS = {
    "hook_strength":       0.25,
    "reframe_insight":     0.20,
    "emotional_resonance": 0.18,
    "standalone_clarity":  0.17,
    "quotability":         0.12,
    "clean_ending":        0.08,
}


# ── Prompts ───────────────────────────────────────────────────────────────────

CATEGORY_WEIGHT_PROMPT = """
You are a social media content strategist and audience psychologist.

Read the transcript excerpt below and determine:

1. What kind of video this is — not just a label, but the specific
   audience it serves and what they came to feel or gain from it.

2. Based on that audience expectation, assign weights to the 6 scoring
   dimensions. Weights must sum to exactly 1.0. Distribute them based
   on what this specific audience rewards most — not a generic formula.

## The 6 dimensions

- hook_strength: Does the opening grab and orient the viewer instantly?
- reframe_insight: Does it offer a fresh idea or perspective?
- emotional_resonance: Does it make the viewer feel something on its own?
- standalone_clarity: Can a new viewer follow it without the full video?
- quotability: Is there a line they would repeat or share?
- clean_ending: Does it land at a peak — punchline, resolution, takeaway?

## Reasoning

Ask: if someone scrolls past in 2 seconds — what dimension failed them?
That gets the highest weight. What dimension barely matters for this
audience? That gets the lowest.

## Transcript excerpt
{transcript_text}
"""

CATEGORY_WEIGHT_RESPONSE_FORMAT = {
    "category":             "string (2-4 words specific)",
    "audience":             "string (1 sentence — who watches this and why)",
    "weights": {
        "hook_strength":       "float",
        "reframe_insight":     "float",
        "emotional_resonance": "float",
        "standalone_clarity":  "float",
        "quotability":         "float",
        "clean_ending":        "float",
    },
    "weight_reasoning":        "string (2-3 sentences why these weights fit)",
    "benefits_from_frames":    "boolean",
    "video_grammar":           "string (how visual cuts/transitions work for this type)",
}


SCENE_DETECT_PROMPT = """
You are analysing frames extracted from a video to identify scene
boundaries and clip-worthy segments.

Video category: {category}
Audience: {audience}
Visual grammar for this category: {video_grammar}

Frames are provided in order. Frame N was captured at approximately
{interval} × N seconds into the video.
Total frames provided: {frame_count}

Detect meaningful visual transitions that affect clip boundaries.
For embedded videos: mark start and end precisely — these are often
the most clip-worthy moments. Mark confidence honestly.

IMPORTANT: For start_time_seconds and end_time_seconds, provide the
most precise timestamp you can estimate based on the frame content and
its position in the sequence. Do NOT round to nearest 20 or 30 seconds
— interpolate based on what you see changing between frames.
"""

SCENE_DETECT_RESPONSE_FORMAT = {
    "scenes": [{
        "start_time_seconds":  "number (precise, not rounded to 20s intervals)",
        "end_time_seconds":    "number or null",
        "scene_type":          "presenter | embedded_video | screen_share | interview_guest | broll | demo | audience | unknown",
        "confidence":          "high | medium | low",
        "visual_evidence":     "string",
        "clip_worthy":         "boolean",
        "clip_reason":         "string or null",
        "speaker_name":        "string or null",
    }]
}


SEQUENCE_PROMPT = """
{feedback_section}

## Video profile
Category: {category}
Audience: {audience}
What this audience rewards most: {weight_reasoning}

You are a pro-level video editor and content strategist. Read the
transcript windows below and find the best shareable clips for
YouTube, Instagram, and TikTok.

Scan through 6 dimensions. A window scoring 75+ becomes a clip.
Below 75 — keep scanning. Quality always beats quantity.

Hook Strength (25 points) — Opening must grab attention and make
complete sense without prior context. Never start with uh/um/ah/hmm.

Reframe / Insight (20 points) — Must offer a perspective that makes
the viewer think differently. Not preamble — the core delivery.

Emotional Resonance (18 points) — Must make the viewer feel something
on its own without the full video.

Standalone Clarity (17 points) — Viewer must understand what, why,
and how it ends without the full video. Extend if anything is missing.

Quotability (12 points) — A line someone would repeat or share.

Clean Ending (8 points) — Ends at a peak. Never on "Great.", "Okay.",
"Right.", "Yeah." — these are transition words, not endings.

Keep clips 30–90 seconds. Duration is a byproduct of the story.

Select {clips_count} DIFFERENT windows. Every window you pick must have
a DIFFERENT WindowId — never repeat a WindowId in your response.

Available windows:
{windows_text}
"""

SEQUENCE_RESPONSE_FORMAT = {
    "clips": [{"WindowId": "number", "reason": "string"}]
}


NONSEQUENCE_PROMPT = """
{feedback_section}

## Video profile
Category: {category}
Audience: {audience}
What this audience rewards most: {weight_reasoning}

You are a pro-level video editor. Assemble clips by combining
non-contiguous moments from the transcript. Decide the story first —
then find the moments that build it.

Hook Strength (25 points) — Strongest opening from anywhere. Never filler.
Reframe / Insight (20 points) — Every segment serves the core idea directly.
Emotional Resonance (18 points) — Whole clip makes viewer feel something.
Standalone Clarity (17 points) — New viewer understands without full video.
Quotability (12 points) — A line someone would repeat or share.
Clean Ending (8 points) — Strongest possible close. Never filler endings.

IMPORTANT: Total duration of selected sentences must be 30–90 seconds.

{sentences_text}
"""

NONSEQUENCE_RESPONSE_FORMAT = {
    "shorts": [{
        "topic":        "string",
        "sentence_ids": ["number"],
        "reason":       "string",
    }]
}


SCORER_PROMPT = """
You are a category-aware social media viewer and analyst.

## Clip to score
Category: {category}
Audience: {audience}
Type: {clip_type}
Duration: {duration}s
Text: "{text_preview}"

## What this audience rewards most
{weight_reasoning}

## Scoring — 6 dimensions, 0 to 20 each

Score through the lens of this specific audience.

1. hook_strength: Opening grabs viewer, makes sense without context?
2. reframe_insight: Fresh perspective or revelation worth thinking about?
3. emotional_resonance: Makes viewer feel something on its own?
4. standalone_clarity: New viewer understands who/what/why/ending?
   Penalise broken references ("he", "that", "as I said").
5. quotability: A line someone would repeat or share?
6. clean_ending: Ends at a peak? "Great.", "Okay.", "Right.", "Yeah."
   are transition words — max score 3 if clip ends on these.

Return fair, context-aware JSON.
"""

SCORER_RESPONSE_FORMAT = {
    "hook_strength":        "integer 0-20",
    "reframe_insight":      "integer 0-20",
    "emotional_resonance":  "integer 0-20",
    "standalone_clarity":   "integer 0-20",
    "quotability":          "integer 0-20",
    "clean_ending":         "integer 0-20",
    "reason":               "string",
    "weakest_factor":       "string",
    "improvise":            "string",
    "recommended_end_trim": "string or null",
}


METADATA_PROMPT = """
You are a social media strategist. Generate titles and platform
recommendations for {short_count} video shorts.

Language: {language_code} — ALL titles must be in the native language.

Title: 6-8 words, punchy, value-focused.
Platforms (2-3 per short): YouTube, Instagram, TikTok, LinkedIn.

{shorts_context}
"""

METADATA_RESPONSE_FORMAT = {
    "shorts": [{
        "short_id":  "string",
        "title":     "string",
        "platforms": ["string"],
    }]
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    index:  int
    start:  float
    end:    float
    text:   str
    source: str = "presenter"


@dataclass
class CandidateWindow:
    id:         int
    start_s:    int
    end_s:      int
    start_time: float
    end_time:   float
    duration:   float
    text:       str
    source:     str = "presenter"


@dataclass
class Scene:
    start_time:      float
    end_time:        Optional[float]
    scene_type:      str
    confidence:      str
    visual_evidence: str
    clip_worthy:     bool
    clip_reason:     Optional[str]
    speaker_name:    Optional[str] = None


# ── LLM client ────────────────────────────────────────────────────────────────

_client = None
_token_usage = {
    "total_input_tokens":  0,
    "total_output_tokens": 0,
    "calls":               [],
}


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def _track_usage(caller, input_tokens, output_tokens):
    _token_usage["total_input_tokens"]  += input_tokens
    _token_usage["total_output_tokens"] += output_tokens
    _token_usage["calls"].append({
        "caller":        caller,
        "model":         CLAUDE_MODEL,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    })
    log.info(
        f"[LLM] {caller} — in:{input_tokens} out:{output_tokens} "
        f"total:{_token_usage['total_input_tokens']+_token_usage['total_output_tokens']}"
    )


def get_token_usage():
    return {
        "total_input_tokens":  _token_usage["total_input_tokens"],
        "total_output_tokens": _token_usage["total_output_tokens"],
        "total_tokens":        _token_usage["total_input_tokens"] + _token_usage["total_output_tokens"],
        "llm_calls":           len(_token_usage["calls"]),
        "calls":               _token_usage["calls"],
    }


def _build_prompt(prompt, response_format=None):
    if not response_format:
        return prompt
    return (
        f"{prompt}\n\n"
        "Return only valid JSON. No markdown, no code fences.\n"
        f"{json.dumps(response_format, ensure_ascii=False, indent=2)}"
    )


def _parse_json_response(text):
    try:
        return json.loads(text)
    except ValueError:
        pass
    for fence in ("```json", "```"):
        if fence in text:
            for part in text.split(fence)[1:]:
                try:
                    return json.loads(part.split("```")[0].strip())
                except ValueError:
                    continue
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except ValueError:
            pass
    if first != -1:
        repaired = _repair_truncated_json(text[first:])
        if repaired is not None:
            return repaired
    log.error(f"JSON parse failed. Start: {text[:200]}")
    raise RuntimeError("LLM returned invalid JSON")


def _repair_truncated_json(text):
    for end_pos in reversed([m.end() for m in re.finditer(r'\}\s*,?\s*', text)]):
        snippet = text[:end_pos].rstrip().rstrip(",")
        ob  = snippet.count("{") - snippet.count("}")
        ob2 = snippet.count("[") - snippet.count("]")
        if ob < 0 or ob2 < 0:
            continue
        try:
            result = json.loads(snippet + "]" * ob2 + "}" * ob)
            if isinstance(result, dict):
                log.warning(f"Repaired truncated JSON at {end_pos}/{len(text)}")
                return result
        except ValueError:
            continue
    return None


def get_response(prompt, system_prompt=None, temperature=0.3,
                 response_format=None, max_tokens=None, caller="unknown"):
    client   = _get_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens or CLAUDE_MAX_TOKENS,
        system=system_prompt or "You are a helpful assistant.",
        temperature=temperature,
        messages=[{"role": "user", "content": _build_prompt(prompt, response_format)}],
    )
    _track_usage(caller, response.usage.input_tokens, response.usage.output_tokens)
    text = response.content[0].text.strip()
    if not text:
        raise RuntimeError("Claude returned empty response")
    return _parse_json_response(text)


def get_multimodal_response(prompt, frames_b64, temperature=0.3,
                            response_format=None, max_tokens=None, caller="unknown"):
    if not frames_b64:
        return get_response(prompt, temperature=temperature,
                            response_format=response_format,
                            max_tokens=max_tokens, caller=caller)
    client  = _get_client()
    content = [
        {"type": "image", "source": {"type": "base64",
                                      "media_type": "image/jpeg",
                                      "data": f}}
        for f in frames_b64
    ]
    content.append({"type": "text", "text": _build_prompt(prompt, response_format)})
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens or CLAUDE_MAX_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
    )
    _track_usage(caller, response.usage.input_tokens, response.usage.output_tokens)
    text = response.content[0].text.strip()
    if not text:
        raise RuntimeError("Claude multimodal returned empty response")
    return _parse_json_response(text)


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames_from_file(video_path):
    if not video_path or not os.path.isfile(video_path):
        log.warning(f"[FRAMES] Video not found: {video_path}")
        return []
    temp_dir = tempfile.mkdtemp()
    try:
        pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps=1/{FRAME_INTERVAL}",
            "-q:v", "2", "-frames:v", str(MAX_FRAMES), pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        except FileNotFoundError:
            log.error("[FRAMES] ffmpeg not found — skipping")
            return []
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.error(f"[FRAMES] ffmpeg error: {repr(e)}")
            return []
        files = sorted(
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )[:MAX_FRAMES]
        encoded = []
        for p in files:
            try:
                with open(p, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode())
            except Exception as e:
                log.error(f"[FRAMES] encode error {p}: {repr(e)}")
        log.info(f"[FRAMES] Extracted {len(encoded)} frames (every {FRAME_INTERVAL}s)")
        return encoded
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Stage 1a: Preprocess ──────────────────────────────────────────────────────

def preprocess(transcription_data):
    items = transcription_data.get("items", [])
    log.info(f"[PREPROCESS] Items: {len(items)}")

    sentences     = []
    current_words = []
    current_start = None
    current_end   = None

    for item in items:
        t = item.get("type")
        if t == "pronunciation":
            word  = item["alternatives"][0]["content"]
            start = float(item.get("start_time", 0))
            end   = float(item.get("end_time",   0))
            if current_start is None:
                current_start = start
            current_end = end
            current_words.append(word)
        elif t == "punctuation":
            punct = item["alternatives"][0]["content"]
            if current_words:
                current_words[-1] += punct
            if punct in (".", "?", "!"):
                if current_words and current_start is not None:
                    sentences.append(Sentence(
                        index=len(sentences) + 1,
                        start=round(current_start, 2),
                        end=round(current_end, 2),
                        text=" ".join(current_words),
                    ))
                current_words, current_start, current_end = [], None, None

    if current_words and current_start is not None:
        sentences.append(Sentence(
            index=len(sentences) + 1,
            start=round(current_start, 2),
            end=round(current_end, 2),
            text=" ".join(current_words),
        ))

    video_duration = 0.0
    for item in items:
        et = item.get("end_time")
        if et:
            video_duration = max(video_duration, float(et))

    language_code = "en-US"
    lang_id = transcription_data.get("language_identification")
    if lang_id and isinstance(lang_id, list) and lang_id:
        language_code = lang_id[0].get("code", "en-US")
    elif transcription_data.get("language_code"):
        language_code = transcription_data["language_code"]

    log.info(f"[PREPROCESS] {len(sentences)} sentences, {video_duration:.1f}s, lang={language_code}")
    return sentences, video_duration, language_code


# ── Stage 1b: Category + weight detection ────────────────────────────────────

def detect_category_weights(sentences):
    sample          = sentences[:CATEGORY_SAMPLE_SENTENCES]
    transcript_text = " ".join(s.text for s in sample)

    try:
        result = get_response(
            CATEGORY_WEIGHT_PROMPT.format(transcript_text=transcript_text),
            response_format=CATEGORY_WEIGHT_RESPONSE_FORMAT,
            caller="category_detector",
        )
    except Exception as e:
        log.error(f"[CATEGORY] Detection failed: {repr(e)} — using defaults")
        return _default_category_profile()

    weights = result.get("weights", {})
    required = set(DEFAULT_DIMENSION_WEIGHTS.keys())
    if not required.issubset(set(weights.keys())):
        log.warning("[CATEGORY] Incomplete weights — using defaults")
        return _default_category_profile()

    try:
        weights = {k: float(v) for k, v in weights.items()}
    except (ValueError, TypeError):
        return _default_category_profile()

    total = sum(weights.values())
    if total <= 0:
        return _default_category_profile()
    if abs(total - 1.0) > 0.01:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        log.warning(f"[CATEGORY] Weights renormalised from {total:.3f} to 1.0")

    profile = {
        "category":             result.get("category",           "general"),
        "audience":             result.get("audience",           "general audience"),
        "weights":              weights,
        "weight_reasoning":     result.get("weight_reasoning",   ""),
        "benefits_from_frames": bool(result.get("benefits_from_frames", True)),
        "video_grammar":        result.get("video_grammar",      "standard cuts"),
    }
    log.info(f"[CATEGORY] '{profile['category']}' — top: {max(weights, key=weights.get)} ({max(weights.values()):.2f})")
    return profile


def _default_category_profile():
    return {
        "category":             "general",
        "audience":             "general audience",
        "weights":              DEFAULT_DIMENSION_WEIGHTS.copy(),
        "weight_reasoning":     "Default weights applied.",
        "benefits_from_frames": True,
        "video_grammar":        "standard cuts and transitions",
    }


def _should_run_scene_detection(category_profile, frames_b64):
    if not frames_b64:
        return False
    if not category_profile.get("benefits_from_frames", True):
        log.info(f"[SCENE] Skipping — LLM says no benefit for this category")
        return False
    cat_lower = category_profile["category"].lower()
    for skip_cat in SKIP_SCENE_DETECTION_CATEGORIES:
        if skip_cat in cat_lower:
            log.info(f"[SCENE] Skipping — known no-benefit category")
            return False
    return True


# ── Stage 1c: Scene detection ─────────────────────────────────────────────────

def detect_scenes(frames_b64, category_profile):
    if not _should_run_scene_detection(category_profile, frames_b64):
        return []

    MAX_SCENE_FRAMES = 30
    if len(frames_b64) > MAX_SCENE_FRAMES:
        step    = len(frames_b64) // MAX_SCENE_FRAMES
        sampled = frames_b64[::step][:MAX_SCENE_FRAMES]
    else:
        sampled = frames_b64

    effective_interval = FRAME_INTERVAL * (len(frames_b64) / max(len(sampled), 1))

    prompt = SCENE_DETECT_PROMPT.format(
        category=category_profile["category"],
        audience=category_profile["audience"],
        video_grammar=category_profile["video_grammar"],
        interval=round(effective_interval, 1),
        frame_count=len(sampled),
    )

    try:
        result = get_multimodal_response(
            prompt, sampled,
            response_format=SCENE_DETECT_RESPONSE_FORMAT,
            caller="scene_detector",
        )
    except Exception as e:
        log.error(f"[SCENE] Detection failed: {repr(e)}")
        return []

    scenes = []
    for raw in result.get("scenes", []):
        try:
            scenes.append(Scene(
                start_time=float(raw.get("start_time_seconds", 0)),
                end_time=float(raw["end_time_seconds"]) if raw.get("end_time_seconds") else None,
                scene_type=raw.get("scene_type", "unknown"),
                confidence=raw.get("confidence", "low"),
                visual_evidence=raw.get("visual_evidence", ""),
                clip_worthy=bool(raw.get("clip_worthy", False)),
                clip_reason=raw.get("clip_reason"),
                speaker_name=raw.get("speaker_name"),
            ))
        except (KeyError, ValueError, TypeError) as e:
            log.warning(f"[SCENE] Skipping malformed scene: {repr(e)}")
            continue

    scenes.sort(key=lambda s: s.start_time)
    for i, scene in enumerate(scenes[:-1]):
        if scene.end_time is None:
            scene.end_time = scenes[i + 1].start_time

    log.info(f"[SCENE] Detected {len(scenes)} scenes")
    for sc in scenes:
        end_str = f"{sc.end_time:.1f}s" if sc.end_time else "?"
        log.info(
            f"  [{sc.start_time:.1f}s-{end_str}] {sc.scene_type} "
            f"conf={sc.confidence} clip_worthy={sc.clip_worthy}"
        )
    return scenes


def tag_sentences_by_scene(sentences, scenes):
    if not scenes:
        return sentences
    for sentence in sentences:
        assigned = "presenter"
        midpoint = (sentence.start + sentence.end) / 2
        for scene in scenes:
            s_end = scene.end_time or float("inf")
            if scene.start_time <= midpoint <= s_end:
                assigned = scene.scene_type
                break
        sentence.source = assigned
    return sentences


def inject_scene_clips(scenes, sentences):
    injected = []
    for scene in scenes:
        if not scene.clip_worthy or scene.end_time is None:
            continue

        scene_sents = [
            s for s in sentences
            if s.start >= scene.start_time
            and s.end   <= (scene.end_time + 2.0)
            and s.source == scene.scene_type
        ]
        if not scene_sents:
            scene_sents = [
                s for s in sentences
                if scene.start_time <= (s.start + s.end) / 2 <= (scene.end_time or float("inf"))
            ]

        if not scene_sents:
            continue

        while scene_sents and scene_sents[-1].text.strip().lower() in FILLER_ENDINGS:
            scene_sents = scene_sents[:-1]

        if not scene_sents:
            continue

        duration = round(scene_sents[-1].end - scene_sents[0].start, 1)
        if duration < 10:
            continue

        text = " ".join(s.text for s in scene_sents)
        clip = {
            "type":       "sequence",
            "source":     "scene_injected",
            "scene_type": scene.scene_type,
            "text":       text,
            "start_time": scene_sents[0].start,
            "end_time":   scene_sents[-1].end,
            "duration":   duration,
            "clips": [{
                "start_time": scene_sents[0].start,
                "end_time":   scene_sents[-1].end,
                "text":       text,
            }],
            "reason": scene.clip_reason or f"Visually detected {scene.scene_type} clip",
        }
        if scene.speaker_name:
            clip["speaker_name"] = scene.speaker_name

        injected.append(clip)
        log.info(
            f"[SCENE] Injected: {scene_sents[0].start:.1f}s–"
            f"{scene_sents[-1].end:.1f}s ({duration}s) [{scene.scene_type}]"
        )
    return injected


# ── Stage 2: Segment ──────────────────────────────────────────────────────────

def segment(sentences, video_duration, scenes=None):
    """
    Topic-aware segmentation — replaces the sliding window approach.

    OLD approach: iterate every sentence as a window start → 274 windows for a
    27-min video, 85% content overlap between neighbours. The LLM saw essentially
    the same content 274 times and kept picking the same "best" window.

    NEW approach:
      1. Detect natural topic boundaries via silence gaps + transition phrases
         + scene boundaries from the visual detector
      2. Group sentences between boundaries into topic chunks
      3. Merge consecutive short chunks until MIN_CLIP_DUR is reached
      4. Split any oversized chunks at their midpoint
      5. Result: ~20-35 non-overlapping windows, each a distinct topic section

    Benefits:
      - Zero content overlap between windows
      - LLM sees genuinely different content each time
      - Window list tokens drop ~97% (100k → ~3k per call)
      - Each window ID maps to a unique, non-repeatable clip
    """
    # Phrase patterns that signal a topic shift
    TRANSITION_TRIGGERS = [
        "let's move", "next", "fact ", "all right", "okay,",
        "now let's", "moving on", "let's dive", "here is",
        "with that said", "number ", "so now", "let's look",
        "fact one", "fact two", "fact three", "fact four", "fact five",
        "fact six", "fact seven", "fact eight", "fact nine", "fact ten",
        "alright", "so let's", "and now", "coming to", "let's come back",
        "let's start", "let's begin", "to summarise", "in summary",
        "moving to", "next up", "first,", "second,", "third,",
    ]

    MIN_DUR = SEQUENTIAL_MIN_DURATION  # 25s
    MAX_DUR = NON_SEQUENTIAL_MAX_DURATION - 2  # 88s — gives split room

    # ── Step 1: detect topic boundaries ──────────────────────────────────────
    boundary_idxs: set = set()
    boundary_idxs.add(0)
    boundary_idxs.add(len(sentences))

    for i in range(len(sentences) - 1):
        gap       = sentences[i + 1].start - sentences[i].end
        next_text = sentences[i + 1].text.lower()

        is_gap        = gap > 1.5
        is_transition = any(t in next_text for t in TRANSITION_TRIGGERS)
        if is_gap or is_transition:
            boundary_idxs.add(i + 1)

    # Add visual scene boundaries
    if scenes:
        for scene in scenes:
            for i, s in enumerate(sentences):
                if abs(s.start - scene.start_time) < 2.0:
                    boundary_idxs.add(i)
                    break

    boundary_list = sorted(boundary_idxs)
    log.info(f"[SEGMENT] {len(boundary_list)-1} raw topic chunks detected")

    # ── Step 2: build raw chunks between boundaries ───────────────────────────
    raw_chunks = []
    for idx in range(len(boundary_list) - 1):
        s_idx = boundary_list[idx]
        e_idx = boundary_list[idx + 1]
        chunk_sents = sentences[s_idx:e_idx]
        if not chunk_sents:
            continue
        dur = round(chunk_sents[-1].end - chunk_sents[0].start, 1)
        raw_chunks.append({
            "sents":  chunk_sents,
            "start":  chunk_sents[0].start,
            "end":    chunk_sents[-1].end,
            "dur":    dur,
        })

    # ── Step 3: merge short chunks until MIN_DUR reached ─────────────────────
    merged = []
    buffer: list = []
    buffer_dur   = 0.0

    for chunk in raw_chunks:
        buffer.append(chunk)
        buffer_dur += chunk["dur"]
        if buffer_dur >= MIN_DUR:
            all_sents = [s for c in buffer for s in c["sents"]]
            merged.append({
                "sents": all_sents,
                "start": all_sents[0].start,
                "end":   all_sents[-1].end,
                "dur":   round(all_sents[-1].end - all_sents[0].start, 1),
            })
            buffer, buffer_dur = [], 0.0

    # Attach any leftover sentences to the last merged chunk
    if buffer:
        extra = [s for c in buffer for s in c["sents"]]
        if merged:
            merged[-1]["sents"].extend(extra)
            merged[-1]["end"] = merged[-1]["sents"][-1].end
            merged[-1]["dur"] = round(
                merged[-1]["end"] - merged[-1]["start"], 1
            )
        else:
            # Edge case: entire video is shorter than MIN_DUR
            if extra:
                merged.append({
                    "sents": extra,
                    "start": extra[0].start,
                    "end":   extra[-1].end,
                    "dur":   round(extra[-1].end - extra[0].start, 1),
                })

    # ── Step 4: split chunks that exceed MAX_DUR ──────────────────────────────
    final_chunks = []
    for chunk in merged:
        if chunk["dur"] <= MAX_DUR:
            final_chunks.append(chunk)
        else:
            # Split at sentence closest to the midpoint time
            mid_time = (chunk["start"] + chunk["end"]) / 2
            split_at = 1
            for k, s in enumerate(chunk["sents"]):
                if s.start >= mid_time:
                    split_at = k
                    break

            # Ensure neither half is empty
            split_at = max(1, min(split_at, len(chunk["sents"]) - 1))
            for half in [chunk["sents"][:split_at], chunk["sents"][split_at:]]:
                if half:
                    final_chunks.append({
                        "sents": half,
                        "start": half[0].start,
                        "end":   half[-1].end,
                        "dur":   round(half[-1].end - half[0].start, 1),
                    })

    # ── Step 5: convert to CandidateWindow objects ────────────────────────────
    windows = []
    for i, chunk in enumerate(final_chunks):
        sents = chunk["sents"]
        windows.append(CandidateWindow(
            id=i + 1,
            start_s=sents[0].index,
            end_s=sents[-1].index,
            start_time=chunk["start"],
            end_time=chunk["end"],
            duration=chunk["dur"],
            text=" ".join(s.text for s in sents),
            source=sents[0].source,
        ))

    log.info(
        f"[SEGMENT] {len(windows)} topic-aware windows "
        f"(old sliding approach would have made ~{len(sentences)} windows)"
    )
    for w in windows:
        log.info(
            f"  W{w.id:02d} [{w.start_time:.1f}s-{w.end_time:.1f}s] "
            f"({w.duration}s) {w.text[:60]}..."
        )
    return windows


# ── Dynamic target score ──────────────────────────────────────────────────────

class DynamicTarget:
    """
    FIX 3: Calibrates TARGET_SCORE based on what this specific video can achieve.

    After the first clip is scored, the target is set to:
        max(MIN_QUALITY_FLOOR, best_score_seen * 0.92)

    This prevents infinite regeneration loops on videos like conference keynotes
    where structural limitations (filler transitions, no narrative peaks) make
    90/100 impossible. The target becomes 'near the best achievable' rather than
    a fixed number that can never be reached.
    """
    def __init__(self):
        self._target     = INITIAL_TARGET_SCORE
        self._calibrated = False
        self._scores:    List[int] = []

    def update(self, score: int):
        self._scores.append(score)
        if not self._calibrated and len(self._scores) >= 1:
            best = max(self._scores)
            calibrated = max(MIN_QUALITY_FLOOR, int(best * 0.92))
            if calibrated < self._target:
                log.info(
                    f"[TARGET] Calibrated from {self._target} → {calibrated} "
                    f"(best seen: {best}, floor: {MIN_QUALITY_FLOOR})"
                )
                self._target     = calibrated
                self._calibrated = True

    @property
    def value(self) -> int:
        return self._target

    def reached(self, score: int) -> bool:
        return score >= self._target


# ── Scorer ────────────────────────────────────────────────────────────────────

def _apply_end_trim(short):
    clips = short.get("clips", [])
    if not clips:
        return short
    if clips[-1]["text"].strip().lower() in FILLER_ENDINGS:
        log.info(f"[TRIM] Dropping filler ending: '{clips[-1]['text']}'")
        clips = clips[:-1]
        if clips:
            short["clips"]    = clips
            short["end_time"] = clips[-1]["end_time"]
            short["duration"] = round(short["end_time"] - short["start_time"], 1)
            short["text"]     = " ".join(c["text"] for c in clips)
    return short


def score_single_short(short, category_profile):
    text_preview = " ".join(short["text"].split()[:100]) + "..."
    weights      = category_profile["weights"]

    prompt = SCORER_PROMPT.format(
        category=category_profile["category"],
        audience=category_profile["audience"],
        weight_reasoning=category_profile["weight_reasoning"],
        clip_type=short["type"],
        duration=short["duration"],
        text_preview=text_preview,
    )

    result = get_response(prompt, response_format=SCORER_RESPONSE_FORMAT, caller="scorer")

    def safe_int(v):
        try:
            return max(0, min(20, int(v)))
        except (ValueError, TypeError):
            return 0

    dim_keys   = list(DEFAULT_DIMENSION_WEIGHTS.keys())
    raw_scores = {k: safe_int(result.get(k, 0)) for k in dim_keys}

    weighted_sum = sum(raw_scores[d] * weights.get(d, 1 / 6) for d in dim_keys)
    total_score  = round((weighted_sum / 20) * 100)
    weakest      = min(raw_scores, key=lambda k: raw_scores[k])

    if result.get("recommended_end_trim"):
        short = _apply_end_trim(short)

    return short, {
        "ConfidenceScore": total_score,
        "total_score":     total_score,
        "ScoreBreakdown":  raw_scores,
        "Category":        category_profile["category"],
        "reason":          result.get("reason",         ""),
        "weakest_factor":  result.get("weakest_factor", weakest),
        "improvise":       result.get("improvise",      ""),
    }


# ── FIX 1+2+5: Unified window filtering ───────────────────────────────────────

def _available_windows(windows: list, used_ids: Set[int]) -> list:
    """
    FIX 2 + FIX 5: Returns only windows not yet used.
    This is the core fix — the LLM only SEES unused windows.
    It cannot pick what it cannot see.
    Regeneration calls with 10 used windows get ~167 windows instead of 177.
    At scale this saves ~95% of regeneration token cost.
    """
    available = [w for w in windows if w.id not in used_ids]
    if len(available) < len(windows):
        log.info(f"[DEDUP] Filtered {len(windows)-len(available)} used windows → {len(available)} available")
    return available


def _format_windows(windows, max_windows=500):
    thinned = windows
    if len(windows) > max_windows:
        step    = max(1, len(windows) // max_windows)
        thinned = windows[::step][:max_windows]
    return "\n".join(
        f"W{w.id} ({w.duration}s) [{w.start_time:.1f}s-{w.end_time:.1f}s] "
        f"[scene:{w.source}] {w.text}"
        for w in thinned
    )


def _calculate_clips_count(video_duration, available_windows):
    if video_duration <= 0 or available_windows == 0:
        return 1
    avg_clip     = (SEQUENTIAL_MIN_DURATION + SEQUENTIAL_MAX_DURATION) / 2
    max_possible = max(1, int(video_duration / avg_clip))
    clips = max(1, min(int(video_duration * 0.15 / avg_clip), 12)) if video_duration < 180 \
        else max(4, min(int(video_duration * 0.15 / avg_clip), 12))
    return min(clips, available_windows, max_possible)


# ── Stage 3a: Sequential selection ───────────────────────────────────────────

def _agent_sequence(windows, clips_count, category_profile,
                    used_window_ids: Set[int],
                    regenerate_feedback=None, frames_b64=None):
    """
    FIX 1: used_window_ids is passed in — never created locally.
    FIX 2: available_windows = windows filtered by used_window_ids.
           LLM only sees unused windows — cannot repeat a pick.
    FIX 5: regeneration gets the filtered list, not the full list.
    """
    # FIX 2+5: filter BEFORE building the prompt
    available = _available_windows(windows, used_window_ids)
    if not available:
        log.warning("[SEQ] No available windows left — stopping")
        return []

    feedback_section = ""
    if regenerate_feedback:
        feedback_section = (
            f"### Regeneration context\n"
            f"Previous score: {regenerate_feedback['score']}/100 "
            f"(target: {regenerate_feedback.get('target', INITIAL_TARGET_SCORE)}+)\n"
            f"Weakest: {regenerate_feedback['weakest_factor']}\n"
            f"Problem: {regenerate_feedback['reason']}\n"
            f"Improve: {regenerate_feedback['improvise']}\n"
            f"Pick a DIFFERENT window that addresses these issues.\n"
        )
        clips_count = 1

    visual_section = ""
    if frames_b64:
        visual_section = (
            "\n### Visual context\nFor interview: include both question AND answer. "
            "REJECT windows that are only a question.\n"
        )

    prompt = SEQUENCE_PROMPT.format(
        feedback_section=feedback_section + visual_section,
        category=category_profile["category"],
        audience=category_profile["audience"],
        weight_reasoning=category_profile["weight_reasoning"],
        clips_count=clips_count,
        windows_text=_format_windows(available),  # FIX 2+5: filtered list only
    )

    if frames_b64:
        result = get_multimodal_response(prompt, frames_b64,
                                         response_format=SEQUENCE_RESPONSE_FORMAT,
                                         caller="agent1_sequence")
    else:
        result = get_response(prompt, response_format=SEQUENCE_RESPONSE_FORMAT,
                              caller="agent1_sequence")

    # Build lookup from available windows only — not full list
    window_by_id = {w.id: w for w in available}
    seq_clips    = []

    for item in result.get("clips", []):
        wid = item.get("WindowId")
        if isinstance(wid, str):
            wid = wid.replace("W", "").replace("w", "").strip()
        try:
            wid = int(wid)
        except (ValueError, TypeError):
            continue
        if wid not in window_by_id:
            log.warning(f"[SEQ] LLM returned unknown/used WindowId {wid} — skipping")
            continue
        w = window_by_id[wid]
        # FIX 1: mark used in the SHARED set immediately
        used_window_ids.add(wid)
        seq_clips.append({
            "type":       "sequence",
            "text":       w.text,
            "start_time": w.start_time,
            "end_time":   w.end_time,
            "duration":   w.duration,
            "clips": [{"start_time": w.start_time,
                       "end_time":   w.end_time,
                       "text":       w.text}],
            "reason":     item.get("reason", ""),
        })
    return seq_clips


def _run_review_loop(short, short_id, regenerate_fn, category_profile,
                     iteration_log, dynamic_target: DynamicTarget):
    """
    FIX 3: Uses DynamicTarget instead of fixed TARGET_SCORE.
    After first score, target calibrates to what this video can achieve.
    """
    current     = short
    iterations  = 0
    best_score  = -1
    best_clip   = None
    best_result = None

    while iterations < MAX_ITERATIONS_PER_SHORT:
        current, score_result = score_single_short(current, category_profile)
        total = score_result["total_score"]
        log.info(f"{short_id} iter {iterations+1}: score={total}/100 target={dynamic_target.value}")

        # Calibrate target after first real score
        dynamic_target.update(total)

        if total > best_score:
            best_score  = total
            best_clip   = dict(current)
            best_result = dict(score_result)

        if dynamic_target.reached(total):
            current.update(score_result)
            current["iterations"] = iterations
            break

        iterations += 1
        iteration_log["iterations"]  += 1
        iteration_log["regenerated"] += 1

        if iterations >= MAX_ITERATIONS_PER_SHORT:
            best_clip.update(best_result)
            best_clip["iterations"] = iterations
            current = best_clip
            break

        feedback = {
            "score":          total,
            "target":         dynamic_target.value,
            "weakest_factor": score_result["weakest_factor"],
            "reason":         score_result["reason"],
            "improvise":      score_result["improvise"],
        }
        regenerated = regenerate_fn(feedback)
        if regenerated:
            current = regenerated[0]
            current["short_id"] = short_id
        else:
            current.update(score_result)
            current["iterations"] = iterations
            break

    return current


def _agent_sequence_reviewer(seq_shorts, windows, category_profile,
                              used_window_ids: Set[int],
                              dynamic_target: DynamicTarget,
                              frames_b64=None):
    """
    FIX 1: Receives the shared used_window_ids set — same object as selector used.
    FIX 2+5: regen_fn passes the same shared set → filtered windows sent to LLM.
    """
    iteration_log = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed      = []

    for idx, short in enumerate(seq_shorts):
        short_id          = f"seq_{idx+1}"
        short["short_id"] = short_id

        def regen_fn(feedback, _uid=used_window_ids):
            return _agent_sequence(
                windows, 1,
                category_profile=category_profile,
                used_window_ids=_uid,      # FIX 1: shared set
                regenerate_feedback=feedback,
                frames_b64=frames_b64,
            )

        reviewed_short = _run_review_loop(
            short, short_id, regen_fn, category_profile,
            iteration_log, dynamic_target
        )
        reviewed.append(reviewed_short)
        iteration_log["final_scores"].append({
            "short_id":    short_id,
            "final_score": reviewed_short.get("ConfidenceScore", 0),
            "iterations":  reviewed_short.get("iterations", 0),
        })

    return reviewed, iteration_log


def sequential_selection(sentences, windows, video_duration,
                         category_profile, frames_b64=None,
                         injected_clips=None):
    """
    CORE FIX for repeated windows:

    The original code called _agent_sequence(clips_count=6) — asking the LLM
    to return 6 clips in ONE response. Even with filtered windows, the LLM
    generates token-by-token within one response with no memory of what it
    already picked. It would return the same WindowId multiple times.

    The fix: loop clips_count times, ONE call per clip.
    Each iteration:
      1. Removes already-used windows from the list (LLM cannot see them)
      2. Asks for exactly 1 clip
      3. Immediately registers the picked window as used
      4. Reviews + scores that 1 clip before moving to the next

    This makes repetition structurally impossible — not just instructed against.
    """
    try:
        # ONE shared set for the entire sequential pipeline
        shared_used_ids: Set[int] = set()

        # Register injected clip windows as used immediately
        for clip in (injected_clips or []):
            for w in windows:
                if abs(w.start_time - clip["start_time"]) < 1.0:
                    shared_used_ids.add(w.id)
                    break

        clips_count    = _calculate_clips_count(video_duration, len(windows))
        dynamic_target = DynamicTarget()
        iteration_log  = {"iterations": 0, "regenerated": 0, "final_scores": []}

        log.info(f"[SEQ] Targeting {clips_count} clips — one call per clip")

        # Start with injected scene clips — they go straight to review
        all_reviewed: list = []

        # Review injected clips first (they don't need selection)
        for idx, clip in enumerate(injected_clips or []):
            short_id          = f"seq_{idx + 1}"
            clip["short_id"]  = short_id

            def regen_fn_inj(feedback, _uid=shared_used_ids):
                return _agent_sequence(
                    windows, 1,
                    category_profile=category_profile,
                    used_window_ids=_uid,
                    regenerate_feedback=feedback,
                    frames_b64=frames_b64,
                )

            reviewed = _run_review_loop(
                clip, short_id, regen_fn_inj,
                category_profile, iteration_log, dynamic_target
            )
            all_reviewed.append(reviewed)
            iteration_log["final_scores"].append({
                "short_id":    short_id,
                "final_score": reviewed.get("ConfidenceScore", 0),
                "iterations":  reviewed.get("iterations", 0),
            })

        injected_count = len(injected_clips or [])

        # ONE-CLIP-PER-CALL loop for agent-selected clips
        for i in range(clips_count):
            available = _available_windows(windows, shared_used_ids)
            if not available:
                log.info(f"[SEQ] No windows left after {i} clips — stopping")
                break

            # Ask LLM for exactly 1 clip from the filtered list
            picks = _agent_sequence(
                windows, 1,
                category_profile=category_profile,
                used_window_ids=shared_used_ids,
                frames_b64=frames_b64,
            )

            if not picks:
                log.warning(f"[SEQ] Call {i+1}: LLM returned no clip — stopping")
                break

            clip     = picks[0]
            short_id = f"seq_{injected_count + i + 1}"
            clip["short_id"] = short_id

            log.info(
                f"[SEQ] Clip {i+1}/{clips_count}: "
                f"{clip['start_time']:.1f}s–{clip['end_time']:.1f}s "
                f"({clip['duration']}s)"
            )

            # Review this single clip immediately
            def regen_fn(feedback, _uid=shared_used_ids):
                return _agent_sequence(
                    windows, 1,
                    category_profile=category_profile,
                    used_window_ids=_uid,
                    regenerate_feedback=feedback,
                    frames_b64=frames_b64,
                )

            reviewed = _run_review_loop(
                clip, short_id, regen_fn,
                category_profile, iteration_log, dynamic_target
            )
            all_reviewed.append(reviewed)
            iteration_log["final_scores"].append({
                "short_id":    short_id,
                "final_score": reviewed.get("ConfidenceScore", 0),
                "iterations":  reviewed.get("iterations", 0),
            })

        log.info(
            f"[SEQ] Done — {len(all_reviewed)} clips, "
            f"{iteration_log['regenerated']} regenerations, "
            f"target={dynamic_target.value}"
        )
        return all_reviewed, iteration_log

    except Exception as e:
        log.error(f"[SEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 3b: Non-sequential selection ───────────────────────────────────────

def _calculate_target_count(video_duration):
    avg = 40
    max_possible = max(1, int(video_duration / avg))
    target = max(1, min(int(video_duration * 0.12 / avg), 10)) if video_duration < 180 \
        else max(4, min(int(video_duration * 0.12 / avg), 10))
    return min(target, max_possible)


def _agent_nonsequence(sentences, clips_count, category_profile,
                       regenerate_feedback=None, used_sentence_ids=None):
    sentences_text = "\n".join(
        f"S{s.index}: [{s.start:.1f}s-{s.end:.1f}s] [scene:{s.source}] {s.text}"
        for s in sentences
    )

    feedback_section = ""
    if regenerate_feedback:
        all_used = used_sentence_ids or set()
        old_ids  = regenerate_feedback.get("old_sentence_ids", [])
        note     = ""
        if old_ids:
            note += f"\nSentences in previous attempt: {', '.join(f'S{s}' for s in old_ids)}"
        if all_used:
            note += f"\nAll used sentences (avoid): {', '.join(f'S{s}' for s in sorted(all_used))}"
        feedback_section = (
            f"### Regeneration context\n{note}\n\n"
            f"Previous score: {regenerate_feedback['score']}/100 "
            f"(target: {regenerate_feedback.get('target', INITIAL_TARGET_SCORE)}+)\n"
            f"Weakest: {regenerate_feedback['weakest_factor']}\n"
            f"Problem: {regenerate_feedback['reason']}\n"
            f"Improve: {regenerate_feedback['improvise']}\n"
            f"Pick a DIFFERENT set of sentences.\n"
        )
        clips_count = 1

    prompt = NONSEQUENCE_PROMPT.format(
        feedback_section=feedback_section,
        category=category_profile["category"],
        audience=category_profile["audience"],
        weight_reasoning=category_profile["weight_reasoning"],
        sentences_text=sentences_text,
    )

    result = get_response(prompt, response_format=NONSEQUENCE_RESPONSE_FORMAT,
                          caller="agent2_nonsequence")

    sentence_by_idx = {s.index: s for s in sentences}
    nonseq_shorts   = []

    for short_data in result.get("shorts", []):
        sentence_ids = short_data.get("sentence_ids", [])
        if not sentence_ids:
            continue
        assembled, text_parts = [], []
        for sid in sentence_ids:
            try:
                sid_int = int(sid)
            except (ValueError, TypeError):
                continue
            if sid_int in sentence_by_idx:
                s = sentence_by_idx[sid_int]
                assembled.append({
                    "start_time": s.start,
                    "end_time":   s.end,
                    "duration":   round(s.end - s.start, 1),
                    "text":       s.text,
                })
                text_parts.append(s.text)
        if not assembled:
            continue
        total_dur = sum(c["duration"] for c in assembled)
        if total_dur < NON_SEQUENTIAL_MIN_DURATION:
            log.warning(f"[NONSEQ] Clip too short ({total_dur:.1f}s) — skipping")
            continue
        if total_dur > NON_SEQUENTIAL_MAX_DURATION:
            log.warning(f"[NONSEQ] Clip too long ({total_dur:.1f}s) — skipping")
            continue
        nonseq_shorts.append({
            "type":          "non-sequence",
            "topic":         short_data.get("topic", ""),
            "text":          " ".join(text_parts),
            "start_time":    assembled[0]["start_time"],
            "end_time":      assembled[-1]["end_time"],
            "duration":      round(total_dur, 1),
            "num_clips":     len(assembled),
            "clips":         assembled,
            "reason":        short_data.get("reason", ""),
            "_sentence_ids": [int(s) for s in sentence_ids
                              if str(s).lstrip("-").isdigit()],
        })
    return nonseq_shorts


def _agent_nonsequence_reviewer(nonseq_shorts, sentences, category_profile):
    iteration_log    = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed         = []
    all_used_ids:set = set()
    dynamic_target   = DynamicTarget()

    for short in nonseq_shorts:
        for sid in short.get("_sentence_ids", []):
            all_used_ids.add(sid)

    for idx, short in enumerate(nonseq_shorts):
        short_id          = f"nonseq_{idx+1}"
        short["short_id"] = short_id

        def regen_fn(feedback, _short=short, _uid=all_used_ids):
            old_ids = _short.get("_sentence_ids", [])
            if not old_ids:
                for clip in _short.get("clips", []):
                    for s in sentences:
                        if abs(s.start - clip["start_time"]) < 0.5:
                            old_ids.append(s.index)
                            break
            feedback["old_sentence_ids"] = old_ids
            result = _agent_nonsequence(
                sentences, 1,
                category_profile=category_profile,
                regenerate_feedback=feedback,
                used_sentence_ids=_uid,
            )
            if result:
                for sid in result[0].get("_sentence_ids", []):
                    _uid.add(sid)
            return result

        reviewed_short = _run_review_loop(
            short, short_id, regen_fn, category_profile,
            iteration_log, dynamic_target
        )
        reviewed.append(reviewed_short)
        iteration_log["final_scores"].append({
            "short_id":    short_id,
            "final_score": reviewed_short.get("ConfidenceScore", 0),
            "iterations":  reviewed_short.get("iterations", 0),
        })

    return reviewed, iteration_log


def non_sequential_selection(sentences, video_duration, category_profile):
    try:
        if video_duration < NON_SEQUENTIAL_MIN_DURATION:
            log.info(f"[NONSEQ] Video too short ({video_duration}s) — skipping")
            return [], {}
        clips_count = _calculate_target_count(video_duration)
        log.info(f"[NONSEQ] Targeting {clips_count} shorts")
        nonseq_shorts = _agent_nonsequence(sentences, clips_count,
                                           category_profile=category_profile)
        log.info(f"[NONSEQ] Agent #2 generated {len(nonseq_shorts)} shorts")
        reviewed, log_data = _agent_nonsequence_reviewer(nonseq_shorts, sentences,
                                                         category_profile=category_profile)
        log.info(f"[NONSEQ] Reviewed {len(reviewed)}, regenerated {log_data['regenerated']}")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[NONSEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── FIX 4+6: Quality filter + duplicate guard ─────────────────────────────────

def _deduplicate_and_filter(clips: list, label: str) -> list:
    """
    FIX 4: Drop clips below MIN_QUALITY_FLOOR.
    FIX 6: Drop clips whose exact time range already exists in the output.
    """
    seen_ranges: set = set()
    out             = []

    for clip in clips:
        score = clip.get("ConfidenceScore", clip.get("confidence_score", 0))

        # FIX 4: quality floor
        if score < MIN_QUALITY_FLOOR:
            log.info(
                f"[FILTER] Dropping {clip.get('short_id','?')} — "
                f"score {score} below floor {MIN_QUALITY_FLOOR}"
            )
            continue

        # FIX 6: duplicate time range guard
        start = round(clip.get("start_time", 0), 1)
        end   = round(clip.get("end_time",   0), 1)
        key   = (start, end)
        if key in seen_ranges:
            log.info(
                f"[FILTER] Dropping {clip.get('short_id','?')} — "
                f"duplicate range {start}s-{end}s"
            )
            continue

        seen_ranges.add(key)
        out.append(clip)

    dropped = len(clips) - len(out)
    if dropped:
        log.info(f"[FILTER] {label}: kept {len(out)}/{len(clips)} clips ({dropped} dropped)")
    return out


# ── Stage 4: Final metadata ───────────────────────────────────────────────────

def final_metadata(sequential, non_sequential, language_code):
    all_shorts = sequential + non_sequential
    if not all_shorts:
        return sequential, non_sequential

    context_text = "\n\n".join(
        f"Short (ID: {s.get('short_id','')})\n"
        f"Type: {s['type']} | Duration: {s['duration']:.1f}s "
        f"| Score: {s.get('ConfidenceScore',0)}/100\n"
        f"Category: {s.get('Category','')}\n"
        f"Text: \"{' '.join(s['text'].split()[:80])}...\""
        for s in all_shorts
    )

    result = get_response(
        METADATA_PROMPT.format(
            short_count=len(all_shorts),
            shorts_context=context_text,
            language_code=language_code,
        ),
        temperature=0.3,
        response_format=METADATA_RESPONSE_FORMAT,
        caller="agent5_metadata",
    )

    by_id = {m["short_id"]: m for m in result.get("shorts", [])}
    for short in all_shorts:
        meta = by_id.get(short.get("short_id", ""), {})
        short["Title"]       = meta.get("title",     short.get("Title", "Untitled Short"))
        short["SocialMedia"] = meta.get("platforms", short.get("SocialMedia",
                                                               ["instagram", "tiktok", "youtube"]))

    seq_count = len(sequential)
    log.info(f"[META] Generated metadata for {len(all_shorts)} shorts")
    return all_shorts[:seq_count], all_shorts[seq_count:]


# ── Stage 5: Ranking ──────────────────────────────────────────────────────────

def ranking(sequential, non_sequential):
    sequential     = sorted(sequential,     key=lambda c: c.get("start_time", 0))
    non_sequential = sorted(non_sequential, key=lambda s: s.get("ConfidenceScore", 0), reverse=True)
    log.info(f"[RANK] {len(sequential)} sequential, {len(non_sequential)} non-sequential")
    return sequential, non_sequential


# ── Output formatting ─────────────────────────────────────────────────────────

def format_sequential(clips):
    return [{
        "debug_id":         c.get("short_id",       ""),
        "title":            c.get("Title",           ""),
        "text":             c["text"],
        "confidence_score": c.get("ConfidenceScore", 0),
        "category":         c.get("Category",        "general"),
        "source":           c.get("source",          "agent"),
        "scene_type":       c.get("scene_type",      ""),
        "speaker_name":     c.get("speaker_name",    ""),
        "social_media":     c.get("SocialMedia",     []),
        "video_start_time": c["start_time"],
        "video_end_time":   c["end_time"],
        "score_breakdown":  c.get("ScoreBreakdown",  {}),
        "reason":           c.get("reason",          ""),
        "weakest_factor":   c.get("weakest_factor",  ""),
        "iterations":       c.get("iterations",      0),
    } for c in clips]


def format_non_sequential(shorts):
    return [{
        "short_id":         s.get("short_id",       ""),
        "debug_id":         s.get("short_id",       ""),
        "title":            s.get("Title",          ""),
        "confidence_score": s.get("ConfidenceScore",0),
        "category":         s.get("Category",       "general"),
        "social_media":     s.get("SocialMedia",    []),
        "total_duration":   s.get("duration",       0),
        "num_clips":        s.get("num_clips",       0),
        "clips": [{
            "startTime": c["start_time"],
            "endTime":   c["end_time"],
            "duration":  c["duration"],
            "text":      c["text"],
        } for c in s.get("clips", [])],
        "score_breakdown":  s.get("ScoreBreakdown", {}),
        "reason":           s.get("reason",         ""),
        "weakest_factor":   s.get("weakest_factor", ""),
        "iterations":       s.get("iterations",     0),
    } for s in shorts]


# ── Entry point ───────────────────────────────────────────────────────────────

def run(transcription_data, video_path=None, clip_mode="both"):
    frames_b64 = []
    scenes     = []
    if video_path:
        frames_b64 = extract_frames_from_file(video_path)

    # 1a — Preprocess
    sentences, video_duration, language_code = preprocess(transcription_data)

    # 1b — Category + adaptive weights
    category_profile = detect_category_weights(sentences)

    # 1c — Scene detection
    if frames_b64:
        scenes    = detect_scenes(frames_b64, category_profile)
        sentences = tag_sentences_by_scene(sentences, scenes)

    injected_clips = inject_scene_clips(scenes, sentences) if scenes else []
    log.info(f"[AGENT] {len(injected_clips)} clips injected from scene detection")

    # 2 — Segment
    windows = segment(sentences, video_duration, scenes=scenes)

    sequential_clips,      seq_log    = [], {}
    non_sequential_shorts, nonseq_log = [], {}

    # 3a — Sequential
    if clip_mode in ("both", "sequential"):
        sequential_clips, seq_log = sequential_selection(
            sentences, windows, video_duration,
            category_profile=category_profile,
            frames_b64=frames_b64,
            injected_clips=injected_clips,
        )
        # FIX 4+6: filter before metadata
        sequential_clips = _deduplicate_and_filter(sequential_clips, "sequential")

    # 3b — Non-sequential
    if clip_mode in ("both", "non_sequential"):
        non_sequential_shorts, nonseq_log = non_sequential_selection(
            sentences, video_duration,
            category_profile=category_profile,
        )
        # FIX 4+6: filter before metadata
        non_sequential_shorts = _deduplicate_and_filter(non_sequential_shorts, "non_sequential")

    # 4 — Metadata
    sequential_clips, non_sequential_shorts = final_metadata(
        sequential_clips, non_sequential_shorts, language_code,
    )

    # 5 — Ranking
    sequential_clips, non_sequential_shorts = ranking(
        sequential_clips, non_sequential_shorts,
    )

    return {
        "video_duration":        video_duration,
        "language_code":         language_code,
        "sentence_count":        len(sentences),
        "window_count":          len(windows),
        "injected_clip_count":   len(injected_clips),
        "category_profile": {
            "category":             category_profile["category"],
            "audience":             category_profile["audience"],
            "weights":              category_profile["weights"],
            "weight_reasoning":     category_profile["weight_reasoning"],
            "benefits_from_frames": category_profile["benefits_from_frames"],
            "video_grammar":        category_profile["video_grammar"],
        },
        "scene_map": [
            {
                "start":       sc.start_time,
                "end":         sc.end_time,
                "type":        sc.scene_type,
                "confidence":  sc.confidence,
                "clip_worthy": sc.clip_worthy,
            }
            for sc in scenes
        ],
        "sequential_clips":      format_sequential(sequential_clips),
        "non_sequential_shorts": format_non_sequential(non_sequential_shorts),
        "iteration_log": {
            "sequential":     seq_log,
            "non_sequential": nonseq_log,
        },
        "token_usage": get_token_usage(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="LFTS video clipping pipeline.")
    p.add_argument("--transcript", "-t", default=DEFAULT_TRANSCRIPT_PATH)
    p.add_argument("--video",      "-v", default=DEFAULT_VIDEO_PATH,
                   help="Optional video file (requires ffmpeg)")
    p.add_argument("--output",     "-o", default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--clip-mode",  choices=("both","sequential","non_sequential"),
                   default="both")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not CLAUDE_API_KEY:
        raise RuntimeError("CLAUDE_API_KEY not set.\n  export CLAUDE_API_KEY='your-key'")

    transcript_path = os.path.abspath(args.transcript)
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    video_path = (args.video or "").strip() or None
    if video_path and not os.path.isfile(video_path):
        log.warning(f"[MAIN] Video not found at {video_path} — frames skipped")
        video_path = None

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    transcription_data = raw.get("results", raw)

    result = run(transcription_data, video_path=video_path, clip_mode=args.clip_mode)

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    seq_count    = len(result["sequential_clips"])
    nonseq_count = len(result["non_sequential_shorts"])
    log.info(f"[DONE] {seq_count} sequential + {nonseq_count} non-sequential clips")
    log.info(f"[DONE] Category: {result['category_profile']['category']}")
    log.info(f"[DONE] Tokens:   {result['token_usage']['total_tokens']:,}")
    log.info(f"[DONE] Output:   {output_path}")
    return result


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        log.error(str(e))
        sys.exit(1)
