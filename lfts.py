#!/usr/bin/env python3
"""Standalone runner for the video clipping agent pipeline.

Runs Stages 1–5 plus ranking, end-to-end:
    Preprocess → Segment → Sequential (Agent #1 + #3) → Non-Sequential (Agent #2 + #4) →
    Final Metadata (Agent #5) → Ranking → write result JSON

Skipped on purpose (for feasibility): execution tracing, ES feedback, S3 uploads, thumbnails.
Frame extraction runs only if VIDEO_PATH is set (needs ffmpeg on PATH).

Quick start:
    pip install -r requirements.txt
    export CLAUDE_API_KEY="your-key"
    python lfts.py

Or pass paths on the CLI:
    python lfts.py --transcript /path/to/transcript.json --output result.json

The transcript JSON may be either the raw AWS Transcribe payload (with a top-level
"results" key) or the inner "results" dict directly.
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
from dataclasses import dataclass

from anthropic import Anthropic


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSCRIPT_PATH = os.path.join(_SCRIPT_DIR, "transcript.json")
DEFAULT_VIDEO_PATH = os.path.join(_SCRIPT_DIR, "input.mp4")
DEFAULT_OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "result.json")


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("clipping_agent_standalone")


# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.environ.get(
    "CLAUDE_API_KEY",
    "",
)
CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS = 8192

SEQUENTIAL_MIN_DURATION = 25
SEQUENTIAL_MAX_DURATION = 55

NON_SEQUENTIAL_MIN_DURATION = 25
NON_SEQUENTIAL_MAX_DURATION = 90

TARGET_SCORE = 90
MAX_ITERATIONS_PER_SHORT = 3
SCORE_MARKET_ADJUSTMENT = 6

FRAME_INTERVAL = 30   # seconds between frames
MAX_FRAMES = 20

# Scorer dimension weights (must sum to 1.0)
DIMENSION_WEIGHTS = {
    "hook_strength": 0.25,
    "reframe_insight": 0.20,
    "emotional_resonance": 0.18,
    "standalone_clarity": 0.17,
    "quotability": 0.12,
    "clean_ending": 0.08,
}


# ── Prompts (copied verbatim from agent/prompts/) ─────────────────────────────

SEQUENCE_PROMPT = """
{feedback_section}

You are a pro-level video editor and content strategist. Watch the full video, understand its story and tone, then identify the category it belongs to — whether that's educational, podcast, interview, tutorial, inspirational, motivational, humorous, or any other video category.
After identifying, consider yourself an expert in that category. Your job is to find the best shareable and engaging short clips for YouTube, Instagram, and TikTok. You do this not by watching and then picking, but by scanning the video through 6 dimensions that together tell you whether a moment is worth clipping at all. A moment that scores 75 or above out of 100 becomes a clip. Anything below 75 gets left behind — keep scanning. Never lower the bar to hit a clip count target. Fewer great clips are always better than more mediocre ones.
Here is how you scan:
Hook Strength (25 points) — As you move through the video, ask whether this moment could open a clip with a line that immediately grabs attention and makes complete sense without any prior context. The very first line must establish who, what, and why — without the viewer needing anything from the full video. If the opening raises an unanswered "who?", "where?", "what?" or "when?" — it is not a hook, it is a mid-entry. Never begin with a filler sound like "uh," "um," "ah," or "hmm." A good hook sounds like: "My father was a chain smoker for thirty years — and I made him a deal he couldn't refuse." or "I failed the exam three times. The fourth time changed everything." A bad hook sounds like: "He used to be a chain smoker" — who is he? Or "And that's when I realized..." — realized what? If the moment doesn't have a strong enough natural opening, go back further in the video until it does.
Reframe / Insight (20 points) — Once a strong opening is found, ask whether what follows offers a perspective, idea, or revelation that makes the viewer think or see something differently. This must come from the core of what the video is delivering — not from a transitional, repetitive, or setup segment that only exists to lead into something else. If the moment is purely preamble, it scores low here regardless of how well it opens.
Emotional Resonance (18 points) — Ask whether this moment makes the viewer feel something — curiosity, inspiration, humour, surprise, or genuine emotion that relates to the viewers of the identified category of the video. The feeling must be self-contained. A viewer who has never seen the full video should feel it just as strongly as someone who has watched everything. If the emotion only lands because of what came before in the full video, this moment is not ready to stand alone.
Standalone Clarity (17 points) — At this point, hold the clip as identified so far and ask: does the viewer know what is being talked about, why it matters, and how it ends? If any of these feel missing, extend the clip in either direction until it does. A clip that needs the full video to make sense scores zero here and must be reworked before moving on.
Quotability (12 points) — Ask whether the clip contains a line or moment that someone would want to remember, repeat, or share. This naturally follows when Hook and Insight are strong — but check that the moment lands cleanly without trailing off into filler or an unfinished thought.
Clean Ending (8 points) — Finally, ask whether the clip ends at an emotional or narrative peak — a punchline, a powerful takeaway, a resolved story, or a line that lands with weight. Never end while a thought is still unfinished. Never end on a filler sound like "uh," "um," or "hmm." The last word the viewer hears should feel like the right last word.
Once a clip passes 75, display it with its score in this format: Hook: /25 | Reframe: /20 | Emotion: /18 | Clarity: /17 | Quotability: /12 | Ending: /8 | Total: /100
Scale the number of clips to the video's length. A 10-minute video should yield around 6 clips. Longer videos should produce more, proportionally. But never compromise the 75-point minimum to hit a number — quality always wins.
Finally, review each passing clip from the perspective of the target audience for that video category. Ask: would someone scrolling through their feed stop, watch, and feel completely satisfied by this clip alone? If anything still feels incomplete, thin, or confusing — extend the boundary, find a better start, or replace it with a stronger moment.

Keep each clip between 30 to 90 seconds. Duration is not a creative decision — it is a byproduct of the story. If the moment scores well across all 6 dimensions but the story genuinely needs more time to feel complete and contextual, extend it. Never cut a story short to fit the range, and never pad a clip to reach it. The right duration is whatever the story honestly needs — 30 to 90 seconds is simply where most strong clips naturally land.

Transcript : {windows_text}
"""

SEQUENCE_RESPONSE_FORMAT = {
    "clips": [{"WindowId": "number", "reason": "string"}]
}

NONSEQUENCE_PROMPT = """
{feedback_section}

You are a pro-level video editor and content strategist. Watch the full video, understand its story and tone, then identify the category it belongs to — whether that's educational, podcast, interview, tutorial, inspirational, motivational, humorous, or any other video category.
After identifying, consider yourself an expert in that category. Your job is to assemble the best shareable and engaging short clips for YouTube, Instagram, and TikTok. Unlike straight cuts, you can select multiple separate moments from across the video and join them into one cohesive clip. You do this not by collecting interesting moments and hoping they form a story, but by first deciding the story, emotion, or message you want the clip to deliver — and then hunting across the entire video for the moments that best build and complete it. The story always comes first. The moments serve it.
You decide whether a story is worth assembling by scanning through 6 dimensions. A combination of moments that scores 75 or above out of 100 becomes a clip. Anything below 75 gets left behind — rethink the story or find better moments. Never lower the bar to hit a clip count target. Fewer great clips are always better than more mediocre ones.
Here is how you scan and assemble:
Hook Strength (25 points) — Before anything else, find the single strongest opening moment from anywhere in the video — a line that immediately grabs attention and makes complete sense without any prior context. The very first line must establish who, what, and why — without the viewer needing anything from the full video. If the opening raises an unanswered "who?", "where?", "what?" or "when?" — it is not a hook, it is a mid-entry. Never begin with a filler sound like "uh," "um," "ah," or "hmm." A good hook sounds like: "My father was a chain smoker for thirty years — and I made him a deal he couldn't refuse." or "I failed the exam three times. The fourth time changed everything." A bad hook sounds like: "He used to be a chain smoker" — who is he? Or "And that's when I realized..." — realized what? If no single moment opens strongly enough, go back further in the video or find an earlier moment that sets the context cleanly.
Reframe / Insight (20 points) — With the opening locked, now find the moments from across the video that carry the core idea, perspective, or revelation that makes this story worth telling. Every segment you pull must directly serve this — not transition into it, not repeat it, not set it up from a distance. If a segment is purely preamble or filler, leave it out even if it feels connected. Every second of the assembled clip must be pulling its weight.
Emotional Resonance (18 points) — As you assemble the segments, ask whether the clip as a whole makes the viewer feel something — curiosity, inspiration, humour, surprise, or genuine emotion that relates to the viewers of the identified category of the video. Each individual segment must carry its share of that feeling. The joins between segments must be invisible — the thought, tone, energy, and emotion must flow seamlessly from one to the next. If a join feels jarring or causes the emotion to drop, re-select or reorder until the feeling holds throughout.
Standalone Clarity (17 points) — Hold the assembled clip and ask: does the viewer know what is being talked about, why it matters, and how it ends — without having seen any of the full video? If anything feels missing, find an additional moment from elsewhere in the video to fill the gap. Never leave a join that creates confusion. Never leave a gap that needs the full video to bridge it. Remove all filler sounds — "uh," "um," "ah," "hmm" — from every segment boundary and join. A clip that needs the full video to make sense scores zero here and must be reassembled.
Quotability (12 points) — Ask whether the assembled clip contains a line or moment that someone would want to remember, repeat, or share. This naturally follows when Hook and Insight are strong — but check that the moment lands cleanly without trailing off into filler, a dangling thought, or an awkward join.
Clean Ending (8 points) — Finally, ask whether the clip ends at an emotional or narrative peak — a punchline, a powerful takeaway, a resolved story, or a line that lands with weight. The ending segment must be the strongest possible close from anywhere in the video. Never end while a thought is still unfinished. Never end on a filler sound. The last word the viewer hears should feel like the only right last word.
Once a clip passes 75, display it with its score in this format: Hook: /25 | Reframe: /20 | Emotion: /18 | Clarity: /17 | Quotability: /12 | Ending: /8 | Total: /100
Scale the number of clips to the video's length. A 10-minute video should yield around 6 clips. Longer videos should produce more, proportionally. But never compromise the 75-point minimum to hit a number — quality always wins.
Finally, review each passing clip from the perspective of the target audience for that video category. Ask: would someone scrolling through their feed stop, watch, and feel completely satisfied by this clip alone — with no knowledge of the original video? If anything feels missing, disjointed, or incomplete — re-select segments, adjust the joins, or rethink the story the clip is trying to tell.

Keep each clip between 30 to 90 seconds. Duration is not a creative decision — it is a byproduct of the story. If the moment scores well across all 6 dimensions but the story genuinely needs more time to feel complete and contextual, extend it. Never cut a story short to fit the range, and never pad a clip to reach it. The right duration is whatever the story honestly needs — 30 to 90 seconds is simply where most strong clips naturally land.

{sentences_text}
"""

NONSEQUENCE_RESPONSE_FORMAT = {
    "shorts": [{
        "topic": "string",
        "sentence_ids": ["number"],
        "reason": "string"
    }]
}

SCORER_PROMPT = """
You are a **category-aware social media viewer and analyst**.

Your job is to evaluate how well this short-form video performs **for its specific type of content and audience**, not based on generic virality standards.

---

## SHORT TO SCORE
Type: {clip_type}
Duration: {duration}s
Text: "{text_preview}"

---

## STEP 1: CATEGORY IDENTIFICATION

First, determine the most accurate category for this clip.

### Predefined categories (use if applicable):
- motivation
- business
- storytelling
- humor
- advice
- framework
- emotional
- confrontational
- relatable
- educational
- controversial

---

## STEP 1A: DYNAMIC CATEGORY DETECTION (IF NEEDED)

If the clip does NOT clearly fit into any predefined category:

1. Create a **custom category label** (2–4 words, specific and descriptive)
2. Define internally what makes this format engaging.
3. Establish a **custom evaluation lens**.

---

## SCORING FACTORS (0–20 each, total 100)

Score the clip on each of the SIX dimensions below. Each must be returned as an integer 0–20.

1. **hook_strength** (0–20) — Does the opening line/visual pull the viewer in AND make sense without prior context?
2. **reframe_insight** (0–20) — Does the clip deliver a fresh perspective, idea, or revelation worth thinking about?
3. **emotional_resonance** (0–20) — Does the clip make the viewer feel something (curiosity, inspiration, humour, surprise, etc.) on its own?
4. **standalone_clarity** (0–20) — Can a new viewer understand who/what/why/how-it-ends without the original video? Penalize broken references ("he", "that", etc.).
5. **quotability** (0–20) — Is there a line/moment a viewer would want to remember, repeat, or share?
6. **clean_ending** (0–20) — Does the clip end on a peak (punchline, takeaway, resolution)? Never end on filler or mid-thought.

---

## ADDITIONAL REQUIRED FIELDS

- **category**: final category name
- **reason**: 2–3 sentences justifying the score, category-aware
- **weakest_factor**: name of the lowest-scoring of the six factors above
- **improvise**: 2–3 actionable sentences on how to push this clip to 90+

---

## IMPORTANT RULES

- Do NOT over-reward shock value if clarity suffers
- Do NOT penalize insight-driven clips for low emotion
- Do NOT ignore context gaps or broken flow
- Do NOT inflate scores without justification

Return a fair, context-aware evaluation as JSON.
"""

SCORER_RESPONSE_FORMAT = {
    "hook_strength": "number (0-20)",
    "reframe_insight": "number (0-20)",
    "emotional_resonance": "number (0-20)",
    "standalone_clarity": "number (0-20)",
    "quotability": "number (0-20)",
    "clean_ending": "number (0-20)",
    "category": "string",
    "reason": "string",
    "weakest_factor": "string",
    "improvise": "string",
}

METADATA_PROMPT = """
You are a social media strategist. Generate titles and platform recommendations for {short_count} video shorts.

### LANGUAGE REQUIREMENT

The video transcript language is: **{language_code}**

**CRITICAL: Generate ALL titles in the NATIVE LANGUAGE of the transcript.**
- If language is "en-US" or "en-*" → English titles
- If language is "hi-IN" → Hindi titles (Devanagari script)
- If language is "es-*" → Spanish titles
- If language is "fr-*" → French titles
- If language is "ta-IN" → Tamil titles
- If language is "te-IN" → Telugu titles
- And so on for any other language

The title MUST be in the same language the speaker is using in the transcript.

### TITLE GUIDELINES

- **6-8 words** (short and punchy)
- Capitalize first letter of each word (where applicable for the language)
- Focus on value proposition: What will viewer learn/feel?
- Must be in the native language: {language_code}

### PLATFORM SELECTION

Choose 2-3 platforms per short based on content type:

- **YouTube**: Business, educational, frameworks, longer explanations
- **Instagram**: Relatable, emotional, visual stories, lifestyle
- **TikTok**: Humor, personality, trends, quick hooks
- **LinkedIn**: Professional, business lessons, startup insights

Here are the {short_count} shorts:

{shorts_context}
"""

METADATA_RESPONSE_FORMAT = {
    "shorts": [{
        "short_id": "string",
        "title": "string",
        "platforms": ["string"],
    }]
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str


@dataclass
class CandidateWindow:
    id: int
    start_s: int
    end_s: int
    start_time: float
    end_time: float
    duration: float
    text: str


# ── LLM client ────────────────────────────────────────────────────────────────

_client = None
_token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "calls": [],
}


def _get_client():
    global _client
    if _client is None:
        _client = Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def _track_usage(caller, input_tokens, output_tokens):
    _token_usage["total_input_tokens"] += input_tokens
    _token_usage["total_output_tokens"] += output_tokens
    _token_usage["calls"].append({
        "caller": caller,
        "model": CLAUDE_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })
    log.info(
        f"[LLM] {caller} — in: {input_tokens}, out: {output_tokens}, "
        f"running_total: {_token_usage['total_input_tokens'] + _token_usage['total_output_tokens']}"
    )


def get_token_usage():
    return {
        "total_input_tokens": _token_usage["total_input_tokens"],
        "total_output_tokens": _token_usage["total_output_tokens"],
        "total_tokens": _token_usage["total_input_tokens"] + _token_usage["total_output_tokens"],
        "llm_calls": len(_token_usage["calls"]),
        "calls": _token_usage["calls"],
    }


def _build_prompt(prompt, response_format=None):
    if not response_format:
        return prompt
    return (
        f"{prompt}\n\n"
        "Return only valid JSON. Do not include markdown, code fences, or any text outside the JSON.\n"
        "The JSON response must follow this structure:\n"
        f"{json.dumps(response_format, ensure_ascii=False, indent=2)}"
    )


def _parse_json_response(text):
    try:
        return json.loads(text)
    except ValueError:
        pass

    for fence in ("```json", "```"):
        if fence in text:
            parts = text.split(fence)
            for i in range(1, len(parts)):
                candidate = parts[i].split("```")[0].strip()
                try:
                    return json.loads(candidate)
                except ValueError:
                    continue

    first = text.find("{")
    last = text.rfind("}")
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
    obj_ends = [m.end() for m in re.finditer(r'\}\s*,?\s*', text)]
    if not obj_ends:
        return None
    for end_pos in reversed(obj_ends):
        snippet = text[:end_pos].rstrip().rstrip(",")
        open_braces = snippet.count("{") - snippet.count("}")
        open_brackets = snippet.count("[") - snippet.count("]")
        if open_braces < 0 or open_brackets < 0:
            continue
        closing = "]" * open_brackets + "}" * open_braces
        try:
            result = json.loads(snippet + closing)
            if isinstance(result, dict):
                log.warning(f"Repaired truncated JSON at position {end_pos}/{len(text)}")
                return result
        except ValueError:
            continue
    return None


def get_response(prompt, system_prompt=None, temperature=0.3, response_format=None,
                 max_tokens=None, caller="unknown"):
    client = _get_client()
    full_prompt = _build_prompt(prompt, response_format)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens or CLAUDE_MAX_TOKENS,
        system=system_prompt or "You are a helpful assistant.",
        temperature=temperature,
        messages=[{"role": "user", "content": full_prompt}],
    )
    _track_usage(caller, response.usage.input_tokens, response.usage.output_tokens)
    text = response.content[0].text.strip()
    if not text:
        raise RuntimeError("Claude returned empty response")
    return _parse_json_response(text)


def get_multimodal_response(prompt, frames_b64, temperature=0.3, response_format=None,
                            max_tokens=None, caller="unknown"):
    if not frames_b64:
        return get_response(prompt, temperature=temperature,
                            response_format=response_format,
                            max_tokens=max_tokens, caller=caller)
    client = _get_client()
    full_prompt = _build_prompt(prompt, response_format)
    content = []
    for frame_b64 in frames_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_b64,
            },
        })
    content.append({"type": "text", "text": full_prompt})
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


# ── Frame extraction (local file via ffmpeg) ──────────────────────────────────

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
            "-q:v", "2",
            "-frames:v", str(MAX_FRAMES),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except FileNotFoundError:
            log.error("[FRAMES] ffmpeg not found on PATH — skipping frame extraction")
            return []
        except subprocess.TimeoutExpired:
            log.error("[FRAMES] ffmpeg timed out")
            return []
        except subprocess.CalledProcessError as e:
            log.error(f"[FRAMES] ffmpeg failed: {e.stderr.decode()[:200]}")
            return []

        files = sorted(
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )[:MAX_FRAMES]

        encoded = []
        for p in files:
            try:
                with open(p, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception as e:
                log.error(f"[FRAMES] encode failed for {p}: {repr(e)}")
        log.info(f"[FRAMES] Extracted {len(encoded)} frames")
        return encoded
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Stage 1: Preprocess ───────────────────────────────────────────────────────

def preprocess(transcription_data):
    items = transcription_data.get("items", [])
    log.info(f"[PREPROCESS] Items: {len(items)}")

    sentences = []
    current_words = []
    current_start = None
    current_end = None

    for item in items:
        t = item.get("type")
        if t == "pronunciation":
            word = item["alternatives"][0]["content"]
            start = float(item.get("start_time", 0))
            end = float(item.get("end_time", 0))
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

    language_code = transcription_data.get("language_code", "en-US")
    log.info(f"[PREPROCESS] {len(sentences)} sentences, {video_duration:.1f}s, lang={language_code}")
    return sentences, video_duration, language_code


# ── Stage 2: Segment (candidate windows for sequential) ───────────────────────

def segment(sentences, video_duration):
    min_dur = SEQUENTIAL_MIN_DURATION
    max_dur = SEQUENTIAL_MAX_DURATION
    if video_duration < 180:
        min_dur = max(10, int(video_duration * 0.2))
        max_dur = min(max_dur, int(video_duration * 0.8))
        log.info(f"[SEGMENT] Short video — window range adjusted to {min_dur}-{max_dur}s")

    target = (min_dur + max_dur) / 2
    windows = []
    n = len(sentences)

    for i in range(n):
        best_window = None
        best_diff = float("inf")
        text_parts = []
        for j in range(i, n):
            text_parts.append(sentences[j].text)
            start_time = sentences[i].start
            end_time = sentences[j].end
            duration = round(end_time - start_time, 1)
            if duration > max_dur:
                break
            if duration >= min_dur:
                diff = abs(duration - target)
                if diff < best_diff:
                    best_diff = diff
                    best_window = CandidateWindow(
                        id=0,
                        start_s=sentences[i].index,
                        end_s=sentences[j].index,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        text=" ".join(text_parts),
                    )
        if best_window:
            best_window.id = len(windows) + 1
            windows.append(best_window)

    log.info(f"[SEGMENT] {len(windows)} candidate windows ({min_dur}-{max_dur}s)")
    return windows


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_single_short(short):
    text_preview = " ".join(short["text"].split()[:100]) + "..."
    prompt = SCORER_PROMPT.format(
        clip_type=short["type"],
        duration=short["duration"],
        text_preview=text_preview,
    )
    result = get_response(prompt, response_format=SCORER_RESPONSE_FORMAT, caller="scorer")

    def safe_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    raw_scores = {k: safe_int(result.get(k, 0)) for k in DIMENSION_WEIGHTS.keys()}
    adjusted_scores = {k: min(v + SCORE_MARKET_ADJUSTMENT, 20) for k, v in raw_scores.items()}

    weighted_sum = sum(adjusted_scores[d] * w for d, w in DIMENSION_WEIGHTS.items())
    total_score = round((weighted_sum / 20) * 100)
    weakest_factor = min(raw_scores, key=lambda k: raw_scores[k])

    return {
        "ConfidenceScore": total_score,
        "total_score": total_score,
        "ScoreBreakdown": adjusted_scores,
        "Category": result.get("category", "general"),
        "reason": result.get("reason", ""),
        "weakest_factor": result.get("weakest_factor", weakest_factor),
        "improvise": result.get("improvise", ""),
    }


# ── Stage 3a: Sequential selection (Agent #1 + #3) ────────────────────────────

def _format_windows(windows, max_windows=500):
    if len(windows) > max_windows:
        step = max(1, len(windows) // max_windows)
        thinned = windows[::step][:max_windows]
    else:
        thinned = windows
    lines = []
    for w in thinned:
        words = w.text.split()
        if len(words) <= 25:
            preview = w.text
        else:
            preview = " ".join(words[:15]) + " ... " + " ".join(words[-10:])
        lines.append(f"W{w.id} ({w.duration}s) [{w.start_time:.1f}s - {w.end_time:.1f}s] {preview}")
    return "\n".join(lines)


def _calculate_clips_count(video_duration, available_windows):
    if video_duration <= 0 or available_windows == 0:
        return 1
    avg_clip = (SEQUENTIAL_MIN_DURATION + SEQUENTIAL_MAX_DURATION) / 2
    max_possible = max(1, int(video_duration / avg_clip))
    if video_duration < 180:
        clips = max(1, min(int(video_duration * 0.15 / avg_clip), 12))
    else:
        clips = max(4, min(int(video_duration * 0.15 / avg_clip), 12))
    return min(clips, available_windows, max_possible)


def _agent_sequence(windows, clips_count, regenerate_feedback=None,
                    used_window_ids=None, frames_b64=None):
    if used_window_ids is None:
        used_window_ids = set()
    windows_text = _format_windows(windows)

    feedback_section = ""
    if regenerate_feedback:
        used_text = ", ".join(f"W{wid}" for wid in sorted(used_window_ids)) if used_window_ids else "None yet"
        feedback_section = f"""
### REGENERATION CONTEXT

**CRITICAL: Windows already used (DO NOT SELECT THESE)**:
{used_text}

You must pick a DIFFERENT window that is NOT in the list above.

---

You previously generated a short that scored {regenerate_feedback['score']}/100 (target: {TARGET_SCORE}+).

**Issues identified**:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

**How to improve**: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT window that addresses these issues.
"""
        clips_count = 1

    visual_section = ""
    if frames_b64:
        visual_section = """
### VISUAL CONTEXT PROVIDED

You have access to video frames extracted every 30 seconds. Use these to:
- **Detect interview format**: Look for multiple people in frame (interviewer + guest)
- **Identify speaker changes**: Visual cues when question transitions to answer
- **Verify content type**: Is this Q&A format or monologue?

**CRITICAL for Interview Content**:
- If you see 2+ people in frames → This is likely interview format
- For interview windows: MUST include both question AND answer
- **REJECT any window that is just an interviewer asking a question**
- The valuable content is the GUEST'S RESPONSE, not the question
- A good interview clip: 10% question + 90% answer
"""

    prompt = SEQUENCE_PROMPT.format(
        feedback_section=feedback_section + visual_section,
        windows_text=windows_text,
    )

    if frames_b64:
        result = get_multimodal_response(
            prompt, frames_b64,
            response_format=SEQUENCE_RESPONSE_FORMAT, caller="agent1_sequence",
        )
    else:
        result = get_response(
            prompt, response_format=SEQUENCE_RESPONSE_FORMAT, caller="agent1_sequence",
        )

    window_by_id = {w.id: w for w in windows}
    seq_clips = []
    for item in result.get("clips", []):
        wid = item.get("WindowId")
        if isinstance(wid, str):
            wid = int(wid.replace("W", "").replace("w", "").strip())
        if wid is not None:
            try:
                wid = int(wid)
            except (ValueError, TypeError):
                continue
        if not wid or wid not in window_by_id:
            continue
        w = window_by_id[wid]
        used_window_ids.add(wid)
        seq_clips.append({
            "type": "sequence",
            "text": w.text,
            "start_time": w.start_time,
            "end_time": w.end_time,
            "duration": w.duration,
            "clips": [{"start_time": w.start_time, "end_time": w.end_time, "text": w.text}],
            "reason": item.get("reason", ""),
        })
    return seq_clips


def _agent_sequence_reviewer(seq_shorts, windows, frames_b64=None):
    iteration_log = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed = []
    used_window_ids = set()
    for short in seq_shorts:
        for w in windows:
            if abs(w.start_time - short["start_time"]) < 1.0:
                used_window_ids.add(w.id)
                break

    for idx, short in enumerate(seq_shorts):
        short_id = f"seq_{idx + 1}"
        short["short_id"] = short_id
        current = short
        iterations = 0
        prev_result = None

        while iterations < MAX_ITERATIONS_PER_SHORT:
            score_result = score_single_short(current)
            total = score_result["total_score"]
            log.info(f"{short_id} iter {iterations + 1}: score={total}/100")

            if total >= TARGET_SCORE:
                current.update(score_result)
                current["iterations"] = iterations
                break

            if prev_result:
                prev_weakest = prev_result["weakest_factor"]
                prev_score = prev_result["ScoreBreakdown"].get(prev_weakest, 0)
                new_score = score_result["ScoreBreakdown"].get(prev_weakest, 0)
                if new_score < prev_score:
                    current.update(prev_result)
                    current["iterations"] = iterations
                    iterations += 1
                    iteration_log["iterations"] += 1
                    if iterations >= MAX_ITERATIONS_PER_SHORT:
                        break
                    prev_result = score_result
                    continue

            iterations += 1
            iteration_log["iterations"] += 1
            iteration_log["regenerated"] += 1

            if iterations >= MAX_ITERATIONS_PER_SHORT:
                current.update(score_result)
                current["iterations"] = iterations
                break

            prev_result = score_result
            feedback = {
                "score": total,
                "weakest_factor": score_result["weakest_factor"],
                "reason": score_result["reason"],
                "improvise": score_result["improvise"],
            }
            regenerated = _agent_sequence(
                windows, 1,
                regenerate_feedback=feedback,
                used_window_ids=used_window_ids,
                frames_b64=frames_b64,
            )
            if regenerated:
                current = regenerated[0]
                current["short_id"] = short_id
                for w in windows:
                    if abs(w.start_time - current["start_time"]) < 1.0:
                        used_window_ids.add(w.id)
                        break
            else:
                current.update(score_result)
                current["iterations"] = iterations
                break

        reviewed.append(current)
        iteration_log["final_scores"].append({
            "short_id": short_id,
            "final_score": current.get("ConfidenceScore", 0),
            "iterations": current.get("iterations", 0),
        })

    return reviewed, iteration_log


def sequential_selection(sentences, windows, video_duration, frames_b64=None):
    try:
        clips_count = _calculate_clips_count(video_duration, len(windows))
        log.info(f"[SEQ] Targeting {clips_count} clips from {len(windows)} windows")
        seq_clips = _agent_sequence(windows, clips_count, frames_b64=frames_b64)
        log.info(f"[SEQ] Agent #1 generated {len(seq_clips)} clips")
        reviewed, log_data = _agent_sequence_reviewer(seq_clips, windows, frames_b64=frames_b64)
        log.info(f"[SEQ] Agent #3 reviewed {len(reviewed)}, regenerated {log_data['regenerated']}")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[SEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 3b: Non-sequential selection (Agent #2 + #4) ────────────────────────

def _calculate_target_count(video_duration):
    avg_short_dur = 40
    max_possible = max(1, int(video_duration / avg_short_dur))
    if video_duration < 180:
        target = max(1, min(int(video_duration * 0.12 / avg_short_dur), 10))
    else:
        target = max(4, min(int(video_duration * 0.12 / avg_short_dur), 10))
    return min(target, max_possible)


def _agent_nonsequence(sentences, clips_count, regenerate_feedback=None):
    sentences_text = "\n".join(
        f"S{s.index}: [{s.start:.1f}s - {s.end:.1f}s] {s.text}" for s in sentences
    )

    feedback_section = ""
    if regenerate_feedback:
        old_note = ""
        if "old_sentence_ids" in regenerate_feedback:
            old_note = (
                f"\n**Sentences used in previous attempt**: "
                f"{', '.join(f'S{sid}' for sid in regenerate_feedback['old_sentence_ids'])}"
            )
        feedback_section = f"""
### REGENERATION CONTEXT
{old_note}

You must pick a DIFFERENT set of sentences (avoid the ones above if provided).

---

You previously generated a short that scored {regenerate_feedback['score']}/100 (target: {TARGET_SCORE}+).

**Issues identified**:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

**How to improve**: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT set of sentences that addresses these issues.
"""
        clips_count = 1

    prompt = NONSEQUENCE_PROMPT.format(
        feedback_section=feedback_section,
        sentences_text=sentences_text,
    )

    result = get_response(
        prompt, response_format=NONSEQUENCE_RESPONSE_FORMAT, caller="agent2_nonsequence",
    )

    sentence_by_idx = {s.index: s for s in sentences}
    nonseq_shorts = []
    raw_shorts = result.get("shorts", [])
    log.info(f"[NONSEQ] LLM returned {len(raw_shorts)} raw shorts")

    for short_data in raw_shorts:
        sentence_ids = short_data.get("sentence_ids", [])
        if not sentence_ids:
            continue
        assembled = []
        full_text_parts = []
        for sid in sentence_ids:
            try:
                sid_int = int(sid)
            except (ValueError, TypeError):
                continue
            if sid_int in sentence_by_idx:
                s = sentence_by_idx[sid_int]
                assembled.append({
                    "start_time": s.start,
                    "end_time": s.end,
                    "duration": round(s.end - s.start, 1),
                    "text": s.text,
                })
                full_text_parts.append(s.text)
        if not assembled:
            continue
        total_dur = sum(c["duration"] for c in assembled)
        nonseq_shorts.append({
            "type": "non-sequence",
            "topic": short_data.get("topic", ""),
            "text": " ".join(full_text_parts),
            "start_time": assembled[0]["start_time"],
            "end_time": assembled[-1]["end_time"],
            "duration": round(total_dur, 1),
            "num_clips": len(assembled),
            "clips": assembled,
            "reason": short_data.get("reason", ""),
            "_sentence_ids": [int(sid) for sid in sentence_ids if str(sid).lstrip("-").isdigit()],
        })
    return nonseq_shorts


def _agent_nonsequence_reviewer(nonseq_shorts, sentences):
    iteration_log = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed = []
    for idx, short in enumerate(nonseq_shorts):
        short_id = f"nonseq_{idx + 1}"
        short["short_id"] = short_id
        current = short
        iterations = 0
        prev_result = None

        while iterations < MAX_ITERATIONS_PER_SHORT:
            score_result = score_single_short(current)
            total = score_result["total_score"]
            log.info(f"{short_id} iter {iterations + 1}: score={total}/100")

            if total >= TARGET_SCORE:
                current.update(score_result)
                current["iterations"] = iterations
                break

            if prev_result:
                prev_weakest = prev_result["weakest_factor"]
                prev_score = prev_result["ScoreBreakdown"].get(prev_weakest, 0)
                new_score = score_result["ScoreBreakdown"].get(prev_weakest, 0)
                if new_score < prev_score:
                    current.update(prev_result)
                    current["iterations"] = iterations
                    iterations += 1
                    iteration_log["iterations"] += 1
                    if iterations >= MAX_ITERATIONS_PER_SHORT:
                        break
                    prev_result = score_result
                    continue

            iterations += 1
            iteration_log["iterations"] += 1
            iteration_log["regenerated"] += 1
            if iterations >= MAX_ITERATIONS_PER_SHORT:
                current.update(score_result)
                current["iterations"] = iterations
                break

            prev_result = score_result
            old_sentence_ids = current.get("_sentence_ids", [])
            if not old_sentence_ids:
                for clip in current.get("clips", []):
                    for s in sentences:
                        if abs(s.start - clip["start_time"]) < 0.5:
                            old_sentence_ids.append(s.index)
                            break

            feedback = {
                "score": total,
                "weakest_factor": score_result["weakest_factor"],
                "reason": score_result["reason"],
                "improvise": score_result["improvise"],
                "old_sentence_ids": old_sentence_ids,
            }
            regenerated = _agent_nonsequence(sentences, 1, regenerate_feedback=feedback)
            if regenerated:
                current = regenerated[0]
                current["short_id"] = short_id
            else:
                current.update(score_result)
                current["iterations"] = iterations
                break

        reviewed.append(current)
        iteration_log["final_scores"].append({
            "short_id": short_id,
            "final_score": current.get("ConfidenceScore", 0),
            "iterations": current.get("iterations", 0),
        })

    return reviewed, iteration_log


def non_sequential_selection(sentences, video_duration):
    try:
        if video_duration < NON_SEQUENTIAL_MIN_DURATION:
            log.info(f"[NONSEQ] Video too short ({video_duration}s) — skipping")
            return [], {}
        clips_count = _calculate_target_count(video_duration)
        log.info(f"[NONSEQ] Targeting {clips_count} shorts")
        nonseq_shorts = _agent_nonsequence(sentences, clips_count)
        log.info(f"[NONSEQ] Agent #2 generated {len(nonseq_shorts)} shorts")
        reviewed, log_data = _agent_nonsequence_reviewer(nonseq_shorts, sentences)
        log.info(f"[NONSEQ] Agent #4 reviewed {len(reviewed)}, regenerated {log_data['regenerated']}")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[NONSEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 4: Final metadata (Agent #5) ────────────────────────────────────────

def final_metadata(sequential, non_sequential, language_code):
    all_shorts = sequential + non_sequential
    if not all_shorts:
        return sequential, non_sequential

    parts = []
    for s in all_shorts:
        parts.append(
            f"Short (ID: {s.get('short_id', '')})\n"
            f"Type: {s['type']} | Duration: {s['duration']:.1f}s | Score: {s.get('ConfidenceScore', 0)}/100\n"
            f"Category: {s.get('Category', '')}\n"
            f"Text: \"{' '.join(s['text'].split()[:80])}...\""
        )
    context_text = "\n\n".join(parts)

    prompt = METADATA_PROMPT.format(
        short_count=len(all_shorts),
        shorts_context=context_text,
        language_code=language_code,
    )

    result = get_response(
        prompt, temperature=0.3,
        response_format=METADATA_RESPONSE_FORMAT, caller="agent5_metadata",
    )

    by_id = {m["short_id"]: m for m in result.get("shorts", [])}
    for short in all_shorts:
        sid = short.get("short_id", "")
        if sid in by_id:
            meta = by_id[sid]
            short["Title"] = meta.get("title", "")
            short["SocialMedia"] = meta.get("platforms", ["instagram", "tiktok", "youtube"])
        else:
            short["Title"] = short.get("Title", "Untitled Short")
            short["SocialMedia"] = short.get("SocialMedia", ["instagram", "tiktok", "youtube"])

    seq_count = len(sequential)
    log.info(f"[META] Generated metadata for {len(all_shorts)} shorts")
    return all_shorts[:seq_count], all_shorts[seq_count:]


# ── Stage 5: Ranking ──────────────────────────────────────────────────────────

def ranking(sequential, non_sequential):
    sequential = sorted(sequential, key=lambda c: c.get("start_time", 0))
    non_sequential = sorted(non_sequential, key=lambda s: s.get("ConfidenceScore", 0), reverse=True)
    log.info(f"[RANK] {len(sequential)} sequential, {len(non_sequential)} non-sequential")
    return sequential, non_sequential


# ── Output formatting ─────────────────────────────────────────────────────────

def format_sequential(clips):
    out = []
    for clip in clips:
        out.append({
            "debug_id": clip.get("short_id", ""),
            "title": clip.get("Title", ""),
            "text": clip["text"],
            "confidence_score": clip.get("ConfidenceScore", 0),
            "category": clip.get("Category", "general"),
            "social_media": clip.get("SocialMedia", []),
            "video_start_time": clip["start_time"],
            "video_end_time": clip["end_time"],
            "score_breakdown": clip.get("ScoreBreakdown", {}),
            "reason": clip.get("reason", ""),
            "weakest_factor": clip.get("weakest_factor", ""),
            "iterations": clip.get("iterations", 0),
        })
    return out


def format_non_sequential(shorts):
    out = []
    for short in shorts:
        out.append({
            "short_id": short.get("short_id", ""),
            "debug_id": short.get("short_id", ""),
            "title": short.get("Title", ""),
            "confidence_score": short.get("ConfidenceScore", 0),
            "category": short.get("Category", "general"),
            "social_media": short.get("SocialMedia", []),
            "total_duration": short.get("duration", 0),
            "num_clips": short.get("num_clips", 0),
            "clips": [
                {
                    "startTime": c["start_time"],
                    "endTime": c["end_time"],
                    "duration": c["duration"],
                    "text": c["text"],
                }
                for c in short.get("clips", [])
            ],
            "score_breakdown": short.get("ScoreBreakdown", {}),
            "reason": short.get("reason", ""),
            "weakest_factor": short.get("weakest_factor", ""),
            "iterations": short.get("iterations", 0),
        })
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def run(transcription_data, video_path=None, clip_mode="both"):
    """Programmatic entry point. Returns the result dict."""
    frames_b64 = []
    if video_path:
        frames_b64 = extract_frames_from_file(video_path)
        log.info(f"[AGENT] Multimodal {'enabled' if frames_b64 else 'disabled'}")

    sentences, video_duration, language_code = preprocess(transcription_data)
    windows = segment(sentences, video_duration)

    sequential_clips, seq_log = [], {}
    non_sequential_shorts, nonseq_log = [], {}

    if clip_mode in ("both", "sequential"):
        sequential_clips, seq_log = sequential_selection(
            sentences, windows, video_duration, frames_b64=frames_b64,
        )

    if clip_mode in ("both", "non_sequential"):
        non_sequential_shorts, nonseq_log = non_sequential_selection(sentences, video_duration)

    sequential_clips, non_sequential_shorts = final_metadata(
        sequential_clips, non_sequential_shorts, language_code,
    )
    sequential_clips, non_sequential_shorts = ranking(sequential_clips, non_sequential_shorts)

    return {
        "video_duration": video_duration,
        "language_code": language_code,
        "sentence_count": len(sentences),
        "window_count": len(windows),
        "sequential_clips": format_sequential(sequential_clips),
        "non_sequential_shorts": format_non_sequential(non_sequential_shorts),
        "iteration_log": {
            "sequential": seq_log,
            "non_sequential": nonseq_log,
        },
        "token_usage": get_token_usage(),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the LFTS video clipping agent pipeline (Stages 1–5 + ranking).",
    )
    parser.add_argument(
        "--transcript", "-t",
        default=DEFAULT_TRANSCRIPT_PATH,
        help=f"AWS Transcribe JSON path (default: {DEFAULT_TRANSCRIPT_PATH})",
    )
    parser.add_argument(
        "--video", "-v",
        default=DEFAULT_VIDEO_PATH,
        help=f"Optional local video file for frame extraction (requires ffmpeg) (default: {DEFAULT_VIDEO_PATH})",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--clip-mode",
        choices=("both", "sequential", "non_sequential"),
        default="both",
        help="Which clip types to generate (default: both)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if not CLAUDE_API_KEY:
        raise RuntimeError(
            "CLAUDE_API_KEY is not set. Export it before running:\n"
            '  export CLAUDE_API_KEY="your-key"'
        )

    transcript_path = os.path.abspath(args.transcript)
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(
            f"Transcript not found: {transcript_path}\n"
            f"Pass --transcript or place a file at {DEFAULT_TRANSCRIPT_PATH}"
        )

    video_path = args.video.strip() or None
    if video_path and not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    transcription_data = raw.get("results", raw)

    result = run(transcription_data, video_path=video_path, clip_mode=args.clip_mode)

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(
        f"[DONE] {len(result['sequential_clips'])} sequential, "
        f"{len(result['non_sequential_shorts'])} non-sequential — written to {output_path}"
    )
    return result


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        log.error(str(e))
        sys.exit(1)
