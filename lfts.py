#!/usr/bin/env python3
"""Standalone runner for the video clipping agent pipeline.

Runs Stages 1–6 end-to-end:
    Preprocess → Category Detection → Segment →
    Sequential (Agent #1 + #3) → Non-Sequential (Agent #2 + #4) →
    Final Metadata (Agent #5) → Ranking → write result JSON

Changes from v1:
    - Category + adaptive weights detected from transcript (new Stage 1b)
    - Scorer dimensions aligned with selector rubric (6 dims, 0-20 each)
    - SCORE_MARKET_ADJUSTMENT removed (was inflating scores artificially)
    - Model string fixed to claude-haiku-4-5-20251001
    - Language code extraction fixed for AWS Transcribe payload shape
    - Non-sequential duration guard added (30–90s enforced)
    - Iteration improvement check uses total score, not single-dimension
    - Selector prompts now receive category + audience + weight reasoning
    - "Watch the full video" wording replaced with "Read the transcript"
    - Non-sequential regeneration tracks used sentence IDs to avoid repeats
    - Weight normalisation + fallback guard in category detector

Quick start:
    pip install anthropic
    export CLAUDE_API_KEY="your-key"
    python lfts.py

Or pass paths on the CLI:
    python lfts.py --transcript /path/to/transcript.json --output result.json
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
DEFAULT_VIDEO_PATH      = os.path.join(_SCRIPT_DIR, "input.mp4")
DEFAULT_OUTPUT_PATH     = os.path.join(_SCRIPT_DIR, "result.json")


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("clipping_agent")


# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE_API_KEY   = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL     = "claude-haiku-4-5-20251001"   # FIX: date suffix required
CLAUDE_MAX_TOKENS = 8192

SEQUENTIAL_MIN_DURATION     = 25
SEQUENTIAL_MAX_DURATION     = 55
NON_SEQUENTIAL_MIN_DURATION = 25
NON_SEQUENTIAL_MAX_DURATION = 90

TARGET_SCORE             = 90
MAX_ITERATIONS_PER_SHORT = 3

# How many sentences to sample for category detection
CATEGORY_SAMPLE_SENTENCES = 60

FRAME_INTERVAL = 30
MAX_FRAMES     = 20

# Default fallback weights (used only if category detection fails entirely)
DEFAULT_DIMENSION_WEIGHTS = {
    "hook_strength":      0.25,
    "reframe_insight":    0.20,
    "emotional_resonance":0.18,
    "standalone_clarity": 0.17,
    "quotability":        0.12,
    "clean_ending":       0.08,
}


# ── Prompts ───────────────────────────────────────────────────────────────────

# ── Stage 1b: Category + weight detection ─────────────────────────────────────
CATEGORY_WEIGHT_PROMPT = """
You are a social media content strategist and audience psychologist.

Read the transcript excerpt below and determine:

1. What kind of video this is — not just a label, but the specific
   audience it serves and what they came to feel or gain from it.

2. Based on that audience expectation, assign weights to the 6 scoring
   dimensions below. The weights must sum to exactly 1.0.
   Distribute them based on what this specific audience rewards most —
   not a generic formula.

## The 6 dimensions

- hook_strength: Does the opening grab and orient the viewer instantly?
- reframe_insight: Does it offer a fresh idea or perspective worth thinking about?
- emotional_resonance: Does it make the viewer feel something on its own?
- standalone_clarity: Can a new viewer follow it without the full video?
- quotability: Is there a line they would repeat or share?
- clean_ending: Does it land at a peak — punchline, resolution, or takeaway?

## Reasoning you must apply

Ask yourself: if someone scrolls past this clip in 2 seconds and keeps
scrolling — what dimension failed them? That dimension deserves the
highest weight. Then ask: what dimension is almost irrelevant for this
audience? That gets the lowest weight.

Examples of how weight logic changes by content:
- Comedy clip: viewer left because the punchline didn't land or the hook
  was slow → hook_strength and clean_ending dominate
- Tutorial: viewer left because they couldn't follow the steps without
  context → standalone_clarity dominates
- Motivational story: viewer left because it didn't make them feel
  anything → emotional_resonance dominates
- Interview insight: viewer left because the idea wasn't fresh enough →
  reframe_insight dominates

Do not apply the same weights to every video. The weights must reflect
this specific transcript's audience and purpose.

## Transcript excerpt
{transcript_text}
"""

CATEGORY_WEIGHT_RESPONSE_FORMAT = {
    "category": "string (2-4 words, specific — e.g. 'AI industry keynote' not just 'educational')",
    "audience": "string (1 sentence — who watches this and why)",
    "weights": {
        "hook_strength":       "float",
        "reframe_insight":     "float",
        "emotional_resonance": "float",
        "standalone_clarity":  "float",
        "quotability":         "float",
        "clean_ending":        "float",
    },
    "weight_reasoning": "string (2-3 sentences explaining why these weights fit this specific video)",
}


# ── Stage 3a: Sequential selector ────────────────────────────────────────────
SEQUENCE_PROMPT = """
{feedback_section}

## Video profile
Category: {category}
Audience: {audience}
What this audience rewards most: {weight_reasoning}

You are a pro-level video editor and content strategist. Read the full
transcript below, understand its story and tone, then apply the video
profile above to find the best shareable and engaging short clips for
YouTube, Instagram, and TikTok.

You do this not by reading and then picking, but by scanning the
transcript through 6 dimensions that together tell you whether a moment
is worth clipping. A moment that scores 75 or above out of 100 becomes
a clip. Anything below 75 gets left behind. Never lower the bar to hit
a clip count target. Fewer great clips are always better than more
mediocre ones.

Here is how you scan:

Hook Strength (25 points) — Ask whether this moment could open a clip
with a line that immediately grabs attention and makes complete sense
without any prior context. The very first line must establish who, what,
and why. If the opening raises an unanswered "who?", "where?", "what?"
or "when?" it is not a hook — it is a mid-entry. Never begin with a
filler sound like "uh," "um," "ah," or "hmm." If the moment does not
have a strong enough natural opening, go back further in the transcript
until it does.

Reframe / Insight (20 points) — Ask whether what follows offers a
perspective, idea, or revelation that makes the viewer think or see
something differently. This must come from the core of what the video
is delivering — not a transitional or setup segment.

Emotional Resonance (18 points) — Ask whether this moment makes the
viewer feel something — curiosity, inspiration, humour, surprise, or
genuine emotion relevant to the video's audience. The feeling must be
self-contained without the full video.

Standalone Clarity (17 points) — Does the viewer know what is being
talked about, why it matters, and how it ends? If anything feels
missing, extend the clip in either direction until it does. A clip that
needs the full video to make sense scores zero here.

Quotability (12 points) — Does the clip contain a line or moment
someone would want to remember, repeat, or share?

Clean Ending (8 points) — Does the clip end at an emotional or
narrative peak — a punchline, a powerful takeaway, a resolved story?
Never end on a filler sound or an unfinished thought.

Once a clip passes 75, display it with its score in this format:
Hook: /25 | Reframe: /20 | Emotion: /18 | Clarity: /17 | Quotability: /12 | Ending: /8 | Total: /100

Scale the number of clips to the video length. A 10-minute video yields
around 6 clips. Longer videos produce more proportionally. Quality
always wins over quantity.

Keep each clip between 30 to 90 seconds. Duration is a byproduct of the
story — never cut short to fit the range, never pad to reach it.

Transcript:
{windows_text}
"""

SEQUENCE_RESPONSE_FORMAT = {
    "clips": [{"WindowId": "number", "reason": "string"}]
}


# ── Stage 3b: Non-sequential assembler ───────────────────────────────────────
NONSEQUENCE_PROMPT = """
{feedback_section}

## Video profile
Category: {category}
Audience: {audience}
What this audience rewards most: {weight_reasoning}

You are a pro-level video editor and content strategist. Read the full
transcript below, understand its story and tone, then apply the video
profile above to assemble the best shareable and engaging short clips
for YouTube, Instagram, and TikTok.

Unlike straight cuts, you can select multiple separate moments from
across the transcript and join them into one cohesive clip. Always
decide the story first — then hunt for the moments that build it.

Scan through 6 dimensions. A combination scoring 75 or above becomes
a clip. Anything below 75 gets left behind.

Hook Strength (25 points) — Find the single strongest opening moment
from anywhere in the transcript. The very first line must establish
who, what, and why — without context from the full video. Never begin
with a filler sound.

Reframe / Insight (20 points) — Find the moments that carry the core
idea or revelation. Every segment you pull must directly serve this —
not transition into it or repeat it. Every second must pull its weight.

Emotional Resonance (18 points) — The assembled clip as a whole must
make the viewer feel something. Each segment carries its share of that
feeling. Joins between segments must be invisible — tone and energy
must flow seamlessly.

Standalone Clarity (17 points) — Does the viewer know what is being
talked about, why it matters, and how it ends — without having seen
any of the full video? Remove all filler sounds at every segment
boundary.

Quotability (12 points) — Is there a line someone would remember,
repeat, or share? Check it lands cleanly without trailing off.

Clean Ending (8 points) — Does the clip end at an emotional or
narrative peak? The ending segment must be the strongest possible
close from anywhere in the transcript.

IMPORTANT: The total duration of selected sentences must be between
30 and 90 seconds. Do not select sentences whose combined duration
falls outside this range.

Scale the number of clips to the video length proportionally.
Quality always wins over quantity.

{sentences_text}
"""

NONSEQUENCE_RESPONSE_FORMAT = {
    "shorts": [{
        "topic":        "string",
        "sentence_ids": ["number"],
        "reason":       "string",
    }]
}


# ── Scorer ────────────────────────────────────────────────────────────────────
SCORER_PROMPT = """
You are a category-aware social media viewer and analyst.

Your job is to evaluate how well this short-form video clip performs
for its specific type of content and audience.

## Clip to score
Category: {category}
Audience: {audience}
Type: {clip_type}
Duration: {duration}s
Text: "{text_preview}"

## What this audience rewards most
{weight_reasoning}

## Scoring — 6 dimensions, 0 to 20 each

Score each dimension as an integer 0–20. Apply your judgement through
the lens of the audience described above — what matters most for them
should be scored most critically.

1. hook_strength (0-20): Does the opening line pull the viewer in AND
   make complete sense without prior context?

2. reframe_insight (0-20): Does the clip deliver a fresh perspective,
   idea, or revelation worth thinking about?

3. emotional_resonance (0-20): Does the clip make the viewer feel
   something — curiosity, inspiration, humour, surprise — on its own?

4. standalone_clarity (0-20): Can a new viewer understand who, what,
   why, and how it ends without the original video? Penalise broken
   references ("he", "that", "as I mentioned").

5. quotability (0-20): Is there a line or moment a viewer would want
   to remember, repeat, or share?

6. clean_ending (0-20): Does the clip end on a peak — punchline,
   takeaway, resolution? Never end on filler or a mid-thought.

## Additional required fields

- reason: 2-3 sentences justifying the total score, category-aware
- weakest_factor: name of the lowest-scoring dimension
- improvise: 2-3 actionable sentences on how to push this clip to 90+

## Rules

- Do not over-reward shock value if clarity suffers
- Do not penalise insight-driven clips for low emotion
- Do not ignore context gaps or broken flow
- Do not inflate scores without justification

Return a fair, context-aware evaluation as JSON.
"""

SCORER_RESPONSE_FORMAT = {
    "hook_strength":       "integer 0-20",
    "reframe_insight":     "integer 0-20",
    "emotional_resonance": "integer 0-20",
    "standalone_clarity":  "integer 0-20",
    "quotability":         "integer 0-20",
    "clean_ending":        "integer 0-20",
    "reason":              "string",
    "weakest_factor":      "string",
    "improvise":           "string",
}


# ── Metadata ──────────────────────────────────────────────────────────────────
METADATA_PROMPT = """
You are a social media strategist. Generate titles and platform
recommendations for {short_count} video shorts.

### Language requirement

The video transcript language is: {language_code}

Generate ALL titles in the native language of the transcript.
- en-US or en-* → English
- hi-IN → Hindi (Devanagari script)
- es-* → Spanish
- fr-* → French
- ta-IN → Tamil
- te-IN → Telugu

### Title guidelines

- 6-8 words, short and punchy
- Capitalise first letter of each word where applicable
- Focus on value: what will the viewer learn or feel?

### Platform selection — choose 2-3 per short

- YouTube: business, educational, frameworks, longer explanations
- Instagram: relatable, emotional, visual stories, lifestyle
- TikTok: humour, personality, trends, quick hooks
- LinkedIn: professional, business lessons, startup insights

Here are the {short_count} shorts:

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
    index: int
    start: float
    end:   float
    text:  str


@dataclass
class CandidateWindow:
    id:         int
    start_s:    int
    end_s:      int
    start_time: float
    end_time:   float
    duration:   float
    text:       str


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
        f"[LLM] {caller} — in: {input_tokens}, out: {output_tokens}, "
        f"running_total: {_token_usage['total_input_tokens'] + _token_usage['total_output_tokens']}"
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
        "Return only valid JSON. Do not include markdown, code fences, "
        "or any text outside the JSON.\n"
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
    last  = text.rfind("}")
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
        snippet       = text[:end_pos].rstrip().rstrip(",")
        open_braces   = snippet.count("{") - snippet.count("}")
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


def get_response(prompt, system_prompt=None, temperature=0.3,
                 response_format=None, max_tokens=None, caller="unknown"):
    client      = _get_client()
    full_prompt = _build_prompt(prompt, response_format)
    response    = client.messages.create(
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


def get_multimodal_response(prompt, frames_b64, temperature=0.3,
                            response_format=None, max_tokens=None, caller="unknown"):
    if not frames_b64:
        return get_response(prompt, temperature=temperature,
                            response_format=response_format,
                            max_tokens=max_tokens, caller=caller)
    client      = _get_client()
    full_prompt = _build_prompt(prompt, response_format)
    content     = []
    for frame_b64 in frames_b64:
        content.append({
            "type":   "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       frame_b64,
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
            "-q:v", "2",
            "-frames:v", str(MAX_FRAMES),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except FileNotFoundError:
            log.error("[FRAMES] ffmpeg not found — skipping frame extraction")
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
    """
    Accepts either:
      - the raw AWS Transcribe payload  (top-level key "results")
      - the inner "results" dict directly
    """
    items = transcription_data.get("items", [])
    log.info(f"[PREPROCESS] Items: {len(items)}")

    sentences    = []
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
                        end=round(current_end,   2),
                        text=" ".join(current_words),
                    ))
                current_words, current_start, current_end = [], None, None

    if current_words and current_start is not None:
        sentences.append(Sentence(
            index=len(sentences) + 1,
            start=round(current_start, 2),
            end=round(current_end,   2),
            text=" ".join(current_words),
        ))

    video_duration = 0.0
    for item in items:
        et = item.get("end_time")
        if et:
            video_duration = max(video_duration, float(et))

    # FIX: AWS Transcribe puts language under language_identification[],
    # not as a top-level "language_code" key on the results dict.
    language_code = "en-US"  # safe default
    lang_id = transcription_data.get("language_identification")
    if lang_id and isinstance(lang_id, list) and lang_id:
        language_code = lang_id[0].get("code", "en-US")
    elif transcription_data.get("language_code"):
        # some payloads do include it at top level
        language_code = transcription_data["language_code"]

    log.info(
        f"[PREPROCESS] {len(sentences)} sentences, "
        f"{video_duration:.1f}s, lang={language_code}"
    )
    return sentences, video_duration, language_code


# ── Stage 1b: Category + adaptive weight detection ────────────────────────────

def detect_category_weights(sentences):
    """
    Reads a sample of the transcript and asks the LLM to:
      - identify the specific video category and target audience
      - produce dimension weights that sum to 1.0 for that audience

    Returns a profile dict used by the scorer and selector agents.
    """
    sample         = sentences[:CATEGORY_SAMPLE_SENTENCES]
    transcript_text = " ".join(s.text for s in sample)

    prompt = CATEGORY_WEIGHT_PROMPT.format(transcript_text=transcript_text)

    try:
        result = get_response(
            prompt,
            response_format=CATEGORY_WEIGHT_RESPONSE_FORMAT,
            caller="category_detector",
        )
    except Exception as e:
        log.error(f"[CATEGORY] Detection failed: {repr(e)} — using defaults")
        return _default_category_profile()

    print("CATEGORY WEIGHT RESPONSE RESULT", result)
    
    weights = result.get("weights", {})

    # Validate all 6 dimensions present
    required = set(DEFAULT_DIMENSION_WEIGHTS.keys())
    if not required.issubset(set(weights.keys())):
        log.warning("[CATEGORY] Incomplete weights returned — using defaults")
        return _default_category_profile()

    # Coerce to float
    try:
        weights = {k: float(v) for k, v in weights.items()}
    except (ValueError, TypeError):
        log.warning("[CATEGORY] Non-numeric weights — using defaults")
        return _default_category_profile()

    # Normalise to sum = 1.0
    total = sum(weights.values())
    if total <= 0:
        log.warning("[CATEGORY] Zero-sum weights — using defaults")
        return _default_category_profile()

    if abs(total - 1.0) > 0.01:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
        log.warning(f"[CATEGORY] Weights renormalised from {total:.3f} to 1.0")

    profile = {
        "category":         result.get("category",         "general"),
        "audience":         result.get("audience",         "general audience"),
        "weights":          weights,
        "weight_reasoning": result.get("weight_reasoning", ""),
    }

    log.info(
        f"[CATEGORY] '{profile['category']}' — "
        f"top weight: {max(weights, key=weights.get)} ({max(weights.values()):.2f})"
    )
    log.info(f"[CATEGORY] Audience: {profile['audience']}")
    log.info(f"[CATEGORY] Reasoning: {profile['weight_reasoning']}")
    return profile


def _default_category_profile():
    return {
        "category":         "general",
        "audience":         "general audience",
        "weights":          DEFAULT_DIMENSION_WEIGHTS.copy(),
        "weight_reasoning": "Default weights applied — category detection unavailable.",
    }


# ── Stage 2: Segment ──────────────────────────────────────────────────────────

def segment(sentences, video_duration):
    min_dur = SEQUENTIAL_MIN_DURATION
    max_dur = SEQUENTIAL_MAX_DURATION
    if video_duration < 180:
        min_dur = max(10, int(video_duration * 0.2))
        max_dur = min(max_dur, int(video_duration * 0.8))
        log.info(f"[SEGMENT] Short video — window range adjusted to {min_dur}-{max_dur}s")

    target  = (min_dur + max_dur) / 2
    windows = []
    n       = len(sentences)

    for i in range(n):
        best_window = None
        best_diff   = float("inf")
        text_parts  = []
        for j in range(i, n):
            text_parts.append(sentences[j].text)
            start_time = sentences[i].start
            end_time   = sentences[j].end
            duration   = round(end_time - start_time, 1)
            if duration > max_dur:
                break
            if duration >= min_dur:
                diff = abs(duration - target)
                if diff < best_diff:
                    best_diff   = diff
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

def score_single_short(short, category_profile):
    """
    Scores a clip using the 6-dimension rubric.
    Weights are taken from the detected category profile, not a global constant.
    No artificial score inflation (SCORE_MARKET_ADJUSTMENT removed).
    Iteration improvement check uses total score comparison.
    """
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

    result = get_response(
        prompt,
        response_format=SCORER_RESPONSE_FORMAT,
        caller="scorer",
    )

    def safe_int(v):
        try:
            return max(0, min(20, int(v)))
        except (ValueError, TypeError):
            return 0

    dim_keys = list(DEFAULT_DIMENSION_WEIGHTS.keys())
    raw_scores = {k: safe_int(result.get(k, 0)) for k in dim_keys}

    # Weighted sum — no market adjustment, no artificial inflation
    weighted_sum = sum(raw_scores[d] * weights.get(d, 1/6) for d in dim_keys)
    total_score  = round((weighted_sum / 20) * 100)
    weakest      = min(raw_scores, key=lambda k: raw_scores[k])

    return {
        "ConfidenceScore": total_score,
        "total_score":     total_score,
        "ScoreBreakdown":  raw_scores,
        "Category":        category_profile["category"],
        "reason":          result.get("reason",         ""),
        "weakest_factor":  result.get("weakest_factor", weakest),
        "improvise":       result.get("improvise",      ""),
    }


# ── Stage 3a: Sequential selection ───────────────────────────────────────────

def _format_windows(windows, max_windows=500):
    if len(windows) > max_windows:
        step   = max(1, len(windows) // max_windows)
        thinned = windows[::step][:max_windows]
    else:
        thinned = windows
    lines = []
    for w in thinned:
        # Pass full text so the model has all content — not a truncated preview
        lines.append(
            f"W{w.id} ({w.duration}s) "
            f"[{w.start_time:.1f}s - {w.end_time:.1f}s] {w.text}"
        )
    return "\n".join(lines)


def _dedup_overlapping_clips(clips, overlap_threshold=0.8):
    if not clips:
        return clips

    sorted_clips = sorted(clips, key=lambda c: (c["start_time"], c["end_time"]))
    deduped = []

    for clip in sorted_clips:
        clip_start = clip["start_time"]
        clip_end = clip["end_time"]
        clip_duration = clip_end - clip_start

        is_duplicate = False
        for existing in deduped:
            existing_start = existing["start_time"]
            existing_end = existing["end_time"]

            overlap_start = max(clip_start, existing_start)
            overlap_end = min(clip_end, existing_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            if overlap_duration > 0:
                overlap_ratio = overlap_duration / clip_duration
                if overlap_ratio >= overlap_threshold:
                    is_duplicate = True
                    break

        if not is_duplicate:
            deduped.append(clip)

    log.info(f"[DEDUP] Removed {len(clips) - len(deduped)} duplicate/overlapping clips "
             f"({len(deduped)} unique clips remain)")
    return deduped


def _calculate_clips_count(video_duration, available_windows):
    if video_duration <= 0 or available_windows == 0:
        return 1
    avg_clip    = (SEQUENTIAL_MIN_DURATION + SEQUENTIAL_MAX_DURATION) / 2
    max_possible = max(1, int(video_duration / avg_clip))
    if video_duration < 180:
        clips = max(1, min(int(video_duration * 0.15 / avg_clip), 12))
    else:
        clips = max(4, min(int(video_duration * 0.15 / avg_clip), 12))
    return min(clips, available_windows, max_possible)


def _agent_sequence(windows, clips_count, category_profile,
                    regenerate_feedback=None, used_window_ids=None,
                    frames_b64=None):
    if used_window_ids is None:
        used_window_ids = set()

    windows_text     = _format_windows(windows)
    feedback_section = ""

    if regenerate_feedback:
        used_text = (
            ", ".join(f"W{wid}" for wid in sorted(used_window_ids))
            if used_window_ids else "None yet"
        )
        feedback_section = f"""
### Regeneration context

**Windows already used (DO NOT SELECT THESE)**: {used_text}

You previously generated a short that scored {regenerate_feedback['score']}/100
(target: {TARGET_SCORE}+).

Issues identified:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

How to improve: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT window that addresses these issues.
"""
        clips_count = 1

    visual_section = ""
    if frames_b64:
        visual_section = """
### Visual context provided

You have access to video frames extracted every 30 seconds. Use these to:
- Detect interview format (multiple people in frame)
- Identify speaker changes
- Verify content type (Q&A vs monologue)

For interview content: MUST include both question AND answer.
REJECT any window that is just an interviewer asking a question.
"""

    prompt = SEQUENCE_PROMPT.format(
        feedback_section=feedback_section + visual_section,
        category=category_profile["category"],
        audience=category_profile["audience"],
        weight_reasoning=category_profile["weight_reasoning"],
        windows_text=windows_text,
    )

    if frames_b64:
        result = get_multimodal_response(
            prompt, frames_b64,
            response_format=SEQUENCE_RESPONSE_FORMAT,
            caller="agent1_sequence",
        )
    else:
        result = get_response(
            prompt,
            response_format=SEQUENCE_RESPONSE_FORMAT,
            caller="agent1_sequence",
        )

    window_by_id = {w.id: w for w in windows}
    seq_clips    = []

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
            "type":       "sequence",
            "text":       w.text,
            "start_time": w.start_time,
            "end_time":   w.end_time,
            "duration":   w.duration,
            "clips": [{"start_time": w.start_time,
                       "end_time":   w.end_time,
                       "text":       w.text}],
            "reason": item.get("reason", ""),
        })
    return seq_clips


def _agent_sequence_reviewer(seq_shorts, windows, category_profile,
                              frames_b64=None):
    iteration_log   = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed        = []
    used_window_ids = set()

    for short in seq_shorts:
        for w in windows:
            if abs(w.start_time - short["start_time"]) < 1.0:
                used_window_ids.add(w.id)
                break

    for idx, short in enumerate(seq_shorts):
        short_id = f"seq_{idx + 1}"
        short["short_id"] = short_id
        current     = short
        iterations  = 0
        best_result = None   # track best scoring version seen

        while iterations < MAX_ITERATIONS_PER_SHORT:
            score_result = score_single_short(current, category_profile)
            total        = score_result["total_score"]
            log.info(f"{short_id} iter {iterations + 1}: score={total}/100")

            # Keep track of the best version seen so far
            if best_result is None or total > best_result["total_score"]:
                best_result = score_result
                best_result["_clip"] = dict(current)

            if total >= TARGET_SCORE:
                current.update(score_result)
                current["iterations"] = iterations
                break

            iterations += 1
            iteration_log["iterations"]  += 1
            iteration_log["regenerated"] += 1

            if iterations >= MAX_ITERATIONS_PER_SHORT:
                # Use the best version seen, not necessarily the last
                best_clip = best_result.pop("_clip", current)
                best_clip.update({k: v for k, v in best_result.items()
                                  if k != "_clip"})
                current = best_clip
                current["iterations"] = iterations
                break

            feedback = {
                "score":          total,
                "weakest_factor": score_result["weakest_factor"],
                "reason":         score_result["reason"],
                "improvise":      score_result["improvise"],
            }
            regenerated = _agent_sequence(
                windows, 1,
                category_profile=category_profile,
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
            "short_id":    short_id,
            "final_score": current.get("ConfidenceScore", 0),
            "iterations":  current.get("iterations",      0),
        })

    return reviewed, iteration_log


def sequential_selection(sentences, windows, video_duration,
                         category_profile, frames_b64=None):
    try:
        clips_count = _calculate_clips_count(video_duration, len(windows))
        log.info(f"[SEQ] Targeting {clips_count} clips from {len(windows)} windows")
        seq_clips = _agent_sequence(
            windows, clips_count,
            category_profile=category_profile,
            frames_b64=frames_b64,
        )
        log.info(f"[SEQ] Agent #1 generated {len(seq_clips)} clips")
        seq_clips = _dedup_overlapping_clips(seq_clips, overlap_threshold=0.75)
        reviewed, log_data = _agent_sequence_reviewer(
            seq_clips, windows,
            category_profile=category_profile,
            frames_b64=frames_b64,
        )
        log.info(
            f"[SEQ] Agent #3 reviewed {len(reviewed)}, "
            f"regenerated {log_data['regenerated']}"
        )
        reviewed = _dedup_overlapping_clips(reviewed, overlap_threshold=0.75)
        log.info(f"[SEQ] After dedup: {len(reviewed)} unique clips")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[SEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 3b: Non-sequential selection ───────────────────────────────────────

def _dedup_nonsequential_shorts(shorts):
    if not shorts:
        return shorts

    deduped = []
    seen_clip_sets = set()

    for short in shorts:
        clips = short.get("clips", [])
        clip_key = tuple(sorted((c["start_time"], c["end_time"]) for c in clips))

        if clip_key not in seen_clip_sets:
            seen_clip_sets.add(clip_key)
            deduped.append(short)

    log.info(f"[NONSEQ-DEDUP] Removed {len(shorts) - len(deduped)} duplicate assembled shorts "
             f"({len(deduped)} unique remain)")
    return deduped


def _calculate_target_count(video_duration):
    avg_short_dur = 40
    max_possible  = max(1, int(video_duration / avg_short_dur))
    if video_duration < 180:
        target = max(1, min(int(video_duration * 0.12 / avg_short_dur), 10))
    else:
        target = max(4, min(int(video_duration * 0.12 / avg_short_dur), 10))
    return min(target, max_possible)


def _agent_nonsequence(sentences, clips_count, category_profile,
                       regenerate_feedback=None, used_sentence_ids=None):
    sentences_text = "\n".join(
        f"S{s.index}: [{s.start:.1f}s - {s.end:.1f}s] {s.text}"
        for s in sentences
    )

    feedback_section = ""
    if regenerate_feedback:
        old_ids  = regenerate_feedback.get("old_sentence_ids", [])
        old_note = (
            f"\n**Sentences used in previous attempt**: "
            f"{', '.join(f'S{sid}' for sid in old_ids)}"
            if old_ids else ""
        )
        # Also exclude all previously used sentence IDs across iterations
        all_used = used_sentence_ids or set()
        if all_used:
            old_note += (
                f"\n**All sentences used so far (avoid these)**: "
                f"{', '.join(f'S{sid}' for sid in sorted(all_used))}"
            )
        feedback_section = f"""
### Regeneration context
{old_note}

You must pick a DIFFERENT set of sentences (avoid the ones listed above).

You previously generated a short that scored {regenerate_feedback['score']}/100
(target: {TARGET_SCORE}+).

Issues identified:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

How to improve: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT set of sentences that addresses these issues.
"""
        clips_count = 1

    prompt = NONSEQUENCE_PROMPT.format(
        feedback_section=feedback_section,
        category=category_profile["category"],
        audience=category_profile["audience"],
        weight_reasoning=category_profile["weight_reasoning"],
        sentences_text=sentences_text,
    )

    result = get_response(
        prompt,
        response_format=NONSEQUENCE_RESPONSE_FORMAT,
        caller="agent2_nonsequence",
    )

    sentence_by_idx = {s.index: s for s in sentences}
    nonseq_shorts   = []
    raw_shorts      = result.get("shorts", [])
    log.info(f"[NONSEQ] LLM returned {len(raw_shorts)} raw shorts")

    for short_data in raw_shorts:
        sentence_ids = short_data.get("sentence_ids", [])
        if not sentence_ids:
            continue

        assembled       = []
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
                    "end_time":   s.end,
                    "duration":   round(s.end - s.start, 1),
                    "text":       s.text,
                })
                full_text_parts.append(s.text)

        if not assembled:
            continue

        total_dur = sum(c["duration"] for c in assembled)

        # FIX: enforce 30–90s duration range on non-sequential clips
        if total_dur < NON_SEQUENTIAL_MIN_DURATION:
            log.warning(
                f"[NONSEQ] Clip too short ({total_dur:.1f}s < "
                f"{NON_SEQUENTIAL_MIN_DURATION}s) — skipping"
            )
            continue
        if total_dur > NON_SEQUENTIAL_MAX_DURATION:
            log.warning(
                f"[NONSEQ] Clip too long ({total_dur:.1f}s > "
                f"{NON_SEQUENTIAL_MAX_DURATION}s) — skipping"
            )
            continue

        nonseq_shorts.append({
            "type":          "non-sequence",
            "topic":         short_data.get("topic", ""),
            "text":          " ".join(full_text_parts),
            "start_time":    assembled[0]["start_time"],
            "end_time":      assembled[-1]["end_time"],
            "duration":      round(total_dur, 1),
            "num_clips":     len(assembled),
            "clips":         assembled,
            "reason":        short_data.get("reason", ""),
            "_sentence_ids": [
                int(sid) for sid in sentence_ids
                if str(sid).lstrip("-").isdigit()
            ],
        })

    return nonseq_shorts


def _agent_nonsequence_reviewer(nonseq_shorts, sentences, category_profile):
    iteration_log    = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed         = []
    # FIX: track used sentence IDs across all shorts and iterations
    all_used_ids: set = set()

    for idx, short in enumerate(nonseq_shorts):
        short_id = f"nonseq_{idx + 1}"
        short["short_id"] = short_id
        current     = short
        iterations  = 0
        best_result = None

        # Register initial sentence IDs as used
        for sid in current.get("_sentence_ids", []):
            all_used_ids.add(sid)

        while iterations < MAX_ITERATIONS_PER_SHORT:
            score_result = score_single_short(current, category_profile)
            total        = score_result["total_score"]
            log.info(f"{short_id} iter {iterations + 1}: score={total}/100")

            if best_result is None or total > best_result["total_score"]:
                best_result = score_result
                best_result["_clip"] = dict(current)

            if total >= TARGET_SCORE:
                current.update(score_result)
                current["iterations"] = iterations
                break

            iterations += 1
            iteration_log["iterations"]  += 1
            iteration_log["regenerated"] += 1

            if iterations >= MAX_ITERATIONS_PER_SHORT:
                best_clip = best_result.pop("_clip", current)
                best_clip.update({k: v for k, v in best_result.items()
                                  if k != "_clip"})
                current = best_clip
                current["iterations"] = iterations
                break

            old_sentence_ids = current.get("_sentence_ids", [])
            if not old_sentence_ids:
                for clip in current.get("clips", []):
                    for s in sentences:
                        if abs(s.start - clip["start_time"]) < 0.5:
                            old_sentence_ids.append(s.index)
                            break

            feedback = {
                "score":            total,
                "weakest_factor":   score_result["weakest_factor"],
                "reason":           score_result["reason"],
                "improvise":        score_result["improvise"],
                "old_sentence_ids": old_sentence_ids,
            }
            regenerated = _agent_nonsequence(
                sentences, 1,
                category_profile=category_profile,
                regenerate_feedback=feedback,
                used_sentence_ids=all_used_ids,
            )
            if regenerated:
                current = regenerated[0]
                current["short_id"] = short_id
                for sid in current.get("_sentence_ids", []):
                    all_used_ids.add(sid)
            else:
                current.update(score_result)
                current["iterations"] = iterations
                break

        reviewed.append(current)
        iteration_log["final_scores"].append({
            "short_id":    short_id,
            "final_score": current.get("ConfidenceScore", 0),
            "iterations":  current.get("iterations",      0),
        })

    return reviewed, iteration_log


def non_sequential_selection(sentences, video_duration, category_profile):
    try:
        if video_duration < NON_SEQUENTIAL_MIN_DURATION:
            log.info(f"[NONSEQ] Video too short ({video_duration}s) — skipping")
            return [], {}
        clips_count = _calculate_target_count(video_duration)
        log.info(f"[NONSEQ] Targeting {clips_count} shorts")
        nonseq_shorts = _agent_nonsequence(
            sentences, clips_count,
            category_profile=category_profile,
        )
        log.info(f"[NONSEQ] Agent #2 generated {len(nonseq_shorts)} shorts")
        reviewed, log_data = _agent_nonsequence_reviewer(
            nonseq_shorts, sentences,
            category_profile=category_profile,
        )
        log.info(
            f"[NONSEQ] Agent #4 reviewed {len(reviewed)}, "
            f"regenerated {log_data['regenerated']}"
        )
        reviewed = _dedup_nonsequential_shorts(reviewed)
        return reviewed, log_data
    except Exception as e:
        log.error(f"[NONSEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 4: Final metadata ───────────────────────────────────────────────────

def final_metadata(sequential, non_sequential, language_code):
    all_shorts = sequential + non_sequential
    if not all_shorts:
        return sequential, non_sequential

    parts = []
    for s in all_shorts:
        parts.append(
            f"Short (ID: {s.get('short_id', '')})\n"
            f"Type: {s['type']} | Duration: {s['duration']:.1f}s "
            f"| Score: {s.get('ConfidenceScore', 0)}/100\n"
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
        response_format=METADATA_RESPONSE_FORMAT,
        caller="agent5_metadata",
    )

    by_id = {m["short_id"]: m for m in result.get("shorts", [])}
    for short in all_shorts:
        sid = short.get("short_id", "")
        if sid in by_id:
            meta = by_id[sid]
            short["Title"]       = meta.get("title",     "")
            short["SocialMedia"] = meta.get("platforms", ["instagram", "tiktok", "youtube"])
        else:
            short["Title"]       = short.get("Title",       "Untitled Short")
            short["SocialMedia"] = short.get("SocialMedia", ["instagram", "tiktok", "youtube"])

    seq_count = len(sequential)
    log.info(f"[META] Generated metadata for {len(all_shorts)} shorts")
    return all_shorts[:seq_count], all_shorts[seq_count:]


# ── Stage 5: Ranking ──────────────────────────────────────────────────────────

def ranking(sequential, non_sequential):
    sequential     = sorted(sequential,     key=lambda c: c.get("start_time",      0))
    non_sequential = sorted(non_sequential, key=lambda s: s.get("ConfidenceScore", 0), reverse=True)
    log.info(f"[RANK] {len(sequential)} sequential, {len(non_sequential)} non-sequential")
    return sequential, non_sequential


# ── Output formatting ─────────────────────────────────────────────────────────

def format_sequential(clips):
    out = []
    for clip in clips:
        out.append({
            "debug_id":         clip.get("short_id", ""),
            "title":            clip.get("Title", ""),
            "text":             clip["text"],
            "confidence_score": clip.get("ConfidenceScore", 0),
            "category":         clip.get("Category", "general"),
            "social_media":     clip.get("SocialMedia", []),
            "video_start_time": clip["start_time"],
            "video_end_time":   clip["end_time"],
            "score_breakdown":  clip.get("ScoreBreakdown", {}),
            "reason":           clip.get("reason",         ""),
            "weakest_factor":   clip.get("weakest_factor", ""),
            "iterations":       clip.get("iterations",     0),
        })
    return out


def format_non_sequential(shorts):
    out = []
    for short in shorts:
        out.append({
            "short_id":         short.get("short_id", ""),
            "debug_id":         short.get("short_id", ""),
            "title":            short.get("Title",    ""),
            "confidence_score": short.get("ConfidenceScore", 0),
            "category":         short.get("Category", "general"),
            "social_media":     short.get("SocialMedia", []),
            "total_duration":   short.get("duration",  0),
            "num_clips":        short.get("num_clips", 0),
            "clips": [
                {
                    "startTime": c["start_time"],
                    "endTime":   c["end_time"],
                    "duration":  c["duration"],
                    "text":      c["text"],
                }
                for c in short.get("clips", [])
            ],
            "score_breakdown":  short.get("ScoreBreakdown", {}),
            "reason":           short.get("reason",         ""),
            "weakest_factor":   short.get("weakest_factor", ""),
            "iterations":       short.get("iterations",     0),
        })
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def run(transcription_data, video_path=None, clip_mode="both"):
    """Programmatic entry point. Returns the full result dict."""

    # Optional frame extraction (requires ffmpeg on PATH)
    frames_b64 = []
    if video_path:
        frames_b64 = extract_frames_from_file(video_path)
        log.info(f"[AGENT] Multimodal {'enabled' if frames_b64 else 'disabled'}")

    # Stage 1: Preprocess
    sentences, video_duration, language_code = preprocess(transcription_data)

    # Stage 1b: Detect category + adaptive weights from transcript
    category_profile = detect_category_weights(sentences)

    # Stage 2: Segment (sequential candidate windows)
    windows = segment(sentences, video_duration)

    sequential_clips,      seq_log     = [], {}
    non_sequential_shorts, nonseq_log  = [], {}

    # Stage 3a: Sequential
    if clip_mode in ("both", "sequential"):
        sequential_clips, seq_log = sequential_selection(
            sentences, windows, video_duration,
            category_profile=category_profile,
            frames_b64=frames_b64,
        )

    # Stage 3b: Non-sequential
    if clip_mode in ("both", "non_sequential"):
        non_sequential_shorts, nonseq_log = non_sequential_selection(
            sentences, video_duration,
            category_profile=category_profile,
        )

    # Stage 4: Metadata
    sequential_clips, non_sequential_shorts = final_metadata(
        sequential_clips, non_sequential_shorts, language_code,
    )

    # Stage 5: Ranking
    sequential_clips, non_sequential_shorts = ranking(
        sequential_clips, non_sequential_shorts,
    )

    return {
        "video_duration":          video_duration,
        "language_code":           language_code,
        "sentence_count":          len(sentences),
        "window_count":            len(windows),
        "category_profile":        {
            "category":         category_profile["category"],
            "audience":         category_profile["audience"],
            "weights":          category_profile["weights"],
            "weight_reasoning": category_profile["weight_reasoning"],
        },
        "sequential_clips":        format_sequential(sequential_clips),
        "non_sequential_shorts":   format_non_sequential(non_sequential_shorts),
        "iteration_log": {
            "sequential":     seq_log,
            "non_sequential": nonseq_log,
        },
        "token_usage": get_token_usage(),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="LFTS video clipping agent pipeline (Stages 1–5 + ranking).",
    )
    parser.add_argument(
        "--transcript", "-t",
        default=DEFAULT_TRANSCRIPT_PATH,
        help=f"AWS Transcribe JSON (default: {DEFAULT_TRANSCRIPT_PATH})",
    )
    parser.add_argument(
        "--video", "-v",
        default=DEFAULT_VIDEO_PATH,
        help="Optional local video for frame extraction (requires ffmpeg)",
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

    video_path = (args.video or "").strip() or None
    if video_path and not os.path.isfile(video_path):
        log.warning(f"[MAIN] Video not found at {video_path} — skipping frames")
        video_path = None

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Accept both raw AWS payload and pre-extracted "results" dict
    transcription_data = raw.get("results", raw)

    result = run(transcription_data, video_path=video_path, clip_mode=args.clip_mode)

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(
        f"[DONE] {len(result['sequential_clips'])} sequential, "
        f"{len(result['non_sequential_shorts'])} non-sequential — "
        f"written to {output_path}"
    )
    log.info(
        f"[DONE] Category detected: {result['category_profile']['category']}"
    )
    log.info(
        f"[DONE] Total tokens used: {result['token_usage']['total_tokens']}"
    )
    return result


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, KeyboardInterrupt) as e:
        log.error(str(e))
        sys.exit(1)
