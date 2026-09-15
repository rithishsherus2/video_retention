"""Gemini call for the one generic rule we're starting with: does the
DURING window look visually worse or wrongly rendered compared to the
BEFORE window -- and does that plausibly explain the retention drop.

Evidence is sampled still frames sent inline, NOT full video+audio clips.
This was previously upgraded to real video clips (uploaded via the Files
API) so Gemini could perceive motion and audio-sync directly -- genuinely
better evidence for temporal artifacts (a face morphing across frames,
audio that doesn't sync to the action), and worth reinstating once quota
isn't the constraint. Reverted because the free tier's rate limit is
request-count-sensitive: a video clip needs ~5 requests per event (2
uploads + polling + generate) vs. 1 inline call for a set of images, and
the video-clip version couldn't finish analyzing even a 25-second reel
before hitting the limit. Stills are the deliberate, disclosed tradeoff
for demoing under a free-tier budget, not the intended long-term design.

Deliberately NOT AI-generation-specific. The eventual use case is
reviewing AI-generated film/reels for scenes that aren't "on par" with
their neighbors, but this same comparison also has to work on ordinary
real-footage quality problems (focus, compression, lighting) -- so the
prompt asks Gemini to judge what's visible without assuming which kind of
content it is. More rules get added once this one flow is validated
end-to-end; nothing here should assume it's the only rule that will ever
run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from app.analysis.evidence import DropEvidence

MODEL = "gemini-flash-latest"  # alias -- avoids re-breaking every time Google retires a dated model name
SYNTHESIS_MODEL = "gemini-flash-latest"  # gemini-pro-latest hit the free-tier quota wall immediately in
# testing (429 RESOURCE_EXHAUSTED on the very first call); Flash is text-only reasoning here (no vision
# needed for synthesis) and handles it fine, so there's no reason to fight Pro's free-tier limits for this.


class QualityFinding(BaseModel):
    quality_issue_found: bool = Field(description="Does the DURING window show a genuine visual quality/rendering problem?")
    description: str = Field(description="What specifically looks wrong, in plain language, as a viewer would describe it")
    affected_elements: list[str] = Field(description="What's affected, e.g. 'character face', 'hands', 'background', "
                                                       "'text overlay', 'object shape', 'lighting continuity'. Empty if none.")
    severity: Literal["none", "low", "medium", "high"]
    confidence: float = Field(ge=0, le=1)
    likely_explains_retention_drop: bool = Field(
        description="Does this visual issue plausibly explain why viewers left here, on its own?")
    reasoning: str = Field(description="How the visual evidence connects (or doesn't) to the retention drop")
    other_observations: str = Field(
        description="Anything else notable about why retention may have dropped here, even if it's NOT a "
                     "visual-quality issue (e.g. an abrupt tone/content shift, a jarring audio change). "
                     "Empty string if nothing else stands out.")
    suggestion: str = Field(description="A concrete, actionable fix, or empty string if quality_issue_found is false")


def _get_client():
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()  # picks up backend/.env if present (see .env.example)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY in backend/.env "
            "(get a free key at https://aistudio.google.com/apikey)."
        )
    return genai.Client(api_key=api_key)


def _build_prompt(evidence: DropEvidence) -> str:
    s = evidence.numeric_summary
    return f"""You are reviewing a short video (a reel) to understand why viewer retention dropped sharply at one point.

You are shown two sets of still frames sampled from the video, in order (these are still images, not video -- you
cannot perceive motion or hear audio directly; judge only what's visible in each frame, and use the numeric audio
signals below for anything audio-related):
- BEFORE: {len(evidence.before_frame_paths)} frame(s) from just before the drop window (t={s['before_window_s'][0]}-{s['before_window_s'][1]}s)
- DURING: {len(evidence.during_frame_paths)} frame(s) from within the drop window itself (t={s['during_window_s'][0]}-{s['during_window_s'][1]}s)

Retention went from {s['retention_before_pct']}% to {s['retention_after_pct']}% of viewers still watching across this window
({s['pct_points_lost']} percentage points lost) -- much steeper than this video's own typical pace (z={s['peak_drop_z']}, where >1.5 is already unusual for this video).

YOUR TASK: compare the DURING frames against the BEFORE frames as a viewer would. Judge whether the DURING frames
show a genuine visual quality or rendering problem -- reduced clarity/sharpness, blurred or malformed
characters/faces/hands/objects, broken continuity, warped geometry, texture breakdown, or anything that looks like
it wasn't shot/rendered/generated properly. Do NOT assume the footage is real or AI-generated -- judge only on what
is visible. A content/subject change alone (new location, new shot) is NOT a quality issue by itself; only flag it
if something about how it's rendered looks wrong.

Supporting numeric signals -- including audio levels and transcript, since you can't hear audio yourself. Your
visual judgment on the actual frames is what matters most for the quality question, but use these numbers to
reason about non-visual causes too (e.g. an audio jump or silence) and mention them in other_observations if
relevant:
{json.dumps(s, indent=2)}

Respond with the structured fields requested."""


def _generate_structured(model: str, contents: list, response_schema: type[BaseModel], max_retries: int = 6, client=None):
    from google.genai import errors, types

    client = client or _get_client()
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return response_schema.model_validate_json(response.text)
        except errors.ClientError as e:
            # 429 quota-exhausted is NOT transient like a 503 -- retrying
            # with backoff just burns more requests against an already-
            # exhausted budget (this is exactly what happened during
            # testing: repeated retries on 429 made the wait longer, not
            # shorter). Fail fast with a clear message instead.
            if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                raise RuntimeError(
                    "Gemini free-tier quota exhausted (429). This does not resolve by retrying -- "
                    "wait for the quota window to reset (check https://ai.dev/rate-limit for timing) "
                    "or enable billing to raise the limit."
                ) from e
            raise
        except errors.ServerError as e:
            # Free-tier 503 "high demand" is transient and common -- worth
            # a short backoff-and-retry rather than failing the whole run
            # over one busy moment.
            last_err = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"    Gemini overloaded (attempt {attempt + 1}/{max_retries}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_err


def analyze_drop_event(evidence: DropEvidence, max_retries: int = 6) -> QualityFinding:
    prompt = _build_prompt(evidence)
    contents: list = [prompt, "BEFORE frames:"]
    contents += [Image.open(p) for p in evidence.before_frame_paths]
    contents.append("DURING frames:")
    contents += [Image.open(p) for p in evidence.during_frame_paths]
    return _generate_structured(MODEL, contents, QualityFinding, max_retries)


class RankedFinding(BaseModel):
    rank: int
    window: str = Field(description="e.g. 't=2-6s'")
    pct_points_lost: float
    cause_summary: str = Field(description="One or two sentences: the most likely reason viewers left here")
    evidence: str = Field(description="What specifically supports this -- cite the actual numbers/observations given")
    suggestion: str = Field(description="A concrete, actionable fix for this specific moment")


class SynthesisReport(BaseModel):
    headline: str = Field(description="One sentence: the single biggest reason this reel lost viewers")
    overview: str = Field(description="2-4 sentence narrative of the whole reel's retention story, start to finish")
    ranked_findings: list[RankedFinding] = Field(description="Drop events ranked by actual impact (pct points lost), biggest first")
    overall_suggestions: list[str] = Field(description="Cross-cutting recommendations beyond the per-event fixes above "
                                                         "-- things that would help the reel as a whole")


def _build_synthesis_prompt(video_summary: str, per_event_results: list[dict]) -> str:
    return f"""You are writing the final analysis report for a short video (a reel), explaining why viewer
retention dropped where it did and what to do about it. You're given:

1. A per-second digest of the whole reel (retention, cuts, quality flags, audio, transcript):
---
{video_summary}
---

2. Structured findings already produced for each major drop event (a vision model already compared the actual
video frames before/during each drop and recorded what it saw -- treat "other_observations" in each as real
evidence, not speculation, since it came from looking at the actual footage):
---
{json.dumps(per_event_results, indent=2)}
---

YOUR TASK: synthesize this into one coherent, prioritized report. Rank the drop events by how many percentage
points of retention they actually cost (biggest impact first, not necessarily chronological order). For each,
give a concrete, specific reason grounded in the evidence above -- not generic advice. Then give a small number
of cross-cutting suggestions that address the reel as a whole, not just one moment.

IMPORTANT: if an event's JSON has a non-null "likely_skip" field, retention recovered shortly afterward --
this is consistent with viewers skipping ahead rather than genuinely disengaging. Describe it as a likely
skip, not full disengagement, and don't overstate its severity as if the audience left for good, but still
suggest why that segment specifically might have invited a skip (it's still worth improving).

Write for someone who made this reel and wants to know exactly what to change next time. Be direct and specific
(reference actual timestamps and numbers), not vague ("improve pacing" is not useful; "the opening text card
holds for 3+ seconds with no motion or audio before the first cut -- cut that to under 1.5s" is)."""


def synthesize_report(video_summary: str, per_event_results: list[dict]) -> SynthesisReport:
    prompt = _build_synthesis_prompt(video_summary, per_event_results)
    return _generate_structured(SYNTHESIS_MODEL, [prompt], SynthesisReport)


def render_report_markdown(report: SynthesisReport) -> str:
    lines = [
        "# Retention Analysis Report", "",
        f"**{report.headline}**", "",
        report.overview, "",
        "## Drop Events (ranked by impact)", "",
    ]
    for f in report.ranked_findings:
        lines += [
            f"### {f.rank}. {f.window} -- {f.pct_points_lost} percentage points lost", "",
            f"**Why:** {f.cause_summary}", "",
            f"**Evidence:** {f.evidence}", "",
            f"**Suggestion:** {f.suggestion}", "",
        ]
    lines.append("## Overall Suggestions")
    lines.append("")
    for s in report.overall_suggestions:
        lines.append(f"- {s}")
    return "\n".join(lines)
