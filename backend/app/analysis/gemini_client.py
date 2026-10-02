"""All Gemini calls for this project, in two families:

1. The original per-drop-event quality check: does the DURING window look
   visually worse or wrongly rendered compared to the BEFORE window, and
   does that plausibly explain the retention drop (QualityFinding /
   analyze_drop_event), synthesized across events into one report
   (SynthesisReport / synthesize_report). Still used by the legacy
   /api/analyze flow (app.analysis.run) when there's no ideal-reel profile
   to compare against.

2. The decode-ideal-reels-into-rules flow and its counterpart, critiquing
   a new reel against those rules (see app.analysis.rules / app.analysis.
   evaluate for the orchestration): ReelDecoding/decode_reel describes one
   reel's hook/pacing/storytelling/audio/technique; ContentRules/
   synthesize_content_rules finds the pattern shared across several ideal
   reels; RuleBasedCritique/critique_against_rules checks a test reel
   against that pattern, optionally folding in the per-drop findings from
   family 1 when a retention recording is available.

Evidence for vision calls is sampled still frames sent inline, NOT full
video+audio clips. This was previously upgraded to real video clips
(uploaded via the Files API) so Gemini could perceive motion and
audio-sync directly -- genuinely better evidence for temporal artifacts
(a face morphing across frames, audio that doesn't sync to the action),
and worth reinstating once quota isn't the constraint. Reverted because
the free tier's rate limit is request-count-sensitive: a video clip needs
~5 requests per event (2 uploads + polling + generate) vs. 1 inline call
for a set of images, and the video-clip version couldn't finish analyzing
even a 25-second reel before hitting the limit. Stills are the deliberate,
disclosed tradeoff for demoing under a free-tier budget, not the intended
long-term design. Multiple GEMINI_API_KEYS can be configured to rotate
past a single key's quota wall (see _load_api_keys) -- that raises the
effective budget without changing this tradeoff.

The quality-check prompt is deliberately NOT AI-generation-specific: the
eventual use case is reviewing AI-generated film/reels for scenes that
aren't "on par" with their neighbors, but this same comparison also has to
work on ordinary real-footage quality problems (focus, compression,
lighting) -- so it asks Gemini to judge what's visible without assuming
which kind of content it is.
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


def _load_api_keys() -> list[str]:
    """GEMINI_API_KEYS="key1,key2,key3" for a rotation pool, falling back
    to the single-key GEMINI_API_KEY/GOOGLE_API_KEY for back-compat. Free
    per-key quotas are small enough that one key alone regularly runs out
    mid-run (see the 429 handling below) -- a second/third key from a
    different Google account is the practical fix, not a retry loop."""
    from dotenv import load_dotenv

    load_dotenv()  # picks up backend/.env if present (see .env.example)
    multi = os.environ.get("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    if not keys:
        single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if single:
            keys = [single]
    if not keys:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GEMINI_API_KEYS=key1,key2,... to "
            "rotate across several) in backend/.env (get a free key at https://aistudio.google.com/apikey)."
        )
    return keys


REQUEST_TIMEOUT_MS = 180_000  # 3 min -- finite (without this the underlying HTTP call has no
# timeout at all and a stalled connection hangs forever instead of failing and letting
# try_each() rotate to the next key -- confirmed happening in practice: a call sat with zero
# progress for 2+ hours before being killed by hand), but generous enough for the
# decode_reel call specifically, which can carry 100+ inline images (see app.analysis.rules'
# DECODE_FRAMES_PER_SECOND) -- 90s wasn't enough even under normal load and produced
# ReadTimeouts indistinguishable from genuine server overload (503s) in testing.


def _get_client(api_key: str):
    from google import genai
    from google.genai import types

    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))


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


def _generate_structured(model: str, contents: list, response_schema: type[BaseModel], max_retries: int = 3):
    import httpx
    from google.genai import errors, types

    from app.infra.rotation import ExhaustedError, try_each

    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)

    def attempt_with_key(api_key: str) -> BaseModel:
        client = _get_client(api_key)
        last_err = None
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                return response_schema.model_validate_json(response.text)
            except errors.ClientError as e:
                # 429 quota-exhausted is NOT transient like a 503 -- retrying
                # the SAME key with backoff just burns more requests against
                # an already-exhausted budget. Signal the caller to rotate
                # to the next key instead of retrying this one.
                if getattr(e, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(e):
                    raise ExhaustedError(f"quota exhausted (429): {e}") from e
                raise
            except (errors.ServerError, httpx.TimeoutException, httpx.TransportError) as e:
                # Free-tier 503 "high demand" is transient and common, and so is a
                # stalled/timed-out connection (genai's own HTTP layer doesn't catch
                # or wrap these -- an unhandled one previously hung a whole run for
                # hours instead of failing, since nothing here caught it and nothing
                # rotated to another key). Both get the same treatment: a short
                # backoff-and-retry on the SAME key first (a different key doesn't
                # fix an overloaded model OR a flaky network path), and only after
                # max_retries do we give up on this key and let the caller rotate.
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 3
                    print(f"    Gemini call failed ({type(e).__name__}), attempt {attempt + 1}/{max_retries}, "
                          f"retrying in {wait}s...")
                    time.sleep(wait)
        raise ExhaustedError(f"exhausted {max_retries} retries: {last_err}") from last_err

    keys = _load_api_keys()
    try:
        return try_each(keys, attempt_with_key, label="Gemini API key")
    except RuntimeError as e:
        raise RuntimeError(
            f"{e} This does not resolve by retrying -- wait for the quota window to reset "
            "(check https://ai.dev/rate-limit for timing), add more keys to GEMINI_API_KEYS, "
            "or enable billing to raise the limit."
        ) from e


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


# ---------------------------------------------------------------------------
# Rule extraction from "ideal" reels + rule-based critique of a test reel.
#
# This is the decode-then-apply flow: a handful of reels that already
# performed well get decoded individually (pacing/hook/storytelling/audio/
# technical execution, grounded in representative frames + deterministic
# stats -- not guessed from the caption alone), then synthesized into one
# reusable rule set for that page/niche. A later reel is decoded the same
# way and checked against that rule set, which is what lets the critique
# be specific ("your hook takes 4s to resolve, the ideal reels resolve
# theirs within 1-2s") instead of generic video-editing advice.
# ---------------------------------------------------------------------------

class ReelDecoding(BaseModel):
    hook_description: str = Field(description="What happens in the first 1-3 seconds and why it does or doesn't grab attention")
    storytelling_arc: str = Field(description="The narrative/structural arc start to finish, e.g. problem->agitation->payoff, "
                                                "listicle, before/after, tutorial steps, punchline-last")
    pacing_style: str = Field(description="How cuts/pacing are used -- fast-cut vs. long takes, rhythm changes over the "
                                            "reel, how cut timing relates to the audio/beat")
    tone_and_style: str = Field(description="Visual/tonal style: aesthetic, energy level, humor vs. seriousness, "
                                              "any recognizable editing trend used")
    audio_strategy: str = Field(description="How music/voiceover/sound design is used and how it supports the content")
    text_overlay_and_captions: str = Field(description="How on-screen text/captions are used, if visible -- timing, "
                                                          "density, purpose. Say so if none is visible")
    notable_techniques: list[str] = Field(description="Specific, reusable techniques worth naming as a rule, e.g. "
                                                        "'text hook appears within first 1s', 'pattern interrupt at 3s'")
    call_to_action: str = Field(description="Any explicit or implicit CTA / loop-back / engagement bait present, "
                                              "empty string if none")


def _build_decode_prompt(features: dict, engagement: dict, n_frames: int) -> str:
    return f"""You are reverse-engineering the creative and technical approach of a short vertical video (a reel),
so its techniques can be turned into reusable rules later. You are shown {n_frames} representative still frames
sampled evenly across the whole reel, in order from start to finish (these are still images -- judge visual style,
framing, text overlays, and apparent pacing from them; you cannot hear audio, so use the transcript/audio stats
below for anything audio-related).

Deterministic stats already measured from the actual video (cuts, shot durations, dialogue, technical quality):
---
{json.dumps(features, indent=2)}
---

Engagement this reel actually received, for context on how well it performed:
---
{json.dumps(engagement, indent=2)}
---

YOUR TASK: describe, concretely and specifically, HOW this reel is made -- not just what it's about. Focus on the
hook, storytelling structure, pacing, tone, audio strategy, text overlay usage, and any other technique that a
creator could deliberately repeat. Ground your description in the frames and stats given; don't invent details you
can't see or infer from the numbers.

Respond with the structured fields requested."""


def decode_reel(features: dict, engagement: dict, frame_paths: list[Path]) -> ReelDecoding:
    prompt = _build_decode_prompt(features, engagement, len(frame_paths))
    contents: list = [prompt, "Representative frames, in chronological order:"]
    contents += [Image.open(p) for p in frame_paths]
    return _generate_structured(MODEL, contents, ReelDecoding)


class ContentRule(BaseModel):
    category: Literal["hook", "pacing", "storytelling", "audio", "visual_style", "text_overlay", "cta",
                       "technical_quality", "other"]
    rule: str = Field(description="A concrete, checkable rule distilled across the ideal reels, e.g. 'Hook resolves "
                                    "within 2 seconds via on-screen text plus motion, never a static opening frame'")
    rationale: str = Field(description="Why this pattern likely drives views/engagement, grounded in what was "
                                         "actually observed across the ideal reels -- not generic best-practice")


class _GeneratedContentRules(BaseModel):
    """Exactly what Gemini fills in via structured output. benchmark_stats
    deliberately isn't a field here: Gemini's Developer API rejects a
    generic dict's schema outright (additionalProperties unsupported,
    confirmed by a real 'ValueError: additionalProperties is only
    supported in Gemini Enterprise Agent Platform mode' when it was tried
    here) -- and even a properly typed nested model for it would let the
    model regenerate/second-guess numbers that must come from deterministic
    computation, never generation. See ContentRules.benchmark_stats below,
    which wraps this with that field attached in code, not by the model."""
    profile_summary: str = Field(description="2-4 sentence summary of this account/niche's winning formula across "
                                                "the analyzed reels")
    rules: list[ContentRule]
    benchmark_duration_s: str = Field(description="Typical duration range observed across the ideal reels, e.g. '18-27s'")
    benchmark_engagement: str = Field(description="Typical views/likes/comments and engagement rate observed across "
                                                     "the ideal reels, in plain language, grounded in the exact "
                                                     "benchmark_stats given in the prompt -- e.g. 'views average "
                                                     "~X, median ~Y, ranging Z1-Z2'")


class EngagementStatSummary(BaseModel):
    avg: float
    median: float
    min: float
    max: float


class EngagementBenchmarkStats(BaseModel):
    n_reels: int
    views: EngagementStatSummary | None = None
    likes: EngagementStatSummary | None = None
    comments: EngagementStatSummary | None = None
    engagement_rate_pct: EngagementStatSummary | None = None


class ContentRules(_GeneratedContentRules):
    benchmark_stats: EngagementBenchmarkStats = Field(
        default_factory=lambda: EngagementBenchmarkStats(n_reels=0),
        description="Deterministic avg/median/min/max engagement computed directly from the ideal reels' own data "
                     "-- always attached in code after generation, never produced by the model itself.",
    )


def _build_rules_synthesis_prompt(decoded_reels: list[dict], benchmark_stats: dict) -> str:
    n_reels = benchmark_stats.get("n_reels", len(decoded_reels))
    return f"""You are extracting a reusable content strategy from a small set of reels from the SAME Instagram
account/niche that all performed well. Each entry below is one reel: its URL, deterministic technical stats, the
engagement it received, and a structured decoding of its creative approach (hook, pacing, storytelling, audio, text
overlay, technique list) produced by actually reviewing its frames.
---
{json.dumps(decoded_reels, indent=2)}
---

Deterministic engagement statistics computed directly from these {n_reels} reels' own numbers (avg/median/min/max
for views, likes, comments, and engagement rate) -- use these EXACT numbers for benchmark_engagement, do not
recompute, round loosely, or contradict them:
---
{json.dumps(benchmark_stats, indent=2)}
---

YOUR TASK: find the PATTERNS these reels share -- not a description of any single one -- and turn them into a small
set of concrete, checkable rules a creator (or an automated checker) could apply to a NEW reel from this same
account/niche. Each rule must be specific enough to check against a new reel's own stats/decoding (e.g. "first cut
lands within 2 seconds" is checkable; "good pacing" is not). Where the reels disagree, prefer the pattern shared by
the majority, and don't force a rule if there isn't a real shared pattern for that category -- fewer, real rules
beat padding the list.

IMPORTANT: with only {n_reels} reels, a single unusually low or high view count is normal small-sample variance,
NOT evidence of a recurring category or segment. Do not invent a categorical explanation for it (e.g. labeling one
reel a "niche case" versus another "iconic" one) unless the reels are actually about obviously, describably
different kinds of subjects AND you say so explicitly grounded in what's different about them -- otherwise just
report the plain average/median and the min-max spread, without implying the spread reflects a stable pattern a
future reel should be sorted into."""


def synthesize_content_rules(decoded_reels: list[dict], benchmark_stats: dict) -> ContentRules:
    prompt = _build_rules_synthesis_prompt(decoded_reels, benchmark_stats)
    generated = _generate_structured(SYNTHESIS_MODEL, [prompt], _GeneratedContentRules)
    return ContentRules(**generated.model_dump(), benchmark_stats=benchmark_stats)


class RuleViolation(BaseModel):
    category: str
    rule: str = Field(description="The specific profile rule this relates to")
    expected: str = Field(description="What the rule calls for")
    observed: str = Field(description="What this reel actually does, grounded in its own stats/decoding")
    impact: Literal["low", "medium", "high"]
    suggestion: str = Field(description="A concrete, actionable fix")


class RuleBasedCritique(BaseModel):
    headline: str = Field(description="One sentence: the single biggest gap versus the ideal-reel rules")
    engagement_assessment: str = Field(description="How this reel's views/likes/comments compare to the profile's "
                                                      "own benchmark, and what that gap (or lack of one) suggests "
                                                      "about audience interest -- specific and numeric where possible")
    violations: list[RuleViolation] = Field(description="Rules this reel breaks or only partially follows, ranked "
                                                           "by impact, biggest first")
    strengths: list[str] = Field(description="What this reel already does right, matching the ideal-reel rules")
    overall_suggestions: list[str] = Field(description="Cross-cutting recommendations beyond the per-rule fixes above")


def _build_critique_prompt(
    rules: ContentRules, test_features: dict, test_decoding: ReelDecoding, test_engagement: dict,
    drop_findings: list[dict] | None,
) -> str:
    drop_section = ""
    if drop_findings:
        drop_section = f"""
A retention recording WAS provided for this reel, and a separate pass already analyzed specific moments where
viewers dropped off, comparing actual before/during video frames (treat these as real evidence, not speculation):
---
{json.dumps(drop_findings, indent=2)}
---
Use these to make your violations/suggestions concrete about WHEN in the reel a rule breaks down, not just that
it does.
"""
    else:
        drop_section = """
No retention recording was provided for this reel -- there is no second-by-second drop-off data. Base your
critique entirely on how this reel's own stats/decoding compare to the rule set below, and be explicit in
engagement_assessment that these are the most likely general reasons for the views/engagement observed, since
no retention curve was available to pinpoint exact moments.
"""

    return f"""You are critiquing a new reel against a rule set already extracted from other reels on the SAME
Instagram account/niche that performed well. Explain concretely why this reel is likely underperforming (or not)
relative to that established pattern.

RULE SET (profile summary, checkable rules, and this account's own normal duration/engagement range):
---
{json.dumps(rules.model_dump(), indent=2)}
---

THIS REEL's deterministic stats:
---
{json.dumps(test_features, indent=2)}
---

THIS REEL's decoded creative approach (from its own frames):
---
{json.dumps(test_decoding.model_dump(), indent=2)}
---

THIS REEL's actual engagement:
---
{json.dumps(test_engagement, indent=2)}
---
{drop_section}
YOUR TASK: go rule by rule (only the ones that matter here -- don't manufacture a violation for a rule this reel
already satisfies) and identify concrete gaps between what the rule set calls for and what this reel actually does.
For engagement_assessment, compare this reel's actual views/likes/comments against the rule set's benchmark_stats
field specifically (avg, median, min-max -- deterministic numbers, not the benchmark_engagement prose) since that's
the exact, computed ground truth; don't invent or repeat a categorical explanation (like "niche" vs "iconic") for
why the benchmark spread looks the way it does unless the rule set itself already grounds that explicitly in a real
difference between the reels, not just a number being lower or higher. Be direct and specific (cite actual
numbers/timestamps/stats from above), and end with a small number of cross-cutting suggestions. Write for someone
who made this reel and wants to know exactly what to change next time."""


def critique_against_rules(
    rules: ContentRules, test_decoding: ReelDecoding, test_features: dict, test_engagement: dict,
    drop_findings: list[dict] | None = None,
) -> RuleBasedCritique:
    prompt = _build_critique_prompt(rules, test_features, test_decoding, test_engagement, drop_findings)
    return _generate_structured(SYNTHESIS_MODEL, [prompt], RuleBasedCritique)


def render_evaluation_markdown(
    profile_name: str, critique: RuleBasedCritique, engagement: dict, drop_results: list[dict] | None = None,
) -> str:
    lines = [
        f"# Reel Evaluation vs. Profile \"{profile_name}\"", "",
        f"**{critique.headline}**", "",
        f"**Engagement:** views={engagement.get('views')}, likes={engagement.get('likes')}, "
        f"comments={engagement.get('comments')}", "",
        critique.engagement_assessment, "",
        "## Rule Violations", "",
    ]
    if not critique.violations:
        lines += ["No significant rule violations found.", ""]
    for v in critique.violations:
        lines += [
            f"### [{v.impact.upper()}] {v.rule}", "",
            f"**Expected:** {v.expected}", "",
            f"**Observed:** {v.observed}", "",
            f"**Suggestion:** {v.suggestion}", "",
        ]
    if critique.strengths:
        lines.append("## What's Already Working")
        lines.append("")
        for s in critique.strengths:
            lines.append(f"- {s}")
        lines.append("")
    if drop_results:
        lines.append("## Retention Drop Events")
        lines.append("")
        for r in drop_results:
            f = r["finding"]
            reason = f.get("other_observations") or f.get("reasoning") or f.get("description") or "No specific cause identified."
            lines += [
                f"### t={r['event']['start_t']:.0f}-{r['event']['end_t']:.0f}s -- "
                f"{r['event']['pct_lost']:.1f} percentage points lost", "",
                reason, "",
            ]
    else:
        lines.append("## Retention Drop Events")
        lines.append("")
        lines.append("_No retention recording was provided for this reel -- suggestions above are based on "
                      "comparing this reel's own stats against the ideal-reel rule set, not a second-by-second "
                      "drop-off curve._")
        lines.append("")
    lines.append("## Overall Suggestions")
    lines.append("")
    for s in critique.overall_suggestions:
        lines.append(f"- {s}")
    return "\n".join(lines)
