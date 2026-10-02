"""Decode a handful of "ideal" reels (ones that already performed well)
from an Instagram page into a reusable, saved rule set covering pacing,
hook, storytelling, audio, and technical-execution patterns -- then apply
that rule set against any number of later reels (see app.analysis.evaluate).

A retention recording is NOT required for an ideal reel: it's decoded from
its own video (frames + deterministic stats) and whatever engagement it
received either way. If one IS supplied for a given ideal reel, it's used
purely to enrich that reel's own features (see compute_reel_features's
retention_at_*s stats) with evidence for which techniques correlate with
viewers actually staying -- not to diagnose a problem, which is what
app.analysis.evaluate is for.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from app.analysis.evidence import extract_frames_at, sample_frame_times
from app.analysis.gemini_client import ContentRules, ReelDecoding, decode_reel, synthesize_content_rules
from app.features.timeline import build_timeline, compute_reel_features
from app.ingest import video_fetch
from app.ingest.engagement import fetch_engagement
from app.ingest.retention_extract import extract_curve_from_frames, extract_frames
from app.schemas import EngagementMetrics

ProgressCallback = Callable[[str], None]

PROFILES_DIR = Path(__file__).resolve().parents[3] / "data" / "profiles"

# 2 frames/sec across the whole reel for ideal-reel decoding -- dense enough
# to catch every cut/text-overlay/pacing beat, not just a handful of evenly
# spaced samples. At a reel's native ~200-450KB/frame resolution this would
# blow past Gemini's ~20MB inline-request limit for anything longer than
# ~15s (a 78s reel at 2fps is ~150 frames, ~50MB+ raw) -- DECODE_FRAME_MAX_WIDTH
# downscales each frame before sending to keep the payload in budget; see
# extract_frames_at's docstring for the actual numbers this is based on.
DECODE_FRAMES_PER_SECOND = 2.0
DECODE_FRAME_MAX_WIDTH = 480


@dataclass
class ReelAnalysis:
    url: str
    video_path: Path
    engagement: EngagementMetrics
    features: dict
    decoding: ReelDecoding
    df: pd.DataFrame | None = field(default=None, repr=False)
    shots_df: pd.DataFrame | None = field(default=None, repr=False)

    def to_summary_dict(self) -> dict:
        """The compact, JSON-safe subset that goes into Gemini prompts and
        saved profile/evidence files -- not the raw per-second DataFrames."""
        return {
            "url": self.url,
            "engagement": self.engagement.model_dump(),
            "features": self.features,
            "decoding": self.decoding.model_dump(),
        }


DECODE_CACHE_FILENAME = "decode_cache.json"


def _find_downloaded_video(run_dir: Path) -> Path | None:
    for p in sorted(run_dir.glob("*.mp4")):
        if p.name != "retention_source.mp4":
            return p
    return None


def extract_retention_curve(retention_source: str, run_dir: Path, on_progress: ProgressCallback | None = None) -> Path | None:
    """Shared by both the ideal-reel decode flow (an ideal reel's OWN
    retention recording is optional, used only to enrich its features --
    see compute_reel_features's retention_at_*s stats) and the test-reel
    evaluate flow (where it also drives full drop-event analysis). Returns
    a path to a saved retention.json, or None if extraction produced
    nothing usable -- callers treat that the same as "no retention
    recording was provided" rather than failing the run over it."""
    def progress(msg: str) -> None:
        print(f"[rules] {msg}")
        if on_progress:
            on_progress(msg)

    retention_source_path = Path(retention_source)
    if retention_source_path.is_dir():
        frames = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in retention_source_path.glob(ext))
    else:
        frames = extract_frames(retention_source_path, run_dir / "_retention_frames", fps=8.0)
    if not frames:
        progress("No frames found in the retention recording -- proceeding without a retention curve.")
        return None

    curve = extract_curve_from_frames(frames, crop_box=None)
    if not curve.points:
        progress("Retention extraction produced zero points -- proceeding without a retention curve.")
        return None

    retention_json_path = run_dir / "retention.json"
    retention_json_path.write_text(json.dumps(curve.model_dump(), indent=2), encoding="utf-8")
    progress(f"Extracted {len(curve.points)} retention points, duration={curve.video_duration_s}s")
    return retention_json_path


def fetch_and_analyze_reel(
    reel_url: str, run_dir: Path, cookies_file: str | None = None,
    whisper_model: str = "small", language: str | None = None,
    retention_json_path: Path | None = None, on_progress: ProgressCallback | None = None,
) -> ReelAnalysis:
    """Shared first half of both the ideal-reel decode flow and the
    test-reel evaluate flow: fetch the video (+ its yt-dlp metadata),
    resolve engagement, build the per-second timeline (with retention
    merged in if retention_json_path is given), compute deterministic
    reel-level features, and have Gemini decode the reel's creative/
    technical approach from representative frames sampled across it.

    Caches engagement/features/decoding to run_dir/decode_cache.json and
    reuses it on a later call with the SAME run_dir (only when no
    retention_json_path is given -- that path always needs a fresh,
    real df/shots_df, never a short-circuited one). This exists because a
    Gemini decode call is the slow, unreliable part of this whole pipeline
    (free-tier can take anywhere from ~1 to 20+ minutes per reel under
    load) -- losing an already-successful decode to a LATER reel's
    unrelated crash was confirmed wasteful in practice, not hypothetical."""
    def progress(msg: str) -> None:
        print(f"[rules] {msg}")
        if on_progress:
            on_progress(msg)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_path = run_dir / DECODE_CACHE_FILENAME
    if retention_json_path is None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        video_path = _find_downloaded_video(run_dir)
        if video_path is not None:
            progress(f"Reusing cached decode for {reel_url} from a previous run "
                     f"(delete {cache_path} to force a redo).")
            return ReelAnalysis(
                url=reel_url, video_path=video_path,
                engagement=EngagementMetrics.model_validate(cached["engagement"]),
                features=cached["features"], decoding=ReelDecoding.model_validate(cached["decoding"]),
            )

    progress(f"Fetching {reel_url} ...")
    video_path, info = video_fetch.fetch_reel_with_info(
        reel_url, run_dir, cookies_file=Path(cookies_file) if cookies_file else None,
    )
    progress(f"Fetched {video_path.name}")

    engagement = fetch_engagement(reel_url, ytdlp_info=info)
    progress(f"Engagement ({engagement.source}): views={engagement.views} likes={engagement.likes} "
             f"comments={engagement.comments}")

    progress("Building feature timeline (quality, shots, audio, transcript)... this is the slow step.")
    df, shots_df = build_timeline(
        str(video_path), str(retention_json_path) if retention_json_path else None,
        whisper_model=whisper_model, language=language,
    )
    duration = float(df["sec"].max()) if not df.empty else 0.0
    features = compute_reel_features(df, shots_df, duration)

    n_frames = max(1, round(duration * DECODE_FRAMES_PER_SECOND))
    frame_times = sample_frame_times(0.0, duration, n_frames)
    frame_paths = extract_frames_at(
        str(video_path), frame_times, run_dir / "decode_frames", "frame", max_width=DECODE_FRAME_MAX_WIDTH,
    )
    progress(f"Sampled {len(frame_paths)} frames at ~{DECODE_FRAMES_PER_SECOND}/s for decoding.")

    progress("Decoding creative/technical approach with Gemini...")
    decoding = decode_reel(features, engagement.model_dump(), frame_paths)

    if retention_json_path is None:
        cache_path.write_text(json.dumps({
            "engagement": engagement.model_dump(), "features": features, "decoding": decoding.model_dump(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    return ReelAnalysis(
        url=reel_url, video_path=video_path, engagement=engagement,
        features=features, decoding=decoding, df=df, shots_df=shots_df,
    )


def compute_engagement_benchmark(analyses: list[ReelAnalysis]) -> dict:
    """Deterministic avg/median/min/max for views/likes/comments/engagement
    rate across the ideal reels -- computed here in code, not narrated by
    Gemini. Exists because with only a handful of ideal reels, letting the
    model freely describe "typical" engagement in prose led it to invent a
    categorical explanation (e.g. calling one reel's lower view count a
    "niche case" vs another "iconic") for what was really just ordinary
    small-sample spread -- see synthesize_content_rules's prompt, which now
    treats this dict as the one source of truth for those numbers instead."""
    def stats(values: list[float | int | None], round_to: int = 0) -> dict | None:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        sv = sorted(clean)
        n = len(sv)
        median = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        return {
            "avg": round(sum(clean) / n, round_to),
            "median": round(median, round_to),
            "min": round(min(clean), round_to),
            "max": round(max(clean), round_to),
        }

    return {
        "n_reels": len(analyses),
        "views": stats([a.engagement.views for a in analyses]),
        "likes": stats([a.engagement.likes for a in analyses]),
        "comments": stats([a.engagement.comments for a in analyses]),
        "engagement_rate_pct": stats([a.engagement.engagement_rate_pct for a in analyses], round_to=2),
    }


def extract_rules_from_ideal_reels(
    reels: list[dict], profile_name: str, run_dir: Path, cookies_file: str | None = None,
    whisper_model: str = "small", language: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> ContentRules:
    """Decode each ideal reel independently, then synthesize one rule set
    from the patterns shared across all of them. Saves the result under
    data/profiles/<profile_name>/ so it can be reused against any number
    of later test reels without re-decoding these ideal ones each time.

    reels: [{"url": str, "retention_source": str | None}, ...]. A given
    ideal reel's retention recording is entirely optional -- when present
    it's merged into that reel's own timeline (see build_timeline) and
    surfaces as extra retention_at_*s stats in compute_reel_features,
    giving the rule-synthesis step actual evidence for which techniques
    correlate with sustained attention, not just aggregate engagement
    counts. Without it, decoding proceeds exactly as before."""
    def progress(msg: str) -> None:
        print(f"[rules] {msg}")
        if on_progress:
            on_progress(msg)

    run_dir = Path(run_dir)
    analyses: list[ReelAnalysis] = []
    for i, reel in enumerate(reels):
        url = reel["url"]
        ideal_dir = run_dir / f"ideal_{i}"
        progress(f"[{i + 1}/{len(reels)}] decoding ideal reel: {url}")

        retention_json_path = None
        retention_source = reel.get("retention_source")
        if retention_source:
            progress(f"[{i + 1}/{len(reels)}] retention recording provided -- extracting curve...")
            ideal_dir.mkdir(parents=True, exist_ok=True)
            retention_json_path = extract_retention_curve(retention_source, ideal_dir, on_progress)

        analysis = fetch_and_analyze_reel(
            url, ideal_dir, cookies_file, whisper_model, language,
            retention_json_path=retention_json_path, on_progress=on_progress,
        )
        analyses.append(analysis)

    benchmark_stats = compute_engagement_benchmark(analyses)
    progress(f"Synthesizing rules across all ideal reels (benchmark: {benchmark_stats})...")
    rules = synthesize_content_rules([a.to_summary_dict() for a in analyses], benchmark_stats)

    save_profile(profile_name, rules, analyses)
    progress(f"Saved profile '{profile_name}' ({len(rules.rules)} rules).")
    return rules


def save_profile(profile_name: str, rules: ContentRules, analyses: list[ReelAnalysis]) -> Path:
    profile_dir = PROFILES_DIR / profile_name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "rules.json").write_text(json.dumps(rules.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    (profile_dir / "source_reels.json").write_text(
        json.dumps([a.to_summary_dict() for a in analyses], indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return profile_dir


def load_profile(profile_name: str) -> ContentRules:
    path = PROFILES_DIR / profile_name / "rules.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved profile named '{profile_name}' (expected {path}). "
            "Create one first via extract_rules_from_ideal_reels / POST /api/projects."
        )
    return ContentRules.model_validate(json.loads(path.read_text(encoding="utf-8")))


def list_profiles() -> list[dict]:
    if not PROFILES_DIR.exists():
        return []
    out = []
    for d in sorted(PROFILES_DIR.iterdir()):
        rules_path = d / "rules.json"
        if not d.is_dir() or not rules_path.exists():
            continue
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        out.append({
            "name": d.name,
            "profile_summary": rules.get("profile_summary"),
            "n_rules": len(rules.get("rules", [])),
            "benchmark_duration_s": rules.get("benchmark_duration_s"),
            "benchmark_engagement": rules.get("benchmark_engagement"),
            "benchmark_stats": rules.get("benchmark_stats") or {},
        })
    return out
