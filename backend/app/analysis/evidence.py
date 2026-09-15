"""Gather the actual before/during frames + numeric context for a
DropEvent -- the package that gets handed to Gemini.

Frame-sampled stills, not full video clips. This was previously upgraded
to real video+audio clips (see git history / DESIGN.md) so Gemini could
perceive motion and audio-sync directly -- genuinely better evidence for
catching temporal artifacts (a face morphing across frames, audio that
doesn't sync to the action). Reverted back to stills because the free
Gemini tier's rate limit is request-count-sensitive, and video clips need
~5 requests per event (2 uploads + polling + generate) vs. 1 inline call
for a set of images -- the video-clip version couldn't complete even a
25-second reel's analysis before hitting the limit. Stills are the
correct tradeoff for demoing under a free-tier budget; switch back once
billing/quota isn't the constraint.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from app.analysis.drops import DropEvent


def sample_frame_times(start_t: float, end_t: float, n: int) -> list[float]:
    if end_t <= start_t or n <= 1:
        return [start_t]
    return [start_t + i * (end_t - start_t) / (n - 1) for i in range(n)]


def extract_frames_at(video_path: str, times: list[float], out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"{prefix}_{i:02d}_t{t:.1f}s.jpg"
        cv2.imwrite(str(p), frame)
        paths.append(p)
    cap.release()
    return paths


@dataclass
class DropEvidence:
    event: DropEvent
    before_frame_paths: list[Path]
    during_frame_paths: list[Path]
    before_window: tuple[float, float]
    during_window: tuple[float, float]
    numeric_summary: dict


def _window_summary(df: pd.DataFrame, t0: float, t1: float) -> dict:
    w = df[(df["sec"] >= int(t0)) & (df["sec"] <= int(t1) + 1)]
    if w.empty:
        return {}
    transcript = " ".join(t for t in w.get("transcript", []) if isinstance(t, str) and t).strip()
    return {
        "sharpness_mean": round(float(w["sharpness_mean"].mean()), 1) if w["sharpness_mean"].notna().any() else None,
        "blockiness_mean": round(float(w["blockiness_mean"].mean()), 3) if w["blockiness_mean"].notna().any() else None,
        "any_cut": bool(w["is_cut_second"].any()),
        "any_redundant_cut": bool(w["is_redundant_cut_second"].any()),
        "any_reused_shot": bool(w["is_reused_shot_second"].any()),
        "audio_rms_db_mean": round(float(w["rms_db_mean"].mean()), 1) if w["rms_db_mean"].notna().any() else None,
        "transcript": transcript or None,
    }


def gather_evidence(
    video_path: str, event: DropEvent, timeline_df: pd.DataFrame,
    out_dir: Path, n_before: int = 3, n_during: int = 5, before_lookback_s: float = 3.0,
) -> DropEvidence:
    during_start, during_end = event.investigate_start_t, event.investigate_end_t
    before_end = during_start
    before_start = max(0.0, before_end - before_lookback_s)
    # A drop right at the start of the video has no earlier material to
    # compare against -- 'before' degenerates to the opening frame(s),
    # which is actually the right comparison there (the hook itself vs.
    # what happens as it collapses), not a bug to special-case away.
    if before_start == before_end and before_end > 0:
        before_start = max(0.0, before_end - 1.0)

    before_times = sample_frame_times(before_start, before_end, n_before)
    # Guard against the 'before' window's last sample and 'during's first
    # landing on the exact same timestamp (happens when a drop starts at
    # or near t=0) -- that would hand Gemini the identical frame in both
    # sets, defeating the before/during comparison.
    if before_times and abs(before_times[-1] - during_start) < 1e-6:
        during_start_adj = min(during_start + 0.3, during_end)
    else:
        during_start_adj = during_start
    during_times = sample_frame_times(during_start_adj, during_end, n_during)

    event_slug = f"t{int(event.start_t)}-{int(event.end_t)}"
    event_dir = out_dir / event_slug
    before_paths = extract_frames_at(video_path, before_times, event_dir, "before")
    during_paths = extract_frames_at(video_path, during_times, event_dir, "during")

    numeric_summary = {
        "retention_before_pct": event.start_pct,
        "retention_after_pct": event.end_pct,
        "pct_points_lost": round(event.pct_lost, 1),
        "peak_drop_z": round(event.peak_z, 2),
        "before_window_s": [round(before_start, 1), round(before_end, 1)],
        "during_window_s": [round(during_start, 1), round(during_end, 1)],
        "before_window_metrics": _window_summary(timeline_df, before_start, before_end),
        "during_window_metrics": _window_summary(timeline_df, during_start, during_end),
    }

    return DropEvidence(
        event=event,
        before_frame_paths=before_paths,
        during_frame_paths=during_paths,
        before_window=(before_start, before_end),
        during_window=(during_start, during_end),
        numeric_summary=numeric_summary,
    )
