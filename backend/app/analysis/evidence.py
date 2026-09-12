"""Gather the actual before/during evidence for a DropEvent -- the package
that gets handed to Gemini.

Evidence is real video CLIPS (with audio), not sampled still frames --
Gemini natively understands motion and audio together, and a lot of what
actually matters here (AI-generation artifacts like a face morphing across
frames, flicker, inconsistent hand geometry within one shot, audio that
doesn't sync to the action) is structurally invisible to a handful of
stills no matter how many you sample. One representative thumbnail per
window is still saved alongside the clip purely so a human can eyeball the
evidence folder without playing video files.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

from app.analysis.drops import DropEvent


def _ensure_min_duration(start: float, end: float, min_s: float, floor: float = 0.0, ceil: float | None = None) -> tuple[float, float]:
    """Widen a (possibly zero-width) window to at least min_s seconds,
    centered where possible, clamped to [floor, ceil]."""
    if end - start >= min_s:
        return start, end
    center = (start + end) / 2
    new_start = max(floor, center - min_s / 2)
    new_end = new_start + min_s
    if ceil is not None and new_end > ceil:
        new_end = ceil
        new_start = max(floor, new_end - min_s)
    return new_start, new_end


def extract_clip(video_path: str, start_t: float, end_t: float, out_path: Path) -> Path:
    """Trim a frame-accurate clip (video + audio) via ffmpeg. -ss/-to placed
    after -i trades some speed for accuracy, which is worth it since these
    clips are only a few seconds long anyway."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ss", f"{max(0.0, start_t):.2f}", "-to", f"{end_t:.2f}",
        "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed:\n{result.stderr[-2000:]}")
    return out_path


def extract_thumbnail(video_path: str, t: float, out_path: Path) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    cv2.imwrite(str(out_path), frame)
    return out_path


@dataclass
class DropEvidence:
    event: DropEvent
    before_clip_path: Path
    during_clip_path: Path
    before_thumb_path: Path | None
    during_thumb_path: Path | None
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
    out_dir: Path, before_lookback_s: float = 3.0, min_clip_s: float = 1.2,
) -> DropEvidence:
    video_duration = float(timeline_df["sec"].max()) if "sec" in timeline_df else None

    during_start, during_end = event.investigate_start_t, event.investigate_end_t
    during_start, during_end = _ensure_min_duration(during_start, during_end, min_clip_s, ceil=video_duration)

    before_end = event.investigate_start_t
    before_start = max(0.0, before_end - before_lookback_s)
    before_start, before_end = _ensure_min_duration(before_start, before_end, min_clip_s, ceil=video_duration)

    event_slug = f"t{int(event.start_t)}-{int(event.end_t)}"
    event_dir = out_dir / event_slug

    before_clip = extract_clip(video_path, before_start, before_end, event_dir / "before.mp4")
    during_clip = extract_clip(video_path, during_start, during_end, event_dir / "during.mp4")
    before_thumb = extract_thumbnail(video_path, (before_start + before_end) / 2, event_dir / "before_thumb.jpg")
    during_thumb = extract_thumbnail(video_path, (during_start + during_end) / 2, event_dir / "during_thumb.jpg")

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
        before_clip_path=before_clip,
        during_clip_path=during_clip,
        before_thumb_path=before_thumb,
        during_thumb_path=during_thumb,
        before_window=(before_start, before_end),
        during_window=(during_start, during_end),
        numeric_summary=numeric_summary,
    )
