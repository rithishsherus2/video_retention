"""Per-frame deterministic image-quality signals.

Nothing here is LLM-judged -- these are all classic CV metrics computed
directly off pixels, cheap enough to run on every sampled frame of a 30s
reel in a couple seconds on CPU. The analysis stage later flags a drop as a
z-score against *this video's own baseline*, not an absolute threshold --
different sources/exports have wildly different absolute sharpness, so only
relative drops within one video are meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- standard focus/blur measure. Lower =
    blurrier (soft focus, motion blur, or an upscaled/re-compressed source
    pretending to be higher-res than it is)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def highfreq_energy_ratio(gray: np.ndarray) -> float:
    """Fraction of the frame's frequency-domain energy that's high-frequency
    (fine detail/texture) vs. low-frequency (flat regions, gradients). Drops
    when footage is effectively lower-resolution than its container (e.g.
    480p source upscaled to 1080p) even though nominal pixel dimensions
    don't change."""
    f = np.fft.fft2(gray.astype(np.float32))
    mag = np.abs(np.fft.fftshift(f))
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    low_mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    total = mag.sum() + 1e-9
    low = mag[low_mask].sum()
    return float((total - low) / total)


def blockiness(gray: np.ndarray) -> float:
    """Proxy for compression artifacts: how much stronger the pixel
    discontinuity is right at 8x8 JPEG/H.264-macroblock-aligned boundaries
    vs. elsewhere. Ratio near 1.0 = no visible blocking; notably above 1.0
    = visible block edges (a re-encode/quality drop)."""
    g = gray.astype(np.float32)
    diffs = np.abs(np.diff(g, axis=1))  # (h, w-1)
    w = diffs.shape[1]
    if w < 8:
        return 1.0
    boundary_mask = np.zeros(w, dtype=bool)
    boundary_mask[7::8] = True
    boundary_mean = diffs[:, boundary_mask].mean() if boundary_mask.any() else 0.0
    nonboundary_mean = diffs[:, ~boundary_mask].mean() if (~boundary_mask).any() else 1e-9
    return float(boundary_mean / (nonboundary_mean + 1e-9))


@dataclass
class QualityFrame:
    t: float
    sharpness: float
    brightness: float
    highfreq_ratio: float
    blockiness: float


def extract_quality_timeline(video_path: str, fps_sample: float = 5.0) -> pd.DataFrame:
    """Sample frames at fps_sample and compute all quality metrics for each.
    Returns a DataFrame with one row per sampled frame (finer than 1/sec --
    timeline.py aggregates to whole seconds to align with retention data,
    but keeping frame-level granularity here lets us pinpoint exactly which
    frame within a second something happened)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native_fps / fps_sample))

    rows: list[QualityFrame] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t = frame_idx / native_fps
            rows.append(QualityFrame(
                t=t,
                sharpness=sharpness(gray),
                brightness=brightness(gray),
                highfreq_ratio=highfreq_energy_ratio(gray),
                blockiness=blockiness(gray),
            ))
        frame_idx += 1
    cap.release()

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def assign_shot_index(t: pd.Series, shots: list) -> pd.Series:
    """Map each frame's timestamp to the shot it falls in (shots must have
    .start_t/.end_t, e.g. app.features.shots.Shot). Frames after the last
    shot's end (rounding at the boundary) get clamped to the last shot."""
    starts = [s.start_t for s in shots]
    idx = np.searchsorted(starts, t, side="right") - 1
    idx = idx.clip(0, len(shots) - 1)
    return pd.Series(idx, index=t.index)


def zscore_flags_per_shot(df: pd.DataFrame, columns: list[str], threshold: float = -1.5) -> pd.DataFrame:
    """Same idea as zscore_flags, but the baseline is each SHOT's own
    mean/std, not the whole video's. This is the metric that should
    actually drive a 'quality dropped' flag: a frame that's blurry
    relative to its own shot's neighbors is a strong real-defect signal
    (motion blur, autofocus hunt, encoding glitch mid-shot). A whole shot
    that merely differs from the video's global average is NOT necessarily
    a defect -- different shots can legitimately have different
    location/lighting/style, and a global baseline can't tell the
    difference. Requires a 'shot_index' column (see assign_shot_index)."""
    out = df.copy()
    for col in columns:
        grp = out.groupby("shot_index")[col]
        mean = grp.transform("mean")
        std = grp.transform("std").fillna(0.0)
        z = np.where(std > 1e-9, (out[col] - mean) / std, 0.0)
        out[f"{col}_shot_z"] = z
        if col == "blockiness":
            out[f"{col}_shot_flag"] = out[f"{col}_shot_z"] > -threshold
        else:
            out[f"{col}_shot_flag"] = out[f"{col}_shot_z"] < threshold
    return out


def shot_level_quality_deltas(shot_agg: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """For each shot, % change in mean quality metrics vs. the immediately
    preceding shot. This is raw comparative evidence, NOT a verdict --
    whether a drop here is a real defect or a deliberate change (new
    location, time of day, intentional stylistic softness) is a judgment
    call that needs the actual frames from both shots, so it's left to the
    analysis stage. shot_agg must have a 'shot_index' column and be one
    row per shot, sorted or sortable by it."""
    out = shot_agg.sort_values("shot_index").reset_index(drop=True).copy()
    for col in columns:
        prev = out[col].shift(1)
        out[f"{col}_pct_change_vs_prev_shot"] = (out[col] - prev) / prev.replace(0, np.nan) * 100
    return out


def zscore_flags(df: pd.DataFrame, columns: list[str], threshold: float = -1.5) -> pd.DataFrame:
    """Add <col>_z columns (z-score against this video's own mean/std) and
    a boolean <col>_flag for values more than `threshold` std below the
    mean (a quality DROP, hence negative threshold -- blockiness is
    inverted since higher = worse there). Kept for reference/comparison,
    but zscore_flags_per_shot is what timeline.py actually uses to decide
    what counts as a real quality drop -- see its docstring for why."""
    out = df.copy()
    for col in columns:
        mean, std = out[col].mean(), out[col].std()
        if std < 1e-9:
            out[f"{col}_z"] = 0.0
        else:
            out[f"{col}_z"] = (out[col] - mean) / std
        if col == "blockiness":
            out[f"{col}_flag"] = out[f"{col}_z"] > -threshold  # higher blockiness = worse
        else:
            out[f"{col}_flag"] = out[f"{col}_z"] < threshold
    return out
