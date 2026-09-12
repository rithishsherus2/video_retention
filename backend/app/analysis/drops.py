"""Retention event detection: drops AND rises, plus a classifier that pairs
a drop with a nearby recovery into a "likely skip" pattern.

Both directions use the same robust (median/MAD) threshold on the
per-second rate of change -- not a fixed percentage -- because different
videos have wildly different baseline decay shapes, so only deviations
from a video's OWN typical pace are a meaningful signal.

Rises matter for platforms/players where retention isn't strictly
monotonic: if a per-second "retention" curve is built from playhead
position samples (the standard approach -- see docs on how this is
computed), a viewer who skips forward stops counting for the seconds they
jumped over and resumes counting at the landing point. Enough viewers
doing the same skip shows up as a dip at the skipped range followed by a
partial recovery at the landing point -- not genuine re-engagement, just
skippers rejoining the counted audience. This reel's own retention curve
is monotonically decreasing throughout (nothing here exercises the rise
path), so this is built and validated against synthetic data instead,
ready for the first source where it actually fires.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A viewer swipes away some time AFTER whatever triggered it, not the
# instant it happens -- so the window worth investigating for a drop
# starting at t reaches back before t, not just from t onward.
LAG_BEFORE_S = 2.0
LAG_AFTER_S = 0.5


@dataclass
class DropEvent:
    start_t: float          # last clean timestamp BEFORE the drop began
    end_t: float             # last timestamp of the flagged fast-drop run
    start_pct: float
    end_pct: float
    peak_z: float            # how extreme the steepest step in this event was

    @property
    def pct_lost(self) -> float:
        return self.start_pct - self.end_pct

    @property
    def investigate_start_t(self) -> float:
        """Where to start pulling 'before' evidence from."""
        return max(0.0, self.start_t - LAG_BEFORE_S)

    @property
    def investigate_end_t(self) -> float:
        """Where to stop pulling 'during' evidence from."""
        return self.end_t + LAG_AFTER_S


@dataclass
class RiseEvent:
    start_t: float           # last timestamp BEFORE the rise began
    end_t: float              # last timestamp of the flagged rise run
    start_pct: float
    end_pct: float
    peak_z: float

    @property
    def pct_gained(self) -> float:
        return self.end_pct - self.start_pct


@dataclass
class SkipCandidate:
    """A drop closely followed by a recovery that gets back most of what
    was lost -- the signature of viewers skipping ahead rather than
    genuinely disengaging. Not a certainty (a real re-engagement moment can
    look the same from the numbers alone), just a pattern worth flagging
    as a different hypothesis from a plain drop."""
    drop: DropEvent
    rise: RiseEvent
    recovered_fraction: float  # rise.pct_gained / drop.pct_lost, clipped to [0, 1]


def _compute_rates_and_z(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[["sec", "retention_pct"]].dropna().sort_values("sec").reset_index(drop=True)
    if len(sub) < 3:
        return sub

    sub["dt"] = sub["sec"].diff()
    sub["drop_rate"] = -sub["retention_pct"].diff() / sub["dt"]  # +ve = pct points LOST per second

    rates = sub["drop_rate"].dropna()
    median = rates.median()
    mad = (rates - median).abs().median()
    scale = 1.4826 * mad if mad > 1e-9 else (rates.std() or 1.0)
    sub["z"] = (sub["drop_rate"] - median) / (scale + 1e-9)
    return sub


def _group_flagged(sub: pd.DataFrame, flagged_idx: list[int], merge_gap_s: float) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for idx in flagged_idx:
        if current and (sub.loc[idx, "sec"] - sub.loc[current[-1], "sec"]) > merge_gap_s:
            groups.append(current)
            current = []
        current.append(idx)
    if current:
        groups.append(current)
    return groups


def detect_drop_events(df: pd.DataFrame, z_threshold: float = 1.5, merge_gap_s: float = 1.5) -> list[DropEvent]:
    """df needs 'sec' and 'retention_pct' (NaN rows -- e.g. cleaned-out OCR
    outliers -- are skipped, not treated as drops). Returns events sorted
    by percentage points lost, biggest first."""
    sub = _compute_rates_and_z(df)
    if "z" not in sub:
        return []

    flagged_idx = sub.index[sub["z"] > z_threshold].tolist()
    if not flagged_idx:
        return []

    groups = _group_flagged(sub, flagged_idx, merge_gap_s)
    events = [_finalize_drop(sub, g) for g in groups]
    return sorted(events, key=lambda e: e.pct_lost, reverse=True)


def detect_rise_events(df: pd.DataFrame, z_threshold: float = 1.5, merge_gap_s: float = 1.5) -> list[RiseEvent]:
    """Mirror of detect_drop_events on the opposite tail: seconds where
    retention increased faster than this video's own typical (usually
    near-zero-or-negative) pace. On a monotonically-decreasing curve this
    naturally returns nothing -- that's correct, not a bug."""
    sub = _compute_rates_and_z(df)
    if "z" not in sub:
        return []

    flagged_idx = sub.index[sub["z"] < -z_threshold].tolist()
    if not flagged_idx:
        return []

    groups = _group_flagged(sub, flagged_idx, merge_gap_s)
    events = [_finalize_rise(sub, g) for g in groups]
    return sorted(events, key=lambda e: e.pct_gained, reverse=True)


def find_skip_patterns(
    drop_events: list[DropEvent], rise_events: list[RiseEvent],
    max_gap_s: float = 20.0, min_recovery_frac: float = 0.5,
) -> list[SkipCandidate]:
    """Pair each drop with the nearest subsequent rise (within max_gap_s)
    that recovers at least min_recovery_frac of what was lost. A drop with
    no qualifying rise is left as a plain DropEvent -- this only reclassifies
    the ones that show the skip-and-recover signature."""
    candidates: list[SkipCandidate] = []
    used_rises: set[int] = set()
    for drop in drop_events:
        best_rise, best_idx = None, None
        for i, rise in enumerate(rise_events):
            if i in used_rises:
                continue
            gap = rise.start_t - drop.end_t
            if 0 <= gap <= max_gap_s:
                if best_rise is None or rise.start_t < best_rise.start_t:
                    best_rise, best_idx = rise, i
        if best_rise is None or drop.pct_lost <= 0:
            continue
        recovered = max(0.0, min(1.0, best_rise.pct_gained / drop.pct_lost))
        if recovered >= min_recovery_frac:
            used_rises.add(best_idx)
            candidates.append(SkipCandidate(drop=drop, rise=best_rise, recovered_fraction=recovered))
    return candidates


def _finalize_drop(sub: pd.DataFrame, group_idx: list[int]) -> DropEvent:
    first_idx, last_idx = group_idx[0], group_idx[-1]
    before_idx = first_idx - 1 if first_idx > 0 else first_idx
    before_row = sub.loc[before_idx]
    after_row = sub.loc[last_idx]
    peak_z = sub.loc[group_idx, "z"].max()
    return DropEvent(
        start_t=float(before_row["sec"]), end_t=float(after_row["sec"]),
        start_pct=float(before_row["retention_pct"]), end_pct=float(after_row["retention_pct"]),
        peak_z=float(peak_z),
    )


def _finalize_rise(sub: pd.DataFrame, group_idx: list[int]) -> RiseEvent:
    first_idx, last_idx = group_idx[0], group_idx[-1]
    before_idx = first_idx - 1 if first_idx > 0 else first_idx
    before_row = sub.loc[before_idx]
    after_row = sub.loc[last_idx]
    peak_z = sub.loc[group_idx, "z"].min()  # most negative = strongest rise
    return RiseEvent(
        start_t=float(before_row["sec"]), end_t=float(after_row["sec"]),
        start_pct=float(before_row["retention_pct"]), end_pct=float(after_row["retention_pct"]),
        peak_z=float(peak_z),
    )
