"""Canonical data shapes shared across every source adapter and pipeline stage.

Every ingestion path (manual drag-recording OCR today, a scraped API later,
a client's own instrumented player eventually) must produce these same
shapes. Nothing downstream should know or care where the data came from.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetentionPoint(BaseModel):
    """One sample of the retention curve: what fraction of viewers were
    still watching at second `t`."""

    t: float = Field(..., description="Seconds from the start of the reel")
    pct: float = Field(..., description="Percent of viewers still watching (can exceed 100 due to rewatch loops)")
    source: str = Field(default="ocr", description="How this point was read: ocr | ocr_llm_fallback | manual")
    confidence: float = Field(default=1.0, description="0-1, extraction confidence for this point")


class RetentionCurve(BaseModel):
    points: list[RetentionPoint]
    video_duration_s: Optional[float] = None

    def sorted_points(self) -> list[RetentionPoint]:
        return sorted(self.points, key=lambda p: p.t)


class ExtractionSourceType(str, Enum):
    DRAG_RECORDING = "drag_recording"   # screen-recorded finger drag over the graph
    TAP_SCREENSHOTS = "tap_screenshots"  # discrete screenshots, one per tapped point
