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


class EngagementMetrics(BaseModel):
    """Views/likes/comments for one reel, however they were obtained.
    `source` records provenance (yt_dlp, a named external provider, or a
    merge of both) since these numbers are frequently incomplete for
    accounts you don't own -- worth knowing when a field is None because
    Instagram withheld it vs. because nothing was configured to fetch it."""

    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    caption: Optional[str] = None
    hashtags: list[str] = Field(default_factory=list)
    posted_at: Optional[str] = None
    source: str = "unknown"

    @property
    def engagement_rate_pct(self) -> Optional[float]:
        """(likes + comments) / views, as a percentage -- None if views is
        missing or zero, since the ratio is meaningless without it."""
        if not self.views or self.likes is None or self.comments is None:
            return None
        return round((self.likes + self.comments) / self.views * 100, 3)
