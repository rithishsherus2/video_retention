"""End-to-end agent: given a reel URL and a retention-graph recording,
runs the entire pipeline (fetch video -> extract retention -> build
feature timeline -> detect drops -> Gemini analysis -> synthesized
report) as one call, making its own sequencing decisions along the way.

This replaces the four separate hand-run CLI commands (video_fetch,
retention_extract, features.timeline, analysis.run) used throughout
development with one entrypoint -- both the web backend and any future
CLI use this, not the individual scripts directly.

This is "the agent" in the sense of removing manual pipeline
orchestration. The OTHER sense from the original brief -- something that
watches your account and fires automatically on a new reel, no manual
link/recording needed at all -- is still open and out of scope here: it
needs Instagram-side automation (polling your own account, detecting new
posts) that this demo doesn't build. What's here is the piece that
automation would call once it has a URL and a retention recording, not a
replacement for it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.analysis.run import analyze_drops_and_report
from app.features.timeline import build_timeline, save_debug_plot, write_human_summary
from app.ingest import video_fetch
from app.ingest.retention_extract import extract_curve_from_frames, extract_frames

ProgressCallback = Callable[[str], None]


@dataclass
class PipelineResult:
    run_dir: str
    video_path: str
    retention_json_path: str
    timeline_csv_path: str
    timeline_summary_path: str
    timeline_plot_path: str
    drop_analysis_path: str
    report_path: str | None
    report_markdown: str | None
    n_retention_points: int
    video_duration_s: float | None


def run_pipeline(
    reel_url: str,
    retention_source: str,  # path to a screen-recording video, OR a directory of tap screenshots
    run_dir: Path,
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,  # NOT "chrome" by default -- App-Bound Encryption blocks
    # reading Chrome's cookie store from an external tool (confirmed during development); the manual
    # cookies.txt export (cookies_file) is the path that actually works. Pass "firefox" here explicitly
    # if that's genuinely a viable option for a given user -- never default it to "chrome".
    whisper_model: str = "small",
    language: str | None = None,
    top_n_events: int = 3,
    on_progress: ProgressCallback | None = None,
) -> PipelineResult:
    def progress(msg: str) -> None:
        print(f"[agent] {msg}")
        if on_progress:
            on_progress(msg)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. fetch the reel ----
    progress("Fetching reel video (yt-dlp + cookies)...")
    video_path = video_fetch.fetch_reel(
        reel_url, run_dir,
        cookies_from_browser=cookies_from_browser,
        cookies_file=Path(cookies_file) if cookies_file else None,
    )
    progress(f"Video fetched: {video_path.name}")

    # ---- 2. extract the retention curve ----
    progress("Extracting retention curve from the provided recording...")
    retention_source_path = Path(retention_source)
    if retention_source_path.is_dir():
        frames = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg") for p in retention_source_path.glob(ext)
        )
    else:
        frames = extract_frames(retention_source_path, run_dir / "_retention_frames", fps=8.0)
    if not frames:
        raise RuntimeError(f"No frames found in retention source: {retention_source}")

    curve = extract_curve_from_frames(frames, crop_box=None)  # None -> auto-detect per source
    if not curve.points:
        raise RuntimeError(
            "Retention extraction produced zero points -- check the recording actually shows the "
            "tap/drag tooltip clearly (see app.ingest.retention_extract --calibrate to debug)."
        )
    retention_json_path = run_dir / "retention.json"
    retention_json_path.write_text(json.dumps(curve.model_dump(), indent=2), encoding="utf-8")
    progress(f"Extracted {len(curve.points)} retention points, duration={curve.video_duration_s}s")

    # ---- 3. build the feature timeline ----
    progress("Building feature timeline (quality, shots, audio, transcript)... this is the slow step.")
    df, shots_df = build_timeline(
        str(video_path), str(retention_json_path), whisper_model=whisper_model, language=language,
    )
    timeline_csv_path = run_dir / "timeline.csv"
    df.to_csv(timeline_csv_path, index=False)
    shots_df.to_csv(run_dir / "shots.csv", index=False)
    timeline_plot_path = run_dir / "timeline_debug.png"
    save_debug_plot(df, str(timeline_plot_path))
    timeline_summary_path = run_dir / "timeline_summary.txt"
    write_human_summary(df, shots_df, str(timeline_summary_path))
    progress("Feature timeline built.")

    # ---- 4. drop detection + Gemini analysis + synthesis ----
    progress("Running drop-event analysis (Gemini)...")
    analysis = analyze_drops_and_report(
        str(video_path), str(timeline_csv_path), run_dir, top_n=top_n_events,
        summary_path=timeline_summary_path, on_progress=progress,
    )

    report_path = run_dir / "report.md"
    progress("Pipeline complete.")
    return PipelineResult(
        run_dir=str(run_dir),
        video_path=str(video_path),
        retention_json_path=str(retention_json_path),
        timeline_csv_path=str(timeline_csv_path),
        timeline_summary_path=str(timeline_summary_path),
        timeline_plot_path=str(timeline_plot_path),
        drop_analysis_path=str(run_dir / "drop_analysis.json"),
        report_path=str(report_path) if report_path.exists() else None,
        report_markdown=analysis["report_markdown"],
        n_retention_points=len(curve.points),
        video_duration_s=curve.video_duration_s,
    )
