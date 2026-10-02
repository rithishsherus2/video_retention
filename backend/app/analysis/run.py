"""Phase 3 end-to-end: detect drop events from a built timeline, gather
before/during evidence for each, run the Gemini quality-consistency check,
save a report. This is the one generic rule to validate the whole flow --
more rules (redundant dialogue, audio smoothness, etc., most of which
already have deterministic detectors sitting in app/features/) get added
as additional checks per event once this loop is proven out, not by
rebuilding this orchestration.

The core logic lives in analyze_drops_and_report() so it's callable
directly (the agent orchestrator and the web backend both use it) --
main() is a thin CLI wrapper around the same function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from app.analysis.drops import detect_drop_events, detect_rise_events, find_skip_patterns
from app.analysis.evidence import gather_evidence
from app.analysis.gemini_client import analyze_drop_event, render_report_markdown, synthesize_report

ProgressCallback = Callable[[str], None]


def run_drop_event_analysis(
    video_path: str, df: pd.DataFrame, out_dir: Path, top_n: int = 3,
    on_progress: ProgressCallback | None = None,
) -> list[dict]:
    """The detect-events -> gather-evidence -> Gemini-per-event loop, operating
    on an in-memory timeline DataFrame (no CSV round-trip required) -- pulled
    out of analyze_drops_and_report so app.analysis.evaluate can reuse it on
    a timeline it already has in memory, without re-reading anything from
    disk. Does NOT write drop_analysis.json itself; the caller decides that."""
    def progress(msg: str) -> None:
        print(msg)
        if on_progress:
            on_progress(msg)

    out_dir = Path(out_dir)
    all_drops = detect_drop_events(df)
    all_rises = detect_rise_events(df)
    skip_patterns = find_skip_patterns(all_drops, all_rises)
    skip_by_drop_id = {id(sk.drop): sk for sk in skip_patterns}

    if all_rises:
        progress(f"{len(all_rises)} rise event(s) detected -- retention recovered somewhere in this video.")
    if skip_patterns:
        progress(f"{len(skip_patterns)} drop(s) look like a SKIP rather than disengagement.")

    events = all_drops[:top_n]
    progress(f"{len(events)} drop event(s) to investigate (of {len(all_drops)} detected total).")

    results = []
    for i, event in enumerate(events):
        skip = skip_by_drop_id.get(id(event))
        progress(f"[{i + 1}/{len(events)}] investigating t={event.start_t:.0f}-{event.end_t:.0f}s "
                 f"({event.pct_lost:.1f} pts lost)" + ("  [likely a skip]" if skip else ""))
        evidence = gather_evidence(video_path, event, df, out_dir / "evidence")
        finding = analyze_drop_event(evidence)
        results.append({
            "event": {
                "start_t": event.start_t, "end_t": event.end_t,
                "start_pct": event.start_pct, "end_pct": event.end_pct,
                "pct_lost": round(event.pct_lost, 1), "peak_z": round(event.peak_z, 2),
            },
            "likely_skip": {
                "recovered_fraction": round(skip.recovered_fraction, 2),
                "rise_start_t": skip.rise.start_t, "rise_end_t": skip.rise.end_t,
                "note": "Retention recovers shortly after this drop -- consistent with viewers skipping "
                        "ahead rather than genuinely disengaging. Treat as a pacing/content signal for "
                        "THIS segment specifically, not the same severity as a drop with no recovery.",
            } if skip else None,
            "evidence_numeric": evidence.numeric_summary,
            "finding": finding.model_dump(),
        })
        progress(f"  -> quality_issue_found={finding.quality_issue_found}  severity={finding.severity}")

    return results


def analyze_drops_and_report(
    video_path: str, timeline_csv_path: str, out_dir: Path, top_n: int = 3,
    summary_path: Path | None = None, on_progress: ProgressCallback | None = None,
) -> dict:
    """Runs the full drop-analysis + synthesis pass and writes drop_analysis.json
    and report.md into out_dir. Returns {"results": [...], "report_markdown": str | None}.
    on_progress, if given, is called with short human-readable status strings --
    the web backend uses this to stream progress to the browser instead of the console."""
    def progress(msg: str) -> None:
        print(msg)
        if on_progress:
            on_progress(msg)

    out_dir = Path(out_dir)
    df = pd.read_csv(timeline_csv_path)
    results = run_drop_event_analysis(video_path, df, out_dir, top_n, on_progress)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drop_analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = summary_path or (out_dir / "timeline_summary.txt")
    report_markdown = None
    if summary_path.exists():
        progress("Synthesizing final report...")
        video_summary = summary_path.read_text(encoding="utf-8")
        report = synthesize_report(video_summary, results)
        report_markdown = render_report_markdown(report)
        (out_dir / "report.md").write_text(report_markdown, encoding="utf-8")
        progress(f"Saved: {out_dir / 'report.md'}")
    else:
        progress(f"Skipping synthesis report -- {summary_path} not found.")

    return {"results": results, "report_markdown": report_markdown}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--timeline", type=Path, required=True, help="timeline.csv from app.features.timeline")
    ap.add_argument("--out-dir", type=Path, default=Path("data/runs"))
    ap.add_argument("--top-n", type=int, default=3, help="investigate only the N biggest drop events")
    ap.add_argument("--summary", type=Path, default=None,
                     help="timeline_summary.txt from app.features.timeline -- feeds the synthesis pass; "
                          "defaults to <out-dir>/timeline_summary.txt")
    args = ap.parse_args()
    analyze_drops_and_report(str(args.video), str(args.timeline), args.out_dir, args.top_n, args.summary)


if __name__ == "__main__":
    main()
