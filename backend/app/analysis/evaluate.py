"""Test a reel against a saved ideal-reel rule profile (see
app.analysis.rules for how that profile gets built).

Two things always happen, regardless of whether a retention recording is
available: engagement is compared against the profile's own benchmark
range, and the reel's decoded creative/technical approach is checked
against the profile's rules -- this is what still explains low views/
engagement even with no retention curve at all.

If a retention recording IS supplied, that's layered in additionally: the
timeline is built WITH retention merged in, drop events are detected and
individually investigated (the original before/during Gemini comparison),
and those per-event findings are folded into the same rule-based critique
so it can point at specific moments, not just general gaps.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.analysis.gemini_client import critique_against_rules, render_evaluation_markdown
from app.analysis.rules import ReelAnalysis, extract_retention_curve, fetch_and_analyze_reel, load_profile
from app.analysis.run import run_drop_event_analysis

ProgressCallback = Callable[[str], None]


def evaluate_reel_against_profile(
    reel_url: str, profile_name: str, run_dir: Path,
    retention_source: str | None = None, cookies_file: str | None = None,
    whisper_model: str = "small", language: str | None = None, top_n_events: int = 3,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Returns a dict with: rules_summary, engagement, features, decoding,
    drop_results (None if no usable retention data), critique, and
    report_markdown. Writes evaluation.json and report.md into run_dir."""
    def progress(msg: str) -> None:
        print(f"[evaluate] {msg}")
        if on_progress:
            on_progress(msg)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rules = load_profile(profile_name)
    progress(f"Loaded profile '{profile_name}' ({len(rules.rules)} rules).")

    retention_json_path = None
    if retention_source:
        progress("Retention recording provided -- extracting curve...")
        retention_json_path = extract_retention_curve(retention_source, run_dir, on_progress)
    else:
        progress("No retention recording provided -- evaluating on rules + engagement only.")

    analysis: ReelAnalysis = fetch_and_analyze_reel(
        reel_url, run_dir, cookies_file, whisper_model, language,
        retention_json_path=retention_json_path, on_progress=on_progress,
    )

    drop_results = None
    if retention_json_path is not None:
        progress("Running drop-event analysis (Gemini)...")
        drop_results = run_drop_event_analysis(
            str(analysis.video_path), analysis.df, run_dir, top_n_events, on_progress,
        )
        (run_dir / "drop_analysis.json").write_text(
            json.dumps(drop_results, indent=2, ensure_ascii=False), encoding="utf-8",
        )

    progress("Critiquing this reel against the profile's rules...")
    critique = critique_against_rules(
        rules, analysis.decoding, analysis.features, analysis.engagement.model_dump(), drop_results,
    )

    report_markdown = render_evaluation_markdown(profile_name, critique, analysis.engagement.model_dump(), drop_results)
    (run_dir / "report.md").write_text(report_markdown, encoding="utf-8")

    result = {
        "reel_url": reel_url,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "profile_name": profile_name,
        "rules_summary": rules.profile_summary,
        "engagement": analysis.engagement.model_dump(),
        "features": analysis.features,
        "decoding": analysis.decoding.model_dump(),
        "drop_results": drop_results,
        "critique": critique.model_dump(),
        "report_markdown": report_markdown,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("Evaluation complete.")
    return result
