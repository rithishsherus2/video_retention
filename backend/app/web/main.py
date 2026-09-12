"""FastAPI backend for the retention-analysis tool.

A thin HTTP layer over app.agent.orchestrator.run_pipeline. Runs each
request's pipeline in a background thread with an in-memory job store --
correct for a single-user local demo, NOT sufficient for multi-user
production (that would need a real queue/worker and a persistent job
store, e.g. Redis + arq, per the original architecture discussion).

Job results are also recoverable from disk (see _build_done_result): if
the server restarts, or a job isn't in the in-memory dict, /status falls
back to reading run_dir directly. That's what makes it possible to
inspect an already-completed run's data without re-running the pipeline
(and re-spending Gemini quota) too.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.orchestrator import run_pipeline

load_dotenv()

APP_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "web_runs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Server-side Instagram auth -- the visitor never sees or provides this.
# Set COOKIES_FILE in backend/.env to point elsewhere; defaults to
# backend/cookies.txt (gitignored). Every request through this server
# fetches using THIS session -- fine for sharing your own analysis tool
# with people you trust, but worth knowing: anyone who can reach this
# server can cause a fetch under your Instagram identity.
COOKIES_FILE = os.environ.get("COOKIES_FILE", str(Path(__file__).resolve().parents[2] / "cookies.txt"))

app = FastAPI(title="Retention Analysis")

_jobs: dict[str, dict] = {}


def _run_job(job_id: str, reel_url: str, retention_path: Path, run_dir: Path) -> None:
    job = _jobs[job_id]

    def on_progress(msg: str) -> None:
        job["messages"].append(msg)

    try:
        if not Path(COOKIES_FILE).exists():
            raise RuntimeError(
                f"Server isn't configured with Instagram cookies (expected at {COOKIES_FILE}). "
                "This is an operator setup issue, not something the person submitting this form can fix."
            )
        run_pipeline(
            reel_url=reel_url,
            retention_source=str(retention_path),
            run_dir=run_dir,
            cookies_file=COOKIES_FILE,
            on_progress=on_progress,
        )
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


def _find_video_file(run_dir: Path) -> Path | None:
    """The fetched reel itself -- distinct from retention_source.* (the
    uploaded screen-recording of the retention graph) and evidence clips
    (under evidence/, which are trimmed excerpts, not the full reel)."""
    for p in sorted(run_dir.glob("*.mp4")):
        if p.name != "retention_source.mp4":
            return p
    return None


def _build_done_result(job_id: str, run_dir: Path) -> dict | None:
    """Shapes everything the interactive UI needs (video, retention curve,
    per-drop reasons, report) from whatever's on disk for this run. Used
    both for a freshly-finished in-memory job and for disk-recovered ones."""
    retention_path = run_dir / "retention.json"
    video_path = _find_video_file(run_dir)
    if not retention_path.exists() or video_path is None:
        return None

    curve = json.loads(retention_path.read_text(encoding="utf-8"))
    retention_points = [{"t": p["t"], "pct": p["pct"]} for p in curve.get("points", [])]

    drop_events = []
    drop_analysis_path = run_dir / "drop_analysis.json"
    if drop_analysis_path.exists():
        for r in json.loads(drop_analysis_path.read_text(encoding="utf-8")):
            f = r["finding"]
            reason = (f.get("other_observations") or f.get("reasoning")
                      or f.get("description") or "No specific cause identified.")
            drop_events.append({
                "start_t": r["event"]["start_t"], "end_t": r["event"]["end_t"],
                "start_pct": r["event"]["start_pct"], "end_pct": r["event"]["end_pct"],
                "pct_lost": r["event"]["pct_lost"],
                "quality_issue_found": f.get("quality_issue_found", False),
                "severity": f.get("severity", "none"),
                "reason": reason,
                "suggestion": f.get("suggestion") or "",
                "likely_skip": r.get("likely_skip") is not None,
            })

    report_path = run_dir / "report.md"
    return {
        "video_url": f"/api/analyze/{job_id}/file/{video_path.name}",
        "plot_url": f"/api/analyze/{job_id}/file/timeline_debug.png",
        "retention_points": retention_points,
        "video_duration_s": curve.get("video_duration_s"),
        "drop_events": drop_events,
        "report_markdown": report_path.read_text(encoding="utf-8") if report_path.exists() else None,
    }


@app.post("/api/analyze")
async def analyze(
    reel_url: str = Form(...),
    retention_recording: UploadFile = File(...),
):
    job_id = uuid.uuid4().hex[:12]
    run_dir = DATA_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    retention_path = run_dir / f"retention_source{Path(retention_recording.filename or '').suffix}"
    with open(retention_path, "wb") as f:
        shutil.copyfileobj(retention_recording.file, f)

    _jobs[job_id] = {"status": "running", "messages": [], "error": None}
    thread = threading.Thread(
        target=_run_job, args=(job_id, reel_url, retention_path, run_dir), daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/analyze/{job_id}/status")
async def status(job_id: str):
    job = _jobs.get(job_id)
    run_dir = DATA_DIR / job_id

    if job is None:
        # Not tracked in this process (e.g. server restarted) -- try disk.
        result = _build_done_result(job_id, run_dir) if run_dir.exists() else None
        if result is None:
            raise HTTPException(404, "job not found")
        return {"status": "done", "messages": [], "result": result}

    resp = {"status": job["status"], "messages": job["messages"]}
    if job["status"] == "error":
        resp["error"] = job["error"]
    if job["status"] == "done":
        resp["result"] = _build_done_result(job_id, run_dir)
    return resp


@app.get("/api/analyze/{job_id}/file/{filename:path}")
async def get_file(job_id: str, filename: str):
    # Never serve back what was uploaded, only what the pipeline produced --
    # cookies.txt is a live Instagram session token, and the raw retention
    # recording is a personal screen recording; neither should be
    # retrievable by anyone who has (or guesses) a job_id.
    name = Path(filename).name
    if name == "cookies.txt" or name.startswith("retention_source"):
        raise HTTPException(404, "file not found")

    run_dir = (DATA_DIR / job_id).resolve()
    path = (run_dir / filename).resolve()
    if not path.is_relative_to(run_dir) or not path.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(path)


# Registered last: API routes above take priority, this only catches
# whatever's left (the frontend's static HTML/JS/CSS).
app.mount("/", StaticFiles(directory=str(APP_DIR / "static"), html=True), name="static")
