"""FastAPI backend for the retention-analysis tool.

A thin HTTP layer over three pipelines, each run in a background thread
with an in-memory job store -- correct for a single-user local demo, NOT
sufficient for multi-user production (that would need a real queue/worker
and a persistent job store, e.g. Redis + arq, per the original
architecture discussion):

- /api/analyze (legacy): app.agent.orchestrator.run_pipeline -- the
  original generic visual-quality-vs-retention-drop check, no ideal-reel
  comparison. Still useful when there's no saved project yet.
- /api/projects: app.analysis.rules.extract_rules_from_ideal_reels --
  decode a handful of well-performing reels (each with an independently
  optional retention recording) into a saved, reusable rule set
  (data/profiles/<name>/ -- "profile" internally, "project" on the wire
  and in the UI, since that's the unit the user creates/reuses/picks).
- /api/evaluate: app.analysis.evaluate.evaluate_reel_against_profile --
  check a new reel against a saved project's rules, with a retention
  recording optional (see that module's docstring for what changes when
  one is or isn't supplied).

Job results for /api/analyze are also recoverable from disk (see
_build_done_result) if the server restarts or a job isn't in the
in-memory dict; /api/evaluate does the same via evaluation.json.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.orchestrator import run_pipeline
from app.analysis import rules as rules_mod
from app.analysis.evaluate import evaluate_reel_against_profile

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


# ---------------------------------------------------------------------------
# Projects: decode a handful of "ideal" reels once into a saved, reusable
# rule set, then evaluate any number of later reels against it. One
# project = one saved rule set (app.analysis.rules calls this a "profile"
# internally -- data/profiles/<name>/ -- but the HTTP surface and UI use
# "project", since that's the unit the user actually creates/reuses/picks
# between: decode a new niche/page's ideal reels -> new project). This is
# the same background-thread + in-memory-job-store pattern as /api/analyze
# above, just for two new job kinds.
# ---------------------------------------------------------------------------

async def _save_upload(upload: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = dest.parent / f"{dest.name}{Path(upload.filename or '').suffix}"
    content = await upload.read()
    path.write_bytes(content)
    return path


def _run_project_job(job_id: str, reels: list[dict], project_name: str, run_dir: Path) -> None:
    job = _jobs[job_id]

    def on_progress(msg: str) -> None:
        job["messages"].append(msg)

    try:
        if not Path(COOKIES_FILE).exists():
            raise RuntimeError(
                f"Server isn't configured with Instagram cookies (expected at {COOKIES_FILE})."
            )
        rules_mod.extract_rules_from_ideal_reels(
            reels=reels, profile_name=project_name, run_dir=run_dir,
            cookies_file=COOKIES_FILE, on_progress=on_progress,
        )
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/projects")
async def create_project(request: Request):
    """multipart/form-data: project_name (str), reel_urls (str, one URL
    per line or comma-separated, 2-8 reels), and OPTIONALLY one file field
    per reel named retention_0, retention_1, ... (0-indexed, matching
    reel_urls' order) -- each is that ideal reel's own retention-graph
    recording, entirely optional and independent per reel. Not using
    typed Form()/File() params here because the number of retention_N
    fields is dynamic (2-8), unknown until reel_urls is parsed."""
    form = await request.form()
    project_name = str(form.get("project_name") or "").strip()
    reel_urls_raw = str(form.get("reel_urls") or "")
    urls = [u.strip() for u in reel_urls_raw.replace(",", "\n").splitlines() if u.strip()]

    if not project_name:
        raise HTTPException(400, "project_name is required.")
    if not (2 <= len(urls) <= 8):
        raise HTTPException(400, f"Give 2-8 ideal reel URLs to decode a pattern from (got {len(urls)}).")

    job_id = uuid.uuid4().hex[:12]
    run_dir = DATA_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reels: list[dict] = []
    for i, url in enumerate(urls):
        field = form.get(f"retention_{i}")
        # request.form() returns starlette.datastructures.UploadFile for file fields,
        # NOT fastapi.UploadFile (they are distinct classes in this stack) -- duck-type
        # on the attribute that only an actual upload has instead of isinstance().
        retention_source = None
        if field is not None and hasattr(field, "filename") and field.filename:
            path = await _save_upload(field, run_dir / f"ideal_{i}" / "retention_source")
            retention_source = str(path)
        reels.append({"url": url, "retention_source": retention_source})

    _jobs[job_id] = {"status": "running", "messages": [], "error": None}
    thread = threading.Thread(
        target=_run_project_job, args=(job_id, reels, project_name, run_dir), daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/projects/jobs/{job_id}")
async def project_job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    resp = {"status": job["status"], "messages": job["messages"]}
    if job["status"] == "error":
        resp["error"] = job["error"]
    if job["status"] == "done":
        resp["projects"] = rules_mod.list_profiles()
    return resp


@app.get("/api/projects")
async def list_projects():
    return {"projects": rules_mod.list_profiles()}


@app.get("/api/projects/{name}")
async def get_project(name: str):
    try:
        rules = rules_mod.load_profile(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return rules.model_dump()


def _scan_tests(project_filter: str | None = None) -> list[dict]:
    """Every reel previously evaluated via /api/evaluate, newest first,
    optionally filtered to one project -- no separate index is kept, this
    just scans DATA_DIR for evaluation.json files (one per past job). Fine
    at this scale (a personal tool, not many jobs); would need a real
    index if run dirs ever grow into the thousands."""
    tests = []
    for job_dir in DATA_DIR.iterdir():
        eval_path = job_dir / "evaluation.json"
        if not job_dir.is_dir() or not eval_path.exists():
            continue
        try:
            result = json.loads(eval_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        project_name = result.get("profile_name")
        if project_filter is not None and project_name != project_filter:
            continue
        critique = result.get("critique") or {}
        engagement = result.get("engagement") or {}
        tests.append({
            "job_id": job_dir.name,
            "project_name": project_name,
            "reel_url": result.get("reel_url"),
            "evaluated_at": result.get("evaluated_at"),
            "headline": critique.get("headline"),
            "views": engagement.get("views"),
            "likes": engagement.get("likes"),
            "comments": engagement.get("comments"),
        })
    tests.sort(key=lambda t: t["evaluated_at"] or "", reverse=True)
    return tests


@app.get("/api/tests")
async def list_all_tests():
    return {"tests": _scan_tests()}


@app.get("/api/projects/{name}/tests")
async def list_project_tests(name: str):
    return {"tests": _scan_tests(project_filter=name)}


@app.delete("/api/projects/{name}")
async def delete_project(name: str):
    project_dir = (rules_mod.PROFILES_DIR / name).resolve()
    if not project_dir.is_relative_to(rules_mod.PROFILES_DIR.resolve()) or not project_dir.exists():
        raise HTTPException(404, "project not found")
    shutil.rmtree(project_dir)
    return {"deleted": name}


# ---------------------------------------------------------------------------
# Evaluate a reel against a saved project's rules. retention_recording is
# optional -- see app.analysis.evaluate's docstring for what changes when
# it's provided vs. omitted.
# ---------------------------------------------------------------------------

def _run_evaluate_job(
    job_id: str, reel_url: str, project_name: str, retention_path: Path | None, run_dir: Path,
) -> None:
    job = _jobs[job_id]

    def on_progress(msg: str) -> None:
        job["messages"].append(msg)

    try:
        if not Path(COOKIES_FILE).exists():
            raise RuntimeError(
                f"Server isn't configured with Instagram cookies (expected at {COOKIES_FILE})."
            )
        result = evaluate_reel_against_profile(
            reel_url=reel_url, profile_name=project_name, run_dir=run_dir,
            retention_source=str(retention_path) if retention_path else None,
            cookies_file=COOKIES_FILE, on_progress=on_progress,
        )
        job["status"] = "done"
        job["evaluate_result"] = result
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/evaluate")
async def evaluate(
    project_name: str = Form(...),
    reel_url: str = Form(...),
    retention_recording: UploadFile | None = File(None),
):
    job_id = uuid.uuid4().hex[:12]
    run_dir = DATA_DIR / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    retention_path = None
    if retention_recording is not None and retention_recording.filename:
        retention_path = run_dir / f"retention_source{Path(retention_recording.filename).suffix}"
        with open(retention_path, "wb") as f:
            shutil.copyfileobj(retention_recording.file, f)

    _jobs[job_id] = {"status": "running", "messages": [], "error": None}
    thread = threading.Thread(
        target=_run_evaluate_job, args=(job_id, reel_url, project_name, retention_path, run_dir), daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


def _build_evaluate_view(job_id: str, run_dir: Path, result: dict) -> dict:
    """Layers on what the UI needs beyond evaluate_reel_against_profile's
    raw result: a playable video URL, and -- only if a retention recording
    was actually supplied and produced a curve -- the retention points and
    shaped drop events, in the exact shape the retention chart component
    (shared with the legacy /api/analyze view) expects."""
    view = dict(result)
    video_path = _find_video_file(run_dir)
    if video_path is not None:
        view["video_url"] = f"/api/runs/{job_id}/file/{video_path.name}"

    retention_path = run_dir / "retention.json"
    if retention_path.exists():
        curve = json.loads(retention_path.read_text(encoding="utf-8"))
        view["retention_points"] = [{"t": p["t"], "pct": p["pct"]} for p in curve.get("points", [])]
        view["video_duration_s"] = curve.get("video_duration_s")
        view["drop_events"] = []
        for r in result.get("drop_results") or []:
            f = r["finding"]
            reason = (f.get("other_observations") or f.get("reasoning")
                      or f.get("description") or "No specific cause identified.")
            view["drop_events"].append({
                "start_t": r["event"]["start_t"], "end_t": r["event"]["end_t"],
                "start_pct": r["event"]["start_pct"], "end_pct": r["event"]["end_pct"],
                "pct_lost": r["event"]["pct_lost"],
                "quality_issue_found": f.get("quality_issue_found", False),
                "reason": reason,
                "suggestion": f.get("suggestion") or "",
                "likely_skip": r.get("likely_skip") is not None,
            })
    return view


@app.get("/api/evaluate/{job_id}/status")
async def evaluate_status(job_id: str):
    job = _jobs.get(job_id)
    run_dir = DATA_DIR / job_id

    if job is None:
        eval_path = run_dir / "evaluation.json"
        if not eval_path.exists():
            raise HTTPException(404, "job not found")
        result = json.loads(eval_path.read_text(encoding="utf-8"))
        return {"status": "done", "messages": [], "result": _build_evaluate_view(job_id, run_dir, result)}

    resp = {"status": job["status"], "messages": job["messages"]}
    if job["status"] == "error":
        resp["error"] = job["error"]
    if job["status"] == "done":
        resp["result"] = _build_evaluate_view(job_id, run_dir, job.get("evaluate_result") or {})
    return resp


@app.get("/api/runs/{job_id}/file/{filename:path}")
async def get_run_file(job_id: str, filename: str):
    # Generic version of /api/analyze/.../file/... -- both job kinds write
    # into DATA_DIR/<job_id>, this just isn't namespaced under /api/analyze.
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
