# Video Retention Analysis

Paste a reel link + a recording of its retention graph, get a causal
explanation of where and why viewers dropped off, and concrete
suggestions -- built to generalize beyond Instagram Reels to any short-form
video review, including AI-generated content review (does a scene/shot
look "off" compared to its neighbors, correlated with where retention
actually fell).

## Why this exists / how it works

See the pipeline stages below -- each one was built and validated against
real data before the next was added. The short version: a deterministic
layer computes hard numbers (quality, cuts, audio, transcript) cheaply and
locally; only the handful of seconds around an actual detected retention
drop gets sent to Gemini, which does the visual/contextual judgment a
z-score can't (is this a real quality defect, or just a new scene?).

## Setup

Requires: Python 3.12, ffmpeg + Tesseract OCR on PATH (installed via
winget during development), a free Gemini API key.

```
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then fill in GEMINI_API_KEY (https://aistudio.google.com/apikey)
```

Instagram auth: Chrome's cookie encryption (App-Bound Encryption) blocks
yt-dlp's `--cookies-from-browser` -- use a manually exported `cookies.txt`
(via the "Get cookies.txt LOCALLY" browser extension) instead. Never
commit this file; it's your login session.

## Getting the retention curve

The retention graph only exists inside the Instagram **Edits** app (not
the main Instagram app), and only reveals exact numbers via a drag/tap
gesture ("62% at 0:04" tooltips). There's no API for it. Screen-record
yourself dragging across the graph (or take discrete tap screenshots),
and the pipeline OCRs the tooltip out of each frame -- see
`app/ingest/retention_extract.py`'s module docstring for the full
extraction design (auto-calibrating crop detection, outlier cleanup).

## Running it

**Web UI** (the easiest way):
```
cd backend
.venv\Scripts\python -m uvicorn app.web.main:app --reload
```
Open http://127.0.0.1:8000, paste the reel URL, upload the retention
recording + your `cookies.txt`, and watch it run.

**CLI, stage by stage** (useful for debugging a specific step):
```
python -m app.ingest.video_fetch <reel_url> --cookies-file cookies.txt
python -m app.ingest.retention_extract --video <recording.mp4> --out retention.json
python -m app.features.timeline --video <reel.mp4> --retention retention.json --out-dir data/runs
python -m app.analysis.run --video <reel.mp4> --timeline data/runs/timeline.csv --out-dir data/runs
```

**One call** (what the web backend actually uses):
```python
from app.agent.orchestrator import run_pipeline
result = run_pipeline(reel_url=..., retention_source=..., run_dir=..., cookies_file=...)
```

## Pipeline stages

1. **Ingest** (`app/ingest/`) -- `video_fetch.py` (yt-dlp + cookies),
   `retention_extract.py` (OCR the drag-recording into a `RetentionCurve`,
   with auto-calibrated crop detection and outlier cleanup).
2. **Feature timeline** (`app/features/`) -- per-second, deterministic,
   no LLM calls: `quality.py` (sharpness/blockiness/etc., z-scored
   *per-shot* not globally), `shots.py` (cut detection + CLIP similarity
   for redundant-cut/reused-shot flags), `audio.py` (loudness curve +
   Whisper transcript, with an explicit >80%-speech-confidence gate and a
   vocal-separation fallback for lyrics/dialogue buried under music),
   `dialogue.py` (semantic redundant-dialogue detection), `timeline.py`
   (aligns everything to the retention curve into one table + a
   human-readable digest + a debug plot).
3. **Analysis** (`app/analysis/`) -- `drops.py` (robust MAD-based drop/rise
   detection, plus a skip-vs-disengagement classifier for platforms where
   retention can recover), `evidence.py` (extracts real before/during
   video+audio clips around each event), `gemini_client.py` (uploads the
   clips to Gemini for the visual/causal judgment, then a text-only
   synthesis pass into the final report), `run.py` (orchestrates the
   above into `drop_analysis.json` + `report.md`).
4. **Agent** (`app/agent/orchestrator.py`) -- chains all of the above into
   one call. This is "the agent" in the sense of replacing manual,
   four-separate-CLI-command orchestration; an agent that watches your
   account and fires on new reels automatically (no manual link/recording
   needed) is a separate, still-open piece.
5. **Web** (`app/web/`) -- a thin FastAPI layer over the orchestrator,
   plus a single-page vanilla-JS frontend (no build step).

## Known limitations / open items

- Song lyrics under heavy instrumentation: vocal separation (Demucs) is
  wired in as a fallback, but only trusted above an explicit >80%
  speech-confidence bar -- on real test content this correctly declined
  to guess rather than hallucinate. For reliable lyric transcription,
  audio-fingerprint + lyrics-database lookup (the song is probably a
  known/trending track) will beat ASR-on-singing; a bigger commercial ASR
  model (Whisper `large-v3` via API, Google Chirp) is the fallback for
  genuinely original audio.
- The web backend is single-process, in-memory job tracking -- fine for
  local/demo use, not multi-user production (would need a real queue +
  persistent job store).
- Only one analysis rule is wired into the Gemini pass so far (generic
  visual/rendering quality vs. neighbors). More rules (the redundant-cut/
  reused-shot/redundant-dialogue deterministic detectors already exist
  and are visible in the timeline digest, but aren't yet each their own
  Gemini-reasoned finding) can be added the same way once needed.
- Retention extraction assumes the drag/tap-tooltip UI Instagram Edits
  currently uses; a client's own instrumented player would skip OCR
  entirely and feed a `RetentionCurve` directly.
