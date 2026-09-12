# Design & Logic Reference

This is the "why" document — every non-obvious decision, algorithm, and
tradeoff made while building this, in enough detail to reconstruct the
reasoning without re-deriving it from scratch. `README.md` is the "how to
run it" companion; this is "how it works and why it works that way."

---

## 1. Problem statement

Given a short-form video (a reel) and its per-second viewer-retention
curve, explain **why** retention dropped where it did, with evidence, and
suggest concrete fixes. Two constraints shaped everything:

- **Retention data has no API.** Instagram's per-second retention chart
  only exists inside the Edits app UI and reveals exact numbers only via
  a drag/tap gesture. There is no documented endpoint for it.
- **The end goal is broader than Instagram.** The real target is
  reviewing *any* short-form video — including AI-generated content —
  for a scene/shot that isn't "on par" with its neighbors, correlated
  with where retention actually fell. Nothing in the design should be
  Instagram-specific or assume real (non-generated) footage.

## 2. Architecture at a glance

```
Retention recording ──OCR──► RetentionCurve ──┐
                                               ├──► Feature Timeline ──► Drop/Rise/Skip Detection ──► Evidence ──► Gemini (per-event) ──► Synthesis ──► Report
Reel URL ──yt-dlp+cookies──► Video file ───────┘         (per-second,          (robust stats,      (real video    (visual+audio       (text-only,
                                                          local, free,          this video's         clips per      judgment,           ranks by
                                                          no LLM calls)         own baseline)         event)         one generic         impact)
                                                                                                                      rule so far)
```

Everything left of "Drop/Rise/Skip Detection" is deterministic and free
(local CV/audio processing, no API calls). Only the handful of seconds
around an actual detected event gets sent to Gemini — never the whole
video, never a fixed uniform chunking. This was a deliberate cost/quality
tradeoff: cheap math finds *where* to look; the model judges *what it
means*, only where it's needed.

## 3. Retention extraction (`app/ingest/retention_extract.py`)

**Why OCR at all**: the Edits-app retention graph has no API. A
screen-recorded drag gesture over the graph produces tooltip text
("62% at 0:04") on each frame; OCR extracts it.

**Auto-calibrating crop, not a hardcoded box.** Different capture setups
(a raw phone screenshot vs. a screen-mirrored recording) put the graph at
different pixel fractions of the frame — a fixed crop broke immediately
across our own two test sources. The fix: `autodetect_crop_from_frames`
locates the "100%" y-axis gridline label (present in *every* frame, at a
fixed position) by voting across ~8 sampled frames, and derives a
full-width horizontal band from the top of the frame down to just above
that label — the tooltip always renders above the gridline regardless of
capture layout. **Single-frame detection isn't safe**: at t=0 the
tooltip itself also reads "100%" (retention starts there), colliding with
the axis label. Voting across many frames resolves this because the axis
label is static (same y in every frame) while the tooltip's "100%" only
appears transiently and at a drifting x-position — the static one wins
the vote.

**Outlier cleanup, two distinct failure modes:**
1. A time value beyond the reel's own actual duration (an OCR digit
   misread — can't be a real timestamp). The reel's true duration is
   itself auto-detected the same voting way (the "0:33"-style end-axis
   label is static across frames; a misread tooltip time essentially
   never repeats).
2. A lone percent-value spike that both neighbors disagree with (a
   misread digit on one frame) — dropped only if neighbors on both sides
   agree with each other and disagree with the flagged point, so a real
   single-step drop/rise (a rewatch spike) isn't discarded, only an
   isolated contradiction is.

## 4. Video fetch (`app/ingest/video_fetch.py`)

`yt-dlp` handles the actual download. Auth is the interesting part:
**`--cookies-from-browser chrome` does not work** — Chrome 127+'s
App-Bound Encryption ties cookie decryption to Chrome's own elevation
service; no external tool can read it, DPAPI-decryption dependencies
(`pycryptodome`, `pywin32`) installed and it still fails. The working
path is a manually-exported `cookies.txt` (Netscape format, via a browser
extension). `fetch_reel()`'s parameter priority is `cookies_from_browser`
over `cookies_file` if both given — so any code calling it must default
`cookies_from_browser=None`, never `"chrome"`, or it'll silently retry
the broken path.

## 5. Feature timeline (`app/features/`) — all deterministic, no LLM calls

### Quality (`quality.py`)
Four per-frame metrics: sharpness (variance of Laplacian), blockiness
(ratio of pixel discontinuity at 8px H.264-macroblock-grid boundaries vs.
elsewhere — real detail follows scene content, compression artifacts
follow the fixed grid regardless of content), high-frequency energy ratio
(FFT-based; catches effective-resolution loss/upscaling), brightness.

**Per-shot z-scoring, not global.** Originally scored against the whole
video's mean/std; this conflates "a frame is blurry relative to its own
shot's neighbors" (a real defect) with "a whole shot just looks different
from the video's average" (could just be a new location/lighting — not a
defect by itself). Fixed to score each frame against *its own shot's*
mean/std (`zscore_flags_per_shot`), which catches genuine mid-shot
glitches without false-flagging ordinary scene-to-scene variation.
Shot-vs-previous-shot deltas (`shot_level_quality_deltas`) are kept as a
*second*, separate signal — raw percentage change, not a verdict, because
telling "this is a real defect" from "this is a deliberate new
scene/style" requires seeing the actual frames, which is deferred to the
Gemini pass, not decided by a z-score.

### Shots (`shots.py`)
PySceneDetect for cut boundaries. CLIP (`ViT-B-32-quickgelu`/openai —
note the `-quickgelu` suffix: the plain `ViT-B-32` config defaults to
GELU and silently produces degraded embeddings against these particular
weights, which were trained with QuickGELU) embeds each shot's middle
frame for similarity. Two distinct flags from the same similarity matrix:
- **Redundant cut**: shot vs. the *immediately preceding* shot (≥0.92) —
  a cut to near-identical content.
- **Reused shot**: shot vs. *any other non-adjacent* shot (≥0.90) — the
  same footage reappearing later.
Known limitation: one embedding per shot (its middle frame) — a shot that
pans from A to B is only represented by whatever's on-screen at its
midpoint; catches "same clip inserted twice" reliably, weaker on partial
overlaps.

### Audio (`audio.py`)
RMS loudness curve (dBFS proxy for LUFS) for silence/jump detection —
this doesn't care whether the signal is speech or music, so it's valid
regardless of content type. Whisper (`faster-whisper`) transcription with:
- **Language auto-detection by default**, not forced — a real product
  reviewing arbitrary content can't assume the language ahead of time.
  Confidence-gated escalation (`small` → `medium`) if
  `info.language_probability` is below 0.6, since `base` was seen to
  misidentify Telugu audio as Tamil and return near-nothing, while
  `small` alone resolved it correctly without forcing anything.
- **Explicit >80%-speech-confidence gate**, applied in our own code
  (`MAX_NO_SPEECH_PROB = 0.20`), not left to library defaults. Every
  segment must clear `no_speech_prob < 0.20` AND `avg_logprob > -1.0`
  before being trusted. This was tuned against two real failure cases: a
  forced-past-the-gate `small`-model attempt produced `no_speech_prob`
  0.83–0.91 with incoherent script-valid-but-meaningless text (clear
  hallucination); a cleaner `medium`-model-on-isolated-vocals attempt
  landed at 0.586–0.737 — more plausible-looking, still correctly
  rejected under the 80% bar. The gate is deliberately strict: *product
  decision was "skip it" over "guess and maybe be wrong."*
- **Vocal-separation fallback** (Demucs, `htdemucs`) for lyrics/dialogue
  buried under background music — only triggered when the full mix
  produces zero confident segments and the audio isn't just silent. Every
  segment from *either* path (full mix or isolated vocals) still has to
  clear the same confidence gate; separation makes real content easier to
  find, it never lowers the bar for trusting what comes out. On real test
  content this correctly declined to guess at lyrics rather than
  hallucinate, even after separation.
- **Per-model-size cache**, not a single global — an earlier bug cached
  one Whisper instance regardless of requested size, which would have
  silently defeated the escalation ladder (asking for `medium` would
  return the already-cached `small`).

### Dialogue redundancy (`dialogue.py`)
Sentence-embedding similarity (`all-MiniLM-L6-v2`, not CLIP's text
encoder — CLIP is image-text aligned, not tuned for general sentence
semantics) over Whisper's *segment*-level transcript (natural
phrase/sentence units, not individual words). Flags a segment if its most
similar *other* segment (excluding itself) clears 0.82 similarity, with a
minimum-word filter so trivially short utterances ("okay", "yeah") don't
register as false "redundancy" from generic high similarity. Not
exercised on real data yet (our test reel has no dialogue at all) —
validated against synthetic paraphrase/negative/empty-transcript cases
instead.

## 6. Drop / rise / skip detection (`app/analysis/drops.py`)

**Robust (median/MAD) threshold on the per-second retention *rate of
change*, not a fixed percentage.** Different videos have wildly different
overall decay shapes; only a deviation from *this video's own* typical
pace is a meaningful signal that something specific happened at that
moment; every video's retention is *always* decreasing on average, so a
raw derivative flags noise.

**Lag window.** A viewer decides to leave some time *after* whatever
triggered it, not the instant it happens — so the investigation window
for a drop starting at `t` reaches back to `t - 2.0s` (before the
retention number visibly moves) through `t_end + 0.5s`. This offset is a
named constant (`LAG_BEFORE_S`/`LAG_AFTER_S`), not folklore baked into
the math.

**Rises and skips, generalized beyond this reel.** Our own Instagram
Reels retention curve is monotonically decreasing throughout (this
particular graph style never shows recovery), so `detect_rise_events`
(the exact mirror of drop detection on the opposite tail) legitimately
returns nothing on real data here — validated against synthetic
dip-then-recover data instead. This matters for other platforms/players
where per-second retention is built from playhead-position samples: a
viewer who skips forward stops counting for the seconds they jumped over
and resumes counting at the landing point, producing a dip-then-partial-
recovery shape that looks like two separate events but is one skip.
`find_skip_patterns` pairs a drop with a nearby (within 20s — tuned
against a "skip 10 seconds" scenario, not guessed) subsequent rise that
recovers ≥50% of what was lost, and tags it as *likely a skip*, not full
disengagement, so the report doesn't overstate severity or suggest a fix
for something that wasn't actually abandonment.

## 7. Evidence gathering (`app/analysis/evidence.py`)

For each event: a "before" window (a clean reference clip, *before* the
lag-adjusted investigation window even starts, not overlapping it) and a
"during" window (the lag-adjusted window itself). Real video+audio clips
get trimmed via ffmpeg (`-ss`/`-to` placed *after* `-i` for frame-accurate
cuts — worth the slower seek since these clips are only seconds long) —
**not sampled still frames**. This was a deliberate upgrade mid-project:
stills cannot show motion/temporal continuity or audio-visual
correlation, which is exactly what matters for the actual target use case
(AI-generation artifacts like a face morphing across frames, or audio
that doesn't sync to the action) — a defect that's invisible to any
number of discrete stills but visible the instant you watch the clip
move. A minimum clip duration (1.2s) is enforced by widening
symmetrically, since a drop right at the very start of a video can
produce a technically-valid but zero-width window.

## 8. Gemini analysis (`app/analysis/gemini_client.py`)

**Two passes, not one:**
1. **Per-event (vision+audio)**: uploads the before/during clips via the
   Files API (polls until `ACTIVE` — video/audio uploads go through a
   processing state), asks Gemini to judge whether the DURING clip shows
   a genuine quality/rendering problem *as a viewer would perceive it* —
   explicitly not assuming real vs. AI-generated content, and explicitly
   told a content/subject change alone isn't a quality issue. Structured
   output includes `other_observations` for anything notable that ISN'T
   a quality defect (tone/pacing/audio shifts) — this is deliberately
   NOT forced into the quality-only schema, because on real test data the
   model correctly and repeatedly said "no rendering defect here" while
   still surfacing the actual cause (a held-too-long silent hook, an
   abrupt audio jump) through that field. Two independent detection paths
   (the deterministic audio-jump signal and Gemini's own vision+audio
   read) converged on the same explanation without one feeding the
   other — real corroboration, not circular reasoning.
2. **Synthesis (text-only)**: reasons over the per-event JSON findings
   *and* the full per-second timeline digest (not just the 3 investigated
   events) — this is why the final report can cite the reused-shot/
   redundant-cut flags at t=22s/27s even though those weren't
   individually sent to Gemini for vision analysis. Explicitly instructed
   to describe a `likely_skip`-tagged event as a probable skip, not full
   disengagement.

**Model naming uses `-latest` aliases** (`gemini-flash-latest`), not
dated model names — `gemini-2.5-flash` and `gemini-2.5-pro` were both
retired mid-project ("no longer available to new users"), breaking a
hardcoded name; aliases avoid re-breaking on the next retirement.

**429 vs. 503 are handled differently, on purpose.** A 503 ("high
demand") is transient — worth a backoff-and-retry (up to 6 attempts,
exponential: 3/6/12/24/48s). A 429 (quota exhausted) is NOT transient —
retrying burns more requests against an already-exhausted budget for no
benefit (this happened during testing: blind retries on a 429 made the
situation worse, not better). The client now fails fast on 429 with an
actionable message instead of retrying.

## 9. Agent (`app/agent/orchestrator.py`)

Honest framing: this is a **sequencer**, not an autonomous
decision-maker. `run_pipeline()` calls the four stages in order
(fetch → retention-extract → timeline → analyze+report), threading a
progress callback through all of them. All the actual judgment/adaptive
behavior (retry policy, model escalation, vocal-separation fallback)
lives inside the individual stage functions, not in the orchestrator
itself — it makes zero decisions; it just calls things in order and
stops if one throws. It replaces the four hand-run CLI commands used
during development with one call, which is the sense in which "agent"
applies here. It is explicitly NOT the other sense from the original
brief — something that watches an account and fires automatically with
no manual link/recording input — which remains unbuilt.

## 10. Web layer (`app/web/`)

Single-process, in-memory job tracking (a dict keyed by job id) —
correct for a local demo, not for multi-user production (would need a
real queue + persistent store). **Disk-recoverable by design**: `/status`
reconstructs a "done" result by reading `run_dir` directly if the job
isn't in memory (e.g. after a server restart), which is also what makes
the `?job=<id>` direct-link feature possible — reopening an already-
computed result costs zero new API calls.

**Server-side credentials, not a per-request upload.** Instagram
`cookies.txt` lives at `backend/cookies.txt` (gitignored, path
overridable via `COOKIES_FILE` in `.env`) and is used for every fetch
regardless of who submits the form. Tradeoff made explicitly: this is
right for sharing *your own* tool with people you trust (no login means
anyone with the link can trigger a fetch under your Instagram identity,
which is why the base URL shouldn't be posted publicly), and wrong for
any multi-tenant use where different users need different credentials.

**File-serving allowlist by exclusion, not inclusion.** `/file/{filename}`
explicitly refuses `cookies.txt` and anything starting with
`retention_source` (the raw uploaded recording) — only pipeline *outputs*
are servable, never uploaded *inputs*. This was a real bug found and
fixed before the tool was ever exposed publicly (via a Cloudflare quick
tunnel) — caught by reasoning about what a stranger with a job_id could
retrieve, not by an automated scan.

## 11. Known gaps (see also README's "Known limitations")

- Only one Gemini-reasoned rule exists (generic quality/rendering vs.
  neighbors). The deterministic redundant-cut/reused-shot/redundant-
  dialogue detectors already run and appear in the timeline digest, but
  aren't each individually reasoned about by Gemini yet — same
  before/during-evidence pattern would extend to them.
- Song-lyric transcription: vocal separation + confidence gating is
  wired in and correctly *declines* to guess on real test content.
  Reliable lyric recovery needs audio-fingerprint + lyrics-database
  lookup (the track is probably identifiable/known), not better ASR.
- The orchestrator has no standalone CLI entrypoint yet — reachable only
  via the web layer or direct Python import.
- No authentication anywhere in the web layer.
