"""Extract a RetentionCurve from a manually captured recording of Instagram
Edits' retention graph.

Why this exists: the retention graph only lives inside the Edits mobile app
and only reveals exact numbers when you drag/tap across it (tooltip like
"62% at 0:04"). There's no API for it. So for the demo, you:

  1. Open the reel's retention graph in Edits.
  2. Screen-record yourself slowly dragging a finger left-to-right across
     the graph (pause briefly every ~0.5-1s of video-time so each point
     gets a few clean, unblurred frames), OR take discrete screenshots
     while tapping point by point.
  3. Save that recording/screenshots into data/samples/<run_id>/.
  4. Run this script to turn it into retention.json.

Design goal: this must be cheap. It's plain OCR on a small, fixed-position
crop -- no model calls per frame. A vision-LLM fallback is only invoked
(later, once wired up) for frames OCR fails to parse.

Nothing else in the pipeline should import anything UI-specific from this
file -- it must only ever produce/consume app.schemas.RetentionCurve.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from app.schemas import RetentionCurve, RetentionPoint

# --- Tesseract binary location -------------------------------------------
# On Windows the pip package only installs the Python wrapper; the actual
# OCR engine (tesseract.exe) has to be installed separately and pointed to
# here if it's not on PATH.
_DEFAULT_WIN_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if Path(_DEFAULT_WIN_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_WIN_TESSERACT


@dataclass
class CropBox:
    """Fixed region where the tooltip renders, as fractions (0-1) of the
    frame so it's resolution-independent across devices."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def parse(cls, s: str) -> "CropBox":
        x0, y0, x1, y1 = (float(v) for v in s.split(","))
        return cls(x0, y0, x1, y1)

    def crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x0, x1 = int(self.x0 * w), int(self.x1 * w)
        y0, y1 = int(self.y0 * h), int(self.y1 * h)
        return img[y0:y1, x0:x1]


# Fallback only -- different capture setups (phone screenshot vs. a
# screen-mirrored recording, different devices/aspect ratios) put the
# tooltip at different fractional positions, so a hardcoded box is fragile.
# See autodetect_crop() below, which is what's actually used by default:
# it locates the "100%" gridline label (always present, always the topmost
# axis text) and derives a full-width band from the top of the frame down
# to just above it -- the tooltip always renders above that gridline
# regardless of capture layout.
DEFAULT_CROP = CropBox(0.0, 0.0, 1.0, 0.20)

_HUNDRED_PCT_RE = re.compile(r"100\s*%")

# Accepts variants like:
#   "62% at 0:04"   "62% · 0:04"   "0:04 62%"   "62%\n0:04"   "4s 62%"
_PCT_RE = r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*%"
_TIME_MMSS_RE = r"(?P<min>\d+):(?P<sec>\d{1,2})"
_TIME_SEC_RE = r"(?P<secs>\d+(?:\.\d+)?)\s*s\b"


def _find_100pct_y_fractions(img: np.ndarray) -> list[float]:
    """All y-positions (as a fraction of frame height) where '100%' text is
    found in this frame. There can be more than one: the static axis
    gridline label AND, whenever retention is still ~100% at that instant,
    the tooltip's own '100%' reading -- these are easy to confuse from a
    single frame (see autodetect_crop_from_frames)."""
    from pytesseract import Output

    h = img.shape[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, config="--psm 11")
    return [data["top"][i] / h for i, txt in enumerate(data["text"]) if _HUNDRED_PCT_RE.search(txt or "")]


def autodetect_crop(img: np.ndarray, margin_frac: float = 0.02) -> CropBox | None:
    """Single-frame version: find the '100%' text closest to the axis
    position (lower/further down = more likely the static gridline, since
    the tooltip floats above it) and derive the tooltip band as [top of
    frame, just above that]. Only used for one-off --calibrate checks --
    the real extraction run uses autodetect_crop_from_frames instead, which
    is robust to the t=0 collision where the tooltip ALSO reads '100%'."""
    fractions = _find_100pct_y_fractions(img)
    if not fractions:
        return None
    top_y_frac = max(fractions)  # furthest down = the gridline, not a floating tooltip
    y1 = max(0.05, top_y_frac - margin_frac)
    return CropBox(0.0, 0.0, 1.0, y1)


def autodetect_crop_from_frames(
    frame_paths: list[Path], n_samples: int = 8, margin_frac: float = 0.02, bucket: float = 0.01
) -> CropBox | None:
    """Robust version for a full recording: sample several frames spread
    across the whole thing and find the y-fraction where '100%' shows up
    MOST CONSISTENTLY. The axis gridline is static and present in every
    frame; a tooltip reading '100%' only happens transiently (near t=0) and
    at a drifting x-position -- so voting across frames isolates the
    gridline even when a single frame is ambiguous."""
    if not frame_paths:
        return None
    step = max(1, len(frame_paths) // n_samples)
    sample_paths = frame_paths[::step][:n_samples]

    votes: dict[float, int] = {}
    for fp in sample_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        for frac in _find_100pct_y_fractions(img):
            key = round(frac / bucket) * bucket
            votes[key] = votes.get(key, 0) + 1

    if not votes:
        return None

    best_y = max(votes, key=lambda k: votes[k])
    y1 = max(0.05, best_y - margin_frac)
    return CropBox(0.0, 0.0, 1.0, y1)


def autodetect_reel_duration(frame_paths: list[Path], n_samples: int = 8) -> float | None:
    """The reel's own duration (e.g. '0:33') is printed as a static x-axis
    label in every frame -- distinct from the screen-recording's own
    length, which is unrelated (you might drag slower/faster than
    real-time). Find it the same way as the gridline: vote for the mm:ss
    value that shows up identically across many sampled frames, since a
    transiently-read tooltip time essentially never repeats verbatim."""
    if not frame_paths:
        return None
    step = max(1, len(frame_paths) // n_samples)
    sample_paths = frame_paths[::step][:n_samples]

    votes: dict[float, int] = {}
    for fp in sample_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 11")
        for m in re.finditer(_TIME_MMSS_RE, text):
            t = int(m.group("min")) * 60 + int(m.group("sec"))
            if t > 0:  # skip the "0:00" start label
                votes[t] = votes.get(t, 0) + 1

    if not votes:
        return None
    return float(max(votes, key=lambda k: votes[k]))


def clean_curve(points: list[RetentionPoint], max_t: float | None, spike_threshold: float = 20.0) -> list[RetentionPoint]:
    """Drop the two OCR-failure shapes that survive individually-valid
    parses: (1) a time reading beyond the reel's actual duration -- a
    digit misread, since it can't be a real timestamp -- and (2) a lone
    percent-value spike that both neighbors disagree with (a misread
    digit on one frame), while leaving real single-step drops/rises
    (rewatch loops) alone since those are corroborated by where the curve
    goes next, not contradicted by it."""
    pts = list(points)

    if max_t is not None:
        tol = 1.0
        pts = [p for p in pts if p.t <= max_t + tol]

    cleaned: list[RetentionPoint] = []
    for i, p in enumerate(pts):
        if 0 < i < len(pts) - 1:
            prev_pct, next_pct = pts[i - 1].pct, pts[i + 1].pct
            neighbors_agree = abs(prev_pct - next_pct) < spike_threshold
            this_disagrees = abs(p.pct - prev_pct) >= spike_threshold and abs(p.pct - next_pct) >= spike_threshold
            if neighbors_agree and this_disagrees:
                continue  # lone spike, neighbors on both sides contradict it
        cleaned.append(p)
    return cleaned


def extract_frames(video_path: Path, out_dir: Path, fps: float = 6.0) -> list[Path]:
    """Pull frames out of the drag-recording at a fixed rate via ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%05d.png")
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", pattern]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return sorted(out_dir.glob("frame_*.png"))


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Upscale + binarize so small UI text OCRs reliably."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_crop(crop: np.ndarray) -> str:
    processed = preprocess_for_ocr(crop)
    return pytesseract.image_to_string(processed, config="--psm 6")


def parse_tooltip_text(text: str) -> tuple[float, float] | None:
    """Return (t_seconds, pct) if both are found in the OCR'd text, else None."""
    pct_match = re.search(_PCT_RE, text)
    if not pct_match:
        return None
    pct = float(pct_match.group("pct"))

    mmss_match = re.search(_TIME_MMSS_RE, text)
    if mmss_match:
        t = int(mmss_match.group("min")) * 60 + int(mmss_match.group("sec"))
        return float(t), pct

    sec_match = re.search(_TIME_SEC_RE, text)
    if sec_match:
        return float(sec_match.group("secs")), pct

    return None


def extract_curve_from_frames(
    frame_paths: list[Path], crop_box: CropBox | None, dedupe_eps_s: float = 0.15
) -> RetentionCurve:
    if crop_box is None:
        crop_box = autodetect_crop_from_frames(frame_paths)
        if crop_box is not None:
            print(f"[retention_extract] auto-detected crop band (voted across "
                  f"{min(8, len(frame_paths))} sampled frames): y1={crop_box.y1:.3f}", file=sys.stderr)
        else:
            print("[retention_extract] could not auto-detect the '100%' gridline across sampled "
                  "frames -- falling back to DEFAULT_CROP. Consider passing --crop explicitly.",
                  file=sys.stderr)
            crop_box = DEFAULT_CROP

    raw_points: list[RetentionPoint] = []
    failures = 0
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        crop = crop_box.crop(img)
        text = ocr_crop(crop)
        parsed = parse_tooltip_text(text)
        if parsed is None:
            failures += 1
            continue
        t, pct = parsed
        raw_points.append(RetentionPoint(t=t, pct=pct, source="ocr", confidence=1.0))

    if failures:
        print(f"[retention_extract] {failures}/{len(frame_paths)} frames had no parseable tooltip "
              f"(finger mid-swipe, no tooltip visible, or crop box is off).", file=sys.stderr)

    raw_points.sort(key=lambda p: p.t)

    # Collapse near-duplicate readings (finger paused on the same point for
    # several frames) -- keep the first clean reading per cluster.
    deduped: list[RetentionPoint] = []
    for p in raw_points:
        if deduped and abs(p.t - deduped[-1].t) <= dedupe_eps_s:
            continue
        deduped.append(p)

    duration = autodetect_reel_duration(frame_paths)
    if duration is not None:
        print(f"[retention_extract] auto-detected reel duration: {duration:.0f}s", file=sys.stderr)
    before = len(deduped)
    cleaned = clean_curve(deduped, max_t=duration)
    if len(cleaned) != before:
        print(f"[retention_extract] dropped {before - len(cleaned)} point(s) as OCR outliers "
              f"(time beyond reel duration, or a lone value spike neighbors disagree with)", file=sys.stderr)

    return RetentionCurve(points=cleaned, video_duration_s=duration)


def resolve_crop_box(img: np.ndarray, explicit: CropBox | None) -> CropBox:
    """explicit --crop wins; otherwise try to autodetect from the '100%'
    gridline label; otherwise fall back to the generic DEFAULT_CROP."""
    if explicit is not None:
        return explicit
    detected = autodetect_crop(img)
    if detected is not None:
        print(f"[retention_extract] auto-detected crop band: y1={detected.y1:.3f} "
              f"(from '100%' gridline)", file=sys.stderr)
        return detected
    print("[retention_extract] could not auto-detect the '100%' gridline label -- "
          "falling back to DEFAULT_CROP. Consider passing --crop explicitly.", file=sys.stderr)
    return DEFAULT_CROP


def run_calibration(image_path: Path, crop_box: CropBox | None, out_path: Path) -> None:
    """Draw the crop box on a sample frame and OCR it, so you can check the
    box is actually over the tooltip before trusting a full extraction run."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    crop_box = resolve_crop_box(img, crop_box)
    h, w = img.shape[:2]
    x0, x1 = int(crop_box.x0 * w), int(crop_box.x1 * w)
    y0, y1 = int(crop_box.y0 * h), int(crop_box.y1 * h)
    annotated = img.copy()
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), annotated)

    crop = crop_box.crop(img)
    text = ocr_crop(crop)
    parsed = parse_tooltip_text(text)
    print(f"Annotated frame saved to: {out_path}")
    print(f"Raw OCR text: {text!r}")
    print(f"Parsed (t_seconds, pct): {parsed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", type=Path, help="Screen-recording of the drag gesture")
    src.add_argument("--screenshots-dir", type=Path, help="Directory of discrete tap screenshots")
    ap.add_argument("--frames-out", type=Path, default=Path("data/runs/_frames"),
                     help="Where to dump extracted frames when --video is used")
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--crop", type=str, default=None,
                     help="x0,y0,x1,y1 as fractions of frame size, e.g. 0.15,0.10,0.85,0.30")
    ap.add_argument("--calibrate", type=Path, default=None,
                     help="Path to ONE sample frame/screenshot -- draws the crop box, OCRs it, "
                          "and exits without processing the whole video.")
    ap.add_argument("--out", type=Path, default=Path("data/runs/retention.json"))
    args = ap.parse_args()

    crop_box = CropBox.parse(args.crop) if args.crop else None  # None => autodetect per-source

    if args.calibrate:
        run_calibration(args.calibrate, crop_box, args.calibrate.with_suffix(".calibrated.png"))
        return

    if not args.video and not args.screenshots_dir:
        ap.error("--video or --screenshots-dir is required (unless using --calibrate)")

    if args.video:
        frames = extract_frames(args.video, args.frames_out, fps=args.fps)
    else:
        frames = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg") for p in args.screenshots_dir.glob(ext)
        )

    if not frames:
        raise SystemExit("No frames found to process.")

    curve = extract_curve_from_frames(frames, crop_box)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(curve.model_dump(), indent=2))
    print(f"Extracted {len(curve.points)} retention points -> {args.out}")


if __name__ == "__main__":
    main()
