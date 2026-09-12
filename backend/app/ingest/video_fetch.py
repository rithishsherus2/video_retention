"""Fetch a reel's video+audio from Instagram, authenticated as you.

Two auth modes:
  --cookies-from-browser chrome|edge|firefox   (default: reads live from
      your browser's cookie store, no file to manage -- close that browser
      first, Chrome/Edge lock the cookie DB while running)
  --cookies-file path/to/cookies.txt           (manual export fallback,
      e.g. via the "Get cookies.txt LOCALLY" extension)

Usage:
  python -m app.ingest.video_fetch <reel_url> --cookies-from-browser chrome
  python -m app.ingest.video_fetch <reel_url> --cookies-file data/samples/cookies.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yt_dlp


def fetch_reel(url: str, out_dir: Path, cookies_from_browser: str | None = None,
                cookies_file: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    elif cookies_file:
        ydl_opts["cookiefile"] = str(cookies_file)
    else:
        raise ValueError("Need either cookies_from_browser or cookies_file for a private/own-account fetch.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
        # merge_output_format forces mp4 even if prepare_filename guesses otherwise
        mp4_path = path.with_suffix(".mp4")
        return mp4_path if mp4_path.exists() else path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="Instagram reel URL")
    ap.add_argument("--cookies-from-browser", choices=["chrome", "edge", "firefox", "brave"], default=None)
    ap.add_argument("--cookies-file", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("data/samples"))
    args = ap.parse_args()

    if not args.cookies_from_browser and not args.cookies_file:
        args.cookies_from_browser = "chrome"  # sensible default

    path = fetch_reel(args.url, args.out_dir, args.cookies_from_browser, args.cookies_file)
    print(f"Downloaded -> {path}")


if __name__ == "__main__":
    main()
