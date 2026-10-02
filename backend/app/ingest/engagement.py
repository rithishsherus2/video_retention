"""Instagram engagement metrics (views/likes/comments) for a reel.

Two layers, cheapest-and-most-available first:

1. yt-dlp's own extract_info() metadata -- already fetched as a side
   effect of downloading the reel (see video_fetch.fetch_reel_with_info),
   so reading it costs no extra request. Instagram frequently omits or
   zeroes some of these fields for accounts you don't own, even when
   logged in via cookies -- that's a platform limitation, not a bug here.

2. Optional external engagement-scraper endpoints, configured in
   config/engagement_providers.json (gitignored -- copy
   config/engagement_providers.example.json and fill in real endpoints/
   keys), tried in order and rotated past on a rate-limit response, used
   to fill in whatever yt-dlp left null. Fully optional: with no config
   file present, engagement is yt-dlp-only and nothing fails because of
   it -- the whole feature degrades gracefully, it doesn't error out.

   Each provider's base_url/query_params/headers strings may reference
   {reel_url}, {api_key}, {shortcode} (the URL's /reel/<this>/ segment),
   and {media_id} (Instagram's internal numeric media ID, derived from
   the shortcode -- see shortcode_to_media_id) -- whichever one a given
   API actually expects as input.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from app.infra.rotation import ExhaustedError, try_each
from app.schemas import EngagementMetrics

CONFIG_PATH_ENV = "ENGAGEMENT_PROVIDERS_FILE"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "engagement_providers.json"

# Instagram's reel/post shortcode (the bit in the URL, e.g. "DJn77p9zwVB")
# is that account-independent numeric media ID base64-encoded with this
# fixed alphabet -- a long-stable, publicly documented encoding, not
# something Instagram's API exposes directly. Several scraper APIs (see
# config/engagement_providers.example.json) want that numeric ID rather
# than the URL/shortcode itself, so it's computed here once and offered
# to every provider config as the {media_id} template variable.
_SHORTCODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


def extract_shortcode(reel_url: str) -> str | None:
    m = re.search(r"/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)", reel_url)
    return m.group(1) if m else None


def shortcode_to_media_id(shortcode: str) -> int:
    media_id = 0
    for char in shortcode:
        media_id = media_id * 64 + _SHORTCODE_ALPHABET.index(char)
    return media_id


def metrics_from_ytdlp_info(info: dict) -> EngagementMetrics:
    description = info.get("description") or ""
    return EngagementMetrics(
        views=info.get("view_count"),
        likes=info.get("like_count"),
        comments=info.get("comment_count"),
        caption=description or None,
        hashtags=re.findall(r"#(\w+)", description),
        posted_at=info.get("upload_date"),
        source="yt_dlp",
    )


def _load_provider_configs() -> list[dict]:
    from dotenv import load_dotenv

    load_dotenv()  # picks up backend/.env if present, so api_key_env lookups work standalone
    path = Path(os.environ.get(CONFIG_PATH_ENV, str(DEFAULT_CONFIG_PATH)))
    if not path.exists():
        return []
    configs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(configs, list):
        raise RuntimeError(f"{path} must contain a JSON list of provider configs.")
    return configs


def _get_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                return None
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _query_provider(cfg: dict, reel_url: str) -> EngagementMetrics:
    """Raises ExhaustedError for ANY failure -- not just a rate-limit status
    code. Confirmed in practice: the same request that got a 400 once
    succeeded moments later with identical parameters, i.e. this is an
    unreliable third-party surface that fails transiently in ways that
    don't always show up as 429. Since engagement is always optional
    (fetch_engagement falls back to yt-dlp-only data), every failure mode
    here should just mean "try the next provider," never "crash the run" --
    unlike Gemini calls, where letting a real bug propagate loudly is
    the point."""
    name = cfg.get("name", "unnamed provider")
    try:
        api_key = ""
        key_env = cfg.get("api_key_env")
        if key_env:
            api_key = os.environ.get(key_env, "")
            if not api_key:
                raise ExhaustedError(f"'{name}': env var {key_env} is not set.")

        shortcode = extract_shortcode(reel_url) or ""
        media_id = str(shortcode_to_media_id(shortcode)) if shortcode else ""

        def fmt(v):
            return v.format(reel_url=reel_url, api_key=api_key, shortcode=shortcode, media_id=media_id) \
                if isinstance(v, str) else v

        params = {k: fmt(v) for k, v in cfg.get("query_params", {}).items()}
        headers = {k: fmt(v) for k, v in cfg.get("headers", {}).items()}
        method = cfg.get("method", "GET").upper()

        resp = requests.request(
            method, fmt(cfg["base_url"]),
            params=params if method == "GET" else None,
            json=params if method != "GET" else None,
            headers=headers, timeout=20,
        )

        rate_limit_codes = cfg.get("rate_limit_status_codes", [429])
        if resp.status_code in rate_limit_codes:
            raise ExhaustedError(f"'{name}' returned {resp.status_code} (rate-limited)")
        resp.raise_for_status()

        data = resp.json()
        paths = cfg.get("response_paths", {})
        return EngagementMetrics(
            views=_get_path(data, paths["views"]) if "views" in paths else None,
            likes=_get_path(data, paths["likes"]) if "likes" in paths else None,
            comments=_get_path(data, paths["comments"]) if "comments" in paths else None,
            caption=_get_path(data, paths["caption"]) if "caption" in paths else None,
            source=f"provider:{name}",
        )
    except ExhaustedError:
        raise
    except Exception as e:
        raise ExhaustedError(f"'{name}' failed: {type(e).__name__}: {e}") from e


def fetch_engagement(reel_url: str, ytdlp_info: dict | None = None) -> EngagementMetrics:
    """Best-effort merge: start from yt-dlp's metadata (free, already
    fetched), then fill any still-missing fields from the first configured
    provider that succeeds, rotating past ones that are rate-limited or
    mis-configured. Never raises for missing/unconfigured providers --
    worst case you get whatever yt-dlp already gave you."""
    base = metrics_from_ytdlp_info(ytdlp_info) if ytdlp_info else EngagementMetrics(source="none")
    if base.views is not None and base.likes is not None and base.comments is not None:
        return base

    try:
        configs = _load_provider_configs()
    except Exception as e:
        print(f"[engagement] couldn't load provider config, keeping yt-dlp-only data: {e}")
        return base
    if not configs:
        return base

    try:
        provider_result = try_each(configs, lambda cfg: _query_provider(cfg, reel_url), label="engagement provider")
    except Exception as e:
        # Broad on purpose: this is a best-effort supplementary data source,
        # and _query_provider already converts everything it can anticipate
        # into ExhaustedError -- this is the final backstop for whatever it
        # didn't, so a flaky/misbehaving provider degrades to yt-dlp-only
        # data instead of taking down the whole reel-analysis run.
        print(f"[engagement] all providers failed/unconfigured, keeping yt-dlp-only data: {e}")
        return base

    return EngagementMetrics(
        views=base.views if base.views is not None else provider_result.views,
        likes=base.likes if base.likes is not None else provider_result.likes,
        comments=base.comments if base.comments is not None else provider_result.comments,
        caption=base.caption or provider_result.caption,
        hashtags=base.hashtags,
        posted_at=base.posted_at,
        source=f"{base.source}+{provider_result.source}",
    )
