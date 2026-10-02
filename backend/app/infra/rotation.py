"""Generic rotate-past-the-exhausted-one helper.

Used for two independent pools that both fail the same way (a rate limit
or quota wall that retrying the SAME credential won't fix, but the NEXT
one might): Gemini API keys (app.analysis.gemini_client) and Instagram
engagement-scraper endpoints (app.ingest.engagement). Kept here instead of
duplicated in both because the failure mode and the fix are identical --
only what's being rotated differs.
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class ExhaustedError(Exception):
    """Raise this from inside the function passed to try_each() to mean
    'this specific item is rate-limited/out of quota -- move on to the
    next one', as distinct from any other exception, which propagates
    immediately without rotating (a bug isn't fixed by trying it again
    with a different key)."""


def try_each(items: list[T], fn: Callable[[T], R], label: str = "item") -> R:
    """Calls fn(item) for each item in order, moving to the next only when
    fn raises ExhaustedError. Returns the first success. Raises once all
    items are exhausted (or if the list is empty)."""
    if not items:
        raise RuntimeError(f"No {label} configured.")

    last_err: Exception | None = None
    for i, item in enumerate(items):
        try:
            return fn(item)
        except ExhaustedError as e:
            last_err = e
            print(f"[rotation] {label} {i + 1}/{len(items)} exhausted/rate-limited ({e}); trying next...")
            continue
    raise RuntimeError(f"All {len(items)} {label}(s) exhausted or rate-limited.") from last_err
