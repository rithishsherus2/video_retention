"""Redundant-dialogue detection: flag when a speaker says essentially the
same thing twice.

Not built on this reel's data yet (it has zero spoken dialogue, confirmed
in app.features.audio) -- built and validated against synthetic
transcripts instead, ready for the first real video with speech. It has
to work on paraphrases, not just exact repeats, so this is semantic
similarity over sentence embeddings, not string matching.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.features.audio import TranscriptSegment

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: small (~80MB), fast on CPU, standard choice for
        # sentence-level semantic similarity -- not the same tool as CLIP
        # (which is image-text aligned, not general sentence semantics),
        # so this is a separate small model rather than reusing CLIP.
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


@dataclass
class RedundancyFlag:
    segment_index: int
    text: str
    start: float
    end: float
    repeats_segment_index: int
    repeats_text: str
    repeats_start: float
    similarity: float


def detect_redundant_dialogue(
    segments: list[TranscriptSegment], similarity_threshold: float = 0.82, min_words: int = 3,
) -> list[RedundancyFlag]:
    """For each segment, find its most similar OTHER segment; flag it if
    similarity clears the threshold. min_words filters out trivially short
    utterances ("okay", "yeah") where high cosine similarity is meaningless
    noise rather than a real repeated statement."""
    candidates = [s for s in segments if len(s.text.split()) >= min_words]
    if len(candidates) < 2:
        return []

    model = _get_embedder()
    embeddings = model.encode([s.text for s in candidates], normalize_embeddings=True)
    sim = embeddings @ embeddings.T

    flags: list[RedundancyFlag] = []
    flagged_pairs: set[tuple[int, int]] = set()
    for i in range(len(candidates)):
        best_j, best_score = None, -1.0
        for j in range(len(candidates)):
            if j == i:
                continue
            if sim[i, j] > best_score:
                best_j, best_score = j, float(sim[i, j])
        if best_j is not None and best_score >= similarity_threshold:
            pair = (min(i, best_j), max(i, best_j))
            if pair in flagged_pairs:
                continue  # already recorded from the other direction
            flagged_pairs.add(pair)
            # Report against whichever of the pair comes SECOND -- that's
            # the one that's actually redundant (repeats something already said).
            first, second = (i, best_j) if candidates[i].start < candidates[best_j].start else (best_j, i)
            flags.append(RedundancyFlag(
                segment_index=second, text=candidates[second].text,
                start=candidates[second].start, end=candidates[second].end,
                repeats_segment_index=first, repeats_text=candidates[first].text,
                repeats_start=candidates[first].start, similarity=best_score,
            ))
    return sorted(flags, key=lambda f: f.start)
