"""Shot-cut detection (PySceneDetect) + CLIP-based shot similarity, which is
how we flag your two priority rules on the editing side: a "redundant shot"
(a cut to near-identical content -- pointless cut) and a "reused shot" (the
same footage appearing again elsewhere in the video, possibly padding).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from scenedetect import ContentDetector, detect

_clip_model = None
_clip_preprocess = None


def _get_clip():
    """Lazy-load CLIP (ViT-B-32/openai) once per process -- first call
    downloads pretrained weights (~350MB) if not already cached."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import open_clip
        # "-quickgelu" matches the activation function the OpenAI weights were
        # actually trained with; the plain "ViT-B-32" config defaults to GELU
        # and silently produces degraded embeddings against these weights.
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
        model.eval()
        _clip_model, _clip_preprocess = model, preprocess
    return _clip_model, _clip_preprocess


@dataclass
class Shot:
    index: int
    start_t: float
    end_t: float

    @property
    def duration(self) -> float:
        return self.end_t - self.start_t

    @property
    def mid_t(self) -> float:
        return (self.start_t + self.end_t) / 2


def detect_shots(video_path: str, threshold: float = 27.0) -> list[Shot]:
    """Hard-cut boundaries via PySceneDetect's ContentDetector (HSV/edge
    content-change based -- catches hard cuts reliably; slow crossfades or
    whip-pans may not register as a cut, which is the correct behavior
    here since those aren't the "random cut" failure mode we're after)."""
    scene_list = detect(video_path, ContentDetector(threshold=threshold))
    if not scene_list:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return [Shot(index=0, start_t=0.0, end_t=(n_frames / fps if fps else 0.0))]
    return [
        Shot(index=i, start_t=start.get_seconds(), end_t=end.get_seconds())
        for i, (start, end) in enumerate(scene_list)
    ]


def _read_frame_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    return frame if ok else None


def compute_shot_embeddings(video_path: str, shots: list[Shot]) -> np.ndarray:
    """One CLIP embedding per shot, taken from its middle frame (more
    representative of shot content than the first frame, which can still
    show cut-transition artifacts)."""
    model, preprocess = _get_clip()
    cap = cv2.VideoCapture(video_path)
    imgs = []
    for shot in shots:
        frame = _read_frame_at(cap, shot.mid_t)
        if frame is None:
            frame = _read_frame_at(cap, shot.start_t)
        if frame is None:
            imgs.append(Image.new("RGB", (224, 224)))
            continue
        imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()

    batch = torch.stack([preprocess(im) for im in imgs])
    with torch.no_grad():
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.numpy()


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


@dataclass
class ShotAnalysis:
    shot: Shot
    prev_shot_similarity: float | None        # similarity to the immediately preceding shot
    most_similar_other_index: int | None       # most similar NON-adjacent shot elsewhere
    most_similar_other_score: float | None
    is_likely_redundant_cut: bool              # cut to near-identical content (pointless cut)
    is_likely_reused_shot: bool                # near-duplicate of an earlier/later, non-adjacent shot


def analyze_shots(
    video_path: str,
    redundant_cut_threshold: float = 0.92,
    reused_shot_threshold: float = 0.90,
) -> list[ShotAnalysis]:
    shots = detect_shots(video_path)
    embeddings = compute_shot_embeddings(video_path, shots)
    sim = cosine_similarity_matrix(embeddings)

    results: list[ShotAnalysis] = []
    for i, shot in enumerate(shots):
        prev_sim = float(sim[i, i - 1]) if i > 0 else None

        other_idxs = [j for j in range(len(shots)) if j not in (i, i - 1)]
        most_similar_idx = most_similar_score = None
        if other_idxs:
            scores = [(j, float(sim[i, j])) for j in other_idxs]
            most_similar_idx, most_similar_score = max(scores, key=lambda x: x[1])

        results.append(ShotAnalysis(
            shot=shot,
            prev_shot_similarity=prev_sim,
            most_similar_other_index=most_similar_idx,
            most_similar_other_score=most_similar_score,
            is_likely_redundant_cut=(prev_sim is not None and prev_sim >= redundant_cut_threshold),
            is_likely_reused_shot=(most_similar_score is not None and most_similar_score >= reused_shot_threshold),
        ))
    return results
