"""
Result fusion + post-processing utilities.

* :func:`rrf_merge` — Reciprocal Rank Fusion across modalities.
* :func:`softmax_calibrate` — turn raw scores into well-spread [0,1] confidences.
* :func:`dedupe_by_time` — collapse near-duplicate timestamps, keeping the peak.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def rrf_merge(ranked_lists: Sequence[Tuple[Sequence[int], float]],
              k: int = 60) -> Dict[int, float]:
    """Fuse multiple ranked lists of timestamps (ms) into a single ranking.

    ``ranked_lists`` is a sequence of ``(timestamps_ms_sorted_best_first, weight)``.
    Returns ``{timestamp_ms: rrf_score}``.
    """
    fused: Dict[int, float] = {}
    for ranked, w in ranked_lists:
        for rank, ts in enumerate(ranked):
            fused[ts] = fused.get(ts, 0.0) + w / (k + rank + 1)
    return fused


# --------------------------------------------------------------------------- #
# Confidence calibration
# --------------------------------------------------------------------------- #
def softmax_calibrate(scores: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """Map arbitrary scores onto [0,1] in a well-spread way.

    Uses a temperature-scaled softmax then renormalises so the top entry is
    1.0. This produces nicer-looking confidences than raw CLIP cosine
    (which usually lives in ~[0.15, 0.35]).
    """
    if scores.size == 0:
        return scores
    s = scores - scores.max()
    e = np.exp(s / max(temperature, 1e-6))
    p = e / e.sum()
    if p.max() > 0:
        p = p / p.max()
    # Blend with a min-max rescale so confidences are not *too* peaky.
    lo, hi = float(scores.min()), float(scores.max())
    mm = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
    blended = 0.5 * p + 0.5 * mm
    return np.clip(blended, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Temporal dedup
# --------------------------------------------------------------------------- #
def dedupe_by_time(items: List[Tuple[int, float, int]],
                   min_gap_ms: int = 1500) -> List[Tuple[int, float, int]]:
    """Collapse results within ``min_gap_ms`` of each other.

    ``items`` is ``[(timestamp_ms, score, frame_number), ...]`` sorted by score
    descending. Returns a similarly-sorted list with neighbours suppressed.
    """
    kept: List[Tuple[int, float, int]] = []
    for ts, score, fn in items:
        if all(abs(ts - kt) >= min_gap_ms for kt, _, _ in kept):
            kept.append((ts, score, fn))
    return kept


def normalize_bm25(scores: Iterable[float]) -> np.ndarray:
    """0-1 normalisation that returns zeros for an all-zero input."""
    arr = np.asarray(list(scores), dtype=np.float32)
    if arr.size == 0:
        return arr
    mx = float(arr.max())
    if mx <= 0:
        return np.zeros_like(arr)
    return arr / mx
