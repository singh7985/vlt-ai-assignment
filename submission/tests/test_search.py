"""Unit tests for fusion, dedup, calibration, BM25 and sampler logic.

These tests intentionally avoid loading CLIP / Whisper / OCR so the suite
runs in <1 s on any laptop. End-to-end behaviour is covered by the
``--check-interface`` script in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``import search.*`` resolve when pytest is run from anywhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from search.audio import AudioSegment, TranscriptIndex, _tokenize  # noqa: E402
from search.fusion import (  # noqa: E402
    dedupe_by_time, normalize_bm25, rrf_merge, softmax_calibrate,
)
from search.sampler import VideoMeta, select_indices  # noqa: E402


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def test_rrf_combines_rankings():
    a = ([1000, 2000, 3000], 1.0)
    b = ([3000, 1000, 4000], 0.5)
    fused = rrf_merge([a, b])
    # 1000 appears at rank 0 in A and rank 1 in B → should beat 3000
    assert fused[1000] > fused[3000]
    assert 4000 in fused  # B-only timestamp still surfaces


def test_softmax_calibrate_orders_preserved():
    scores = np.array([0.1, 0.4, 0.25, 0.35])
    out = softmax_calibrate(scores)
    assert out.max() == pytest.approx(1.0, abs=1e-6) or out.max() > 0.9
    # Argmax preserved
    assert int(np.argmax(out)) == int(np.argmax(scores))
    # All in [0,1]
    assert ((out >= 0) & (out <= 1)).all()


def test_softmax_calibrate_handles_empty():
    out = softmax_calibrate(np.array([], dtype=np.float32))
    assert out.size == 0


def test_dedupe_by_time_keeps_peak_per_window():
    items = [
        (1000, 0.9, 30),
        (1200, 0.8, 36),   # within 1.5 s of 1000 → drop
        (5000, 0.85, 150),
        (5200, 0.7, 156),  # within 1.5 s of 5000 → drop
        (9000, 0.6, 270),
    ]
    kept = dedupe_by_time(items, min_gap_ms=1500)
    assert [k[0] for k in kept] == [1000, 5000, 9000]


def test_normalize_bm25_handles_zeros():
    assert (normalize_bm25([0, 0, 0]) == 0).all()
    out = normalize_bm25([1.0, 2.0, 4.0])
    assert out[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# BM25 transcript index
# --------------------------------------------------------------------------- #
def test_bm25_scores_matching_segments_higher():
    segs = [
        AudioSegment(0, 1000, "hello world",
                     tokens=_tokenize("hello world")),
        AudioSegment(2000, 3000, "the quick brown fox",
                     tokens=_tokenize("the quick brown fox")),
        AudioSegment(4000, 5000, "another sentence",
                     tokens=_tokenize("another sentence")),
    ]
    idx = TranscriptIndex(segs)
    scores = idx.score("quick fox")
    assert np.argmax(scores) == 1
    assert scores[0] == 0 and scores[2] == 0


def test_bm25_empty_index_safe():
    idx = TranscriptIndex([])
    assert idx.score("anything") == []


# --------------------------------------------------------------------------- #
# Sampler index selection
# --------------------------------------------------------------------------- #
def test_select_indices_uniform_plus_scenes_capped():
    meta = VideoMeta(fps=30.0, total_frames=18000,  # 10 minutes
                     width=640, height=360, duration_ms=600000)
    scenes = [(0, 100), (5000, 6000), (15000, 17999)]
    idx = select_indices(meta, scenes, stride_seconds=1.0, max_frames=300)
    assert len(idx) <= 300
    assert idx == sorted(set(idx))
    assert idx[0] >= 0 and idx[-1] < meta.total_frames


def test_select_indices_short_video():
    meta = VideoMeta(fps=24.0, total_frames=240, width=320, height=240,
                     duration_ms=10000)
    idx = select_indices(meta, [], stride_seconds=1.0)
    # 10s at 1s stride => 10 frames
    assert len(idx) == 10
