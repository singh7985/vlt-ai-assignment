"""
Persistent on-disk cache of indexed videos.

The cache key is a sha1 of (video file content prefix + size + mtime +
config string). Cache hits make second-run TTFR ~instant.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


CACHE_VERSION = 2
DEFAULT_CACHE_DIR = Path(
    os.environ.get("VIDEO_SEARCH_CACHE",
                   str(Path.home() / ".cache" / "video_search"))
)


@dataclass
class IndexPayload:
    version: int
    config_key: str
    frame_numbers: List[int]
    timestamps_ms: List[int]
    visual_embeddings: np.ndarray          # (N, D) float32, L2-normalized
    audio_segments: List[Any]              # search.audio.AudioSegment
    ocr_frames: List[Any]                  # search.ocr.OCRFrame
    thumbnail_paths: List[Optional[str]]   # parallel to frame_numbers
    video_meta: Any                        # search.sampler.VideoMeta


def _hash_file(path: str, max_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha1()
    st = os.stat(path)
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def cache_key(video_path: str, config: str) -> str:
    return f"{_hash_file(video_path)}__{hashlib.sha1(config.encode()).hexdigest()[:8]}"


def cache_path(key: str, base: Path = DEFAULT_CACHE_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"index_{key}.pkl"


def load(key: str) -> Optional[IndexPayload]:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, IndexPayload) and obj.version == CACHE_VERSION:
            return obj
    except Exception:
        return None
    return None


def save(key: str, payload: IndexPayload) -> None:
    p = cache_path(key)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)
