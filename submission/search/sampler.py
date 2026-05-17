"""
Scene-aware adaptive frame sampling.

We combine two strategies and take their union:

1. **Uniform safety net** — one frame every ``stride_seconds`` (default 1 s).
   Guarantees we never miss long static scenes.
2. **Shot-boundary refinement** — PySceneDetect's ContentDetector finds
   cuts; we sample the *middle* frame of every shot. This catches very
   short shots that uniform sampling would skip and lets us run with a
   wide uniform stride for big throughput gains.

If PySceneDetect is unavailable we silently fall back to uniform sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SampledFrame:
    frame_number: int          # 0-based index into the source video
    timestamp_ms: int          # decoded from frame_number / fps
    image_bgr: np.ndarray      # OpenCV BGR frame


@dataclass
class VideoMeta:
    fps: float
    total_frames: int
    width: int
    height: int
    duration_ms: int


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def probe_video(video_path: str) -> VideoMeta:
    """Return basic video metadata without decoding any frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_ms = int(total * 1000 / fps) if fps > 0 else 0
    return VideoMeta(fps=fps, total_frames=total, width=w, height=h,
                     duration_ms=duration_ms)


def detect_scene_indices(video_path: str,
                         threshold: float = 27.0) -> List[Tuple[int, int]]:
    """Return list of (start_frame, end_frame) tuples for each shot.

    Falls back to a single segment covering the whole video on any error.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        sm = SceneManager()
        sm.add_detector(ContentDetector(threshold=threshold))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list()
        if not scenes:
            return []
        return [(s[0].get_frames(), s[1].get_frames() - 1) for s in scenes]
    except Exception:
        return []


def select_indices(meta: VideoMeta,
                   scenes: List[Tuple[int, int]],
                   stride_seconds: float = 1.0,
                   max_frames: int = 1200) -> List[int]:
    """Compute the final sorted, deduped frame indices to embed.

    * Uniform every ``stride_seconds`` seconds.
    * Plus middle frame of every detected scene.
    * Capped at ``max_frames`` (evenly thinned) so a 10-min video never
      blows up the embedding budget.
    """
    if meta.total_frames <= 0:
        return []

    stride = max(1, int(round(meta.fps * stride_seconds)))
    uniform = list(range(0, meta.total_frames, stride))

    scene_mids = [
        (s + e) // 2 for s, e in scenes
        if 0 <= (s + e) // 2 < meta.total_frames
    ]

    indices = sorted(set(uniform + scene_mids))

    if len(indices) > max_frames:
        # Even thinning preserves coverage across the timeline
        idx_arr = np.linspace(0, len(indices) - 1, max_frames).astype(int)
        indices = [indices[i] for i in idx_arr]

    return indices


def decode_frames(video_path: str,
                  indices: List[int]) -> List[SampledFrame]:
    """Decode only the requested frame indices.

    Uses sequential read + skip rather than CAP_PROP_POS_FRAMES seeking,
    which is unreliable on many codecs. This is fast because we don't have
    to decode the full pixel data of skipped frames if we use ``grab()``.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: List[SampledFrame] = []
    wanted = sorted(set(indices))
    if not wanted:
        cap.release()
        return out

    wanted_iter = iter(wanted)
    next_target: Optional[int] = next(wanted_iter, None)
    current = 0
    while next_target is not None:
        # Cheap skip until just before the target
        while current < next_target:
            if not cap.grab():
                next_target = None
                break
            current += 1
        if next_target is None:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        out.append(SampledFrame(
            frame_number=current,
            timestamp_ms=int(current * 1000 / fps),
            image_bgr=frame,
        ))
        current += 1
        next_target = next(wanted_iter, None)

    cap.release()
    return out
