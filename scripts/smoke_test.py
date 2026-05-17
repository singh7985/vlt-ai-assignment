"""End-to-end smoke test: generate a tiny synthetic video, index it, run a search.

Not part of the submission grading — purely a sanity check that the full
pipeline (sampling -> CLIP encode -> fusion -> ranked results) works.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission"))

from main import VideoSearch  # noqa: E402


def make_synthetic_video(path: Path, seconds: int = 8, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    total = seconds * fps
    # Three "scenes": red square, green circle, blue text
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        scene = i // (total // 3 if total >= 3 else 1)
        if scene == 0:
            cv2.rectangle(frame, (60, 60), (260, 180), (0, 0, 200), -1)  # red
        elif scene == 1:
            cv2.circle(frame, (160, 120), 70, (0, 200, 0), -1)  # green
        else:
            frame[:] = (200, 80, 30)  # blue-ish
            cv2.putText(frame, "HELLO", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        vw.write(frame)
    vw.release()


def main() -> int:
    video = REPO / "data" / "smoke.mp4"
    if not video.exists():
        print(f"[smoke] creating synthetic video at {video}")
        make_synthetic_video(video)

    vs = VideoSearch()
    print("[smoke] loading video ...")
    t0 = time.time()
    vs.load_video(str(video))
    print(f"[smoke] load_video took {time.time() - t0:.2f}s")

    stats = vs.get_processing_stats()
    print(f"[smoke] stats: {stats}")

    for q in ["a red square", "a green circle", "the word hello on screen"]:
        t1 = time.time()
        results = vs.search(q, top_k=3)
        dt_ms = (time.time() - t1) * 1000
        print(f"\n[smoke] query={q!r}  latency={dt_ms:.1f}ms")
        for r in results:
            print(f"   t={r.timestamp_ms:>6d}ms  conf={r.confidence:.3f}  frame={r.frame_number}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
