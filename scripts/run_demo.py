"""Run the VideoSearch pipeline against the Big Buck Bunny trailer."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "submission"))

from main import VideoSearch  # noqa: E402

VIDEO = REPO / "data" / "demo.mp4"

QUERIES = [
    "a fluffy white rabbit",
    "a butterfly flying",
    "a green forest with trees",
    "two small animals together",
    "a close-up of an angry face",
    "the title text on screen",
    "a sunny outdoor scene",
    "a character looking surprised",
]


def main() -> int:
    if not VIDEO.exists():
        print(f"missing video: {VIDEO}")
        return 1

    vs = VideoSearch()
    print(f"[demo] loading {VIDEO.name} ({VIDEO.stat().st_size / 1e6:.1f} MB) ...")
    t0 = time.time()
    vs.load_video(str(VIDEO))
    load_s = time.time() - t0
    stats = vs.get_processing_stats()
    print(f"[demo] load_video took {load_s:.2f}s   cache_hit={stats['cache_hit']}")
    print(f"[demo] {stats['total_frames']} total frames, "
          f"{stats['sampled_frames']} sampled, "
          f"{stats['scenes_detected']} scenes detected, "
          f"reported FPS={stats['fps']:.1f}, "
          f"peak RAM={stats['memory_mb']:.0f} MB")
    print(f"[demo] modalities active: {stats['modalities']}")

    for q in QUERIES:
        t1 = time.time()
        results = vs.search(q, top_k=3)
        dt_ms = (time.time() - t1) * 1000
        print(f"\n[demo] query={q!r}   latency={dt_ms:.1f} ms")
        for i, r in enumerate(results, 1):
            secs = r.timestamp_ms / 1000
            print(f"   #{i}  t={secs:6.2f}s  conf={r.confidence:.3f}  "
                  f"frame={r.frame_number}  thumb={r.thumbnail_path or '-'}")

    print("\n[demo] done. Launch the UI with `python -m submission.app` and "
          f"upload {VIDEO.relative_to(REPO)} to explore interactively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
