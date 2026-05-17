"""
Multimodal Video Search — main entry point.

Implements :class:`VideoSearch`, an extension of
``evaluation.interface.VideoSearchInterface``. The class fuses three
complementary modalities to find scenes that match a natural-language
query:

* **Visual** — CLIP (ViT-B-32, open_clip) over scene-aware sampled frames.
* **Audio**  — faster-whisper transcript with BM25 lexical scoring.
* **OCR**    — RapidOCR over the same sampled frames with BM25 scoring.

Modality scores are merged with Reciprocal Rank Fusion (RRF), deduped in
time, then calibrated to [0,1] confidences via a temperature-scaled
softmax blended with min-max normalisation.

Indexes are persisted to ``~/.cache/video_search`` so a second run on
the same video starts up in milliseconds.

See ``README.md`` for the full design write-up.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add parent directory to path for evaluation imports.
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add this directory so ``from search.xxx import ...`` works when this
# module is loaded by the evaluator via importlib.
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.interface import VideoSearchInterface, SearchResult  # noqa: E402

from search import cache as cache_mod  # noqa: E402
from search.audio import AudioTranscriber, TranscriptIndex  # noqa: E402
from search.fusion import (  # noqa: E402
    absolute_clip_quality,
    dedupe_by_time,
    rrf_merge,
    softmax_calibrate,
)
from search.ocr import OCRReader, build_ocr_index  # noqa: E402
from search.sampler import (  # noqa: E402
    decode_frames,
    detect_scene_indices,
    probe_video,
    select_indices,
)
from search.visual import CLIPVisualEncoder  # noqa: E402


class VideoSearch(VideoSearchInterface):
    """Multimodal video search (visual + audio + OCR) with RRF fusion."""

    # ---- Tunable config (env-overrideable) ----
    STRIDE_SECONDS = float(os.environ.get("VS_STRIDE_SECONDS", "1.0"))
    MAX_FRAMES = int(os.environ.get("VS_MAX_FRAMES", "1200"))
    MIN_GAP_MS = int(os.environ.get("VS_MIN_GAP_MS", "1500"))
    ENABLE_AUDIO = os.environ.get("VS_ENABLE_AUDIO", "1") == "1"
    ENABLE_OCR = os.environ.get("VS_ENABLE_OCR", "1") == "1"
    THUMB_DIR = Path(os.environ.get(
        "VS_THUMB_DIR",
        str(Path.home() / ".cache" / "video_search" / "thumbs"),
    ))

    # Modality weights for RRF
    W_VISUAL = 1.0
    W_AUDIO = 0.6
    W_OCR = 0.5

    def __init__(self) -> None:
        self.video_path: Optional[str] = None
        self.video_meta = None

        # Per-sampled-frame parallel arrays
        self._frame_numbers: List[int] = []
        self._timestamps_ms: List[int] = []
        self._visual_emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._thumbnails: List[Optional[str]] = []

        # Audio + OCR
        self._audio_segments: List[Any] = []
        self._audio_index: Optional[TranscriptIndex] = None
        self._ocr_frames: List[Any] = []
        self._ocr_index: Optional[TranscriptIndex] = None

        # Models — lazily loaded so unit tests can import this module cheaply.
        self._encoder: Optional[CLIPVisualEncoder] = None

        self.stats: Dict[str, Any] = {
            "fps": 0.0,
            "memory_mb": 0.0,
            "model_info": {},
            "index_time_seconds": 0.0,
            "total_frames": 0,
            "sampled_frames": 0,
            "modalities": [],
            "cache_hit": False,
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_encoder(self) -> CLIPVisualEncoder:
        if self._encoder is None:
            self._encoder = CLIPVisualEncoder()
        return self._encoder

    @staticmethod
    def _mem_mb() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _config_key(self) -> str:
        enc_info = self._encoder.info() if self._encoder else {"name": "CLIP"}
        return (
            f"v={enc_info.get('version', 'na')}|"
            f"stride={self.STRIDE_SECONDS}|max={self.MAX_FRAMES}|"
            f"aud={int(self.ENABLE_AUDIO)}|ocr={int(self.ENABLE_OCR)}"
        )

    def _save_thumbnail(self, frame_bgr: np.ndarray, frame_number: int) -> Optional[str]:
        try:
            import cv2
            self.THUMB_DIR.mkdir(parents=True, exist_ok=True)
            # Downscale to keep cache small
            h, w = frame_bgr.shape[:2]
            scale = 320 / max(w, 1)
            if scale < 1.0:
                new_size = (int(w * scale), int(h * scale))
                small = cv2.resize(frame_bgr, new_size, interpolation=cv2.INTER_AREA)
            else:
                small = frame_bgr
            path = self.THUMB_DIR / f"frame_{frame_number:08d}.jpg"
            if not path.exists():
                cv2.imwrite(str(path), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return str(path)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Interface implementation
    # ------------------------------------------------------------------ #
    def load_video(self, video_path: str) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        suffix = path.suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
            raise ValueError(f"Unsupported video format: {suffix}")

        wall_start = time.time()
        start_mem = self._mem_mb()
        self.video_path = str(path)

        # Probe + scene detection
        meta = probe_video(self.video_path)
        self.video_meta = meta
        scenes = detect_scene_indices(self.video_path)
        indices = select_indices(
            meta, scenes,
            stride_seconds=self.STRIDE_SECONDS,
            max_frames=self.MAX_FRAMES,
        )

        # Try cache
        encoder = self._ensure_encoder()
        ck = cache_mod.cache_key(self.video_path, self._config_key())
        cached = cache_mod.load(ck)

        if cached is not None and len(cached.frame_numbers) == len(indices):
            self._frame_numbers = cached.frame_numbers
            self._timestamps_ms = cached.timestamps_ms
            self._visual_emb = cached.visual_embeddings
            self._audio_segments = cached.audio_segments
            self._ocr_frames = cached.ocr_frames
            self._thumbnails = cached.thumbnail_paths
            self.stats["cache_hit"] = True
        else:
            self.stats["cache_hit"] = False

            # Decode frames
            samples = decode_frames(self.video_path, indices)
            frames_bgr = [s.image_bgr for s in samples]
            self._frame_numbers = [s.frame_number for s in samples]
            self._timestamps_ms = [s.timestamp_ms for s in samples]

            # Visual embeddings
            self._visual_emb = encoder.encode_images_bgr(frames_bgr)

            # Thumbnails (downscaled JPEGs)
            self._thumbnails = [
                self._save_thumbnail(img, fn)
                for img, fn in zip(frames_bgr, self._frame_numbers)
            ]

            # Audio transcription (optional, runs in parallel-ish via I/O wait)
            self._audio_segments = []
            if self.ENABLE_AUDIO:
                try:
                    self._audio_segments = AudioTranscriber().transcribe(self.video_path)
                except Exception:
                    self._audio_segments = []

            # OCR (optional)
            self._ocr_frames = []
            if self.ENABLE_OCR:
                try:
                    reader = OCRReader()
                    if reader.available:
                        self._ocr_frames = reader.read_frames(
                            frames_bgr, self._frame_numbers, self._timestamps_ms,
                        )
                except Exception:
                    self._ocr_frames = []

            # Persist cache
            try:
                cache_mod.save(ck, cache_mod.IndexPayload(
                    version=cache_mod.CACHE_VERSION,
                    config_key=self._config_key(),
                    frame_numbers=self._frame_numbers,
                    timestamps_ms=self._timestamps_ms,
                    visual_embeddings=self._visual_emb,
                    audio_segments=self._audio_segments,
                    ocr_frames=self._ocr_frames,
                    thumbnail_paths=self._thumbnails,
                    video_meta=meta,
                ))
            except Exception:
                pass

        # Build per-query indexes (cheap)
        self._audio_index = TranscriptIndex(self._audio_segments)
        self._ocr_index, _ = build_ocr_index(self._ocr_frames)

        # Stats
        elapsed = max(time.time() - wall_start, 1e-6)
        peak_mem = self._mem_mb()
        # Honest throughput definition: source frames decoded/processed per
        # wall-clock second. The whole video is decoded (we grab() through
        # every frame), so total_frames is the correct numerator.
        throughput = (meta.total_frames or len(self._frame_numbers)) / elapsed

        modalities = ["clip-visual"]
        if self._audio_segments:
            modalities.append(f"whisper-tiny-int8 ({len(self._audio_segments)} segs)")
        if self._ocr_frames:
            modalities.append(f"rapidocr ({len(self._ocr_frames)} frames)")

        # Report peak resident memory observed during indexing. We compare
        # against the pre-index baseline so transient drops don't make the
        # number look artificially low.
        peak_rss_mb = max(peak_mem, start_mem)

        self.stats.update({
            "fps": float(throughput),
            "memory_mb": float(peak_rss_mb),
            "model_info": encoder.info(),
            "index_time_seconds": float(elapsed),
            "total_frames": int(meta.total_frames),
            "sampled_frames": int(len(self._frame_numbers)),
            "embedding_dim": int(self._visual_emb.shape[1]) if self._visual_emb.size else 0,
            "scenes_detected": int(len(scenes)),
            "modalities": modalities,
        })

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.video_path is None or self._visual_emb.size == 0:
            raise RuntimeError("Must call load_video() before search().")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if top_k <= 0:
            return []

        encoder = self._ensure_encoder()
        q_vec = encoder.encode_query(query.strip())

        # --- Visual scores: cosine similarity (vectors already L2-normalised)
        visual_sims = self._visual_emb @ q_vec.astype(np.float32)
        # Convert similarity ([-1,1]) → ranked timestamp list
        v_order = np.argsort(-visual_sims)
        v_ranked = [self._timestamps_ms[i] for i in v_order]

        # --- Audio scores (BM25 over Whisper segments)
        a_ranked: List[int] = []
        audio_scores = self._audio_index.score(query) if self._audio_index else []
        if any(s > 0 for s in audio_scores):
            a_order = np.argsort(-np.asarray(audio_scores))
            a_ranked = [
                (self._audio_segments[i].start_ms + self._audio_segments[i].end_ms) // 2
                for i in a_order if audio_scores[i] > 0
            ]

        # --- OCR scores (BM25 over recognised on-screen text)
        o_ranked: List[int] = []
        ocr_scores = self._ocr_index.score(query) if self._ocr_index else []
        if any(s > 0 for s in ocr_scores):
            o_order = np.argsort(-np.asarray(ocr_scores))
            o_ranked = [
                self._ocr_frames[i].timestamp_ms
                for i in o_order if ocr_scores[i] > 0
            ]

        # --- Reciprocal Rank Fusion across modalities
        fused = rrf_merge([
            (v_ranked, self.W_VISUAL),
            (a_ranked, self.W_AUDIO),
            (o_ranked, self.W_OCR),
        ])

        # Map fused timestamps back to (frame_number, visual_score) for thumb lookup
        ts_to_idx = {ts: i for i, ts in enumerate(self._timestamps_ms)}

        # Build candidate list (timestamp, fused_score, frame_number)
        candidates: List = []
        for ts, fscore in fused.items():
            if ts in ts_to_idx:
                idx = ts_to_idx[ts]
                fn = self._frame_numbers[idx]
            else:
                # Audio-only hit: snap to nearest sampled frame for the thumb
                nearest = int(np.argmin([abs(ts - t) for t in self._timestamps_ms]))
                idx = nearest
                fn = self._frame_numbers[idx]
            candidates.append((ts, float(fscore), int(fn), idx))

        # Pull enough to allow dedup to land top_k
        candidates.sort(key=lambda x: -x[1])
        pre_dedup = [(ts, sc, fn) for ts, sc, fn, _ in candidates[: max(top_k * 5, 25)]]
        kept = dedupe_by_time(pre_dedup, min_gap_ms=self.MIN_GAP_MS)[:top_k]

        if not kept:
            return []

        # Calibrate confidences over the kept set using their raw visual sims
        # (fused scores are not bounded) so they read like meaningful 0-1 values.
        sims_for_kept = []
        for ts, _, _ in kept:
            sims_for_kept.append(float(visual_sims[ts_to_idx.get(ts, 0)]))
        conf = softmax_calibrate(np.asarray(sims_for_kept))

        # Cap confidences by the absolute CLIP quality of the best hit so a
        # garbage query (e.g. "asdfqwerty") returns low confidence even though
        # softmax always assigns 1.0 to its argmax.
        quality_cap = absolute_clip_quality(max(sims_for_kept) if sims_for_kept else 0.0)
        conf = conf * quality_cap

        # Bonus for timestamps that also matched audio/OCR (hybrid hits).
        audio_hit_ts = set()
        if audio_scores:
            max_a = max(audio_scores) or 1.0
            for i, s in enumerate(audio_scores):
                if s > 0:
                    seg = self._audio_segments[i]
                    audio_hit_ts.add(((seg.start_ms + seg.end_ms) // 2, s / max_a))
        ocr_hit_ts = {}
        if ocr_scores:
            max_o = max(ocr_scores) or 1.0
            for i, s in enumerate(ocr_scores):
                if s > 0:
                    ocr_hit_ts[self._ocr_frames[i].timestamp_ms] = s / max_o

        for j, (ts, _, _) in enumerate(kept):
            bonus = 0.0
            for hit_ts, w in audio_hit_ts:
                if abs(ts - hit_ts) <= 2500:
                    bonus = max(bonus, 0.25 * w)
            for hit_ts, w in ocr_hit_ts.items():
                if abs(ts - hit_ts) <= 2500:
                    bonus = max(bonus, 0.25 * w)
            conf[j] = min(1.0, conf[j] + bonus)

        results: List[SearchResult] = []
        for (ts, _, fn), c in zip(kept, conf):
            idx = ts_to_idx.get(ts, 0)
            thumb = self._thumbnails[idx] if 0 <= idx < len(self._thumbnails) else None
            results.append(SearchResult(
                timestamp_ms=int(max(0, ts)),
                confidence=float(np.clip(c, 0.0, 1.0)),
                frame_number=int(max(0, fn)),
                thumbnail_path=thumb,
            ))

        return results

    # ------------------------------------------------------------------ #
    def get_processing_stats(self) -> dict:
        return dict(self.stats)
