"""
Optional OCR modality — on-screen text via ``rapidocr_onnxruntime``.

Run once per sampled frame; the recognised strings are indexed with the
same tiny BM25 we use for audio. Enables queries like "scene with the
word PARIS" or "subtitle that says hello".

If the rapidocr package is missing or fails to initialise we return an
empty index and the rest of the system carries on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .audio import TranscriptIndex, _tokenize  # reuse BM25 + tokenizer
from .audio import AudioSegment as _Seg  # alias only — same shape


@dataclass
class OCRFrame:
    frame_number: int
    timestamp_ms: int
    text: str


class OCRReader:
    def __init__(self) -> None:
        self._engine = None
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
            self._engine = RapidOCR()
        except Exception:
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def read_frames(self,
                    frames_bgr: Sequence[np.ndarray],
                    frame_numbers: Sequence[int],
                    timestamps_ms: Sequence[int]) -> List[OCRFrame]:
        if not self.available or not frames_bgr:
            return []
        out: List[OCRFrame] = []
        for img, fn, ts in zip(frames_bgr, frame_numbers, timestamps_ms):
            try:
                result, _ = self._engine(img)  # type: ignore[misc]
            except Exception:
                continue
            if not result:
                continue
            text = " ".join(box[1] for box in result if box and len(box) > 1).strip()
            if text:
                out.append(OCRFrame(frame_number=int(fn),
                                    timestamp_ms=int(ts),
                                    text=text))
        return out


def build_ocr_index(ocr_frames: List[OCRFrame]
                    ) -> "tuple[TranscriptIndex, List[OCRFrame]]":
    """Wrap OCR frames in the BM25 index by reusing the AudioSegment shape."""
    segs = [
        _Seg(start_ms=f.timestamp_ms, end_ms=f.timestamp_ms,
             text=f.text, tokens=_tokenize(f.text))
        for f in ocr_frames
    ]
    return TranscriptIndex(segs), ocr_frames
