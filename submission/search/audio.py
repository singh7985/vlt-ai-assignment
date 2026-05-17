"""
Optional audio modality — Whisper speech-to-text via ``faster_whisper``.

We transcribe the video's audio track into timestamped segments. At query
time we do a lightweight BM25-ish lexical scoring of the query against
each segment and convert the best-matching segments into search results.

This lets users find moments by what people *say*, which CLIP cannot do
(e.g. "someone introduces themselves", "the speaker says thank you").

All failures degrade silently — the rest of the pipeline still works.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass
class AudioSegment:
    start_ms: int
    end_ms: int
    text: str
    tokens: List[str]


class AudioTranscriber:
    """Thin wrapper around faster-whisper."""

    def __init__(self, model_size: str = "tiny", compute_type: str = "int8") -> None:
        self.model_size = model_size
        self.compute_type = compute_type
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore
        # CPU int8 keeps memory <100 MB for tiny.
        self._model = WhisperModel(self.model_size, device="cpu",
                                   compute_type=self.compute_type)

    def transcribe(self, video_path: str) -> List[AudioSegment]:
        """Extract audio (via ffmpeg) and transcribe. Returns [] on failure."""
        try:
            self._ensure_model()
        except Exception:
            return []

        wav_path: Optional[str] = None
        try:
            wav_path = _extract_wav(video_path)
            if wav_path is None:
                return []
            segments, _ = self._model.transcribe(  # type: ignore[union-attr]
                wav_path, vad_filter=True, beam_size=1,
            )
            out: List[AudioSegment] = []
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                out.append(AudioSegment(
                    start_ms=int(seg.start * 1000),
                    end_ms=int(seg.end * 1000),
                    text=text,
                    tokens=_tokenize(text),
                ))
            return out
        except Exception:
            return []
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass


def _extract_wav(video_path: str) -> Optional[str]:
    """Use ffmpeg to extract a 16 kHz mono WAV into a temp file."""
    out_fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="vs_audio_")
    os.close(out_fd)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=300)
        if os.path.getsize(out_path) == 0:
            return None
        return out_path
    except Exception:
        try:
            os.unlink(out_path)
        except OSError:
            pass
        return None


# --------------------------------------------------------------------------- #
# Lightweight BM25 over transcript segments
# --------------------------------------------------------------------------- #
class TranscriptIndex:
    """Tiny BM25-like scorer specialized for short segment lists."""

    def __init__(self, segments: List[AudioSegment],
                 k1: float = 1.2, b: float = 0.75) -> None:
        self.segments = segments
        self.k1 = k1
        self.b = b
        self.N = len(segments)
        self.avgdl = (sum(len(s.tokens) for s in segments) / self.N) if self.N else 0
        self.df: dict = {}
        for seg in segments:
            for term in set(seg.tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query: str) -> List[float]:
        if not self.segments:
            return []
        q_terms = _tokenize(query)
        scores = [0.0] * self.N
        for term in q_terms:
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, seg in enumerate(self.segments):
                tf = seg.tokens.count(term)
                if tf == 0:
                    continue
                dl = len(seg.tokens) or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores
