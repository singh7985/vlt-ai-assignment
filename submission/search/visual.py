"""
Visual encoder — OpenAI CLIP via ``open_clip_torch``.

We use **ViT-B-32 / laion2b_s34b_b79k** by default: a great accuracy /
memory / speed sweet-spot (~150 MB, 512-d embeddings, runs well on CPU
and accelerates nicely on CUDA/MPS in FP16).

Key features:

* **Prompt ensembling** — the query is encoded with multiple templates
  ("a photo of {q}", "a video frame of {q}", ...) and the embeddings are
  averaged. This is a classic CLIP recall booster.
* **Hardware-aware** — auto-selects CUDA → MPS → CPU and uses FP16 on GPU.
* **Batched image encoding** — frames are processed in batches of 32.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from PIL import Image


_PROMPT_TEMPLATES: Sequence[str] = (
    "a photo of {}.",
    "a video frame showing {}.",
    "a scene of {}.",
    "an image containing {}.",
    "{}",
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CLIPVisualEncoder:
    """Wraps an open_clip model + tokenizer for video frames + queries."""

    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 batch_size: int = 32) -> None:
        import open_clip  # local import keeps eval-only paths light

        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = _select_device()
        self.use_fp16 = self.device.type in ("cuda", "mps")

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        model.eval()
        if self.use_fp16:
            try:
                model = model.half()
            except Exception:
                self.use_fp16 = False

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.embedding_dim: int = int(model.visual.output_dim)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Image side
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_images_bgr(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a list of OpenCV BGR frames into L2-normalized vectors."""
        if not frames_bgr:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        out_chunks: List[np.ndarray] = []
        for i in range(0, len(frames_bgr), self.batch_size):
            batch = frames_bgr[i:i + self.batch_size]
            tensors = []
            for bgr in batch:
                # BGR -> RGB -> PIL -> CLIP preprocess
                rgb = bgr[:, :, ::-1]
                pil = Image.fromarray(rgb)
                tensors.append(self.preprocess(pil))
            stack = torch.stack(tensors).to(self.device)
            if self.use_fp16:
                stack = stack.half()
            feats = self.model.encode_image(stack)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out_chunks.append(feats.float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Text side
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a natural-language query as a single L2-normalized vector,
        averaging over an ensemble of prompt templates."""
        prompts = [tpl.format(query) for tpl in _PROMPT_TEMPLATES]
        tokens = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean = feats.mean(dim=0, keepdim=True)
        mean = mean / mean.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return mean.float().cpu().numpy().reshape(-1)

    def info(self) -> dict:
        return {
            "name": "CLIP",
            "version": f"{self.model_name}/{self.pretrained}",
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "fp16": self.use_fp16,
        }
