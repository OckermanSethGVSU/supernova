"""
open_clip backend: dense text and image embeddings from OpenCLIP checkpoints.

Covers the CLIP / SigLIP / EVA / MetaCLIP families published through
open_clip, including the many LAION and DataComp weights that were never
ported to sentence-transformers. One class serves both modalities: page the
image column through it with `modality: image` and the caption (or query)
column with `modality: text`; two entries with identical backend settings
(batch_size included) share one loaded model.

A checkpoint is named by TWO fields, because the same architecture with
different weights is a different embedding space:

* ``model``      — the architecture, e.g. ``ViT-B-32``, or a Hub id
                   ``hf-hub:timm/ViT-B-16-SigLIP`` (weights included).
* ``pretrained`` — the weight tag, e.g. ``laion2b_s34b_b79k``. Leave unset
                   for ``hf-hub:`` models.

Example config entries:
    embedders:
      - name: clip_image
        kind: dense
        type: open_clip
        model: ViT-B-32
        pretrained: laion2b_s34b_b79k
        input_column: image
        modality: image
      - name: clip_caption
        kind: dense
        type: open_clip
        model: ViT-B-32
        pretrained: laion2b_s34b_b79k
        input_column: caption
        modality: text
"""

import asyncio
import logging
import threading

from typing import Any

import open_clip
import torch
import torch.nn.functional as F

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.media import Modality
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)

# config dtype -> open_clip `precision`
_PRECISION = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


@EMBEDDERS.register("open_clip")
class OpenCLIPDenseEmbedder(Embedder):
    output_kind = OutputKind.DENSE
    supported_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(
        self,
        model: str = "ViT-B-32",
        pretrained: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
        # None = float16 on CUDA, float32 elsewhere (half precision on CPU is
        # slow and poorly supported)
        dtype: str | None = None,
        # L2-normalize every vector; CLIP spaces are compared by cosine
        normalize: bool = True,
        cache_dir: str | None = None,
    ):
        self._device = device or detect_device()
        if dtype is None:
            dtype = "float16" if self._device.startswith("cuda") else "float32"
        if dtype not in _PRECISION:
            raise ValueError(
                f"open_clip dtype {dtype!r} not supported; choose from {sorted(_PRECISION)}"
            )
        logger.info(
            "Loading open_clip %s (pretrained=%s) on %s (dtype=%s)",
            model, pretrained or "none", self._device, dtype,
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model,
            pretrained=pretrained,
            device=self._device,
            precision=_PRECISION[dtype],
            cache_dir=cache_dir,
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model)
        # Half precision casts only some weights (positional embeddings stay
        # float32), so the input dtype must come from open_clip, not from a
        # parameter. None means float32 input.
        self._input_dtype = open_clip.get_input_dtype(_PRECISION[dtype])
        self._model_name = model if pretrained is None else f"{model}/{pretrained}"
        self._batch_size = batch_size
        self._normalize = normalize
        self._encode_lock = threading.Lock()
        # Not every architecture exposes its output width as an attribute, so
        # measure it with one empty caption.
        with torch.inference_mode():
            probe = self._model.encode_text(self._tokenizer([""]).to(self._device))
        self._dimensions_val = int(probe.shape[-1])
        self._max_tokens = getattr(self._tokenizer, "context_length", None)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    def _encode_slice(self, items: list[Any]) -> torch.Tensor:
        if all(isinstance(x, str) for x in items):
            tokens = self._tokenizer(items).to(self._device)
            return self._model.encode_text(tokens)
        if any(isinstance(x, str) for x in items):
            raise TypeError("open_clip batch mixes text and images")
        pixels = torch.stack([self._preprocess(img) for img in items])
        return self._model.encode_image(
            pixels.to(self._device, dtype=self._input_dtype)
        )

    def _encode(self, batch: list[Any]) -> list[Any]:
        out: list[Any] = []
        with self._encode_lock, torch.inference_mode():
            for start in range(0, len(batch), self._batch_size):
                feats = self._encode_slice(batch[start : start + self._batch_size])
                feats = feats.float()
                if self._normalize:
                    feats = F.normalize(feats, dim=-1)
                # float32 ndarray rows, kept as-is (see the ST backend note)
                out.extend(feats.cpu().numpy())
        return out

    async def embed(self, batch: list[Any]) -> list[Any]:
        return await asyncio.to_thread(self._encode, batch)
