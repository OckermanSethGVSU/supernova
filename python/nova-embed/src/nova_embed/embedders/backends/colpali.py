"""
ColPali backend: multivector embeddings of document page images (colpali-engine).

ColPali-style models (ColPali, ColQwen2, ColQwen2.5, ColSmol/ColIdefics3)
embed a rendered page image directly into one vector per image patch, with no
OCR step, and embed text queries into one vector per token in the same space.
Retrieval scores a query against a page with MaxSim, like ColBERT.

One class serves both sides: an entry with `modality: image` on the page
column and one with `modality: text` on the query column. Entries with
identical backend settings (batch_size included) share one loaded model. The side is chosen from the
input itself (images go through the processor's image path, strings through
its query path), so there is no `query` flag here.

The model family is detected from the model id; set `variant:` explicitly for
a checkpoint whose name doesn't say (see `_VARIANTS` for the names).

Page embeddings are large (on the order of a thousand vectors per page), so
lower `pipeline.flush_threshold` well below the text defaults.

Example config entries:
    embedders:
      - name: page
        kind: multivector
        type: colpali
        model: vidore/colqwen2.5-v0.2
        input_column: image
        modality: image
      - name: query
        kind: multivector
        type: colpali
        model: vidore/colqwen2.5-v0.2
        input_column: query
        modality: text
"""

import asyncio
import logging
import threading

from typing import Any

import numpy as np
import torch

import colpali_engine.models as colpali_models

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.media import Modality
from nova_embed.models import MultiVectorEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)

# variant -> (model class, processor class) in colpali_engine.models. Looked
# up by name at load time so a colpali-engine release without one of them
# only fails for that variant.
_VARIANTS = {
    "colpali": ("ColPali", "ColPaliProcessor"),
    "colqwen2": ("ColQwen2", "ColQwen2Processor"),
    "colqwen2_5": ("ColQwen2_5", "ColQwen2_5_Processor"),
    "colidefics3": ("ColIdefics3", "ColIdefics3Processor"),
}

# model-id substrings -> variant, checked in order (longest names first, so
# "colqwen2.5" is not taken for "colqwen2")
_NAME_HINTS = [
    ("colqwen2.5", "colqwen2_5"),
    ("colqwen2_5", "colqwen2_5"),
    ("colqwen2", "colqwen2"),
    ("colsmol", "colidefics3"),
    ("colidefics3", "colidefics3"),
    ("colpali", "colpali"),
]

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def detect_variant(model: str) -> str:
    name = model.lower()
    for hint, variant in _NAME_HINTS:
        if hint in name:
            return variant
    raise ValueError(
        f"can't tell the ColPali model family from {model!r}; set `variant:` "
        f"to one of {sorted(_VARIANTS)}"
    )


def _load_classes(variant: str):
    if variant not in _VARIANTS:
        raise ValueError(
            f"unknown ColPali variant {variant!r}; choose from {sorted(_VARIANTS)}"
        )
    model_name, proc_name = _VARIANTS[variant]
    try:
        return getattr(colpali_models, model_name), getattr(colpali_models, proc_name)
    except AttributeError as e:
        raise ValueError(
            f"the installed colpali-engine has no {model_name}/{proc_name} "
            f"(variant {variant!r}); upgrade colpali-engine"
        ) from e


def _trim(out: torch.Tensor, mask: torch.Tensor) -> list[np.ndarray]:
    """One (n_real_tokens, dim) float32 array per row of a padded batch.

    The model returns every row padded to the longest in the batch. Padded
    positions must be dropped, not kept as zero vectors: a zero vector scores
    0 against any query token, which can outrank real negative similarities
    in MaxSim.
    """
    mask = mask.bool()
    return [out[i][mask[i]].float().cpu().numpy() for i in range(out.shape[0])]


@EMBEDDERS.register("colpali")
class ColPaliMultiVectorEmbedder(Embedder):
    output_kind = OutputKind.MULTIVECTOR
    supported_modalities = frozenset({Modality.IMAGE, Modality.TEXT})

    def __init__(
        self,
        model: str = "vidore/colqwen2.5-v0.2",
        variant: str | None = None,
        batch_size: int = 4,
        device: str | None = None,
        # None = bfloat16 on CUDA, float32 elsewhere
        dtype: str | None = None,
        revision: str | None = None,
        # forwarded to the model's from_pretrained (attn_implementation, ...)
        **model_kwargs: Any,
    ):
        self._device = device or detect_device()
        if dtype is None:
            dtype = "bfloat16" if self._device.startswith("cuda") else "float32"
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"colpali dtype {dtype!r} not supported; choose from {sorted(DTYPE_MAP)}"
            )
        self._variant = variant or detect_variant(model)
        model_cls, proc_cls = _load_classes(self._variant)
        logger.info(
            "Loading ColPali %s (%s) on %s (dtype=%s, revision=%s)",
            model, self._variant, self._device, dtype, revision or "default",
        )
        self._model = model_cls.from_pretrained(
            model,
            torch_dtype=DTYPE_MAP[dtype],
            device_map=self._device,
            revision=revision,
            **model_kwargs,
        ).eval()
        self._processor = proc_cls.from_pretrained(model, revision=revision)
        self._model_name = model
        self._batch_size = batch_size
        self._dimensions_val = getattr(self._model, "dim", None)
        self._encode_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions_val

    def _process(self, items: list[Any]):
        if all(isinstance(x, str) for x in items):
            return self._processor.process_queries(items)
        if any(isinstance(x, str) for x in items):
            raise TypeError("colpali batch mixes text and images")
        # processors expect 3-channel input; grayscale / RGBA pages are common
        return self._processor.process_images([img.convert("RGB") for img in items])

    def _encode(self, batch: list[Any]) -> list[MultiVectorEmbedding]:
        out: list[MultiVectorEmbedding] = []
        with self._encode_lock, torch.inference_mode():
            for start in range(0, len(batch), self._batch_size):
                inputs = self._process(batch[start : start + self._batch_size])
                inputs = inputs.to(self._model.device)
                emb = self._model(**inputs)
                out.extend(
                    MultiVectorEmbedding(vectors=v)
                    for v in _trim(emb, inputs["attention_mask"])
                )
        return out

    async def embed(self, batch: list[Any]) -> list[MultiVectorEmbedding]:
        return await asyncio.to_thread(self._encode, batch)
