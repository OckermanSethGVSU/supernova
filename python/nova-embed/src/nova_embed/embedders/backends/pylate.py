"""
PyLate backend: ColBERT-family multivector text embeddings.

PyLate is the maintained library for late-interaction text models
(GTE-ModernColBERT, answerai-colbert, jina-colbert, colbertv2, ...). Each input
becomes one vector per token, written as a multivector column.

Documents and queries are encoded differently: queries are padded with
[MASK] tokens up to a fixed length ("query augmentation") and use the query
marker token. Set `query: true` on an entry whose column holds queries. The
flag is recorded in the run manifest, since a corpus and its queries must be
encoded with the matching side. A document entry and a query entry on the
same model share one loaded copy: `query` and `batch_size` only change how
it is called, not what is loaded.

Example config entry:
    embedders:
      - name: colbert
        kind: multivector
        type: pylate
        model: lightonai/GTE-ModernColBERT-v1
        input_column: text
        modality: text
"""

import asyncio
import logging
import threading

from typing import Any

import numpy as np

from pylate import models as pylate_models

from nova_embed.embedders.backends.device import detect_device
from nova_embed.embedders.base import Embedder, OutputKind
from nova_embed.models import MultiVectorEmbedding
from nova_embed.registry import EMBEDDERS

logger = logging.getLogger(__name__)

# load settings -> (loaded ColBERT model, its encode lock). Shared across
# entries so the document and query sides don't each load the weights.
_LOADED: dict[str, tuple[Any, threading.Lock]] = {}


def _load(model: str, device: str, **load_kwargs: Any) -> tuple[Any, threading.Lock]:
    key = repr((model, device, sorted(load_kwargs.items())))
    if key not in _LOADED:
        logger.info("Loading PyLate ColBERT %s on %s", model, device)
        loaded = pylate_models.ColBERT(
            model_name_or_path=model, device=device, **load_kwargs
        )
        _LOADED[key] = (loaded, threading.Lock())
    else:
        logger.info("Reusing the loaded PyLate ColBERT %s", model)
    return _LOADED[key]


@EMBEDDERS.register("pylate")
class PyLateMultiVectorEmbedder(Embedder):
    output_kind = OutputKind.MULTIVECTOR

    def __init__(
        self,
        model: str = "lightonai/GTE-ModernColBERT-v1",
        batch_size: int = 32,
        device: str | None = None,
        # encode with the query side (is_query=True) instead of the document side
        query: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        # token caps; None keeps the model's own configured lengths
        query_length: int | None = None,
        document_length: int | None = None,
        # forwarded to pylate.models.ColBERT
        **pylate_kwargs: Any,
    ):
        self._device = device or detect_device()
        if query_length is not None:
            pylate_kwargs["query_length"] = query_length
        if document_length is not None:
            pylate_kwargs["document_length"] = document_length
        self._model, self._encode_lock = _load(
            model,
            self._device,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **pylate_kwargs,
        )
        self._model_name = model
        self._batch_size = batch_size
        self._query = query
        self._dimensions_val = self._model.get_sentence_embedding_dimension()
        self._max_tokens = (
            self._model.query_length if query else self._model.document_length
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int | None:
        return self._dimensions_val

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    def _encode(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        with self._encode_lock:
            rows = self._model.encode(
                texts,
                batch_size=self._batch_size,
                is_query=self._query,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        # one (num_tokens, dim) array per input; padding is already removed
        # by PyLate on the document side
        return [
            MultiVectorEmbedding(vectors=np.asarray(r, dtype=np.float32)) for r in rows
        ]

    async def embed(self, texts: list[str]) -> list[MultiVectorEmbedding]:
        return await asyncio.to_thread(self._encode, texts)
