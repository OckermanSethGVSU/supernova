"""open_clip, pylate and colpali backends, with each library's model stubbed
(no weight downloads). Each block skips when its library isn't installed,
since the backend module only registers when its import succeeds.
"""

from __future__ import annotations

import asyncio
import importlib

import numpy as np
import pytest
import torch

pytest.importorskip("obstore")  # nova_embed.embedders package pulls in storage

from nova_embed.config import EmbedderEntry
from nova_embed.embedders.engine import build_engine
from nova_embed.models import MultiVectorEmbedding


def entry(name, kind, type_, column, modality, **extra) -> EmbedderEntry:
    data = {
        "name": name,
        "kind": kind,
        "type": type_,
        "input_column": column,
        "modality": modality,
    }
    data.update(extra)
    return EmbedderEntry.model_validate(data)


def run(engine, rows):
    return asyncio.run(engine.embed(rows))


def image(value=0):
    from PIL import Image

    return Image.new("L", (4, 4), color=value)  # grayscale on purpose


def backend(name):
    try:
        return importlib.import_module(f"nova_embed.embedders.backends.{name}")
    except ImportError as e:
        pytest.skip(f"{name} backend unavailable: {e}")


# --------------------------------------------------------------------------
# open_clip
# --------------------------------------------------------------------------


class _ClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def encode_text(self, tokens):
        # length-of-text vector, un-normalized
        return tokens.float().sum(dim=1, keepdim=True).repeat(1, 3) + 1.0

    def encode_image(self, pixels):
        _OpenClipStub.image_dtypes.append(pixels.dtype)
        return torch.full((pixels.shape[0], 3), 2.0)


class _ClipTokenizer:
    context_length = 77

    def __call__(self, texts):
        return torch.tensor([[len(t)] for t in texts])


class _OpenClipStub:
    calls: list[dict] = []
    image_dtypes: list = []

    @staticmethod
    def get_input_dtype(precision):
        return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)

    @classmethod
    def create_model_and_transforms(cls, model, **kwargs):
        cls.calls.append({"model": model, **kwargs})
        return _ClipModel(), None, lambda img: torch.zeros(3, 2, 2)

    @staticmethod
    def get_tokenizer(model):
        return _ClipTokenizer()


@pytest.fixture
def open_clip_stub(monkeypatch):
    mod = backend("open_clip")
    _OpenClipStub.calls.clear()
    _OpenClipStub.image_dtypes.clear()
    monkeypatch.setattr(mod, "open_clip", _OpenClipStub)
    return mod


def test_open_clip_text_and_image_share_one_model(open_clip_stub):
    kw = {"model": "ViT-B-32", "pretrained": "laion2b"}
    engine = build_engine([
        entry("img", "dense", "open_clip", "image", "image", **kw),
        entry("cap", "dense", "open_clip", "caption", "text", **kw),
    ])
    out = run(engine, [{"image": image(), "caption": "abcd"}])
    assert len(_OpenClipStub.calls) == 1  # shared instance
    for name in ("img", "cap"):
        (vec,) = out[name]
        assert np.isclose(np.linalg.norm(vec), 1.0)
    spec = {s.name: s for s in engine.output_specs}
    assert spec["img"].dimensions == 3
    assert spec["img"].model_name == "ViT-B-32/laion2b"


def test_open_clip_cpu_defaults_to_fp32_and_can_skip_normalize(open_clip_stub):
    engine = build_engine([
        entry("cap", "dense", "open_clip", "caption", "text", normalize=False, device="cpu"),
    ])
    (vec,) = run(engine, [{"caption": "ab"}])["cap"]
    assert _OpenClipStub.calls[0]["precision"] == "fp32"
    assert np.allclose(vec, 3.0)


@pytest.mark.parametrize("dtype, expected", [
    ("float16", torch.float16), ("bfloat16", torch.bfloat16), ("float32", torch.float32),
])
def test_open_clip_pixels_use_open_clip_input_dtype(open_clip_stub, dtype, expected):
    # the stub model's parameters stay float32, as open_clip's half-precision
    # conversion leaves some of them: the input dtype must not be read from them
    engine = build_engine([
        entry("img", "dense", "open_clip", "image", "image", dtype=dtype, device="cpu"),
    ])
    run(engine, [{"image": image()}])
    assert _OpenClipStub.image_dtypes == [expected]


def test_open_clip_batches_by_batch_size(open_clip_stub):
    engine = build_engine([
        entry("cap", "dense", "open_clip", "caption", "text", batch_size=2),
    ])
    out = run(engine, [{"caption": "x" * n} for n in range(1, 6)])["cap"]
    assert len(out) == 5


# --------------------------------------------------------------------------
# pylate
# --------------------------------------------------------------------------


class _ColBERT:
    instances: list["_ColBERT"] = []

    def __init__(self, model_name_or_path, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.kwargs = kwargs
        self.query_length = kwargs.get("query_length", 32)
        self.document_length = kwargs.get("document_length", 180)
        self.encode_calls: list[dict] = []
        _ColBERT.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.encode_calls.append(kwargs)
        tag = 2.0 if kwargs["is_query"] else 1.0
        return [np.full((len(t), 4), tag, dtype=np.float32) for t in texts]


@pytest.fixture
def pylate_stub(monkeypatch):
    mod = backend("pylate")
    _ColBERT.instances.clear()
    monkeypatch.setattr(mod, "_LOADED", {})
    monkeypatch.setattr(mod, "pylate_models", type("M", (), {"ColBERT": _ColBERT}))
    return mod


def test_pylate_document_side(pylate_stub):
    engine = build_engine([entry("cb", "multivector", "pylate", "text", "text")])
    (vec,) = run(engine, [{"text": "abc"}])["cb"]
    assert isinstance(vec, MultiVectorEmbedding)
    assert np.asarray(vec.vectors).shape == (3, 4)
    assert _ColBERT.instances[0].encode_calls[0]["is_query"] is False
    (spec,) = engine.output_specs
    assert spec.dimensions == 4
    assert spec.max_tokens == 180


def test_pylate_query_side_and_lengths(pylate_stub):
    engine = build_engine([
        entry("cb", "multivector", "pylate", "text", "text", query=True, query_length=16),
    ])
    (vec,) = run(engine, [{"text": "abc"}])["cb"]
    assert np.all(np.asarray(vec.vectors) == 2.0)
    inst = _ColBERT.instances[0]
    assert inst.kwargs["query_length"] == 16
    assert "document_length" not in inst.kwargs
    assert engine.output_specs[0].max_tokens == 16


def test_pylate_document_and_query_sides_share_one_model(pylate_stub):
    engine = build_engine([
        entry("doc", "multivector", "pylate", "text", "text", batch_size=64),
        entry("qry", "multivector", "pylate", "query", "text", query=True, batch_size=8),
    ])
    out = run(engine, [{"text": "abc", "query": "ab"}])
    assert len(_ColBERT.instances) == 1
    calls = _ColBERT.instances[0].encode_calls
    assert sorted((c["is_query"], c["batch_size"]) for c in calls) == [(False, 64), (True, 8)]
    assert np.all(np.asarray(out["doc"][0].vectors) == 1.0)
    assert np.all(np.asarray(out["qry"][0].vectors) == 2.0)
    spec = {s.name: s for s in engine.output_specs}
    assert (spec["doc"].max_tokens, spec["qry"].max_tokens) == (180, 32)


def test_pylate_different_load_settings_load_separately(pylate_stub):
    build_engine([
        entry("a", "multivector", "pylate", "text", "text", document_length=100),
        entry("b", "multivector", "pylate", "text2", "text", document_length=200),
    ])
    assert len(_ColBERT.instances) == 2


def test_pylate_pooling_derives_dense(pylate_stub):
    engine = build_engine([
        entry("cb", "multivector", "pylate", "text", "text", pooling={"type": "mean"}),
    ])
    out = run(engine, [{"text": "abc"}])
    (pooled,) = out["cb_pooled"]
    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_pylate_rejects_images_before_load(pylate_stub):
    with pytest.raises(ValueError, match="modality"):
        build_engine([entry("cb", "multivector", "pylate", "img", "image")])
    assert _ColBERT.instances == []


# --------------------------------------------------------------------------
# colpali
# --------------------------------------------------------------------------


class _Batch(dict):
    def to(self, device):
        return self


class _Processor:
    seen_modes: list[str] = []

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        return cls()

    def process_images(self, images):
        assert all(img.mode == "RGB" for img in images)
        _Processor.seen_modes.append("image")
        # ragged pages: row i has i+2 real patches, padded to the max
        lens = [i + 2 for i in range(len(images))]
        return self._batch(lens)

    def process_queries(self, texts):
        _Processor.seen_modes.append("query")
        return self._batch([len(t) for t in texts])

    @staticmethod
    def _batch(lens):
        n = max(lens)
        mask = torch.zeros(len(lens), n, dtype=torch.long)
        for i, length in enumerate(lens):
            mask[i, :length] = 1
        return _Batch(input_ids=torch.zeros(len(lens), n), attention_mask=mask)


class _ColModel:
    loads: list[dict] = []
    dim = 5
    device = "cpu"

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.loads.append({"model": model, **kwargs})
        return cls()

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        # padded positions come back as zeros, as the real models do
        out = torch.ones(*attention_mask.shape, self.dim)
        return out * attention_mask.unsqueeze(-1)


@pytest.fixture
def colpali_stub(monkeypatch):
    mod = backend("colpali")
    _ColModel.loads.clear()
    _Processor.seen_modes.clear()
    stub = type("M", (), {
        "ColQwen2_5": _ColModel, "ColQwen2_5_Processor": _Processor,
        "ColIdefics3": _ColModel, "ColIdefics3Processor": _Processor,
    })
    monkeypatch.setattr(mod, "colpali_models", stub)
    return mod


@pytest.mark.parametrize(
    "model, variant",
    [
        ("vidore/colqwen2.5-v0.2", "colqwen2_5"),
        ("vidore/colqwen2-v1.0", "colqwen2"),
        ("vidore/colSmol-256M", "colidefics3"),
        ("vidore/colpali-v1.3", "colpali"),
    ],
)
def test_colpali_detect_variant(colpali_stub, model, variant):
    assert colpali_stub.detect_variant(model) == variant


def test_colpali_unknown_name_needs_variant(colpali_stub):
    with pytest.raises(ValueError, match="variant"):
        colpali_stub.detect_variant("someone/page-retriever")


def test_colpali_pages_and_queries_share_model_and_trim_padding(colpali_stub):
    engine = build_engine([
        entry("page", "multivector", "colpali", "image", "image", batch_size=8),
        entry("query", "multivector", "colpali", "query", "text", batch_size=8),
    ])
    rows = [
        {"image": image(0), "query": "ab"},
        {"image": image(1), "query": "abcde"},
        {"image": image(2), "query": "a"},
    ]
    out = run(engine, rows)
    assert len(_ColModel.loads) == 1  # one model for both sides
    assert set(_Processor.seen_modes) == {"image", "query"}
    page_shapes = [np.asarray(v.vectors).shape for v in out["page"]]
    assert page_shapes == [(2, 5), (3, 5), (4, 5)]  # no padding rows kept
    query_shapes = [np.asarray(v.vectors).shape for v in out["query"]]
    assert query_shapes == [(2, 5), (5, 5), (1, 5)]
    for v in out["page"] + out["query"]:
        assert np.all(np.asarray(v.vectors) == 1.0)  # no zero (pad) vectors
        assert np.asarray(v.vectors).dtype == np.float32
    assert engine.output_specs[0].dimensions == 5


def test_colpali_explicit_variant_and_cpu_dtype(colpali_stub):
    build_engine([
        entry("page", "multivector", "colpali", "image", "image",
              model="me/page-model", variant="colidefics3", device="cpu"),
    ])
    (load,) = _ColModel.loads
    assert load["model"] == "me/page-model"
    assert load["torch_dtype"] == torch.float32


def test_colpali_missing_class_is_a_clear_error(colpali_stub):
    with pytest.raises(ValueError, match="upgrade colpali-engine"):
        build_engine([
            entry("page", "multivector", "colpali", "image", "image",
                  model="vidore/colpali-v1.3"),
        ])
