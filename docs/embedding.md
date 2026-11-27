# Embedding

`nova embed` streams rows from a dataset, runs one or more embedding models
over them, and writes the results as parquet. The output is what
[`nova load`](loading/overview.md) and [`nova bf`](brute-force/overview.md) read.

![Embedding Pipeline](fig/embedding_pipeline.svg)

## Config

A config has four sections: where rows come from (`source`), which models run
(`embedders`), how the run is batched (`pipeline`), and where parquet goes
(`storage`). More examples live in `configs/embedder/`.

```yaml
source:
  type: huggingface
  dataset_name: mteb/tweet_sentiment_extraction
  split: train

embedders:
  - name: dense                   # output column: dense_embedding
    kind: dense                   # dense | sparse | multivector
    type: sentence_transformer    # backend, see the table below
    model: sentence-transformers/all-MiniLM-L6-v2
    input_column: text
    modality: text                # text | image | multimodal (required)
    batch_size: 64

  - name: bm25                    # output column: bm25_embedding
    kind: sparse
    type: fastembed
    model: Qdrant/bm25
    input_column: text
    modality: text

pipeline:
  chunk_size: 10000               # rows per batch
  flush_threshold: 100000         # rows per output parquet file

storage:
  type: object_store
  path: s3://my-bucket/tweets/minilm
```

Each entry in `embedders:` writes one column, named `{name}_embedding` unless
you set `output_column`. Keys the pipeline doesn't know (`batch_size`, `dtype`,
`device`, `revision`, ...) are passed to the backend. `${VAR}` and
`${VAR:-default}` are expanded from the environment.

## Embedders

| `type` | `kind` | Inputs | Notes |
|---|---|---|---|
| `sentence_transformer` | dense, sparse | text, image (dense only) | Any sentence-transformers model. Sparse uses `SparseEncoder` (SPLADE-family, BM42). CLIP-family models take images. |
| `fastembed` | dense, sparse, multivector | text | fastembed's ONNX models: e.g. `BAAI/bge-small-en-v1.5` (dense), `Qdrant/bm25` (sparse), `colbert-ir/colbertv2.0` (multivector). Runs well on CPU. |
| `openai` | dense | text | OpenAI API, or any OpenAI-compatible server via `base_url` (`api_key: none` for local servers). |
| `bge_m3` | dense, sparse, multivector | text | `BAAI/bge-m3`. |
| `vllm` | dense | text, image, multimodal | vLLM pooling runner. The only backend that takes `modality: multimodal`. |
| `open_clip` | dense | text, image | OpenCLIP checkpoints (CLIP, SigLIP, EVA, MetaCLIP). `model` is the architecture (`ViT-B-32`, or `hf-hub:...`), `pretrained` the weight tag (`laion2b_s34b_b79k`). Vectors are L2-normalized unless `normalize: false`. Install the `open_clip` extra. |
| `pylate` | multivector | text | ColBERT-family models through PyLate (`lightonai/GTE-ModernColBERT-v1`, `answerdotai/answerai-colbert-small-v1`, ...). Set `query: true` when the column holds queries. Install the `pylate` extra. |
| `colpali` | multivector | image, text | Page-image retrieval models through colpali-engine (ColPali, ColQwen2/2.5, ColSmol). Images are embedded as pages, text as queries, in one space. The family is read from the model id, or set `variant:`. Page rows hold about a thousand vectors each, so lower `pipeline.flush_threshold`. Install the `colpali` extra. |

Entries on the same model and input column are fused into one forward pass
when the backend supports it (e.g. `bge_m3` dense, sparse, and multivector
together). This happens automatically.

## Sources

| `type` | Reads | Sharded by |
|---|---|---|
| `huggingface` | Parquet datasets on the Hugging Face Hub | Row range |
| `huggingface_jsonl` | JSON Lines datasets on the Hub (e.g. `MedRAG/pubmed`) | Whole files |

Both take `dataset_name` and `split`, plus:

- `render_columns` builds a new column from others, e.g.
  `combined: "{title}: {abstract}"`. Use it as an `input_column`.
- `exclude_columns` drops columns before anything is read or embedded.

To split long text before embedding, add a `chunking:` block with
`strategy: fixed_char` (`chunk_chars`, `overlap`). Splitting requires every
entry to read the same text column.

## Pipeline options

| Key | Default | Meaning |
|---|---|---|
| `chunk_size` | `10000` | Rows embedded per batch. |
| `flush_threshold` | `100000` | Rows buffered before a parquet file is written. |
| `row_group_size` | pyarrow default | Parquet row-group size. Smaller groups let readers fetch and decode a file in parallel. |
| `on_empty_input` | `skip` | Empty input: `skip` the row, write a `null` embedding, or `error`. |
| `drop_columns` | `[]` | Source columns to embed but leave out of the output (e.g. raw image bytes). |
| `content_addressed_files` | `false` | Name each file by a hash of its contents. Makes re-runs idempotent. |
| `shard_output_buckets` | none | Spread files across N subdirectories. |
| `include_source_provenance` | `false` | Add `source_file_name` and `source_row_number` columns. |

## Storage

| `type` | Writes to | Key |
|---|---|---|
| `object_store` (alias `s3`) | S3 (`s3://`), GCS (`gs://`), Azure (`az://`), or any S3-compatible store (set `endpoint`) | `path` |
| `hf` | A Hugging Face Storage Bucket | `bucket_id` |
| `local` | Local disk | `output_dir` |

Credentials come from each provider's usual environment variables or config
files. Extra provider options go under `config:` and are passed to
[obstore](https://developmentseed.org/obstore/latest/api/store/). Downstream
tools read S3 reliably; check that yours can read GCS or Azure before a big run.

## Output

Every source column is kept, plus one column per embedder entry:

| `kind` | Arrow type |
|---|---|
| `dense` | `list<float32>` |
| `sparse` | `struct{indices: list<uint32>, values: list<float32>}` |
| `multivector` | `list<list<float32>>` |

There is no id column. `nova load` derives point ids from each row's file path
and row number, so re-reading the same files always gives the same ids.

Each run also writes `_manifest.json` (`rank<NN>__manifest.json` per rank in a
fleet run) next to the parquet. It records the models, their resolved
revisions and dtypes, library versions, the git commit, the rank's row range,
and whether the rank finished (`complete`).

## Running

```bash
nova embed <config>                          # run locally
nova embed <config> --dry-run                # print the plan, embed nothing
nova embed <config> --num-jobs 8 --job-rank 0  # one shard of a fleet run
```

`--job-rank` defaults to `$SKYPILOT_JOB_RANK`. See
[Distributed](distributed.md) for launching a fleet.

To estimate throughput and cost before renting GPUs, run `nova embed predict`.
It samples the dataset, tokenizes it with each model's tokenizer, and prices
the run for a GPU type. It doesn't need a GPU. Treat the numbers as a way to
compare options (GPU types, batch sizes, cutoffs): in validation runs the
median prediction was off by about 1.6x.

```bash
nova embed predict <config> --gpu h100 --num-gpus 8
nova embed predict --help                    # every flag
```
