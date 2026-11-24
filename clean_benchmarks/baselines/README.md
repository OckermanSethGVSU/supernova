# Baselines: Credits and Modifications

We benchmark FAISS, cuVS, SPARe, and PyLate, with a few modifications and additional implementations described below. All other functionality uses the original libraries. Versions are pinned in `env/versions.env`.

## Libraries

| Library | Authors | Version | License |
|---|---|---|---|
| [FAISS](https://github.com/facebookresearch/faiss) | Meta AI Research | 1.16.0 | MIT |
| [cuVS](https://github.com/NVIDIA/cuvs) | NVIDIA (RAPIDS) | 26.10.00 | Apache-2.0 |
| [SPARe](https://github.com/ieeta-pt/SPARe) | Tiago Almeida and Sérgio Matos | `e7f4ad1` | Apache-2.0 |
| [PyLate](https://github.com/lightonai/pylate) | LightOn (Antoine Chaffin and Raphaël Sourty) | 1.6.0 (`3389837`) | MIT |

The environment uses PyTorch 2.15.0+cu132 and CUDA 13.2. Each library retains its original license, including any modified library files. Our additions are released under this repository's Apache-2.0 license.

## Modifications and Additions

**FAISS:** Used without modifications.

**cuVS:** Applied two patches. The first fixes incorrect sparse inner-product results across multiple batches and when a batch contains fewer than k rows. The second disables cuTile for compatibility with CUDA 13.2 without changing search behavior. We also added a Python/C++ binding for sparse brute-force search.

**SPARe:** The original library is unchanged. We added batched versions of its dot-product and iterative search methods, preserving the original scoring and top-k operations. We also added a version of the original iterative method that removes per-term GPU synchronization. The original implementations are benchmarked separately.

**PyLate:** The original library is unchanged. We added batched exhaustive MaxSim search using PyLate's scoring functions. We also benchmark the original reranking implementation and a version with corrected padding. The original zero padding can produce incorrect MaxSim scores. All modes use float32.

## Known Limitation

cuVS dense filtered search can return incorrect results when fewer than k rows match a query in its SDDMM selection path. Our configurations avoid this path, and the benchmark checks that it is never used.

## References

Please cite the original libraries when using these baselines:

- **FAISS:** Douze et al., *The Faiss Library* (2024); Johnson et al., *Billion-scale Similarity Search with GPUs* (2019).
- **cuVS:** NVIDIA cuVS. See the repository for citations to the underlying algorithms.
- **SPARe:** Almeida and Matos, *SPARe: Supercharged Lexical Retrievers on GPU with Sparse Kernels* (ECIR 2024).
- **PyLate:** Chaffin and Sourty, *PyLate: Flexible Training and Retrieval for Late Interaction Models* (CIKM 2025).