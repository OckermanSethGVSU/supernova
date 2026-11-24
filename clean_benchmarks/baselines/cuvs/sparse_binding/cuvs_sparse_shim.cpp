/*
 * Minimal C entry points over cuVS's SPARSE exact brute-force kNN, which
 * cuVS 26.10 exposes only in C++ (cuvs/neighbors/brute_force.hpp:
 * brute_force::build(res, device_csr_matrix_view<const float, int, int, int>,
 * DistanceType, float) -> sparse_index<float, int>, and
 * brute_force::search(res, sparse_search_params, sparse_index, csr query view,
 * device_matrix_view<int, int64_t> neighbors, device_matrix_view<float, int64_t>
 * distances)). Neither the C API (libcuvs_c) nor the Python package wraps it.
 *
 * Compiled with the HOST compiler only (no nvcc: these headers are host
 * code, as cuVS's own C API sources include them from .cpp files) against
 * the libcuvs / libraft / librmm / rapids_logger wheels' headers, linked
 * against the wheel's libcuvs.so. Called from Python through ctypes
 * (cuvs_sparse.py). See build.sh.
 *
 * Resources: the caller passes a cuvsResources_t (the C API's handle, e.g.
 * cuvs.common.Resources().get_c_obj()), which IS a raft::resources* (c_api.cpp
 * cuvsResourcesCreate: `new raft::resources{}` cast to uintptr_t, and every C
 * API entry point casts it back the same way). Its stream is whatever the
 * caller bound with cuvsStreamSet; every kernel of the call runs there,
 * except what cuVS itself puts on stream 0 (see cuvs_sparse.py: the caller
 * must bind cudaStreamPerThread).
 *
 * All pointers are DEVICE pointers. Nothing is allocated here except cuVS's
 * own workspace (RMM's current device resource). Errors never cross the C
 * boundary as exceptions: they return non-zero with the message in errbuf.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace {

void set_err(char* buf, size_t len, const char* msg)
{
  if (buf != nullptr && len > 0) {
    std::snprintf(buf, len, "%s", msg);
  }
}

auto csr_view(const int32_t* indptr,
              const int32_t* indices,
              const float* values,
              int32_t n_rows,
              int32_t n_cols,
              int32_t nnz)
{
  // raft's structure view takes non-const index pointers; cuVS only reads them
  auto structure = raft::make_device_compressed_structure_view<int, int, int>(
    const_cast<int32_t*>(indptr), const_cast<int32_t*>(indices), n_rows, n_cols, nnz);
  return raft::make_device_csr_matrix_view<const float, int, int, int>(values, structure);
}

}  // namespace

#define NOVA_API __attribute__((visibility("default")))

extern "C" {

/* ABI version of this file's entry points (bumped on any signature change). */
NOVA_API int nova_cuvs_sparse_abi(void) { return 1; }

/*
 * Exact top-k of every query row against every index row, both CSR (int32
 * indptr / indices, float32 values), one cuVS build + search.
 *
 * metric: 0 = inner product (raw dot products, LARGEST first), 1 = cosine
 * (cuVS CosineExpanded: 1 - cos, SMALLEST first). n_cols is the column count
 * of BOTH matrices (every index must be < n_cols; cuVS's dense shared-memory
 * strategy indexes a row's columns directly when n_cols is small).
 * batch_size_index / batch_size_query: sparse_search_params (cuVS default
 * 2 << 14 each); each (query batch x index batch) tile is a DENSE float32
 * distance matrix plus an int32 id matrix. k <= n_index_rows.
 * out_neighbors: int32 [n_query_rows, k]; out_distances: float32 [n_query_rows, k].
 * Returns 0 on success, 1 on error (message in errbuf).
 */
NOVA_API int nova_cuvs_sparse_knn(uintptr_t res,
                         const int32_t* idx_indptr,
                         const int32_t* idx_indices,
                         const float* idx_values,
                         int32_t n_idx_rows,
                         int32_t idx_nnz,
                         const int32_t* q_indptr,
                         const int32_t* q_indices,
                         const float* q_values,
                         int32_t n_q_rows,
                         int32_t q_nnz,
                         int32_t n_cols,
                         int32_t k,
                         int32_t metric,
                         int32_t batch_size_index,
                         int32_t batch_size_query,
                         int32_t* out_neighbors,
                         float* out_distances,
                         char* errbuf,
                         size_t errlen)
{
  try {
    if (res == 0) {
      set_err(errbuf, errlen, "null cuvsResources_t");
      return 1;
    }
    if (n_idx_rows <= 0 || n_q_rows <= 0 || n_cols <= 0 || k <= 0 || k > n_idx_rows ||
        idx_nnz < 0 || q_nnz < 0 || batch_size_index <= 0 || batch_size_query <= 0) {
      set_err(errbuf, errlen, "bad sizes (need rows > 0, n_cols > 0, 0 < k <= index rows, "
                              "nnz >= 0, batch sizes > 0)");
      return 1;
    }
    cuvs::distance::DistanceType dt;
    if (metric == 0) {
      dt = cuvs::distance::DistanceType::InnerProduct;
    } else if (metric == 1) {
      dt = cuvs::distance::DistanceType::CosineExpanded;
    } else {
      set_err(errbuf, errlen, "metric must be 0 (inner product) or 1 (cosine)");
      return 1;
    }
    auto const& handle = *reinterpret_cast<raft::resources const*>(res);
    auto index_view =
      csr_view(idx_indptr, idx_indices, idx_values, n_idx_rows, n_cols, idx_nnz);
    auto query_view = csr_view(q_indptr, q_indices, q_values, n_q_rows, n_cols, q_nnz);
    auto index = cuvs::neighbors::brute_force::build(handle, index_view, dt, 0.0f);
    cuvs::neighbors::brute_force::sparse_search_params params;
    params.batch_size_index = batch_size_index;
    params.batch_size_query = batch_size_query;
    auto neighbors =
      raft::make_device_matrix_view<int, int64_t, raft::row_major>(out_neighbors, n_q_rows, k);
    auto distances =
      raft::make_device_matrix_view<float, int64_t, raft::row_major>(out_distances, n_q_rows, k);
    cuvs::neighbors::brute_force::search(handle, params, index, query_view, neighbors, distances);
    return 0;
  } catch (std::exception const& e) {
    set_err(errbuf, errlen, e.what());
    return 1;
  } catch (...) {
    set_err(errbuf, errlen, "unknown C++ exception");
    return 1;
  }
}

/* cuVS's own defaults for sparse_search_params (2 << 14 each in 26.10), read
 * from the header this file was compiled against. */
NOVA_API void nova_cuvs_sparse_default_batches(int32_t* batch_size_index, int32_t* batch_size_query)
{
  cuvs::neighbors::brute_force::sparse_search_params p;
  *batch_size_index = p.batch_size_index;
  *batch_size_query = p.batch_size_query;
}

}  // extern "C"
