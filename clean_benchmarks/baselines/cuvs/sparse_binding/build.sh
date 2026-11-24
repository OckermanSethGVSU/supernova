#!/usr/bin/env bash
# Build libnova_cuvs_sparse.so (cuvs_sparse_shim.cpp) against the RAPIDS wheels
# installed in PYTHON's environment: libcuvs-cuXX, libraft-cuXX, librmm-cuXX,
# rapids-logger (all pulled in by `pip install cuvs-cu13` / `cuvs-cu12`) and the
# CUDA runtime headers (nvidia-cuda-runtime wheel, or $CUDA_HOME/include).
# (The paper env's conda-built cuVS uses env/build.sh's shim step instead.)
# Host compiler only (g++ >= 11, C++20); building needs no nvcc and no GPU.
#
#   PYTHON=~/vcuvs/bin/python ./build.sh          # -> ./libnova_cuvs_sparse.so
#
# Env: PYTHON (default: python3), CXX (default: g++), CUDA_HOME (optional),
# OUT (default: this directory's libnova_cuvs_sparse.so).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
CXX="${CXX:-g++}"
OUT="${OUT:-$here/libnova_cuvs_sparse.so}"

# every wheel's root dir (include/ + lib64/), found without importing (and so
# without dlopening) anything
read -r CUVS RAFT RMM LOGGER CUDA_INC < <("$PY" - "${CUDA_HOME:-}" <<'EOF'
import importlib.util, os, sys
def root(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"build.sh: python module {name!r} not found: pip install the cuvs wheel "
                 "(cuvs-cu13 / cuvs-cu12 from https://pypi.nvidia.com) into this env")
    return list(spec.submodule_search_locations)[0]
dirs = [root(m) for m in ("libcuvs", "libraft", "librmm", "rapids_logger")]
cuda_home = sys.argv[1]
cands = [os.path.join(cuda_home, "include")] if cuda_home else []
for ns in ("nvidia.cu13", "nvidia.cu12", "nvidia.cuda_runtime"):
    try:
        spec = importlib.util.find_spec(ns)
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.submodule_search_locations:
        cands.append(os.path.join(list(spec.submodule_search_locations)[0], "include"))
cands += ["/usr/local/cuda/include"]
inc = next((c for c in cands if os.path.exists(os.path.join(c, "cuda_runtime_api.h"))), None)
if inc is None:
    sys.exit("build.sh: no cuda_runtime_api.h (set CUDA_HOME, or pip install "
             "nvidia-cuda-runtime matching the wheel's CUDA major)")
print(*dirs, inc)
EOF
)

echo "libcuvs:       $CUVS"
echo "libraft:       $RAFT"
echo "librmm:        $RMM"
echo "rapids_logger: $LOGGER"
echo "CUDA headers:  $CUDA_INC"

# the defines cuvs / raft's exported CMake targets put on their consumers
# (lib64/cmake/{cuvs,raft}/*-targets.cmake: RAFT_COMPILED,
# RAFT_LOG_ACTIVE_LEVEL, RAFT_SYSTEM_LITTLE_ENDIAN); librmm/include/rapids is
# the CCCL (thrust / cub / libcu++) the wheels were built with
"$CXX" -std=c++20 -O2 -fPIC -shared -fvisibility=hidden \
  -DRAFT_COMPILED -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
  -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO \
  -Wno-deprecated-declarations \
  -I"$CUVS/include" -I"$RAFT/include" -I"$RMM/include" -I"$RMM/include/rapids" \
  -I"$LOGGER/include" -I"$CUDA_INC" \
  "$here/cuvs_sparse_shim.cpp" -o "$OUT" \
  -L"$CUVS/lib64" -lcuvs -Wl,-rpath,"$CUVS/lib64" -Wl,--no-undefined -Wl,--allow-shlib-undefined

echo "built $OUT"
