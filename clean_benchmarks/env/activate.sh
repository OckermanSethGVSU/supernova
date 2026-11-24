# Activate the pinned benchmark environment.
# Refuses environments built with different version pins.
# Prioritizes PyTorch's CUDA libraries to avoid conflicts with conda's copies.
# Configures the cuVS sparse binding and environment paths.
#
# Usage: . clean_benchmarks/env/activate.sh

_pe_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_pe_name="${ENV_NAME:-paper}"
_pe_conda="${CONDA:-$HOME/miniforge3}"
_pe_stamp="$_pe_conda/envs/$_pe_name/.versions.env.sha256"
# the hash of versions.env's assignments (comments and the freeze date left out: filling in
# STACK_FROZEN_ON is not a new stack); build.sh computes the same (VERSIONS_SHA)
_pe_want="$(grep -v '^STACK_FROZEN_ON=' "$_pe_here/versions.env" | sed 's/[[:space:]]*#.*$//' \
            | grep -v '^[[:space:]]*$' | sha256sum | cut -d' ' -f1)"
if [ -d "$_pe_conda/envs/$_pe_name" ] && [ "$(cat "$_pe_stamp" 2>/dev/null)" != "$_pe_want" ]; then
  echo "activate.sh: env '$_pe_name' was not built from this versions.env" \
       "(stamp ${_pe_stamp}: '$(cat "$_pe_stamp" 2>/dev/null || echo missing)', versions.env: $_pe_want);" \
       "rebuild it (REBUILD_ENV=1 bash build.sh)" >&2
  return 1
fi
_pe_u=0; case $- in *u*) _pe_u=1; set +u;; esac
. "$_pe_conda/etc/profile.d/conda.sh" && conda activate "$_pe_name" || { [ $_pe_u = 1 ] && set -u; return 1; }
[ $_pe_u = 1 ] && set -u
_pe_cuda="$(sed -n 's/^CUDA_VERSION=\([0-9]*\).*/\1/p' "$_pe_here/versions.env")"
_pe_pip=""
for _pe_d in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cu"$_pe_cuda"/lib; do
  [ -d "$_pe_d" ] && _pe_pip="$_pe_d"
done
export LD_LIBRARY_PATH="${_pe_pip:+$_pe_pip:}$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 4. the sparse cuVS shim built against this env's libcuvs (build.sh's shim step copies it into
#    the env), whatever tree the job runs from; an explicit NOVA_CUVS_SPARSE_LIB wins
if [ -z "${NOVA_CUVS_SPARSE_LIB:-}" ] && [ -f "$CONDA_PREFIX/lib/libnova_cuvs_sparse.so" ]; then
  export NOVA_CUVS_SPARSE_LIB="$CONDA_PREFIX/lib/libnova_cuvs_sparse.so"
fi
# 5. the paper env's bin first on PATH, once: conda activate puts it first, but a caller that
#    prepends after activation (e.g. ~/.local/bin) would win; env/check.py refuses a python /
#    nova-bf / bench that resolves outside the env
_pe_path=":$PATH:"; _pe_path="${_pe_path//:$CONDA_PREFIX\/bin:/:}"; _pe_path="${_pe_path#:}"
export PATH="$CONDA_PREFIX/bin${_pe_path:+:${_pe_path%:}}"
unset _pe_path _pe_here _pe_name _pe_conda _pe_stamp _pe_want _pe_u _pe_cuda _pe_pip _pe_d
