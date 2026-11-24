#!/usr/bin/env bash
# Build the pinned benchmark environment from versions.env.
# Installs cuVS with verified patches, FAISS, PyTorch, and all benchmark dependencies.
# Runs validation and saves environment lock files.
#
# bash build.sh [steps...]     Build all or selected steps.
# REBUILD_ENV=1 bash build.sh  Rebuild an environment with different version pins.
#
# All engines share one environment to ensure consistent dependencies.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CB="$(cd "$HERE/.." && pwd)"                           # clean_benchmarks
REPO="$(cd "$CB/.." && pwd)"                           # supernova
# the cuVS source patches; CUVS_PATCH_DIR= overrides
CUVS_PATCH_DIR="${CUVS_PATCH_DIR:-$CB/baselines/cuvs/patches}"
set -a; . "$HERE/versions.env"; set +a

ENV_NAME="${ENV_NAME:-paper}"
CONDA="${CONDA:-$HOME/miniforge3}"
WORK="${WORK:-$HOME/paper_env_build}"                  # clones and logs
PAR="${PAR:-24}"                                       # nvcc jobs: ~5 GiB each on a 124 GiB node (the 2026-11-06 build)
REBUILD_ENV="${REBUILD_ENV:-0}"                        # 1: remove an env built from another versions.env
export ENV_NAME CONDA
mkdir -p "$WORK/logs"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$WORK/logs/progress.log"; }
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(patches conda cuvs shim faiss pip check freeze)
# hash of versions.env's assignments (comments and STACK_FROZEN_ON left out), as activate.sh does
VERSIONS_SHA="$(grep -v '^STACK_FROZEN_ON=' "$HERE/versions.env" | sed 's/[[:space:]]*#.*$//' \
                | grep -v '^[[:space:]]*$' | sha256sum | cut -d' ' -f1)"
STAMP="$CONDA/envs/$ENV_NAME/.versions.env.sha256"     # which versions.env built the env (activate.sh checks it)

# The CUDA libraries, as "conda-package pip-wheel version" rows. One copy of each runs, for every
# engine: the pip wheel beside torch, at exactly the versions.env version. The conda packages (lib
# and -dev) are what the cuVS build and FAISS link; they are installed at whatever version those
# packages require (not pinned: the solver picks, conda-explicit.lock.txt records it), and never
# loaded (activate.sh, bench/record/cudalibs.py; check.py fails if one is).
CUDA_LIBS="libcublas nvidia-cublas $CUBLAS_VERSION
libcusparse nvidia-cusparse $CUSPARSE_VERSION
libcusolver nvidia-cusolver $CUSOLVER_VERSION
libcufft nvidia-cufft $CUFFT_VERSION
libcurand nvidia-curand $CURAND_VERSION
cuda-cudart nvidia-cuda-runtime $CUDART_VERSION
cuda-nvrtc nvidia-cuda-nvrtc $NVRTC_VERSION
libnvjitlink nvidia-nvjitlink $NVJITLINK_VERSION"
PIP_PINS=(); CONDA_PKGS=()
while read -r cpkg ppkg ver; do
  PIP_PINS+=("$ppkg==$ver")
  CONDA_PKGS+=("$cpkg" "$cpkg-dev")
done <<< "$CUDA_LIBS"
# what the FAISS install must not move under the cuVS build (it is built against them)
FROZEN_BY_CUVS='^(cuda-version|librmm|rmm|libraft[a-z-]*|pylibraft|raft-dask|libcuvs|cuvs|libkvikio|kvikio|rapids-logger|libucxx|ucxx|nccl|libnccl)$'
want() { local x; for x in "${STEPS[@]}"; do [ "$x" = "$1" ] && return 0; done; return 1; }
# libfaiss / libcuvs are built (or packaged) against the conda copies but run on the pip copies:
# a conda copy newer than its pip copy may need symbols the pip copy lacks, and libcuvs then fails
# to load (seen with nvJitLink 12.9 vs torch's 12.8). So every conda copy must be <= its pip pin;
# the env's conda-meta/pinned keeps later installs (FAISS) from moving one above it.
#
conda_not_newer() {  # conda_not_newer STEP
  local cpkg ppkg ver have bad=0
  while read -r cpkg ppkg ver; do
    for p in "$cpkg" "$cpkg-dev"; do
      have=$(conda list -n "$ENV_NAME" "^$p\$" | awk '!/^#/ {print $2; exit}')
      [ -n "$have" ] || continue   # a missing one is check.py's failure
      if [ "$(printf '%s\n%s\n' "$have" "$ver" | sort -V | tail -1)" != "$ver" ]; then
        log "after $1: conda $p $have is NEWER than the pip copy every engine runs ($ppkg $ver)"
        bad=1
      fi
    done
  done <<< "$CUDA_LIBS"
  [ $bad = 0 ] || { log "add an upper bound for it to the cuVS env (or move the pip pins up)"; exit 1; }
}

# activate.sh: refuses an env built from another versions.env, activates it, and puts torch's pip
# CUDA libraries before the env's own lib dir on LD_LIBRARY_PATH (the env's lib dir is there for
# its libstdc++: GLIBCXX not found otherwise)
activate() { . "$HERE/activate.sh"; }
# every patch in CUVS_PATCHES must be a file in $CUVS_PATCH_DIR/ whose sha256 is the one pinned
# in CUVS_PATCH_SHA256 (name=sha256 entries).
verify_cuvs_patches() {
  local p f pin got e
  for p in $CUVS_PATCHES; do
    f="$CUVS_PATCH_DIR/$p"
    if [ ! -f "$f" ]; then
      log "PATCH MISSING: $f not found (baselines/cuvs/patches/$p)"
      return 1
    fi
    pin=""
    for e in ${CUVS_PATCH_SHA256:-}; do [ "${e%%=*}" = "$p" ] && pin="${e#*=}"; done
    if [ -z "$pin" ]; then
      log "PATCH NOT PINNED: $p has no name=sha256 entry in versions.env's CUVS_PATCH_SHA256"
      return 1
    fi
    got="$(sha256sum "$f" | cut -d' ' -f1)"
    if [ "$got" != "$pin" ]; then
      log "PATCH MISMATCH: $f sha256 $got, versions.env CUVS_PATCH_SHA256 pins $pin: refusing to apply it"
      return 1
    fi
  done
  log "cuvs patches verified against CUVS_PATCH_SHA256 ($CUVS_PATCH_DIR): ${CUVS_PATCHES:-none}"
}

if want patches; then
  log "STEP patches"
  verify_cuvs_patches || exit 1
fi

if want conda; then
  log "STEP conda"
  [ -x "$CONDA/bin/conda" ] || { curl -fsSL -o /tmp/mf.sh \
      https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
      && bash /tmp/mf.sh -b -p "$CONDA" > "$WORK/logs/miniforge.log" 2>&1; }
  . "$CONDA/etc/profile.d/conda.sh"
  [ -d "$WORK/cuvs" ] || git clone -q --branch "$CUVS_GIT_TAG" --depth 1 \
      https://github.com/rapidsai/cuvs.git "$WORK/cuvs"
  # cuVS's own CUDA 13 build environment at the tag, with these edits:
  #   cuda-version 13.3 -> $CUDA_VERSION (one CUDA minor for every engine; if the solver refuses,
  #     build at 13.3 and record that);
  #   no rapidsai-nightly channel (release packages only); python pinned;
  src="$WORK/cuvs/conda/environments/all_cuda-133_arch-x86_64.yaml"
  #   toolchains the libcuvs + python build does not use dropped (go / java / node / rust /
  #     clang tooling / pre-commit);
  #   the CUDA lib / -dev packages the env file does not name added (check.py requires every
  #     one, -dev included).
  # Each CUDA lib / -dev package gets an upper bound at its pip pin (<=, not ==: an exact pin is
  # unsatisfiable, FAISS takes an older cuBLAS), so conda's copy is never newer than the copy
  # every engine runs (conda_not_newer). nvJitLink needs it most: cuda-version=13.2 does not bound
  # it, and the solver has picked libnvjitlink 13.5.37 against the pip 13.2.86.
  # A bare "- pkg" line in the env file gets the bound; one with its own spec is kept as it is.
  EXTRA=""; BOUNDS=()
  while read -r cpkg ppkg ver; do
    for p in "$cpkg" "$cpkg-dev"; do
      if grep -Eq "^- $p$" "$src"; then BOUNDS+=(-e "s/^- $p$/- $p<=$ver/")
      elif ! grep -Eq "^- $p([ =><].*)?$" "$src"; then EXTRA+="\n- $p<=$ver"; fi
    done
  done <<< "$CUDA_LIBS"
  sed "${BOUNDS[@]}" \
      -e "s/^- cuda-version=.*/- cuda-version=$CUDA_VERSION/" \
      -e "/^- rapidsai-nightly$/d" \
      -e "/^- \(_go_select\|go\|maven\|nodejs\|openjdk\|rust\|clang-tools\|clang\|pre-commit\)\([ =><].*\)\?$/d" \
      -e "s/^name: .*/name: $ENV_NAME/" \
      -e "/^dependencies:/a - python=$PYTHON_VERSION$EXTRA" \
      "$src" > "$WORK/cuvs_build_env.yaml"
  cp "$WORK/cuvs_build_env.yaml" "$HERE/cuvs_build_env.generated.yaml"   # kept as part of the record
  # an existing env is reused only if it was built from this versions.env (its stamp); otherwise
  # refuse, or remove it with REBUILD_ENV=1 (a versions.env change is a new stack: build it fresh)
  if [ -d "$CONDA/envs/$ENV_NAME" ] && [ "$(cat "$STAMP" 2>/dev/null)" != "$VERSIONS_SHA" ]; then
    if [ "$REBUILD_ENV" = 1 ]; then
      log "env $ENV_NAME was built from another versions.env: removing it (REBUILD_ENV=1)"
      conda env remove -y -n "$ENV_NAME" > "$WORK/logs/conda_env_remove.log" 2>&1
      rm -rf "$CONDA/envs/$ENV_NAME"
    else
      log "env $ENV_NAME exists but was not built from this versions.env" \
          "(stamp: $(cat "$STAMP" 2>/dev/null || echo missing)); refusing to reuse it." \
          "REBUILD_ENV=1 removes and rebuilds it"
      exit 1
    fi
  fi
  if [ ! -d "$CONDA/envs/$ENV_NAME" ]; then
    conda env create -y -n "$ENV_NAME" -f "$WORK/cuvs_build_env.yaml" > "$WORK/logs/conda_env.log" 2>&1
    echo "$VERSIONS_SHA" > "$STAMP"
  fi
  conda_not_newer "conda env create"
  while read -r cpkg ppkg ver; do
    printf '%s <=%s\n%s-dev <=%s\n' "$cpkg" "$ver" "$cpkg" "$ver"
  done <<< "$CUDA_LIBS" > "$CONDA/envs/$ENV_NAME/conda-meta/pinned"
fi

if want cuvs; then
  log "STEP cuvs ($CUVS_GIT_TAG + patches)"
  verify_cuvs_patches || exit 1   # again: the step can run on its own
  activate
  cd "$WORK/cuvs"
  git checkout -q "$CUVS_GIT_TAG" && git reset -q --hard && git clean -qfd -e cpp/build -e python/cuvs/build
  for p in $CUVS_PATCHES; do
    git apply "$CUVS_PATCH_DIR/$p" || { log "git apply $p FAILED"; exit 1; }
  done
  git diff --stat | tee "$WORK/logs/cuvs_patch_stat.txt"
  git diff | sha256sum | tee "$WORK/logs/cuvs_patch_sha256.txt"
  # which patch files this env's cuVS is built with, in order: check.py compares the stamp
  # with baselines/cuvs/patches/, and every harness cuVS row records it (cuvs_patches)
  ( cd "$CUVS_PATCH_DIR" && sha256sum $CUVS_PATCHES ) > "$CONDA_PREFIX/.cuvs_patches.sha256"
  t0=$(date +%s)
  # CUVS_ENABLE_CUTILE=OFF (with patch 0002): tileiras 13.2 cannot build the
  # sm_90 cuTile fused 1-NN kernels v26.10 always asks for. They are not on any kNN path.
  PARALLEL_LEVEL=$PAR ./build.sh libcuvs python tests --limit-tests=NEIGHBORS_TEST \
      '--cmake-args="-DCUVS_ENABLE_CUTILE=OFF"' \
      > "$WORK/logs/cuvs_build.log" 2>&1 || { tail -40 "$WORK/logs/cuvs_build.log"; exit 1; }
  log "cuvs build $(( ($(date +%s) - t0) / 60 )) min"
  T=$(find cpp/build -name NEIGHBORS_TEST -type f -perm -u+x | head -1)
  "$T" --gtest_filter='*SparseKNN*' > "$WORK/logs/cuvs_gtest_sparse.log" 2>&1 \
      || { log "SparseKNN gtests FAILED"; exit 1; }
fi

if want shim; then
  log "STEP shim"
  activate
  P=$CONDA_PREFIX
  "${CXX:-g++}" -std=c++20 -O2 -fPIC -shared -fvisibility=hidden \
    -DRAFT_COMPILED -DRAFT_SYSTEM_LITTLE_ENDIAN=1 \
    -DRAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_INFO -Wno-deprecated-declarations \
    -I"$P/include" -I"$P/include/rapids" -I"$P/targets/x86_64-linux/include" \
    "$CB/baselines/cuvs/sparse_binding/cuvs_sparse_shim.cpp" -o "$CB/baselines/cuvs/sparse_binding/libnova_cuvs_sparse.so" \
    -L"$P/lib" -lcuvs -ldl -Wl,-rpath,"$P/lib" -Wl,--no-undefined -Wl,--allow-shlib-undefined \
    > "$WORK/logs/shim.log" 2>&1 || { cat "$WORK/logs/shim.log"; exit 1; }
  ldd "$CB/baselines/cuvs/sparse_binding/libnova_cuvs_sparse.so" | grep -i cuvs | tee -a "$WORK/logs/progress.log"
  # and into the env, so a job from another tree (a `bench launch` stage, which ships the shim's
  # source only) loads this build: activate.sh points NOVA_CUVS_SPARSE_LIB at it
  cp "$CB/baselines/cuvs/sparse_binding/libnova_cuvs_sparse.so" "$P/lib/libnova_cuvs_sparse.so"
fi

if want faiss; then
  log "STEP faiss ($FAISS_CONDA_SPEC)"
  activate
  # exact build string: the channel also carries *_nightly builds of the same version. The conda
  # CUDA libraries are left to the solver (FAISS's build needs e.g. libcublas >=13.3,<13.4, which
  # may move the cuVS env's libcublas): they are never loaded, only recorded
  pkgs() { conda list -n "$ENV_NAME" | awk '!/^#/ {print $1, $2, $3}' | sort; }
  pkgs > "$WORK/logs/before_faiss.txt"
  # set +u: the `conda` shell function re-activates the env after installing, and cuda-nvcc's
  # activate.d script reads an unset NVCC_PREPEND_FLAGS (with -u the step dies after a good install)
  set +u
  conda install -y -n "$ENV_NAME" -c pytorch -c conda-forge "$FAISS_CONDA_SPEC" \
      "cuda-version=$CUDA_VERSION" \
      > "$WORK/logs/faiss_install.log" 2>&1 || { set -u; tail -30 "$WORK/logs/faiss_install.log"; exit 1; }
  set -u
  pkgs > "$WORK/logs/after_faiss.txt"
  # everything the solver changed, kept as a build log
  diff "$WORK/logs/before_faiss.txt" "$WORK/logs/after_faiss.txt" \
      | tee "$WORK/logs/faiss_moved.txt" || true
  # ... but it must not have moved what the cuVS build links against (RAPIDS, cuda-version)
  moved=$(diff <(awk -v re="$FROZEN_BY_CUVS" '$1 ~ re' "$WORK/logs/before_faiss.txt") \
               <(awk -v re="$FROZEN_BY_CUVS" '$1 ~ re' "$WORK/logs/after_faiss.txt") || true)
  if [ -n "$moved" ]; then
    log "the FAISS install moved packages the cuVS build is built against:"; echo "$moved" | tee -a "$WORK/logs/progress.log"
    exit 1
  fi
  conda_not_newer "the FAISS install"
  grep -E "^($(IFS='|'; echo "${CONDA_PKGS[*]}")) " "$WORK/logs/after_faiss.txt" \
      | tee "$WORK/logs/conda_cuda_libs.txt"   # installed, never loaded: recorded only
fi

if want pip; then
  log "STEP pip"
  activate
  command -v cargo >/dev/null || { curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q; }
  . "$HOME/.cargo/env"
  PIP="python -m pip install --no-input"
  # the CUDA libraries every engine runs on. torch pins these wheels itself (cuda-toolkit==13.2.2),
  # except cuRAND, which torch does not pull and which is added at cuda-toolkit 13.2.2's own pin;
  # naming the exact builds makes any drift fail here instead of in a timing
  $PIP --index-url "$TORCH_INDEX_URL" --extra-index-url https://pypi.org/simple \
      "torch==$TORCH_VERSION" "${PIP_PINS[@]}" > "$WORK/logs/pip_torch.log" 2>&1
  $PIP "numpy==$NUMPY_VERSION" "pyarrow==$PYARROW_VERSION" "duckdb==$DUCKDB_VERSION" pyyaml \
      > "$WORK/logs/pip_basics.log" 2>&1
  $PIP -e "$REPO/python/nova-bf[compute]" "$REPO/python/nova-bf/rust_io" > "$WORK/logs/pip_novabf.log" 2>&1
  # the bench CLI (check.py refuses a `bench` outside this env); no dependencies
  $PIP --no-deps -e "$CB" > "$WORK/logs/pip_bench.log" 2>&1
  # psutil: SPARe's backend_torch imports it but its package does not declare it (without it every
  # --backend spare run fails "No module named 'psutil'")
  $PIP "$SPARE_PIP" psutil==7.2.2 > "$WORK/logs/pip_spare.log" 2>&1
  $PIP --no-deps "$PYLATE_PIP" > "$WORK/logs/pip_pylate.log" 2>&1
  # what pylate.scores / pylate.rank import (via pylate.utils) but PyLate installed with --no-deps
  # lacks: transformers + sentence-transformers at PyLate 1.6.0's own pins, and datasets.
  # Constrained to everything already installed so torch / numpy / pyarrow / the CUDA wheels cannot
  # move. PyLate's index / training extras (fast-plaid, fastkmeans, accelerate, ...) stay out: the
  # harness never imports them
  python -m pip freeze | grep -v -E '^-e |@ (file|git)' > "$WORK/pip_constraints_pylate.txt"
  $PIP -c "$WORK/pip_constraints_pylate.txt" "transformers==5.3.0" "sentence-transformers==5.3.0" \
      "datasets==5.2.0" > "$WORK/logs/pip_pylate_deps.log" 2>&1
  python -c "import pylate.scores, pylate.rank, spare.backend_torch" \
      || { log "pylate.scores / pylate.rank / spare.backend_torch do not import"; exit 1; }
  # pip must not have moved torch / numpy under the engines: check.py is the judge
  python -m pip check | tee "$WORK/logs/pip_check.txt" || true
fi

if want check; then
  log "STEP check"
  activate
  python "$HERE/check.py" --versions "$HERE/versions.env" | tee "$WORK/logs/check_env.txt"
fi

if want freeze; then
  log "STEP freeze"
  activate
  conda list -n "$ENV_NAME" --explicit --md5 > "$HERE/conda-explicit.lock.txt"
  python -m pip freeze > "$HERE/pip-freeze.lock.txt"
  python "$HERE/check.py" --versions "$HERE/versions.env" --json > "$HERE/check_env.lock.json"
  nvidia-smi --query-gpu=name,driver_version,uuid --format=csv > "$HERE/gpu.lock.txt" || true
  log "lock files written to $HERE; commit them"
fi
log "done: ${STEPS[*]}"
