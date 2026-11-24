#!/usr/bin/env bash
# Run a benchmark job with the pinned environment and reproducibility checks.
#
# Usage: bash clean_benchmarks/env/paper_run.sh [--run-dir DIR] [--] CMD [ARGS...]
#
# Before execution:
#   - Activates the pinned environment and refuses version mismatches.
#   - Validates the installed and loaded libraries; aborts if checks fail.
#   - Hashes the actual code on disk, including uncommitted changes.
#   - Records the environment and initial GPU state.
#   - Exports PAPER_* metadata for engines to verify and record.
#
# Optionally logs GPU metrics when PAPER_SMI_MS is set.
# On exit, stops logging, records final GPU state, and returns CMD's exit status.
#
# Records go to PAPER_RUN_DIR (--run-dir), or a timestamped directory by default.
# PAPER_CODE_ROOTS overrides which code paths are hashed.


set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CB="$(cd "$HERE/.." && pwd)"   # clean_benchmarks (or a stage: a copy of it)
RUN_DIR="${PAPER_RUN_DIR:-}"
while [ $# -gt 0 ]; do
  case $1 in
    --run-dir) RUN_DIR=$2; shift 2 ;;
    --run-dir=*) RUN_DIR=${1#*=}; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: paper_run.sh [--run-dir DIR] [--] CMD [ARGS...]" >&2; exit 2; }
RUN_DIR="${RUN_DIR:-$PWD/paper_run_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR" && RUN_DIR="$(cd "$RUN_DIR" && pwd)" || { echo "paper_run.sh: cannot create $RUN_DIR" >&2; exit 2; }
say() { echo "paper_run.sh: $*" | tee -a "$RUN_DIR/launch.log" >&2; }
# nothing of an outer job's (or a hand-set) stack identity survives into this one
unset PAPER_STACK_SHA PAPER_CODE_SHA PAPER_ENV_PREFIX PAPER_LAUNCH_STUBBED PAPER_GPU_LOG

# 1. the env
ACTIVATE="${PAPER_ACTIVATE:-$HERE/activate.sh}"
CHECK="${PAPER_CHECK_ENV:-$HERE/check.py}"
STUBBED=0
if [ "$ACTIVATE" != "$HERE/activate.sh" ] || [ "$CHECK" != "$HERE/check.py" ]; then
  STUBBED=1
  say "WARNING: STUBBED launch (PAPER_ACTIVATE / PAPER_CHECK_ENV set): NOT a paper-stack job"
fi
# shellcheck disable=SC1090
. "$ACTIVATE" || { say "activate failed ($ACTIVATE): not the pinned env"; exit 3; }
[ -n "${CONDA_PREFIX:-}" ] || { say "no CONDA_PREFIX after activation"; exit 3; }
command -v python >/dev/null || { say "no python after activation"; exit 3; }

# 2. the stack check, in the env every engine runs in
say "check.py -> $RUN_DIR/check_env.json"
python "$CHECK" --versions "$HERE/versions.env" --json "$RUN_DIR/check_env.json" \
    ${PAPER_CHECK_ENGINES:+--engines $PAPER_CHECK_ENGINES} > "$RUN_DIR/check_env.txt" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  tail -40 "$RUN_DIR/check_env.txt" >&2
  say "check.py FAILED (exit $rc; $RUN_DIR/check_env.txt): not the pinned stack, NOT running: $*"
  exit 3
fi

# 3. what the job runs on, for every process of it
# versions.env's assignments, comments and STACK_FROZEN_ON left out: activate.sh's / build.sh's hash
STACK_SHA="$(grep -v '^STACK_FROZEN_ON=' "$HERE/versions.env" | sed 's/[[:space:]]*#.*$//' \
             | grep -v '^[[:space:]]*$' | sha256sum | cut -d' ' -f1)"
if [ -n "${PAPER_CODE_ROOTS:-}" ]; then
  read -ra ROOTS <<< "$PAPER_CODE_ROOTS"; BASE="$CB"
else
  # code dirs called data/ (code_sha.py never walks into one) and the test fixtures: named as
  # roots of their own. bench/launch.py DATA_CODE_DIRS is the same list (the stage's STAGED
  # hash; a test checks it).
  DATA_CODE_DIRS="bench/data tests/data tests/fixtures"
  if [ -d "$CB/../python/nova-bf" ]; then   # a repo checkout
    BASE="$(cd "$CB/.." && pwd)"
    ROOTS=()
    for p in bench env baselines experiments tests $DATA_CODE_DIRS; do
      [ -e "$CB/$p" ] && ROOTS+=("$(cd "$CB/$p" && pwd)")
    done
    ROOTS+=("$BASE/python/nova-bf")
    # bench's own package file, and for a filtered experiment the text scanner nova-bf's
    # filtered runs use (a stage ships it as nova_textscan/ + the root Cargo.lock / Cargo.toml:
    # launch.py TEXTSCAN_PATHS). The modality is the resolved experiment's (read through its
    # base: chain; a sweep file names none itself)
    [ -f "$CB/pyproject.toml" ] && ROOTS+=("$CB/pyproject.toml")
    for a in "$@"; do
      case "$a" in *.yaml|*.yml) ;; *) continue ;; esac
      [ -f "$a" ] || continue
      if [ "$(cd "$CB" && python -m bench.experiment "$(cd "$(dirname "$a")" && pwd)/$(basename "$a")" \
              --get modality 2>/dev/null)" = filtered ] && [ -d "$BASE/crates/nova-textscan" ]; then
        ROOTS+=("$BASE/crates/nova-textscan")
        for f in Cargo.lock Cargo.toml; do [ -f "$BASE/$f" ] && ROOTS+=("$BASE/$f"); done
        break
      fi
    done
  else                                      # a stage (bench launch): all of it + its data-named code dirs
    BASE="$CB"; ROOTS=("$CB")
    for p in $DATA_CODE_DIRS; do [ -d "$CB/$p" ] && ROOTS+=("$CB/$p"); done
  fi
fi
read -ra EXTRA <<< "${PAPER_CODE_EXTRA:-}"
ROOTS+=("${EXTRA[@]}")
for a in "$@"; do [ -f "$a" ] && ROOTS+=("$a"); done   # the job's own script / config
CODE_SHA="$(python "$HERE/code_sha.py" --base "$BASE" --json "$RUN_DIR/code.json" "${ROOTS[@]}")" \
  || { say "code_sha.py failed"; exit 3; }
if [ -f "$CB/STAGED" ]; then   # a stage records the hash it was built with
  staged="$(sed -n 's/^code sha256: //p' "$CB/STAGED")"
  if [ -n "$staged" ] && [ "$staged" != "$CODE_SHA" ]; then
    say "WARNING: code sha $CODE_SHA differs from the stage's ($staged, STAGED): the code was changed after staging"
  fi
fi
export PAPER_STACK_SHA="$STACK_SHA" PAPER_CODE_SHA="$CODE_SHA" PAPER_ENV_PREFIX="$CONDA_PREFIX" \
       PAPER_RUN_DIR="$RUN_DIR"
[ $STUBBED = 1 ] && export PAPER_LAUNCH_STUBBED=1
{
  echo "date_utc=$(date -u +%FT%TZ)"; echo "host=$(hostname)"; echo "stubbed=$STUBBED"
  printf 'cmd='; printf '%q ' "$@"; echo
  echo "PAPER_STACK_SHA=$PAPER_STACK_SHA"; echo "PAPER_CODE_SHA=$PAPER_CODE_SHA"
  echo "PAPER_ENV_PREFIX=$PAPER_ENV_PREFIX"; echo "PAPER_RUN_DIR=$PAPER_RUN_DIR"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  env | grep -E '^(OMP_|KMP_|MKL_|GOMP_|OPENBLAS_|CUDA_|PYTORCH_|TORCH_|CUBLAS|NVIDIA_TF32|NOVA_|NOVABF_)' | sort
} > "$RUN_DIR/launch.env"
say "stack $PAPER_STACK_SHA, code $PAPER_CODE_SHA, env $PAPER_ENV_PREFIX"

# 4.-5. the GPU: state at start and end, and (PAPER_SMI_MS set only) a log for the whole job
SMI_PID=""; STATE_Q=""; LOG_Q=""
smi_ok() { nvidia-smi --query-gpu="$1" --format=csv,noheader > /dev/null 2>&1; }
smi_fields() {  # the fields this driver accepts, in order (one unknown name fails a whole query)
  local f out=()
  for f in "$@"; do
    case $f in   # renamed in newer drivers: either spelling
      *throttle*|*event_reasons*) smi_ok clocks_event_reasons.active && f=clocks_event_reasons.active \
                                  || f=clocks_throttle_reasons.active ;;
    esac
    smi_ok "$f" && out+=("$f")
  done
  local IFS=,; echo "${out[*]}"
}
STATE_FIELDS=(timestamp index uuid name pci.bus_id driver_version vbios_version persistence_mode
  ecc.mode.current ecc.mode.pending compute_mode mig.mode.current
  clocks.applications.graphics clocks.applications.memory
  clocks.default_applications.graphics clocks.default_applications.memory clocks.max.sm
  clocks.max.mem power.limit power.default_limit enforced.power.limit pcie.link.gen.max
  pcie.link.width.max pcie.link.gen.current pcie.link.width.current temperature.gpu
  clocks_event_reasons.active retired_pages.pending)
# these five columns first, in this order (timestamp,index,uuid,memory.used,utilization.gpu)
LOG_FIELDS=(timestamp index uuid memory.used utilization.gpu utilization.memory clocks.sm
  clocks.mem clocks.gr power.draw temperature.gpu temperature.memory clocks_event_reasons.active
  pcie.link.gen.current pcie.link.width.current pstate)
gpu_state() {  # gpu_state start|end
  [ -n "$STATE_Q" ] && nvidia-smi --query-gpu="$STATE_Q" --format=csv > "$RUN_DIR/gpu_state_$1.csv" 2>&1
  nvidia-smi -q > "$RUN_DIR/nvidia_smi_q_$1.txt" 2>&1
}
cleanup() {
  if [ -n "$SMI_PID" ]; then kill "$SMI_PID" 2>/dev/null; wait "$SMI_PID" 2>/dev/null; fi
  command -v nvidia-smi > /dev/null && gpu_state end
}
if command -v nvidia-smi > /dev/null; then
  STATE_Q="$(smi_fields "${STATE_FIELDS[@]}")"
  [ -n "${PAPER_SMI_MS:-}" ] && LOG_Q="$(smi_fields "${LOG_FIELDS[@]}")"
  gpu_state start
  if [ -n "$LOG_Q" ]; then
    nvidia-smi --query-gpu="$LOG_Q" --format=csv -lms "$PAPER_SMI_MS" > "$RUN_DIR/gpu_smi.csv" 2>&1 &
    SMI_PID=$!
    export PAPER_GPU_LOG="$RUN_DIR/gpu_smi.csv"
  fi
else
  say "WARNING: no nvidia-smi: no GPU state or GPU log recorded"
fi
trap cleanup EXIT

# CMD, as a child (so the logger outlives it and stops after it)
child=""
forward() { [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; }
trap forward TERM INT HUP
say "running: $*"
"$@" &
child=$!
rc=0
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
wait "$child" 2>/dev/null; r=$?; [ $r -ne 127 ] && rc=$r
say "exit $rc: $*"
exit $rc
