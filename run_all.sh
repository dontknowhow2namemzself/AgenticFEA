#!/usr/bin/env bash
# Orchestrator (plan "run_all.sh"): the deterministic chain end to end --
#   mesh -> BC -> inp -> solve -> physics validation -> [gate] -> postprocess
#   -> numerical verification
#
# Physics validation (layer 5) is the cheap FAST-FAIL GATE: on a Critical FAIL
# the expensive postprocess is SKIPPED and the run goes to diagnostics. Any stage
# failing stops the pipeline and invokes the layer-8 diagnostics agent on that
# stage's log + reports (a Human Review Report, then a human decides).
#
# Each stage's output is tee'd to out/stage_<name>.log so diagnostics can read it.
#
# Flags:  --skip-checkpoints   skip the optional human-review pauses (agent auto)
# Env:    PYTHON=...            interpreter to use (default: python from the env)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/out"
SKIP=0
for arg in "$@"; do
  [ "$arg" = "--skip-checkpoints" ] && SKIP=1
done
mkdir -p "$OUT"
cd "$ROOT"

step()       { echo; echo "########## $1 ##########"; }
checkpoint() {  # $1 = what to inspect
  if [ "$SKIP" -eq 1 ]; then
    echo ">> checkpoint skipped (--skip-checkpoints): $1"
  else
    echo ">> CHECKPOINT: $1"
    read -r -p ">> press Enter to continue (Ctrl-C to stop)... " _
  fi
}
run_stage() {  # $1 = label, $2.. = command;  tees output to out/stage_<label>.log
  local label="$1"; shift
  step "stage: $label"
  "$@" 2>&1 | tee "$OUT/stage_${label}.log"
  return "${PIPESTATUS[0]}"
}
diagnose() {  # $1 = stage label -> layer-8 Human Review Report
  echo
  echo "########## stage '$1' FAILED -> diagnostics (layer 8) ##########"
  "$PY" diagnostics.py --stage "$1" --log "$OUT/stage_$1.log" \
    || echo "(diagnostics agent could not run; inspect $OUT/stage_$1.log and the reports)"
}

# --- pre-processing ---
run_stage mesh "$PY" mesh.py                 || { diagnose mesh; exit 1; }
checkpoint "mesh: gmsh out/beam.msh"
run_stage bc "$PY" boundary_conditions.py    || { diagnose bc; exit 1; }
checkpoint "BC mapping (FIXED / LOAD)"
run_stage inp "$PY" build_inp.py             || { diagnose inp; exit 1; }

# --- solve ---
run_stage solve bash solve.sh                || { diagnose solve; exit 1; }

# --- physics validation: the fast-fail gate ---
if run_stage physics "$PY" physics_validation.py; then
  echo ">> physics PASSED -> proceed to postprocess"
else
  echo ">> physics FAILED (Critical) -> SKIP postprocess (the expensive stage)"
  diagnose physics
  exit 1
fi

# --- postprocess (reached only when the gate passes) ---
run_stage postprocess "$PY" postprocess.py   || { diagnose postprocess; exit 1; }
checkpoint "results: open out/beam.vtu in ParaView"

# --- numerical verification ---
if run_stage verification "$PY" verification.py; then
  echo ">> verification PASSED"
else
  echo ">> verification FAILED"
  diagnose verification
  exit 1
fi

echo
echo "=================================================================="
echo "PIPELINE COMPLETE: all stages PASSED."
echo "out/: beam.msh  beam.inp  beam.frd  beam.vtu"
echo "      stress_x.png  stress_y.png  stress_z.png  displacement.png"
echo "      physics_report.txt  verification_report.txt"
echo "=================================================================="
