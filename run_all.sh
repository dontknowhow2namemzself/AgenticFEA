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
# Every stage reads and writes ONE run directory (default out/). Each stage's
# output is tee'd to <run>/stage_<name>.log so diagnostics can read it.
#
# Flags:  --skip-checkpoints   skip the optional human-review pauses (agent auto)
#         --workdir DIR        run directory to use instead of out/ -- keeps two
#                              runs from overwriting each other's files
# Env:    PYTHON=...            interpreter to use (default: python from the env)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/out"
SKIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-checkpoints) SKIP=1 ;;
    --workdir)          [ $# -ge 2 ] || { echo "ERROR: --workdir needs a directory"; exit 2; }
                        OUT="$2"; shift ;;
    --workdir=*)        OUT="${1#--workdir=}" ;;
    *)                  echo "ERROR: unknown argument $1"; exit 2 ;;
  esac
  shift
done
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"     # absolute, so every stage sees the same directory
cd "$ROOT"

# The active geometry (params.ACTIVE_CASE) decides which renders the run
# produces, so the closing summary asks params.py rather than assuming a beam.
GEOM="$("$PY" -c 'from params import PARAMS; print(PARAMS["geometry"]["type"])' 2>/dev/null || echo unknown)"

step()       { echo; echo "########## $1 ##########"; }
checkpoint() {  # $1 = what to inspect
  if [ "$SKIP" -eq 1 ]; then
    echo ">> checkpoint skipped (--skip-checkpoints): $1"
  else
    echo ">> CHECKPOINT: $1"
    read -r -p ">> press Enter to continue (Ctrl-C to stop)... " _
  fi
}
run_stage() {  # $1 = label, $2.. = command;  tees output to <run>/stage_<label>.log
  local label="$1"; shift
  step "stage: $label"
  "$@" 2>&1 | tee "$OUT/stage_${label}.log"
  return "${PIPESTATUS[0]}"
}
diagnose() {  # $1 = stage label -> layer-8 Human Review Report
  echo
  echo "########## stage '$1' FAILED -> diagnostics (layer 8) ##########"
  "$PY" diagnostics.py --stage "$1" --workdir "$OUT" --log "$OUT/stage_$1.log" \
    || echo "(diagnostics agent could not run; inspect $OUT/stage_$1.log and the reports)"
}

# --- pre-processing ---
run_stage mesh "$PY" mesh.py --workdir "$OUT"               || { diagnose mesh; exit 1; }
checkpoint "mesh: gmsh $OUT/beam.msh"
run_stage bc "$PY" boundary_conditions.py --workdir "$OUT"  || { diagnose bc; exit 1; }
checkpoint "BC mapping (FIXED / LOAD)"
run_stage inp "$PY" build_inp.py --workdir "$OUT"           || { diagnose inp; exit 1; }

# --- solve ---
run_stage solve bash solve.sh "$OUT"                        || { diagnose solve; exit 1; }

# --- physics validation: the fast-fail gate ---
if run_stage physics "$PY" physics_validation.py --workdir "$OUT"; then
  echo ">> physics PASSED -> proceed to postprocess"
else
  echo ">> physics FAILED (Critical) -> SKIP postprocess (the expensive stage)"
  diagnose physics
  exit 1
fi

# --- postprocess (reached only when the gate passes) ---
run_stage postprocess "$PY" postprocess.py --workdir "$OUT" || { diagnose postprocess; exit 1; }
checkpoint "results: open $OUT/beam.vtu in ParaView"

# --- numerical verification ---
if run_stage verification "$PY" verification.py --workdir "$OUT"; then
  echo ">> verification PASSED"
else
  echo ">> verification FAILED"
  diagnose verification
  exit 1
fi

echo
echo "=================================================================="
echo "PIPELINE COMPLETE: all stages PASSED.   (geometry: $GEOM)"
echo "run directory: $OUT"
echo "      beam.msh  beam.inp  beam.frd  beam.vtu"
if [ "$GEOM" = "plate_with_hole" ]; then
  echo "      stress_xx.png  stress_yy.png  stress_xy.png  stress_vm.png"
  echo "      displacement.png"
else
  echo "      stress_x.png  stress_y.png  stress_z.png  displacement.png"
fi
echo "      viewer.npz  (interactive 3D view in the dashboard)"
echo "      physics_report.txt  verification_report.txt"
echo "=================================================================="
