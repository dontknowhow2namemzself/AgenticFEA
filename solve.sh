#!/usr/bin/env bash
# Solver layer 4 (plan section 4): run CalculiX ccx on out/beam.inp.
#
# Threads: OMP_NUM_THREADS defaults to the number of Apple-silicon PERFORMANCE
# cores (hw.perflevel0.physicalcpu). FEM solves are barrier-heavy; spilling onto
# the slower efficiency cores makes them the straggler and hurts throughput, so
# we fill the P-cores only. A caller-provided OMP_NUM_THREADS is respected.
#
# Output: out/beam.frd (U / RF / S), out/beam.dat, out/beam.sta, out/ccx.log.
# Acceptance (plan layer 4): exit 0, no *ERROR in the log, frd has the result
# blocks. The frd also carries an ERROR block = CalculiX stress error estimator,
# which is informational, not a failure.
set -euo pipefail

# Project root = this script's directory (works from any CWD).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/out"
JOB="$OUT/beam"          # ccx -i <job> reads <job>.inp, writes <job>.frd, ...
LOG="$OUT/ccx.log"

# Preconditions.
command -v ccx >/dev/null 2>&1 || { echo "ERROR: ccx not on PATH (activate the fea env)"; exit 1; }
[ -f "$JOB.inp" ] || { echo "ERROR: $JOB.inp not found; run build_inp.py first"; exit 1; }

# Thread count = performance cores unless the caller already set OMP_NUM_THREADS.
PCORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)"
ECORES="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo '?')"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$PCORES}"
export CCX_NPROC_EQUATION_SOLVER="${CCX_NPROC_EQUATION_SOLVER:-$OMP_NUM_THREADS}"

echo "=================================================================="
echo "solve.sh (plan layer 4): CalculiX ccx"
echo "=================================================================="
echo "job             : $JOB.inp"
echo "P-cores/E-cores : $PCORES/$ECORES   (filling performance cores only)"
echo "OMP_NUM_THREADS : $OMP_NUM_THREADS   CCX_NPROC_EQUATION_SOLVER: $CCX_NPROC_EQUATION_SOLVER"
echo "ccx             : $(command -v ccx)"
echo "------------------------------------------------------------------"

# Run the solve; capture the exit code without aborting (it is an acceptance gate).
set +e
ccx -i "$JOB" > "$LOG" 2>&1
RC=$?
set -e
echo "ccx exit code   : $RC"
echo "log             : $LOG"

# Deterministic acceptance gates (plan layer 4).
PASS=1
check () {  # $1 = name, $2 = 1(pass)/0(fail)
  if [ "$2" -eq 1 ]; then echo "  [PASS] $1"; else echo "  [FAIL] $1"; PASS=0; fi
}

[ "$RC" -eq 0 ] && C_EXIT=1 || C_EXIT=0
grep -aq '\*ERROR' "$LOG" && C_ERR=0 || C_ERR=1            # CalculiX marker is *ERROR
grep -aq 'Job finished' "$LOG" && C_FIN=1 || C_FIN=0
[ -s "$JOB.frd" ] && C_FRD=1 || C_FRD=0
if [ -f "$JOB.frd" ] \
   && grep -aq -- '-4  DISP' "$JOB.frd" \
   && grep -aq -- '-4  STRESS' "$JOB.frd" \
   && grep -aq -- '-4  FORC' "$JOB.frd"; then C_BLK=1; else C_BLK=0; fi

echo "------------------------------------------------------------------"
check "ccx exit code == 0" "$C_EXIT"
check "no *ERROR in ccx log" "$C_ERR"
check "ccx reported 'Job finished'" "$C_FIN"
check "out/beam.frd exists and non-empty" "$C_FRD"
check "frd has U(DISP) / S(STRESS) / RF(FORC) blocks" "$C_BLK"
echo "------------------------------------------------------------------"
if [ "$PASS" -eq 1 ]; then
  echo "ACCEPTANCE: PASS"
  echo "Note: frd also has an ERROR block = CalculiX stress error estimator (informational)."
  echo "Next: physics validation (layer 5) reads out/beam.frd."
  echo "=================================================================="
  exit 0
else
  echo "ACCEPTANCE: FAIL  (inspect $LOG; the diagnostics layer would triage this)"
  echo "=================================================================="
  exit 1
fi
