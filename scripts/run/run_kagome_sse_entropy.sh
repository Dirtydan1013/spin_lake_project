#!/bin/bash
#SBATCH --job-name=SSEent6x6
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/sse_entropy_6x6_%j.out
#SBATCH --error=logs/sse_entropy_6x6_%j.err

# ─── SSE thermal-entropy β-ladder (Wang–Pollet PRL Eq. 7) ───────────────────
#
# Ascending log β-ladder of E(β) → S(T)/N by thermodynamic integration
#   S(β) = N ln2 − β0 E_∞ + ½β0² varH − ∫_{β0}^{β} E dβ' + β E(β)
# with the analytic β→0 anchor (E_∞, varH exact) applied at the hottest rung.
# Each rung warm-starts (in memory) from the previous one — the op-string is
# valid at any β — so equilibration is paid once at the hot end.  Per-rung
# output is the standard chunked format  <RUN_DIR>/beta_{k:03d}/rank{r}.h5,
# plus a top-level meta.h5 (betas + anchor).  Analyse with
#   python plots/plot_sse/plot_entropy.py --run_dir <RUN_DIR>
#
# ⚠ Two error sources decide accuracy (see docs/progress/experiments/E13 + memory):
#   1. STATISTICAL — dominated by the endpoint term β·SEM(E)/N (largest at
#      β_max).  Scale N_SAMPLES to shrink it (∝ 1/√N_SAMPLES).
#   2. TRAPEZOID DISCRETISATION — the ∫E dβ' quadrature is NOT converged at
#      ~40 rungs (bias ~0.008); use a DENSE ladder (N_BETA≈72–80 for β_max=20)
#      and always run the rung-decimation convergence check in analysis.
# NEVER `cp` over the live qaqmc_cpp .so while this runs — deploy with `mv`.
#
# Submit (from repo root):
#   ./scripts/submit.sh scripts/run/run_kagome_sse_entropy.sh
#   EXCLUSIVE=1 ./scripts/submit.sh scripts/run/run_kagome_sse_entropy.sh
# Override any tunable via env, e.g.  N_SAMPLES=2560000 N_BETA=80 DELTA=4.25 ...

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>`.  Run from the repo root.
source scripts/common/env.sh

mkdir -p logs data

# ─── Tunables ────────────────────────────────────────────────────────────────
LATTICE=${LATTICE:-kagome_bond}
# periodic (torus) is only valid for kagome_bond; the triangle patch errors.
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
OMEGA=${OMEGA:-1.0}
DELTA=${DELTA:-4.5}
RB=${RB:-2.4}
EPSILON=${EPSILON:-0.01}
SEED=${SEED:-42}
# Ladder: geometric grid β∈[BETA_MIN, BETA_MAX] with N_BETA rungs.  β_min must
# be tiny so the analytic high-T anchor (O(β²)) is exact there.
BETA_MIN=${BETA_MIN:-3e-4}
BETA_MAX=${BETA_MAX:-20}
N_BETA=${N_BETA:-72}
# Equilibration: once at the hottest rung, then a short re-equil per warm rung.
N_EQUIL0=${N_EQUIL0:-4000}
N_EQUIL_WARM=${N_EQUIL_WARM:-1000}
# TOTAL samples per rung across ranks (∝ 1/√ statistical error).  cpunode02 has
# 256 GB; memory is set by the op-string length (∝β·N), not N_SAMPLES, so this
# can be pushed hard.
N_SAMPLES=${N_SAMPLES:-1280000}
CHECKPOINT=${CHECKPOINT:-2000}
RUN_DIR=${RUN_DIR:-"data/sse_entropy_${NX}x${NY}_delta${DELTA}_prod"}

echo "SSE entropy ladder ${NX}x${NY} ${LATTICE} δ=${DELTA} ${BOUNDARY}"
echo "Node(s): $NODE_DESC   MPI tasks: $NTASKS (omp/rank=$CPT)"
echo "Ladder: ${N_BETA} rungs β∈[${BETA_MIN},${BETA_MAX}], ${N_SAMPLES} samples/rung"
echo "Run dir: $RUN_DIR"

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC \
    python -m src.mpi.sse_entropy_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" \
    --a "$A_LAT" \
    --Omega "$OMEGA" \
    --delta "$DELTA" \
    --Rb "$RB" \
    --epsilon "$EPSILON" \
    --boundary "$BOUNDARY" \
    --seed "$SEED" \
    --beta-min "$BETA_MIN" \
    --beta-max "$BETA_MAX" \
    --n-beta "$N_BETA" \
    --n-equil0 "$N_EQUIL0" \
    --n-equil-warm "$N_EQUIL_WARM" \
    --n-samples "$N_SAMPLES" \
    --checkpoint "$CHECKPOINT" \
    --run-dir "$RUN_DIR"
    # NOTES:
    #   - full 1/r⁶ (neighbor_cutoff=-1) is fixed inside the driver.
    #   - only E(β) is measured, so per-rank site-label permutation is
    #     irrelevant here (energy is permutation-invariant).
    #   - reuse a high-stat single-β run (e.g. the β=20 sse_mpi run) as the
    #     endpoint in analysis to shrink the β_max error bar (mixed scheme).

echo "Entropy ladder finished successfully!"
