#!/bin/bash
#SBATCH --job-name=Probe_SSE
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/probe_sse_%j.out
#SBATCH --error=logs/probe_sse_%j.err
#SBATCH --time=00:30:00

# Runtime probe for the finite-temperature SSE engine: times engine init +
# a handful of mc_steps at the PRODUCTION geometry (after M has grown to its
# steady state) and extrapolates to the full equil + sample budget.  Mirrors
# run_kagome_sse.sh defaults (triangle lattice) so the estimate is realistic.

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source scripts/common/env.sh

mkdir -p logs

# ── Probe target: mirror run_kagome_sse.sh production defaults ───────────────
LATTICE=${LATTICE:-kagome_bond_triangle}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
OMEGA=${OMEGA:-1.0}
DELTA=${DELTA:-3.0}
RB=${RB:-2.4}
BETA=${BETA:-16.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}

# Production budget being estimated
TARGET_EQUIL=${TARGET_EQUIL:-20000}
TARGET_SAMPLES=${TARGET_SAMPLES:-2000000}

# Probe scale (small; just enough to reach steady-state M and time mc_step)
PROBE_WARMUP=${PROBE_WARMUP:-300}
PROBE_STEPS=${PROBE_STEPS:-200}

echo "=== Probing SSE runtime ==="
echo "Node: $NODE_DESC, ranks=$NTASKS"
echo "Geometry: $LATTICE ${NX}x${NY}, beta=${BETA}, delta=${DELTA}"
echo "Estimating: equil=${TARGET_EQUIL}, samples=${TARGET_SAMPLES} (per rank)"
echo

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC python -u scripts/probe/probe_runtime_sse.py \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --Omega "$OMEGA" --delta "$DELTA" --Rb "$RB" --beta "$BETA" \
    --epsilon "$EPSILON" --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --probe-warmup "$PROBE_WARMUP" --probe-steps "$PROBE_STEPS" \
    --target-equil "$TARGET_EQUIL" --target-samples "$TARGET_SAMPLES"

echo "=== probe done ==="
