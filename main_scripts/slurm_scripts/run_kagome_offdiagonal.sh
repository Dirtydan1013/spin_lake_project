#!/bin/bash
#SBATCH --job-name=SSE_OffDiag
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=240gb
#SBATCH --output=logs/sse_offdiag_%j.out
#SBATCH --error=logs/sse_offdiag_%j.err
#SBATCH --time=08:00:00

# ─── Thermal SSE off-diagonal string-work (Jarzynski) production run ─────────
#
# Computes  O_C(beta) = Tr[X_C e^{-beta H}] / Tr[e^{-beta H}]  for an X-string
# / X-loop via the periodic-tau seam + half-line-toggle Jarzynski estimator
# (SSEEngine string-seam primitives, driver src.mpi.sse_string_work_mpi).
# Physics: a classical (diagonal) ensemble has O_C == 0 identically for any
# closed X-loop — a resolved non-zero value is a direct coherence witness,
# and O_C(l) vs loop size l grades perimeter-law quality (E14).
#
# Same conventions as run_kagome_string_work.sh (its QAQMC sibling):
# incremental checkpointing per CKPT_TRAJ trajectories, warm start via
# CONFIG_IN (must match N/beta/delta/boundary — SSE configs are
# temperature-specific), n_eff / zero_frac convergence diagnostics in the log.

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source main_scripts/common/env.sh

mkdir -p logs data

# ─── Tunables ────────────────────────────────────────────────────────────────
LATTICE=${LATTICE:-kagome_bond}
# open (finite patch) or periodic (torus, kagome_bond only).
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
RB=${RB:-2.4}
DELTA=${DELTA:-4.25}
BETA=${BETA:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
K_VALUES=${K_VALUES:-"200"}
SCHEDULE=${SCHEDULE:-cosine}
DIRECTION=${DIRECTION:-forward}
N_TRAJ=${N_TRAJ:-4000}
N_THERMALIZE=${N_THERMALIZE:-5000}
EQUIL_PRINT_EVERY=${EQUIL_PRINT_EVERY:-500}
DECORR=${DECORR:-50}
SEED=${SEED:-7}
CKPT_TRAJ=${CKPT_TRAJ:-250}
CONFIG_IN=${CONFIG_IN:-""}
CONFIG_OUT=${CONFIG_OUT:-""}
PERMUTE_SITES=${PERMUTE_SITES:-1}

# String selection: explicit STRING_SITES ("s0,s1,..."), or STRING_SIZE to
# auto-pick the most central size-s string/loop copy on the lattice.
# LOOP=1 selects a closed Z_l-geometry loop (the X-loop coherence witness);
# LOOP=0 selects an open C_m-geometry string.
STRING_SITES=${STRING_SITES:-""}
STRING_SIZE=${STRING_SIZE:-2}
LOOP=${LOOP:-1}

if [ -z "$STRING_SITES" ]; then
    STRING_SITES=$(python - <<PY
import numpy as np
nx, ny, s, want_loop = $NX, $NY, $STRING_SIZE, $LOOP
lattice = "$LATTICE"
if lattice == "kagome_bond_triangle":
    from src.rydberg.lattices import (generate_kagome_bond_triangle_lattice,
                                      kagome_triangle_multi_size_translations)
    pos = generate_kagome_bond_triangle_lattice(nx, ny, 1.0)
    lsets, ssets, _, _ = kagome_triangle_multi_size_translations(
        nx, ny, loop_sizes=(s,) if want_loop else (),
        string_sizes=() if want_loop else (s,))
else:
    from src.rydberg.lattices import (generate_kagome_bond_lattice,
                                      kagome_loop_string_translations)
    pos = generate_kagome_bond_lattice(nx, ny, 1.0)
    lsets, ssets = kagome_loop_string_translations(
        nx, ny, loop_size=s if want_loop else 2,
        string_size=2 if want_loop else s)
sets = lsets if want_loop else ssets
centre = pos.mean(axis=0)
best = min(sets, key=lambda st: np.linalg.norm(pos[st].mean(axis=0) - centre))
print(",".join(str(x) for x in best))
PY
)
    KIND=$( [ "$LOOP" = "1" ] && echo loop || echo string )
    echo "Auto-selected central size-${STRING_SIZE} ${KIND}: sites ${STRING_SITES}"
fi

OUT_NAME=${OUT_NAME:-"sse_offdiag_${LATTICE}_${NX}x${NY}_beta${BETA}_delta${DELTA}_K${K_VALUES//,/-}_n${N_TRAJ}_${JOB_TAG}.h5"}
FILEPATH="data/${OUT_NAME}"

echo "Thermal SSE off-diagonal string-work production run"
echo "Node: $NODE_DESC, ranks=$NTASKS, omp_threads/rank=$CPT"
echo "Geometry: $LATTICE ${NX}x${NY} ($BOUNDARY), a=$A_LAT; string sites: $STRING_SITES"
echo "SSE: beta=$BETA, delta=$DELTA, Rb=$RB"
echo "Protocol: K=$K_VALUES ($SCHEDULE, $DIRECTION), n_traj=$N_TRAJ, "
echo "          thermalize=$N_THERMALIZE, decorr=$DECORR, ckpt every $CKPT_TRAJ traj"
echo "Output: $FILEPATH"
echo

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC \
    python -u -m src.mpi.sse_string_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --Omega 1.0 --Rb "$RB" \
    --delta "$DELTA" --beta "$BETA" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --string-sites "$STRING_SITES" \
    --K-values "$K_VALUES" \
    --schedule "$SCHEDULE" \
    --direction "$DIRECTION" \
    --n-trajectories "$N_TRAJ" \
    --n-thermalize "$N_THERMALIZE" \
    --equil-progress-every "$EQUIL_PRINT_EVERY" \
    --decorrelation-steps "$DECORR" \
    --seed "$SEED" \
    --checkpoint-every-trajectories "$CKPT_TRAJ" \
    ${CONFIG_IN:+--config-in "$CONFIG_IN"} \
    ${CONFIG_OUT:+--config-out "$CONFIG_OUT"} \
    $( [ "$PERMUTE_SITES" = "1" ] || printf %s --no-permute-site-labels ) \
    --filepath "$FILEPATH"

echo
echo "Finished: $(date --iso-8601=seconds)"
