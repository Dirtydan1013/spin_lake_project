#!/bin/bash
#SBATCH --job-name=StrWork_Kag
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=240gb
#SBATCH --output=logs/string_work_%j.out
#SBATCH --error=logs/string_work_%j.err
#SBATCH --time=08:00:00

# ─── Off-diagonal string-work (Jarzynski) production run ────────────────────
#
# Computes O_C = Z_C / Z_empty for a C_m string via the half-line-toggle
# Jarzynski estimator (QAQMCEngine string-seam primitives, driver
# src.mpi.qaqmc_string_work_mpi).  Default lattice: kagome_bond_triangle
# (cropped patch; every atom in a complete blockade triangle).  Set
# LATTICE=kagome_bond for the plain bond lattice.
#
# Incremental checkpointing flushes each rank's log_J samples every
# CKPT_TRAJ trajectories to <filepath minus .h5>_chunks/K{K}/rank{r}/
# chunk{c}.h5, so a crash loses at most one chunk per rank.
#
# Convergence diagnostics to watch in the log: n_eff/n_traj (Jarzynski
# effective sample size) and zero_frac (trajectories ending with J=0 —
# refine the λ schedule / raise K if it is large).

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source main_scripts/common/env.sh

mkdir -p logs data

# ─── Tunables ────────────────────────────────────────────────────────────────
LATTICE=${LATTICE:-kagome_bond_triangle}
# Spatial lattice boundary: open (finite patch) or periodic (torus, kagome_bond
# only).  The driver errors on the cropped triangle + periodic.
BOUNDARY=${BOUNDARY:-open}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-100000}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
DELTA_GROUPS=${DELTA_GROUPS:-600}
K_VALUES=${K_VALUES:-"200"}
SCHEDULE=${SCHEDULE:-cosine}
DIRECTION=${DIRECTION:-forward}
N_TRAJ=${N_TRAJ:-4000}
N_THERMALIZE=${N_THERMALIZE:-5000}
# Print rank-0 thermalization progress every this many steps (<= 0 disables).
EQUIL_PRINT_EVERY=${EQUIL_PRINT_EVERY:-500}
DECORR=${DECORR:-100}
SEED=${SEED:-7}
CKPT_TRAJ=${CKPT_TRAJ:-250}
# Warm start: CONFIG_IN = previous run's final-config dir (rank{r}.h5) —
# thermalization is skipped.  Final configs saved to CONFIG_OUT
# (default <filepath>_configs).
CONFIG_IN=${CONFIG_IN:-""}
CONFIG_OUT=${CONFIG_OUT:-""}

# String selection.  Either give explicit site indices via STRING_SITES
# ("s0,s1,..."), or set STRING_SIZE to auto-pick the most central size-s
# C_m string copy on the (kagome_bond) lattice.
STRING_SITES=${STRING_SITES:-""}
STRING_SIZE=${STRING_SIZE:-2}

if [ -z "$STRING_SITES" ]; then
    STRING_SITES=$(python - <<PY
import numpy as np
nx, ny, s = $NX, $NY, $STRING_SIZE
lattice = "$LATTICE"
if lattice == "kagome_bond_triangle":
    from src.rydberg.lattices import (generate_kagome_bond_triangle_lattice,
                                      kagome_triangle_multi_size_translations)
    pos = generate_kagome_bond_triangle_lattice(nx, ny, 1.0)
    _, ssets, _, _ = kagome_triangle_multi_size_translations(
        nx, ny, loop_sizes=(), string_sizes=(s,))
else:
    from src.rydberg.lattices import (generate_kagome_bond_lattice,
                                      kagome_loop_string_translations)
    pos = generate_kagome_bond_lattice(nx, ny, 1.0)
    _, ssets = kagome_loop_string_translations(nx, ny, loop_size=2, string_size=s)
centre = pos.mean(axis=0)
best = min(ssets, key=lambda st: np.linalg.norm(pos[st].mean(axis=0) - centre))
print(",".join(str(x) for x in best))
PY
)
    echo "Auto-selected central size-${STRING_SIZE} string: sites ${STRING_SITES}"
fi

OUT_NAME=${OUT_NAME:-"strwork_${LATTICE}_${NX}x${NY}_M${M}_K${K_VALUES//,/-}_n${N_TRAJ}_${JOB_TAG}.h5"}
FILEPATH="data/${OUT_NAME}"

echo "Off-diagonal string-work production run"
echo "Node: $NODE_DESC, ranks=$NTASKS, omp_threads/rank=$CPT"
echo "Geometry: $LATTICE ${NX}x${NY}, a=$A_LAT; string sites: $STRING_SITES"
echo "QAQMC: M=$M, delta=[$DELTA_MIN,$DELTA_MAX], Rb=$RB, groups=$DELTA_GROUPS"
echo "Protocol: K=$K_VALUES ($SCHEDULE, $DIRECTION), n_traj=$N_TRAJ, "
echo "          thermalize=$N_THERMALIZE, decorr=$DECORR, ckpt every $CKPT_TRAJ traj"
echo "Output: $FILEPATH"
echo

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC \
    python -u -m src.mpi.qaqmc_string_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" \
    --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --delta-groups "$DELTA_GROUPS" \
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
    --filepath "$FILEPATH"

echo
echo "Finished: $(date --iso-8601=seconds)"
