#!/bin/bash
#SBATCH --job-name=Probe_Work
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=2
#SBATCH --mem=120gb
#SBATCH --output=logs/probe_work_%j.out
#SBATCH --error=logs/probe_work_%j.err
#SBATCH --time=00:30:00

# Runtime probe for QAQMC Renyi work engine.  Times the basic primitives
# (engine init, backend mc_step, toggle-ratio compute, full probe trajectory)
# and extrapolates to a target production scale to estimate wall-clock.
#
# These values should mirror the production run_kagome_renyi_work.sh defaults
# so the estimate is realistic.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qaqmc

mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$CONDA_PREFIX/bin:$PATH"

unset PMI_SIZE PMI_RANK PMI_FD PMI_PORT
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SECURITY_MODE

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Probe target: which production run are we estimating ─────────────────────
LATTICE=${LATTICE:-kagome_bond_triangle}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-2257000}
OMEGA=${OMEGA:-1.0}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
DELTA_GROUPS=${DELTA_GROUPS:-600}
KP_M=${KP_M:-2}

# Probe region: nested KP pair (start → end), e.g. A → AB.  Both must be set.
PROBE_KP_START=${PROBE_KP_START:-"A"}
PROBE_KP_END=${PROBE_KP_END:-"AB"}
if [ -z "$PROBE_KP_START" ] || [ -z "$PROBE_KP_END" ]; then
    echo "ERROR: PROBE_KP_START and PROBE_KP_END must both be set" >&2
    exit 1
fi

# Production parameters being estimated
TARGET_K=${TARGET_K:-200}
TARGET_N_TRAJ=${TARGET_N_TRAJ:-4000}
TARGET_N_THERMALIZE=${TARGET_N_THERMALIZE:-5000}
TARGET_DECORR=${TARGET_DECORR:-1000}
TARGET_N_REGIONS=${TARGET_N_REGIONS:-1}     # 7 for KP-all, 1 for single region/pair

# Probe scale (small; just enough to time primitives)
PROBE_WARMUP=${PROBE_WARMUP:-10}
PROBE_MC=${PROBE_MC:-50}
PROBE_TOGGLE=${PROBE_TOGGLE:-200}
PROBE_TRAJ=${PROBE_TRAJ:-3}
PROBE_K=${PROBE_K:-20}
PROBE_DECORR=${PROBE_DECORR:-20}

echo "=== Probing QAQMC Renyi work-engine runtime ==="
echo "Node: $SLURM_NODELIST, ranks=$SLURM_NTASKS, omp_threads/rank=$OMP_NUM_THREADS"
echo "Geometry: $LATTICE ${NX}x${NY} (m=${KP_M}), probe pair: ${PROBE_KP_START} → ${PROBE_KP_END}"
echo "Estimating: K=${TARGET_K}, n_traj=${TARGET_N_TRAJ}, thermalize=${TARGET_N_THERMALIZE},"
echo "            decorr=${TARGET_DECORR}, n_regions=${TARGET_N_REGIONS}"
echo

mpiexec --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core -n $SLURM_NTASKS \
    python -u "scripts/python script/probe_runtime_renyi_work.py" \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" \
    --Omega "$OMEGA" --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --delta-groups "$DELTA_GROUPS" \
    --kp-pair-start "$PROBE_KP_START" --kp-pair-end "$PROBE_KP_END" \
    --kp-m "$KP_M" \
    --probe-warmup-mc-steps "$PROBE_WARMUP" \
    --probe-mc-steps "$PROBE_MC" \
    --probe-toggle-attempts "$PROBE_TOGGLE" \
    --probe-trajectories "$PROBE_TRAJ" \
    --probe-K "$PROBE_K" \
    --probe-decorrelation-steps "$PROBE_DECORR" \
    --target-K "$TARGET_K" \
    --target-n-trajectories "$TARGET_N_TRAJ" \
    --target-n-thermalize "$TARGET_N_THERMALIZE" \
    --target-decorrelation-steps "$TARGET_DECORR" \
    --target-n-regions "$TARGET_N_REGIONS"
