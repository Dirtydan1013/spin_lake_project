#!/bin/bash
#SBATCH --job-name=SSE_Kag6x6
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/sse_6x6_%j.out
#SBATCH --error=logs/sse_6x6_%j.err

# ─── Finite-temperature SSE production run (thermal ensemble) ────────────────
#
# Lattice: kagome_bond_triangle — the cropped patch whose every atom belongs
# to at least one complete blockade triangle (density/SF over all atoms), same
# convention as run_kagome_otf.sh.  Each rank runs an independent chain and
# writes ONE self-contained file  <RUN_DIR>/rank{r}.h5  holding a chunk{i}
# group per bin plus a final_config group for warm starting.
#
# Storage: --checkpoint is the merged bin==flush size.  Every CHECKPOINT
# samples each rank forms one bin (means of energy/density/mz/n_ops) and
# appends it as the next chunk; a crash loses at most one bin.  NEVER `cp` over
# the live qaqmc_cpp .so while this runs — deploy with `mv` (atomic rename).

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source scripts/common/env.sh

mkdir -p logs data

# ─── Tunables ────────────────────────────────────────────────────────────────
# EXCLUSIVE=1 (via submit.sh) → sbatch --exclusive: whole node, one full
# physical core per rank.  Default allows co-scheduling: two 64-task jobs
# share the node's hyperthreads (~60-70% speed each, higher total throughput).
LATTICE=${LATTICE:-kagome_bond}
# Spatial lattice boundary: open (finite patch) or periodic (torus).  periodic
# is only valid for kagome_bond (the driver errors on the cropped triangle).
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
OMEGA=${OMEGA:-1.0}
DELTA=${DELTA:-4.5}
RB=${RB:-2.4}
BETA=${BETA:-20}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
N_EQUIL=${N_EQUIL:-4000}
# Print rank-0 equilibration progress every this many steps (<= 0 disables).
EQUIL_PRINT_EVERY=${EQUIL_PRINT_EVERY:-500}
N_SAMPLES=${N_SAMPLES:-1280000}
# checkpoint = samples per bin == chunk flush size (merged).  Pick so one
# chunk ~= a few minutes of wall time.
CHECKPOINT=${CHECKPOINT:-250}
# Diagonal observables (QAQMC-profile parity: Z_l/C_m/A_v/VBS/SS are ON
# automatically on kagome lattices when the deployed qaqmc_cpp supports them).
# occ-SF matrix q-grid side (0 = off) and full-state snapshots per rank/chunk.
OCC_SF_GRID_N=${OCC_SF_GRID_N:-12}
N_SNAPSHOTS=${N_SNAPSHOTS:-4}
# Per-rank random site-label permutation (breaks the shared update-scan-order
# domain selection; see scripts/experiments/).  Default ON; PERMUTE_SITES=0
# restores pre-2026-07 fixed-seed trajectories.
PERMUTE_SITES=${PERMUTE_SITES:-1}
SEED=${SEED:-42}
RUN_DIR=${RUN_DIR:-"data/sse_${NX}x${NY}_${LATTICE}_beta${BETA}_delta${DELTA}"}
# Warm start: point CONFIG_IN at a previous run's RUN_DIR (rank{r}.h5 with a
# final_config group) to skip thermalization entirely.
CONFIG_IN=${CONFIG_IN:-""}

echo "Starting SSE ${NX}x${NY} ${LATTICE} thermal simulation (beta=${BETA})"
echo "Node(s): $NODE_DESC"
echo "MPI tasks: $NTASKS (omp_threads/rank=$CPT)"
echo "Run dir: $RUN_DIR"

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC \
    python -m src.mpi.sse_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" \
    --a "$A_LAT" \
    --Omega "$OMEGA" \
    --delta "$DELTA" \
    --Rb "$RB" \
    --beta "$BETA" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --seed "$SEED" \
    --n-equil "$N_EQUIL" \
    --equil-progress-every "$EQUIL_PRINT_EVERY" \
    --n-samples "$N_SAMPLES" \
    --checkpoint "$CHECKPOINT" \
    --occ-sf-grid-n "$OCC_SF_GRID_N" \
    --n-snapshots "$N_SNAPSHOTS" \
    $( [ "$PERMUTE_SITES" = "1" ] || printf %s --no-permute-site-labels ) \
    --run-dir "$RUN_DIR" \
    ${CONFIG_IN:+--config-in "$CONFIG_IN"}
    # NOTES:
    #   - triangle lattice ⇒ observables over ALL atoms (complete blockade
    #     triangles everywhere), same as run_kagome_otf.sh.
    #   - output layout: <RUN_DIR>/rank{r}.h5 (chunk{i} groups + final_config)
    #     plus a one-time meta.h5 (geometry + params) from rank 0.
    #   - post-process with src.mpi.sse_mpi.combine_run(RUN_DIR) for
    #     burn-in-trimmed observable means/errors.

echo "Simulation Finished successfully!"
