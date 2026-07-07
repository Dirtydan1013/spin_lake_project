#!/bin/bash
#SBATCH --job-name=SSE_Kag6x6
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qaqmc

mkdir -p logs data
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$CONDA_PREFIX/bin:$PATH"

# Avoid Open MPI / Slurm PMIx conflicts
unset PMI_SIZE PMI_RANK PMI_FD PMI_PORT
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SECURITY_MODE

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ─── Tunables ────────────────────────────────────────────────────────────────
LATTICE=${LATTICE:-kagome_bond_triangle}
# Spatial lattice boundary: open (finite patch) or periodic (torus).  periodic
# is only valid for kagome_bond (the driver errors on the cropped triangle).
BOUNDARY=${BOUNDARY:-open}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
OMEGA=${OMEGA:-1.0}
DELTA=${DELTA:-3.0}
RB=${RB:-2.4}
BETA=${BETA:-16.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
N_EQUIL=${N_EQUIL:-20000}
N_SAMPLES=${N_SAMPLES:-2000000}
# checkpoint = samples per bin == chunk flush size (merged).  Pick so one
# chunk ~= a few minutes of wall time.
CHECKPOINT=${CHECKPOINT:-250}
SEED=${SEED:-42}
RUN_DIR=${RUN_DIR:-"data/sse_${NX}x${NY}_${LATTICE}_beta${BETA}_delta${DELTA}"}
# Warm start: point CONFIG_IN at a previous run's RUN_DIR (rank{r}.h5 with a
# final_config group) to skip thermalization entirely.
CONFIG_IN=${CONFIG_IN:-""}

echo "Starting SSE ${NX}x${NY} ${LATTICE} thermal simulation (beta=${BETA})"
echo "Target Node: $SLURM_NODELIST"
echo "Total MPI Tasks: $SLURM_NTASKS"
echo "Run dir: $RUN_DIR"

# Bind each rank to its own block of cores (NUMA-local memory).
mpiexec --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core -n $SLURM_NTASKS \
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
    --n-samples "$N_SAMPLES" \
    --checkpoint "$CHECKPOINT" \
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
