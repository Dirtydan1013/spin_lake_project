#!/bin/bash
#SBATCH --job-name=QAQMC_Kag6x6_profile
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/qaqmc_6x6_profile_%j.out
#SBATCH --error=logs/qaqmc_6x6_profile_%j.err

# ─── QAQMC diagonal-profile production run (single-replica engine) ──────────
#
# Lattice: kagome_bond_triangle — the cropped patch whose every atom belongs
# to at least one complete blockade triangle, so NO bulk restriction is used
# (density/SF over all atoms) and the effective observable region is larger
# than plain kagome_bond (6x6: A_v 63→94, loop s=2 copies 16→25, VBS 25→36).
#
# Incremental checkpointing writes self-contained chunks to
#   data/M=<M>_<nx>x<ny>_<stamp>/rank{r}.h5   (chunk{i} groups, one file/rank)
# so a crash loses at most one chunk per rank.  CHECKPOINT is the merged
# bin==flush size: every that many raw samples get averaged into ONE row
# of the Z_l/C_m_l/density arrays AND written out as the next chunk.
#
# 32 ranks × 4 cores matches cpunode02 (sampling is single-threaded per rank;
# OMP only accelerates engine construction).  NEVER `cp` over the live
# qaqmc_cpp .so while this runs — deploy with `mv` (atomic rename).

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
LATTICE=${LATTICE:-kagome_bond}
# Spatial lattice boundary: open (finite cropped patch) or periodic (torus).
# periodic is only valid for kagome_bond (NOT the cropped triangle patch); the
# driver errors out on kagome_bond_triangle + periodic.
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-100000000}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2}
DELTA_MAX=${DELTA_MAX:-6}
N_EQUIL=${N_EQUIL:-4000}
N_SAMPLES=${N_SAMPLES:-72000}
PROFILE_STEP=${PROFILE_STEP:-500000}
# Merged bin==chunk: every CHECKPOINT raw samples per rank get averaged into
# ONE row and written as the next chunk (~30-60 min each at the default).
CHECKPOINT=${CHECKPOINT:-200}
SEED=${SEED:-42}
# Warm start: point CONFIG_IN at a previous run's run dir (the driver finds
# its configs/ subdir automatically) to skip thermalization entirely (n_equil
# forced to 0).  Final configs are saved to <run_dir>/configs/rank{r}.h5 —
# separate from the observable data files, so `rm -rf <run_dir>/configs`
# reclaims that space without touching the data or needing an h5repack.
CONFIG_IN=${CONFIG_IN:-""}
CONFIG_OUT=${CONFIG_OUT:-""}

echo "Starting QAQMC ${NX}x${NY} ${LATTICE} Profile Simulation"
echo "Target Node: $SLURM_NODELIST"
echo "Total MPI Tasks: $SLURM_NTASKS"

# Resolve qaqmc_mpi module path (handles both flat `src/qaqmc_mpi.py` on main
# and nested `src/mpi/qaqmc_mpi.py` on feature branches).
QAQMC_MPI_MOD=$(python - <<'PY'
import importlib.util as u
for m in ("src.qaqmc_mpi", "src.mpi.qaqmc_mpi"):
    if u.find_spec(m):
        print(m); break
PY
)
if [ -z "$QAQMC_MPI_MOD" ]; then
    echo "ERROR: cannot find src.qaqmc_mpi or src.mpi.qaqmc_mpi on PYTHONPATH" >&2
    exit 1
fi
echo "Using module: $QAQMC_MPI_MOD"

# Bind each rank to its own block of cores (NUMA-local memory for the ~GB/rank
# op arrays; --bind-to none let ranks migrate across sockets).
mpiexec --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core -n $SLURM_NTASKS python -m "$QAQMC_MPI_MOD" \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" \
    --M  "$M" \
    --a "$A_LAT" \
    --Rb "$RB" \
    --Omega 1.0 \
    --seed "$SEED" \
    --epsilon 0.01 \
    --delta_min "$DELTA_MIN" \
    --delta_max "$DELTA_MAX" \
    --n_equil "$N_EQUIL" \
    --n_samples "$N_SAMPLES" \
    --omp_threads "$SLURM_CPUS_PER_TASK" \
    --delta_groups 600 \
    --neighbor_cutoff -1 \
    --boundary "$BOUNDARY" \
    --mode profile \
    --measure_every 1 \
    --profile_step "$PROFILE_STEP" \
    --checkpoint "$CHECKPOINT" \
    --snapshot_deltas -1 0.0 1.0 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 \
    --n_snapshots 1 \
    --occ_sf_delta_points -1 0.0 1.0 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 \
    --occ_sf_grid_n 12 \
    --occ_sf_nbatch 4 \
    ${CONFIG_IN:+--config_in "$CONFIG_IN"} \
    ${CONFIG_OUT:+--config_out "$CONFIG_OUT"} \
    --filepath "data/${NX}x${NY}_${LATTICE}_M=${M}.h5"
    # NOTES:
    #   - triangle lattice ⇒ bulk = ALL atoms (complete blockade triangles
    #     everywhere); the h5 'bulk' conventions still apply, just with the
    #     full site set.
    #   - occ-SF MATRIX enabled: sublattice-resolved S_αβ(q) on a 12×12 2D BZ
    #     grid at the listed δ, 4 super-bins/rank → /occ_sf group.  Hexagon
    #     void cell (S_full/S_bulk) AND up+down triangle pair (S_tri) both
    #     written for cross-comparison.
    #   - with --checkpoint > 0 the run dir layout is
    #     data/M=..._<stamp>/rank{r}.h5 (chunk{i} groups) + meta.h5 +
    #     configs/rank{r}.h5 (warm-start configs, deletable independently).

echo "Simulation Finished successfully!"
