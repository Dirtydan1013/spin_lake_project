#!/bin/bash
#SBATCH --job-name=QAQMC_CUDA_Kag6x6
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/qaqmc_cuda_%j.out
#SBATCH --error=logs/qaqmc_cuda_%j.err

# ─── CUDA QAQMC diagonal-profile production run (single-replica engine) ─────
#
# GPU sibling of run_kagome_otf.sh: device-resident standard QAQMC producing
# rank-local batched profiles (density/Z_l/C_m_l/A_v/VBS/SS + occ-SF at
# selected deltas) with exact Philox checkpoint/resume.
# Runner: python -m src.runners.qaqmc_cuda (needs build_cuda/qaqmc_cuda).
#
# One job owns one GPU and runs one independent chain.  Submit multiple jobs
# (or a job array with a shared absolute RUN_DIR) for multiple chains; Slurm
# selects an idle A100/V100 and masks it as device 0.
#
# Output: $RUN_DIR/rank{r}.h5 (chunked, resumable).
# Usage:  sbatch scripts/run/cuda/run_kagome_qaqmc_cuda.sh
#         RUN_DIR=$PWD/data/qaqmc_cuda_ensemble \
#           sbatch --array=0-2 scripts/run/cuda/run_kagome_qaqmc_cuda.sh
#
# NOTE: the process starts outside the repository (cd $SLURM_TMPDIR) so a
# stale root-level native qaqmc_cpp*.so cannot shadow build_cuda via
# Python's empty import path ("Illegal instruction" trap — see README).

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$ROOT/logs" "$ROOT/data"
export PYTHONPATH="$ROOT/build_cuda:$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

LATTICE=${LATTICE:-kagome_bond}
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-2760000}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2}
DELTA_MAX=${DELTA_MAX:-6}
N_EQUIL=${N_EQUIL:-4000}
N_SAMPLES=${N_SAMPLES:-100000}
PROFILE_STEP=${PROFILE_STEP:-10000}
CHECKPOINT=${CHECKPOINT:-200}
OCC_SF_GRID_N=${OCC_SF_GRID_N:-12}
SEED=${SEED:-42}
RANK=${RANK:-${SLURM_ARRAY_TASK_ID:-0}}
RUN_DIR=${RUN_DIR:-$ROOT/data/qaqmc_cuda_${NX}x${NY}_M${M}_${SLURM_JOB_ID:-manual}}

cd "${SLURM_TMPDIR:-/tmp}"
python -m src.runners.qaqmc_cuda \
    --lattice "$LATTICE" --boundary "$BOUNDARY" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon 0.01 --neighbor-cutoff -1 --delta-groups 600 \
    --n-equil "$N_EQUIL" --n-samples "$N_SAMPLES" \
    --profile-step "$PROFILE_STEP" --checkpoint "$CHECKPOINT" \
    --occ-sf-grid-n "$OCC_SF_GRID_N" \
    --occ-sf-deltas -1 0.0 1.0 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 \
    --seed "$SEED" --rank "$RANK" --device 0 --run-dir "$RUN_DIR"
