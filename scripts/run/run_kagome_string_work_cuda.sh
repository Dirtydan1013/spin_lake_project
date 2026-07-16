#!/bin/bash
#SBATCH --job-name=StrWork_CUDA
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/string_work_cuda_%j.out
#SBATCH --error=logs/string_work_cuda_%j.err

# ─── CUDA string-work production run (X_C Jarzynski, GPU backend) ────────────
#
# GPU sibling of run_kagome_string_work.sh: same driver
# (src.mpi.qaqmc_string_work_mpi) with --backend cuda — one MPI rank per
# allocated GPU (the driver maps node-local ranks to visible devices, and
# handles Slurm installs that expose only one CUDA device per task).
# Adds EXACT trajectory resume from the last committed chunk (operator
# checkpoint + Philox counters, RESUME=1).
#
# Output: data/string_work_cuda_..._<RUN_TAG>.h5 (+ _chunks/K{K}/rank{r}.h5).
# Usage:  sbatch scripts/run/run_kagome_string_work_cuda.sh
#         # resume, reusing the original job's tag/paths:
#         sbatch --export=ALL,RESUME=1,RUN_TAG=<original_job_id> \
#           scripts/run/run_kagome_string_work_cuda.sh
# Key knobs (env): STRING_SITES / STRING_SIZE, K_VALUES, N_TRAJ, DECORR, M.

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$ROOT/logs" "$ROOT/data"
export PYTHONPATH="$ROOT/build_cuda:$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

NTASKS=${SLURM_NTASKS:-3}
LATTICE=${LATTICE:-kagome_bond_triangle}
BOUNDARY=${BOUNDARY:-open}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-100000}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-6.0}
DELTA_GROUPS=${DELTA_GROUPS:-600}
STRING_SITES=${STRING_SITES:-0,1}
K_VALUES=${K_VALUES:-200}
N_TRAJ=${N_TRAJ:-4000}
N_THERMALIZE=${N_THERMALIZE:-5000}
DECORR=${DECORR:-100}
SEED=${SEED:-7}
CKPT_TRAJ=${CKPT_TRAJ:-250}
RESUME=${RESUME:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME_TARGET_EXPLICIT=0
if [[ -n "${RUN_TAG+x}" || -n "${FILEPATH+x}" || -n "${CKPT_DIR+x}" ]]; then
    RESUME_TARGET_EXPLICIT=1
fi
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-manual}}
FILEPATH=${FILEPATH:-$ROOT/data/string_work_cuda_${NX}x${NY}_M${M}_K${K_VALUES}_${RUN_TAG}.h5}
CKPT_DIR=${CKPT_DIR:-${FILEPATH%.h5}_chunks}
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
    if [[ "$RESUME_TARGET_EXPLICIT" != "1" ]]; then
        echo "RESUME=1 requires the previous RUN_TAG, FILEPATH, or CKPT_DIR" >&2
        exit 2
    fi
    RESUME_ARGS+=(--resume)
elif [[ "$RESUME" != "0" ]]; then
    echo "RESUME must be 0 or 1" >&2
    exit 2
elif compgen -G "$CKPT_DIR/K*/rank*.h5" > /dev/null && [[ "$OVERWRITE" != "1" ]]; then
    echo "checkpoint data already exists in $CKPT_DIR; use RESUME=1 or OVERWRITE=1" >&2
    exit 2
fi

cd "${SLURM_TMPDIR:-/tmp}"
mpiexec --bind-to core -n "$NTASKS" \
    python -u -m src.mpi.qaqmc_string_work_mpi \
    --backend cuda \
    --lattice "$LATTICE" --boundary "$BOUNDARY" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon 0.01 --neighbor-cutoff -1 --delta-groups "$DELTA_GROUPS" \
    --string-sites "$STRING_SITES" --K-values "$K_VALUES" \
    --schedule cosine --direction forward \
    --n-trajectories "$N_TRAJ" --n-thermalize "$N_THERMALIZE" \
    --decorrelation-steps "$DECORR" --seed "$SEED" \
    --checkpoint-every-trajectories "$CKPT_TRAJ" \
    --checkpoint-dir "$CKPT_DIR" "${RESUME_ARGS[@]}" \
    --filepath "$FILEPATH"
