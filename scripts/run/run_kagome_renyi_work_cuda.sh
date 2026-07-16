#!/bin/bash
#SBATCH --job-name=RenyiWork_CUDA
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/renyi_work_cuda_%j.out
#SBATCH --error=logs/renyi_work_cuda_%j.err

# Resume example (reuse the original job's tag/path):
#   sbatch --export=ALL,RESUME=1,RUN_TAG=<original_job_id> \
#     scripts/run/run_kagome_renyi_work_cuda.sh

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
M=${M:-225700}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-4.5}
DELTA_GROUPS=${DELTA_GROUPS:-600}
KP_START=${KP_START:-A}
KP_END=${KP_END:-AB}
KP_M=${KP_M:-2}
K_VALUES=${K_VALUES:-400}
N_TRAJ=${N_TRAJ:-32000}
N_THERMALIZE=${N_THERMALIZE:-4000}
DECORR=${DECORR:-200}
SEED=${SEED:-7}
CKPT_TRAJ=${CKPT_TRAJ:-200}
RESUME=${RESUME:-0}
OVERWRITE=${OVERWRITE:-0}
RESUME_TARGET_EXPLICIT=0
if [[ -n "${RUN_TAG+x}" || -n "${FILEPATH+x}" || -n "${CKPT_DIR+x}" ]]; then
    RESUME_TARGET_EXPLICIT=1
fi
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-manual}}
FILEPATH=${FILEPATH:-$ROOT/data/renyi_work_cuda_${NX}x${NY}_M${M}_K${K_VALUES}_${KP_START}_to_${KP_END}_${RUN_TAG}.h5}
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
elif compgen -G "$CKPT_DIR/*/K*/rank*.h5" > /dev/null && [[ "$OVERWRITE" != "1" ]]; then
    echo "checkpoint data already exists in $CKPT_DIR; use RESUME=1 or OVERWRITE=1" >&2
    exit 2
fi

cd "${SLURM_TMPDIR:-/tmp}"
mpiexec --bind-to core -n "$NTASKS" \
    python -u -m src.mpi.qaqmc_renyi_work_mpi \
    --backend cuda \
    --lattice "$LATTICE" --boundary "$BOUNDARY" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon 0.01 --neighbor-cutoff -1 --delta-groups "$DELTA_GROUPS" \
    --kp-start "$KP_START" --kp-end "$KP_END" --kp-m "$KP_M" \
    --K-values "$K_VALUES" --n-trajectories "$N_TRAJ" \
    --n-thermalize "$N_THERMALIZE" --decorrelation-steps "$DECORR" \
    --seed "$SEED" --skip-ed \
    --checkpoint-every-trajectories "$CKPT_TRAJ" \
    --checkpoint-dir "$CKPT_DIR" "${RESUME_ARGS[@]}" \
    --filepath "$FILEPATH"
