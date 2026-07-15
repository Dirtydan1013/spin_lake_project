#!/bin/bash
#SBATCH --job-name=bench_qaqmc_batch
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_qaqmc_batch_%j.out
#SBATCH --error=logs/bench_qaqmc_batch_%j.err

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$ROOT/logs"
export PYTHONPATH="$ROOT/build_cuda:$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_TMPDIR:-/tmp}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python "$ROOT/main_scripts/python_scripts/probe_qaqmc_batch_cuda.py" \
    --engines "${ENGINES:-standard,string,renyi}" \
    --batch-sizes "${BATCH_SIZES:-1,2,4,8}" \
    --M "${M:-2760000}" --warmup "${WARMUP:-1}" --steps "${STEPS:-5}"
