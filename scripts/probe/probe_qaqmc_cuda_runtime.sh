#!/bin/bash
#SBATCH --job-name=probe_QAQMC_CUDA
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/probe_qaqmc_cuda_%j.out
#SBATCH --error=logs/probe_qaqmc_cuda_%j.err

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$ROOT/logs"
export PYTHONPATH="$ROOT/build_cuda:$ROOT"
export OMP_NUM_THREADS=1

M=${M:-2760000}
CPU_STEPS=${CPU_STEPS:-1}
GPU_STEPS=${GPU_STEPS:-5}
FULL_STEPS=${FULL_STEPS:-3}

cd "${SLURM_TMPDIR:-/tmp}"
python -m src.probes.runtime_qaqmc_cuda \
    --M "$M" --cpu-steps "$CPU_STEPS" --gpu-steps "$GPU_STEPS" \
    --full-steps "$FULL_STEPS" --gpu-warmup 1 --event-builds 1

