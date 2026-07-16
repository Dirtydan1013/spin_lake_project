#!/bin/bash
#SBATCH --job-name=test_work_allgpu
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/test_work_allgpu_%j.out
#SBATCH --error=logs/test_work_allgpu_%j.err

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
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi --query-gpu=index,name,memory.total \
    --format=csv,noheader

cd "${SLURM_TMPDIR:-/tmp}"
for device in 0 1 2; do
    echo "[CUDA-GATE] visible device ${device}"
    CUDA_VISIBLE_DEVICES=$device python -c \
        "import qaqmc_cuda; print(qaqmc_cuda.device_info()[0], flush=True)"
    CUDA_VISIBLE_DEVICES=$device python -m pytest -q "$ROOT/tests/gpu"
    CUDA_VISIBLE_DEVICES=$device python \
        -m src.probes.qaqmc_work_cuda \
        --engine string --lattice 1d_chain --N 8 --M 2000 \
        --neighbor-cutoff 1 --delta-groups 32 --sites 2,4 \
        --warmup 1 --gpu-steps 5 --cpu-steps 1 --topology-sweeps 5
    CUDA_VISIBLE_DEVICES=$device python \
        -m src.probes.qaqmc_work_cuda \
        --engine renyi --lattice 1d_chain --N 8 --M 2000 \
        --neighbor-cutoff 1 --delta-groups 32 --sites 2,4 \
        --warmup 1 --gpu-steps 5 --cpu-steps 1 --topology-sweeps 5
done
