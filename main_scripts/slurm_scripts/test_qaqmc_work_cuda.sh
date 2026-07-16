#!/bin/bash
#SBATCH --job-name=test_work_cuda
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_work_cuda_%j.out
#SBATCH --error=logs/test_work_cuda_%j.err

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
python -m pytest -q "$ROOT/tests/gpu"

python "$ROOT/main_scripts/python_scripts/probe_qaqmc_work_cuda.py" \
    --engine string --lattice 1d_chain --N 8 --M 2000 \
    --neighbor-cutoff 1 --delta-groups 32 --sites 2,4 \
    --warmup 1 --gpu-steps 5 --cpu-steps 1 --topology-sweeps 5

python "$ROOT/main_scripts/python_scripts/probe_qaqmc_work_cuda.py" \
    --engine renyi --lattice 1d_chain --N 8 --M 2000 \
    --neighbor-cutoff 1 --delta-groups 32 --sites 2,4 \
    --warmup 1 --gpu-steps 5 --cpu-steps 1 --topology-sweeps 5
