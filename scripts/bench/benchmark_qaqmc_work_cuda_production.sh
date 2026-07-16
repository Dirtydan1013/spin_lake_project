#!/bin/bash
#SBATCH --job-name=bench_work_prod
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/bench_work_prod_%j.out
#SBATCH --error=logs/bench_work_prod_%j.err

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

M=${M:-2760000}
GPU_STEPS=${GPU_STEPS:-3}
CPU_STEPS=${CPU_STEPS:-1}
TOPOLOGY_SWEEPS=${TOPOLOGY_SWEEPS:-5}

cd "${SLURM_TMPDIR:-/tmp}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for engine in string renyi; do
    python -m src.probes.qaqmc_work_cuda \
        --engine "$engine" --lattice kagome_bond_triangle \
        --nx 6 --ny 6 --a 4.0 --M "$M" --Rb 2.4 \
        --delta-min -2.0 --delta-max 4.5 \
        --neighbor-cutoff -1 --delta-groups 600 --sites 0,1 \
        --warmup 1 --gpu-steps "$GPU_STEPS" --cpu-steps "$CPU_STEPS" \
        --topology-sweeps "$TOPOLOGY_SWEEPS"
done
