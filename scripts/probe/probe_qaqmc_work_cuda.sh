#!/bin/bash
#SBATCH --job-name=probe_work_cuda
#SBATCH --partition=gpu
#SBATCH --nodelist=gpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/probe_work_cuda_%j.out
#SBATCH --error=logs/probe_work_cuda_%j.err

# ─── Runtime probe: CUDA work engines (string / Renyi) vs CPU ────────────────
#
# Times full step + topology sweep of the selected CUDA work engine at
# production geometry and compares against the CPU reference, reporting
# timings, speedup and VRAM as JSON to stdout.
# Instrument: python -m src.probes.qaqmc_work_cuda (also reused by the
# test/ gates and bench/ production benchmark with different parameters).
#
# Output: JSON in the job log only (no data/ files).
# Knobs (env): ENGINE=renyi|string, M, GPU_STEPS / CPU_STEPS.
# Usage:  sbatch scripts/probe/probe_qaqmc_work_cuda.sh
#         ENGINE=string M=100000 sbatch scripts/probe/probe_qaqmc_work_cuda.sh

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$ROOT/logs"
export PYTHONPATH="$ROOT/build_cuda:$ROOT"
export OMP_NUM_THREADS=1

ENGINE=${ENGINE:-renyi}
M=${M:-225700}
GPU_STEPS=${GPU_STEPS:-5}
CPU_STEPS=${CPU_STEPS:-1}

cd "${SLURM_TMPDIR:-/tmp}"
python -m src.probes.qaqmc_work_cuda \
    --engine "$ENGINE" --M "$M" --gpu-steps "$GPU_STEPS" \
    --cpu-steps "$CPU_STEPS" --device 0
