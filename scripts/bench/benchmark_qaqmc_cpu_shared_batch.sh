#!/bin/bash
#SBATCH --job-name=bench_cpu_shared
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bench_cpu_shared_%j.out
#SBATCH --error=logs/bench_cpu_shared_%j.err

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$ROOT/logs"
export PYTHONPATH="${QAQMC_BUILD:-$ROOT/build_cpu_shared}:$ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_TMPDIR:-/tmp}"
for storage in ${EVENT_STORAGES:-packed64 p_bond16 p_only32}; do
    for batch in ${BATCH_SIZES:-1 2 4 8 16}; do
        echo "[CPU-SHARED] storage=$storage B=$batch"
        python "$ROOT/scripts/bench/probe_qaqmc_cpu_shared_batch.py" \
            --batch-size "$batch" --event-storage "$storage" \
            --M "${M:-2760000}" --warmup "${WARMUP:-2}" \
            --steps "${STEPS:-5}"
    done
done
