#!/bin/bash
#SBATCH --job-name=bench_cpu_mem64
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bench_cpu_mem64_%j.out
#SBATCH --error=logs/bench_cpu_mem64_%j.err

set -euo pipefail

ROOT=${ROOT:-$PWD}
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
mkdir -p "$ROOT/logs"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

cd "${SLURM_TMPDIR:-/tmp}"
for variant in baseline optimized; do
    if [ "$variant" = baseline ]; then
        build="$ROOT/build_cpu_memory_ab/baseline"
    else
        build="${QAQMC_BUILD:-$ROOT/build_cpu_shared}"
    fi
    echo "[CPU-MEM64] variant=$variant build=$build"
    PYTHONPATH="$build:$ROOT" mpiexec -n 64 --bind-to core --map-by core \
        python -m src.probes.qaqmc_cpu_memory_mpi \
        --label "$variant" --M "${M:-2760000}" \
        --warmup "${WARMUP:-1}" --steps "${STEPS:-3}" \
        --event-storage packed64
done

echo "[CPU-MEM64] optimized aggressive event mode"
PYTHONPATH="${QAQMC_BUILD:-$ROOT/build_cpu_shared}:$ROOT" \
    mpiexec -n 64 --bind-to core --map-by core \
    python -m src.probes.qaqmc_cpu_memory_mpi \
    --label optimized_p_bond16 --M "${M:-2760000}" \
    --warmup "${WARMUP:-1}" --steps "${STEPS:-3}" \
    --event-storage p_bond16

PYTHONPATH="${QAQMC_BUILD:-$ROOT/build_cpu_shared}:$ROOT" \
    mpiexec -n 64 --bind-to core --map-by core \
    python -m src.probes.qaqmc_cpu_memory_mpi \
    --label optimized_p_only32 --M "${M:-2760000}" \
    --warmup "${WARMUP:-1}" --steps "${STEPS:-3}" \
    --event-storage p_only32
