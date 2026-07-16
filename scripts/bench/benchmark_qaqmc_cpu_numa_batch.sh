#!/bin/bash
#SBATCH --job-name=bench_cpu_numa
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bench_cpu_numa_%j.out
#SBATCH --error=logs/bench_cpu_numa_%j.err

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
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp

cd "${SLURM_TMPDIR:-/tmp}"
for storage in ${EVENT_STORAGES:-packed64 p_bond16 p_only32}; do
    for chains in ${CHAINS_PER_RANK:-1 2 4 8 16}; do
        echo "[CPU-NUMA] storage=$storage mpi=4 chains/rank=$chains"
        mpiexec -n 4 --bind-to core --map-by ppr:1:socket:PE=16 \
            python "$ROOT/scripts/bench/probe_qaqmc_cpu_shared_batch_mpi.py" \
            --chains-per-rank "$chains" --event-storage "$storage" \
            --M "${M:-2760000}" --warmup "${WARMUP:-2}" \
            --steps "${STEPS:-5}"
    done
done
