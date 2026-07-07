#!/bin/bash
#SBATCH --job-name=Probe_OTF
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/probe_otf_%j.out
#SBATCH --error=logs/probe_otf_%j.err

# Runtime probe for the QAQMC diagonal-profile engine: times engine init +
# a handful of mc_steps at the PRODUCTION geometry and extrapolates to the
# full sample budget.  Mirrors run_kagome_otf.sh defaults (triangle lattice,
# no bulk restriction) so the estimate is realistic.

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qaqmc

mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$CONDA_PREFIX/bin:$PATH"

unset PMI_SIZE PMI_RANK PMI_FD PMI_PORT
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SECURITY_MODE

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LATTICE=${LATTICE:-kagome_bond_triangle}
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
M=${M:-100000000}
TARGET_SAMPLES=${TARGET_SAMPLES:-100000}

echo "Probing QAQMC ${NX}x${NY} ${LATTICE} profile runtime (M=${M})"

mpiexec --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core -n $SLURM_NTASKS \
    python -u "scripts/python script/probe_runtime_qaqmc_otf.py" \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --M "$M" --delta-groups 600 \
    --delta-min -2.0 --delta-max 6.0 \
    --Rb 2.4 --neighbor-cutoff -1 \
    --probe-equil 24 --probe-samples 24 \
    --target-equil 4000 --target-samples "$TARGET_SAMPLES" \
    --omp-threads "$SLURM_CPUS_PER_TASK"
