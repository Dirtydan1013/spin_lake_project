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

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source main_scripts/common/env.sh

mkdir -p logs

LATTICE=${LATTICE:-kagome_bond_triangle}
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
M=${M:-27600000}
TARGET_SAMPLES=${TARGET_SAMPLES:-100000}

echo "Probing QAQMC ${NX}x${NY} ${LATTICE} profile runtime (M=${M})"

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC python -u main_scripts/python_scripts/probe_runtime_qaqmc_otf.py \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --M "$M" --delta-groups 600 \
    --delta-min -2.0 --delta-max 6.0 \
    --Rb 2.4 --neighbor-cutoff -1 \
    --probe-equil 24 --probe-samples 24 \
    --target-equil 4000 --target-samples "$TARGET_SAMPLES" \
    --omp-threads "$CPT"
