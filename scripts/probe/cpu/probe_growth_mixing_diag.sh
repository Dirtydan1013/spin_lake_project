#!/bin/bash
#SBATCH --job-name=MixDiag
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=120gb
#SBATCH --output=logs/mixdiag_%j.out
#SBATCH --error=logs/mixdiag_%j.err
#SBATCH --time=06:00:00

# Growth-anchor mixing diagnostic (see src/mpi/growth_mixing_diag_mpi.py):
# two-arm (ON/OFF start) occupancy time series on the drifting stages of the
# production growth ladder.  ~0.1 s/sample -> T=48000 on 2 stages ~ 2.8 h.
# Usage: sbatch scripts/probe/cpu/probe_growth_mixing_diag.sh

set -euo pipefail
source scripts/common/env.sh
mkdir -p logs data

T=${T:-48000}
RECORD_STAGES=${RECORD_STAGES:-1,3}
OUT=${OUT:-data/growth_mixing_diag_${JOB_TAG}.h5}

echo "=== Growth-anchor mixing diagnostic ==="
echo "Node: $(hostname), ranks=$NTASKS, T=$T, record stages: $RECORD_STAGES"

$MPIEXEC python -u -m src.mpi.growth_mixing_diag_mpi \
    --T "$T" --record-stages "$RECORD_STAGES" --out "$OUT"

echo "Finished: $(date -Iseconds)"
