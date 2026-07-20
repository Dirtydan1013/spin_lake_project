#!/bin/bash
#SBATCH --job-name=Probe_SSEOffD
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=120gb
#SBATCH --output=logs/probe_sse_offdiag_%j.out
#SBATCH --error=logs/probe_sse_offdiag_%j.err
#SBATCH --time=01:00:00

# Runtime probe for the thermal SSE off-diagonal string-work engine: a SHORT
# real batch at the production geometry/beta/K (a few trajectories per rank),
# extrapolated to the production budget.  Mirrors run_kagome_offdiagonal.sh
# defaults; sibling of probe_string_work_runtime.sh (QAQMC version).
#
# Output: timing/extrapolation report in the job log only (no data/ files).
# Usage:  sbatch scripts/probe/cpu/probe_sse_offdiagonal_runtime.sh

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source scripts/common/env.sh

mkdir -p logs

# ─── Production target being estimated (mirror run_kagome_offdiagonal.sh) ───
LATTICE=${LATTICE:-kagome_bond}
BOUNDARY=${BOUNDARY:-periodic}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
RB=${RB:-2.4}
DELTA=${DELTA:-4.25}
BETA=${BETA:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
K=${K:-200}
SCHEDULE=${SCHEDULE:-cosine}
STRING_SIZE=${STRING_SIZE:-2}
LOOP=${LOOP:-1}
TARGET_N_TRAJ=${TARGET_N_TRAJ:-4000}
TARGET_THERMALIZE=${TARGET_THERMALIZE:-5000}
DECORR=${DECORR:-50}

# Probe scale: a couple of trajectories per rank + a short thermalize.
PROBE_TRAJ_PER_RANK=${PROBE_TRAJ_PER_RANK:-2}
PROBE_THERMALIZE=${PROBE_THERMALIZE:-100}

PROBE_TRAJ=$(( PROBE_TRAJ_PER_RANK * NTASKS ))

STRING_SITES=$(python - <<PY
import numpy as np
nx, ny, s, want_loop = $NX, $NY, $STRING_SIZE, $LOOP
lattice = "$LATTICE"
if lattice == "kagome_bond_triangle":
    from src.rydberg.lattices import (generate_kagome_bond_triangle_lattice,
                                      kagome_triangle_multi_size_translations)
    pos = generate_kagome_bond_triangle_lattice(nx, ny, 1.0)
    lsets, ssets, _, _ = kagome_triangle_multi_size_translations(
        nx, ny, loop_sizes=(s,) if want_loop else (),
        string_sizes=() if want_loop else (s,))
else:
    from src.rydberg.lattices import (generate_kagome_bond_lattice,
                                      kagome_loop_string_translations)
    pos = generate_kagome_bond_lattice(nx, ny, 1.0)
    lsets, ssets = kagome_loop_string_translations(
        nx, ny, loop_size=s if want_loop else 2,
        string_size=2 if want_loop else s)
sets = lsets if want_loop else ssets
centre = pos.mean(axis=0)
best = min(sets, key=lambda st: np.linalg.norm(pos[st].mean(axis=0) - centre))
print(",".join(str(x) for x in best))
PY
)

echo "=== Probing thermal SSE off-diagonal string-work runtime ==="
echo "Node: $NODE_DESC, ranks=$NTASKS"
echo "Geometry: $LATTICE ${NX}x${NY} ($BOUNDARY), beta=$BETA, delta=$DELTA, K=$K"
echo "String sites: $STRING_SITES  (size $STRING_SIZE, loop=$LOOP)"
echo "Probe: $PROBE_TRAJ trajectories total ($PROBE_TRAJ_PER_RANK/rank), thermalize=$PROBE_THERMALIZE"
echo "Estimating production: n_traj=$TARGET_N_TRAJ, thermalize=$TARGET_THERMALIZE"
echo

T0=$(date +%s.%N)
# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC python -u -m src.mpi.sse_string_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --Omega 1.0 --Rb "$RB" \
    --delta "$DELTA" --beta "$BETA" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --string-sites "$STRING_SITES" \
    --K-values "$K" \
    --schedule "$SCHEDULE" \
    --n-trajectories "$PROBE_TRAJ" \
    --n-thermalize "$PROBE_THERMALIZE" \
    --decorrelation-steps "$DECORR" \
    --seed 7
T1=$(date +%s.%N)

python - <<PY
probe_elapsed = $T1 - $T0
probe_traj_per_rank = $PROBE_TRAJ_PER_RANK
target_per_rank = ($TARGET_N_TRAJ + $NTASKS - 1) // $NTASKS
per_traj = probe_elapsed / max(probe_traj_per_rank, 1)
print(f"probe wall = {probe_elapsed:.1f}s for {probe_traj_per_rank} traj/rank "
      f"(includes init+thermalize {$PROBE_THERMALIZE} sweeps)")
print(f"UPPER-BOUND production estimate ≈ {per_traj:.1f}s/traj × {target_per_rank} traj/rank "
      f"= {per_traj * target_per_rank / 3600:.2f} h  (plus thermalize scaling)")
from src.probes import costreport
costreport.report(per_traj * target_per_rank, $NTASKS, $CPT)
print("For a tighter estimate use the per-K 'elapsed=' line printed by the driver.")
PY
