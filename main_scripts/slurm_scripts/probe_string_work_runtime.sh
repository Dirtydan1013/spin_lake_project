#!/bin/bash
#SBATCH --job-name=Probe_StrWork
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=120gb
#SBATCH --output=logs/probe_strwork_%j.out
#SBATCH --error=logs/probe_strwork_%j.err
#SBATCH --time=01:00:00

# Runtime probe for the off-diagonal string-work engine: runs a SHORT real
# batch at the production geometry/M/K (a few trajectories per rank) and
# extrapolates elapsed-per-trajectory to the production budget.  Mirrors
# run_kagome_string_work.sh defaults.

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

# ─── Production target being estimated (mirror run_kagome_string_work.sh) ───
LATTICE=${LATTICE:-kagome_bond_triangle}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-100000}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}
DELTA_GROUPS=${DELTA_GROUPS:-600}
K=${K:-200}
SCHEDULE=${SCHEDULE:-cosine}
STRING_SIZE=${STRING_SIZE:-2}
TARGET_N_TRAJ=${TARGET_N_TRAJ:-4000}
TARGET_THERMALIZE=${TARGET_THERMALIZE:-5000}
DECORR=${DECORR:-100}

# Probe scale: a couple of trajectories per rank + a short thermalize.
PROBE_TRAJ_PER_RANK=${PROBE_TRAJ_PER_RANK:-2}
PROBE_THERMALIZE=${PROBE_THERMALIZE:-50}

PROBE_TRAJ=$(( PROBE_TRAJ_PER_RANK * SLURM_NTASKS ))

STRING_SITES=$(python - <<PY
import numpy as np
nx, ny, s = $NX, $NY, $STRING_SIZE
lattice = "$LATTICE"
if lattice == "kagome_bond_triangle":
    from src.rydberg.lattices import (generate_kagome_bond_triangle_lattice,
                                      kagome_triangle_multi_size_translations)
    pos = generate_kagome_bond_triangle_lattice(nx, ny, 1.0)
    _, ssets, _, _ = kagome_triangle_multi_size_translations(
        nx, ny, loop_sizes=(), string_sizes=(s,))
else:
    from src.rydberg.lattices import (generate_kagome_bond_lattice,
                                      kagome_loop_string_translations)
    pos = generate_kagome_bond_lattice(nx, ny, 1.0)
    _, ssets = kagome_loop_string_translations(nx, ny, loop_size=2, string_size=s)
centre = pos.mean(axis=0)
best = min(ssets, key=lambda st: np.linalg.norm(pos[st].mean(axis=0) - centre))
print(",".join(str(x) for x in best))
PY
)

echo "=== Probing string-work runtime ==="
echo "Node: $SLURM_NODELIST, ranks=$SLURM_NTASKS"
echo "Geometry: $LATTICE ${NX}x${NY}, M=$M, K=$K ($SCHEDULE), string sites: $STRING_SITES"
echo "Probe: $PROBE_TRAJ trajectories total ($PROBE_TRAJ_PER_RANK/rank), thermalize=$PROBE_THERMALIZE"
echo "Estimating production: n_traj=$TARGET_N_TRAJ, thermalize=$TARGET_THERMALIZE"
echo

T0=$(date +%s.%N)
mpiexec --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core -n $SLURM_NTASKS \
    python -u -m src.mpi.qaqmc_string_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" \
    --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --delta-groups "$DELTA_GROUPS" \
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
target_per_rank = ($TARGET_N_TRAJ + $SLURM_NTASKS - 1) // $SLURM_NTASKS
# Rough split: engine init + thermalize is a fixed overhead; per-trajectory
# cost scales linearly.  The K-line "elapsed=" in the log gives the pure
# sampling time; here we scale the whole probe conservatively.
per_traj = probe_elapsed / max(probe_traj_per_rank, 1)
est = per_traj * target_per_rank + ($TARGET_THERMALIZE - $PROBE_THERMALIZE) * 0.0
print(f"probe wall = {probe_elapsed:.1f}s for {probe_traj_per_rank} traj/rank "
      f"(includes init+thermalize {$PROBE_THERMALIZE} steps)")
print(f"UPPER-BOUND production estimate ≈ {per_traj:.1f}s/traj × {target_per_rank} traj/rank "
      f"= {per_traj * target_per_rank / 3600:.2f} h  (plus thermalize scaling)")
print("For a tighter estimate use the per-K 'elapsed=' line printed by the driver.")
PY
