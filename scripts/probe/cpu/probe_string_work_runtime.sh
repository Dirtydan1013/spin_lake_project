#!/bin/bash
#SBATCH --job-name=Probe_StrWork
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=120gb
#SBATCH --output=logs/probe_strwork_%j.out
#SBATCH --error=logs/probe_strwork_%j.err
#SBATCH --time=01:00:00

# Runtime probe for the off-diagonal string-work engine: runs a SHORT real
# batch at the production geometry/M/K (a few trajectories per rank) and
# extrapolates elapsed-per-trajectory to the production budget.  Mirrors
# run_kagome_string_work.sh defaults.
#
# Output: timing/extrapolation report in the job log only (no data/ files).
# Usage:  sbatch scripts/probe/cpu/probe_string_work_runtime.sh

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source scripts/common/env.sh

mkdir -p logs

# ─── Production target being estimated (mirror run_kagome_string_work.sh) ───
LATTICE=${LATTICE:-kagome_bond_triangle}
BOUNDARY=${BOUNDARY:-open}
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
DIRECTION=${DIRECTION:-forward}
STRING_SIZE=${STRING_SIZE:-2}
TARGET_N_TRAJ=${TARGET_N_TRAJ:-4000}
TARGET_THERMALIZE=${TARGET_THERMALIZE:-5000}
DECORR=${DECORR:-100}
# Relaxation per lambda step of the anchor bridge (the vs-ED gates used 4/4;
# 1/1 under-relaxes hard bridges — watch n_eff / zero_frac).
N_TOPO_SWEEPS=${N_TOPO_SWEEPS:-1}
N_QMC_SWEEPS=${N_QMC_SWEEPS:-1}

# Drag-ladder phase target (empty DRAG_DELTAS = phase disabled, both in the
# probe and in the production estimate).  The probe runs the drag over the
# FULL target grid/spr with only PROBE_DRAG_SAMPLES samples/rung, so the
# production drag time is the measured drag time x DRAG_SAMPLES/
# PROBE_DRAG_SAMPLES x DRAG_REPEATS (rung count identical by construction).
DRAG_DELTAS=${DRAG_DELTAS:-""}
DRAG_SPR=${DRAG_SPR:-64}
DRAG_BIDIR=${DRAG_BIDIR:-0}
DRAG_SAMPLES=${DRAG_SAMPLES:-400}
DRAG_SWEEPS=${DRAG_SWEEPS:-1}
DRAG_BURN=${DRAG_BURN:-5}
DRAG_REPEATS=${DRAG_REPEATS:-1}
PROBE_DRAG_SAMPLES=${PROBE_DRAG_SAMPLES:-4}

# Growth residence-ladder anchor probe (GROWTH=1): runs the ladder at
# PROBE_GROWTH_SAMPLES per stage — the timing scales, and the reported
# min_stage_flips tells you the mixing rate that sets the production
# GROWTH_SAMPLES.  K=none skips the lambda anchor.
GROWTH=${GROWTH:-0}
GROWTH_SAMPLES=${GROWTH_SAMPLES:-4000}
GROWTH_SWEEPS=${GROWTH_SWEEPS:-1}
GROWTH_EQUIL=${GROWTH_EQUIL:-200}
GROWTH_TUNE=${GROWTH_TUNE:-300}
PROBE_GROWTH_SAMPLES=${PROBE_GROWTH_SAMPLES:-200}

# Probe scale: a couple of trajectories per rank + a short thermalize.
PROBE_TRAJ_PER_RANK=${PROBE_TRAJ_PER_RANK:-2}
PROBE_THERMALIZE=${PROBE_THERMALIZE:-50}

PROBE_TRAJ=$(( PROBE_TRAJ_PER_RANK * NTASKS ))

# Explicit STRING_SITES overrides the auto-pick (same contract as the run
# script — e.g. a vertex hexagon loop that is smaller than any Z_l copy).
STRING_SITES=${STRING_SITES:-""}
if [ -z "$STRING_SITES" ]; then
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
fi

echo "=== Probing string-work runtime ==="
echo "Node: $NODE_DESC, ranks=$NTASKS"
echo "Geometry: $LATTICE ${NX}x${NY} ($BOUNDARY), M=$M, K=$K ($SCHEDULE), string sites: $STRING_SITES"
echo "Probe: $PROBE_TRAJ trajectories total ($PROBE_TRAJ_PER_RANK/rank), thermalize=$PROBE_THERMALIZE"
echo "Estimating production: n_traj=$TARGET_N_TRAJ, thermalize=$TARGET_THERMALIZE"
if [ -n "$DRAG_DELTAS" ]; then
    echo "Drag probe: deltas=$DRAG_DELTAS spr=$DRAG_SPR, $PROBE_DRAG_SAMPLES samples/rung "
    echo "            (target $DRAG_SAMPLES/rung x $DRAG_REPEATS repeats/rank)"
fi
echo

PROBE_LOG=$(mktemp)
trap 'rm -f "$PROBE_LOG"' EXIT

T0=$(date +%s.%N)
# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC python -u -m src.mpi.qaqmc_string_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" \
    --Omega 1.0 --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --delta-groups "$DELTA_GROUPS" \
    --string-sites "$STRING_SITES" \
    --K-values "$K" \
    --schedule "$SCHEDULE" \
    --direction "$DIRECTION" \
    --n-trajectories "$PROBE_TRAJ" \
    --n-thermalize "$PROBE_THERMALIZE" \
    --decorrelation-steps "$DECORR" \
    --n-topology-sweeps-per-lambda "$N_TOPO_SWEEPS" \
    --n-qaqmc-sweeps-per-lambda "$N_QMC_SWEEPS" \
    $( [ "$GROWTH" = "1" ] && printf %s --growth-anchor ) \
    $( [ "$GROWTH" = "1" ] && printf "%s" "--growth-samples-per-stage $PROBE_GROWTH_SAMPLES --growth-sweeps-between-samples $GROWTH_SWEEPS --growth-equil-per-stage $GROWTH_EQUIL --growth-tune-samples $GROWTH_TUNE" ) \
    ${DRAG_DELTAS:+--drag-deltas "$DRAG_DELTAS"} \
    ${DRAG_DELTAS:+--drag-slots-per-rung "$DRAG_SPR"} \
    $( [ -n "$DRAG_DELTAS" ] && [ "$DRAG_BIDIR" = "1" ] && printf "%s" "--drag-bidirectional" ) \
    ${DRAG_DELTAS:+--drag-samples-per-rung "$PROBE_DRAG_SAMPLES"} \
    ${DRAG_DELTAS:+--drag-sweeps-between-samples "$DRAG_SWEEPS"} \
    ${DRAG_DELTAS:+--drag-burn-per-rung "$DRAG_BURN"} \
    ${DRAG_DELTAS:+--drag-repeats 1} \
    ${DRAG_DELTAS:+--drag-thermalize "$PROBE_THERMALIZE"} \
    --seed 7 2>&1 | tee "$PROBE_LOG"
T1=$(date +%s.%N)

python - <<PY
import re
probe_elapsed = $T1 - $T0
probe_traj_per_rank = $PROBE_TRAJ_PER_RANK
target_per_rank = ($TARGET_N_TRAJ + $NTASKS - 1) // $NTASKS

# Parse the driver's own phase timings from the tee'd log.
log = open("$PROBE_LOG").read()
drag_m = re.search(r"drag: .* elapsed=([0-9.]+)s", log)
drag_probe = float(drag_m.group(1)) if drag_m else 0.0
growth_m = re.search(r"growth: .* elapsed=([0-9.]+)s", log)
growth_probe = float(growth_m.group(1)) if growth_m else 0.0
lam_probe = probe_elapsed - drag_probe - growth_probe

# Rough split: engine init + thermalize is a fixed overhead; per-trajectory
# cost scales linearly.  The K-line "elapsed=" in the log gives the pure
# sampling time; here we scale the whole lambda part conservatively.
per_traj = lam_probe / max(probe_traj_per_rank, 1)
lam_est = per_traj * target_per_rank
print(f"probe wall = {probe_elapsed:.1f}s "
      f"(lambda part {lam_probe:.1f}s for {probe_traj_per_rank} traj/rank, "
      f"includes init+thermalize {$PROBE_THERMALIZE} steps)")
print(f"UPPER-BOUND lambda-anchor estimate ≈ {per_traj:.1f}s/traj × "
      f"{target_per_rank} traj/rank = {lam_est / 3600:.2f} h")
growth_est = 0.0
if growth_probe > 0.0:
    n_stages = len("$STRING_SITES".split(","))
    step_g = re.findall(r"equilibration .*, ([0-9.]+) step/s", log)
    t_step_g = 1.0 / float(step_g[-1]) if step_g else 0.0
    fixed_g = n_stages * ($GROWTH_EQUIL + 3 * $GROWTH_TUNE) * t_step_g
    samp_g = max(growth_probe - fixed_g, 0.0)
    growth_est = samp_g * ($GROWTH_SAMPLES / $PROBE_GROWTH_SAMPLES) + fixed_g
    flips_m = re.search(r"min_stage_flips=(\d+)", log)
    print(f"growth probe = {growth_probe:.1f}s at $PROBE_GROWTH_SAMPLES samples/stage → "
          f"production growth ≈ {growth_est / 3600:.2f} h ($GROWTH_SAMPLES samples/stage); "
          f"min_stage_flips={flips_m.group(1) if flips_m else '?'} at probe scale — scale "
          f"GROWTH_SAMPLES so every stage gets >~200 flips")
drag_est = 0.0
if drag_probe > 0.0:
    # Rung count is identical between probe and production (same grid/spr),
    # so only the SAMPLING part scales with samples/rung; per-rung burn-in
    # and thermalization are fixed per pass.  t_step comes from the drag
    # equilibration progress line, n_rungs from the driver's drag summary.
    rungs_m = re.search(r"drag: .*rungs=(\d+)", log)
    step_m = re.findall(r"drag\] equilibration .*, ([0-9.]+) step/s", log)
    n_rungs = int(rungs_m.group(1)) if rungs_m else 0
    t_step = 1.0 / float(step_m[-1]) if step_m else 0.0
    fixed = n_rungs * $DRAG_BURN * t_step
    samp_part = max(drag_probe - fixed, 0.0)
    scale = $DRAG_SAMPLES / $PROBE_DRAG_SAMPLES
    drag_est = (samp_part * scale + fixed) * $DRAG_REPEATS
    print(f"drag probe = {drag_probe:.1f}s at $PROBE_DRAG_SAMPLES samples/rung "
          f"({n_rungs} rungs; burn share {fixed:.0f}s) → production drag ≈ "
          f"{drag_est / 3600:.2f} h ($DRAG_SAMPLES samples/rung × "
          f"$DRAG_REPEATS repeats/rank)")
from src.probes import costreport
costreport.report(lam_est + growth_est + drag_est, $NTASKS, $CPT)
print("For a tighter estimate use the per-K 'elapsed=' / 'drag:' lines printed by the driver.")
PY
