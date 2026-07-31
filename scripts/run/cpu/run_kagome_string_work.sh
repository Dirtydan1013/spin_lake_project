#!/bin/bash
#SBATCH --job-name=StrWork_Kag
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=240gb
#SBATCH --output=logs/string_work_%j.out
#SBATCH --error=logs/string_work_%j.err
#SBATCH --time=08:00:00

# ─── Off-diagonal string-work (Jarzynski) production run ────────────────────
#
# Computes O_C = Z_C / Z_empty for a C_m string via the half-line-toggle
# Jarzynski estimator (QAQMCEngine string-seam primitives, driver
# src.mpi.qaqmc_string_work_mpi).  Default lattice: kagome_bond_triangle
# (cropped patch; every atom in a complete blockade triangle).  Set
# LATTICE=kagome_bond for the plain bond lattice.
#
# Incremental checkpointing flushes each rank's log_J samples every
# CKPT_TRAJ trajectories to <filepath minus .h5>_chunks/K{K}/rank{r}/
# chunk{c}.h5, so a crash loses at most one chunk per rank.
#
# Convergence diagnostics to watch in the log: n_eff/n_traj (Jarzynski
# effective sample size) and zero_frac (trajectories ending with J=0 —
# refine the λ schedule / raise K if it is large).
#
# Output: data/string_work_... .h5 (+ _chunks/K{K}/rank{r}.h5, configs/).
# Usage:  ./scripts/submit.sh scripts/run/cpu/run_kagome_string_work.sh
#         STRING_SIZE=2 K_VALUES=200,400 ./scripts/submit.sh scripts/run/cpu/run_kagome_string_work.sh

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source scripts/common/env.sh

mkdir -p logs data

# ─── Tunables ────────────────────────────────────────────────────────────────
# EXCLUSIVE=1 (via submit.sh) → sbatch --exclusive: whole node, one full
# physical core per rank.  Default allows co-scheduling: two 64-task jobs
# share the node's hyperthreads (~60-70% speed each, higher total throughput).
LATTICE=${LATTICE:-kagome_bond_triangle}
# Spatial lattice boundary: open (finite patch) or periodic (torus, kagome_bond
# only).  The driver errors on the cropped triangle + periodic.
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
K_VALUES=${K_VALUES:-"200"}
SCHEDULE=${SCHEDULE:-cosine}
DIRECTION=${DIRECTION:-forward}
N_TRAJ=${N_TRAJ:-4000}
N_THERMALIZE=${N_THERMALIZE:-5000}
# Print rank-0 thermalization progress every this many steps (<= 0 disables).
EQUIL_PRINT_EVERY=${EQUIL_PRINT_EVERY:-500}
DECORR=${DECORR:-100}
# Relaxation per lambda step of the anchor bridge (the vs-ED gates used 4/4;
# 1/1 under-relaxes hard bridges — watch n_eff / zero_frac).
N_TOPO_SWEEPS=${N_TOPO_SWEEPS:-1}
N_QMC_SWEEPS=${N_QMC_SWEEPS:-1}
SEED=${SEED:-7}
CKPT_TRAJ=${CKPT_TRAJ:-250}
# Warm start: CONFIG_IN = previous run's final-config dir (rank{r}.h5) —
# thermalization is skipped.  Final configs saved to CONFIG_OUT
# (default <filepath>_configs).
CONFIG_IN=${CONFIG_IN:-""}
CONFIG_OUT=${CONFIG_OUT:-""}
# Per-rank random site-label permutation (breaks the shared update-scan-order
# domain selection; see scripts/experiments/).  Default ON; PERMUTE_SITES=0
# restores pre-2026-07 fixed-seed trajectories.
PERMUTE_SITES=${PERMUTE_SITES:-1}

# ── Drag-ladder phase (whole-curve O_C(delta); docs/design/seam_drag_curve.md)
# DRAG_DELTAS non-empty enables the phase after the lambda-anchor: the RB
# ladder drags the seam cut along the ramp and records log[Z_X(m)/Z_X(M)] at
# each delta; the driver composes O_C(delta) = O_C(anchor;K) x ratio per K.
# DRAG_SPR = slots per RB rung — bigger cuts rung count, but beyond the
# worldline correlation crossover the rung log-sd grows ~linearly and
# efficiency drops again: keep the rung log-sd <~0.3 (kagome 4x4/M=4096
# calibration → 64; re-probe per geometry/M, design doc SS7).
DRAG_DELTAS=${DRAG_DELTAS:-""}          # e.g. "5.0,4.5,4.25,4.0,3.5,3.0"
DRAG_MIRROR=${DRAG_MIRROR:-1}           # mirror-average the two branches
DRAG_SAMPLES=${DRAG_SAMPLES:-400}       # equilibrium samples per rung
DRAG_SWEEPS=${DRAG_SWEEPS:-1}           # mc_steps between rung samples
DRAG_BURN=${DRAG_BURN:-5}               # mc_steps after each rung move
DRAG_SPR=${DRAG_SPR:-64}               # slots per rung
DRAG_REPEATS=${DRAG_REPEATS:-1}         # independent ladder passes per rank
DRAG_THERMALIZE=${DRAG_THERMALIZE:--1}  # reverse-sector equil (-1 = N_THERMALIZE)

# ── Growth residence-ladder anchor (docs/design/seam_drag_curve.md SS9) ─────
# GROWTH=1 anchors O_C by growing the string one seam bit per stage
# (balanced-lambda sector residence) — the robust choice when half-line
# mixing is slow (kagome loops at Rb=2.4).  Combine with K_VALUES=none to
# skip the lambda-Jarzynski anchor entirely, or keep both as a cross-check.
GROWTH=${GROWTH:-0}
GROWTH_SAMPLES=${GROWTH_SAMPLES:-4000}   # residence samples per stage
GROWTH_SWEEPS=${GROWTH_SWEEPS:-1}        # mc_steps between samples
GROWTH_EQUIL=${GROWTH_EQUIL:-200}        # equilibration per stage
GROWTH_TUNE=${GROWTH_TUNE:-300}          # samples per lambda-autotune round

# String selection.  Either give explicit site indices via STRING_SITES
# ("s0,s1,..."), or set STRING_SIZE to auto-pick the most central size-s
# C_m string copy on the (kagome_bond) lattice.
STRING_SITES=${STRING_SITES:-""}
STRING_SIZE=${STRING_SIZE:-2}

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
    echo "Auto-selected central size-${STRING_SIZE} string: sites ${STRING_SITES}"
fi

OUT_NAME=${OUT_NAME:-"string_work_${LATTICE}_${NX}x${NY}_M${M}_K${K_VALUES//,/-}_n${N_TRAJ}_${JOB_TAG}.h5"}
FILEPATH="data/${OUT_NAME}"

echo "Off-diagonal string-work production run"
echo "Node: $NODE_DESC, ranks=$NTASKS, omp_threads/rank=$CPT"
echo "Geometry: $LATTICE ${NX}x${NY}, a=$A_LAT; string sites: $STRING_SITES"
echo "QAQMC: M=$M, delta=[$DELTA_MIN,$DELTA_MAX], Rb=$RB, groups=$DELTA_GROUPS"
echo "Protocol: K=$K_VALUES ($SCHEDULE, $DIRECTION), n_traj=$N_TRAJ, "
echo "          thermalize=$N_THERMALIZE, decorr=$DECORR, ckpt every $CKPT_TRAJ traj"
if [ -n "$DRAG_DELTAS" ]; then
    echo "Drag:     deltas=$DRAG_DELTAS, mirror=$DRAG_MIRROR, spr=$DRAG_SPR, "
    echo "          samples/rung=$DRAG_SAMPLES (sweeps=$DRAG_SWEEPS, burn=$DRAG_BURN), repeats/rank=$DRAG_REPEATS"
fi
echo "Output: $FILEPATH"
echo

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC \
    python -u -m src.mpi.qaqmc_string_work_mpi \
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
    --K-values "$K_VALUES" \
    --schedule "$SCHEDULE" \
    --direction "$DIRECTION" \
    --n-trajectories "$N_TRAJ" \
    --n-thermalize "$N_THERMALIZE" \
    --equil-progress-every "$EQUIL_PRINT_EVERY" \
    --decorrelation-steps "$DECORR" \
    --n-topology-sweeps-per-lambda "$N_TOPO_SWEEPS" \
    --n-qaqmc-sweeps-per-lambda "$N_QMC_SWEEPS" \
    --seed "$SEED" \
    --checkpoint-every-trajectories "$CKPT_TRAJ" \
    ${CONFIG_IN:+--config-in "$CONFIG_IN"} \
    ${CONFIG_OUT:+--config-out "$CONFIG_OUT"} \
    $( [ "$PERMUTE_SITES" = "1" ] || printf %s --no-permute-site-labels ) \
    $( [ "$GROWTH" = "1" ] && printf %s --growth-anchor ) \
    $( [ "$GROWTH" = "1" ] && printf "%s" "--growth-samples-per-stage $GROWTH_SAMPLES --growth-sweeps-between-samples $GROWTH_SWEEPS --growth-equil-per-stage $GROWTH_EQUIL --growth-tune-samples $GROWTH_TUNE" ) \
    ${DRAG_DELTAS:+--drag-deltas "$DRAG_DELTAS"} \
    ${DRAG_DELTAS:+--drag-samples-per-rung "$DRAG_SAMPLES"} \
    ${DRAG_DELTAS:+--drag-sweeps-between-samples "$DRAG_SWEEPS"} \
    ${DRAG_DELTAS:+--drag-burn-per-rung "$DRAG_BURN"} \
    ${DRAG_DELTAS:+--drag-slots-per-rung "$DRAG_SPR"} \
    ${DRAG_DELTAS:+--drag-repeats "$DRAG_REPEATS"} \
    ${DRAG_DELTAS:+--drag-thermalize "$DRAG_THERMALIZE"} \
    $( [ -n "$DRAG_DELTAS" ] && [ "$DRAG_MIRROR" != "1" ] && printf %s --no-drag-mirror ) \
    --filepath "$FILEPATH"

echo
echo "Finished: $(date --iso-8601=seconds)"
