#!/bin/bash
#SBATCH --job-name=Work_Kag
#SBATCH --partition=cpu
#SBATCH --nodelist=cpunode02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=240gb
#SBATCH --output=logs/work_kagome_%j.out
#SBATCH --error=logs/work_kagome_%j.err
#SBATCH --time=08:00:00

# ─── QAQMC Renyi work-engine production run on cropped kagome-bond lattice ──
#
# Computes ΔS_2 = S_2(A_end) − S_2(A_start) for one nested region pair using
# the Jarzynski work estimator.  No ED reference (system too large).
#
# Tunables you probably want to adjust:
#   --nx --ny            6x6 ⇒ N=216, 8x8 ⇒ N=384.
#   --M                  imaginary-time slices per half-ramp.  For work mode
#                        100k is plenty (we don't accumulate sample-per-slice).
#   --K-values           "200" for a single production run; comma-separated
#                        for a K-convergence sweep.
#   --n-trajectories     Jarzynski sample size (split across ranks).
#   --decorrelation-steps   QAQMC steps in start sector between trajectories.
#   --A-end / --A-start  region pair.  Use the "0:5:12" site-list form for
#                        readability with large N (avoid 216-character bit
#                        strings).  Omit --A-start for ∅→A_end.
#   --filepath           HDF5 output path; result includes per-K work_samples
#                        for offline analysis.
#
# Performance note: ~32 ranks × 4 OMP threads matches cpunode02's topology.
# OMP only speeds up the per-step alias-table refresh and cluster builder;
# sampling itself is single-threaded per rank.

set -euo pipefail

# Shared env: conda, PMIx workarounds, OMP threads, NTASKS/CPT/JOB_TAG/MPIEXEC.
# Works under sbatch AND as plain `bash <script>` on a server without SLURM.
# Run from the repo root.
source main_scripts/common/env.sh

mkdir -p logs data

echo "QAQMC Renyi work-engine production run"
echo "Node: $NODE_DESC, ranks=$NTASKS, omp_threads/rank=$CPT"
echo "Started: $(date --iso-8601=seconds)"
echo

# ─── Defaults — override by editing or copy & change ─────────────────────────
LATTICE=${LATTICE:-kagome_bond_triangle}    # cropped (matches KP geometry)
# Spatial lattice boundary: open (finite patch) or periodic (torus, kagome_bond
# only).  KP region geometry requires an OPEN cropped patch, so leave this
# 'open' for γ runs; periodic is for bulk (non-KP) studies on kagome_bond.
BOUNDARY=${BOUNDARY:-open}
NX=${NX:-6}
NY=${NY:-6}
A_LAT=${A_LAT:-4.0}
M=${M:-100000}
OMEGA=${OMEGA:-1.0}
RB=${RB:-2.4}
DELTA_MIN=${DELTA_MIN:--2.0}
DELTA_MAX=${DELTA_MAX:-6.0}
EPSILON=${EPSILON:-0.01}
NEIGHBOR_CUTOFF=${NEIGHBOR_CUTOFF:--1}     # all-to-all
DELTA_GROUPS=${DELTA_GROUPS:-600}
K_VALUES=${K_VALUES:-"200"}
N_TRAJ=${N_TRAJ:-4000}
N_THERMALIZE=${N_THERMALIZE:-5000}
# Print rank-0 thermalization progress every this many steps (<= 0 disables).
EQUIL_PRINT_EVERY=${EQUIL_PRINT_EVERY:-500}
DECORR=${DECORR:-1000}
SEED=${SEED:-7}
# Incremental checkpointing: flush per-trajectory arrays every this many
# trajectories per rank to <filepath minus .h5>_chunks/[region/]K{K}/rank{r}/
# chunk{c}.h5.  0 = disabled.  A crash then loses at most one chunk per rank.
CKPT_TRAJ=${CKPT_TRAJ:-250}
# Warm start: CONFIG_IN = previous run's final-config dir
# ([region/]rank{r}.h5 in KP-regions mode) — thermalization is skipped.
CONFIG_IN=${CONFIG_IN:-""}
CONFIG_OUT=${CONFIG_OUT:-""}
# Per-rank random site-label permutation (breaks the shared update-scan-order
# domain selection; see scripts/experiments/).  Default ON; PERMUTE_SITES=0
# restores pre-2026-07 fixed-seed trajectories.
PERMUTE_SITES=${PERMUTE_SITES:-1}

# KP region selection.  Three modes (in priority order):
#
#  1) KP_START + KP_END   (nested pair, e.g. A → AB)
#     ΔS_2 = S_2(KP_END) − S_2(KP_START); a single trajectory batch.
#     Requires nested: start ⊆ end.
#     Example:  KP_START=A KP_END=AB
#
#  2) KP_REGIONS           (∅ → each, e.g. all 7 for γ calculation)
#     Loops over each name, returns ΔS_2 = S_2(name); rank 0 prints γ
#     if the standard 7-set is computed.
#     Example: KP_REGIONS="A,B,C,AB,BC,CA,ABC"  or  KP_REGIONS=all
#
#  3) Custom A_END (with optional A_START) — non-KP mode, see below.
#
# Default = all 7 regions (γ run).  Set KP_START/KP_END to switch to pair mode.
KP_REGIONS=${KP_REGIONS:-all}
KP_START=${KP_START:-""}
KP_END=${KP_END:-""}
KP_M=${KP_M:-1}
KP_CENTER=${KP_CENTER:-""}    # e.g. "C24"; empty = auto

# If a pair is specified, suppress the regions-loop mode.
if [ -n "$KP_START" ] || [ -n "$KP_END" ]; then
    KP_REGIONS=""
fi

# Custom region pair (only used when KP_REGIONS is unset / empty).
# Format: bit string "1,0,..." or site-list "0:2:5".
A_END=${A_END:-""}
A_START=${A_START:-""}

OUT_NAME=${OUT_NAME:-"renyi_work_${LATTICE}_${NX}x${NY}_M${M}_K${K_VALUES//,/-}_n${N_TRAJ}_${JOB_TAG}.h5"}
FILEPATH="data/${OUT_NAME}"

echo "Geometry: $LATTICE nx=$NX, ny=$NY, a=$A_LAT"
echo "Hamiltonian: Omega=$OMEGA, Rb=$RB, delta=[$DELTA_MIN,$DELTA_MAX]"
echo "QAQMC: M=$M, delta_groups=$DELTA_GROUPS, neighbor_cutoff=$NEIGHBOR_CUTOFF"
echo "Protocol: K=$K_VALUES, n_trajectories=$N_TRAJ, thermalize=$N_THERMALIZE, decorrelation=$DECORR"
if [ -n "$KP_START" ] && [ -n "$KP_END" ]; then
    echo "KP pair mode: $KP_START → $KP_END, m=$KP_M, center=${KP_CENTER:-auto}"
elif [ -n "$KP_REGIONS" ]; then
    echo "KP regions mode: ∅ → {$KP_REGIONS}, m=$KP_M, center=${KP_CENTER:-auto}"
else
    echo "Custom region: A_start=\"$A_START\", A_end=\"$A_END\""
fi
echo "Output: $FILEPATH"
echo

# ─── Build CLI args conditionally ───────────────────────────────────────────
EXTRA_ARGS=()
if [ -n "$KP_START" ] && [ -n "$KP_END" ]; then
    EXTRA_ARGS+=(--kp-start "$KP_START" --kp-end "$KP_END" --kp-m "$KP_M")
    [ -n "$KP_CENTER" ] && EXTRA_ARGS+=(--kp-preferred-center "$KP_CENTER")
elif [ -n "$KP_REGIONS" ]; then
    EXTRA_ARGS+=(--kp-regions "$KP_REGIONS" --kp-m "$KP_M")
    [ -n "$KP_CENTER" ] && EXTRA_ARGS+=(--kp-preferred-center "$KP_CENTER")
else
    [ -z "$A_END" ] && { echo "ERROR: set KP_START+KP_END, KP_REGIONS, or A_END"; exit 1; }
    EXTRA_ARGS+=(--A-end "$A_END")
    [ -n "$A_START" ] && EXTRA_ARGS+=(--A-start "$A_START")
fi

# $MPIEXEC (from env.sh) = mpiexec + core binding + -n $NTASKS
$MPIEXEC python -u -m src.mpi.qaqmc_renyi_work_mpi \
    --lattice "$LATTICE" \
    --nx "$NX" --ny "$NY" --a "$A_LAT" \
    --M "$M" \
    --Omega "$OMEGA" --Rb "$RB" \
    --delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX" \
    --epsilon "$EPSILON" \
    --neighbor-cutoff "$NEIGHBOR_CUTOFF" \
    --boundary "$BOUNDARY" \
    --delta-groups "$DELTA_GROUPS" \
    "${EXTRA_ARGS[@]}" \
    --K-values "$K_VALUES" \
    --n-trajectories "$N_TRAJ" \
    --n-thermalize "$N_THERMALIZE" \
    --equil-progress-every "$EQUIL_PRINT_EVERY" \
    --decorrelation-steps "$DECORR" \
    --seed "$SEED" \
    --skip-ed \
    --checkpoint-every-trajectories "$CKPT_TRAJ" \
    ${CONFIG_IN:+--config-in "$CONFIG_IN"} \
    ${CONFIG_OUT:+--config-out "$CONFIG_OUT"} \
    $( [ "$PERMUTE_SITES" = "1" ] || printf %s --no-permute-site-labels ) \
    --filepath "$FILEPATH"

echo
echo "Finished: $(date --iso-8601=seconds)"
