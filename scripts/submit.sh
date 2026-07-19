#!/bin/bash
# Launcher-agnostic submit wrapper: sbatch when SLURM exists, background bash
# otherwise.  Run from the repo root.
#
#     ./scripts/submit.sh scripts/run/cpu/run_kagome_sse.sh
#     ./scripts/submit.sh scripts/run/cpu/run_kagome_otf.sh --nodelist=cpunode02
#
# Extra args go to sbatch (override the #SBATCH headers).  Without SLURM the
# script runs via nohup with output in logs/<script>_<stamp>.log; tune
# resources with NTASKS/CPT env vars (see scripts/common/env.sh).
#
# EXCLUSIVE=1 requests the whole node (sbatch --exclusive): each rank gets a
# full physical core.  Default allows co-scheduling — cpunode02 advertises
# 128 logical CPUs (64 cores × 2 hyperthreads), so two 64-task jobs stack on
# hyperthread pairs (each ~60-70% speed, but higher combined throughput).
#
# Site scheduler overrides: the #SBATCH partition/nodelist/nodes headers baked
# into the run scripts are THIS cluster's single-node defaults.  Set
# SBATCH_PARTITION / SBATCH_NODELIST / SBATCH_NODES / SBATCH_NTASKS_PER_NODE
# (env or scripts/common/site.conf) and this wrapper injects them as sbatch
# args, which override the headers — no per-script editing.  Explicit CLI args
# here still win over both.  Multi-node example (uniform 32 ranks/node; the
# baked-in --nodelist pin can only be REPLACED, not cleared, so name the node
# set explicitly):
#     SBATCH_NODES=2 SBATCH_NTASKS_PER_NODE=32 SBATCH_NODELIST=cpunode[02-03] \
#         ./scripts/submit.sh scripts/run/cpu/run_kagome_otf.sh
# (uneven per-node counts: SLURM can't express them here — allocate the nodes
# and pass HOSTS="nodeA:32,nodeB:16" so mpiexec places ranks, see env.sh)
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_script.sh> [sbatch args...]" >&2
    exit 1
fi
script=$1
shift
[ -f "$script" ] || { echo "ERROR: no such script: $script" >&2; exit 1; }

# site.conf may set SBATCH_PARTITION/SBATCH_NODELIST (keep ${VAR:-} form
# there so launch-time environment variables still win).
SITE_CONF="$(dirname "$0")/common/site.conf"
[ -f "$SITE_CONF" ] && source "$SITE_CONF"

if command -v sbatch >/dev/null 2>&1; then
    SB_ARGS=()
    [ -n "${SBATCH_PARTITION:-}" ] && SB_ARGS+=(--partition "$SBATCH_PARTITION")
    [ -n "${SBATCH_NODELIST:-}" ]  && SB_ARGS+=(--nodelist "$SBATCH_NODELIST")
    [ -n "${SBATCH_NODES:-}" ]     && SB_ARGS+=(--nodes "$SBATCH_NODES")
    [ -n "${SBATCH_NTASKS_PER_NODE:-}" ] && SB_ARGS+=(--ntasks-per-node "$SBATCH_NTASKS_PER_NODE")
    [ "${EXCLUSIVE:-0}" = "1" ]    && SB_ARGS+=(--exclusive)
    # user CLI args come last: for repeated flags, sbatch takes the last one
    sbatch "${SB_ARGS[@]}" "$@" "$script"
else
    if [ $# -gt 0 ]; then
        echo "ERROR: extra args are sbatch-only, but sbatch is not available: $*" >&2
        exit 1
    fi
    mkdir -p logs
    log="logs/$(basename "${script%.sh}")_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$script" > "$log" 2>&1 &
    echo "no sbatch — started in background: PID $!  log: $log"
fi
