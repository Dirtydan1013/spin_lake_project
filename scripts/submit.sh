#!/bin/bash
# Launcher-agnostic submit wrapper: sbatch when SLURM exists, background bash
# otherwise.  Run from the repo root.
#
#     ./scripts/submit.sh scripts/run/run_kagome_sse.sh
#     ./scripts/submit.sh scripts/run/run_kagome_otf.sh --nodelist=cpunode02
#
# Extra args go to sbatch (override the #SBATCH headers).  Without SLURM the
# script runs via nohup with output in logs/<script>_<stamp>.log; tune
# resources with NTASKS/CPT env vars (see scripts/common/env.sh).
#
# EXCLUSIVE=1 requests the whole node (sbatch --exclusive): each rank gets a
# full physical core.  Default allows co-scheduling — cpunode02 advertises
# 128 logical CPUs (64 cores × 2 hyperthreads), so two 64-task jobs stack on
# hyperthread pairs (each ~60-70% speed, but higher combined throughput).
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_script.sh> [sbatch args...]" >&2
    exit 1
fi
script=$1
shift
[ -f "$script" ] || { echo "ERROR: no such script: $script" >&2; exit 1; }

if command -v sbatch >/dev/null 2>&1; then
    if [ "${EXCLUSIVE:-0}" = "1" ]; then
        sbatch --exclusive "$@" "$script"
    else
        sbatch "$@" "$script"
    fi
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
