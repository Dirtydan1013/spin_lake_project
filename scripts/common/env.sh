# ─── Shared runtime environment for scripts production runs ─────────────
#
# Sourced (not executed) by every run_*.sh, from the REPO ROOT:
#     source scripts/common/env.sh
#
# The same run script then works both ways:
#     sbatch scripts/run/cpu/run_x.sh   # SLURM: resources from the job
#     bash   scripts/run/cpu/run_x.sh   # bare server: env vars / autodetect
#
# The launcher is mpiexec in BOTH cases, so behavior is identical inside and
# outside a job — only where NTASKS/CPT come from differs.  The conda-forge
# Open MPI 5 runtime (PRRTE) ships slurm+ssh launch plugins: inside a
# multi-node SLURM allocation mpiexec reads the node list itself and spawns
# remote daemons via srun; outside SLURM use HOSTS/HOSTFILE below (needs
# passwordless ssh and identical repo/conda paths on every node).  The PMI
# unsets below only stop stale client-side PMI vars leaking into the job.
#
# Everything is overridable from the environment or scripts/common/site.conf
# (gitignored; copy site.conf.example):
#   CONDA_ROOT, CONDA_ENV   conda install dir / env name
#   NTASKS                  MPI ranks            (bare default: phys_cores / CPT)
#   CPT                     cores per rank == OMP threads per rank (bare default: 1)
#   BIND_FLAGS              Open MPI binding (default: --map-by slot:PE=$CPT --bind-to core)
#                           binding is topology-based (hwloc) — CPU vendor doesn't
#                           matter; override for NUMA-heavy hosts (EPYC:
#                           "--map-by numa:PE=$CPT --bind-to core") or disable in
#                           cgroup-limited containers ("--bind-to none")
#   MPI_EXTRA_FLAGS         extra mpiexec flags, e.g. --oversubscribe
#   HOSTS                   explicit per-node rank placement, e.g.
#                           "cpunode02:32,cpunode03:16" (slots per node; becomes
#                           mpiexec --host).  Set NTASKS to the TOTAL rank count.
#   HOSTFILE                same via an Open MPI hostfile ("nodeX slots=N" lines)
#
# Exports: NTASKS, CPT, JOB_TAG (job id or timestamp.pid), NODE_DESC, MPIEXEC.

# ── site-local overrides ─────────────────────────────────────────────────────
_QAQMC_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$_QAQMC_COMMON_DIR/site.conf" ]; then
    source "$_QAQMC_COMMON_DIR/site.conf"
fi

# ── conda environment ────────────────────────────────────────────────────────
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniconda3}
CONDA_ENV=${CONDA_ENV:-qaqmc}
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    # conda-forge activate.d scripts (binutils/gcc from cxx-compiler) read
    # possibly-unset vars ($ADDR2LINE, ...) and die under `set -u`
    case $- in *u*) _QAQMC_HAD_U=1;; *) _QAQMC_HAD_U=0;; esac
    set +u
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    [ "$_QAQMC_HAD_U" = 1 ] && set -u
    export PATH="$CONDA_PREFIX/bin:$PATH"
elif [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    echo "[env.sh] WARNING: $CONDA_ROOT not found and conda env '$CONDA_ENV' not" \
         "active — assuming python/mpiexec are already on PATH" >&2
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Avoid Open MPI / SLURM PMIx conflicts (harmless outside SLURM)
unset PMI_SIZE PMI_RANK PMI_FD PMI_PORT
unset PMIX_RANK PMIX_SERVER_URI2 PMIX_SECURITY_MODE

# ── resources: SLURM job vs bare server ──────────────────────────────────────
if [ -n "${SLURM_JOB_ID:-}" ]; then
    CPT=${CPT:-${SLURM_CPUS_PER_TASK:-1}}
    NTASKS=${NTASKS:-${SLURM_NTASKS:-1}}
    JOB_TAG=$SLURM_JOB_ID
    NODE_DESC=${SLURM_NODELIST:-$(hostname)}
else
    # physical cores (exclude SMT/hyperthreads: MC ranks on sibling threads
    # just fight over cache); fall back to nproc if lscpu is unavailable
    _PHYS_CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    [ "${_PHYS_CORES:-0}" -gt 0 ] || _PHYS_CORES=$(nproc)
    CPT=${CPT:-1}
    NTASKS=${NTASKS:-$(( _PHYS_CORES / CPT ))}
    [ "$NTASKS" -ge 1 ] || NTASKS=1
    JOB_TAG="$(date +%Y%m%d_%H%M%S).$$"
    NODE_DESC=$(hostname)
fi

export OMP_NUM_THREADS=$CPT
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── launcher ─────────────────────────────────────────────────────────────────
# Bind each rank to its own block of cores (NUMA-local memory for the
# ~GB/rank op arrays; unbound ranks migrate across sockets).
BIND_FLAGS=${BIND_FLAGS:---map-by slot:PE=$CPT --bind-to core}
MPI_EXTRA_FLAGS=${MPI_EXTRA_FLAGS:-}
PLACE_FLAGS=
[ -n "${HOSTS:-}" ]    && PLACE_FLAGS="--host $HOSTS"
[ -n "${HOSTFILE:-}" ] && PLACE_FLAGS="$PLACE_FLAGS --hostfile $HOSTFILE"
MPIEXEC="mpiexec $PLACE_FLAGS $BIND_FLAGS $MPI_EXTRA_FLAGS -n $NTASKS"

echo "[env.sh] mode=$([ -n "${SLURM_JOB_ID:-}" ] && echo slurm || echo bare)" \
     "node=$NODE_DESC ntasks=$NTASKS cpt=$CPT bind='$BIND_FLAGS'" \
     "${HOSTS:+hosts=$HOSTS}${HOSTFILE:+hostfile=$HOSTFILE}"
