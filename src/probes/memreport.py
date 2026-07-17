"""End-of-probe memory report shared by the runtime probes.

Three layers of information:

1. Measured per rank: current RSS (``VmRSS``) and the kernel's high-water
   mark (``VmHWM``) — the HWM automatically captures any transient peak that
   happened DURING the probe (alias-table build, first cluster event
   allocation, SSE ``adjust_M`` growth ...).
2. Extrapolated to production: chains are independent, so per-rank RSS does
   not depend on rank count; node total ≈ target_ranks × per-rank.  A fit
   verdict is printed against ``--node-mem-gb`` (default 240, cpunode02).
3. Analytic notes for known transients the probe did NOT exercise (e.g.
   warm-start config export, rank-0 result gathers) — passed in by each
   probe as plain strings and printed verbatim.

Output: a few human-readable lines plus one machine-parsable JSON line.
"""

from __future__ import annotations

import json
import os


def _status_kib(field: str) -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    if field == "VmHWM":  # portable fallback (KiB on Linux)
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return 0


def rss_mib() -> float:
    """Current resident set size of this process, MiB."""
    return _status_kib("VmRSS") / 1024.0


def peak_mib() -> float:
    """High-water-mark RSS of this process (lifetime peak), MiB."""
    return _status_kib("VmHWM") / 1024.0


def default_target_ranks(fallback: int) -> int:
    for var in ("TARGET_RANKS", "SLURM_NTASKS", "NTASKS"):
        value = os.environ.get(var)
        if value and value.isdigit():
            return int(value)
    return int(fallback)


def qaqmc_engine_core_mib(cpp_engine) -> float | None:
    """Standard-engine core bytes from the exact C++ breakdown, MiB."""
    try:
        breakdown = cpp_engine.memory_breakdown   # readonly property
        if callable(breakdown):
            breakdown = breakdown()
        return float(dict(breakdown)["total_capacity_bytes"]) / 2**20
    except Exception:
        return None


def sse_engine_core_mib(cpp_engine) -> float:
    """Analytic SSE core estimate, MiB (dominant arrays only).

    op_types/op_sites are int32 × M capacity slots; the packed bond-event
    list is 8 B per endpoint (2/bond op) plus site lists and the int8
    bond_spin cache.  Uses the worst case "every op is a bond op".
    """
    m_cap = int(cpp_engine.M)
    n_ops = int(cpp_engine.n_ops)
    op_arrays = m_cap * 8            # int32 types + int32 sites
    bond_events = n_ops * 2 * 8      # packed int64, both endpoints
    bond_spin = m_cap                # int8 per slot
    return (op_arrays + bond_events + bond_spin) / 2**20


def renyi_work_core_mib(m_total: int) -> float:
    """Analytic two-replica work-engine core estimate, MiB.

    Dominant terms per replica pair: int32 op strings (2 × 8 B/slot),
    the raw-W cache W_by_op_ (2 × 4 doubles/slot), packed uint32 channel
    events (≤ 2 endpoints × 4 B/slot) and the int8 bond-spin caches.
    """
    per_slot = 2 * (8 + 32 + 8 + 1)
    return m_total * per_slot / 2**20


def report(tag: str, comm=None, *, engine_core_mib: float | None = None,
           core_label: str = "engine core", notes: tuple = (),
           target_ranks: int | None = None, node_mem_gb: float = 240.0) -> None:
    """Collective (when ``comm`` is given): every rank must call this."""
    now = rss_mib()
    peak = peak_mib()

    if comm is not None:
        rank = comm.Get_rank()
        n_ranks = comm.Get_size()
        rows = comm.gather((now, peak), root=0)
        if rank != 0:
            return
        now_values = sorted(row[0] for row in rows)
        peak_values = sorted(row[1] for row in rows)
        now_med = now_values[len(now_values) // 2]
        now_max = now_values[-1]
        peak_med = peak_values[len(peak_values) // 2]
        peak_max = peak_values[-1]
    else:
        n_ranks = 1
        now_med = now_max = now
        peak_med = peak_max = peak

    ranks = default_target_ranks(target_ranks or n_ranks)
    node_now_gib = ranks * now_max / 1024.0
    node_peak_gib = ranks * peak_max / 1024.0
    node_gib = node_mem_gb * 1000**3 / 2**30   # GB (Slurm convention) → GiB
    headroom = 1.0 - node_peak_gib / node_gib
    verdict = "OK" if headroom > 0.10 else ("TIGHT" if headroom > 0.0 else "DOES NOT FIT")

    print(f"[{tag}] memory: rank RSS {now_med:.0f} MiB (max {now_max:.0f}), "
          f"peak(HWM) {peak_med:.0f} MiB (max {peak_max:.0f})")
    if engine_core_mib is not None:
        print(f"[{tag}]   {core_label} ≈ {engine_core_mib:.1f} MiB")
    for note in notes:
        print(f"[{tag}]   note: {note}")
    print(f"[{tag}] node @ {ranks} ranks: now ~{node_now_gib:.1f} GiB, "
          f"peak ~{node_peak_gib:.1f} GiB → {node_mem_gb:.0f} GB node "
          f"{verdict} (headroom {100 * headroom:.0f}%)")
    print(f"[{tag}] memory-json: " + json.dumps(dict(
        rank_rss_mib_median=round(now_med, 1), rank_rss_mib_max=round(now_max, 1),
        rank_peak_mib_median=round(peak_med, 1), rank_peak_mib_max=round(peak_max, 1),
        engine_core_mib=None if engine_core_mib is None else round(engine_core_mib, 1),
        target_ranks=ranks, node_peak_gib=round(node_peak_gib, 2),
        node_mem_gb=node_mem_gb, verdict=verdict)))
