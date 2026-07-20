"""Estimated CPU-hour / cost lines appended to the runtime-probe reports.

CPU-hours are billed on what the job occupies: ranks × threads-per-rank
cores for the estimated wall time of the slowest rank.  The price defaults
to 0.1 per CPU-hour; override with the CPU_HOUR_RATE environment variable.
"""

from __future__ import annotations

import os

DEFAULT_RATE = 0.1  # cost per CPU-hour


def report(wall_seconds: float, n_ranks: int, threads_per_rank: int = 1) -> None:
    """Print the estimated total CPU-hours and cost for a production run."""
    rate = float(os.environ.get("CPU_HOUR_RATE", DEFAULT_RATE))
    threads = max(int(threads_per_rank), 1)
    cores = int(n_ranks) * threads
    cpu_h = wall_seconds / 3600.0 * cores
    print(f"est CPU-hours : {wall_seconds / 3600.0:.2f} h wall x {cores} cores "
          f"({n_ranks} ranks x {threads} threads) = {cpu_h:.1f} CPU-h")
    print(f"est cost      : {cpu_h:.1f} CPU-h x {rate:g}/CPU-h = {cpu_h * rate:.2f}")
