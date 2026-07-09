"""Rank-0 progress printing for MPI equilibration / thermalization loops.

Every driver used to run its whole equilibration in one opaque call and only
print a single "done in Xs" line at the end.  This helper chunks the loop so
rank 0 can report progress + elapsed time periodically (default every 500
steps), which makes multi-hour thermalizations observable in SLURM logs.
"""

from __future__ import annotations

import time


def run_equil_with_progress(advance, n_steps: int, *, label: str,
                            rank: int = 0, print_every: int = 500,
                            verbose: bool = True) -> float:
    """Run ``n_steps`` MC steps via ``advance(chunk)``, printing progress.

    ``advance`` must accept an int chunk size and run exactly that many MC
    steps, such that repeated calls continue the same chain (true for all four
    engines: QAQMC ``run(n, 0)``, SSE ``mc_step`` loops, and both work engines'
    ``thermalize``).

    Rank 0 prints ``[label] equilibration <done>/<total> (elapsed Xs, Y step/s)``
    every ``print_every`` steps.  ``print_every <= 0`` disables intermediate
    prints (single chunk).  Returns this rank's elapsed seconds.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        return 0.0
    every = int(print_every) if int(print_every) > 0 else n_steps
    t0 = time.perf_counter()
    done = 0
    while done < n_steps:
        chunk = min(every, n_steps - done)
        advance(chunk)
        done += chunk
        if verbose and rank == 0:
            elapsed = time.perf_counter() - t0
            rate = done / max(elapsed, 1e-12)
            print(f"[{label}] equilibration {done}/{n_steps} "
                  f"(elapsed {elapsed:.1f}s, {rate:.1f} step/s)", flush=True)
    return time.perf_counter() - t0
