"""
rydberg_sse_ed.py — backward-compatibility shim.

This file re-exports everything from the modular packages so that any
existing code that imports from rydberg_sse_ed continues to work unchanged.

New code should import directly from:
  rydberg_ed      — Exact Diagonalization
  rydberg_sse     — SSE QMC (Numba core + SSE_Rydberg class)
  rydberg_compare — ED vs SSE benchmark
"""

from rydberg_ed import (                    # noqa: F401
    build_rydberg_hamiltonian,
    ed_thermal_energy,
    ed_thermal_density,
)
from rydberg_sse import (                   # noqa: F401
    build_alias_table,
    diagonal_update,
    cluster_update,
    SSE_Rydberg,
)
from rydberg_compare import run_comparison  # noqa: F401

if __name__ == '__main__':
    run_comparison()
