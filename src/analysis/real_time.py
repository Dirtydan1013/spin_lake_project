"""
Real-time dynamics via exact diagonalization for small Rydberg systems.

This module re-exports ramp_real_time from ed_core for backward compatibility.
The implementation lives in src/ed_core.py with numba-optimized kernels.
"""

from src.analysis.ed_core import ramp_real_time

__all__ = ['ramp_real_time']
