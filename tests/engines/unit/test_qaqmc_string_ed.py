import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.ed_core import qaqmc_exact_string_zratio
from src.rydberg.lattices import generate_1d_chain


def _test_empty_set_is_trivial():
    # B = {} must always give Z_B == Z_empty, i.e. O_B == 1 exactly.
    N = 4
    result = qaqmc_exact_string_zratio(
        N=N, Omega=1.0, delta_min=0.0, delta_max=1.5, Rb=1.2, M=20,
        B_sets=[[]],
        pos=generate_1d_chain(N),
        neighbor_cutoff=1,
    )
    assert result["Z_empty"] != 0.0
    assert np.isclose(result["O_B"][frozenset()], 1.0)


def _test_mirror_symmetry_single_site():
    # A 1D chain (no external field breaking left-right symmetry) must give
    # identical <sigma_i^x>-type ratios for mirror-partner sites i <-> N-1-i,
    # independent of m_star: the Hamiltonian, ramp schedule and |0...0> trial
    # state are all invariant under site i -> N-1-i.
    N = 6
    pos = generate_1d_chain(N)
    for m_star in (10, 20, 30):  # M=20 -> M_total=40; try an off-center cut too
        result = qaqmc_exact_string_zratio(
            N=N, Omega=1.0, delta_min=0.0, delta_max=1.5, Rb=1.2, M=20,
            B_sets=[[0], [N - 1], [1], [N - 2], [2], [N - 3]],
            m_star=m_star,
            pos=pos,
            neighbor_cutoff=1,
        )
        o = result["O_B"]
        assert np.isclose(o[frozenset([0])], o[frozenset([N - 1])], atol=1e-10), m_star
        assert np.isclose(o[frozenset([1])], o[frozenset([N - 2])], atol=1e-10), m_star
        assert np.isclose(o[frozenset([2])], o[frozenset([N - 3])], atol=1e-10), m_star


def _test_mirror_symmetry_two_site_string():
    # Same mirror argument for a genuine multi-site string.
    N = 6
    pos = generate_1d_chain(N)
    result = qaqmc_exact_string_zratio(
        N=N, Omega=1.0, delta_min=0.0, delta_max=1.5, Rb=1.2, M=20,
        B_sets=[[0, 1], [N - 1, N - 2], [0, 1, 2], [N - 1, N - 2, N - 3]],
        pos=pos,
        neighbor_cutoff=1,
    )
    o = result["O_B"]
    assert np.isclose(o[frozenset([0, 1])], o[frozenset([N - 1, N - 2])], atol=1e-10)
    assert np.isclose(o[frozenset([0, 1, 2])], o[frozenset([N - 1, N - 2, N - 3])], atol=1e-10)


def main():
    _test_empty_set_is_trivial()
    _test_mirror_symmetry_single_site()
    _test_mirror_symmetry_two_site_string()
    print("QAQMC exact string Z_B/Z_empty (ED) unit checks passed")


if __name__ == "__main__":
    main()
