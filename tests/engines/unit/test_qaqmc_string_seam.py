import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qaqmc_cpp


def _make_engine(N=6, M=20, seed=42, delta_groups=40):
    pos = np.arange(N).reshape(-1, 1).astype(np.float64)
    return qaqmc_cpp.QAQMCEngine(
        N, 1.0, 0.0, 1.5, 1.2, M, 0.01, seed, pos,
        neighbor_cutoff=1, delta_groups=delta_groups,
    )


def _expected_plus(minus, string_sites, seam_mask, N):
    b = np.zeros(N, dtype=np.int32)
    for k, s in enumerate(string_sites):
        if (seam_mask >> k) & 1:
            b[s] = 1
    return minus ^ b


def _assert_seam_holds(eng, n_steps):
    N = eng.N
    for step in range(n_steps):
        eng.mc_step()
        minus = np.array(eng.state_at_seam_minus)
        plus = np.array(eng.state_at_seam_plus)
        expected = _expected_plus(minus, eng.string_sites, eng.seam_mask, N)
        assert np.array_equal(plus, expected), (
            f"seam constraint n_i^+ = n_i^- xor b_i violated at step {step}: "
            f"minus={minus} plus={plus} expected={expected}"
        )


def _test_seam_constraint_single_site():
    eng = _make_engine()
    eng.set_string_sites([2], eng.M)
    eng.set_seam_mask(0b1)
    _assert_seam_holds(eng, 50)


def _test_seam_constraint_multi_site():
    eng = _make_engine()
    eng.set_string_sites([0, 1, 2, 4], eng.M)
    eng.set_seam_mask(0b1011)  # sites 0, 1, 4 active; site 2 inactive
    _assert_seam_holds(eng, 50)


def _test_seam_constraint_empty_mask_is_trivial():
    # B = empty set: n_i^+ must equal n_i^- exactly for every configured site.
    eng = _make_engine()
    eng.set_string_sites([0, 3, 5], eng.M)
    eng.set_seam_mask(0)
    for _ in range(20):
        eng.mc_step()
    minus = np.array(eng.state_at_seam_minus)
    plus = np.array(eng.state_at_seam_plus)
    assert np.array_equal(plus, minus)


def _test_seam_constraint_off_center_m_star():
    # m_star need not coincide with the existing M-midpoint convention.
    eng = _make_engine(M=30)
    eng.set_string_sites([1, 3], eng.M_total // 4)
    eng.set_seam_mask(0b10)
    _assert_seam_holds(eng, 50)


def _test_unconfigured_string_defaults_to_no_seam():
    # Without set_string_sites(), m_star_ stays at the -1 sentinel: the seam
    # XOR must never fire, so ordinary (non-string) behavior is unaffected.
    eng = _make_engine()
    assert eng.m_star == -1
    assert len(eng.string_sites) == 0
    for _ in range(10):
        eng.mc_step()
    # state_at_seam_minus/plus stay at their all-zero initial allocation size 0
    assert len(np.array(eng.state_at_seam_minus)) == 0
    assert len(np.array(eng.state_at_seam_plus)) == 0


def main():
    _test_seam_constraint_single_site()
    _test_seam_constraint_multi_site()
    _test_seam_constraint_empty_mask_is_trivial()
    _test_seam_constraint_off_center_m_star()
    _test_unconfigured_string_defaults_to_no_seam()
    print("QAQMC Phase A seam constraint (Test 1) checks passed")


if __name__ == "__main__":
    main()
