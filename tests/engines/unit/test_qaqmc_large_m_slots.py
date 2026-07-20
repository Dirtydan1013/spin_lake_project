"""Large-M slot arithmetic for the QAQMC engine (int64 migration).

The production arrays at M ~ 1e10-1e11 are terabytes, so these tests exercise
the exact production *arithmetic* through allocation-free bindings
(``_delta_at``, ``_packed64_roundtrip``) plus the constructor guards, which
run before any O(M) allocation.

The one genuinely allocating test (crossing the old int32 wall with a real
mc_step) costs ~30 GB RAM and minutes of runtime; it is gated behind
``QAQMC_BIG_M_TESTS=1`` and meant for manual runs on a large-memory node.
"""

import os

import numpy as np
import pytest

qaqmc_cpp = pytest.importorskip("qaqmc_cpp")

M_HUGE = 10**11            # target production scale
M_TOTAL_HUGE = 2 * M_HUGE


class TestPacked64:
    def test_layout_limits(self):
        assert qaqmc_cpp._packed64_slot_max == 2**41 - 1
        assert qaqmc_cpp._packed64_bond_max == 2**21 - 1
        assert M_TOTAL_HUGE < qaqmc_cpp._packed64_slot_max

    @pytest.mark.parametrize("p", [0, 1, 2**31 - 1, 2**31, 2**32,
                                   M_TOTAL_HUGE - 1, 2**41 - 1])
    @pytest.mark.parametrize("b", [0, 1, 65535, 2**21 - 1])
    @pytest.mark.parametrize("e", [0, 1])
    def test_roundtrip(self, p, b, e):
        packed, p2, b2, e2 = qaqmc_cpp._packed64_roundtrip(p, b, e)
        assert (p2, b2, e2) == (p, b, e)
        assert packed >= 0                      # sign bit never set

    def test_order_is_p_major(self):
        # upper_bound over packed keys relies on (p, b, e) lexicographic order.
        entries = [(p, b, e)
                   for p in (0, 5, 2**31, M_TOTAL_HUGE - 1)
                   for b in (0, 3, 2**21 - 1)
                   for e in (0, 1)]
        packed = [qaqmc_cpp._packed64_roundtrip(*t)[0] for t in entries]
        assert packed == sorted(packed)
        assert entries == sorted(entries)


class TestDeltaSchedule:
    DMIN, DMAX = -2.0, 4.5

    def d(self, p, M=M_HUGE):
        return qaqmc_cpp._delta_at(p, M, self.DMIN, self.DMAX)

    def test_endpoints_exact(self):
        assert self.d(0) == self.DMIN
        assert self.d(M_HUGE) == self.DMAX

    def test_forward_backward_symmetry(self):
        # Algebraically equal; the two ramp branches round differently at the
        # last ulp (identical to the pre-migration formula), so compare at
        # ulp-level tolerance rather than bit-exactly.
        for off in (1, 17, 12345, 10**9):
            a, b = self.d(M_HUGE - off), self.d(M_HUGE + off)
            assert a == pytest.approx(b, rel=1e-14, abs=1e-14)

    def test_adjacent_slots_resolved(self):
        # Neighbouring slots at M=1e11 differ by span/M ~ 6.5e-11; double
        # resolves that with ~5 decimal digits to spare.  This is the
        # "floating-point rounding" concern, answered quantitatively.
        for p in (1, M_HUGE // 3, M_HUGE - 2, M_HUGE + 5, M_TOTAL_HUGE - 2):
            step = abs(self.d(p + 1) - self.d(p))
            span_per_slot = (self.DMAX - self.DMIN) / M_HUGE
            assert step > 0.0
            assert abs(step - span_per_slot) < 1e-4 * span_per_slot

    def test_monotone_on_forward_ramp(self):
        ps = np.linspace(0, M_HUGE, 1000, dtype=np.int64)
        vals = [self.d(int(p)) for p in ps]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_matches_small_m_engine(self):
        # The static hook must be the production expression: compare against
        # an actual engine's exported schedule at small M.
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        eng = qaqmc_cpp.QAQMCEngine(N=4, Omega=1.0, delta_min=self.DMIN,
                                    delta_max=self.DMAX, Rb=2.4, M=50,
                                    epsilon=0.01, seed=1, pos=pos,
                                    neighbor_cutoff=-1, delta_groups=10)
        sched = eng.delta_schedule
        for p in range(100):
            assert sched[p] == qaqmc_cpp._delta_at(p, 50, self.DMIN, self.DMAX)


class TestConstructorGuards:
    POS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def _make(self, M):
        return qaqmc_cpp.QAQMCEngine(N=4, Omega=1.0, delta_min=-2.0,
                                     delta_max=4.5, Rb=2.4, M=M,
                                     epsilon=0.01, seed=1, pos=self.POS,
                                     neighbor_cutoff=-1, delta_groups=10)

    def test_rejects_m_beyond_packed_slot_field(self):
        # Guard runs before any O(M) allocation, so this is cheap.
        with pytest.raises(ValueError, match="2\\^41"):
            self._make(2**41)

    def test_int64_m_accepted_at_binding(self):
        # M > 2^31 must reach C++ intact (old int binding raised TypeError).
        # Constructing would allocate ~10 GB, so only check the boundary is
        # the guard, not the binding: 2^41 passes the binding and dies in the
        # guard with our message (previous test), while a plain int32 M works.
        eng = self._make(1000)
        assert eng.M == 1000 and eng.M_total == 2000


@pytest.mark.skipif(os.environ.get("QAQMC_BIG_M_TESTS") != "1",
                    reason="~30 GB RAM + minutes; set QAQMC_BIG_M_TESTS=1")
class TestCrossInt32Wall:
    def test_mc_step_beyond_int32(self):
        # M_total = 2.4e9 > 2^31: every slot index, event offset and packed
        # key in this step exceeds what the old int32 layout could hold.
        M = 1_200_000_000
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                        [2.0, 0.0], [2.0, 1.0]])
        eng = qaqmc_cpp.QAQMCEngine(N=6, Omega=1.0, delta_min=-2.0,
                                    delta_max=4.5, Rb=2.4, M=M,
                                    epsilon=0.01, seed=7, pos=pos,
                                    neighbor_cutoff=1, delta_groups=60)
        assert eng.M_total == 2 * M
        eng.set_observable_sites([[0, 1, 2, 3]], [[0, 1]])
        eng.mc_step()
        bd = eng.memory_breakdown
        assert bd["operator_slots"] == 2 * M
        assert (bd["site_operator_count"] + bd["bond_operator_count"]) == 2 * M
        # A full on-the-fly measurement walks the entire 2.4e9-slot string.
        res = eng.run_onthefly(n_equil=0, n_samples=1)
        assert np.all(np.isfinite(res["density"]))
        assert np.all(res["density"] >= 0.0) and np.all(res["density"] <= 1.0)
