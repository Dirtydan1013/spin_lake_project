"""
Python wrapper for the single-replica QAQMC off-diagonal string estimator.

Implements the nonequilibrium-work / Jarzynski method from
document/QAQMC_LineCluster_Jarzynski_String_Implementation.md: measures

    O_C = Z_C / Z_empty = <psi_L| P_L X_C P_R |psi_R> / <psi_L| P_L P_R |psi_R>

for X_C = prod_{i in C} sigma_i^x inserted at imaginary-time slice m_star, via
an interpolating ensemble g_lambda(B) = lambda^|B| (1-lambda)^(L_C-|B|) over
subsets B of C. Topology sampling uses the engine's half-line move
(`QAQMCEngine.attempt_string_toggle`/`topology_sweep`, Phase B); ordinary
relaxation uses the seam-aware `diagonal_update`/`cluster_update` (Phase A).

This module is Python-level orchestration around C++ primitives -- the hot
per-step work (topology_sweep, mc_step) runs in C++, matching the design of
`QAQMCRenyiWorkRydberg` (src/engines/qaqmc_renyi_work.py), the analogous
two-replica Renyi-entropy estimator this mirrors.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for _candidate in (_repo_root, os.path.join(_repo_root, "build")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    import qaqmc_cpp
except ImportError as exc:
    raise ImportError(
        "qaqmc_cpp with QAQMCEngine string-seam support is required for "
        "src.engines.qaqmc_string_work"
    ) from exc


def _log_g(lam: float, active: int, length: int) -> float:
    """Endpoint-safe log g_lambda(B) = |B| log(lambda) + (L-|B|) log(1-lambda)."""
    if lam <= 0.0:
        return 0.0 if active == 0 else -math.inf
    if lam >= 1.0:
        return 0.0 if active == length else -math.inf
    return active * math.log(lam) + (length - active) * math.log1p(-lam)


@dataclass
class StringWorkTrajectoryResult:
    log_j: float
    zero_weight: bool
    final_active_count: int


@dataclass
class StringDragRunResult:
    """Per-grid-point Jarzynski estimates of Z_X(m) / Z_X(m_anchor)."""
    m_anchor: int
    m_grid: np.ndarray            # (n_grid,) cut positions, in protocol order
    log_r: np.ndarray             # (n_grid,) log of the Jarzynski mean
    r: np.ndarray                 # (n_grid,) exp(log_r)
    n_eff: np.ndarray             # (n_grid,) effective sample size
    p_max: np.ndarray             # (n_grid,) largest normalized weight
    zero_weight_fraction: np.ndarray  # (n_grid,)
    n_trajectories: int
    log_j_samples: np.ndarray     # (n_trajectories, n_grid) raw accumulated work


@dataclass
class StringDragLadderResult:
    """Rao-Blackwellized ladder estimates of Z_X(m) / Z_X(m_anchor)."""
    m_anchor: int
    m_grid: np.ndarray        # (n_grid,) record points, in protocol order
    log_r: np.ndarray         # (n_grid,) accumulated log ratio at each record point
    log_r_sem: np.ndarray     # (n_grid,) propagated per-rung SEM (quadrature)
    r: np.ndarray             # (n_grid,) exp(log_r)
    rung_m: np.ndarray        # (n_rungs,) target cut of each single-slot rung
    rung_log: np.ndarray      # (n_rungs,) log of each rung's mean RB ratio
    rung_sem: np.ndarray      # (n_rungs,) SEM of log rung ratio
    n_samples_per_rung: int


@dataclass
class StringDragMirroredResult:
    """Mirror-averaged drag curve: geometric mean of the two branches.

    The palindromic operator sequence satisfies Z_X(m; v) ~= Z_X(2M - m; -v)
    (exact up to a one-slot schedule shift), so averaging the two branches at
    the same delta (log-space mean == geo mean) cancels the odd-in-v part.
    Empirically (ED M-scaling, docs/design/seam_drag_curve.md SS6) the
    single-branch lag is ALREADY nearly even in v -- the mirror average does
    not change the ~v^2 convergence order, but it removes the residual odd
    component and the single-branch zero-crossing artifacts, yielding a
    clean monotone 1/M^2 tail that is safe to Richardson-extrapolate to the
    ground-state limit.
    """
    m_forward: np.ndarray     # (n_grid,) forward-branch cut positions (< M)
    m_mirror: np.ndarray      # (n_grid,) 2M - m_forward (same delta)
    log_r_mirror: np.ndarray  # (n_grid,) (log_r_L + log_r_R) / 2
    log_r_sem: np.ndarray     # (n_grid,) quadrature/2 of the branch sems
    r_mirror: np.ndarray      # exp(log_r_mirror) = Z-ratio geo mean
    left: "StringDragLadderResult"
    right: "StringDragLadderResult"


@dataclass
class StringWorkRunResult:
    o_c: float                  # estimate of Z_C / Z_empty
    log_o_c: float
    n_trajectories: int
    n_eff: float                # Jarzynski effective sample size
    p_max: float                # largest normalized trajectory weight
    zero_weight_fraction: float # fraction of trajectories that ended with J=0
    log_j_samples: np.ndarray


class QAQMCStringWorkRydberg:
    """Driver around `qaqmc_cpp.QAQMCEngine`'s Phase A/B string-seam primitives.

    Typical usage::

        eng = QAQMCStringWorkRydberg(N=5, M=16, Omega=1.0, Rb=1.2,
                                     delta_min=0.0, delta_max=1.5, pos=pos)
        eng.set_string_sites([2])                       # L_C = 1, m_star = M
        eng.set_lambda_schedule(cosine_schedule(200))
        eng.thermalize(2000)
        res = eng.run_trajectories(n_trajectories=3000, decorrelation_steps=100)
        print(res.o_c, res.n_eff)
    """

    def __init__(self, N: int, M: int, Omega: float = 1.0, Rb: float = 1.2,
                 delta_min: float = 0.0, delta_max: float = 1.0,
                 epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray | None = None,
                 neighbor_cutoff: int | None = None,
                 delta_groups: int = 600,
                 box_vectors: np.ndarray | None = None):
        if pos is None:
            pos = np.arange(N).reshape(-1, 1).astype(np.float64)
        pos_arr = np.ascontiguousarray(pos, dtype=np.float64)
        nc = neighbor_cutoff if neighbor_cutoff is not None else -1
        box = (np.ascontiguousarray(box_vectors, dtype=np.float64)
               if box_vectors is not None else None)
        self._eng = qaqmc_cpp.QAQMCEngine(
            N, Omega, delta_min, delta_max, Rb, M, epsilon, seed, pos_arr,
            neighbor_cutoff=nc, delta_groups=int(delta_groups), box_vectors=box,
        )
        self.N = N
        self.M = M
        self.M_total = 2 * M
        self._length = 0
        self._lambda_schedule: np.ndarray | None = None

    # ── Configuration ────────────────────────────────────────────────────

    def set_string_sites(self, sites, m_star: int | None = None) -> None:
        sites = list(sites)
        if m_star is None:
            m_star = self.M
        self._eng.set_string_sites(sites, int(m_star))
        self._length = len(sites)

    def set_lambda_schedule(self, lambdas: np.ndarray) -> None:
        lambdas = np.asarray(lambdas, dtype=np.float64)
        if lambdas.ndim != 1 or lambdas.size < 2:
            raise ValueError("lambda_schedule must be a 1D array of length >= 2")
        if lambdas[0] != 0.0 or lambdas[-1] != 1.0:
            raise ValueError("lambda_schedule must run from 0.0 to 1.0")
        if np.any(np.diff(lambdas) < 0.0):
            raise ValueError("lambda_schedule must be monotonically non-decreasing")
        self._lambda_schedule = lambdas

    def _full_mask(self) -> int:
        return (1 << self._length) - 1

    def thermalize(self, n_steps: int, direction: str = "forward") -> None:
        """Equilibrate at the trajectory's starting sector: B=empty for a
        forward (lambda: 0->1) run, B=C (all bits active) for a reverse
        (lambda: 1->0) run -- document section 29/33."""
        if direction == "forward":
            self._eng.set_seam_mask_consistent(0)
        elif direction == "reverse":
            self._eng.set_seam_mask_consistent(self._full_mask())
        else:
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")
        for _ in range(n_steps):
            self._eng.mc_step()

    # ── Trajectories ─────────────────────────────────────────────────────

    def run_trajectory(self, n_topology_sweeps_per_lambda: int = 1,
                       n_qaqmc_sweeps_per_lambda: int = 1,
                       direction: str = "forward") -> StringWorkTrajectoryResult:
        if self._lambda_schedule is None:
            raise RuntimeError("call set_lambda_schedule() first")

        if direction == "forward":
            schedule = self._lambda_schedule
            if self._eng.seam_mask != 0:
                raise RuntimeError("forward trajectory must start in the empty sector (seam_mask=0)")
        elif direction == "reverse":
            schedule = self._lambda_schedule[::-1]
            if self._eng.seam_mask != self._full_mask():
                raise RuntimeError("reverse trajectory must start in the full sector (seam_mask=all-ones)")
        else:
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")

        length = self._length
        log_j = 0.0
        K = len(schedule) - 1
        for m in range(K):
            lam_now = schedule[m]
            lam_next = schedule[m + 1]
            active = bin(self._eng.seam_mask).count("1")
            new_log_g = _log_g(lam_next, active, length)
            old_log_g = _log_g(lam_now, active, length)
            if not math.isfinite(new_log_g):
                # B_m != C at lambda -> 1 (or B_m != empty at lambda -> 0):
                # exact zero, per document section 19 -- not a "multiply by
                # one" prescription, this is a different method than the
                # Renyi TEE work engine. Symmetric for the reverse direction
                # (schedule ends at 0, requiring B_m -> empty).
                return StringWorkTrajectoryResult(
                    log_j=-math.inf, zero_weight=True, final_active_count=active)
            log_j += new_log_g - old_log_g
            if m + 1 < K:
                for _ in range(n_topology_sweeps_per_lambda):
                    self._eng.topology_sweep(float(lam_next))
                for _ in range(n_qaqmc_sweeps_per_lambda):
                    self._eng.mc_step()

        final_active = bin(self._eng.seam_mask).count("1")
        return StringWorkTrajectoryResult(log_j=log_j, zero_weight=False,
                                          final_active_count=final_active)

    def run_trajectories(self, n_trajectories: int, decorrelation_steps: int = 100,
                         n_topology_sweeps_per_lambda: int = 1,
                         n_qaqmc_sweeps_per_lambda: int = 1,
                         direction: str = "forward") -> StringWorkRunResult:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"direction must be 'forward' or 'reverse', got {direction!r}")
        reset_mask = 0 if direction == "forward" else self._full_mask()

        log_j_samples = np.empty(n_trajectories, dtype=np.float64)
        zero_count = 0
        for r in range(n_trajectories):
            # Trajectories always start at the sector matching lambda=0 (B=
            # empty, forward) or lambda=1 (B=C, reverse). The reset must go
            # through the closure-repairing setter: the fixed |0...0>
            # boundaries at both tau ends impose parity(sigma^x ops) == seam
            # bit per string site, and a raw mask write breaks it whenever a
            # bit changes -- every kernel preserves parity, so the following
            # decorrelation steps can never leave the unphysical sector and
            # the trajectory's J sample is garbage (alternating trajectories,
            # since each reset re-flips the parity).
            self._eng.set_seam_mask_consistent(reset_mask)
            for _ in range(decorrelation_steps):
                self._eng.mc_step()
            result = self.run_trajectory(n_topology_sweeps_per_lambda, n_qaqmc_sweeps_per_lambda,
                                         direction=direction)
            log_j_samples[r] = result.log_j
            if result.zero_weight:
                zero_count += 1

        finite = np.isfinite(log_j_samples)
        if not np.any(finite):
            return StringWorkRunResult(
                o_c=0.0, log_o_c=-math.inf, n_trajectories=n_trajectories,
                n_eff=0.0, p_max=0.0, zero_weight_fraction=1.0,
                log_j_samples=log_j_samples)

        max_log = log_j_samples[finite].max()
        weights = np.zeros(n_trajectories, dtype=np.float64)
        weights[finite] = np.exp(log_j_samples[finite] - max_log)
        sum_w = weights.sum()
        log_mean_j = max_log + math.log(sum_w / n_trajectories)

        # Forward: mean(J_fwd) = Z_C/Z_empty = O_C directly.
        # Reverse: mean(J_rev) = Z_empty/Z_C = 1/O_C (document section 33),
        # so invert to report both directions on the same O_C scale.
        if direction == "forward":
            log_o_c = log_mean_j
        else:
            log_o_c = -log_mean_j
        o_c = math.exp(log_o_c)

        p = weights / sum_w
        n_eff = 1.0 / float(np.sum(p ** 2))
        p_max = float(p.max())

        return StringWorkRunResult(
            o_c=o_c, log_o_c=log_o_c, n_trajectories=n_trajectories,
            n_eff=n_eff, p_max=p_max,
            zero_weight_fraction=zero_count / n_trajectories,
            log_j_samples=log_j_samples,
        )


    # ── Seam-drag trajectories (cut-position Jarzynski) ──────────────────
    # docs/design/seam_drag_curve.md: the driven parameter is the cut
    # position m, not the seam strength lambda. Every grid point along one
    # trajectory yields a work sample, so a single family of trajectories
    # estimates the whole curve Z_X(m)/Z_X(m_anchor); combined with the
    # existing run_trajectories() anchor at m_anchor this gives O_C(m) for
    # all m in the grid.

    def _validate_drag_grid(self, m_grid) -> np.ndarray:
        m_grid = np.asarray(m_grid, dtype=np.int64)
        if m_grid.ndim != 1 or m_grid.size < 1:
            raise ValueError("m_grid must be a non-empty 1D array of cut positions")
        if np.any(m_grid < 0) or np.any(m_grid >= self.M_total):
            raise ValueError(f"m_grid entries must lie in [0, {self.M_total - 1}]")
        diffs = np.diff(m_grid)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise ValueError("m_grid must be strictly monotonic (one drag direction "
                             "per trajectory family; run the other side separately)")
        return m_grid

    def run_drag_trajectory(self, m_grid: np.ndarray,
                            n_qaqmc_sweeps_per_shift: int = 1,
                            slots_per_block: int = 1) -> np.ndarray:
        """One drag trajectory from the CURRENT (equilibrated) cut position.

        The protocol moves the cut in switch blocks of ``slots_per_block``
        slots, relaxing with ``n_qaqmc_sweeps_per_shift`` mc_steps between
        blocks; ``m_grid`` is only the set of RECORD points (accumulated
        log-work is snapshotted whenever the cut passes a grid entry).
        Protocol speed = block size / relaxation, NOT the grid spacing --
        a whole grid gap in one block is a fast quench (wide work
        distribution, collapsing n_eff), so keep slots_per_block small.
        A zero-weight crossing makes that and all later entries -inf
        (exact-zero sample). Returns (n_grid,) accumulated log-work.
        """
        m_grid = self._validate_drag_grid(m_grid)
        if slots_per_block < 1:
            raise ValueError("slots_per_block must be >= 1")
        log_j = 0.0
        out = np.empty(m_grid.size, dtype=np.float64)
        m_curr = int(self._eng.m_star)
        for j, m in enumerate(m_grid):
            m = int(m)
            step = 1 if m > m_curr else -1
            while m_curr != m:
                m_next = m_curr + step * min(slots_per_block, abs(m - m_curr))
                log_j += self._eng.seam_drag_to(m_next)
                m_curr = m_next
                if m_curr != m or j + 1 < m_grid.size:
                    for _ in range(n_qaqmc_sweeps_per_shift):
                        self._eng.mc_step()
            out[j] = log_j
        return out

    def run_drag_trajectories(self, m_grid: np.ndarray, n_trajectories: int,
                              decorrelation_steps: int = 100,
                              n_qaqmc_sweeps_per_shift: int = 1,
                              slots_per_block: int = 1,
                              m_anchor: int | None = None) -> StringDragRunResult:
        """Family of drag trajectories anchored at ``m_anchor`` (default: the
        m_star at call time -- pass it explicitly when a previous family left
        the cut parked at its far end).

        Each trajectory re-anchors the cut at ``m_anchor`` via
        ``seam_set_position`` -- the configuration left at the
        far end of the previous trajectory is a sample of the wrong ensemble,
        so ``decorrelation_steps`` mc_steps re-equilibrate before dragging
        (same contract as the lambda-protocol's set_seam_mask_consistent
        reset). The seam mask is never touched: dragging preserves worldline
        closure by construction.
        """
        m_grid = self._validate_drag_grid(m_grid)
        if m_anchor is None:
            m_anchor = int(self._eng.m_star)
        m_anchor = int(m_anchor)
        if m_anchor < 0:
            raise RuntimeError("call set_string_sites() first")
        if not (0 <= m_anchor < self.M_total):
            raise ValueError(f"m_anchor must lie in [0, {self.M_total - 1}]")

        n_grid = m_grid.size
        log_j_samples = np.empty((n_trajectories, n_grid), dtype=np.float64)
        for r in range(n_trajectories):
            self._eng.seam_set_position(m_anchor)
            for _ in range(decorrelation_steps):
                self._eng.mc_step()
            log_j_samples[r] = self.run_drag_trajectory(
                m_grid, n_qaqmc_sweeps_per_shift=n_qaqmc_sweeps_per_shift,
                slots_per_block=slots_per_block)

        log_r = np.empty(n_grid, dtype=np.float64)
        n_eff = np.zeros(n_grid, dtype=np.float64)
        p_max = np.zeros(n_grid, dtype=np.float64)
        zero_frac = np.empty(n_grid, dtype=np.float64)
        for j in range(n_grid):
            col = log_j_samples[:, j]
            finite = np.isfinite(col)
            zero_frac[j] = 1.0 - finite.mean()
            if not np.any(finite):
                log_r[j] = -math.inf
                continue
            max_log = col[finite].max()
            weights = np.zeros(n_trajectories, dtype=np.float64)
            weights[finite] = np.exp(col[finite] - max_log)
            sum_w = weights.sum()
            log_r[j] = max_log + math.log(sum_w / n_trajectories)
            p = weights / sum_w
            n_eff[j] = 1.0 / float(np.sum(p ** 2))
            p_max[j] = float(p.max())

        return StringDragRunResult(
            m_anchor=m_anchor, m_grid=m_grid,
            log_r=log_r, r=np.exp(log_r),
            n_eff=n_eff, p_max=p_max, zero_weight_fraction=zero_frac,
            n_trajectories=n_trajectories, log_j_samples=log_j_samples,
        )


    def run_drag_ladder(self, m_grid: np.ndarray, n_samples_per_rung: int = 400,
                        n_sweeps_between_samples: int = 1,
                        n_burn_per_rung: int = 5,
                        n_equil_at_anchor: int = 0,
                        slots_per_rung: int = 1,
                        m_anchor: int | None = None) -> StringDragLadderResult:
        """RB ladder for the drag curve -- the recommended estimator.

        Walks the cut from the anchor through ``m_grid`` (record points) in
        rungs of ``slots_per_rung`` slots. Each rung m -> m' is an
        EQUILIBRIUM estimate

            Z_X(m')/Z_X(m) = E_{Z_X(m)}[ exp(seam_rb_log_ratio_to(m')) ]

        (the block RB conditional factorizes exactly into per-slot Lambda
        ratios). Raising ``slots_per_rung`` cuts the rung count -- what makes
        production M ~ 1e5 feasible -- but the per-slot log contributions are
        NOT independent along a block (imaginary-time correlations): beyond a
        correlation crossover the rung log-sd grows ~linearly in the block
        size and efficiency degrades again. Calibrate so the rung log-sd
        stays <~0.3 (kagome 4x4/M=4096: ~64 slots; see the design doc SS7
        and the probe script).

        averaged over ``n_samples_per_rung`` samples separated by
        ``n_sweeps_between_samples`` mc_steps; the RB conditional
        (Lambda_tgt/Lambda_cur over the diagonal-op menu, see
        docs/design/seam_drag_curve.md) keeps per-sample values O(1), unlike
        the raw per-config ratio whose ~1/epsilon whales give the plain
        Jarzynski drag a one-sided bias at reachable sample sizes
        (run_drag_trajectories is kept as a bracketing diagnostic: forward /
        inverted-reverse runs bracket the ladder answer). After each rung
        the cut moves via ``seam_drag_to`` and ``n_burn_per_rung`` mc_steps
        re-equilibrate at the new m. The caller must have equilibrated the
        full-mask sector at the anchor (thermalize(direction="reverse"));
        when the current configuration was left at a DIFFERENT cut (e.g. a
        previous family parked it at its far end), pass ``n_equil_at_anchor``
        > 0 to re-equilibrate after the re-anchor and before the first rung.

        Error bars: per-rung SEM propagated in quadrature. Samples within a
        rung are Markov-chain correlated -- increase
        ``n_sweeps_between_samples`` until the quoted SEM is honest (the
        rate-independence check in the vs-ED gate covers this).
        """
        m_grid = self._validate_drag_grid(m_grid)
        if slots_per_rung < 1:
            raise ValueError("slots_per_rung must be >= 1")
        if m_anchor is None:
            m_anchor = int(self._eng.m_star)
        m_anchor = int(m_anchor)
        if not (0 <= m_anchor < self.M_total):
            raise ValueError(f"m_anchor must lie in [0, {self.M_total - 1}]")
        self._eng.seam_set_position(m_anchor)
        for _ in range(n_equil_at_anchor):
            self._eng.mc_step()

        n_grid = m_grid.size
        log_r = np.empty(n_grid, dtype=np.float64)
        log_r_var = np.empty(n_grid, dtype=np.float64)
        rung_m, rung_log, rung_sem = [], [], []
        log_cum, var_cum = 0.0, 0.0
        m_curr = m_anchor
        samples = np.empty(n_samples_per_rung, dtype=np.float64)
        for j, m in enumerate(m_grid):
            m = int(m)
            step = 1 if m > m_curr else -1
            while m_curr != m:
                m_next = m_curr + step * min(slots_per_rung, abs(m - m_curr))
                for s in range(n_samples_per_rung):
                    samples[s] = math.exp(self._eng.seam_rb_log_ratio_to(m_next))
                    for _ in range(n_sweeps_between_samples):
                        self._eng.mc_step()
                mean = float(samples.mean())
                sem = float(samples.std(ddof=1)) / math.sqrt(n_samples_per_rung)
                log_cum += math.log(mean)
                var_cum += (sem / mean) ** 2
                m_curr = m_next
                rung_m.append(m_curr)
                rung_log.append(math.log(mean))
                rung_sem.append(sem / mean)
                self._eng.seam_drag_to(m_curr)
                for _ in range(n_burn_per_rung):
                    self._eng.mc_step()
            log_r[j] = log_cum
            log_r_var[j] = var_cum
        return StringDragLadderResult(
            m_anchor=m_anchor, m_grid=m_grid,
            log_r=log_r, log_r_sem=np.sqrt(log_r_var), r=np.exp(log_r),
            rung_m=np.array(rung_m), rung_log=np.array(rung_log),
            rung_sem=np.array(rung_sem),
            n_samples_per_rung=n_samples_per_rung,
        )


    def run_drag_curve_mirrored(self, m_grid_forward: np.ndarray,
                                n_samples_per_rung: int = 400,
                                n_sweeps_between_samples: int = 1,
                                n_burn_per_rung: int = 5,
                                n_equil_at_anchor: int = 100,
                                slots_per_rung: int = 1) -> StringDragMirroredResult:
        """Mirror-averaged drag curve about the symmetric anchor m = M.

        Runs the left family down ``m_grid_forward`` (strictly decreasing,
        all < M) and the right family up the mirror grid ``2M - m`` (same
        delta values), then geo-means the two branches per point -- even in
        the sweep velocity by the palindrome's transpose symmetry. See the
        class of caveats in StringDragMirroredResult: the win over a single
        branch is odd-part removal and zero-crossing smoothing (clean ~1/M^2
        tail for extrapolation), not a change of convergence order.
        Estimator correctness is gated vs ED in
        tests/engines/integration/test_qaqmc_string_drag_vs_ed.py; the
        convergence systematics in docs/design/seam_drag_curve.md SS6.
        The anchor sits at the symmetric point and is shared by both
        families (its error is common-mode for the whole curve).

        The caller must have equilibrated the full-mask sector at m = M;
        ``n_equil_at_anchor`` re-equilibrates before each family (the second
        family starts from a configuration parked at the first family's far
        end, so this must be > 0 in any real run).
        """
        m_grid_forward = self._validate_drag_grid(m_grid_forward)
        m_anchor = self.M
        if np.any(m_grid_forward >= m_anchor) or np.any(np.diff(m_grid_forward) >= 0):
            raise ValueError("m_grid_forward must be strictly decreasing and < M "
                             "(the mirror pairs are m and 2M - m about the anchor M)")
        kwargs = dict(n_samples_per_rung=n_samples_per_rung,
                      n_sweeps_between_samples=n_sweeps_between_samples,
                      n_burn_per_rung=n_burn_per_rung,
                      n_equil_at_anchor=n_equil_at_anchor,
                      slots_per_rung=slots_per_rung)
        left = self.run_drag_ladder(m_grid_forward, m_anchor=m_anchor, **kwargs)
        m_mirror = 2 * m_anchor - m_grid_forward
        right = self.run_drag_ladder(m_mirror, m_anchor=m_anchor, **kwargs)
        log_r_mirror = 0.5 * (left.log_r + right.log_r)
        log_r_sem = 0.5 * np.sqrt(left.log_r_sem ** 2 + right.log_r_sem ** 2)
        return StringDragMirroredResult(
            m_forward=m_grid_forward, m_mirror=m_mirror,
            log_r_mirror=log_r_mirror, log_r_sem=log_r_sem,
            r_mirror=np.exp(log_r_mirror), left=left, right=right,
        )


def cosine_schedule(k_steps: int) -> np.ndarray:
    """lambda(t) = 0.5*(1 - cos(pi*t)), t = m/K -- slower near the endpoints
    than a linear schedule (document section 20)."""
    t = np.linspace(0.0, 1.0, k_steps + 1)
    return 0.5 * (1.0 - np.cos(np.pi * t))
