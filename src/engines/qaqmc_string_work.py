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
    bidirectional: bool = False


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
class StringGrowthAnchorResult:
    """Sector-growth residence ladder estimate of O_C = Z_X / Z_empty.

    Stage k freezes seam bits 0..k-1 ON and lets only bit k toggle at a
    per-stage lambda_k tuned so the two-sector occupancy is balanced; then

        Z_{k+1}/Z_k = P_on/P_off * (1 - lambda_k)/lambda_k     (exact)

    and O_C = prod_k Z_{k+1}/Z_k.  Equilibrium per stage, so slow half-line
    mixing costs sweeps, not validity — unlike the lambda-Jarzynski bridge,
    whose work distribution explodes when a trajectory must thread every
    stage's bottleneck in one sweep (kagome hexagon loop at Rb=2.4: toggle
    acceptance 0-2% on the interior stages, n_eff ~ 2% at K=3600).
    """
    o_c: float
    log_o_c: float
    log_o_c_sem: float        # quadrature over stages
    lambdas: np.ndarray       # (L,) tuned lambda per stage
    p_on: np.ndarray          # (L,) occupancy of the toggling bit
    n_flips: np.ndarray       # (L,) observed sector transitions (mixing check)
    log_r: np.ndarray         # (L,) per-stage log Z-ratio
    log_r_sem: np.ndarray     # (L,)
    n_samples_per_stage: int


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
                        m_anchor: int | None = None,
                        bidirectional: bool = False) -> StringDragLadderResult:
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

        ``bidirectional=True`` (the large-M mode): each rung is estimated
        from BOTH endpoint ensembles -- the forward RB mean sampled at the
        rung's start plus the reverse RB mean sampled at its end (the very
        ensemble the walk equilibrates next anyway) -- and the rung log
        ratio is the symmetric average 0.5*(log fwd_mean - log rev_mean).
        The finite-sample Jensen bias of log<exp(.)> is ~ -sigma^2/(2n) with
        OPPOSITE sign in the two directions, so the symmetric average
        cancels it to leading order.  One-sided ladders accumulate that
        bias coherently over all rungs and across all passes (probes
        27140/27141 at M=3e6: -540 vs -1210 for a -0.6 truth, bias ~ spr^2)
        -- at large M this mode is mandatory; the safe rung log-sd window
        widens from ~0.3 to ~0.5-0.8.  Reverse evaluations consume no RNG,
        so the sampled worldline stream is identical to the one-sided walk;
        the only extra chain work is one final session at the last grid
        point.  Adjacent rung estimates share an endpoint session, so the
        quadrature SEM ignores their (small) covariance -- pass-scatter
        aggregation across independent chains remains the honest error.
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

        if not bidirectional:
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

        # -- bidirectional: symmetric two-ensemble rung estimates ----------
        # Precompute every rung boundary; each position hosts ONE sampling
        # session that serves the incoming rung (reverse eval) and the
        # outgoing rung (forward eval) with the same worldline samples.
        positions = [m_anchor]
        m_curr = m_anchor
        for m in m_grid:
            m = int(m)
            step = 1 if m > m_curr else -1
            while m_curr != m:
                m_curr += step * min(slots_per_rung, abs(m - m_curr))
                positions.append(m_curr)
        grid_at = {}
        gi = 0
        for idx, p in enumerate(positions):
            if gi < n_grid and p == int(m_grid[gi]):
                grid_at[idx] = gi
                gi += 1

        fwd = np.empty(n_samples_per_rung, dtype=np.float64)
        rev = np.empty(n_samples_per_rung, dtype=np.float64)
        prev_fwd = None  # (log mean, rel-sem^2) of the outgoing forward eval
        n_pos = len(positions)
        for i in range(n_pos):
            if i > 0:
                self._eng.seam_drag_to(positions[i])
                for _ in range(n_burn_per_rung):
                    self._eng.mc_step()
            has_fwd = i + 1 < n_pos
            for s in range(n_samples_per_rung):
                if i > 0:
                    rev[s] = math.exp(
                        self._eng.seam_rb_log_ratio_to(positions[i - 1]))
                if has_fwd:
                    fwd[s] = math.exp(
                        self._eng.seam_rb_log_ratio_to(positions[i + 1]))
                for _ in range(n_sweeps_between_samples):
                    self._eng.mc_step()
            if i > 0:
                r_mean = float(rev.mean())
                r_sem = float(rev.std(ddof=1)) / math.sqrt(n_samples_per_rung)
                f_log, f_var = prev_fwd
                step_log = 0.5 * (f_log - math.log(r_mean))
                step_var = 0.25 * (f_var + (r_sem / r_mean) ** 2)
                log_cum += step_log
                var_cum += step_var
                rung_m.append(positions[i])
                rung_log.append(step_log)
                rung_sem.append(math.sqrt(step_var))
            if has_fwd:
                f_mean = float(fwd.mean())
                f_sem = float(fwd.std(ddof=1)) / math.sqrt(n_samples_per_rung)
                prev_fwd = (math.log(f_mean), (f_sem / f_mean) ** 2)
            if i in grid_at:
                log_r[grid_at[i]] = log_cum
                log_r_var[grid_at[i]] = var_cum
        return StringDragLadderResult(
            m_anchor=m_anchor, m_grid=m_grid,
            log_r=log_r, log_r_sem=np.sqrt(log_r_var), r=np.exp(log_r),
            rung_m=np.array(rung_m), rung_log=np.array(rung_log),
            rung_sem=np.array(rung_sem),
            n_samples_per_rung=n_samples_per_rung,
            bidirectional=True,
        )


    def run_drag_curve_mirrored(self, m_grid_forward: np.ndarray,
                                n_samples_per_rung: int = 400,
                                n_sweeps_between_samples: int = 1,
                                n_burn_per_rung: int = 5,
                                n_equil_at_anchor: int = 100,
                                slots_per_rung: int = 1,
                                bidirectional: bool = False) -> StringDragMirroredResult:
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
                      slots_per_rung=slots_per_rung,
                      bidirectional=bidirectional)
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


    # ── Growth residence-ladder anchor ───────────────────────────────────

    def _stage_occupancy(self, k: int, lam: float, n_samples: int,
                         n_sweeps: int, n_attempts: int):
        """Sample bit k's occupancy at fixed lambda (other bits frozen);
        returns (occupancy series, n_flips)."""
        occ = np.empty(n_samples, dtype=np.int8)
        flips = 0
        prev = (self._eng.seam_mask >> k) & 1
        for i in range(n_samples):
            # cluster_update stales the seam snapshots and only
            # diagonal_update refreshes them; a raw attempt_string_toggle
            # trusts the cache (same guard topology_sweep applies on entry —
            # skipping this biases the residence ratio lambda-dependently).
            self._eng.recompute_seam_snapshots()
            for _ in range(n_attempts):
                self._eng.attempt_string_toggle(k, lam)
            for _ in range(n_sweeps):
                self._eng.mc_step()
            cur = (self._eng.seam_mask >> k) & 1
            flips += int(cur != prev)
            prev = cur
            occ[i] = cur
        return occ, flips

    def run_growth_residence_ladder(self, n_samples_per_stage: int = 4000,
                                    n_sweeps_between_samples: int = 1,
                                    n_equil_per_stage: int = 200,
                                    n_tune_samples: int = 300,
                                    tune_rounds: int = 3,
                                    n_toggle_attempts: int = 4,
                                    n_equil_at_lambda: int = 200,
                                    start_bit_on: bool = False,
                                    stage_lambdas: np.ndarray | None = None
                                    ) -> StringGrowthAnchorResult:
        """Anchor O_C via the sector-growth residence ladder.

        Grows the string one seam bit at a time IN THE ORDER of
        ``string_sites`` (order affects efficiency — grow along the
        string/loop adjacency — not correctness; the product telescopes).
        Each stage: freeze bits 0..k-1 ON (set_seam_mask_consistent, so
        worldline closure is repaired), equilibrate, autotune lambda_k so
        the toggling bit's occupancy is balanced, then sample the residence
        ratio. Ends with the FULL mask set consistently, ready for a drag
        phase (decorrelate before using the configuration).

        Errors: per-stage sem of the log-odds from max(blocked scatter,
        sqrt(8/n_flips)) — the flip count is the honest effective sample
        size when transitions are rare; stages with n_flips < ~10 should be
        rerun with more samples (check ``n_flips`` in the result).
        ``stage_lambdas`` overrides the autotune (e.g. rank 0's tuned values
        broadcast to all ranks for a deterministic protocol).

        Slow-mixing hygiene (the production kagome lesson — consistency
        probes 27114/15/16 disagreed by z~8-20 before these): recording only
        starts after ``n_equil_at_lambda`` samples AT the final lambda (a
        tune-to-production lambda jump otherwise leaves a long occupancy
        transient); ``start_bit_on`` lets half the chains start in the ON
        sector so initialization transients cancel to first order across a
        pooled ensemble; ``n_toggle_attempts`` > 1 multiplies the flip rate
        cheaply (a toggle walk costs far less than an mc_step).
        """
        L = self._length
        if L < 1:
            raise RuntimeError("call set_string_sites() first")
        if stage_lambdas is not None:
            stage_lambdas = np.asarray(stage_lambdas, dtype=np.float64)
            if stage_lambdas.shape != (L,):
                raise ValueError(f"stage_lambdas must have shape ({L},)")

        lambdas = np.empty(L)
        p_on = np.empty(L)
        n_flips_arr = np.zeros(L, dtype=np.int64)
        log_r = np.empty(L)
        log_r_sem = np.empty(L)
        for k in range(L):
            base_mask = (1 << k) - 1
            self._eng.set_seam_mask_consistent(base_mask)
            for _ in range(n_equil_per_stage):
                self._eng.mc_step()

            if stage_lambdas is not None:
                lam = float(stage_lambdas[k])
            else:
                lam = 0.5
                for _ in range(tune_rounds):
                    occ, _ = self._stage_occupancy(
                        k, lam, n_tune_samples, n_sweeps_between_samples,
                        n_toggle_attempts)
                    p = float(occ.mean())
                    if p <= 0.0 or p >= 1.0:
                        # bit never crossed: push the g-odds hard toward the
                        # unseen side and try again
                        odds = lam / (1.0 - lam)
                        odds *= 30.0 if p <= 0.0 else 1.0 / 30.0
                        lam = odds / (1.0 + odds)
                    else:
                        # balance: choose lambda' so g-odds cancels the
                        # measured physical ratio
                        r_phys = (p / (1.0 - p)) * (1.0 - lam) / lam
                        lam = 1.0 / (1.0 + r_phys)
                    lam = min(max(lam, 1e-9), 1.0 - 1e-9)
            lambdas[k] = lam

            if start_bit_on:
                self._eng.set_seam_mask_consistent(base_mask | (1 << k))
                for _ in range(max(n_equil_per_stage // 4, 10)):
                    self._eng.mc_step()
            # burn in AT the final lambda before recording (toggles active)
            if n_equil_at_lambda > 0:
                self._stage_occupancy(k, lam, n_equil_at_lambda,
                                      n_sweeps_between_samples,
                                      n_toggle_attempts)
            occ, flips = self._stage_occupancy(
                k, lam, n_samples_per_stage, n_sweeps_between_samples,
                n_toggle_attempts)
            p = float(occ.mean())
            n_flips_arr[k] = flips
            if p <= 0.0 or p >= 1.0:
                # sector never visited: the ratio is unresolved at this
                # budget — poison the estimate rather than fake a number
                log_r[k] = -math.inf if p <= 0.0 else math.inf
                log_r_sem[k] = math.inf
                p_on[k] = p
                continue
            p_on[k] = p
            # blocked scatter on the logit (delta method), floored by the
            # flip-count bound
            n_blocks = 16
            blocks = occ[:(occ.size // n_blocks) * n_blocks].reshape(n_blocks, -1)
            bp = blocks.mean(axis=1)
            valid = (bp > 0) & (bp < 1)
            if valid.sum() >= 4:
                sem_p = float(bp.std(ddof=1)) / math.sqrt(n_blocks)
                sem_block = sem_p / (p * (1.0 - p))
            else:
                sem_block = 0.0
            sem_flip = math.sqrt(8.0 / max(flips, 1))
            log_r[k] = (math.log(p / (1.0 - p))
                        + math.log((1.0 - lam) / lam))
            log_r_sem[k] = max(sem_block, sem_flip)

        # leave the engine in the full-mask sector for a subsequent drag
        self._eng.set_seam_mask_consistent(self._full_mask())

        log_o_c = float(np.sum(log_r))
        log_o_c_sem = float(np.sqrt(np.sum(np.minimum(log_r_sem, 1e150) ** 2)))
        return StringGrowthAnchorResult(
            o_c=float(np.exp(log_o_c)), log_o_c=log_o_c,
            log_o_c_sem=log_o_c_sem,
            lambdas=lambdas, p_on=p_on, n_flips=n_flips_arr,
            log_r=log_r, log_r_sem=log_r_sem,
            n_samples_per_stage=n_samples_per_stage,
        )


def cosine_schedule(k_steps: int) -> np.ndarray:
    """lambda(t) = 0.5*(1 - cos(pi*t)), t = m/K -- slower near the endpoints
    than a linear schedule (document section 20)."""
    t = np.linspace(0.0, 1.0, k_steps + 1)
    return 0.5 * (1.0 - np.cos(np.pi * t))
