"""
SSE QMC for Rydberg atom arrays — Numba-accelerated core.

Implementation based on Merali, De Vlugt, Melko (arXiv:2107.00766v3).

Operator string encoding (flat int32 arrays):
  op_types[p]:  0 = identity, 1 = diagonal site, -1 = off-diagonal site,
                2 = diagonal bond
  op_sites[p]:  site index (for ±1) or bond index (for 2), -1 for identity

Energy formula:
  E_physical = -⟨n_ops⟩/β + energy_shift
  where energy_shift = NΩ/2 + Σ_b c_ij accounts for constant terms in the
  operator decomposition.
"""

import numpy as np
from numba import njit, int32


# ─── Alias-table sampling ────────────────────────────────────────────────────

def build_alias_table(weights):
    """Build Walker's alias table for O(1) categorical sampling.

    Args:
        weights: List of non-negative probabilities (need not be normalised).

    Returns:
        (prob_arr, alias_arr): Arrays for alias sampling.
    """
    n = len(weights)
    total = sum(weights)
    prob_arr = np.array([w * n / total for w in weights], dtype=np.float64)
    alias_arr = np.arange(n, dtype=np.int64)
    small, large = [], []
    for i in range(n):
        (small if prob_arr[i] < 1.0 else large).append(i)
    while small and large:
        s_ = small.pop()
        l_ = large.pop()
        alias_arr[s_] = l_
        prob_arr[l_] -= (1.0 - prob_arr[s_])
        (small if prob_arr[l_] < 1.0 else large).append(l_)
    return prob_arr, alias_arr


@njit(cache=True)
def _alias_sample(prob, alias, n):
    """Sample index from alias table in O(1)."""
    i = np.random.randint(0, n)
    if np.random.random() < prob[i]:
        return i
    return alias[i]


# ─── Diagonal update (Eqs. 41-42) ────────────────────────────────────────────

@njit(cache=True)
def diagonal_update(op_types, op_sites, state, M, n_ops,
                    beta, norm_N,
                    bond_sites, bond_W, bond_W_max,
                    alias_prob, alias_idx, n_alias,
                    op_map_kind, op_map_loc, N):
    """Finite-temperature diagonal (insertion/removal) update.

    Traverses the operator string once. For each position:
      - Identity  → attempt insertion with prob min(1, β𝒩/(M-n))
      - Diagonal  → attempt removal  with prob min(1, (M-n+1)/(β𝒩))
      - Off-diag  → propagate the spin state (no accept/reject)

    Args:
        op_types, op_sites: Operator string (modified in place).
        state:    Current spin configuration (propagated and modified in place).
        M:        Operator-string length.
        n_ops:    Current number of non-identity operators.
        beta:     Inverse temperature.
        norm_N:   𝒩 = Σ_x max_α W^α_x (alias-table normalisation).
        bond_sites: (n_bonds, 2) array of bond endpoint site indices.
        bond_W:   (n_bonds, 4) bond weight lookup table.
        bond_W_max: (n_bonds,) maximum bond weight per bond.
        alias_prob, alias_idx, n_alias: Alias table for operator sampling.
        op_map_kind: 0 = site op, 1 = bond op.
        op_map_loc:  site or bond index for each alias entry.
        N:        Number of sites.

    Returns:
        Updated n_ops.
    """
    for p in range(M):
        ot = op_types[p]

        if ot == -1:
            # Off-diagonal site op: propagate state
            state[op_sites[p]] ^= 1

        elif ot == 1 or ot == 2:
            # Diagonal op: attempt removal
            prob_remove = (M - n_ops + 1) / (beta * norm_N)
            if prob_remove > 1.0:
                prob_remove = 1.0
            if np.random.random() < prob_remove:
                op_types[p] = 0
                op_sites[p] = -1
                n_ops -= 1

        elif ot == 0:
            # Identity: attempt insertion
            prob_insert = beta * norm_N / (M - n_ops)
            if prob_insert > 1.0:
                prob_insert = 1.0
            if np.random.random() < prob_insert:
                idx = _alias_sample(alias_prob, alias_idx, n_alias)
                kind = op_map_kind[idx]
                loc = op_map_loc[idx]

                if kind == 0:
                    # Site diagonal op — always accepted (weight is constant)
                    op_types[p] = 1
                    op_sites[p] = loc
                    n_ops += 1
                else:
                    # Bond diagonal op — second-step acceptance (Eq. 40)
                    b = loc
                    si = bond_sites[b, 0]
                    sj = bond_sites[b, 1]
                    w_idx = state[si] * 2 + state[sj]
                    w_actual = bond_W[b, w_idx]
                    w_max = bond_W_max[b]
                    if w_max > 0.0 and np.random.random() < w_actual / w_max:
                        op_types[p] = 2
                        op_sites[p] = b
                        n_ops += 1

    return n_ops


# ─── Cluster update helpers ───────────────────────────────────────────────────

@njit(cache=True)
def _flip_segment_range(state_at, site_i, p_start_excl, p_end_incl, M):
    """Flip state_at[p, site_i] for p in (p_start, p_end] (inclusive, modular).

    Special case: p_end_incl == p_start_excl → flip ALL positions (full loop).
    """
    if p_end_incl > p_start_excl:
        for p in range(p_start_excl + 1, p_end_incl + 1):
            state_at[p, site_i] ^= 1
    elif p_end_incl < p_start_excl:
        # Wrapping segment
        for p in range(p_start_excl + 1, M):
            state_at[p, site_i] ^= 1
        for p in range(0, p_end_incl + 1):
            state_at[p, site_i] ^= 1
    else:
        # Full loop (single segment wrapping onto itself)
        for p in range(M):
            state_at[p, site_i] ^= 1


@njit(cache=True)
def _segment_log_weight_ratio(state_at, op_types, op_sites, site_i,
                               p_start_excl, p_end_incl, M, N,
                               bond_sites, bond_W):
    """Compute log(W'/W) for bond ops inside segment (p_start, p_end].

    Flipping site_i's occupation in this segment changes the weight of
    every bond op that involves site_i and lies within the segment.
    """
    log_w_old = 0.0
    log_w_new = 0.0

    def _proc(p):
        nonlocal log_w_old, log_w_new
        if op_types[p] == 2:
            b = op_sites[p]
            si = bond_sites[b, 0]
            sj = bond_sites[b, 1]
            if si == site_i or sj == site_i:
                ni = state_at[p, si]
                nj = state_at[p, sj]
                w_old = bond_W[b, ni * 2 + nj]
                if si == site_i:
                    w_new = bond_W[b, (1 - ni) * 2 + nj]
                else:
                    w_new = bond_W[b, ni * 2 + (1 - nj)]
                log_w_old += np.log(w_old) if w_old > 1e-300 else -1e10
                log_w_new += np.log(w_new) if w_new > 1e-300 else -1e10

    if p_end_incl > p_start_excl:
        for p in range(p_start_excl + 1, p_end_incl + 1):
            _proc(p)
    elif p_end_incl < p_start_excl:
        for p in range(p_start_excl + 1, M):
            _proc(p)
        for p in range(0, p_end_incl + 1):
            _proc(p)
    else:
        for p in range(M):
            _proc(p)

    return log_w_new - log_w_old


@njit(cache=True)
def _segment_contains_time0(p_start_excl, p_end_incl, M):
    """Return True if time slice 0 lies inside segment (p_start, p_end]."""
    if p_end_incl > p_start_excl:
        # Normal (non-wrapping) segment: check if 0 is in (p_start, p_end]
        return p_start_excl < 0 <= p_end_incl
    else:
        # Wrapping or full-loop: always contains 0
        return True


# ─── Line cluster update (Sec. 3.2) ──────────────────────────────────────────

@njit(cache=True)
def cluster_update(op_types, op_sites, state, M, N,
                   bond_sites, bond_W):
    """Line cluster update (Sec. 3.2 of arXiv:2107.00766v3).

    Operates on the worldline configuration state_at[p, i] — the propagated
    site occupation at each imaginary-time slice.

    For each site i, the algorithm:
    1. Builds state_at by propagating `state` through the operator string.
    2. Identifies all site operators on site i → divides the worldline into
       segments. Segment k spans (site_ops[k], site_ops[(k+1) % n_sops]].
    3. For each segment, computes the Metropolis ratio W'/W from bond ops
       within the segment (site op weights cancel because Ω/2 is constant).
    4. Flips the segment with probability min(1, W'/W).
       Flipping = toggling state_at in the segment; both boundary site ops
       then change type (diag ↔ offdiag) when types are re-derived.
    5. Re-derives all site op types from the updated state_at.

    Parity invariant: each segment flip toggles exactly 2 site op types,
    preserving the even-parity constraint (even # of off-diagonal ops per site).
    """
    if M == 0:
        return

    # Build propagated state array
    state_at = np.empty((M, N), dtype=int32)
    cur = state.copy()
    for p in range(M):
        for s in range(N):
            state_at[p, s] = cur[s]
        if op_types[p] == -1:
            cur[op_sites[p]] ^= 1

    # Process each site independently
    for site_i in range(N):
        # Collect site-op positions in time order
        site_ops = np.empty(M, dtype=int32)
        n_sops = 0
        for p in range(M):
            ot = op_types[p]
            if (ot == 1 or ot == -1) and op_sites[p] == site_i:
                site_ops[n_sops] = p
                n_sops += 1

        if n_sops == 0:
            # No site ops → one full-loop segment; only bond weights matter
            log_ratio = _segment_log_weight_ratio(
                state_at, op_types, op_sites, site_i,
                0, 0, M, N, bond_sites, bond_W)
            if log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio):
                _flip_segment_range(state_at, site_i, 0, 0, M)
                state[site_i] ^= 1
            continue

        for seg in range(n_sops):
            p_start = site_ops[seg]
            p_end   = site_ops[(seg + 1) % n_sops]

            log_ratio = _segment_log_weight_ratio(
                state_at, op_types, op_sites, site_i,
                p_start, p_end, M, N, bond_sites, bond_W)

            do_flip = log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio)
            if do_flip:
                _flip_segment_range(state_at, site_i, p_start, p_end, M)
                if _segment_contains_time0(p_start, p_end, M):
                    state[site_i] ^= 1

    # Re-derive site op types from modified state_at
    for p in range(M):
        ot = op_types[p]
        if ot == 1 or ot == -1:
            site = op_sites[p]
            n_before = state_at[p, site]
            n_after  = state_at[p + 1, site] if p < M - 1 else state[site]
            op_types[p] = 1 if n_before == n_after else -1


# ─── Python wrapper class ─────────────────────────────────────────────────────

class SSE_Rydberg:
    """Finite-temperature SSE QMC simulation for a 1D Rydberg atom chain.

    Uses Numba JIT-compiled kernels for the diagonal update and line cluster
    update. Operator decomposition follows Eq. (13-15) of arXiv:2107.00766v3.

    Args:
        N:       Number of sites.
        Omega:   Rabi frequency.
        delta:   Laser detuning.
        Rb:      Blockade radius (lattice units).
        beta:    Inverse temperature.
        epsilon: Small regularisation to keep all bond weights positive.
        seed:    NumPy random seed.
    """

    def __init__(self, N: int, Omega: float, delta: float, Rb: float,
                 beta: float, epsilon: float = 0.01, seed: int = 42,
                 pos: np.ndarray = None):
        self.N = N
        self.Omega = Omega
        self.delta = delta
        self.Rb = Rb
        self.beta = beta
        self.pos = pos

        np.random.seed(seed)

        # ── Build bonds ──────────────────────────────────────────────────
        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1)

        bonds_i, bonds_j, vij_list = [], [], []
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(self.pos[i] - self.pos[j])
                vij = Omega * (Rb / dist) ** 6
                bonds_i.append(i)
                bonds_j.append(j)
                vij_list.append(vij)
        n_bonds = len(bonds_i)
        self.n_bonds = n_bonds

        self.bond_sites = np.zeros((max(n_bonds, 1), 2), dtype=np.int32)
        for b in range(n_bonds):
            self.bond_sites[b, 0] = bonds_i[b]
            self.bond_sites[b, 1] = bonds_j[b]

        # ── Bond weights (Eq. 15c-f) ─────────────────────────────────────
        # Split detuning equally across bonds: δ_b = δ/(N-1) each
        delta_b = delta / (N - 1) if N > 1 else delta
        bond_W     = np.zeros((max(n_bonds, 1), 4), dtype=np.float64)
        bond_W_max = np.zeros(max(n_bonds, 1), dtype=np.float64)

        # Constant energy shift from operator decomposition
        self.energy_shift = N * Omega / 2.0
        for b in range(n_bonds):
            vij = vij_list[b]
            m1  = min(0.0, delta_b, 2 * delta_b - vij)
            m2  = min(delta_b, 2 * delta_b - vij)
            cij = abs(m1) + epsilon * abs(m2)
            self.energy_shift += cij
            bond_W[b, 0] = cij                          # W(00)
            bond_W[b, 1] = delta_b + cij                # W(01)
            bond_W[b, 2] = delta_b + cij                # W(10)
            bond_W[b, 3] = -vij + 2 * delta_b + cij    # W(11)
            bond_W_max[b] = np.max(bond_W[b])
        self.bond_W     = bond_W
        self.bond_W_max = bond_W_max

        # ── Alias table for operator insertion ───────────────────────────
        weights      = []
        op_map_kind  = []
        op_map_loc   = []
        for i in range(N):
            weights.append(Omega / 2.0)
            op_map_kind.append(0)
            op_map_loc.append(i)
        for b in range(n_bonds):
            weights.append(bond_W_max[b])
            op_map_kind.append(1)
            op_map_loc.append(b)
        self.norm_N    = sum(weights)
        self.alias_prob, self.alias_idx = build_alias_table(weights)
        self.n_alias   = len(weights)
        self.op_map_kind = np.array(op_map_kind, dtype=np.int32)
        self.op_map_loc  = np.array(op_map_loc,  dtype=np.int32)

        # ── SSE operator string ──────────────────────────────────────────
        self.M        = max(20, N * 4)
        self.op_types = np.zeros(self.M,  dtype=np.int32)
        self.op_sites = np.full(self.M, -1, dtype=np.int32)
        self.n_ops    = 0
        self.state    = np.random.randint(0, 2, size=N).astype(np.int32)

    # ── MC step ──────────────────────────────────────────────────────────────

    def mc_step(self):
        """One diagonal update + one line cluster update."""
        self.n_ops = diagonal_update(
            self.op_types, self.op_sites, self.state, self.M, self.n_ops,
            self.beta, self.norm_N,
            self.bond_sites, self.bond_W, self.bond_W_max,
            self.alias_prob, self.alias_idx, self.n_alias,
            self.op_map_kind, self.op_map_loc, self.N)

        cluster_update(
            self.op_types, self.op_sites, self.state, self.M, self.N,
            self.bond_sites, self.bond_W)

    def adjust_M(self):
        """Grow the operator-string buffer if it is more than 80% full."""
        if self.n_ops > 0 and self.n_ops / self.M > 0.8:
            new_M = int(self.M * 1.5) + 10
            new_types = np.zeros(new_M, dtype=np.int32)
            new_sites = np.full(new_M, -1, dtype=np.int32)
            new_types[:self.M] = self.op_types
            new_sites[:self.M] = self.op_sites
            self.op_types = new_types
            self.op_sites = new_sites
            self.M = new_M

    # ── Observables ──────────────────────────────────────────────────────────

    def measure_energy(self) -> float:
        """Instantaneous energy estimator: E = -⟨n_ops⟩/β + energy_shift."""
        return -self.n_ops / self.beta + self.energy_shift

    def measure_density(self) -> float:
        """Instantaneous average site occupation ⟨n⟩ = (1/N) Σ_i n_i."""
        return float(np.mean(self.state))

    # ── Full run ─────────────────────────────────────────────────────────────

    def run(self, n_equil: int = 10000, n_measure: int = 50000) -> dict:
        """Equilibrate then measure.

        Args:
            n_equil:   Number of equilibration MC steps.
            n_measure: Number of measurement MC steps.

        Returns:
            Dictionary with keys:
              energy_mean, energy_err  — binned mean and std-error
              density_mean, density_err
              M                        — final operator-string length
        """
        # Warm-up + equilibration
        self.mc_step()
        self.adjust_M()
        for _ in range(n_equil):
            self.mc_step()
            self.adjust_M()

        # Measurement
        energies  = np.empty(n_measure)
        densities = np.empty(n_measure)
        for step in range(n_measure):
            self.mc_step()
            energies[step]  = self.measure_energy()
            densities[step] = self.measure_density()

        # Bin into 100 bins for error estimation
        n_bins = 100
        bs     = n_measure // n_bins
        e_bins = np.array([np.mean(energies[i*bs:(i+1)*bs])  for i in range(n_bins)])
        d_bins = np.array([np.mean(densities[i*bs:(i+1)*bs]) for i in range(n_bins)])

        return {
            'energy_mean':  float(np.mean(e_bins)),
            'energy_err':   float(np.std(e_bins) / np.sqrt(n_bins)),
            'density_mean': float(np.mean(d_bins)),
            'density_err':  float(np.std(d_bins) / np.sqrt(n_bins)),
            'M':            self.M,
        }
