"""
Core Numba-accelerated QMC update kernels for SSE and QAQMC.
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

@njit(cache=True, nogil=True)
def _alias_sample(prob, alias, n):
    """Sample index from alias table in O(1)."""
    i = np.random.randint(0, n)
    if np.random.random() < prob[i]:
        return i
    return alias[i]

# ─── SSE Updates ─────────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def sse_diagonal_update(op_types, op_sites, state, M, n_ops,
                    beta, norm_N,
                    bond_sites, bond_W, bond_W_max,
                    alias_prob, alias_idx, n_alias,
                    op_map_kind, op_map_loc, N):
    """Finite-temperature SSE diagonal update."""
    for p in range(M):
        ot = op_types[p]
        if ot == -1:
            state[op_sites[p]] ^= 1
        elif ot == 1 or ot == 2:
            prob_remove = (M - n_ops + 1) / (beta * norm_N)
            if prob_remove > 1.0: prob_remove = 1.0
            if np.random.random() < prob_remove:
                op_types[p] = 0
                op_sites[p] = -1
                n_ops -= 1
        elif ot == 0:
            prob_insert = beta * norm_N / (M - n_ops)
            if prob_insert > 1.0: prob_insert = 1.0
            if np.random.random() < prob_insert:
                idx = _alias_sample(alias_prob, alias_idx, n_alias)
                kind = op_map_kind[idx]
                loc = op_map_loc[idx]
                if kind == 0:
                    op_types[p] = 1
                    op_sites[p] = loc
                    n_ops += 1
                else:
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

@njit(cache=True, nogil=True)
def _flip_segment_range_sse(state_at, site_i, p_start_excl, p_end_incl, M):
    """Flip state_at[p, site_i] for p in (p_start, p_end] (mod M)."""
    if p_end_incl > p_start_excl:
        for p in range(p_start_excl + 1, p_end_incl + 1):
            state_at[p, site_i] ^= 1
    elif p_end_incl < p_start_excl:
        for p in range(p_start_excl + 1, M):
            state_at[p, site_i] ^= 1
        for p in range(0, p_end_incl + 1):
            state_at[p, site_i] ^= 1
    else:
        for p in range(M):
            state_at[p, site_i] ^= 1

@njit(cache=True, nogil=True)
def _segment_log_weight_ratio_sse(state_at, op_types, op_sites, site_i,
                               p_start_excl, p_end_incl, M, N,
                               bond_sites, bond_W):
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
        for p in range(p_start_excl + 1, p_end_incl + 1): _proc(p)
    elif p_end_incl < p_start_excl:
        for p in range(p_start_excl + 1, M): _proc(p)
        for p in range(0, p_end_incl + 1): _proc(p)
    else:
        for p in range(M): _proc(p)
    return log_w_new - log_w_old

@njit(cache=True, nogil=True)
def _segment_contains_time0(p_start_excl, p_end_incl, M):
    if p_end_incl > p_start_excl:
        return p_start_excl < 0 <= p_end_incl
    else:
        return True

@njit(cache=True, nogil=True)
def sse_cluster_update(op_types, op_sites, state, M, N, bond_sites, bond_W):
    """Line cluster update for SSE."""
    if M == 0: return
    state_at = np.empty((M, N), dtype=int32)
    cur = state.copy()
    for p in range(M):
        for s in range(N):
            state_at[p, s] = cur[s]
        if op_types[p] == -1:
            cur[op_sites[p]] ^= 1

    for site_i in range(N):
        site_ops = np.empty(M, dtype=int32)
        n_sops = 0
        for p in range(M):
            ot = op_types[p]
            if (ot == 1 or ot == -1) and op_sites[p] == site_i:
                site_ops[n_sops] = p
                n_sops += 1

        if n_sops == 0:
            log_ratio = _segment_log_weight_ratio_sse(state_at, op_types, op_sites, site_i, 0, 0, M, N, bond_sites, bond_W)
            if log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio):
                _flip_segment_range_sse(state_at, site_i, 0, 0, M)
                state[site_i] ^= 1
            continue

        for seg in range(n_sops):
            p_start = site_ops[seg]
            p_end   = site_ops[(seg + 1) % n_sops]
            log_ratio = _segment_log_weight_ratio_sse(state_at, op_types, op_sites, site_i, p_start, p_end, M, N, bond_sites, bond_W)
            do_flip = log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio)
            if do_flip:
                _flip_segment_range_sse(state_at, site_i, p_start, p_end, M)
                if _segment_contains_time0(p_start, p_end, M):
                    state[site_i] ^= 1

    for p in range(M):
        ot = op_types[p]
        if ot == 1 or ot == -1:
            site = op_sites[p]
            n_before = state_at[p, site]
            n_after  = state_at[p + 1, site] if p < M - 1 else state[site]
            op_types[p] = 1 if n_before == n_after else -1

