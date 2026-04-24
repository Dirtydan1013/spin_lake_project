"""
Core Numba-accelerated QMC update kernels for SSE and QAQMC.
"""

import numpy as np
from numba import njit, int32

# ─── Alias-table sampling ────────────────────────────────────────────────────

def build_qaqmc_alias_tables(M_total, N, n_bonds, Omega, delta_sched, bond_vij, bond_si, bond_sj, coord_number, epsilon=0.01):
    max_alias = N + n_bonds
    
    bond_W_all = np.zeros((M_total, max(n_bonds, 1), 4), dtype=np.float64)
    bond_W_max_all = np.zeros((M_total, max(n_bonds, 1)), dtype=np.float64)
    
    n_alias_all = np.zeros(M_total, dtype=np.int32)
    alias_prob_all = np.zeros((M_total, max_alias), dtype=np.float64)
    alias_idx_all = np.zeros((M_total, max_alias), dtype=np.int64) # Alias arrays output int64
    op_map_kind_all = np.zeros((M_total, max_alias), dtype=np.int32)
    op_map_loc_all = np.zeros((M_total, max_alias), dtype=np.int32)
    
    for p in range(M_total):
        delta = delta_sched[p]
        
        weights = []
        op_map_kind = []
        op_map_loc = []
        
        for i in range(N):
            weights.append(Omega / 2.0)
            op_map_kind.append(0)
            op_map_loc.append(i)
            
        for b in range(n_bonds):
            vij = bond_vij[b]
            si, sj = bond_si[b], bond_sj[b]
            delta_i = delta / coord_number[si] if coord_number[si] > 0 else 0.0
            delta_j = delta / coord_number[sj] if coord_number[sj] > 0 else 0.0
            
            raw0 = 0.0
            raw1 = delta_j
            raw2 = delta_i
            raw3 = -vij + delta_i + delta_j
            m_min = min(raw0, raw1, raw2, raw3)
            m_abs = min(abs(raw0), abs(raw1), abs(raw2), abs(raw3))
            cij = ((-m_min) if m_min < 0.0 else 0.0) + epsilon * m_abs
            
            bond_W_all[p, b, 0] = raw0 + cij
            bond_W_all[p, b, 1] = raw1 + cij
            bond_W_all[p, b, 2] = raw2 + cij
            bond_W_all[p, b, 3] = raw3 + cij
            bmax = np.max(bond_W_all[p, b])
            bond_W_max_all[p, b] = bmax
            
            weights.append(bmax)
            op_map_kind.append(1)
            op_map_loc.append(b)
            
        n_a = len(weights)
        n_alias_all[p] = n_a
        op_map_kind_all[p, :n_a] = op_map_kind
        op_map_loc_all[p, :n_a] = op_map_loc
        
        total = sum(weights)
        prob_arr = np.array([w * n_a / total for w in weights], dtype=np.float64)
        alias_arr = np.arange(n_a, dtype=np.int64)
        small, large = [], []
        for i in range(n_a):
            (small if prob_arr[i] < 1.0 else large).append(i)
        while small and large:
            s_ = small.pop()
            l_ = large.pop()
            alias_arr[s_] = l_
            prob_arr[l_] -= (1.0 - prob_arr[s_])
            (small if prob_arr[l_] < 1.0 else large).append(l_)
            
        alias_prob_all[p, :n_a] = prob_arr
        alias_idx_all[p, :n_a] = alias_arr

    return bond_W_all, bond_W_max_all, n_alias_all, alias_prob_all, alias_idx_all, op_map_kind_all, op_map_loc_all


@njit(cache=True, nogil=True)
def _alias_sample(prob, alias, n):
    """Sample index from alias table in O(1)."""
    i = np.random.randint(0, n)
    if np.random.random() < prob[i]:
        return i
    return alias[i]

# ─── QAQMC Updates ───────────────────────────────────────────────────────────

@njit(cache=True, nogil=True)
def qaqmc_diagonal_update(op_types, op_sites, state, M_total,
                          bond_sites, bond_W_all, bond_W_max_all,
                          n_alias_all, alias_prob_all, alias_idx_all,
                          op_map_kind_all, op_map_loc_all,
                          site_W, site_W_max, N):
    """Zero-temperature QAQMC diagonal update (fixed length, no identity ops)."""
    for p in range(M_total):
        ot = op_types[p]
        if ot == -1:
            state[op_sites[p]] ^= 1
        elif ot == 1 or ot == 2:
            inserted = False
            while not inserted:
                n_alias_p = n_alias_all[p]
                i = np.random.randint(0, n_alias_p)
                if np.random.random() < alias_prob_all[p, i]: idx = i
                else: idx = alias_idx_all[p, i]
                    
                kind = op_map_kind_all[p, idx]
                loc = op_map_loc_all[p, idx]
                
                if kind == 0:
                    op_types[p] = 1
                    op_sites[p] = loc
                    inserted = True
                else:
                    b = loc
                    si = bond_sites[b, 0]
                    sj = bond_sites[b, 1]
                    w_idx = state[si] * 2 + state[sj]
                    w_actual = bond_W_all[p, b, w_idx]
                    w_max = bond_W_max_all[p, b]
                    if w_max > 0.0 and np.random.random() < w_actual / w_max:
                        op_types[p] = 2
                        op_sites[p] = b
                        inserted = True

@njit(cache=True, nogil=True)
def _flip_segment_range_qaqmc(state_at, site_i, p_start_excl, p_end_incl, M_total):
    """Strictly open bounded segment flip function."""
    for p in range(max(0, p_start_excl + 1), min(M_total, p_end_incl + 1)):
        state_at[p, site_i] ^= 1

@njit(cache=True, nogil=True)
def _segment_log_weight_ratio_qaqmc(state_at, op_types, op_sites, site_i,
                               p_start_excl, p_end_incl, M_total, N,
                               bond_sites, bond_W_all):
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
                w_old = bond_W_all[p, b, ni * 2 + nj]
                if si == site_i: w_new = bond_W_all[p, b, (1 - ni) * 2 + nj]
                else: w_new = bond_W_all[p, b, ni * 2 + (1 - nj)]
                log_w_old += np.log(w_old) if w_old > 1e-300 else -1e30
                log_w_new += np.log(w_new) if w_new > 1e-300 else -1e30

    for p in range(max(0, p_start_excl + 1), min(M_total, p_end_incl + 1)):
        _proc(p)
    return log_w_new - log_w_old

@njit(cache=True, nogil=True)
def qaqmc_cluster_update(op_types, op_sites, state, M_total, N, bond_sites, bond_W_all):
    """Open boundary cluster update for QAQMC."""
    if M_total == 0: return
    state_at = np.empty((M_total, N), dtype=int32)
    cur = state.copy()
    for p in range(M_total):
        for s in range(N): state_at[p, s] = cur[s]
        if op_types[p] == -1: cur[op_sites[p]] ^= 1

    for site_i in range(N):
        site_ops = np.empty(M_total + 2, dtype=int32)
        n_sops = 0
        site_ops[n_sops] = -1
        n_sops += 1
        
        for p in range(M_total):
            ot = op_types[p]
            if (ot == 1 or ot == -1) and op_sites[p] == site_i:
                site_ops[n_sops] = p
                n_sops += 1
                
        site_ops[n_sops] = M_total
        n_sops += 1
        
        for seg in range(n_sops - 1):
            p_start = site_ops[seg]
            p_end = site_ops[seg + 1]
            if p_start == -1 or p_end == M_total:
                continue
            log_ratio = _segment_log_weight_ratio_qaqmc(
                state_at, op_types, op_sites, site_i,
                p_start, p_end, M_total, N, bond_sites, bond_W_all)
            do_flip = log_ratio >= 0.0 or np.random.random() < np.exp(log_ratio)
            if do_flip:
                _flip_segment_range_qaqmc(state_at, site_i, p_start, p_end, M_total)

    for p in range(M_total):
        ot = op_types[p]
        if ot == 1 or ot == -1:
            site = op_sites[p]
            n_before = state_at[p, site]
            n_after = state_at[p + 1, site] if p < M_total - 1 else cur[site]  
            op_types[p] = 1 if n_before == n_after else -1
