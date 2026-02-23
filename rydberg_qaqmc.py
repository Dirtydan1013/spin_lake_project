"""
Quasi-Adiabatic Quantum Monte Carlo (QAQMC) for Rydberg atom arrays.
Based on arXiv:1212.4815 and arXiv:2107.00766v3.
"""

import numpy as np
from numba import njit, int32

@njit(cache=True)
def qaqmc_diagonal_update(op_types, op_sites, state,
                            M_total,
                            bond_sites, bond_W_all, bond_W_max_all,
                            n_alias_all, alias_prob_all, alias_idx_all,
                            op_map_kind_all, op_map_loc_all,
                            site_W, site_W_max, N):
    """Zero-temperature QAQMC diagonal update.
    
    The projector length 2M is fixed and there are NO identity operators.
    At every imaginary time slice p, we remove the existing diagonal operator 
    and insert a new one sampled from the local Hamiltonian H(p).
    
    The parameter sweep is encoded via the index p.
    
    Args:
        op_types, op_sites: Operator string of length 2M.
        state: Initial spin config (boundary condition |0>).
        M_total: Total length 2M.
        bond_sites: (n_bonds, 2)
        bond_W_all: (2M, n_bonds, 4) - bond weights at every time step
        bond_W_max_all: (2M, n_bonds)
        n_alias_all: (2M,)
        alias_prob_all: (2M, n_alias_max)
        alias_idx_all: (2M, n_alias_max)
        op_map_kind_all: (2M, n_alias_max)
        op_map_loc_all: (2M, n_alias_max)
        site_W: Weight of site ops = Omega / 2
        site_W_max: Max weight
        N: Number of sites
    """
    
    for p in range(M_total):
        ot = op_types[p]
        
        if ot == -1:
            # Off-diagonal, propagate and continue
            state[op_sites[p]] ^= 1
        elif ot == 1 or ot == 2:
            # It's a diagonal operator. Remove it and propose to replace it
            inserted = False
            while not inserted:
                # Sample from alias table at time slice p
                n_alias_p = n_alias_all[p]
                
                # Inline alias sample
                i = np.random.randint(0, n_alias_p)
                if np.random.random() < alias_prob_all[p, i]:
                    idx = i
                else:
                    idx = alias_idx_all[p, i]
                    
                kind = op_map_kind_all[p, idx]
                loc = op_map_loc_all[p, idx]
                
                if kind == 0:
                    # Site op
                    op_types[p] = 1
                    op_sites[p] = loc
                    inserted = True
                else:
                    # Bond op
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

@njit(cache=True)
def _segment_log_weight_ratio_qaqmc(state_at, op_types, op_sites, site_i,
                               p_start_excl, p_end_incl, M_total, N,
                               bond_sites, bond_W_all):
    """Compute log(W'/W) for QAQMC line clusters.
    Similar to SSE but bond weights are time-dependent.
    """
    log_w_old = 0.0
    log_w_new = 0.0
    
    # In QAQMC with open boundaries, segments do not wrap around if they touch the boundary.
    # The pseudo-boundaries at p=0 and p=M_total are fixed.
    
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
                if si == site_i:
                    w_new = bond_W_all[p, b, (1 - ni) * 2 + nj]
                else:
                    w_new = bond_W_all[p, b, ni * 2 + (1 - nj)]
                log_w_old += np.log(w_old) if w_old > 1e-300 else -1e30
                log_w_new += np.log(w_new) if w_new > 1e-300 else -1e30

    # In strict open boundaries, start < end.
    for p in range(max(0, p_start_excl + 1), min(M_total, p_end_incl + 1)):
        _proc(p)
        
    return log_w_new - log_w_old


@njit(cache=True)
def _flip_segment_range_qaqmc(state_at, site_i, p_start_excl, p_end_incl, M_total):
    """Flip state inside segment. Strictly open boundary version."""
    for p in range(max(0, p_start_excl + 1), min(M_total, p_end_incl + 1)):
        state_at[p, site_i] ^= 1


@njit(cache=True)
def qaqmc_cluster_update(op_types, op_sites, state, M_total, N,
                   bond_sites, bond_W_all):
    """Line cluster update for QAQMC with open boundaries in imaginary time.
    
    At p=0 and p=2M, the boundary state is pinned to |00...0> corresponding to large negative delta.
    Segments that reach exactly p=0 or p=2M cannot be flipped since the boundaries are constrained.
    """
    if M_total == 0:
        return
        
    state_at = np.empty((M_total, N), dtype=int32)
    cur = state.copy()
    for p in range(M_total):
        for s in range(N):
            state_at[p, s] = cur[s]
        if op_types[p] == -1:
            cur[op_sites[p]] ^= 1

    for site_i in range(N):
        # Positions:
        site_ops = np.empty(M_total + 2, dtype=int32)
        
        # Artificial boundaries corresponding to the fixed ground state at large negative delta
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
            
            # If the segment touches either extreme boundary, it represents a line that connects 
            # to the pinned edge. We freeze this segment (cannot flip occupation).
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
            
            
def build_qaqmc_alias_tables(M_total, N, n_bonds, Omega, delta_sched, bond_vij, epsilon=0.01):
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
        delta_b = delta / (N - 1) if N > 1 else delta
        
        weights = []
        op_map_kind = []
        op_map_loc = []
        
        for i in range(N):
            weights.append(Omega / 2.0)
            op_map_kind.append(0)
            op_map_loc.append(i)
            
        for b in range(n_bonds):
            vij = bond_vij[b]
            m1 = min(0.0, delta_b, 2 * delta_b - vij)
            m2 = min(delta_b, 2 * delta_b - vij)
            cij = abs(m1) + epsilon * abs(m2)
            
            bond_W_all[p, b, 0] = cij
            bond_W_all[p, b, 1] = delta_b + cij
            bond_W_all[p, b, 2] = delta_b + cij
            bond_W_all[p, b, 3] = -vij + 2 * delta_b + cij
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
    

class QAQMC_Rydberg:
    def __init__(self, N: int, Omega: float, delta_min: float, delta_max: float, Rb: float,
                 M: int, epsilon: float = 0.01, seed: int = 42, pos: np.ndarray = None):
        self.N = N
        self.Omega = Omega
        self.Rb = Rb
        
        self.M = M
        self.M_total = 2 * M
        
        self.pos = pos
        if self.pos is None:
            self.pos = np.arange(N).reshape(-1, 1)

        np.random.seed(seed)
        
        bonds_i, bonds_j, vij_list = [], [], []
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(self.pos[i] - self.pos[j])
                vij = Omega * (Rb / dist) ** 6
                bonds_i.append(i)
                bonds_j.append(j)
                vij_list.append(vij)
        
        n_bonds = len(bonds_i)
        self.bond_sites = np.zeros((max(n_bonds, 1), 2), dtype=np.int32)
        for b in range(n_bonds):
            self.bond_sites[b, 0] = bonds_i[b]
            self.bond_sites[b, 1] = bonds_j[b]
            
        # The evolution sweep delta_min -> delta_max -> delta_min
        delta_sched = np.empty(self.M_total, dtype=np.float64)
        for p in range(self.M):
            # linear iter
            delta_sched[p] = delta_min + (delta_max - delta_min) * (p / self.M)
        for p in range(self.M, self.M_total):
            delta_sched[p] = delta_max - (delta_max - delta_min) * ((p - self.M) / self.M)
            
        # Build memory for parameters across all imaginary time slices
        res = build_qaqmc_alias_tables(self.M_total, N, n_bonds, Omega, delta_sched, vij_list, epsilon)
        self.bond_W_all = res[0]
        self.bond_W_max_all = res[1]
        self.n_alias_all = res[2]
        self.alias_prob_all = res[3]
        self.alias_idx_all = res[4]
        self.op_map_kind_all = res[5]
        self.op_map_loc_all = res[6]
        
        self.site_W = Omega / 2.0
        self.site_W_max = Omega / 2.0
        
        # Initialize QAQMC config
        self.op_types = np.ones(self.M_total, dtype=np.int32)  # initialize to diagonal sites
        self.op_sites = np.zeros(self.M_total, dtype=np.int32)
        
        # Initial state is pinned boundary |0...0> corresponding to delta_min << 0
        self.state = np.zeros(N, dtype=np.int32)

    def mc_step(self):
        # We start propagation always from |0> due to strong negative delta_min boundaries
        boundary_state = np.zeros(self.N, dtype=np.int32)
        
        qaqmc_diagonal_update(
            self.op_types, self.op_sites, boundary_state,
            self.M_total,
            self.bond_sites, self.bond_W_all, self.bond_W_max_all,
            self.n_alias_all, self.alias_prob_all, self.alias_idx_all,
            self.op_map_kind_all, self.op_map_loc_all,
            self.site_W, self.site_W_max, self.N)
            
        # Boundary state |0>
        boundary_state = np.zeros(self.N, dtype=np.int32)
            
        qaqmc_cluster_update(
            self.op_types, self.op_sites, boundary_state, 
            self.M_total, self.N,
            self.bond_sites, self.bond_W_all)
            
    def measure_symmetric(self):
        """Measures observables at the midpoint p=M where parameter is delta_max."""
        cur = np.zeros(self.N, dtype=np.int32)
        for p in range(self.M):
            if self.op_types[p] == -1:
                cur[self.op_sites[p]] ^= 1
        
        return np.mean(cur)
        
    def run(self, n_equil=5000, n_measure=10000):
        for _ in range(n_equil):
            self.mc_step()
            
        densities = np.empty(n_measure)
        for step in range(n_measure):
            self.mc_step()
            densities[step] = self.measure_symmetric()
            
        n_bins = 50
        bs = n_measure // n_bins
        d_bins = np.array([np.mean(densities[i*bs:(i+1)*bs]) for i in range(n_bins)])
        
        return {
            'density_mean': float(np.mean(d_bins)),
            'density_err':  float(np.std(d_bins) / np.sqrt(n_bins)),
        }
