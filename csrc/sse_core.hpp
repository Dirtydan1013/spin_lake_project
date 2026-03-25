#pragma once
#include "qaqmc_core.hpp"  // RydbergVij, build_rydberg_vij, QAQMCEngine::compute_bond_W_inline
#include <string>
#include <cstdint>

// ─── SSEEngine ────────────────────────────────────────────────────────────────
//
// Finite-temperature Stochastic Series Expansion QMC for the Rydberg Hamiltonian.
//
//   H = -(Omega/2) sum_i sigma_x_i  +  delta * sum_i n_i  +  sum_{i<j} V_ij n_i n_j
//
// The operator string has variable effective length (non-identity ops = n_ops_)
// within a fixed-capacity buffer of size M_.  M_ grows automatically via
// adjust_M_if_needed() after each mc_step().
//
// Operator types:
//   0  = identity (empty slot — can be filled by diagonal update)
//  -1  = off-diagonal single-site op  (sigma_x: flips spin i)
//   1  = diagonal   single-site op
//   2  = diagonal   bond op  (V_ij + detuning term)
//
// Bond weights follow the QAQMC convention (asymmetric delta per endpoint):
//   delta_i = delta / coord_number[i],  delta_j = delta / coord_number[j]
//   raw0 = 0,  raw1 = delta_j,  raw2 = delta_i,  raw3 = -V_ij + delta_i + delta_j
//   cij  = max(0, -min(raws)) + epsilon * min(|raws|)
//   W[s] = raw_s + cij  (all >= 0)
//
// When neighbor_cutoff is not set (== -1), all N*(N-1)/2 bonds are included and
// coord_number[i] == N-1 for all i — reproducing the Python SSE formula
// delta_b = delta / (N-1).

class SSEEngine {
public:
    // ── Constructor ──────────────────────────────────────────────────────────
    SSEEngine(int N, double Omega, double delta, double Rb,
              double beta, double epsilon, uint64_t seed,
              const double* pos, int pos_dim,
              int neighbor_cutoff = -1);

    // ── Core update ──────────────────────────────────────────────────────────
    void mc_step();

    // ── Scalar accessors ─────────────────────────────────────────────────────
    int    get_N()      const { return N_; }
    int    get_M()      const { return M_; }
    int    get_n_ops()  const { return n_ops_; }
    double get_beta()   const { return beta_; }
    double get_norm_N() const { return norm_N_; }

    // ── Instant observables (measured on current state_) ─────────────────────
    double measure_energy()  const;   // -n_ops/beta + sum_b cij_b
    double measure_density() const;   // mean(state_)
    double measure_mz()      const;   // staggered mz = (1/N) sum_i (-1)^i (n_i - 0.5)

    // ── Array accessors ──────────────────────────────────────────────────────
    const std::vector<int32_t>& get_state()           const { return state_; }
    const std::vector<int32_t>& get_op_types()        const { return op_types_; }
    const std::vector<int32_t>& get_op_sites()        const { return op_sites_; }
    const std::vector<int>&     get_bond_sites_flat() const { return vij_.bond_sites_flat; }

    // ── Profiling ────────────────────────────────────────────────────────────
    double get_time_diag() const { return time_diag_; }
    double get_time_clus() const { return time_clus_; }
    int    get_mc_steps()  const { return mc_steps_; }
    void   reset_timers()        { time_diag_ = 0; time_clus_ = 0; mc_steps_ = 0; }

    // ── RNG checkpoint ───────────────────────────────────────────────────────
    std::string get_rng_state() const;
    void        set_rng_state(const std::string& s);

private:
    int    N_, M_, n_ops_;
    double Omega_, delta_, Rb_, beta_, epsilon_;
    double norm_N_;        // total alias-table normalisation constant
    double energy_shift_;  // sum_b W[b,0] = sum_b cij_b

    double time_diag_{0}, time_clus_{0};
    int    mc_steps_{0};

    std::mt19937_64 rng_;
    RydbergVij      vij_;

    // Bond weights: flat (n_bonds_pad * 4), index = b*4 + (ni*2+nj)
    std::vector<double>  bond_W_;
    std::vector<double>  bond_W_max_;  // max over 4 spin states for bond b

    // Single alias table (one per beta/delta, not per time-slice)
    int                  n_alias_;
    std::vector<double>  alias_prob_;
    std::vector<int64_t> alias_idx_;
    std::vector<int>     op_map_kind_;  // 0 = site op, 1 = bond op
    std::vector<int>     op_map_loc_;   // site index or bond index

    std::vector<int32_t> state_;     // current spin state (boundary at tau=0)
    std::vector<int32_t> op_types_;  // length M_
    std::vector<int32_t> op_sites_;  // length M_

    // ── Vertex lists for O(M) cluster update ──────────────────────────────
    std::vector<int32_t> site_op_count_;   // [N_] # single-site ops per site
    std::vector<int32_t> site_op_head_;    // [N_] offset into site_op_list_
    std::vector<int32_t> site_op_list_;    // packed positions of single-site ops

    std::vector<int32_t> site_bond_count_; // [N_] # bond ops touching each site
    std::vector<int32_t> site_bond_head_;  // [N_] offset into site_bond_list_
    std::vector<int32_t> site_bond_list_;  // packed positions of bond ops per site

    std::vector<int32_t> bond_spin_;       // [M_] spin-index (ni*2+nj) at bond-op positions
    std::vector<int32_t> spin_now_;        // [N_] working spin array
    std::vector<int8_t>  seg_flipped_;     // scratch for segment flip decisions

    void diagonal_update();
    void cluster_update();
    void build_vertex_lists();
    void adjust_M_if_needed();
};
