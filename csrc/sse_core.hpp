#pragma once
#include "qaqmc_core.hpp"  // RydbergVij, build_rydberg_vij, AliasEntry, RNG helpers
#include "diagonal_observables.hpp"
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
              int neighbor_cutoff = -1,
              const double* box = nullptr, int n_box = 0);

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

    // ── Diagonal observables shared with the QAQMC profile engine ────────────
    // Geometry (loop/string sets, A_v, VBS/SS triangles, occ-SF maps) is set
    // through this member; per-sample measurement on state_ happens in the
    // bindings' run() loop.  Same estimator as measure_density: state_ is the
    // τ=0 basis state of the SSE configuration.
    DiagonalObservables diag_obs;

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

    // ── Checkpoint / warm start ──────────────────────────────────────────────
    std::string get_rng_state() const;
    void        set_rng_state(const std::string& s);

    // Install a saved configuration: spin state at the tau=0 boundary plus the
    // full operator string (length = new M_).  Used by warm-started MPI runs to
    // skip thermalization.  Validates op types / site+bond indices and throws
    // std::runtime_error on inconsistent input.
    void set_config(const int32_t* state, int n_state,
                    const int32_t* types, const int32_t* sites, int len);

private:
    int    N_, M_, n_ops_;
    double Omega_, delta_, Rb_, beta_, epsilon_;
    double norm_N_;         // total alias-table normalisation constant
    double beta_norm_;      // beta * norm_N_ (insert-acceptance numerator)
    double inv_beta_norm_;  // 1 / (beta * norm_N_) — no division in the sweep
    double energy_shift_;   // sum_b W[b,0] = sum_b cij_b

    double time_diag_{0}, time_clus_{0};
    int    mc_steps_{0};

    std::mt19937_64 rng_;
    RydbergVij      vij_;

    // Bond weights: flat (n_bonds_pad * 4), index = b*4 + (ni*2+nj)
    std::vector<double>  bond_W_;
    std::vector<double>  bond_W_rmax_;  // 1/max_s W[b,s] (0 if max <= 0)

    // Single alias table (one per beta/delta, not per time-slice), stored as
    // 16-byte AoS entries so each proposal touches 1-2 cache lines.
    int                     n_alias_;
    std::vector<AliasEntry> alias_entries_;

    std::vector<int32_t> state_;     // current spin state (boundary at tau=0)
    std::vector<int32_t> op_types_;  // length M_
    std::vector<int32_t> op_sites_;  // length M_

    // ── Vertex lists for O(M) cluster update ──────────────────────────────
    std::vector<int32_t> site_op_count_;   // [N_] # single-site ops per site
    std::vector<int32_t> site_op_head_;    // [N_] offset into site_op_list_
    std::vector<int32_t> site_op_list_;    // packed positions of single-site ops

    std::vector<int32_t> site_bond_count_; // [N_] # bond ops touching each site
    std::vector<int32_t> site_bond_head_;  // [N_] offset into site_bond_list_
    // Packed bond-op vertex events: [ p : 32 ][ b : 31 ][ endpoint : 1 ]
    // (endpoint = 0 if the owning site is bonds_i[b], 1 if bonds_j[b]).
    // Filled in ascending p per site with p in the high bits, so packed order
    // == p order and upper_bound can search the packed key directly.  Carrying
    // b avoids the dependent op_sites_[p] load in the segment-Metropolis loop.
    std::vector<int64_t> site_bond_list_;

    // Bond-op spin cache, values 0..3 — int8 quarters the footprint of the
    // random-access reads/XORs in the segment Metropolis.
    std::vector<int8_t>  bond_spin_;
    // True while site_op_count_/site_bond_count_ filled by the last diagonal
    // sweep still describe op_types_/op_sites_ (cluster's 1 <-> -1 toggles are
    // count-neutral); cleared by set_config.
    bool vertex_counts_valid_{false};

    std::vector<int32_t> spin_now_;        // [N_] working spin array
    std::vector<int8_t>  seg_flipped_;     // scratch for segment flip decisions

    void diagonal_update();
    void cluster_update();
    void build_vertex_lists();
    void adjust_M_if_needed();
};
