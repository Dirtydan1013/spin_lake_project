#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef QAQMC_USE_OPENMP
#include <omp.h>
#endif

// ─── V_ij builder ────────────────────────────────────────────────────────────

struct RydbergVij {
    std::vector<int> bonds_i;
    std::vector<int> bonds_j;
    std::vector<double> vij_list;
    std::vector<int> bond_sites_flat; // (n_bonds * 2), row-major
    std::vector<int> coord_number;    // z_eff[i]: # active bonds touching site i
    int n_bonds;
};

RydbergVij build_rydberg_vij(int N, double Omega, double Rb,
                              const double* pos, int pos_dim,
                              int neighbor_cutoff = -1);

// ─── Alias Table ─────────────────────────────────────────────────────────────

struct AliasTable {
    // All arrays: first dimension = M_total (time slices)
    std::vector<double> bond_W_all;      // (M_total * n_bonds_pad * 4)
    std::vector<double> bond_W_max_all;  // (M_total * n_bonds_pad)
    std::vector<int>    n_alias_all;     // (M_total)
    std::vector<double> alias_prob_all;  // (M_total * max_alias)
    std::vector<int64_t> alias_idx_all;  // (M_total * max_alias)
    std::vector<int>    op_map_kind_all; // (M_total * max_alias)
    std::vector<int>    op_map_loc_all;  // (M_total * max_alias)
    int max_alias;
    int n_bonds_pad; // max(n_bonds, 1)
};

AliasTable build_qaqmc_alias_tables(int M_total, int N, int n_bonds,
                                     double Omega,
                                     const double* delta_sched,
                                     const double* bond_vij,
                                     const int* bond_si, const int* bond_sj,
                                     const int* coord_number,
                                     double epsilon,
                                     int p_start = 0, int p_end = -1);

// ─── QAQMCEngine ─────────────────────────────────────────────────────────────

class QAQMCEngine {
public:
    QAQMCEngine(int N, double Omega, double delta_min, double delta_max,
                double Rb, int M, double epsilon, uint64_t seed,
                const double* pos, int pos_dim,
                int neighbor_cutoff = -1, int delta_groups = 600);

    void mc_step();

    // Accessors
    int get_N() const { return N_; }
    int get_M() const { return M_; }
    int get_M_total() const { return M_total_; }
    const std::vector<int32_t>& get_op_types() const { return op_types_; }
    const std::vector<int32_t>& get_op_sites() const { return op_sites_; }
    const std::vector<int>& get_bond_sites_flat() const { return vij_.bond_sites_flat; }
    const std::vector<double>& get_delta_schedule() const { return delta_sched_; }

    // ── On-the-fly observable support ─────────────────────────────────────
    // Set loop/string site index arrays for Z(l) and C_m(l) measurement.
    // Each inner vector is a list of site indices forming one translated copy.
    void set_observable_sites(const std::vector<std::vector<int>>& loop_sets,
                              const std::vector<std::vector<int>>& string_sets);
    void set_bulk_sites(const std::vector<int>& bulk_sites);

    // Number of registered loop / string translation copies.
    int get_n_loops()   const { return (int)loop_site_sets_.size(); }
    int get_n_strings() const { return (int)string_site_sets_.size(); }

    // Number of distinct size groups (copies are grouped by their set length,
    // i.e. number of sites in the path; ordering = first occurrence in *_site_sets_).
    int get_n_loop_size_groups()   const { return (int)loop_group_n_copies_.size(); }
    int get_n_string_size_groups() const { return (int)string_group_n_copies_.size(); }

    // Measure diagonal observables using the captured state at the midpoint (p=M).
    // Z_l_by_size[g]   = mean over copies-in-group-g of Π(1-2nᵢ)
    // C_m_l_by_size[g] = same for strings
    struct MidpointObservables {
        double density;
        std::vector<double> Z_l_by_size;    // [n_loop_size_groups]
        std::vector<double> C_m_l_by_size;  // [n_string_size_groups]
    };
    MidpointObservables measure_at_midpoint() const;

    // Asymmetric profile: measure density/Z_l/C_m_l at every profile_step slices.
    // Z_l_by_size[g][pt]   = mean over copies-in-group-g at profile point pt
    // C_m_l_by_size[g][pt] = same for strings
    struct ProfileObservables {
        std::vector<double> density;                       // [n_points]
        std::vector<std::vector<double>> Z_l_by_size;      // [n_loop_size_groups][n_points]
        std::vector<std::vector<double>> C_m_l_by_size;    // [n_string_size_groups][n_points]
        // Dimer structure factor at the same profile points (only filled if
        // dimer_q_points_ has been set via set_dimer_sf_q_points; otherwise
        // these vectors are empty).
        //   s_q_real[q][pt] = Re Σ_{i ∈ bulk} n_i e^{i q·r_i}
        //   s_q_imag[q][pt] = Im of same
        //   s_q_abs1[q][pt] = |Σ_{i ∈ bulk} n_i e^{i q·r_i}|        (m_q numerator)
        //   s_q_abs2[q][pt] = |Σ_{i ∈ bulk} n_i e^{i q·r_i}|²       (m_q²)
        //   s_q_abs4[q][pt] = |Σ_{i ∈ bulk} n_i e^{i q·r_i}|⁴       (Binder cumulant)
        std::vector<std::vector<double>> s_q_real;         // [n_q][n_points]
        std::vector<std::vector<double>> s_q_imag;         // [n_q][n_points]
        std::vector<std::vector<double>> s_q_abs1;         // [n_q][n_points]
        std::vector<std::vector<double>> s_q_abs2;         // [n_q][n_points]
        std::vector<std::vector<double>> s_q_abs3;         // [n_q][n_points]
        std::vector<std::vector<double>> s_q_abs4;         // [n_q][n_points]
        // Raw spin/occupation snapshots at requested profile points (only
        // filled if snapshot_point_indices_ is non-empty).  Each snapshot is
        // the full state vector (all N sites, 0/1) at that ramp slice.
        //   snapshots[k][i] = n_i at profile point snapshot_point_indices_[k]
        std::vector<std::vector<int8_t>> snapshots;        // [n_snapshot_points][N]
        int n_points;
    };
    ProfileObservables measure_profile(int profile_step) const;

    // ── Snapshot support ──────────────────────────────────────────────────
    // Request that measure_profile also dump the full state vector (all N
    // sites) at the given profile-point indices (0-based, into the n_points
    // grid).  Indices are sorted/deduped.  Empty list disables snapshotting.
    void set_snapshot_point_indices(const std::vector<int>& point_indices);
    int  get_n_snapshot_points() const { return (int)snapshot_point_indices_.size(); }
    const std::vector<int>& get_snapshot_point_indices() const { return snapshot_point_indices_; }

    // ── Dimer (density-density) structure factor measurement ──────────────
    // S_d(q) = (1/N_d) Σ_ij e^{iq·(r_i-r_j)} [<n_i n_j> - <n_i><n_j>]
    //       = (1/N_d) [<|s_q|²> - |<s_q>|²]  with  s_q = Σ_i n_i e^{iq·r_i}
    //
    // The engine stores precomputed cos(q·r_i)/sin(q·r_i) tables and a list
    // of slice indices on the forward ramp (p < M) at which to measure s_q.
    // Per-bin averages of Re(s_q), Im(s_q), and |s_q|² are produced; the
    // user does the connected-piece subtraction and N_d normalisation in
    // post-processing.
    void set_dimer_sf_q_points(const std::vector<std::vector<double>>& q_points);
    // Convert target delta values into nearest forward-ramp slice indices
    // (argmin over p ∈ [0, M) of |delta_sched_[p] - target|).
    void set_dimer_sf_measure_deltas(const std::vector<double>& deltas);
    // Optional: directly specify p indices on the forward ramp.
    void set_dimer_sf_measure_p_indices(const std::vector<int>& p_indices);
    int  get_n_q_points()             const { return (int)dimer_q_points_.size(); }
    int  get_n_dimer_measure_points() const { return (int)dimer_p_indices_.size(); }
    const std::vector<int>&     get_dimer_p_indices()  const { return dimer_p_indices_; }
    const std::vector<double>&  get_dimer_deltas_used() const { return dimer_deltas_used_; }

    // Single-sample measurement: forward-propagate state from p=0 through
    // all offdiag flips, and at each requested p compute s_q for every q.
    // Returns flat arrays: s_q_real / s_q_imag are (n_p × n_q) row-major,
    // density is (n_p,).
    struct DimerSFSample {
        std::vector<double> density;          // [n_p]
        std::vector<double> s_q_real;          // [n_p * n_q] row-major
        std::vector<double> s_q_imag;          // [n_p * n_q]
        std::vector<double> s_q_abs2;          // [n_p * n_q]
    };
    DimerSFSample measure_dimer_sf() const;

    // Profiling
    double get_time_diag() const { return time_diag_; }
    double get_time_clus() const { return time_clus_; }
    int get_mc_steps() const { return mc_steps_; }
    void reset_timers() { time_diag_ = 0; time_clus_ = 0; mc_steps_ = 0; }

    // Checkpoint: RNG state serialization
    std::string get_rng_state() const;
    void set_rng_state(const std::string& state_str);

    // Checkpoint: restore operator string from external data
    void set_op_string(const int32_t* types, const int32_t* sites, int len);

    // Compute 4 bond weights with asymmetric delta per endpoint
    // delta_i = delta / z_eff[site_i],  delta_j = delta / z_eff[site_j]
    static inline void compute_bond_W_inline(double delta_i, double delta_j,
                                              double vij, double epsilon,
                                              double W[4], double& W_max) {
        // raw matrix elements: -V_ij * ni*nj + delta_i * ni + delta_j * nj
        double raw0 = 0.0;                     // |00>: both empty
        double raw1 = delta_j;                  // |01>: j excited
        double raw2 = delta_i;                  // |10>: i excited
        double raw3 = -vij + delta_i + delta_j;  // |11>: both excited
        // C_ij: shift to make all W >= 0, plus safety margin
        double m_min = std::min({raw0, raw1, raw2, raw3});
        // Exclude |raw0| (== 0 always) so epsilon*m_abs is non-trivial.
        // This matches the SSE C++ convention and keeps W[|11>] > 0.
        double m_abs = std::min({std::abs(raw1), std::abs(raw2),
                                 std::abs(raw3)});
        double cij = (m_min < 0.0 ? -m_min : 0.0) + epsilon * m_abs;
        W[0] = raw0 + cij;
        W[1] = raw1 + cij;
        W[2] = raw2 + cij;
        W[3] = raw3 + cij;
        W_max = std::max({W[0], W[1], W[2], W[3]});
    }

private:
    int N_, M_, M_total_;
    double Omega_, Rb_, delta_min_, delta_max_;
    double site_W_, site_W_max_;
    double epsilon_;
    int delta_groups_;  // # groups for shared alias tables (must be > 0)

    double time_diag_{0.0};
    double time_clus_{0.0};
    int mc_steps_{0};

    std::mt19937_64 rng_;

    RydbergVij vij_;
    std::vector<double> delta_sched_;
    // Site coordinates stored row-major (N × pos_dim) for downstream use
    // (e.g. dimer structure factor q·r_i phases).
    std::vector<double> pos_flat_;
    int pos_dim_{0};

    // ── Grouped alias tables for O(G) diagonal update ─────────────────────
    struct GroupedAlias {
        int n_groups;
        std::vector<int> slice_to_group;      // [M_total] -> group index
        // Per-group alias table (same layout as AliasTable but indexed by group)
        int max_alias;
        int n_bonds_pad;
        std::vector<double>  bond_W_max_all;  // [n_groups * n_bonds_pad]
        std::vector<int>     n_alias_all;     // [n_groups]
        std::vector<double>  alias_prob_all;  // [n_groups * max_alias]
        std::vector<int64_t> alias_idx_all;   // [n_groups * max_alias]
        std::vector<int>     op_map_kind_all; // [n_groups * max_alias]
        std::vector<int>     op_map_loc_all;  // [n_groups * max_alias]
    };
    GroupedAlias grp_alias_;

    std::vector<int32_t> op_types_;
    std::vector<int32_t> op_sites_;

    // ── On-the-fly observable data ──────────────────────────────────────
    std::vector<int32_t> state_at_M_;          // spin config at symmetry point
    std::vector<std::vector<int>> loop_site_sets_;
    std::vector<std::vector<int>> string_site_sets_;
    std::vector<int> bulk_sites_;              // interior sites for density (empty = all sites)

    // Size-group bookkeeping: copies sharing the same set length (= number of sites
    // in the path, which is bijective with the logical "loop_size" / "string_size")
    // are aggregated. Group order = first-occurrence of each unique set length.
    std::vector<int> loop_group_of_;          // [n_loop_copies] -> group index
    std::vector<int> loop_group_n_copies_;    // [n_loop_size_groups]
    std::vector<int> string_group_of_;        // [n_string_copies] -> group index
    std::vector<int> string_group_n_copies_;  // [n_string_size_groups]

    // ── Dimer structure factor data ──────────────────────────────────────
    std::vector<std::vector<double>> dimer_q_points_;     // [n_q][pos_dim]
    std::vector<double> dimer_phase_cos_;                 // [n_q * N] row-major
    std::vector<double> dimer_phase_sin_;                 // [n_q * N]
    std::vector<int>    dimer_p_indices_;                 // forward-ramp slices, sorted ascending
    std::vector<double> dimer_deltas_used_;               // delta_sched_[p] for each p in indices

    // ── Snapshot data ─────────────────────────────────────────────────────
    std::vector<int> snapshot_point_indices_;             // profile-point indices to snapshot, sorted ascending

    // ── Vertex lists for O(M) cluster update ──────────────────────────────
    std::vector<int32_t> site_op_count_;
    std::vector<int32_t> site_op_head_;
    std::vector<int32_t> site_op_list_;

    std::vector<int32_t> site_bond_count_;
    std::vector<int32_t> site_bond_head_;
    std::vector<int32_t> site_bond_list_;

    std::vector<int32_t> bond_spin_;
    std::vector<int32_t> spin_now_;
    std::vector<int8_t>  seg_flipped_;

    // Internal update functions
    void diagonal_update();
    void cluster_update();
    void build_vertex_lists();
};