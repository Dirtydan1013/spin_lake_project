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
                int neighbor_cutoff = -1, bool precompute = true,
                int chunk_slices = 0);

    void mc_step();

    // Accessors
    int get_N() const { return N_; }
    int get_M() const { return M_; }
    int get_M_total() const { return M_total_; }
    const std::vector<int32_t>& get_op_types() const { return op_types_; }
    const std::vector<int32_t>& get_op_sites() const { return op_sites_; }
    const std::vector<int>& get_bond_sites_flat() const { return vij_.bond_sites_flat; }
    const std::vector<double>& get_delta_schedule() const { return delta_sched_; }

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
        double m_abs = std::min({std::abs(raw0), std::abs(raw1),
                                 std::abs(raw2), std::abs(raw3)});
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
    bool precompute_;
    int chunk_slices_;  // 0 = full precompute, >0 = chunked

    double time_diag_{0.0};
    double time_clus_{0.0};
    int mc_steps_{0};

    std::mt19937_64 rng_;

    RydbergVij vij_;
    AliasTable alias_;
    std::vector<double> delta_sched_;

    std::vector<int32_t> op_types_;
    std::vector<int32_t> op_sites_;

    // Internal update functions
    void diagonal_update();
    void cluster_update();
};