#pragma once

#include "qaqmc_core.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class QAQMCRenyiEngine {
public:
    enum class Mode {
        PairToggle = 0,
        Expanded = 1,
    };

    struct ReplicaState {
        std::vector<int32_t> op_types;
        std::vector<int32_t> op_sites;
        std::vector<int32_t> state_at_M;
    };

    struct Ensemble {
        std::vector<uint8_t> A_mask;
        int size{0};
    };

    struct SiteEvent {
        int32_t p{0};
        int8_t replica{0};
    };

    struct BondEvent {
        int32_t p{0};
        int8_t replica{0};
        int32_t bond{0};
        int8_t endpoint{0};
    };

    struct OffdiagPaths {
        std::vector<int32_t> count;
        std::vector<int32_t> head;
        std::vector<int32_t> list;
    };

    QAQMCRenyiEngine(int N, double Omega, double delta_min, double delta_max,
                     double Rb, int M, double epsilon, uint64_t seed,
                     const double* pos, int pos_dim, int neighbor_cutoff = -1,
                     int delta_groups = 0);

    void mc_step();
    void run_steps(int n_steps);

    int get_N() const { return N_; }
    int get_M() const { return M_; }
    int get_M_total() const { return M_total_; }

    const std::vector<int32_t>& get_op_types(int replica) const { return replicas_[replica].op_types; }
    const std::vector<int32_t>& get_op_sites(int replica) const { return replicas_[replica].op_sites; }
    const std::vector<int32_t>& get_state_at_M(int replica) const { return replicas_[replica].state_at_M; }
    const std::vector<int>& get_bond_sites_flat() const { return vij_.bond_sites_flat; }
    const std::vector<double>& get_delta_schedule() const { return delta_sched_; }
    const std::vector<uint8_t>& get_A_mask() const { return A_mask_; }
    const std::vector<uint8_t>& get_topology_mask(int topology) const { return A_masks_[topology]; }
    const std::array<int64_t, 2>& get_visit_counts() const { return visit_count_; }
    const std::vector<int64_t>& get_visit_counts_ext() const { return visit_count_ext_; }
    const std::vector<int64_t>& get_transition_counts() const { return transition_count_; }
    const std::vector<double>& get_collection_counts() const { return collection_count_; }
    int get_current_topology() const { return cur_topology_; }
    int get_current_ensemble() const { return cur_ens_; }
    int get_ensemble_count() const { return static_cast<int>(ensembles_.size()); }
    int get_diff_site() const { return diff_site_; }
    int get_mode() const { return static_cast<int>(mode_); }
    int get_delta_groups() const { return delta_groups_; }

    void set_A_mask(const uint8_t* mask, int len);
    void set_topology_pair(const uint8_t* A_k, const uint8_t* A_kp1, int len, int diff_site);
    void reset_visit_counts();
    void set_ensemble_ladder(const std::vector<std::vector<uint8_t>>& masks,
                             const std::vector<std::vector<int>>& neighbors,
                             int initial_ensemble);
    void set_log_g(const std::vector<double>& log_g);
    void reset_visit_counts_ext();
    void reset_transition_counts();
    void reset_collection_counts();
    void topology_toggle();
    void ensemble_switch();
    double log_weight_ratio_for_site(int site, int from_topology, int to_topology) const;
    void set_indicator_site(int site);
    int get_indicator_site() const { return indicator_site_; }
    void reset_indicator();
    double get_indicator_avg() const;
    int current_indicator() const;

    void set_replica_op_string(int replica, const int32_t* types, const int32_t* sites, int len);
    void recompute_midpoint_states();
    std::array<std::vector<int32_t>, 4> get_site_paths(int site) const;

private:
    int N_, M_, M_total_;
    double Omega_, Rb_, delta_min_, delta_max_, epsilon_;
    int delta_groups_{0};

    RydbergVij vij_;
    AliasTable alias_;
    std::vector<double> delta_sched_;

    struct GroupedAlias {
        int n_groups{0};
        std::vector<int> slice_to_group;
        int max_alias{0};
        int n_bonds_pad{1};
        std::vector<double> bond_W_max_all;
        std::vector<int> n_alias_all;
        std::vector<double> alias_prob_all;
        std::vector<int64_t> alias_idx_all;
        std::vector<int> op_map_kind_all;
        std::vector<int> op_map_loc_all;
    };
    GroupedAlias grp_alias_;

    std::array<ReplicaState, 2> replicas_;
    std::array<std::mt19937_64, 2> rngs_;

    Mode mode_{Mode::PairToggle};
    std::vector<uint8_t> A_mask_;
    std::array<std::vector<uint8_t>, 2> A_masks_;
    int cur_topology_{0};
    int diff_site_{-1};
    std::array<int64_t, 2> visit_count_{0, 0};
    std::vector<Ensemble> ensembles_;
    std::vector<std::vector<int>> ens_neighbors_;
    int cur_ens_{0};
    std::vector<double> log_g_;
    std::vector<int64_t> visit_count_ext_;
    std::vector<int64_t> transition_count_;
    std::vector<double> collection_count_;
    int indicator_site_{-1};
    int64_t indicator_sum_{0};
    int64_t indicator_count_{0};

    int n_bonds_pad_{1};
    int max_alias_{0};

    // Greedy bond-graph coloring used by the parallel cluster_update.  Sites in
    // the same color share no bond, so their per-site segment Metropolis can run
    // concurrently without racing on per-bond spin-cache writes.  Size = n_colors.
    std::vector<std::vector<int>> color_groups_;
    // Per-site RNG (one stream per site) so the OpenMP cluster_update doesn't
    // need to synchronize the original two-channel RNGs.  Size = N.
    std::vector<std::mt19937_64> site_rngs_;

    // Channel-space vertex lists for the Renyi cluster update.  The flat
    // channel/site index is channel * N_ + site; event lists are sorted by p.
    std::vector<int32_t> ch_site_op_count_;
    std::vector<int32_t> ch_site_op_head_;
    std::vector<SiteEvent> ch_site_op_list_;
    std::vector<int32_t> ch_site_bond_count_;
    std::vector<int32_t> ch_site_bond_head_;
    std::vector<BondEvent> ch_site_bond_list_;
    std::vector<int32_t> bond_spin_by_replica_;  // [2 * M_total_]
    OffdiagPaths paths_scratch_from_;
    OffdiagPaths paths_scratch_to_;
    std::vector<OffdiagPaths> paths_scratch_targets_;

    static inline void compute_bond_W_inline(double delta_i, double delta_j,
                                             double vij, double epsilon,
                                             double W[4], double& W_max) {
        QAQMCEngine::compute_bond_W_inline(delta_i, delta_j, vij, epsilon, W, W_max);
    }

    int replica_for_with_mask(int channel, int site, int p, const std::vector<uint8_t>& mask) const;
    int channel_for_actual_with_mask(int replica, int site, int p, const std::vector<uint8_t>& mask) const;
    int replica_for(int channel, int site, int p) const;
    int channel_for_actual(int replica, int site, int p) const;

    void recompute_midpoint_states_from_ops();
    void reproject_site_ops_for_mask_with_paths(const std::vector<uint8_t>& mask,
                                                const OffdiagPaths& paths);
    void reproject_site_ops_at_site_with_paths(int diff_site,
                                               const std::vector<uint8_t>& mask,
                                               const OffdiagPaths& paths);
    void accumulate_indicator();
    int diff_site_between_masks(const std::vector<uint8_t>& from_mask,
                                const std::vector<uint8_t>& to_mask) const;
    double log_weight_ratio_between_masks(int site,
                                          const std::vector<uint8_t>& from_mask,
                                          const std::vector<uint8_t>& to_mask) const;
    void build_offdiag_paths(const std::vector<uint8_t>& mask, OffdiagPaths& paths) const;
    int occupancy_from_paths(const OffdiagPaths& paths, int channel, int site, int p) const;
    double log_weight_for_site_with_paths(int site, const std::vector<uint8_t>& mask,
                                          const OffdiagPaths& paths) const;
    double actual_bond_weight(int p, int b, int w_idx) const;
    void build_grouped_alias_tables();
    void compute_site_coloring();
    void build_channel_vertex_lists();
    void build_bond_spins_from_ops();

    void diagonal_update();
    void cluster_update();
};
