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
        Work = 2,        // dynamic single-bit topology toggles driven by an outer
                         // nonequilibrium-work engine; mc_step() skips topology /
                         // ensemble update so the outer engine owns mask transitions
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

    // Bit-packed channel-space vertex events.  After A.1 the cluster_update
    // hot loop only needs (p, replica, endpoint) — bond/site can be derived
    // from replicas_[replica].op_sites[p] when ever needed (they aren't in the
    // current segment Metropolis), and the original 16-byte BondEvent/SiteEvent
    // structs blew the L3 budget at production sizes (M_total≈4.5M produces
    // ~9M bond events ≈ 144 MB).  Packing into uint32_t mirrors the original
    // single-replica QAQMC vertex-list layout (4 B/event) and quarters the
    // memory bandwidth pressure on cluster_build / cluster_sweep.
    //
    // Layout:
    //   BondEvent : [ p : 30 bits ][ replica : 1 ][ endpoint : 1 ]
    //   SiteEvent : [ p : 31 bits ][ replica : 1 ]
    // Sorted ascending: lists are filled in p-ascending order, so monotonicity
    // is preserved by the high-bit p prefix and std::upper_bound by p just
    // needs to compare on (event >> shift).
    using BondEvent = uint32_t;
    using SiteEvent = uint32_t;

    static inline BondEvent pack_bond_event(int32_t p, int8_t replica, int8_t endpoint) {
        return (static_cast<uint32_t>(p) << 2)
             | (static_cast<uint32_t>(replica & 1) << 1)
             | static_cast<uint32_t>(endpoint & 1);
    }
    static inline int bond_event_p(BondEvent e)        { return static_cast<int>(e >> 2); }
    static inline int bond_event_replica(BondEvent e)  { return static_cast<int>((e >> 1) & 1u); }
    static inline int bond_event_endpoint(BondEvent e) { return static_cast<int>(e & 1u); }

    static inline SiteEvent pack_site_event(int32_t p, int8_t replica) {
        return (static_cast<uint32_t>(p) << 1) | static_cast<uint32_t>(replica & 1);
    }
    static inline int site_event_p(SiteEvent e)       { return static_cast<int>(e >> 1); }
    static inline int site_event_replica(SiteEvent e) { return static_cast<int>(e & 1u); }

    struct OffdiagPaths {
        std::vector<int32_t> count;
        std::vector<int32_t> head;
        std::vector<int32_t> list;
    };

    QAQMCRenyiEngine(int N, double Omega, double delta_min, double delta_max,
                     double Rb, int M, double epsilon, uint64_t seed,
                     const double* pos, int pos_dim, int neighbor_cutoff = -1,
                     int delta_groups = 0,
                     const double* box = nullptr, int n_box = 0);

    void mc_step();
    void run_steps(int n_steps);

    int get_N() const { return N_; }
    int get_M() const { return M_; }
    int get_M_total() const { return M_total_; }
    int get_cut() const { return cut_; }

    const std::vector<int32_t>& get_op_types(int replica) const { return replicas_[replica].op_types; }
    const std::vector<int32_t>& get_op_sites(int replica) const { return replicas_[replica].op_sites; }
    const std::vector<int32_t>& get_state_at_M(int replica) const { return replicas_[replica].state_at_M; }
    const std::vector<int>& get_bond_sites_flat() const { return vij_.bond_sites_flat; }
    const std::vector<double>& get_delta_schedule() const { return delta_sched_; }

    // delta at slice p, bit-identical to delta_sched_[p] (same expression as
    // the ctor) — lets hot loops avoid touching the schedule array.
    inline double delta_at(int p) const {
        return (p < M_)
            ? delta_min_ + (delta_max_ - delta_min_) * (static_cast<double>(p) / M_)
            : delta_max_ - (delta_max_ - delta_min_) * (static_cast<double>(p - M_) / M_);
    }

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
    double get_time_diag() const { return time_diag_; }
    double get_time_clus_build() const { return time_clus_build_; }
    double get_time_clus_sweep() const { return time_clus_sweep_; }
    double get_time_topology() const { return time_topology_; }
    double get_time_ensemble() const { return time_ensemble_; }
    int64_t get_mc_steps() const { return mc_steps_; }
    int64_t get_diag_update_slots() const { return diag_update_slots_; }
    int64_t get_diag_proposal_attempts() const { return diag_proposal_attempts_; }
    int64_t get_diag_site_proposals() const { return diag_site_proposals_; }
    int64_t get_diag_bond_proposals() const { return diag_bond_proposals_; }
    int64_t get_diag_bond_accepts() const { return diag_bond_accepts_; }
    std::array<int64_t, 3> get_operator_counts() const;
    void reset_timers() {
        time_diag_ = 0.0;
        time_clus_build_ = 0.0;
        time_clus_sweep_ = 0.0;
        time_topology_ = 0.0;
        time_ensemble_ = 0.0;
        mc_steps_ = 0;
        diag_update_slots_ = 0;
        diag_proposal_attempts_ = 0;
        diag_site_proposals_ = 0;
        diag_bond_proposals_ = 0;
        diag_bond_accepts_ = 0;
    }

    void set_A_mask(const uint8_t* mask, int len);
    // Move the entanglement cut (swap boundary) to slice m_star in
    // [0, M_total].  Default is the schedule midpoint M.  The A-mask swap and
    // the "midpoint" state capture then act at p >= m_star / p < m_star.
    // MUST be called while the operator strings are valid under BOTH the old
    // and new mapping — in practice: right after construction, or through
    // QAQMCRenyiWorkEngine::set_cut which vacuum-resets the ops.
    void set_cut(int m_star);
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

    // ── Single-bit-toggle primitives for Mode::Work ───────────────────────────
    // These let an outer engine (e.g. QAQMCRenyiWorkEngine) drive dynamic
    // topology proposals without re-running set_topology_pair's full rebuild.
    void set_mode(Mode m) { mode_ = m; }
    // Work-mode counterpart of set_A_mask: switches A_mask_ to the given mask,
    // **reprojects all site operators** under the new mask (critical, since
    // diagonal_update preserves offdiag op_types and would otherwise leave
    // stale -1 slots from a previous boundary), recomputes midpoint states,
    // and leaves mode_ == Mode::Work.
    void set_A_mask_for_work(const uint8_t* mask, int len);
    // log[ Z(A_mask_ ^ {site}) / Z(A_mask_) ], based on current operator strings.
    double log_weight_ratio_for_toggle(int site) const;
    // Apply A_mask_[site] ^= 1 and reproject affected site ops at p >= M.
    // Caller is responsible for the Metropolis accept/reject before calling this.
    void apply_single_bit_toggle(int site);

    // ── Op-string checkpoint (for the work engine's checkpoint chain) ─────
    // Save / restore the current operator string for BOTH replicas.  The mask
    // is NOT saved; the caller is responsible for arranging A_mask_ to match
    // the mask the checkpoint was saved under (typically A_start) before
    // restoring.  After restore, state_at_M is recomputed.
    void save_op_string_checkpoint();
    void restore_op_string_checkpoint();
    bool has_op_string_checkpoint() const { return op_ckpt_valid_; }
    void invalidate_op_string_checkpoint() { op_ckpt_valid_ = false; }
    // Read access to the saved checkpoint (warm-start export).  Only valid
    // when has_op_string_checkpoint() is true.
    const std::vector<int32_t>& get_checkpoint_op_types(int replica) const {
        return op_ckpt_types_[replica];
    }
    const std::vector<int32_t>& get_checkpoint_op_sites(int replica) const {
        return op_ckpt_sites_[replica];
    }
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
    int cut_;  // swap-boundary slice; default M_ (schedule midpoint)
    double Omega_, Rb_, delta_min_, delta_max_, epsilon_;
    int delta_groups_{0};

    RydbergVij vij_;
    AliasTable alias_;
    std::vector<double> delta_sched_;

    struct GroupedAlias {
        int n_groups{0};
        // (slice -> group is computed incrementally in diagonal_update; no
        //  per-slice map is stored.)
        int max_alias{0};
        int n_bonds_pad{1};
        std::vector<double> bond_W_max_all;
        std::vector<double> bond_W_rmax_all;  // 1/W_max (0 if W_max <= 0)
        std::vector<int> n_alias_all;
        std::vector<AliasEntry> entries;      // [n_groups * max_alias] (AoS)
    };
    GroupedAlias grp_alias_;

    std::array<ReplicaState, 2> replicas_;
    std::array<std::mt19937_64, 2> rngs_;

    // Op-string checkpoint (work-engine checkpoint chain).  Not saved by ctor.
    bool op_ckpt_valid_{false};
    std::array<std::vector<int32_t>, 2> op_ckpt_types_;
    std::array<std::vector<int32_t>, 2> op_ckpt_sites_;

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
    double time_diag_{0.0};
    double time_clus_build_{0.0};
    double time_clus_sweep_{0.0};
    double time_topology_{0.0};
    double time_ensemble_{0.0};
    int64_t mc_steps_{0};
    int64_t diag_update_slots_{0};
    int64_t diag_proposal_attempts_{0};
    int64_t diag_site_proposals_{0};
    int64_t diag_bond_proposals_{0};
    int64_t diag_bond_accepts_{0};

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
    // Values 0..3 — int8 quarters the random-access footprint in the sweep.
    std::vector<int8_t> bond_spin_by_replica_;   // [2 * M_total_]
    // Per-bond-op raw-W cache, layout [(replica * M_total + p) * 4 + w_idx].
    // Populated by build_bond_spins_from_ops() at the start of each
    // cluster_update so the inner segment-Metropolis loop avoids
    // compute_bond_W_inline() in the hot path.  Raw weights (not logs):
    // the segment Metropolis accumulates the plain ratio product, so no
    // std::log is needed anywhere in the cluster path.
    std::vector<double> W_by_op_;
    OffdiagPaths paths_scratch_from_;
    OffdiagPaths paths_scratch_to_;
    std::vector<OffdiagPaths> paths_scratch_targets_;

    static inline void compute_bond_W_inline(double delta_i, double delta_j,
                                             double vij, double epsilon,
                                             double W[4], double& W_max) {
        QAQMCEngine::compute_bond_W_inline(delta_i, delta_j, vij, epsilon, W, W_max);
    }

    inline int replica_for_with_mask(int channel, int site, int p,
                                     const std::vector<uint8_t>& mask) const {
        if (p < cut_) return channel;
        return mask[site] ? (1 - channel) : channel;
    }

    inline int channel_for_actual_with_mask(int replica, int site, int p,
                                            const std::vector<uint8_t>& mask) const {
        if (p < cut_) return replica;
        return mask[site] ? (1 - replica) : replica;
    }

    inline int replica_for(int channel, int site, int p) const {
        if (p < cut_) return channel;
        return A_mask_[site] ? (1 - channel) : channel;
    }

    inline int channel_for_actual(int replica, int site, int p) const {
        if (p < cut_) return replica;
        return A_mask_[site] ? (1 - replica) : replica;
    }

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
