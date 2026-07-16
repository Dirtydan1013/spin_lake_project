#pragma once

#include "api.cuh"

#include <memory>

namespace qaqmc_cuda {
// Two-replica QAQMC transition backend.  The operator strings and channel
// event streams remain device-resident; after the cut, physical site s maps
// actual replica r to channel r XOR A_mask[s].
class RenyiEngine {
public:
    RenyiEngine(int n_sites,
                int half_length,
                double delta_min,
                double delta_max,
                double epsilon,
                int n_groups,
                int max_alias,
                int n_bonds,
                const int32_t* bond_sites,
                const double* bond_vij,
                const double* inv_coord,
                const double* alias_prob,
                const int32_t* alias_index,
                const int32_t* alias_loc_kind,
                const double* bond_rmax,
                const int32_t* op_types_2xl,
                const int32_t* op_sites_2xl,
                int device_index = 0);
    ~RenyiEngine();

    RenyiEngine(const RenyiEngine&) = delete;
    RenyiEngine& operator=(const RenyiEngine&) = delete;
    RenyiEngine(RenyiEngine&&) noexcept;
    RenyiEngine& operator=(RenyiEngine&&) noexcept;

    void set_cut(int cut);
    void set_mask(const uint8_t* mask, int count);
    void get_mask(uint8_t* mask, int count) const;
    DiagonalStats diagonal_update(uint64_t seed, uint64_t sweep_id);
    EventStats build_events();
    ClusterStats cluster_update(uint64_t seed, uint64_t sweep_id);
    TopologyStats topology_sweep(const int32_t* topology_sites,
                                 int count,
                                 double lambda,
                                 uint64_t seed,
                                 uint64_t sweep_id);
    // Read-only white-box primitive used to compare the interacting topology
    // weight ratio against the trusted CPU implementation.  It leaves the
    // mask and both operator strings unchanged.
    TopologyRatio log_weight_ratio_for_toggle(int site);
    // Operator-string-only D2D checkpoint.  The work driver restores A_start
    // with set_mask() before restore_checkpoint(); the dynamic mask is
    // intentionally not part of the rolling equilibrium checkpoint.
    void save_checkpoint();
    void restore_checkpoint();
    bool has_checkpoint() const;
    void get_site_events(uint64_t* keys, uint32_t* values) const;
    void get_bond_events(uint64_t* keys, uint32_t* values) const;
    void get_bond_spin(int8_t* values_2xl) const;
    void get_operator_strings(int32_t* types_2xl, int32_t* sites_2xl) const;
    void set_operator_strings(const int32_t* types_2xl, const int32_t* sites_2xl);

    int n_sites() const;
    int half_length() const;
    std::size_t length() const;
    int cut() const;
    int packed_words() const;
    std::size_t device_bytes() const;

private:
    friend class BatchedRenyiEngine;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Batched two-replica work engine. Each chain owns two operator strings and
// its own mask/topology/checkpoint state while the immutable model is shared.
class BatchedRenyiEngine {
public:
    BatchedRenyiEngine(
        int batch_size,
        int n_sites,
        int half_length,
        double delta_min,
        double delta_max,
        double epsilon,
        int n_groups,
        int max_alias,
        int n_bonds,
        const int32_t* bond_sites,
        const double* bond_vij,
        const double* inv_coord,
        const double* alias_prob,
        const int32_t* alias_index,
        const int32_t* alias_loc_kind,
        const double* bond_rmax,
        const int32_t* op_types_bx2xl,
        const int32_t* op_sites_bx2xl,
        int device_index = 0);
    ~BatchedRenyiEngine();

    BatchedRenyiEngine(const BatchedRenyiEngine&) = delete;
    BatchedRenyiEngine& operator=(const BatchedRenyiEngine&) = delete;
    BatchedRenyiEngine(BatchedRenyiEngine&&) noexcept;
    BatchedRenyiEngine& operator=(BatchedRenyiEngine&&) noexcept;

    void set_cut(int cut);
    void set_masks(const uint8_t* masks_bxn);
    void get_masks(uint8_t* masks_bxn) const;
    std::vector<DiagonalStats> diagonal_update(
        const uint64_t* seeds, const uint64_t* sweep_ids);
    std::vector<ClusterStats> cluster_update(
        const uint64_t* seeds, const uint64_t* sweep_ids);
    std::vector<TopologyStats> topology_sweep(
        const int32_t* topology_sites, int count, double lambda,
        const uint64_t* seeds, const uint64_t* sweep_ids);
    void save_checkpoint();
    void restore_checkpoint();
    bool has_checkpoint() const;
    void get_operator_strings(int32_t* types_bx2xl, int32_t* sites_bx2xl) const;
    void set_operator_strings(
        const int32_t* types_bx2xl, const int32_t* sites_bx2xl);

    int batch_size() const;
    int n_sites() const;
    std::size_t length() const;
    std::size_t shared_model_bytes() const;
    std::size_t device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qaqmc_cuda
