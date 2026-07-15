#pragma once

#include "qaqmc_cuda_api.cuh"

#include <memory>

namespace qaqmc_cuda {
// Device-resident standard QAQMC engine with an optional off-diagonal string
// seam. Geometry and grouped-alias tables are supplied by the caller in SoA
// form; operator strings remain resident across update calls.
class DiagonalEngine {
public:
    DiagonalEngine(int n_sites,
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
                   const int32_t* op_types,
                   const int32_t* op_sites,
                   int device_index = 0);
    ~DiagonalEngine();

    DiagonalEngine(const DiagonalEngine&) = delete;
    DiagonalEngine& operator=(const DiagonalEngine&) = delete;
    DiagonalEngine(DiagonalEngine&&) noexcept;
    DiagonalEngine& operator=(DiagonalEngine&&) noexcept;

    DiagonalStats diagonal_update(uint64_t seed, uint64_t sweep_id);
    EventStats build_events();
    ClusterStats cluster_update(uint64_t seed, uint64_t sweep_id);

    // Configure an off-diagonal string seam.  The cut lies immediately
    // before operator m_star.  string_sites contains physical site ids and is
    // limited to 64 entries so its active topology fits in one uint64 mask.
    void set_string_sites(const int32_t* string_sites, int count, int m_star);
    // Install a topology mask and repair fixed-boundary worldline closure on
    // the device.  Unlike the raw setter in the CPU white-box API this method
    // is safe for trajectory resets.
    void set_seam_mask_consistent(uint64_t mask);
    HalfLineProposal half_line_proposal(int local_index, bool direction_right);
    TopologyStats topology_sweep(double lambda, uint64_t seed, uint64_t sweep_id);
    // Lazily allocated device-to-device checkpoint used by string-work
    // trajectory resets.  The seam state is included so restore never needs
    // an O(|C|*M) closure-repair pass.
    void save_checkpoint();
    void restore_checkpoint();
    bool has_checkpoint() const;
    uint64_t seam_mask() const;
    int string_site_count() const;
    int seam_cut() const;
    // Return packed propagated states after slices profile_step-1,
    // 2*profile_step-1, ... .  Only O(n_points * ceil(N/64)) data crosses
    // PCIe, so diagonal observables do not require downloading 2M operators.
    void get_profile_states(int profile_step, uint64_t* host_output) const;
    void get_operator_string(int32_t* host_types, int32_t* host_sites) const;
    void set_operator_string(const int32_t* host_types, const int32_t* host_sites);
    void get_site_events(uint64_t* host_keys, uint32_t* host_values) const;
    void get_bond_events(uint64_t* host_keys, uint64_t* host_values) const;
    void get_bond_spin(int8_t* host_bond_spin) const;

    int n_sites() const;
    int half_length() const;
    std::size_t length() const;
    int packed_words() const;
    std::size_t device_bytes() const;

private:
    friend class BatchedDiagonalEngine;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One CUDA process managing B statistically independent standard or
// off-diagonal chains. Immutable Hamiltonian/alias tables are shared once;
// operator strings, workspaces, topology state and checkpoints are per-chain.
class BatchedDiagonalEngine {
public:
    BatchedDiagonalEngine(
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
        const int32_t* op_types_bxl,
        const int32_t* op_sites_bxl,
        int device_index = 0);
    ~BatchedDiagonalEngine();

    BatchedDiagonalEngine(const BatchedDiagonalEngine&) = delete;
    BatchedDiagonalEngine& operator=(const BatchedDiagonalEngine&) = delete;
    BatchedDiagonalEngine(BatchedDiagonalEngine&&) noexcept;
    BatchedDiagonalEngine& operator=(BatchedDiagonalEngine&&) noexcept;

    std::vector<DiagonalStats> diagonal_update(
        const uint64_t* seeds, const uint64_t* sweep_ids);
    std::vector<ClusterStats> cluster_update(
        const uint64_t* seeds, const uint64_t* sweep_ids);
    void set_string_sites(const int32_t* sites, int count, int m_star);
    void set_seam_masks_consistent(const uint64_t* masks);
    std::vector<TopologyStats> topology_sweep(
        double lambda, const uint64_t* seeds, const uint64_t* sweep_ids);
    void save_checkpoint();
    void restore_checkpoint();
    bool has_checkpoint() const;
    void get_operator_strings(int32_t* types_bxl, int32_t* sites_bxl) const;
    void set_operator_strings(const int32_t* types_bxl, const int32_t* sites_bxl);
    void get_profile_states(int profile_step, uint64_t* output_bxpxw) const;
    void get_seam_masks(uint64_t* masks) const;

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
