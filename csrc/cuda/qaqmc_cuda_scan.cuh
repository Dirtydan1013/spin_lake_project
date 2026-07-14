#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace qaqmc_cuda {

struct DeviceInfo {
    int index;
    std::string name;
    std::size_t total_memory;
    int compute_major;
    int compute_minor;
    int multiprocessors;
};

// Return false (rather than throwing) when the CUDA runtime has no usable
// device.  This lets the Python test suite skip GPU tests on login/CPU nodes.
bool is_available();

std::vector<DeviceInfo> device_info();

// Compute the state immediately BEFORE every operator slice.  Only type=-1
// operators change the propagated state; type=1/2 slots are diagonal.
//
// Output is row-major [length, ceil(n_sites/64)] packed uint64 words.  This is
// a white-box/debug primitive for exact CPU/GPU validation.  Production
// diagonal resampling reuses the same tiled scan without materialising the
// full per-slice state array.
void prefix_xor_states(const int32_t* host_types,
                       const int32_t* host_sites,
                       std::size_t length,
                       int n_sites,
                       uint64_t* host_output);

struct DiagonalStats {
    uint64_t updated_slots;
    uint64_t proposal_attempts;
    uint64_t bond_proposals;
    uint64_t bond_accepts;
    uint64_t failed_slots;
    float elapsed_ms;
};

struct EventStats {
    uint64_t site_events;
    uint64_t bond_events;
    float elapsed_ms;
};

struct ClusterStats {
    uint64_t proposed_segments;
    uint64_t accepted_segments;
    float event_ms;
    float sweep_ms;
};

// Device-resident diagonal-update prototype.  Geometry and grouped-alias
// tables are supplied by the caller in SoA form for this milestone; the full
// Rydberg wrapper will own table construction after the transition kernel is
// validated.  Operator strings remain resident across diagonal_update calls.
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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qaqmc_cuda
