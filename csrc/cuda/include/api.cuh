#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
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

struct TopologyStats {
    uint64_t attempts;
    uint64_t accepts;
    uint64_t invalid;
    uint64_t active_count;
    float elapsed_ms;
};

struct TopologyRatio {
    double log_ratio;
    bool current_valid;
    bool proposed_valid;
};

struct HalfLineProposal {
    bool valid;
    int terminal_p;
    double log_physical_ratio;
};

}  // namespace qaqmc_cuda
