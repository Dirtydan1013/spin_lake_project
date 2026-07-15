#pragma once

#include "../qaqmc_cuda_diagonal.cuh"
#include "qaqmc_cuda_diagonal_kernels.cuh"
#include "qaqmc_cuda_offdiagonal_kernels.cuh"

namespace qaqmc_cuda {
using namespace detail;
struct DiagonalEngine::Impl {
    std::shared_ptr<DeviceHamiltonian> model;
    int device_index;
    int n_sites;
    int half_length;
    std::size_t length;
    int words;
    double delta_min;
    double delta_max;
    double epsilon;
    int n_groups;
    int max_alias;
    int n_bonds;
    std::size_t n_tiles;
    std::size_t scan_temp_bytes;

    DeviceBuffer<int32_t> types;
    DeviceBuffer<int32_t> sites;
    DeviceBuffer<int32_t> bond_sites;
    DeviceBuffer<double> bond_vij;
    DeviceBuffer<double> inv_coord;
    DeviceBuffer<double> alias_prob;
    DeviceBuffer<int32_t> alias_index;
    DeviceBuffer<int32_t> alias_loc_kind;
    DeviceBuffer<double> bond_rmax;
    DeviceBuffer<uint8_t> tile_parity;
    DeviceBuffer<uint8_t> tile_prefix;
    DeviceBuffer<uint8_t> scan_temp;
    DeviceBuffer<DeviceDiagonalStats> stats;

    // Optional off-diagonal string seam.  seam_words is indexed by physical
    // site while seam_mask is indexed by the user-provided string-site list.
    std::vector<int32_t> host_string_sites;
    int seam_cut{-1};
    uint64_t host_seam_mask{0};
    DeviceBuffer<int32_t> string_sites;
    DeviceBuffer<uint64_t> seam_words;
    DeviceBuffer<uint64_t> seam_mask;
    DeviceBuffer<DeviceRepairState> repair_state;
    DeviceBuffer<DeviceTopologyStats> topology_stats;
    DeviceBuffer<DeviceHalfLineProposal> half_line_proposal;
    bool events_valid{false};
    DeviceBuffer<int32_t> checkpoint_types, checkpoint_sites;
    DeviceBuffer<uint64_t> checkpoint_seam_words, checkpoint_seam_mask;
    uint64_t host_checkpoint_seam_mask{0};
    bool checkpoint_valid{false};

    // Lazily allocated by build_events: the diagonal-only benchmark should
    // not pay the radix-sort MVP's substantial temporary-memory footprint.
    DeviceBuffer<uint64_t> site_keys_in, site_keys_out;
    DeviceBuffer<uint32_t> site_values_in, site_values_out;
    DeviceBuffer<uint64_t> bond_keys_in, bond_keys_out;
    DeviceBuffer<uint64_t> bond_values_in, bond_values_out;
    DeviceBuffer<int8_t> bond_spin;
    DeviceBuffer<DeviceEventCounts> event_counts;
    DeviceBuffer<uint8_t> event_sort_temp;
    std::size_t event_sort_temp_bytes{0};
    uint64_t last_site_events{0};
    uint64_t last_bond_events{0};
    DeviceBuffer<int32_t> site_heads, site_counts, bond_heads, bond_counts;
    DeviceBuffer<uint8_t> segment_flags;
    DeviceBuffer<DeviceClusterStats> cluster_stats;
    DeviceBuffer<uint64_t> profile_output;

    std::size_t state_bytes() const {
        return types.size() * sizeof(int32_t)
             + sites.size() * sizeof(int32_t)
             + tile_parity.size() + tile_prefix.size() + scan_temp.size()
             + stats.size() * sizeof(DeviceDiagonalStats)
             + string_sites.size() * sizeof(int32_t)
             + seam_words.size() * sizeof(uint64_t)
             + seam_mask.size() * sizeof(uint64_t)
             + repair_state.size() * sizeof(DeviceRepairState)
             + topology_stats.size() * sizeof(DeviceTopologyStats)
             + half_line_proposal.size() * sizeof(DeviceHalfLineProposal)
             + (checkpoint_types.size() + checkpoint_sites.size()) * sizeof(int32_t)
             + (checkpoint_seam_words.size() + checkpoint_seam_mask.size())
                 * sizeof(uint64_t)
             + site_keys_in.size() * sizeof(uint64_t)
             + site_keys_out.size() * sizeof(uint64_t)
             + site_values_in.size() * sizeof(uint32_t)
             + site_values_out.size() * sizeof(uint32_t)
             + bond_keys_in.size() * sizeof(uint64_t)
             + bond_keys_out.size() * sizeof(uint64_t)
             + bond_values_in.size() * sizeof(uint64_t)
             + bond_values_out.size() * sizeof(uint64_t)
             + bond_spin.size() * sizeof(int8_t)
             + event_counts.size() * sizeof(DeviceEventCounts)
             + event_sort_temp.size()
             + site_heads.size() * sizeof(int32_t)
             + site_counts.size() * sizeof(int32_t)
             + bond_heads.size() * sizeof(int32_t)
             + bond_counts.size() * sizeof(int32_t)
             + segment_flags.size()
             + cluster_stats.size() * sizeof(DeviceClusterStats)
             + profile_output.size() * sizeof(uint64_t);
    }

    std::size_t allocated_bytes() const {
        return state_bytes() + (model ? model->allocated_bytes() : 0);
    }
};

}  // namespace qaqmc_cuda
