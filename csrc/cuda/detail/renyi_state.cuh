#pragma once

#include "../include/renyi.cuh"
#include "renyi_topology_kernels.cuh"
#include "renyi_transition_kernels.cuh"

namespace qaqmc_cuda {
using namespace detail;
struct RenyiEngine::Impl {
    std::shared_ptr<DeviceHamiltonian> model;
    int device_index{0};
    int n_sites{0};
    int half_length{0};
    std::size_t length{0};
    int words{0};
    int cut{0};
    double delta_min{0.0};
    double delta_max{0.0};
    double epsilon{0.0};
    int n_groups{0};
    int max_alias{0};
    int n_bonds{0};
    std::size_t n_tiles{0};
    std::size_t scan_temp_bytes{0};

    DeviceBuffer<int32_t> types, sites;
    DeviceBuffer<int32_t> bond_sites;
    DeviceBuffer<double> bond_vij, inv_coord;
    DeviceBuffer<double> alias_prob;
    DeviceBuffer<int32_t> alias_index, alias_loc_kind;
    DeviceBuffer<double> bond_rmax;
    DeviceBuffer<uint64_t> mask_words;
    DeviceBuffer<uint8_t> tile_parity, tile_prefix, scan_temp;
    DeviceBuffer<DeviceDiagonalStats> diagonal_stats;

    DeviceBuffer<uint64_t> site_keys_in, site_keys_out;
    DeviceBuffer<uint32_t> site_values_in, site_values_out;
    DeviceBuffer<uint64_t> bond_keys_in, bond_keys_out;
    DeviceBuffer<uint32_t> bond_values_in, bond_values_out;
    DeviceBuffer<int8_t> bond_spin;
    DeviceBuffer<DeviceEventCounts> event_counts;
    DeviceBuffer<uint8_t> event_sort_temp;
    std::size_t event_sort_temp_bytes{0};
    uint64_t last_site_events{0};
    uint64_t last_bond_events{0};
    bool events_valid{false};
    DeviceBuffer<int32_t> site_heads, site_counts, bond_heads, bond_counts;
    DeviceBuffer<uint8_t> segment_flags;
    DeviceBuffer<DeviceClusterStats> cluster_stats;
    DeviceBuffer<uint64_t> actual_boundaries;
    DeviceBuffer<int32_t> topology_sites;
    DeviceBuffer<DeviceTopologyStats> topology_stats;
    DeviceBuffer<DeviceTopologyRatio> topology_ratio;
    bool actual_boundaries_valid{false};
    DeviceBuffer<int32_t> checkpoint_types, checkpoint_sites;
    bool checkpoint_valid{false};

    std::size_t state_bytes() const {
        return (types.size() + sites.size()) * sizeof(int32_t)
             + mask_words.size() * sizeof(uint64_t)
             + tile_parity.size() + tile_prefix.size() + scan_temp.size()
             + diagonal_stats.size() * sizeof(DeviceDiagonalStats)
             + (site_keys_in.size() + site_keys_out.size()
                + bond_keys_in.size() + bond_keys_out.size()) * sizeof(uint64_t)
             + (site_values_in.size() + site_values_out.size()
                + bond_values_in.size() + bond_values_out.size()) * sizeof(uint32_t)
             + bond_spin.size() * sizeof(int8_t)
             + event_counts.size() * sizeof(DeviceEventCounts)
             + event_sort_temp.size()
             + (site_heads.size() + site_counts.size()
                + bond_heads.size() + bond_counts.size()) * sizeof(int32_t)
             + segment_flags.size()
             + cluster_stats.size() * sizeof(DeviceClusterStats)
             + actual_boundaries.size() * sizeof(uint64_t)
             + topology_sites.size() * sizeof(int32_t)
             + topology_stats.size() * sizeof(DeviceTopologyStats)
             + topology_ratio.size() * sizeof(DeviceTopologyRatio)
             + (checkpoint_types.size() + checkpoint_sites.size()) * sizeof(int32_t);
    }

    std::size_t allocated_bytes() const {
        return state_bytes() + (model ? model->allocated_bytes() : 0);
    }
};

}  // namespace qaqmc_cuda
