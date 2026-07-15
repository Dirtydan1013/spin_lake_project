#include "qaqmc_cuda_renyi.cuh"
#include "detail/qaqmc_cuda_renyi_state.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qaqmc_cuda {
using namespace detail;
RenyiEngine::RenyiEngine(int n_sites,
                         int half_length,
                         double delta_min,
                         double delta_max,
                         double epsilon,
                         int n_groups,
                         int max_alias,
                         int n_bonds,
                         const int32_t* host_bond_sites,
                         const double* host_bond_vij,
                         const double* host_inv_coord,
                         const double* host_alias_prob,
                         const int32_t* host_alias_index,
                         const int32_t* host_alias_loc_kind,
                         const double* host_bond_rmax,
                         const int32_t* host_types,
                         const int32_t* host_sites,
                         int device_index)
    : impl_(std::make_unique<Impl>()) {
    if (n_sites <= 0 || n_sites > 64 * kMaxWords)
        throw std::invalid_argument("n_sites must be in [1, 384]");
    if (half_length <= 0)
        throw std::invalid_argument("half_length must be positive");
    if (n_groups <= 0 || max_alias != n_sites + n_bonds)
        throw std::invalid_argument("invalid grouped alias dimensions");
    validate_operator_string(
        host_types, host_sites,
        static_cast<std::size_t>(4) * static_cast<std::size_t>(half_length),
        n_sites, n_bonds, "Renyi operator string");
    check_cuda(cudaSetDevice(device_index), "cudaSetDevice for Renyi constructor");
    Impl& x = *impl_;
    x.device_index = device_index;
    x.n_sites = n_sites;
    x.half_length = half_length;
    x.length = static_cast<std::size_t>(2) * half_length;
    x.words = (n_sites + 63) / 64;
    x.cut = half_length;
    x.delta_min = delta_min;
    x.delta_max = delta_max;
    x.epsilon = epsilon;
    x.n_groups = n_groups;
    x.max_alias = max_alias;
    x.n_bonds = n_bonds;
    x.n_tiles = (x.length + kBlockSize - 1) / kBlockSize;
    x.model = make_device_hamiltonian(
        n_sites, delta_min, delta_max, epsilon, n_groups, max_alias, n_bonds,
        host_bond_sites, host_bond_vij, host_inv_coord, host_alias_prob,
        host_alias_index, host_alias_loc_kind, host_bond_rmax, device_index);
    switch (x.words) {
        case 1: x.scan_temp_bytes = dual_tile_scan_temp_bytes<1>(x.n_tiles); break;
        case 2: x.scan_temp_bytes = dual_tile_scan_temp_bytes<2>(x.n_tiles); break;
        case 3: x.scan_temp_bytes = dual_tile_scan_temp_bytes<3>(x.n_tiles); break;
        case 4: x.scan_temp_bytes = dual_tile_scan_temp_bytes<4>(x.n_tiles); break;
        case 5: x.scan_temp_bytes = dual_tile_scan_temp_bytes<5>(x.n_tiles); break;
        case 6: x.scan_temp_bytes = dual_tile_scan_temp_bytes<6>(x.n_tiles); break;
        default: throw std::logic_error("unreachable Renyi packed width");
    }
    const std::size_t two_l = 2 * x.length;
    x.types = DeviceBuffer<int32_t>(two_l);
    x.sites = DeviceBuffer<int32_t>(two_l);
    const std::size_t alias_count = static_cast<std::size_t>(n_groups) * max_alias;
    const std::size_t rmax_count = static_cast<std::size_t>(n_groups) * n_bonds;
    x.bond_sites = DeviceBuffer<int32_t>::view(
        x.model->bond_sites.get(), static_cast<std::size_t>(2) * n_bonds);
    x.bond_vij = DeviceBuffer<double>::view(x.model->bond_vij.get(), n_bonds);
    x.inv_coord = DeviceBuffer<double>::view(x.model->inv_coord.get(), n_sites);
    x.alias_prob = DeviceBuffer<double>::view(x.model->alias_prob.get(), alias_count);
    x.alias_index = DeviceBuffer<int32_t>::view(x.model->alias_index.get(), alias_count);
    x.alias_loc_kind = DeviceBuffer<int32_t>::view(
        x.model->alias_loc_kind.get(), alias_count);
    x.bond_rmax = DeviceBuffer<double>::view(x.model->bond_rmax.get(), rmax_count);
    x.mask_words = DeviceBuffer<uint64_t>(x.words);
    const std::size_t tile_bytes = x.n_tiles * 2 * x.words * sizeof(uint64_t);
    x.tile_parity = DeviceBuffer<uint8_t>(tile_bytes);
    x.tile_prefix = DeviceBuffer<uint8_t>(tile_bytes);
    x.scan_temp = DeviceBuffer<uint8_t>(x.scan_temp_bytes);
    x.diagonal_stats = DeviceBuffer<DeviceDiagonalStats>(1);
    x.actual_boundaries = DeviceBuffer<uint64_t>(
        static_cast<std::size_t>(4) * x.words);
    x.topology_stats = DeviceBuffer<DeviceTopologyStats>(1);
    x.topology_ratio = DeviceBuffer<DeviceTopologyRatio>(1);

    check_cuda(cudaMemcpy(x.types.get(), host_types, two_l * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy Renyi operator types");
    check_cuda(cudaMemcpy(x.sites.get(), host_sites, two_l * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy Renyi operator sites");
    check_cuda(cudaMemset(x.mask_words.get(), 0,
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t)),
               "clear Renyi mask");
}

RenyiEngine::~RenyiEngine() = default;
RenyiEngine::RenyiEngine(RenyiEngine&&) noexcept = default;
RenyiEngine& RenyiEngine::operator=(RenyiEngine&&) noexcept = default;

void RenyiEngine::set_cut(int cut) {
    Impl& x = *impl_;
    if (cut < 0 || static_cast<std::size_t>(cut) > x.length)
        throw std::invalid_argument("Renyi cut must be in [0, operator length]");
    x.cut = cut;
    x.events_valid = false;
    x.actual_boundaries_valid = false;
}

void RenyiEngine::set_mask(const uint8_t* mask, int count) {
    Impl& x = *impl_;
    if (count != x.n_sites)
        throw std::invalid_argument("Renyi mask length must equal n_sites");
    std::array<uint64_t, kMaxWords> packed{};
    for (int site = 0; site < count; ++site) {
        if (mask[site] > 1)
            throw std::invalid_argument("Renyi mask entries must be zero or one");
        if (mask[site]) packed[site >> 6] |= uint64_t{1} << (site & 63);
    }
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi mask");
    check_cuda(cudaMemcpy(x.mask_words.get(), packed.data(),
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice), "copy Renyi mask");
    x.events_valid = false;
}

void RenyiEngine::get_mask(uint8_t* mask, int count) const {
    const Impl& x = *impl_;
    if (count != x.n_sites)
        throw std::invalid_argument("Renyi mask length must equal n_sites");
    std::array<uint64_t, kMaxWords> packed{};
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi mask download");
    check_cuda(cudaMemcpy(packed.data(), x.mask_words.get(),
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost), "download Renyi mask");
    for (int site = 0; site < count; ++site)
        mask[site] = static_cast<uint8_t>((packed[site >> 6] >> (site & 63)) & 1ULL);
}

DiagonalStats RenyiEngine::diagonal_update(uint64_t seed, uint64_t sweep_id) {
    Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi diagonal");
    check_cuda(cudaMemset(x.diagonal_stats.get(), 0, sizeof(DeviceDiagonalStats)),
               "clear Renyi diagonal stats");
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create Renyi diagonal start");
    try {
        check_cuda(cudaEventCreate(&stop), "create Renyi diagonal stop");
        check_cuda(cudaEventRecord(start), "record Renyi diagonal start");
        auto run = [&](auto tag) {
            constexpr int Words = decltype(tag)::value;
            run_renyi_diagonal_scan<Words>(
                x.types.get(), x.sites.get(), x.length, x.cut, x.mask_words.get(),
                x.half_length, x.delta_min, x.delta_max, x.epsilon,
                x.n_groups, x.max_alias, x.n_bonds, x.bond_sites.get(),
                x.bond_vij.get(), x.inv_coord.get(), x.alias_prob.get(),
                x.alias_index.get(), x.alias_loc_kind.get(), x.bond_rmax.get(),
                x.tile_parity.get(), x.tile_prefix.get(), x.scan_temp.get(),
                x.scan_temp_bytes, x.diagonal_stats.get(), seed, sweep_id);
        };
        switch (x.words) {
            case 1: run(std::integral_constant<int, 1>{}); break;
            case 2: run(std::integral_constant<int, 2>{}); break;
            case 3: run(std::integral_constant<int, 3>{}); break;
            case 4: run(std::integral_constant<int, 4>{}); break;
            case 5: run(std::integral_constant<int, 5>{}); break;
            case 6: run(std::integral_constant<int, 6>{}); break;
        }
        check_cuda(cudaEventRecord(stop), "record Renyi diagonal stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize Renyi diagonal");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "time Renyi diagonal");
        DeviceDiagonalStats raw{};
        check_cuda(cudaMemcpy(&raw, x.diagonal_stats.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost), "copy Renyi diagonal stats");
        x.events_valid = false;
        x.actual_boundaries_valid = false;
        x.last_site_events = 0;
        x.last_bond_events = 0;
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return DiagonalStats{raw.updated_slots, raw.proposal_attempts,
                             raw.bond_proposals, raw.bond_accepts,
                             raw.failed_slots, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

EventStats RenyiEngine::build_events() {
    Impl& x = *impl_;
    if (x.length > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 4)
        throw std::runtime_error("Renyi operator string exceeds packed event limit");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi events");
    const std::size_t site_capacity = 2 * x.length;
    const std::size_t bond_capacity = 4 * x.length;
    if (x.site_keys_in.size() == 0) {
        x.site_keys_in = DeviceBuffer<uint64_t>(site_capacity);
        x.site_keys_out = DeviceBuffer<uint64_t>(site_capacity);
        x.site_values_in = DeviceBuffer<uint32_t>(site_capacity);
        x.site_values_out = DeviceBuffer<uint32_t>(site_capacity);
        x.bond_keys_in = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_keys_out = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_values_in = DeviceBuffer<uint32_t>(bond_capacity);
        x.bond_values_out = DeviceBuffer<uint32_t>(bond_capacity);
        x.bond_spin = DeviceBuffer<int8_t>(2 * x.length);
        x.event_counts = DeviceBuffer<DeviceEventCounts>(1);
        std::size_t site_temp = 0, bond_temp = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, site_temp, x.site_keys_in.get(), x.site_keys_out.get(),
                       x.site_values_in.get(), x.site_values_out.get(),
                       static_cast<int>(site_capacity)),
                   "size Renyi site-event radix sort");
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, bond_temp, x.bond_keys_in.get(), x.bond_keys_out.get(),
                       x.bond_values_in.get(), x.bond_values_out.get(),
                       static_cast<int>(bond_capacity)),
                   "size Renyi bond-event radix sort");
        x.event_sort_temp_bytes = std::max(site_temp, bond_temp);
        x.event_sort_temp = DeviceBuffer<uint8_t>(x.event_sort_temp_bytes);
    }
    check_cuda(cudaMemset(x.event_counts.get(), 0, sizeof(DeviceEventCounts)),
               "clear Renyi event counts");
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create Renyi event start");
    try {
        check_cuda(cudaEventCreate(&stop), "create Renyi event stop");
        check_cuda(cudaEventRecord(start), "record Renyi event start");
        auto run = [&](auto tag) {
            constexpr int Words = decltype(tag)::value;
            run_renyi_event_scan<Words>(
                x.types.get(), x.sites.get(), x.length, x.cut, x.mask_words.get(),
                x.n_sites, x.bond_sites.get(), x.tile_parity.get(),
                x.tile_prefix.get(), x.scan_temp.get(), x.scan_temp_bytes,
                x.site_keys_in.get(), x.site_values_in.get(),
                x.bond_keys_in.get(), x.bond_values_in.get(),
                x.bond_spin.get(), x.event_counts.get());
        };
        switch (x.words) {
            case 1: run(std::integral_constant<int, 1>{}); break;
            case 2: run(std::integral_constant<int, 2>{}); break;
            case 3: run(std::integral_constant<int, 3>{}); break;
            case 4: run(std::integral_constant<int, 4>{}); break;
            case 5: run(std::integral_constant<int, 5>{}); break;
            case 6: run(std::integral_constant<int, 6>{}); break;
        }
        std::size_t temp_bytes = x.event_sort_temp_bytes;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       x.event_sort_temp.get(), temp_bytes,
                       x.site_keys_in.get(), x.site_keys_out.get(),
                       x.site_values_in.get(), x.site_values_out.get(),
                       static_cast<int>(site_capacity)),
                   "sort Renyi site events");
        temp_bytes = x.event_sort_temp_bytes;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       x.event_sort_temp.get(), temp_bytes,
                       x.bond_keys_in.get(), x.bond_keys_out.get(),
                       x.bond_values_in.get(), x.bond_values_out.get(),
                       static_cast<int>(bond_capacity)),
                   "sort Renyi bond events");
        check_cuda(cudaEventRecord(stop), "record Renyi event stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize Renyi events");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "time Renyi events");
        DeviceEventCounts raw{};
        check_cuda(cudaMemcpy(&raw, x.event_counts.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost), "copy Renyi event counts");
        x.last_site_events = raw.site_events;
        x.last_bond_events = raw.bond_events;
        x.events_valid = true;
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return EventStats{raw.site_events, raw.bond_events, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

ClusterStats RenyiEngine::cluster_update(uint64_t seed, uint64_t sweep_id) {
    Impl& x = *impl_;
    const EventStats event = build_events();
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi cluster");
    const int channel_sites = 2 * x.n_sites;
    if (x.site_heads.size() == 0) {
        x.site_heads = DeviceBuffer<int32_t>(channel_sites);
        x.site_counts = DeviceBuffer<int32_t>(channel_sites);
        x.bond_heads = DeviceBuffer<int32_t>(channel_sites);
        x.bond_counts = DeviceBuffer<int32_t>(channel_sites);
        x.cluster_stats = DeviceBuffer<DeviceClusterStats>(1);
    }
    const std::size_t flag_capacity = x.last_site_events + channel_sites;
    if (x.segment_flags.size() < flag_capacity)
        x.segment_flags = DeviceBuffer<uint8_t>(flag_capacity);
    check_cuda(cudaMemset(x.cluster_stats.get(), 0, sizeof(DeviceClusterStats)),
               "clear Renyi cluster stats");
    constexpr int bounds_block = 128;
    event_bounds_kernel<<<(channel_sites + bounds_block - 1) / bounds_block,
                          bounds_block>>>(
        x.site_keys_out.get(), x.last_site_events,
        x.bond_keys_out.get(), x.last_bond_events, channel_sites,
        x.site_heads.get(), x.site_counts.get(), x.bond_heads.get(), x.bond_counts.get());
    check_cuda(cudaGetLastError(), "launch Renyi event bounds");
    std::vector<int32_t> site_heads(channel_sites), site_counts(channel_sites);
    std::vector<int32_t> bond_heads(channel_sites), bond_counts(channel_sites);
    check_cuda(cudaMemcpy(site_heads.data(), x.site_heads.get(),
                          channel_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "copy Renyi site heads");
    check_cuda(cudaMemcpy(site_counts.data(), x.site_counts.get(),
                          channel_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "copy Renyi site counts");
    check_cuda(cudaMemcpy(bond_heads.data(), x.bond_heads.get(),
                          channel_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "copy Renyi bond heads");
    check_cuda(cudaMemcpy(bond_counts.data(), x.bond_counts.get(),
                          channel_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "copy Renyi bond counts");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create Renyi cluster start");
    try {
        check_cuda(cudaEventCreate(&stop), "create Renyi cluster stop");
        check_cuda(cudaEventRecord(start), "record Renyi cluster start");
        // Preserve the trusted CPU engine's physical-site then channel order.
        // Later batching can execute independent chains concurrently without
        // changing this within-chain transition ordering.
        for (int site = 0; site < x.n_sites; ++site) {
            for (int channel = 0; channel < 2; ++channel) {
                const int group = channel * x.n_sites + site;
                const int n_sops = site_counts[group];
                if (n_sops == 0) continue;
                const int segment_head = site_heads[group] + group;
                renyi_cluster_segments_kernel<<<n_sops + 1, kClusterBlockSize>>>(
                    group, site_heads[group], n_sops,
                    bond_heads[group], bond_counts[group],
                    x.site_values_out.get(), x.bond_values_out.get(),
                    x.types.get(), x.sites.get(), x.length,
                    x.bond_sites.get(), x.bond_vij.get(), x.inv_coord.get(),
                    x.bond_spin.get(), x.segment_flags.get(), segment_head,
                    x.half_length, x.delta_min, x.delta_max, x.epsilon,
                    seed, sweep_id, x.cluster_stats.get());
                check_cuda(cudaGetLastError(), "launch Renyi cluster segments");
                const int blocks = (n_sops + kBlockSize - 1) / kBlockSize;
                renyi_apply_site_segment_flips_kernel<<<blocks, kBlockSize>>>(
                    site_heads[group], n_sops, x.site_values_out.get(),
                    x.segment_flags.get(), segment_head, x.length, x.types.get());
                check_cuda(cudaGetLastError(), "launch Renyi site flips");
            }
        }
        check_cuda(cudaEventRecord(stop), "record Renyi cluster stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize Renyi cluster");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "time Renyi cluster");
        DeviceClusterStats raw{};
        check_cuda(cudaMemcpy(&raw, x.cluster_stats.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost), "copy Renyi cluster stats");
        // Event positions and bond spins are updated in-place and remain
        // valid because cluster moves only toggle type 1 <-> -1.
        x.events_valid = true;
        x.actual_boundaries_valid = false;
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return ClusterStats{raw.proposed_segments, raw.accepted_segments,
                            event.elapsed_ms, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

TopologyRatio RenyiEngine::log_weight_ratio_for_toggle(int site) {
    Impl& x = *impl_;
    if (site < 0 || site >= x.n_sites)
        throw std::invalid_argument("Renyi topology site is out of range");
    check_cuda(cudaSetDevice(x.device_index),
               "cudaSetDevice for Renyi topology ratio");
    if (!x.actual_boundaries_valid) {
        auto materialise = [&](auto tag) {
            constexpr int Words = decltype(tag)::value;
            run_actual_boundary_materialisation<Words>(
                x.types.get(), x.sites.get(), x.length, x.cut,
                x.tile_parity.get(), x.tile_prefix.get(), x.scan_temp.get(),
                x.scan_temp_bytes, x.actual_boundaries.get());
        };
        switch (x.words) {
            case 1: materialise(std::integral_constant<int, 1>{}); break;
            case 2: materialise(std::integral_constant<int, 2>{}); break;
            case 3: materialise(std::integral_constant<int, 3>{}); break;
            case 4: materialise(std::integral_constant<int, 4>{}); break;
            case 5: materialise(std::integral_constant<int, 5>{}); break;
            case 6: materialise(std::integral_constant<int, 6>{}); break;
        }
        x.actual_boundaries_valid = true;
    }
    auto run = [&](auto tag) {
        constexpr int Words = decltype(tag)::value;
        renyi_compact_topology_ratio_kernel<Words><<<1, 1>>>(
            x.mask_words.get(), x.actual_boundaries.get(), site,
            x.topology_ratio.get());
        check_cuda(cudaGetLastError(), "launch Renyi topology ratio");
    };
    switch (x.words) {
        case 1: run(std::integral_constant<int, 1>{}); break;
        case 2: run(std::integral_constant<int, 2>{}); break;
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
    }
    DeviceTopologyRatio raw{};
    check_cuda(cudaMemcpy(&raw, x.topology_ratio.get(), sizeof(raw),
                          cudaMemcpyDeviceToHost),
               "copy Renyi topology ratio");
    return TopologyRatio{raw.log_ratio, raw.current_valid != 0,
                         raw.proposed_valid != 0};
}

TopologyStats RenyiEngine::topology_sweep(const int32_t* host_topology_sites,
                                          int count,
                                          double lambda,
                                          uint64_t seed,
                                          uint64_t sweep_id) {
    Impl& x = *impl_;
    if (count < 0 || count > x.n_sites)
        throw std::invalid_argument("invalid Renyi topology-site count");
    if (lambda <= 0.0 || lambda >= 1.0)
        throw std::invalid_argument("topology lambda must lie strictly inside (0, 1)");
    if (count == 0) return TopologyStats{0, 0, 0, 0, 0.0f};
    std::vector<uint8_t> seen(static_cast<std::size_t>(x.n_sites), 0);
    for (int k = 0; k < count; ++k) {
        const int site = host_topology_sites[k];
        if (site < 0 || site >= x.n_sites || seen[site])
            throw std::invalid_argument("topology sites must be unique valid sites");
        seen[site] = 1;
    }
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi topology");
    if (x.topology_sites.size() < static_cast<std::size_t>(count))
        x.topology_sites = DeviceBuffer<int32_t>(static_cast<std::size_t>(count));
    check_cuda(cudaMemcpy(x.topology_sites.get(), host_topology_sites,
                          static_cast<std::size_t>(count) * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy Renyi topology sites");
    if (!x.actual_boundaries_valid) {
        auto materialise = [&](auto tag) {
            constexpr int Words = decltype(tag)::value;
            run_actual_boundary_materialisation<Words>(
                x.types.get(), x.sites.get(), x.length, x.cut,
                x.tile_parity.get(), x.tile_prefix.get(), x.scan_temp.get(),
                x.scan_temp_bytes, x.actual_boundaries.get());
        };
        switch (x.words) {
            case 1: materialise(std::integral_constant<int, 1>{}); break;
            case 2: materialise(std::integral_constant<int, 2>{}); break;
            case 3: materialise(std::integral_constant<int, 3>{}); break;
            case 4: materialise(std::integral_constant<int, 4>{}); break;
            case 5: materialise(std::integral_constant<int, 5>{}); break;
            case 6: materialise(std::integral_constant<int, 6>{}); break;
        }
        x.actual_boundaries_valid = true;
    }
    check_cuda(cudaMemset(x.topology_stats.get(), 0, sizeof(DeviceTopologyStats)),
               "clear Renyi topology stats");
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create Renyi topology start");
    try {
        check_cuda(cudaEventCreate(&stop), "create Renyi topology stop");
        check_cuda(cudaEventRecord(start), "record Renyi topology start");
        auto run = [&](auto tag) {
            constexpr int Words = decltype(tag)::value;
            renyi_compact_topology_sweep_kernel<Words><<<1, 1>>>(
                x.mask_words.get(), x.actual_boundaries.get(), x.topology_sites.get(),
                count, lambda, seed, sweep_id, x.topology_stats.get());
            check_cuda(cudaGetLastError(), "launch Renyi topology sweep");
        };
        switch (x.words) {
            case 1: run(std::integral_constant<int, 1>{}); break;
            case 2: run(std::integral_constant<int, 2>{}); break;
            case 3: run(std::integral_constant<int, 3>{}); break;
            case 4: run(std::integral_constant<int, 4>{}); break;
            case 5: run(std::integral_constant<int, 5>{}); break;
            case 6: run(std::integral_constant<int, 6>{}); break;
        }
        check_cuda(cudaEventRecord(stop), "record Renyi topology stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize Renyi topology");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "time Renyi topology");
        DeviceTopologyStats raw{};
        check_cuda(cudaMemcpy(&raw, x.topology_stats.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost), "copy Renyi topology stats");
        x.events_valid = false;
        x.last_site_events = 0;
        x.last_bond_events = 0;
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return TopologyStats{raw.attempts, raw.accepts, raw.invalid,
                             raw.active_count, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

void RenyiEngine::save_checkpoint() {
    Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi checkpoint");
    const std::size_t count = 2 * x.length;
    if (x.checkpoint_types.size() == 0) {
        x.checkpoint_types = DeviceBuffer<int32_t>(count);
        x.checkpoint_sites = DeviceBuffer<int32_t>(count);
    }
    check_cuda(cudaMemcpy(x.checkpoint_types.get(), x.types.get(),
                          count * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "save Renyi type checkpoint");
    check_cuda(cudaMemcpy(x.checkpoint_sites.get(), x.sites.get(),
                          count * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "save Renyi site checkpoint");
    x.checkpoint_valid = true;
}

void RenyiEngine::restore_checkpoint() {
    Impl& x = *impl_;
    if (!x.checkpoint_valid)
        throw std::runtime_error("no Renyi checkpoint has been saved");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi restore");
    const std::size_t count = 2 * x.length;
    check_cuda(cudaMemcpy(x.types.get(), x.checkpoint_types.get(),
                          count * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "restore Renyi type checkpoint");
    check_cuda(cudaMemcpy(x.sites.get(), x.checkpoint_sites.get(),
                          count * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "restore Renyi site checkpoint");
    x.events_valid = false;
    x.actual_boundaries_valid = false;
    x.last_site_events = 0;
    x.last_bond_events = 0;
}

bool RenyiEngine::has_checkpoint() const { return impl_->checkpoint_valid; }

void RenyiEngine::get_site_events(uint64_t* host_keys,
                                  uint32_t* host_values) const {
    const Impl& x = *impl_;
    if (!x.events_valid)
        throw std::runtime_error("build_events must be called before download");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi site download");
    check_cuda(cudaMemcpy(host_keys, x.site_keys_out.get(),
                          x.last_site_events * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost), "download Renyi site keys");
    check_cuda(cudaMemcpy(host_values, x.site_values_out.get(),
                          x.last_site_events * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "download Renyi site values");
}

void RenyiEngine::get_bond_events(uint64_t* host_keys,
                                  uint32_t* host_values) const {
    const Impl& x = *impl_;
    if (!x.events_valid)
        throw std::runtime_error("build_events must be called before download");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi bond download");
    check_cuda(cudaMemcpy(host_keys, x.bond_keys_out.get(),
                          x.last_bond_events * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost), "download Renyi bond keys");
    check_cuda(cudaMemcpy(host_values, x.bond_values_out.get(),
                          x.last_bond_events * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost), "download Renyi bond values");
}

void RenyiEngine::get_bond_spin(int8_t* host_values) const {
    const Impl& x = *impl_;
    if (!x.events_valid)
        throw std::runtime_error("build_events must be called before download");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi spin download");
    check_cuda(cudaMemcpy(host_values, x.bond_spin.get(),
                          2 * x.length * sizeof(int8_t), cudaMemcpyDeviceToHost),
               "download Renyi bond spins");
}

void RenyiEngine::get_operator_strings(int32_t* host_types,
                                       int32_t* host_sites) const {
    const Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi download");
    const std::size_t count = 2 * x.length;
    check_cuda(cudaMemcpy(host_types, x.types.get(), count * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "download Renyi operator types");
    check_cuda(cudaMemcpy(host_sites, x.sites.get(), count * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "download Renyi operator sites");
}

void RenyiEngine::set_operator_strings(const int32_t* host_types,
                                       const int32_t* host_sites) {
    Impl& x = *impl_;
    validate_operator_string(host_types, host_sites, 2 * x.length,
                             x.n_sites, x.n_bonds, "Renyi operator string");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for Renyi upload");
    const std::size_t count = 2 * x.length;
    check_cuda(cudaMemcpy(x.types.get(), host_types, count * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "upload Renyi operator types");
    check_cuda(cudaMemcpy(x.sites.get(), host_sites, count * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "upload Renyi operator sites");
    x.events_valid = false;
    x.actual_boundaries_valid = false;
    x.last_site_events = 0;
    x.last_bond_events = 0;
}

int RenyiEngine::n_sites() const { return impl_->n_sites; }
int RenyiEngine::half_length() const { return impl_->half_length; }
std::size_t RenyiEngine::length() const { return impl_->length; }
int RenyiEngine::cut() const { return impl_->cut; }
int RenyiEngine::packed_words() const { return impl_->words; }
std::size_t RenyiEngine::device_bytes() const { return impl_->allocated_bytes(); }


}  // namespace qaqmc_cuda
