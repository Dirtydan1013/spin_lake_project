#include "qaqmc_cuda_diagonal.cuh"
#include "detail/qaqmc_cuda_diagonal_state.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qaqmc_cuda {
using namespace detail;
DiagonalEngine::DiagonalEngine(int n_sites,
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
    if (half_length <= 0) throw std::invalid_argument("half_length must be positive");
    if (n_groups <= 0 || max_alias <= 0)
        throw std::invalid_argument("n_groups and max_alias must be positive");
    if (n_bonds < 0) throw std::invalid_argument("n_bonds must be non-negative");
    if (max_alias != n_sites + n_bonds)
        throw std::invalid_argument("max_alias must equal n_sites + n_bonds");

    check_cuda(cudaSetDevice(device_index), "cudaSetDevice");
    Impl& x = *impl_;
    x.device_index = device_index;
    x.n_sites = n_sites;
    x.half_length = half_length;
    x.length = static_cast<std::size_t>(2) * half_length;
    x.words = (n_sites + 63) / 64;
    x.delta_min = delta_min;
    x.delta_max = delta_max;
    x.epsilon = epsilon;
    x.n_groups = n_groups;
    x.max_alias = max_alias;
    x.n_bonds = n_bonds;
    x.n_tiles = (x.length + kBlockSize - 1) / kBlockSize;

    switch (x.words) {
        case 1: x.scan_temp_bytes = tile_scan_temp_bytes<1>(x.n_tiles); break;
        case 2: x.scan_temp_bytes = tile_scan_temp_bytes<2>(x.n_tiles); break;
        case 3: x.scan_temp_bytes = tile_scan_temp_bytes<3>(x.n_tiles); break;
        case 4: x.scan_temp_bytes = tile_scan_temp_bytes<4>(x.n_tiles); break;
        case 5: x.scan_temp_bytes = tile_scan_temp_bytes<5>(x.n_tiles); break;
        case 6: x.scan_temp_bytes = tile_scan_temp_bytes<6>(x.n_tiles); break;
        default: throw std::logic_error("unreachable packed-state width");
    }

    x.types = DeviceBuffer<int32_t>(x.length);
    x.sites = DeviceBuffer<int32_t>(x.length);
    x.bond_sites = DeviceBuffer<int32_t>(static_cast<std::size_t>(2) * n_bonds);
    x.bond_vij = DeviceBuffer<double>(n_bonds);
    x.inv_coord = DeviceBuffer<double>(n_sites);
    const std::size_t alias_count = static_cast<std::size_t>(n_groups) * max_alias;
    const std::size_t rmax_count = static_cast<std::size_t>(n_groups) * n_bonds;
    x.alias_prob = DeviceBuffer<double>(alias_count);
    x.alias_index = DeviceBuffer<int32_t>(alias_count);
    x.alias_loc_kind = DeviceBuffer<int32_t>(alias_count);
    x.bond_rmax = DeviceBuffer<double>(rmax_count);
    const std::size_t tile_bytes = x.n_tiles * x.words * sizeof(uint64_t);
    x.tile_parity = DeviceBuffer<uint8_t>(tile_bytes);
    x.tile_prefix = DeviceBuffer<uint8_t>(tile_bytes);
    x.scan_temp = DeviceBuffer<uint8_t>(x.scan_temp_bytes);
    x.stats = DeviceBuffer<DeviceDiagonalStats>(1);
    x.seam_words = DeviceBuffer<uint64_t>(x.words);
    x.seam_mask = DeviceBuffer<uint64_t>(1);
    x.repair_state = DeviceBuffer<DeviceRepairState>(1);
    x.topology_stats = DeviceBuffer<DeviceTopologyStats>(1);
    x.half_line_proposal = DeviceBuffer<DeviceHalfLineProposal>(1);
    check_cuda(cudaMemset(x.seam_words.get(), 0,
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t)),
               "clear seam words");
    check_cuda(cudaMemset(x.seam_mask.get(), 0, sizeof(uint64_t)),
               "clear seam mask");

    set_operator_string(host_types, host_sites);
    if (n_bonds > 0) {
        check_cuda(cudaMemcpy(x.bond_sites.get(), host_bond_sites,
                              static_cast<std::size_t>(2) * n_bonds * sizeof(int32_t),
                              cudaMemcpyHostToDevice), "copy bond sites to device");
        check_cuda(cudaMemcpy(x.bond_vij.get(), host_bond_vij,
                              static_cast<std::size_t>(n_bonds) * sizeof(double),
                              cudaMemcpyHostToDevice), "copy bond strengths to device");
        check_cuda(cudaMemcpy(x.bond_rmax.get(), host_bond_rmax,
                              rmax_count * sizeof(double), cudaMemcpyHostToDevice),
                   "copy bond envelopes to device");
    }
    check_cuda(cudaMemcpy(x.inv_coord.get(), host_inv_coord,
                          static_cast<std::size_t>(n_sites) * sizeof(double),
                          cudaMemcpyHostToDevice), "copy inverse coordination to device");
    check_cuda(cudaMemcpy(x.alias_prob.get(), host_alias_prob,
                          alias_count * sizeof(double), cudaMemcpyHostToDevice),
               "copy alias probabilities to device");
    check_cuda(cudaMemcpy(x.alias_index.get(), host_alias_index,
                          alias_count * sizeof(int32_t), cudaMemcpyHostToDevice),
               "copy alias indices to device");
    check_cuda(cudaMemcpy(x.alias_loc_kind.get(), host_alias_loc_kind,
                          alias_count * sizeof(int32_t), cudaMemcpyHostToDevice),
               "copy alias locations to device");
}

DiagonalEngine::~DiagonalEngine() = default;
DiagonalEngine::DiagonalEngine(DiagonalEngine&&) noexcept = default;
DiagonalEngine& DiagonalEngine::operator=(DiagonalEngine&&) noexcept = default;

void DiagonalEngine::set_string_sites(const int32_t* host_sites,
                                      int count,
                                      int m_star) {
    Impl& x = *impl_;
    if (count < 0 || count > 64)
        throw std::invalid_argument("string site count must be in [0, 64]");
    if (m_star < 0 || static_cast<std::size_t>(m_star) >= x.length)
        throw std::invalid_argument("m_star must be in [0, operator length)");
    std::vector<uint8_t> seen(static_cast<std::size_t>(x.n_sites), 0);
    x.host_string_sites.assign(host_sites, host_sites + count);
    for (const int site : x.host_string_sites) {
        if (site < 0 || site >= x.n_sites)
            throw std::invalid_argument("string site is outside the lattice");
        if (seen[site])
            throw std::invalid_argument("string sites must be unique");
        seen[site] = 1;
    }
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for string seam");
    x.string_sites = DeviceBuffer<int32_t>(static_cast<std::size_t>(count));
    if (count > 0) {
        check_cuda(cudaMemcpy(x.string_sites.get(), host_sites,
                              static_cast<std::size_t>(count) * sizeof(int32_t),
                              cudaMemcpyHostToDevice),
                   "copy string sites to device");
    }
    x.seam_cut = m_star;
    x.host_seam_mask = 0;
    check_cuda(cudaMemset(x.seam_mask.get(), 0, sizeof(uint64_t)),
               "reset seam mask");
    check_cuda(cudaMemset(x.seam_words.get(), 0,
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t)),
               "reset physical seam words");
    x.events_valid = false;
    x.last_site_events = 0;
    x.last_bond_events = 0;
    x.checkpoint_valid = false;
}

void DiagonalEngine::set_seam_mask_consistent(uint64_t mask) {
    Impl& x = *impl_;
    const int count = static_cast<int>(x.host_string_sites.size());
    if (x.seam_cut < 0)
        throw std::runtime_error("set_string_sites must be called first");
    if (count < 64 && (mask >> count) != 0)
        throw std::invalid_argument("seam mask has bits outside string_sites");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for seam repair");
    const unsigned blocks = static_cast<unsigned>(
        (x.length + kBlockSize - 1) / kBlockSize);
    for (int local = 0; local < count; ++local) {
        reset_repair_state_kernel<<<1, 1>>>(x.repair_state.get());
        check_cuda(cudaGetLastError(), "launch reset closure repair");
        scan_closure_repair_kernel<<<blocks, kBlockSize>>>(
            x.types.get(), x.sites.get(), x.length,
            x.host_string_sites[local], x.repair_state.get());
        check_cuda(cudaGetLastError(), "launch closure repair scan");
        commit_closure_repair_kernel<<<1, 1>>>(
            x.types.get(), x.sites.get(), x.host_string_sites[local],
            static_cast<int>((mask >> local) & 1ULL), x.repair_state.get());
        check_cuda(cudaGetLastError(), "launch closure repair commit");
        DeviceRepairState result{};
        check_cuda(cudaMemcpy(&result, x.repair_state.get(), sizeof(result),
                              cudaMemcpyDeviceToHost),
                   "copy closure repair status");
        if (result.failed)
            throw std::runtime_error(
                "cannot repair string worldline closure: no repurposable operator");
    }

    std::array<uint64_t, kMaxWords> packed{};
    for (int local = 0; local < count; ++local) {
        if (((mask >> local) & 1ULL) == 0) continue;
        const int site = x.host_string_sites[local];
        packed[site >> 6] ^= uint64_t{1} << (site & 63);
    }
    check_cuda(cudaMemcpy(x.seam_words.get(), packed.data(),
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t),
                          cudaMemcpyHostToDevice),
               "copy physical seam mask");
    check_cuda(cudaMemcpy(x.seam_mask.get(), &mask, sizeof(mask),
                          cudaMemcpyHostToDevice),
               "copy seam topology mask");
    x.host_seam_mask = mask;
    x.events_valid = false;
    x.last_site_events = 0;
    x.last_bond_events = 0;
}

HalfLineProposal DiagonalEngine::half_line_proposal(int local_index,
                                                    bool direction_right) {
    Impl& x = *impl_;
    if (x.seam_cut < 0)
        throw std::runtime_error("set_string_sites must be called first");
    if (local_index < 0
        || local_index >= static_cast<int>(x.host_string_sites.size()))
        throw std::invalid_argument("string local index is out of range");
    if (!x.events_valid) build_events();
    check_cuda(cudaSetDevice(x.device_index),
               "cudaSetDevice for half-line proposal");
    if (x.site_heads.size() == 0) {
        x.site_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.site_counts = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_counts = DeviceBuffer<int32_t>(x.n_sites);
    }
    constexpr int bounds_block = 128;
    event_bounds_kernel<<<(x.n_sites + bounds_block - 1) / bounds_block,
                           bounds_block>>>(
        x.site_keys_out.get(), x.last_site_events,
        x.bond_keys_out.get(), x.last_bond_events, x.n_sites,
        x.site_heads.get(), x.site_counts.get(),
        x.bond_heads.get(), x.bond_counts.get());
    check_cuda(cudaGetLastError(), "launch half-line event bounds");
    offdiagonal_half_line_proposal_kernel<<<1, kClusterBlockSize>>>(
        x.half_length, x.delta_min, x.delta_max, x.epsilon,
        x.bond_sites.get(), x.bond_vij.get(), x.inv_coord.get(),
        x.site_values_out.get(), x.bond_values_out.get(),
        x.site_heads.get(), x.site_counts.get(),
        x.bond_heads.get(), x.bond_counts.get(), x.bond_spin.get(),
        x.host_string_sites[local_index], x.seam_cut, direction_right,
        x.half_line_proposal.get());
    check_cuda(cudaGetLastError(), "launch half-line proposal diagnostic");
    DeviceHalfLineProposal raw{};
    check_cuda(cudaMemcpy(&raw, x.half_line_proposal.get(), sizeof(raw),
                          cudaMemcpyDeviceToHost),
               "copy half-line proposal diagnostic");
    return HalfLineProposal{raw.valid != 0, raw.terminal_p,
                            raw.log_physical_ratio};
}

TopologyStats DiagonalEngine::topology_sweep(double lambda,
                                             uint64_t seed,
                                             uint64_t sweep_id) {
    Impl& x = *impl_;
    if (lambda <= 0.0 || lambda >= 1.0)
        throw std::invalid_argument("topology lambda must lie strictly inside (0, 1)");
    if (x.seam_cut < 0)
        throw std::runtime_error("set_string_sites must be called first");
    if (x.host_string_sites.empty()) return TopologyStats{0, 0, 0, 0, 0.0f};
    if (!x.events_valid) build_events();
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for topology sweep");
    if (x.site_heads.size() == 0) {
        x.site_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.site_counts = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_counts = DeviceBuffer<int32_t>(x.n_sites);
    }
    constexpr int bounds_block = 128;
    event_bounds_kernel<<<(x.n_sites + bounds_block - 1) / bounds_block, bounds_block>>>(
        x.site_keys_out.get(), x.last_site_events,
        x.bond_keys_out.get(), x.last_bond_events, x.n_sites,
        x.site_heads.get(), x.site_counts.get(), x.bond_heads.get(), x.bond_counts.get());
    check_cuda(cudaGetLastError(), "launch topology event bounds");
    check_cuda(cudaMemset(x.topology_stats.get(), 0, sizeof(DeviceTopologyStats)),
               "clear topology stats");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create topology start event");
    try {
        check_cuda(cudaEventCreate(&stop), "create topology stop event");
        check_cuda(cudaEventRecord(start), "record topology start");
        offdiagonal_topology_sweep_kernel<<<1, kClusterBlockSize>>>(
            x.types.get(), x.length, x.half_length,
            x.delta_min, x.delta_max, x.epsilon,
            x.bond_sites.get(), x.bond_vij.get(), x.inv_coord.get(),
            x.site_values_out.get(), x.bond_values_out.get(),
            x.site_heads.get(), x.site_counts.get(),
            x.bond_heads.get(), x.bond_counts.get(), x.bond_spin.get(),
            x.string_sites.get(), static_cast<int>(x.host_string_sites.size()),
            x.seam_cut, x.seam_mask.get(), x.seam_words.get(),
            lambda, seed, sweep_id, x.topology_stats.get());
        check_cuda(cudaGetLastError(), "launch off-diagonal topology sweep");
        check_cuda(cudaEventRecord(stop), "record topology stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize topology sweep");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "time topology sweep");
        DeviceTopologyStats raw{};
        check_cuda(cudaMemcpy(&raw, x.topology_stats.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost),
                   "copy topology stats");
        check_cuda(cudaMemcpy(&x.host_seam_mask, x.seam_mask.get(), sizeof(uint64_t),
                              cudaMemcpyDeviceToHost),
                   "copy seam mask after topology sweep");
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

void DiagonalEngine::save_checkpoint() {
    Impl& x = *impl_;
    if (x.seam_cut < 0)
        throw std::runtime_error("set_string_sites must be called before checkpoint");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for string checkpoint");
    if (x.checkpoint_types.size() == 0) {
        x.checkpoint_types = DeviceBuffer<int32_t>(x.length);
        x.checkpoint_sites = DeviceBuffer<int32_t>(x.length);
        x.checkpoint_seam_words = DeviceBuffer<uint64_t>(x.words);
        x.checkpoint_seam_mask = DeviceBuffer<uint64_t>(1);
    }
    check_cuda(cudaMemcpy(x.checkpoint_types.get(), x.types.get(),
                          x.length * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "save string type checkpoint");
    check_cuda(cudaMemcpy(x.checkpoint_sites.get(), x.sites.get(),
                          x.length * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "save string site checkpoint");
    check_cuda(cudaMemcpy(x.checkpoint_seam_words.get(), x.seam_words.get(),
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t),
                          cudaMemcpyDeviceToDevice),
               "save string seam-word checkpoint");
    check_cuda(cudaMemcpy(x.checkpoint_seam_mask.get(), x.seam_mask.get(),
                          sizeof(uint64_t), cudaMemcpyDeviceToDevice),
               "save string seam-mask checkpoint");
    x.host_checkpoint_seam_mask = x.host_seam_mask;
    x.checkpoint_valid = true;
}

void DiagonalEngine::restore_checkpoint() {
    Impl& x = *impl_;
    if (!x.checkpoint_valid)
        throw std::runtime_error("no string-work checkpoint has been saved");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for string restore");
    check_cuda(cudaMemcpy(x.types.get(), x.checkpoint_types.get(),
                          x.length * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "restore string type checkpoint");
    check_cuda(cudaMemcpy(x.sites.get(), x.checkpoint_sites.get(),
                          x.length * sizeof(int32_t), cudaMemcpyDeviceToDevice),
               "restore string site checkpoint");
    check_cuda(cudaMemcpy(x.seam_words.get(), x.checkpoint_seam_words.get(),
                          static_cast<std::size_t>(x.words) * sizeof(uint64_t),
                          cudaMemcpyDeviceToDevice),
               "restore string seam-word checkpoint");
    check_cuda(cudaMemcpy(x.seam_mask.get(), x.checkpoint_seam_mask.get(),
                          sizeof(uint64_t), cudaMemcpyDeviceToDevice),
               "restore string seam-mask checkpoint");
    x.host_seam_mask = x.host_checkpoint_seam_mask;
    x.last_site_events = 0;
    x.last_bond_events = 0;
    x.events_valid = false;
}

bool DiagonalEngine::has_checkpoint() const { return impl_->checkpoint_valid; }

DiagonalStats DiagonalEngine::diagonal_update(uint64_t seed, uint64_t sweep_id) {
    Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for diagonal update");
    check_cuda(cudaMemset(x.stats.get(), 0, sizeof(DeviceDiagonalStats)),
               "clear diagonal stats");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create start event");
    try {
        check_cuda(cudaEventCreate(&stop), "create stop event");
        check_cuda(cudaEventRecord(start), "record start event");

        auto run = [&](auto words_tag) {
            constexpr int Words = decltype(words_tag)::value;
            run_diagonal_scan_and_resample<Words>(
                x.types.get(), x.sites.get(), x.length, x.half_length,
                x.delta_min, x.delta_max, x.epsilon, x.n_groups, x.max_alias,
                x.n_bonds, x.bond_sites.get(), x.bond_vij.get(), x.inv_coord.get(),
                x.alias_prob.get(), x.alias_index.get(), x.alias_loc_kind.get(),
                x.bond_rmax.get(), x.seam_cut, x.seam_words.get(),
                x.tile_parity.get(), x.tile_prefix.get(),
                x.scan_temp.get(), x.scan_temp_bytes, x.stats.get(), seed, sweep_id);
        };
        switch (x.words) {
            case 1: run(std::integral_constant<int, 1>{}); break;
            case 2: run(std::integral_constant<int, 2>{}); break;
            case 3: run(std::integral_constant<int, 3>{}); break;
            case 4: run(std::integral_constant<int, 4>{}); break;
            case 5: run(std::integral_constant<int, 5>{}); break;
            case 6: run(std::integral_constant<int, 6>{}); break;
        }

        check_cuda(cudaEventRecord(stop), "record stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize diagonal update");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "time diagonal update");
        DeviceDiagonalStats raw{};
        check_cuda(cudaMemcpy(&raw, x.stats.get(), sizeof(raw), cudaMemcpyDeviceToHost),
                   "copy diagonal stats to host");
        x.last_site_events = 0;
        x.last_bond_events = 0;
        x.events_valid = false;
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

EventStats DiagonalEngine::build_events() {
    Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for event build");
    if (x.length > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2) {
        throw std::runtime_error("operator string is too long for CUB radix-sort item count");
    }

    if (x.site_keys_in.size() == 0) {
        x.site_keys_in = DeviceBuffer<uint64_t>(x.length);
        x.site_keys_out = DeviceBuffer<uint64_t>(x.length);
        x.site_values_in = DeviceBuffer<uint32_t>(x.length);
        x.site_values_out = DeviceBuffer<uint32_t>(x.length);
        const std::size_t bond_capacity = 2 * x.length;
        x.bond_keys_in = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_keys_out = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_values_in = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_values_out = DeviceBuffer<uint64_t>(bond_capacity);
        x.bond_spin = DeviceBuffer<int8_t>(x.length);
        x.event_counts = DeviceBuffer<DeviceEventCounts>(1);

        std::size_t site_temp = 0, bond_temp = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, site_temp, x.site_keys_in.get(), x.site_keys_out.get(),
                       x.site_values_in.get(), x.site_values_out.get(),
                       static_cast<int>(x.length)),
                   "size site-event radix sort");
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, bond_temp, x.bond_keys_in.get(), x.bond_keys_out.get(),
                       x.bond_values_in.get(), x.bond_values_out.get(),
                       static_cast<int>(bond_capacity)),
                   "size bond-event radix sort");
        x.event_sort_temp_bytes = std::max(site_temp, bond_temp);
        x.event_sort_temp = DeviceBuffer<uint8_t>(x.event_sort_temp_bytes);
    }

    check_cuda(cudaMemset(x.event_counts.get(), 0, sizeof(DeviceEventCounts)),
               "clear event counts");
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create event-build start event");
    try {
        check_cuda(cudaEventCreate(&stop), "create event-build stop event");
        check_cuda(cudaEventRecord(start), "record event-build start");

        auto run = [&](auto words_tag) {
            constexpr int Words = decltype(words_tag)::value;
            run_event_scan_and_generation<Words>(
                x.types.get(), x.sites.get(), x.length, x.bond_sites.get(),
                x.seam_cut, x.seam_words.get(),
                x.tile_parity.get(), x.tile_prefix.get(), x.scan_temp.get(),
                x.scan_temp_bytes, x.site_keys_in.get(), x.site_values_in.get(),
                x.bond_keys_in.get(), x.bond_values_in.get(), x.bond_spin.get(),
                x.event_counts.get());
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
                       static_cast<int>(x.length)),
                   "sort site events");
        temp_bytes = x.event_sort_temp_bytes;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       x.event_sort_temp.get(), temp_bytes,
                       x.bond_keys_in.get(), x.bond_keys_out.get(),
                       x.bond_values_in.get(), x.bond_values_out.get(),
                       static_cast<int>(2 * x.length)),
                   "sort bond events");

        check_cuda(cudaEventRecord(stop), "record event-build stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize event build");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "time event build");
        DeviceEventCounts counts{};
        check_cuda(cudaMemcpy(&counts, x.event_counts.get(), sizeof(counts),
                              cudaMemcpyDeviceToHost), "copy event counts to host");
        x.last_site_events = counts.site_events;
        x.last_bond_events = counts.bond_events;
        x.events_valid = true;
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return EventStats{x.last_site_events, x.last_bond_events, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

ClusterStats DiagonalEngine::cluster_update(uint64_t seed, uint64_t sweep_id) {
    Impl& x = *impl_;
    const EventStats event_stats = build_events();
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for cluster update");

    if (x.site_heads.size() == 0) {
        x.site_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.site_counts = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_heads = DeviceBuffer<int32_t>(x.n_sites);
        x.bond_counts = DeviceBuffer<int32_t>(x.n_sites);
        x.cluster_stats = DeviceBuffer<DeviceClusterStats>(1);
    }
    const std::size_t flag_capacity = x.last_site_events + x.n_sites;
    if (x.segment_flags.size() < flag_capacity)
        x.segment_flags = DeviceBuffer<uint8_t>(flag_capacity);
    check_cuda(cudaMemset(x.cluster_stats.get(), 0, sizeof(DeviceClusterStats)),
               "clear cluster stats");

    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create cluster start event");
    try {
        check_cuda(cudaEventCreate(&stop), "create cluster stop event");
        check_cuda(cudaEventRecord(start), "record cluster start");
        constexpr int bounds_block = 128;
        event_bounds_kernel<<<(x.n_sites + bounds_block - 1) / bounds_block, bounds_block>>>(
            x.site_keys_out.get(), x.last_site_events,
            x.bond_keys_out.get(), x.last_bond_events, x.n_sites,
            x.site_heads.get(), x.site_counts.get(), x.bond_heads.get(), x.bond_counts.get());
        check_cuda(cudaGetLastError(), "launch event_bounds_kernel");

        std::vector<int32_t> site_heads(x.n_sites), site_counts(x.n_sites);
        std::vector<int32_t> bond_heads(x.n_sites), bond_counts(x.n_sites);
        check_cuda(cudaMemcpy(site_heads.data(), x.site_heads.get(),
                              x.n_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "copy site heads to host");
        check_cuda(cudaMemcpy(site_counts.data(), x.site_counts.get(),
                              x.n_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "copy site counts to host");
        check_cuda(cudaMemcpy(bond_heads.data(), x.bond_heads.get(),
                              x.n_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "copy bond heads to host");
        check_cuda(cudaMemcpy(bond_counts.data(), x.bond_counts.get(),
                              x.n_sites * sizeof(int32_t), cudaMemcpyDeviceToHost),
                   "copy bond counts to host");

        for (int site = 0; site < x.n_sites; ++site) {
            const int n_sops = site_counts[site];
            if (n_sops == 0) continue;
            const int segment_head = site_heads[site] + site;
            const int segment_blocks = n_sops + 1;
            cluster_segments_for_site_kernel<<<segment_blocks, kClusterBlockSize>>>(
                site, site_heads[site], n_sops, bond_heads[site], bond_counts[site],
                x.site_values_out.get(), x.bond_values_out.get(), x.bond_sites.get(),
                x.bond_vij.get(), x.inv_coord.get(), x.bond_spin.get(),
                x.segment_flags.get(), segment_head, x.half_length,
                x.delta_min, x.delta_max, x.epsilon, seed, sweep_id,
                x.cluster_stats.get());
            check_cuda(cudaGetLastError(), "launch cluster_segments_for_site_kernel");
            const int op_blocks = (n_sops + kBlockSize - 1) / kBlockSize;
            apply_site_segment_flips_kernel<<<op_blocks, kBlockSize>>>(
                site_heads[site], n_sops, x.site_values_out.get(),
                x.segment_flags.get(), segment_head, x.types.get());
            check_cuda(cudaGetLastError(), "launch apply_site_segment_flips_kernel");
        }

        check_cuda(cudaEventRecord(stop), "record cluster stop");
        check_cuda(cudaEventSynchronize(stop), "synchronize cluster update");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "time cluster update");
        DeviceClusterStats raw{};
        check_cuda(cudaMemcpy(&raw, x.cluster_stats.get(), sizeof(raw),
                              cudaMemcpyDeviceToHost), "copy cluster stats to host");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return ClusterStats{raw.proposed_segments, raw.accepted_segments,
                            event_stats.elapsed_ms, elapsed_ms};
    } catch (...) {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        throw;
    }
}

void DiagonalEngine::get_profile_states(int profile_step, uint64_t* host_output) const {
    // Scratch scan buffers are mutated, while the Markov-chain state remains
    // logically const.
    Impl& x = *impl_;
    if (profile_step <= 0 || static_cast<std::size_t>(profile_step) > x.length)
        throw std::invalid_argument("profile_step must be in [1, operator length]");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for profile states");
    const std::size_t n_points = x.length / static_cast<std::size_t>(profile_step);
    const std::size_t output_size = n_points * static_cast<std::size_t>(x.words);
    if (x.profile_output.size() < output_size)
        x.profile_output = DeviceBuffer<uint64_t>(output_size);

    auto run = [&](auto words_tag) {
        constexpr int Words = decltype(words_tag)::value;
        run_profile_state_scan<Words>(
            x.types.get(), x.sites.get(), x.length, profile_step,
            x.seam_cut, x.seam_words.get(),
            x.tile_parity.get(), x.tile_prefix.get(), x.scan_temp.get(),
            x.scan_temp_bytes, x.profile_output.get());
    };
    switch (x.words) {
        case 1: run(std::integral_constant<int, 1>{}); break;
        case 2: run(std::integral_constant<int, 2>{}); break;
        case 3: run(std::integral_constant<int, 3>{}); break;
        case 4: run(std::integral_constant<int, 4>{}); break;
        case 5: run(std::integral_constant<int, 5>{}); break;
        case 6: run(std::integral_constant<int, 6>{}); break;
    }
    check_cuda(cudaMemcpy(host_output, x.profile_output.get(),
                          output_size * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost),
               "copy profile states to host");
}

void DiagonalEngine::get_operator_string(int32_t* host_types, int32_t* host_sites) const {
    const Impl& x = *impl_;
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for operator download");
    check_cuda(cudaMemcpy(host_types, x.types.get(), x.length * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "copy op_types to host");
    check_cuda(cudaMemcpy(host_sites, x.sites.get(), x.length * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "copy op_sites to host");
}

void DiagonalEngine::set_operator_string(const int32_t* host_types,
                                         const int32_t* host_sites) {
    Impl& x = *impl_;
    validate_operator_string(host_types, host_sites, x.length,
                             x.n_sites, x.n_bonds, "operator string");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for operator upload");
    check_cuda(cudaMemcpy(x.types.get(), host_types, x.length * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy op_types to device");
    check_cuda(cudaMemcpy(x.sites.get(), host_sites, x.length * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy op_sites to device");
    x.last_site_events = 0;
    x.last_bond_events = 0;
    x.events_valid = false;
}

void DiagonalEngine::get_site_events(uint64_t* host_keys, uint32_t* host_values) const {
    const Impl& x = *impl_;
    if (x.site_keys_out.size() == 0)
        throw std::runtime_error("build_events must be called before get_site_events");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for site-event download");
    check_cuda(cudaMemcpy(host_keys, x.site_keys_out.get(),
                          x.last_site_events * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy site-event keys to host");
    check_cuda(cudaMemcpy(host_values, x.site_values_out.get(),
                          x.last_site_events * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "copy site-event values to host");
}

void DiagonalEngine::get_bond_events(uint64_t* host_keys, uint64_t* host_values) const {
    const Impl& x = *impl_;
    if (x.bond_keys_out.size() == 0)
        throw std::runtime_error("build_events must be called before get_bond_events");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for bond-event download");
    check_cuda(cudaMemcpy(host_keys, x.bond_keys_out.get(),
                          x.last_bond_events * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy bond-event keys to host");
    check_cuda(cudaMemcpy(host_values, x.bond_values_out.get(),
                          x.last_bond_events * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy bond-event values to host");
}

void DiagonalEngine::get_bond_spin(int8_t* host_bond_spin) const {
    const Impl& x = *impl_;
    if (x.bond_spin.size() == 0)
        throw std::runtime_error("build_events must be called before get_bond_spin");
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for bond-spin download");
    check_cuda(cudaMemcpy(host_bond_spin, x.bond_spin.get(),
                          x.length * sizeof(int8_t), cudaMemcpyDeviceToHost),
               "copy bond_spin to host");
}

int DiagonalEngine::n_sites() const { return impl_->n_sites; }
int DiagonalEngine::half_length() const { return impl_->half_length; }
std::size_t DiagonalEngine::length() const { return impl_->length; }
int DiagonalEngine::packed_words() const { return impl_->words; }
std::size_t DiagonalEngine::device_bytes() const { return impl_->allocated_bytes(); }
uint64_t DiagonalEngine::seam_mask() const { return impl_->host_seam_mask; }
int DiagonalEngine::string_site_count() const {
    return static_cast<int>(impl_->host_string_sites.size());
}
int DiagonalEngine::seam_cut() const { return impl_->seam_cut; }

}  // namespace qaqmc_cuda
