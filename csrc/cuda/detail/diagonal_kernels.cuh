#pragma once

#include "scan_primitives.cuh"

#include <cub/cub.cuh>

namespace qaqmc_cuda::detail {
namespace {
template <int Words>
__global__ void diagonal_resample_kernel(
    int32_t* types,
    int32_t* sites,
    std::size_t length,
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
    const PackedState<Words>* tile_prefix,
    int seam_cut,
    const uint64_t* seam_words,
    uint64_t seed,
    uint64_t sweep_id,
    DeviceDiagonalStats* stats) {
    using Scan = cub::BlockScan<PackedState<Words>, kBlockSize>;
    using Reduce = cub::BlockReduce<unsigned long long, kBlockSize>;
    union SharedTemp {
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage reduce;
    };
    __shared__ SharedTemp temp;

    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const PackedState<Words> flip = flip_for_slice<Words>(types, sites, p, length);
    PackedState<Words> local_prefix;
    Scan(temp.scan).ExclusiveScan(flip, local_prefix, PackedState<Words>::zero(),
                                  PackedXor<Words>{});
    __syncthreads();  // shared storage is reused for block reductions below.

    const bool should_update = p < length && types[p] != -1;
    PackedState<Words> state = PackedState<Words>::zero();
    const PackedState<Words> base = tile_prefix[blockIdx.x];
#pragma unroll
    for (int w = 0; w < Words; ++w) state.word[w] = base.word[w] ^ local_prefix.word[w];
    apply_seam_if_needed(state, p, seam_cut, seam_words);

    uint64_t attempts = 0;
    uint64_t bond_proposals = 0;
    uint64_t bond_accepts = 0;
    bool inserted = !should_update;
    if (should_update) {
        curandStatePhilox4_32_10_t rng;
        // A 2^32-draw offset per sweep keeps variable-length rejection streams
        // disjoint for every practical proposal count while preserving replay.
        curand_init(static_cast<unsigned long long>(seed),
                    static_cast<unsigned long long>(p),
                    static_cast<unsigned long long>(sweep_id << 32), &rng);

        const int group = static_cast<int>(
            (p * static_cast<std::size_t>(n_groups)) / length);
        const std::size_t alias_base = static_cast<std::size_t>(group) * max_alias;
        const std::size_t rmax_base = static_cast<std::size_t>(group) * n_bonds;
        const double fraction = (p < static_cast<std::size_t>(half_length))
            ? static_cast<double>(p) / static_cast<double>(half_length)
            : static_cast<double>(p - static_cast<std::size_t>(half_length))
                / static_cast<double>(half_length);
        const double delta = (p < static_cast<std::size_t>(half_length))
            ? delta_min + (delta_max - delta_min) * fraction
            : delta_max - (delta_max - delta_min) * fraction;

        constexpr uint64_t kAttemptLimit = uint64_t{1} << 20;
        while (!inserted && attempts < kAttemptLimit) {
            ++attempts;
            const int i = philox_randint(rng, max_alias);
            const int idx = (philox_uniform01(rng) < alias_prob[alias_base + i])
                ? i : alias_index[alias_base + i];
            const int loc_kind = alias_loc_kind[alias_base + idx];
            const int loc = loc_kind >> 1;
            if ((loc_kind & 1) == 0) {
                types[p] = 1;
                sites[p] = loc;
                inserted = true;
                continue;
            }

            ++bond_proposals;
            const int b = loc;
            const int si = bond_sites[2 * b];
            const int sj = bond_sites[2 * b + 1];
            const int ni = static_cast<int>(
                (state.word[si >> 6] >> (si & 63)) & 1ULL);
            const int nj = static_cast<int>(
                (state.word[sj >> 6] >> (sj & 63)) & 1ULL);
            double weights[4];
            compute_bond_weights(delta * inv_coord[si], delta * inv_coord[sj],
                                 bond_vij[b], epsilon, weights);
            const double rmax = bond_rmax[rmax_base + b];
            if (philox_uniform01(rng) < weights[2 * ni + nj] * rmax) {
                types[p] = 2;
                sites[p] = b;
                inserted = true;
                ++bond_accepts;
            }
        }
    }

    auto reduce_counter = [&](unsigned long long local,
                              unsigned long long* global) {
        const unsigned long long total = Reduce(temp.reduce).Sum(local);
        if (threadIdx.x == 0) atomicAdd(global, total);
        __syncthreads();
    };
    reduce_counter(should_update ? 1ULL : 0ULL, &stats->updated_slots);
    reduce_counter(attempts, &stats->proposal_attempts);
    reduce_counter(bond_proposals, &stats->bond_proposals);
    reduce_counter(bond_accepts, &stats->bond_accepts);
    reduce_counter(should_update && !inserted ? 1ULL : 0ULL, &stats->failed_slots);
}

template <int Words>
__global__ void generate_event_streams_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    const int32_t* bond_sites,
    const PackedState<Words>* tile_prefix,
    int seam_cut,
    const uint64_t* seam_words,
    uint64_t* site_keys,
    uint32_t* site_values,
    uint64_t* bond_keys,
    uint64_t* bond_values,
    int8_t* bond_spin,
    DeviceEventCounts* counts) {
    using Scan = cub::BlockScan<PackedState<Words>, kBlockSize>;
    using Reduce = cub::BlockReduce<unsigned long long, kBlockSize>;
    union SharedTemp {
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage reduce;
    };
    __shared__ SharedTemp temp;

    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const PackedState<Words> flip = flip_for_slice<Words>(types, sites, p, length);
    PackedState<Words> local_prefix;
    Scan(temp.scan).ExclusiveScan(flip, local_prefix, PackedState<Words>::zero(),
                                  PackedXor<Words>{});
    __syncthreads();

    constexpr uint64_t invalid = std::numeric_limits<uint64_t>::max();
    unsigned long long is_site_event = 0;
    unsigned long long n_bond_events = 0;
    if (p < length) {
        const int type = types[p];
        if (type == 1 || type == -1) {
            const uint32_t site = static_cast<uint32_t>(sites[p]);
            site_keys[p] = (static_cast<uint64_t>(site) << 32)
                         | static_cast<uint32_t>(p);
            site_values[p] = static_cast<uint32_t>(p);
            is_site_event = 1;
            bond_keys[2 * p] = invalid;
            bond_keys[2 * p + 1] = invalid;
            bond_values[2 * p] = 0;
            bond_values[2 * p + 1] = 0;
        } else {
            site_keys[p] = invalid;
            site_values[p] = 0;
            const int b = sites[p];
            const int si = bond_sites[2 * b];
            const int sj = bond_sites[2 * b + 1];
            const uint64_t packed0 = (static_cast<uint64_t>(p) << 32)
                                   | (static_cast<uint64_t>(b) << 1);
            const uint64_t packed1 = packed0 | 1ULL;
            bond_keys[2 * p] = (static_cast<uint64_t>(si) << 32)
                             | static_cast<uint32_t>(p);
            bond_keys[2 * p + 1] = (static_cast<uint64_t>(sj) << 32)
                                 | static_cast<uint32_t>(p);
            bond_values[2 * p] = packed0;
            bond_values[2 * p + 1] = packed1;
            PackedState<Words> state;
            const PackedState<Words> base = tile_prefix[blockIdx.x];
#pragma unroll
            for (int w = 0; w < Words; ++w)
                state.word[w] = base.word[w] ^ local_prefix.word[w];
            apply_seam_if_needed(state, p, seam_cut, seam_words);
            const int ni = static_cast<int>(
                (state.word[si >> 6] >> (si & 63)) & 1ULL);
            const int nj = static_cast<int>(
                (state.word[sj >> 6] >> (sj & 63)) & 1ULL);
            bond_spin[p] = static_cast<int8_t>(2 * ni + nj);
            n_bond_events = 2;
        }
    }

    const unsigned long long site_total = Reduce(temp.reduce).Sum(is_site_event);
    if (threadIdx.x == 0) atomicAdd(&counts->site_events, site_total);
    __syncthreads();
    const unsigned long long bond_total = Reduce(temp.reduce).Sum(n_bond_events);
    if (threadIdx.x == 0) atomicAdd(&counts->bond_events, bond_total);
}


__global__ void cluster_segments_for_site_kernel(
    int site,
    int op_head,
    int n_sops,
    int bond_head,
    int n_bops,
    const uint32_t* site_values,
    const uint64_t* bond_values,
    const int32_t* bond_sites,
    const double* bond_vij,
    const double* inv_coord,
    int8_t* bond_spin,
    uint8_t* segment_flags,
    int segment_head,
    int half_length,
    double delta_min,
    double delta_max,
    double epsilon,
    uint64_t seed,
    uint64_t sweep_id,
    DeviceClusterStats* stats) {
    // One block owns one segment.  A typical production segment contains
    // dozens of bond vertices; reducing those vertices cooperatively keeps
    // the GPU occupied even though sites must retain the CPU engine's
    // sequential update order.
    const int seg = static_cast<int>(blockIdx.x);
    if (seg > n_sops) return;
    if (seg == 0 || seg == n_sops) {
        if (threadIdx.x == 0) segment_flags[segment_head + seg] = 0;
        return;
    }

    __shared__ int j0;
    __shared__ int j1;
    __shared__ int accepted;
    if (threadIdx.x == 0) {
        const uint32_t p_start = site_values[op_head + seg - 1];
        const uint32_t p_end = site_values[op_head + seg];
        const int bond_end = bond_head + n_bops;
        j0 = upper_bound_bond_p(bond_values, bond_head, bond_end, p_start);
        j1 = upper_bound_bond_p(bond_values, bond_head, bond_end, p_end);
    }
    __syncthreads();

    LogRatioTerm local{0.0, 0};
    for (int j = j0 + static_cast<int>(threadIdx.x); j < j1;
         j += kClusterBlockSize) {
        const uint64_t event = bond_values[j];
        const uint32_t p = static_cast<uint32_t>(event >> 32);
        const int b = static_cast<int>((event >> 1) & 0x7FFFFFFFULL);
        const int endpoint = static_cast<int>(event & 1ULL);
        const int si = bond_sites[2 * b];
        const int sj = bond_sites[2 * b + 1];
        const int old_index = bond_spin[p];
        const int new_index = old_index ^ (endpoint == 0 ? 2 : 1);
        const double fraction = (p < static_cast<uint32_t>(half_length))
            ? static_cast<double>(p) / static_cast<double>(half_length)
            : static_cast<double>(p - static_cast<uint32_t>(half_length))
                / static_cast<double>(half_length);
        const double delta = (p < static_cast<uint32_t>(half_length))
            ? delta_min + (delta_max - delta_min) * fraction
            : delta_max - (delta_max - delta_min) * fraction;
        double weights[4];
        compute_bond_weights(delta * inv_coord[si], delta * inv_coord[sj],
                             bond_vij[b], epsilon, weights);
        const double old_weight = weights[old_index];
        const double new_weight = weights[new_index];
        if (new_weight > 1e-300) {
            if (old_weight > 1e-300) {
                local.value += log(new_weight) - log(old_weight);
            } else {
                ++local.balance;
            }
        } else if (old_weight > 1e-300) {
            --local.balance;
        }
    }

    using Reduce = cub::BlockReduce<LogRatioTerm, kClusterBlockSize>;
    __shared__ typename Reduce::TempStorage reduce_temp;
    const LogRatioTerm total = Reduce(reduce_temp).Reduce(local, LogRatioSum{});
    if (threadIdx.x == 0) {
        bool accept;
        if (total.balance != 0) {
            accept = total.balance > 0;
        } else if (total.value >= 0.0) {
            accept = true;
        } else {
            curandStatePhilox4_32_10_t rng;
            const uint64_t sequence = (static_cast<uint64_t>(site) << 32)
                                    | static_cast<uint32_t>(seg);
            curand_init(static_cast<unsigned long long>(seed),
                        static_cast<unsigned long long>(sequence),
                        static_cast<unsigned long long>(sweep_id << 32), &rng);
            const double u = philox_uniform01(rng);
            accept = u > 0.0 && log(u) < total.value;
        }
        accepted = accept ? 1 : 0;
        segment_flags[segment_head + seg] = static_cast<uint8_t>(accepted);
        atomicAdd(&stats->proposed_segments, 1ULL);
        if (accepted) atomicAdd(&stats->accepted_segments, 1ULL);
    }
    __syncthreads();
    if (accepted) {
        for (int j = j0 + static_cast<int>(threadIdx.x); j < j1;
             j += kClusterBlockSize) {
            const uint64_t event = bond_values[j];
            const uint32_t p = static_cast<uint32_t>(event >> 32);
            const int endpoint = static_cast<int>(event & 1ULL);
            bond_spin[p] ^= static_cast<int8_t>(endpoint == 0 ? 2 : 1);
        }
    }
}

__global__ void apply_site_segment_flips_kernel(
    int op_head,
    int n_sops,
    const uint32_t* site_values,
    const uint8_t* segment_flags,
    int segment_head,
    int32_t* types) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n_sops) return;
    if (segment_flags[segment_head + k] != segment_flags[segment_head + k + 1]) {
        const uint32_t p = site_values[op_head + k];
        types[p] = types[p] == 1 ? -1 : 1;
    }
}


template <int Words>
std::size_t tile_scan_temp_bytes(std::size_t n_tiles) {
    std::size_t bytes = 0;
    const PackedState<Words> initial = PackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   nullptr, bytes,
                   static_cast<const PackedState<Words>*>(nullptr),
                   static_cast<PackedState<Words>*>(nullptr),
                   PackedXor<Words>{}, initial, n_tiles),
               "size persistent CUB tile scan");
    return bytes;
}

template <int Words>
void run_diagonal_scan_and_resample(
    int32_t* d_types,
    int32_t* d_sites,
    std::size_t length,
    int half_length,
    double delta_min,
    double delta_max,
    double epsilon,
    int n_groups,
    int max_alias,
    int n_bonds,
    const int32_t* d_bond_sites,
    const double* d_bond_vij,
    const double* d_inv_coord,
    const double* d_alias_prob,
    const int32_t* d_alias_index,
    const int32_t* d_alias_loc_kind,
    const double* d_bond_rmax,
    int seam_cut,
    const uint64_t* d_seam_words,
    void* d_tile_parity_raw,
    void* d_tile_prefix_raw,
    void* d_scan_temp,
    std::size_t scan_temp_bytes,
    DeviceDiagonalStats* d_stats,
    uint64_t seed,
    uint64_t sweep_id) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    auto* parity = static_cast<PackedState<Words>*>(d_tile_parity_raw);
    auto* prefix = static_cast<PackedState<Words>*>(d_tile_prefix_raw);
    tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types, d_sites, length, parity);
    check_cuda(cudaGetLastError(), "launch persistent tile parity");
    const PackedState<Words> initial = PackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   d_scan_temp, scan_temp_bytes, parity, prefix,
                   PackedXor<Words>{}, initial, n_tiles),
               "run persistent CUB tile scan");
    diagonal_resample_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types, d_sites, length, half_length, delta_min, delta_max, epsilon,
        n_groups, max_alias, n_bonds, d_bond_sites, d_bond_vij, d_inv_coord,
        d_alias_prob, d_alias_index, d_alias_loc_kind, d_bond_rmax, prefix,
        seam_cut, d_seam_words,
        seed, sweep_id, d_stats);
    check_cuda(cudaGetLastError(), "launch diagonal_resample_kernel");
}

template <int Words>
void run_event_scan_and_generation(
    const int32_t* d_types,
    const int32_t* d_sites,
    std::size_t length,
    const int32_t* d_bond_sites,
    int seam_cut,
    const uint64_t* d_seam_words,
    void* d_tile_parity_raw,
    void* d_tile_prefix_raw,
    void* d_scan_temp,
    std::size_t scan_temp_bytes,
    uint64_t* d_site_keys,
    uint32_t* d_site_values,
    uint64_t* d_bond_keys,
    uint64_t* d_bond_values,
    int8_t* d_bond_spin,
    DeviceEventCounts* d_counts) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    auto* parity = static_cast<PackedState<Words>*>(d_tile_parity_raw);
    auto* prefix = static_cast<PackedState<Words>*>(d_tile_prefix_raw);
    tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types, d_sites, length, parity);
    check_cuda(cudaGetLastError(), "launch event tile parity");
    const PackedState<Words> initial = PackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   d_scan_temp, scan_temp_bytes, parity, prefix,
                   PackedXor<Words>{}, initial, n_tiles),
               "run event CUB tile scan");
    generate_event_streams_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types, d_sites, length, d_bond_sites, prefix, seam_cut, d_seam_words,
        d_site_keys, d_site_values, d_bond_keys, d_bond_values,
        d_bond_spin, d_counts);
    check_cuda(cudaGetLastError(), "launch generate_event_streams_kernel");
}

template <int Words>
void run_profile_state_scan(
    const int32_t* d_types,
    const int32_t* d_sites,
    std::size_t length,
    int profile_step,
    int seam_cut,
    const uint64_t* d_seam_words,
    void* d_tile_parity_raw,
    void* d_tile_prefix_raw,
    void* d_scan_temp,
    std::size_t scan_temp_bytes,
    uint64_t* d_output) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    const std::size_t n_points = length / static_cast<std::size_t>(profile_step);
    auto* parity = static_cast<PackedState<Words>*>(d_tile_parity_raw);
    auto* prefix = static_cast<PackedState<Words>*>(d_tile_prefix_raw);
    tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types, d_sites, length, parity);
    check_cuda(cudaGetLastError(), "launch profile tile parity");
    const PackedState<Words> initial = PackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   d_scan_temp, scan_temp_bytes, parity, prefix,
                   PackedXor<Words>{}, initial, n_tiles),
               "run profile CUB tile scan");
    materialise_profile_states_kernel<Words>
        <<<static_cast<unsigned>(n_points), kBlockSize>>>(
            d_types, d_sites, length, profile_step, prefix,
            seam_cut, d_seam_words, d_output);
    check_cuda(cudaGetLastError(), "launch materialise_profile_states_kernel");
}


}  // namespace
}  // namespace qaqmc_cuda::detail
