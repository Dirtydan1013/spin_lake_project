#pragma once

#include "scan_primitives.cuh"

#include <cub/cub.cuh>

namespace qaqmc_cuda::detail {
namespace {
template <int Words>
__global__ void renyi_diagonal_resample_kernel(
    int32_t* types,
    int32_t* sites,
    std::size_t length,
    int cut,
    const uint64_t* mask_words,
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
    const DualPackedState<Words>* tile_prefix,
    uint64_t seed,
    uint64_t sweep_id,
    DeviceDiagonalStats* stats) {
    using Scan = cub::BlockScan<DualPackedState<Words>, kBlockSize>;
    using Reduce = cub::BlockReduce<unsigned long long, kBlockSize>;
    union SharedTemp {
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage reduce;
    };
    __shared__ SharedTemp temp;

    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const auto flip = dual_flip_for_slice<Words>(
        types, sites, length, p, cut, mask_words);
    DualPackedState<Words> local_prefix;
    Scan(temp.scan).ExclusiveScan(flip, local_prefix,
                                  DualPackedState<Words>::zero(),
                                  DualPackedXor<Words>{});
    __syncthreads();

    DualPackedState<Words> state = DualPackedState<Words>::zero();
    const auto base = tile_prefix[blockIdx.x];
#pragma unroll
    for (int w = 0; w < 2 * Words; ++w)
        state.word[w] = base.word[w] ^ local_prefix.word[w];

    unsigned long long updated = 0;
    unsigned long long attempts = 0;
    unsigned long long bond_proposals = 0;
    unsigned long long bond_accepts = 0;
    unsigned long long failed = 0;
    if (p < length) {
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

        for (int replica = 0; replica < 2; ++replica) {
            const std::size_t op = static_cast<std::size_t>(replica) * length + p;
            if (types[op] == -1) continue;
            ++updated;
            curandStatePhilox4_32_10_t rng;
            curand_init(static_cast<unsigned long long>(seed),
                        static_cast<unsigned long long>(2 * p + replica),
                        static_cast<unsigned long long>(sweep_id << 32), &rng);
            bool inserted = false;
            uint64_t local_attempts = 0;
            while (!inserted && local_attempts < kAttemptLimit) {
                ++local_attempts;
                ++attempts;
                const int i = philox_randint(rng, max_alias);
                const int alias_slot =
                    (philox_uniform01(rng) < alias_prob[alias_base + i])
                    ? i : alias_index[alias_base + i];
                const int loc_kind = alias_loc_kind[alias_base + alias_slot];
                const int loc = loc_kind >> 1;
                if ((loc_kind & 1) == 0) {
                    types[op] = 1;
                    sites[op] = loc;
                    inserted = true;
                    continue;
                }
                ++bond_proposals;
                const int b = loc;
                const int si = bond_sites[2 * b];
                const int sj = bond_sites[2 * b + 1];
                const int ci = renyi_channel(replica, si, p, cut, mask_words);
                const int cj = renyi_channel(replica, sj, p, cut, mask_words);
                const int ni = static_cast<int>(
                    (state.word[ci * Words + (si >> 6)] >> (si & 63)) & 1ULL);
                const int nj = static_cast<int>(
                    (state.word[cj * Words + (sj >> 6)] >> (sj & 63)) & 1ULL);
                double weights[4];
                compute_bond_weights(delta * inv_coord[si], delta * inv_coord[sj],
                                     bond_vij[b], epsilon, weights);
                if (philox_uniform01(rng) <
                    weights[2 * ni + nj] * bond_rmax[rmax_base + b]) {
                    types[op] = 2;
                    sites[op] = b;
                    inserted = true;
                    ++bond_accepts;
                }
            }
            if (!inserted) ++failed;
        }
    }

    auto reduce_counter = [&](unsigned long long local,
                              unsigned long long* global) {
        const unsigned long long total = Reduce(temp.reduce).Sum(local);
        if (threadIdx.x == 0) atomicAdd(global, total);
        __syncthreads();
    };
    reduce_counter(updated, &stats->updated_slots);
    reduce_counter(attempts, &stats->proposal_attempts);
    reduce_counter(bond_proposals, &stats->bond_proposals);
    reduce_counter(bond_accepts, &stats->bond_accepts);
    reduce_counter(failed, &stats->failed_slots);
}


template <int Words>
__global__ void renyi_generate_event_streams_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int cut,
    const uint64_t* mask_words,
    int n_sites,
    const int32_t* bond_sites,
    const DualPackedState<Words>* tile_prefix,
    uint64_t* site_keys,
    uint32_t* site_values,
    uint64_t* bond_keys,
    uint32_t* bond_values,
    int8_t* bond_spin,
    DeviceEventCounts* counts) {
    using Scan = cub::BlockScan<DualPackedState<Words>, kBlockSize>;
    using Reduce = cub::BlockReduce<unsigned long long, kBlockSize>;
    union SharedTemp {
        typename Scan::TempStorage scan;
        typename Reduce::TempStorage reduce;
    };
    __shared__ SharedTemp temp;
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const auto flip = dual_flip_for_slice<Words>(
        types, sites, length, p, cut, mask_words);
    DualPackedState<Words> local_prefix;
    Scan(temp.scan).ExclusiveScan(flip, local_prefix,
                                  DualPackedState<Words>::zero(),
                                  DualPackedXor<Words>{});
    __syncthreads();
    constexpr uint64_t invalid = std::numeric_limits<uint64_t>::max();
    unsigned long long site_count = 0;
    unsigned long long bond_count = 0;
    if (p < length) {
        DualPackedState<Words> state = tile_prefix[blockIdx.x];
#pragma unroll
        for (int w = 0; w < 2 * Words; ++w) state.word[w] ^= local_prefix.word[w];
        for (int replica = 0; replica < 2; ++replica) {
            const std::size_t op = static_cast<std::size_t>(replica) * length + p;
            const std::size_t site_slot = 2 * p + replica;
            const std::size_t bond_slot = 4 * p + 2 * replica;
            const int type = types[op];
            if (type == 1 || type == -1) {
                const int site = sites[op];
                const int channel = renyi_channel(replica, site, p, cut, mask_words);
                const uint32_t group = static_cast<uint32_t>(channel * n_sites + site);
                site_keys[site_slot] = (static_cast<uint64_t>(group) << 32)
                                           | static_cast<uint32_t>(p);
                site_values[site_slot] = (static_cast<uint32_t>(p) << 1)
                                       | static_cast<uint32_t>(replica);
                bond_keys[bond_slot] = invalid;
                bond_keys[bond_slot + 1] = invalid;
                bond_values[bond_slot] = 0;
                bond_values[bond_slot + 1] = 0;
                ++site_count;
            } else {
                site_keys[site_slot] = invalid;
                site_values[site_slot] = 0;
                const int b = sites[op];
                const int si = bond_sites[2 * b];
                const int sj = bond_sites[2 * b + 1];
                const int ci = renyi_channel(replica, si, p, cut, mask_words);
                const int cj = renyi_channel(replica, sj, p, cut, mask_words);
                const uint32_t gi = static_cast<uint32_t>(ci * n_sites + si);
                const uint32_t gj = static_cast<uint32_t>(cj * n_sites + sj);
                bond_keys[bond_slot] = (static_cast<uint64_t>(gi) << 32)
                                           | static_cast<uint32_t>(p);
                bond_keys[bond_slot + 1] = (static_cast<uint64_t>(gj) << 32)
                                               | static_cast<uint32_t>(p);
                const uint32_t packed = (static_cast<uint32_t>(p) << 2)
                                      | (static_cast<uint32_t>(replica) << 1);
                bond_values[bond_slot] = packed;
                bond_values[bond_slot + 1] = packed | 1u;
                const int ni = static_cast<int>(
                    (state.word[ci * Words + (si >> 6)] >> (si & 63)) & 1ULL);
                const int nj = static_cast<int>(
                    (state.word[cj * Words + (sj >> 6)] >> (sj & 63)) & 1ULL);
                bond_spin[static_cast<std::size_t>(replica) * length + p] =
                    static_cast<int8_t>(2 * ni + nj);
                bond_count += 2;
            }
        }
    } else {
        // Sort capacities are full 2L/4L arrays.  The final partial tile has
        // no corresponding p slots, so nothing needs to be initialised here.
    }
    const auto site_total = Reduce(temp.reduce).Sum(site_count);
    if (threadIdx.x == 0) atomicAdd(&counts->site_events, site_total);
    __syncthreads();
    const auto bond_total = Reduce(temp.reduce).Sum(bond_count);
    if (threadIdx.x == 0) atomicAdd(&counts->bond_events, bond_total);
}


__device__ int renyi_upper_bound_bond_p(const uint32_t* values,
                                        int begin,
                                        int end,
                                        uint32_t p) {
    int lo = begin, hi = end;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if ((values[mid] >> 2) <= p) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void renyi_cluster_segments_kernel(
    int channel_site,
    int op_head,
    int n_sops,
    int bond_head,
    int n_bops,
    const uint32_t* site_values,
    const uint32_t* bond_values,
    int32_t* types,
    const int32_t* sites,
    std::size_t length,
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
        const uint32_t p_start = site_values[op_head + seg - 1] >> 1;
        const uint32_t p_end = site_values[op_head + seg] >> 1;
        const int bond_end = bond_head + n_bops;
        j0 = renyi_upper_bound_bond_p(bond_values, bond_head, bond_end, p_start);
        j1 = renyi_upper_bound_bond_p(bond_values, bond_head, bond_end, p_end);
    }
    __syncthreads();

    LogRatioTerm local{0.0, 0};
    for (int j = j0 + static_cast<int>(threadIdx.x); j < j1;
         j += kClusterBlockSize) {
        const uint32_t event = bond_values[j];
        const uint32_t p = event >> 2;
        const int replica = static_cast<int>((event >> 1) & 1u);
        const int endpoint = static_cast<int>(event & 1u);
        const std::size_t op = static_cast<std::size_t>(replica) * length + p;
        const int b = sites[op];
        const int si = bond_sites[2 * b];
        const int sj = bond_sites[2 * b + 1];
        const int old_index = bond_spin[op];
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
            if (old_weight > 1e-300)
                local.value += log(new_weight) - log(old_weight);
            else
                ++local.balance;
        } else if (old_weight > 1e-300) {
            --local.balance;
        }
    }
    using Reduce = cub::BlockReduce<LogRatioTerm, kClusterBlockSize>;
    __shared__ typename Reduce::TempStorage reduce_temp;
    const auto total = Reduce(reduce_temp).Reduce(local, LogRatioSum{});
    if (threadIdx.x == 0) {
        bool accept;
        if (total.balance != 0) {
            accept = total.balance > 0;
        } else if (total.value >= 0.0) {
            accept = true;
        } else {
            curandStatePhilox4_32_10_t rng;
            const uint64_t sequence = (static_cast<uint64_t>(channel_site) << 32)
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
            const uint32_t event = bond_values[j];
            const uint32_t p = event >> 2;
            const int replica = static_cast<int>((event >> 1) & 1u);
            const int endpoint = static_cast<int>(event & 1u);
            const std::size_t op = static_cast<std::size_t>(replica) * length + p;
            bond_spin[op] ^= static_cast<int8_t>(endpoint == 0 ? 2 : 1);
        }
    }
}

__global__ void renyi_apply_site_segment_flips_kernel(
    int op_head,
    int n_sops,
    const uint32_t* site_values,
    const uint8_t* segment_flags,
    int segment_head,
    std::size_t length,
    int32_t* types) {
    const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= n_sops) return;
    if (segment_flags[segment_head + k] == segment_flags[segment_head + k + 1])
        return;
    const uint32_t event = site_values[op_head + k];
    const uint32_t p = event >> 1;
    const int replica = static_cast<int>(event & 1u);
    const std::size_t op = static_cast<std::size_t>(replica) * length + p;
    types[op] = types[op] == 1 ? -1 : 1;
}


template <int Words>
std::size_t dual_tile_scan_temp_bytes(std::size_t n_tiles) {
    std::size_t bytes = 0;
    const auto initial = DualPackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   nullptr, bytes,
                   static_cast<const DualPackedState<Words>*>(nullptr),
                   static_cast<DualPackedState<Words>*>(nullptr),
                   DualPackedXor<Words>{}, initial, n_tiles),
               "size Renyi CUB tile scan");
    return bytes;
}

template <int Words>
void run_renyi_diagonal_scan(
    int32_t* types,
    int32_t* sites,
    std::size_t length,
    int cut,
    const uint64_t* mask_words,
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
    void* tile_parity_raw,
    void* tile_prefix_raw,
    void* scan_temp,
    std::size_t scan_temp_bytes,
    DeviceDiagonalStats* stats,
    uint64_t seed,
    uint64_t sweep_id) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    auto* parity = static_cast<DualPackedState<Words>*>(tile_parity_raw);
    auto* prefix = static_cast<DualPackedState<Words>*>(tile_prefix_raw);
    dual_tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        types, sites, length, cut, mask_words, parity);
    check_cuda(cudaGetLastError(), "launch Renyi tile parity");
    const auto initial = DualPackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   scan_temp, scan_temp_bytes, parity, prefix,
                   DualPackedXor<Words>{}, initial, n_tiles),
               "run Renyi CUB tile scan");
    renyi_diagonal_resample_kernel<Words>
        <<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
            types, sites, length, cut, mask_words, half_length,
            delta_min, delta_max, epsilon, n_groups, max_alias, n_bonds,
            bond_sites, bond_vij, inv_coord, alias_prob, alias_index,
            alias_loc_kind, bond_rmax, prefix, seed, sweep_id, stats);
    check_cuda(cudaGetLastError(), "launch Renyi diagonal resample");
}

template <int Words>
void run_renyi_event_scan(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int cut,
    const uint64_t* mask_words,
    int n_sites,
    const int32_t* bond_sites,
    void* tile_parity_raw,
    void* tile_prefix_raw,
    void* scan_temp,
    std::size_t scan_temp_bytes,
    uint64_t* site_keys,
    uint32_t* site_values,
    uint64_t* bond_keys,
    uint32_t* bond_values,
    int8_t* bond_spin,
    DeviceEventCounts* counts) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    auto* parity = static_cast<DualPackedState<Words>*>(tile_parity_raw);
    auto* prefix = static_cast<DualPackedState<Words>*>(tile_prefix_raw);
    dual_tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        types, sites, length, cut, mask_words, parity);
    check_cuda(cudaGetLastError(), "launch Renyi event tile parity");
    const auto initial = DualPackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   scan_temp, scan_temp_bytes, parity, prefix,
                   DualPackedXor<Words>{}, initial, n_tiles),
               "run Renyi event CUB tile scan");
    renyi_generate_event_streams_kernel<Words>
        <<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
            types, sites, length, cut, mask_words, n_sites, bond_sites, prefix,
            site_keys, site_values, bond_keys, bond_values, bond_spin, counts);
    check_cuda(cudaGetLastError(), "launch Renyi event generation");
}

template <int Words>
void run_actual_boundary_materialisation(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int cut,
    void* tile_parity_raw,
    void* tile_prefix_raw,
    void* scan_temp,
    std::size_t scan_temp_bytes,
    uint64_t* output) {
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;
    auto* parity = static_cast<DualPackedState<Words>*>(tile_parity_raw);
    auto* prefix = static_cast<DualPackedState<Words>*>(tile_prefix_raw);
    actual_tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        types, sites, length, parity);
    check_cuda(cudaGetLastError(), "launch compact actual-replica tile parity");
    const auto initial = DualPackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   scan_temp, scan_temp_bytes, parity, prefix,
                   DualPackedXor<Words>{}, initial, n_tiles),
               "run compact actual-replica CUB tile scan");
    materialise_actual_boundaries_kernel<Words><<<1, kBlockSize>>>(
        types, sites, length, cut, parity, prefix, output);
    check_cuda(cudaGetLastError(), "materialise actual-replica boundaries");
}

}  // namespace
}  // namespace qaqmc_cuda::detail
