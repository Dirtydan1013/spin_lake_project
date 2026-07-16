#pragma once

#include "scan_primitives.cuh"

namespace qaqmc_cuda::detail {
namespace {
template <int Words>
__global__ void renyi_compact_topology_ratio_kernel(
    const uint64_t* mask_words,
    const uint64_t* actual_boundaries,
    int target_site,
    DeviceTopologyRatio* result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int old_mask_bit = renyi_mask_bit(mask_words, target_site);
    int current_valid = 1;
    int proposed_valid = 1;
    for (int channel = 0; channel < 2; ++channel) {
        if (channel_terminal_from_boundaries<Words>(
                actual_boundaries, channel, target_site, old_mask_bit) != 0)
            current_valid = 0;
        if (channel_terminal_from_boundaries<Words>(
                actual_boundaries, channel, target_site, old_mask_bit ^ 1) != 0)
            proposed_valid = 0;
    }
    result->current_valid = current_valid;
    result->proposed_valid = proposed_valid;
    result->log_ratio = (current_valid && proposed_valid) ? 0.0 : -1e30;
}

template <int Words>
__global__ void renyi_compact_topology_sweep_kernel(
    uint64_t* mask_words,
    const uint64_t* actual_boundaries,
    const int32_t* topology_sites,
    int topology_count,
    double lambda,
    uint64_t seed,
    uint64_t sweep_id,
    DeviceTopologyStats* stats) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int order[384];
    curandStatePhilox4_32_10_t rng;
    curand_init(static_cast<unsigned long long>(seed),
                0x52454E5949544F50ULL,
                static_cast<unsigned long long>(sweep_id << 32), &rng);
    for (int k = 0; k < topology_count; ++k) order[k] = k;
    for (int k = topology_count - 1; k > 0; --k) {
        const int q = philox_randint(rng, k + 1);
        const int tmp = order[k];
        order[k] = order[q];
        order[q] = tmp;
    }
    const double log_odds = log(lambda) - log1p(-lambda);
    for (int proposal = 0; proposal < topology_count; ++proposal) {
        const int target_site = topology_sites[order[proposal]];
        const int old_mask_bit = renyi_mask_bit(mask_words, target_site);
        bool valid = true;
        for (int channel = 0; channel < 2; ++channel) {
            if (channel_terminal_from_boundaries<Words>(
                    actual_boundaries, channel, target_site,
                    old_mask_bit) != 0
                || channel_terminal_from_boundaries<Words>(
                    actual_boundaries, channel, target_site,
                    old_mask_bit ^ 1) != 0) {
                valid = false;
            }
        }
        ++stats->attempts;
        if (!valid) {
            ++stats->invalid;
            continue;
        }
        const double log_accept = old_mask_bit ? -log_odds : log_odds;
        const double u = philox_uniform01(rng);
        if (log_accept < 0.0 && !(u > 0.0 && log(u) < log_accept)) continue;
        mask_words[target_site >> 6] ^= uint64_t{1} << (target_site & 63);
        ++stats->accepts;
    }
    unsigned long long active = 0;
    for (int k = 0; k < topology_count; ++k) {
        const int site = topology_sites[k];
        active += static_cast<unsigned long long>(
            (mask_words[site >> 6] >> (site & 63)) & 1ULL);
    }
    stats->active_count = active;
}

}  // namespace
}  // namespace qaqmc_cuda::detail
