#pragma once

#include "qaqmc_cuda_common.cuh"

#include <cub/cub.cuh>
#include <climits>

namespace qaqmc_cuda::detail {
namespace {
struct DeviceRepairState {
    unsigned int parity;
    int first_pm;
    int first_steal;
    int failed;
};

__global__ void reset_repair_state_kernel(DeviceRepairState* state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    state->parity = 0;
    state->first_pm = INT_MAX;
    state->first_steal = INT_MAX;
    state->failed = 0;
}

__global__ void scan_closure_repair_kernel(const int32_t* types,
                                           const int32_t* sites,
                                           std::size_t length,
                                           int target_site,
                                           DeviceRepairState* state) {
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + static_cast<std::size_t>(threadIdx.x);
    if (p >= length) return;
    const int type = types[p];
    const int site = sites[p];
    if (type == -1 && site == target_site) atomicXor(&state->parity, 1u);
    if ((type == 1 || type == -1) && site == target_site)
        atomicMin(&state->first_pm, static_cast<int>(p));
    if (type == 2 || (type == 1 && site != target_site))
        atomicMin(&state->first_steal, static_cast<int>(p));
}

__global__ void commit_closure_repair_kernel(int32_t* types,
                                             int32_t* sites,
                                             int target_site,
                                             int wanted_parity,
                                             DeviceRepairState* state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (static_cast<int>(state->parity & 1u) == wanted_parity) return;
    if (state->first_pm != INT_MAX) {
        const int p = state->first_pm;
        types[p] = types[p] == 1 ? -1 : 1;
        return;
    }
    if (state->first_steal != INT_MAX) {
        const int p = state->first_steal;
        types[p] = -1;
        sites[p] = target_site;
        return;
    }
    state->failed = 1;
}

__global__ void offdiagonal_topology_sweep_kernel(
    int32_t* types,
    std::size_t length,
    int half_length,
    double delta_min,
    double delta_max,
    double epsilon,
    const int32_t* bond_sites,
    const double* bond_vij,
    const double* inv_coord,
    const uint32_t* site_values,
    const uint64_t* bond_values,
    const int32_t* site_heads,
    const int32_t* site_counts,
    const int32_t* bond_heads,
    const int32_t* bond_counts,
    int8_t* bond_spin,
    const int32_t* string_sites,
    int string_count,
    int seam_cut,
    uint64_t* seam_mask,
    uint64_t* seam_words,
    double lambda,
    uint64_t seed,
    uint64_t sweep_id,
    DeviceTopologyStats* stats) {
    if (blockIdx.x != 0) return;
    __shared__ int order[64];
    __shared__ int target_site;
    __shared__ int local_index;
    __shared__ int terminal_p;
    __shared__ int j0;
    __shared__ int j1;
    __shared__ int proposal_valid;
    __shared__ int accepted;
    __shared__ double log_topology_ratio;

    curandStatePhilox4_32_10_t rng;
    if (threadIdx.x == 0) {
        curand_init(static_cast<unsigned long long>(seed),
                    0x535452494E47554CULL,
                    static_cast<unsigned long long>(sweep_id << 32), &rng);
        for (int k = 0; k < string_count; ++k) order[k] = k;
        for (int k = string_count - 1; k > 0; --k) {
            const int q = philox_randint(rng, k + 1);
            const int tmp = order[k];
            order[k] = order[q];
            order[q] = tmp;
        }
    }
    __syncthreads();

    using Reduce = cub::BlockReduce<LogRatioTerm, kClusterBlockSize>;
    __shared__ typename Reduce::TempStorage reduce_temp;
    const double log_odds = log(lambda) - log1p(-lambda);

    for (int proposal = 0; proposal < string_count; ++proposal) {
        if (threadIdx.x == 0) {
            local_index = order[proposal];
            target_site = string_sites[local_index];
            const bool direction_right = philox_uniform01(rng) < 0.5;
            const int sh = site_heads[target_site];
            const int se = sh + site_counts[target_site];
            const int split = lower_bound_site_p(site_values, sh, se,
                                                 static_cast<uint32_t>(seam_cut));
            terminal_p = direction_right
                ? (split < se ? static_cast<int>(site_values[split]) : -1)
                : (split > sh ? static_cast<int>(site_values[split - 1]) : -1);
            proposal_valid = terminal_p >= 0;
            accepted = 0;
            if (proposal_valid) {
                const int bh = bond_heads[target_site];
                const int be = bh + bond_counts[target_site];
                if (direction_right) {
                    j0 = lower_bound_bond_p(bond_values, bh, be,
                                            static_cast<uint32_t>(seam_cut));
                    j1 = lower_bound_bond_p(bond_values, bh, be,
                                            static_cast<uint32_t>(terminal_p));
                } else {
                    j0 = lower_bound_bond_p(bond_values, bh, be,
                                            static_cast<uint32_t>(terminal_p + 1));
                    j1 = lower_bound_bond_p(bond_values, bh, be,
                                            static_cast<uint32_t>(seam_cut));
                }
                const bool active = ((*seam_mask >> local_index) & 1ULL) != 0;
                log_topology_ratio = active ? -log_odds : log_odds;
            }
            ++stats->attempts;
            if (!proposal_valid) ++stats->invalid;
        }
        __syncthreads();

        LogRatioTerm local{0.0, 0};
        if (proposal_valid) {
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
                if (new_weight > 0.0 && old_weight > 0.0)
                    local.value += log(new_weight) - log(old_weight);
                else
                    ++local.balance;
            }
        }

        const LogRatioTerm total = Reduce(reduce_temp).Reduce(local, LogRatioSum{});
        if (threadIdx.x == 0 && proposal_valid) {
            bool accept;
            if (total.balance != 0) {
                // The CPU half-line move rejects the entire proposal when
                // any touching bond has a non-positive old or new weight.
                proposal_valid = 0;
                ++stats->invalid;
                accept = false;
            } else {
                const double log_accept = total.value + log_topology_ratio;
                const double u = philox_uniform01(rng);
                accept = log_accept >= 0.0 || (u > 0.0 && log(u) < log_accept);
            }
            accepted = accept ? 1 : 0;
            if (accepted) {
                types[terminal_p] = types[terminal_p] == 1 ? -1 : 1;
                *seam_mask ^= uint64_t{1} << local_index;
                seam_words[target_site >> 6] ^= uint64_t{1} << (target_site & 63);
                ++stats->accepts;
            }
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
        __syncthreads();
    }
    if (threadIdx.x == 0)
        stats->active_count = static_cast<unsigned long long>(__popcll(*seam_mask));
}

__global__ void offdiagonal_half_line_proposal_kernel(
    int half_length,
    double delta_min,
    double delta_max,
    double epsilon,
    const int32_t* bond_sites,
    const double* bond_vij,
    const double* inv_coord,
    const uint32_t* site_values,
    const uint64_t* bond_values,
    const int32_t* site_heads,
    const int32_t* site_counts,
    const int32_t* bond_heads,
    const int32_t* bond_counts,
    const int8_t* bond_spin,
    int target_site,
    int seam_cut,
    bool direction_right,
    DeviceHalfLineProposal* result) {
    if (blockIdx.x != 0) return;
    __shared__ int terminal_p;
    __shared__ int j0;
    __shared__ int j1;
    __shared__ int proposal_valid;
    if (threadIdx.x == 0) {
        const int sh = site_heads[target_site];
        const int se = sh + site_counts[target_site];
        const int split = lower_bound_site_p(
            site_values, sh, se, static_cast<uint32_t>(seam_cut));
        terminal_p = direction_right
            ? (split < se ? static_cast<int>(site_values[split]) : -1)
            : (split > sh ? static_cast<int>(site_values[split - 1]) : -1);
        proposal_valid = terminal_p >= 0;
        j0 = j1 = 0;
        if (proposal_valid) {
            const int bh = bond_heads[target_site];
            const int be = bh + bond_counts[target_site];
            if (direction_right) {
                j0 = lower_bound_bond_p(
                    bond_values, bh, be, static_cast<uint32_t>(seam_cut));
                j1 = lower_bound_bond_p(
                    bond_values, bh, be, static_cast<uint32_t>(terminal_p));
            } else {
                j0 = lower_bound_bond_p(
                    bond_values, bh, be, static_cast<uint32_t>(terminal_p + 1));
                j1 = lower_bound_bond_p(
                    bond_values, bh, be, static_cast<uint32_t>(seam_cut));
            }
        }
    }
    __syncthreads();

    LogRatioTerm local{0.0, 0};
    if (proposal_valid) {
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
            if (old_weight > 0.0 && new_weight > 0.0)
                local.value += log(new_weight) - log(old_weight);
            else
                ++local.balance;
        }
    }
    using Reduce = cub::BlockReduce<LogRatioTerm, kClusterBlockSize>;
    __shared__ typename Reduce::TempStorage reduce_temp;
    const auto total = Reduce(reduce_temp).Reduce(local, LogRatioSum{});
    if (threadIdx.x == 0) {
        result->terminal_p = terminal_p;
        result->valid = proposal_valid && total.balance == 0;
        result->log_physical_ratio = result->valid ? total.value : 0.0;
    }
}

}  // namespace
}  // namespace qaqmc_cuda::detail
