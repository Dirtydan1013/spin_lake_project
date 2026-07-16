#pragma once

#include "common.cuh"

#include <cub/cub.cuh>

namespace qaqmc_cuda::detail {
namespace {
template <int Words>
struct PackedState {
    uint64_t word[Words];

    __host__ __device__ static PackedState zero() {
        PackedState out{};
#pragma unroll
        for (int w = 0; w < Words; ++w) out.word[w] = 0;
        return out;
    }
};

template <int Words>
struct PackedXor {
    __host__ __device__ PackedState<Words> operator()(
        const PackedState<Words>& a, const PackedState<Words>& b) const {
        PackedState<Words> out;
#pragma unroll
        for (int w = 0; w < Words; ++w) out.word[w] = a.word[w] ^ b.word[w];
        return out;
    }
};

template <int Words>
__device__ void apply_seam_if_needed(PackedState<Words>& state,
                                     std::size_t p,
                                     int seam_cut,
                                     const uint64_t* seam_words) {
    if (seam_cut < 0 || p < static_cast<std::size_t>(seam_cut)) return;
#pragma unroll
    for (int w = 0; w < Words; ++w) state.word[w] ^= seam_words[w];
}

template <int Words>
__device__ PackedState<Words> flip_for_slice(const int32_t* types,
                                             const int32_t* sites,
                                             std::size_t p,
                                             std::size_t length) {
    PackedState<Words> out = PackedState<Words>::zero();
    if (p < length && types[p] == -1) {
        const int site = sites[p];
        out.word[site >> 6] = uint64_t{1} << (site & 63);
    }
    return out;
}

template <int Words>
__global__ void tile_parity_kernel(const int32_t* types,
                                   const int32_t* sites,
                                   std::size_t length,
                                   PackedState<Words>* tile_parity) {
    using Reduce = cub::BlockReduce<PackedState<Words>, kBlockSize>;
    __shared__ typename Reduce::TempStorage temp;

    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const PackedState<Words> local = flip_for_slice<Words>(types, sites, p, length);
    const PackedState<Words> aggregate = Reduce(temp).Reduce(local, PackedXor<Words>{});
    if (threadIdx.x == 0) tile_parity[blockIdx.x] = aggregate;
}

template <int Words>
__global__ void materialise_prefix_kernel(const int32_t* types,
                                          const int32_t* sites,
                                          std::size_t length,
                                          const PackedState<Words>* tile_prefix,
                                          uint64_t* output) {
    using Scan = cub::BlockScan<PackedState<Words>, kBlockSize>;
    __shared__ typename Scan::TempStorage temp;

    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const PackedState<Words> local = flip_for_slice<Words>(types, sites, p, length);
    PackedState<Words> local_prefix;
    Scan(temp).ExclusiveScan(local, local_prefix, PackedState<Words>::zero(),
                             PackedXor<Words>{});

    if (p < length) {
        const PackedState<Words> base = tile_prefix[blockIdx.x];
#pragma unroll
        for (int w = 0; w < Words; ++w) {
            output[p * Words + w] = base.word[w] ^ local_prefix.word[w];
        }
    }
}

template <int Words>
__global__ void materialise_profile_states_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int profile_step,
    const PackedState<Words>* tile_prefix,
    int seam_cut,
    const uint64_t* seam_words,
    uint64_t* output) {
    using Scan = cub::BlockScan<PackedState<Words>, kBlockSize>;
    __shared__ typename Scan::TempStorage temp;

    const std::size_t point = static_cast<std::size_t>(blockIdx.x);
    const std::size_t target = (point + 1) * static_cast<std::size_t>(profile_step) - 1;
    const std::size_t tile = target / kBlockSize;
    const std::size_t tile_begin = tile * kBlockSize;
    const std::size_t p = tile_begin + static_cast<std::size_t>(threadIdx.x);
    const PackedState<Words> local = flip_for_slice<Words>(types, sites, p, length);
    PackedState<Words> inclusive;
    Scan(temp).InclusiveScan(local, inclusive, PackedXor<Words>{});

    if (p == target) {
        PackedState<Words> state = tile_prefix[tile];
#pragma unroll
        for (int w = 0; w < Words; ++w) state.word[w] ^= inclusive.word[w];
        // Profile samples the state after target, so a seam immediately
        // before target+1 must already be visible in the returned state.
        apply_seam_if_needed(state, target + 1, seam_cut, seam_words);
#pragma unroll
        for (int w = 0; w < Words; ++w)
            output[point * Words + w] = state.word[w];
    }
}


template <int Words>
struct DualPackedState {
    uint64_t word[2 * Words];

    __host__ __device__ static DualPackedState zero() {
        DualPackedState out{};
#pragma unroll
        for (int w = 0; w < 2 * Words; ++w) out.word[w] = 0;
        return out;
    }
};

template <int Words>
struct DualPackedXor {
    __host__ __device__ DualPackedState<Words> operator()(
        const DualPackedState<Words>& a,
        const DualPackedState<Words>& b) const {
        DualPackedState<Words> out;
#pragma unroll
        for (int w = 0; w < 2 * Words; ++w) out.word[w] = a.word[w] ^ b.word[w];
        return out;
    }
};

__device__ int renyi_mask_bit(const uint64_t* mask_words, int site) {
    return static_cast<int>((mask_words[site >> 6] >> (site & 63)) & 1ULL);
}

__device__ int renyi_channel(int replica,
                             int site,
                             std::size_t p,
                             int cut,
                             const uint64_t* mask_words) {
    return replica ^ ((p >= static_cast<std::size_t>(cut))
        ? renyi_mask_bit(mask_words, site) : 0);
}

template <int Words>
__device__ DualPackedState<Words> dual_flip_for_slice(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    std::size_t p,
    int cut,
    const uint64_t* mask_words) {
    DualPackedState<Words> out = DualPackedState<Words>::zero();
    if (p >= length) return out;
#pragma unroll
    for (int replica = 0; replica < 2; ++replica) {
        const std::size_t idx = static_cast<std::size_t>(replica) * length + p;
        if (types[idx] != -1) continue;
        const int site = sites[idx];
        const int channel = renyi_channel(replica, site, p, cut, mask_words);
        out.word[channel * Words + (site >> 6)] ^= uint64_t{1} << (site & 63);
    }
    return out;
}

template <int Words>
__global__ void dual_tile_parity_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int cut,
    const uint64_t* mask_words,
    DualPackedState<Words>* tile_parity) {
    using Reduce = cub::BlockReduce<DualPackedState<Words>, kBlockSize>;
    __shared__ typename Reduce::TempStorage temp;
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const auto local = dual_flip_for_slice<Words>(
        types, sites, length, p, cut, mask_words);
    const auto aggregate = Reduce(temp).Reduce(local, DualPackedXor<Words>{});
    if (threadIdx.x == 0) tile_parity[blockIdx.x] = aggregate;
}

template <int Words>
__device__ DualPackedState<Words> actual_flip_for_slice(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    std::size_t p) {
    DualPackedState<Words> out = DualPackedState<Words>::zero();
    if (p >= length) return out;
#pragma unroll
    for (int replica = 0; replica < 2; ++replica) {
        const std::size_t op = static_cast<std::size_t>(replica) * length + p;
        if (types[op] != -1) continue;
        const int site = sites[op];
        out.word[replica * Words + (site >> 6)] ^= uint64_t{1} << (site & 63);
    }
    return out;
}

template <int Words>
__global__ void actual_tile_parity_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    DualPackedState<Words>* tile_parity) {
    using Reduce = cub::BlockReduce<DualPackedState<Words>, kBlockSize>;
    __shared__ typename Reduce::TempStorage temp;
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * kBlockSize
                        + static_cast<std::size_t>(threadIdx.x);
    const auto local = actual_flip_for_slice<Words>(types, sites, length, p);
    const auto aggregate = Reduce(temp).Reduce(local, DualPackedXor<Words>{});
    if (threadIdx.x == 0) tile_parity[blockIdx.x] = aggregate;
}

// Compact states needed by a single-site Renyi topology toggle.  Layout is
// [replica, {cut, terminal}, word].  A valid current and proposed toggle
// implies the two cut occupations at the target site are equal.  Therefore
// changing the channel label leaves every actual-replica occupation (and all
// diagonal bond weights) unchanged; no O(length) path materialisation is
// needed for the topology ratio or accepted update.
template <int Words>
__global__ void materialise_actual_boundaries_kernel(
    const int32_t* types,
    const int32_t* sites,
    std::size_t length,
    int cut,
    const DualPackedState<Words>* tile_parity,
    const DualPackedState<Words>* tile_prefix,
    uint64_t* output) {
    using Scan = cub::BlockScan<DualPackedState<Words>, kBlockSize>;
    __shared__ typename Scan::TempStorage temp;
    const bool interior = cut > 0 && static_cast<std::size_t>(cut) < length;
    const std::size_t tile = interior ? static_cast<std::size_t>(cut) / kBlockSize : 0;
    const std::size_t tile_begin = tile * kBlockSize;
    const std::size_t p = tile_begin + static_cast<std::size_t>(threadIdx.x);
    const auto flip = interior
        ? actual_flip_for_slice<Words>(types, sites, length, p)
        : DualPackedState<Words>::zero();
    DualPackedState<Words> local_prefix;
    Scan(temp).ExclusiveScan(flip, local_prefix,
                             DualPackedState<Words>::zero(),
                             DualPackedXor<Words>{});
    __syncthreads();

    if (threadIdx.x == 0) {
        const std::size_t last = (length + kBlockSize - 1) / kBlockSize - 1;
        for (int replica = 0; replica < 2; ++replica) {
            for (int w = 0; w < Words; ++w) {
                output[(replica * 2 + 1) * Words + w] =
                    tile_prefix[last].word[replica * Words + w]
                    ^ tile_parity[last].word[replica * Words + w];
                if (cut == 0)
                    output[(replica * 2) * Words + w] = 0;
                else if (static_cast<std::size_t>(cut) == length)
                    output[(replica * 2) * Words + w] =
                        output[(replica * 2 + 1) * Words + w];
            }
        }
    }
    if (interior
        && threadIdx.x == static_cast<unsigned>(cut) % kBlockSize) {
        const auto base = tile_prefix[tile];
        for (int replica = 0; replica < 2; ++replica) {
            for (int w = 0; w < Words; ++w) {
                output[(replica * 2) * Words + w] =
                    base.word[replica * Words + w]
                    ^ local_prefix.word[replica * Words + w];
            }
        }
    }
}

template <int Words>
__device__ int actual_boundary_bit(const uint64_t* boundaries,
                                   int replica,
                                   int boundary,
                                   int site) {
    const std::size_t idx = (replica * 2 + boundary) * Words + (site >> 6);
    return static_cast<int>((boundaries[idx] >> (site & 63)) & 1ULL);
}

template <int Words>
__device__ int channel_terminal_from_boundaries(const uint64_t* boundaries,
                                                int channel,
                                                int site,
                                                int mask_bit) {
    const int source = channel ^ mask_bit;
    return actual_boundary_bit<Words>(boundaries, channel, 0, site)
         ^ actual_boundary_bit<Words>(boundaries, source, 1, site)
         ^ actual_boundary_bit<Words>(boundaries, source, 0, site);
}

}  // namespace
}  // namespace qaqmc_cuda::detail
