#include "qaqmc_cuda_scan.cuh"

#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace qaqmc_cuda {
namespace {

constexpr int kBlockSize = 256;
constexpr int kClusterBlockSize = 128;
constexpr int kMaxWords = 6;  // N <= 384 in the first backend milestone.

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream oss;
    oss << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(oss.str());
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)),
                       "cudaMalloc");
        }
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
};

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
        const PackedState<Words> base = tile_prefix[tile];
#pragma unroll
        for (int w = 0; w < Words; ++w)
            output[point * Words + w] = base.word[w] ^ inclusive.word[w];
    }
}

struct DeviceDiagonalStats {
    unsigned long long updated_slots;
    unsigned long long proposal_attempts;
    unsigned long long bond_proposals;
    unsigned long long bond_accepts;
    unsigned long long failed_slots;
};

struct DeviceEventCounts {
    unsigned long long site_events;
    unsigned long long bond_events;
};

struct DeviceClusterStats {
    unsigned long long proposed_segments;
    unsigned long long accepted_segments;
};

__device__ uint64_t philox_u64(curandStatePhilox4_32_10_t& rng) {
    const uint64_t hi = static_cast<uint64_t>(curand(&rng));
    const uint64_t lo = static_cast<uint64_t>(curand(&rng));
    return (hi << 32) | lo;
}

__device__ double philox_uniform01(curandStatePhilox4_32_10_t& rng) {
    return static_cast<double>(philox_u64(rng) >> 11) * 0x1.0p-53;
}

__device__ int philox_randint(curandStatePhilox4_32_10_t& rng, int n) {
    const uint64_t bound = static_cast<uint64_t>(n);
    uint64_t x = philox_u64(rng);
    uint64_t lo = x * bound;
    uint64_t hi = __umul64hi(x, bound);
    if (lo < bound) {
        const uint64_t threshold = (uint64_t{0} - bound) % bound;
        while (lo < threshold) {
            x = philox_u64(rng);
            lo = x * bound;
            hi = __umul64hi(x, bound);
        }
    }
    return static_cast<int>(hi);
}

__device__ void compute_bond_weights(double delta_i,
                                     double delta_j,
                                     double vij,
                                     double epsilon,
                                     double* weights) {
    const double raw0 = 0.0;
    const double raw1 = delta_j;
    const double raw2 = delta_i;
    const double raw3 = -vij + delta_i + delta_j;
    const double m_min = fmin(fmin(raw0, raw1), fmin(raw2, raw3));
    const double m_abs = fmin(fabs(raw1), fmin(fabs(raw2), fabs(raw3)));
    const double cij = (m_min < 0.0 ? -m_min : 0.0) + epsilon * m_abs;
    weights[0] = raw0 + cij;
    weights[1] = raw1 + cij;
    weights[2] = raw2 + cij;
    weights[3] = raw3 + cij;
}

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

__device__ std::size_t lower_bound_event_key(const uint64_t* keys,
                                             std::size_t count,
                                             uint64_t target) {
    std::size_t lo = 0, hi = count;
    while (lo < hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void event_bounds_kernel(const uint64_t* site_keys,
                                    std::size_t n_site_events,
                                    const uint64_t* bond_keys,
                                    std::size_t n_bond_events,
                                    int n_sites,
                                    int32_t* site_heads,
                                    int32_t* site_counts,
                                    int32_t* bond_heads,
                                    int32_t* bond_counts) {
    const int site = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (site >= n_sites) return;
    const uint64_t begin_key = static_cast<uint64_t>(site) << 32;
    const uint64_t end_key = static_cast<uint64_t>(site + 1) << 32;
    const std::size_t sb = lower_bound_event_key(site_keys, n_site_events, begin_key);
    const std::size_t se = lower_bound_event_key(site_keys, n_site_events, end_key);
    const std::size_t bb = lower_bound_event_key(bond_keys, n_bond_events, begin_key);
    const std::size_t be = lower_bound_event_key(bond_keys, n_bond_events, end_key);
    site_heads[site] = static_cast<int32_t>(sb);
    site_counts[site] = static_cast<int32_t>(se - sb);
    bond_heads[site] = static_cast<int32_t>(bb);
    bond_counts[site] = static_cast<int32_t>(be - bb);
}

__device__ int upper_bound_bond_p(const uint64_t* values,
                                  int begin,
                                  int end,
                                  uint32_t p) {
    int lo = begin, hi = end;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        const uint32_t event_p = static_cast<uint32_t>(values[mid] >> 32);
        if (event_p <= p) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

struct LogRatioTerm {
    double value;
    int balance;
};

struct LogRatioSum {
    __device__ LogRatioTerm operator()(const LogRatioTerm& a,
                                       const LogRatioTerm& b) const {
        return LogRatioTerm{a.value + b.value, a.balance + b.balance};
    }
};

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
void prefix_xor_impl(const int32_t* host_types,
                     const int32_t* host_sites,
                     std::size_t length,
                     uint64_t* host_output) {
    if (length == 0) return;
    const std::size_t n_tiles = (length + kBlockSize - 1) / kBlockSize;

    DeviceBuffer<int32_t> d_types(length);
    DeviceBuffer<int32_t> d_sites(length);
    DeviceBuffer<PackedState<Words>> d_tile_parity(n_tiles);
    DeviceBuffer<PackedState<Words>> d_tile_prefix(n_tiles);
    DeviceBuffer<uint64_t> d_output(length * Words);

    check_cuda(cudaMemcpy(d_types.get(), host_types, length * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "copy op_types to device");
    check_cuda(cudaMemcpy(d_sites.get(), host_sites, length * sizeof(int32_t),
                          cudaMemcpyHostToDevice),
               "copy op_sites to device");

    tile_parity_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types.get(), d_sites.get(), length, d_tile_parity.get());
    check_cuda(cudaGetLastError(), "launch tile_parity_kernel");

    std::size_t temp_bytes = 0;
    const PackedState<Words> initial = PackedState<Words>::zero();
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   nullptr, temp_bytes, d_tile_parity.get(), d_tile_prefix.get(),
                   PackedXor<Words>{}, initial, n_tiles),
               "size CUB tile scan");
    DeviceBuffer<uint8_t> d_temp(temp_bytes);
    check_cuda(cub::DeviceScan::ExclusiveScan(
                   d_temp.get(), temp_bytes, d_tile_parity.get(), d_tile_prefix.get(),
                   PackedXor<Words>{}, initial, n_tiles),
               "run CUB tile scan");

    materialise_prefix_kernel<Words><<<static_cast<unsigned>(n_tiles), kBlockSize>>>(
        d_types.get(), d_sites.get(), length, d_tile_prefix.get(), d_output.get());
    check_cuda(cudaGetLastError(), "launch materialise_prefix_kernel");
    check_cuda(cudaMemcpy(host_output, d_output.get(),
                          length * Words * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "copy prefix states to host");
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
        seed, sweep_id, d_stats);
    check_cuda(cudaGetLastError(), "launch diagonal_resample_kernel");
}

template <int Words>
void run_event_scan_and_generation(
    const int32_t* d_types,
    const int32_t* d_sites,
    std::size_t length,
    const int32_t* d_bond_sites,
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
        d_types, d_sites, length, d_bond_sites, prefix,
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
            d_types, d_sites, length, profile_step, prefix, d_output);
    check_cuda(cudaGetLastError(), "launch materialise_profile_states_kernel");
}

}  // namespace

struct DiagonalEngine::Impl {
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

    std::size_t allocated_bytes() const {
        return types.size() * sizeof(int32_t)
             + sites.size() * sizeof(int32_t)
             + bond_sites.size() * sizeof(int32_t)
             + bond_vij.size() * sizeof(double)
             + inv_coord.size() * sizeof(double)
             + alias_prob.size() * sizeof(double)
             + alias_index.size() * sizeof(int32_t)
             + alias_loc_kind.size() * sizeof(int32_t)
             + bond_rmax.size() * sizeof(double)
             + tile_parity.size() + tile_prefix.size() + scan_temp.size()
             + stats.size() * sizeof(DeviceDiagonalStats)
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
};

bool is_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        // Clear the runtime error so a later call on a real GPU is not polluted.
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

std::vector<DeviceInfo> device_info() {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    std::vector<DeviceInfo> result;
    result.reserve(count);
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        check_cuda(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties");
        result.push_back(DeviceInfo{index, prop.name, prop.totalGlobalMem,
                                    prop.major, prop.minor, prop.multiProcessorCount});
    }
    return result;
}

void prefix_xor_states(const int32_t* host_types,
                       const int32_t* host_sites,
                       std::size_t length,
                       int n_sites,
                       uint64_t* host_output) {
    if (n_sites <= 0 || n_sites > 64 * kMaxWords) {
        throw std::invalid_argument("n_sites must be in [1, 384]");
    }
    const int words = (n_sites + 63) / 64;
    switch (words) {
        case 1: prefix_xor_impl<1>(host_types, host_sites, length, host_output); break;
        case 2: prefix_xor_impl<2>(host_types, host_sites, length, host_output); break;
        case 3: prefix_xor_impl<3>(host_types, host_sites, length, host_output); break;
        case 4: prefix_xor_impl<4>(host_types, host_sites, length, host_output); break;
        case 5: prefix_xor_impl<5>(host_types, host_sites, length, host_output); break;
        case 6: prefix_xor_impl<6>(host_types, host_sites, length, host_output); break;
        default: throw std::logic_error("unreachable packed-state width");
    }
}

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
                x.bond_rmax.get(), x.tile_parity.get(), x.tile_prefix.get(),
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
    check_cuda(cudaSetDevice(x.device_index), "cudaSetDevice for operator upload");
    check_cuda(cudaMemcpy(x.types.get(), host_types, x.length * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy op_types to device");
    check_cuda(cudaMemcpy(x.sites.get(), host_sites, x.length * sizeof(int32_t),
                          cudaMemcpyHostToDevice), "copy op_sites to device");
    x.last_site_events = 0;
    x.last_bond_events = 0;
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

}  // namespace qaqmc_cuda
