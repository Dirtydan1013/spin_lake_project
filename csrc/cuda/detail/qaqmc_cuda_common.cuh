#pragma once

#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qaqmc_cuda::detail {
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

void validate_operator_string(const int32_t* types,
                              const int32_t* sites,
                              std::size_t count,
                              int n_sites,
                              int n_bonds,
                              const char* label) {
    for (std::size_t op = 0; op < count; ++op) {
        const int type = types[op];
        const int site = sites[op];
        if (type == -1 || type == 1) {
            if (site >= 0 && site < n_sites) continue;
            std::ostringstream oss;
            oss << label << " single-site operator " << op
                << " has site " << site << " outside [0, " << n_sites << ')';
            throw std::invalid_argument(oss.str());
        }
        if (type == 2) {
            if (site >= 0 && site < n_bonds) continue;
            std::ostringstream oss;
            oss << label << " bond operator " << op
                << " has bond " << site << " outside [0, " << n_bonds << ')';
            throw std::invalid_argument(oss.str());
        }
        std::ostringstream oss;
        oss << label << " operator " << op << " has unsupported type " << type;
        throw std::invalid_argument(oss.str());
    }
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
        if (ptr_ && owns_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          count_(std::exchange(other.count_, 0)),
          owns_(std::exchange(other.owns_, true)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_ && owns_) cudaFree(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
            owns_ = std::exchange(other.owns_, true);
        }
        return *this;
    }
    static DeviceBuffer view(T* ptr, std::size_t count) {
        DeviceBuffer result;
        result.ptr_ = ptr;
        result.count_ = count;
        result.owns_ = false;
        return result;
    }
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }

private:
    T* ptr_{nullptr};
    std::size_t count_{0};
    bool owns_{true};
};

// Immutable Hamiltonian and proposal tables shared by every chain in one
// batched engine.  Keeping this object reference-counted prevents B chains
// from duplicating the O(G * (N + Nbonds)) alias/envelope data on the GPU.
struct DeviceHamiltonian {
    int device_index{0};
    int n_sites{0};
    double delta_min{0.0};
    double delta_max{0.0};
    double epsilon{0.0};
    int n_groups{0};
    int max_alias{0};
    int n_bonds{0};

    DeviceBuffer<int32_t> bond_sites;
    DeviceBuffer<double> bond_vij;
    DeviceBuffer<double> inv_coord;
    DeviceBuffer<double> alias_prob;
    DeviceBuffer<int32_t> alias_index;
    DeviceBuffer<int32_t> alias_loc_kind;
    DeviceBuffer<double> bond_rmax;

    std::size_t allocated_bytes() const {
        return bond_sites.size() * sizeof(int32_t)
             + bond_vij.size() * sizeof(double)
             + inv_coord.size() * sizeof(double)
             + alias_prob.size() * sizeof(double)
             + alias_index.size() * sizeof(int32_t)
             + alias_loc_kind.size() * sizeof(int32_t)
             + bond_rmax.size() * sizeof(double);
    }
};

inline std::shared_ptr<DeviceHamiltonian> make_device_hamiltonian(
    int n_sites,
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
    int device_index) {
    if (n_sites <= 0 || n_sites > 64 * kMaxWords)
        throw std::invalid_argument("n_sites must be in [1, 384]");
    if (n_groups <= 0 || max_alias <= 0)
        throw std::invalid_argument("n_groups and max_alias must be positive");
    if (n_bonds < 0)
        throw std::invalid_argument("n_bonds must be non-negative");
    if (max_alias != n_sites + n_bonds)
        throw std::invalid_argument("max_alias must equal n_sites + n_bonds");

    check_cuda(cudaSetDevice(device_index), "cudaSetDevice for shared model");
    auto model = std::make_shared<DeviceHamiltonian>();
    model->device_index = device_index;
    model->n_sites = n_sites;
    model->delta_min = delta_min;
    model->delta_max = delta_max;
    model->epsilon = epsilon;
    model->n_groups = n_groups;
    model->max_alias = max_alias;
    model->n_bonds = n_bonds;
    model->bond_sites = DeviceBuffer<int32_t>(static_cast<std::size_t>(2) * n_bonds);
    model->bond_vij = DeviceBuffer<double>(n_bonds);
    model->inv_coord = DeviceBuffer<double>(n_sites);
    const std::size_t alias_count = static_cast<std::size_t>(n_groups) * max_alias;
    const std::size_t rmax_count = static_cast<std::size_t>(n_groups) * n_bonds;
    model->alias_prob = DeviceBuffer<double>(alias_count);
    model->alias_index = DeviceBuffer<int32_t>(alias_count);
    model->alias_loc_kind = DeviceBuffer<int32_t>(alias_count);
    model->bond_rmax = DeviceBuffer<double>(rmax_count);
    if (n_bonds > 0) {
        check_cuda(cudaMemcpy(model->bond_sites.get(), host_bond_sites,
                              static_cast<std::size_t>(2) * n_bonds * sizeof(int32_t),
                              cudaMemcpyHostToDevice), "copy shared bond sites");
        check_cuda(cudaMemcpy(model->bond_vij.get(), host_bond_vij,
                              static_cast<std::size_t>(n_bonds) * sizeof(double),
                              cudaMemcpyHostToDevice), "copy shared bond strengths");
        check_cuda(cudaMemcpy(model->bond_rmax.get(), host_bond_rmax,
                              rmax_count * sizeof(double), cudaMemcpyHostToDevice),
                   "copy shared bond envelopes");
    }
    check_cuda(cudaMemcpy(model->inv_coord.get(), host_inv_coord,
                          static_cast<std::size_t>(n_sites) * sizeof(double),
                          cudaMemcpyHostToDevice), "copy shared inverse coordination");
    check_cuda(cudaMemcpy(model->alias_prob.get(), host_alias_prob,
                          alias_count * sizeof(double), cudaMemcpyHostToDevice),
               "copy shared alias probabilities");
    check_cuda(cudaMemcpy(model->alias_index.get(), host_alias_index,
                          alias_count * sizeof(int32_t), cudaMemcpyHostToDevice),
               "copy shared alias indices");
    check_cuda(cudaMemcpy(model->alias_loc_kind.get(), host_alias_loc_kind,
                          alias_count * sizeof(int32_t), cudaMemcpyHostToDevice),
               "copy shared alias locations");
    return model;
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

struct DeviceTopologyStats {
    unsigned long long attempts;
    unsigned long long accepts;
    unsigned long long invalid;
    unsigned long long active_count;
};

struct DeviceTopologyRatio {
    double log_ratio;
    int current_valid;
    int proposed_valid;
};

struct DeviceHalfLineProposal {
    double log_physical_ratio;
    int terminal_p;
    int valid;
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

__device__ int lower_bound_bond_p(const uint64_t* values,
                                  int begin,
                                  int end,
                                  uint32_t p) {
    int lo = begin, hi = end;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        const uint32_t event_p = static_cast<uint32_t>(values[mid] >> 32);
        if (event_p < p) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ int lower_bound_site_p(const uint32_t* values,
                                  int begin,
                                  int end,
                                  uint32_t p) {
    int lo = begin, hi = end;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (values[mid] < p) lo = mid + 1;
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


}  // namespace
}  // namespace qaqmc_cuda::detail
