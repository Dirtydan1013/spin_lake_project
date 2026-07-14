#include "qaqmc_cuda_scan.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qaqmc_cuda {
namespace {

constexpr int kBlockSize = 256;
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

}  // namespace

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

}  // namespace qaqmc_cuda
