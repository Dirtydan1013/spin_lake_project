#include "include/api.cuh"
#include "detail/prefix_kernels.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace qaqmc_cuda {
using namespace detail;
namespace {
// Lowest architecture compiled into this fatbin.  nvcc defines
// __CUDA_ARCH_LIST__ (e.g. "700,800") for every translation unit it
// compiles; devices at or above the minimum run natively or via PTX JIT,
// devices below it fail with "no kernel image is available".
constexpr int kMinCompiledArch = []() constexpr {
#ifdef __CUDA_ARCH_LIST__
    constexpr int archs[] = {__CUDA_ARCH_LIST__};
    int lowest = archs[0];
    for (int arch : archs) {
        if (arch < lowest) lowest = arch;
    }
    return lowest;
#else
    return 700;  // conservative default: Volta
#endif
}();

inline bool prop_supported(const cudaDeviceProp& prop) {
    return prop.major * 100 + prop.minor * 10 >= kMinCompiledArch;
}
}  // namespace

int min_supported_compute() { return kMinCompiledArch; }

bool is_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        // Clear the runtime error so a later call on a real GPU is not polluted.
        cudaGetLastError();
        return false;
    }
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
            cudaGetLastError();
            continue;
        }
        if (prop_supported(prop)) return true;
    }
    return false;
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
                                    prop.major, prop.minor, prop.multiProcessorCount,
                                    prop_supported(prop)});
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
