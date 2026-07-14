#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qaqmc_cuda {

struct DeviceInfo {
    int index;
    std::string name;
    std::size_t total_memory;
    int compute_major;
    int compute_minor;
    int multiprocessors;
};

// Return false (rather than throwing) when the CUDA runtime has no usable
// device.  This lets the Python test suite skip GPU tests on login/CPU nodes.
bool is_available();

std::vector<DeviceInfo> device_info();

// Compute the state immediately BEFORE every operator slice.  Only type=-1
// operators change the propagated state; type=1/2 slots are diagonal.
//
// Output is row-major [length, ceil(n_sites/64)] packed uint64 words.  This is
// a white-box/debug primitive for exact CPU/GPU validation.  Production
// diagonal resampling reuses the same tiled scan without materialising the
// full per-slice state array.
void prefix_xor_states(const int32_t* host_types,
                       const int32_t* host_sites,
                       std::size_t length,
                       int n_sites,
                       uint64_t* host_output);

}  // namespace qaqmc_cuda
