#pragma once

#include "qaqmc_cuda_scan_primitives.cuh"

#include <cub/cub.cuh>

namespace qaqmc_cuda::detail {
namespace {
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
}  // namespace qaqmc_cuda::detail
