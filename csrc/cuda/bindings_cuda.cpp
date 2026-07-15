#include "bindings_cuda.cuh"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(qaqmc_cuda, module) {
    module.doc() = "CUDA primitives and engine backend for QAQMC";
    qaqmc_cuda::bindings::bind_runtime(module);
    qaqmc_cuda::bindings::bind_diagonal_engine(module);
    qaqmc_cuda::bindings::bind_renyi_engine(module);
    qaqmc_cuda::bindings::bind_batched_diagonal_engine(module);
    qaqmc_cuda::bindings::bind_batched_renyi_engine(module);
}
