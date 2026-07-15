#pragma once

#include <pybind11/pybind11.h>

namespace qaqmc_cuda::bindings {

void bind_runtime(pybind11::module_& module);
void bind_diagonal_engine(pybind11::module_& module);
void bind_renyi_engine(pybind11::module_& module);

}  // namespace qaqmc_cuda::bindings
