#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Parse an optional (n_box, dim) periodic box-vectors array. The holder keeps
// the NumPy storage alive until the engine constructor has copied the data.
inline const double* parse_box(py::object obj, int& n_box,
                               py::array_t<double>& holder) {
    n_box = 0;
    if (obj.is_none()) return nullptr;
    holder = obj.cast<py::array_t<double>>();
    auto buf = holder.request();
    if (buf.ndim != 2) {
        throw std::runtime_error(
            "box_vectors must be a 2D array (n_box, dim)");
    }
    n_box = static_cast<int>(buf.shape[0]);
    return static_cast<const double*>(buf.ptr);
}

void bind_qaqmc(py::module_& m);
void bind_qaqmc_renyi(py::module_& m);
void bind_sse(py::module_& m);
void bind_qaqmc_renyi_work(py::module_& m);
