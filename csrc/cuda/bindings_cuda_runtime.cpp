#include "bindings_cuda.cuh"
#include "qaqmc_cuda_api.cuh"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace qaqmc_cuda::bindings {

void bind_runtime(py::module_& m) {
    m.def("is_available", &qaqmc_cuda::is_available,
          "Return true when the CUDA runtime sees at least one usable GPU.");

    m.def("device_info", []() {
        py::list out;
        for (const auto& info : qaqmc_cuda::device_info()) {
            py::dict item;
            item["index"] = info.index;
            item["name"] = info.name;
            item["total_memory"] = py::int_(info.total_memory);
            item["compute_capability"] = py::make_tuple(info.compute_major,
                                                        info.compute_minor);
            item["multiprocessors"] = info.multiprocessors;
            out.append(std::move(item));
        }
        return out;
    });

    m.def("prefix_xor_states", [](
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> types,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites,
              int n_sites) {
        const auto t = types.request();
        const auto s = sites.request();
        if (t.ndim != 1 || s.ndim != 1 || t.shape[0] != s.shape[0]) {
            throw std::invalid_argument(
                "op_types and op_sites must be one-dimensional arrays of equal length");
        }
        if (n_sites <= 0 || n_sites > 384) {
            throw std::invalid_argument("n_sites must be in [1, 384]");
        }
        const auto* type_ptr = static_cast<const int32_t*>(t.ptr);
        const auto* site_ptr = static_cast<const int32_t*>(s.ptr);
        const std::size_t length = static_cast<std::size_t>(t.shape[0]);
        for (std::size_t p = 0; p < length; ++p) {
            if (type_ptr[p] != -1 && type_ptr[p] != 1 && type_ptr[p] != 2) {
                throw std::invalid_argument("op_types entries must be -1, 1, or 2");
            }
            if (type_ptr[p] == -1 && (site_ptr[p] < 0 || site_ptr[p] >= n_sites)) {
                throw std::invalid_argument("off-diagonal op_sites entry is out of range");
            }
        }

        const int words = (n_sites + 63) / 64;
        py::array_t<uint64_t> output(
            {static_cast<py::ssize_t>(length), static_cast<py::ssize_t>(words)});
        {
            py::gil_scoped_release release;
            qaqmc_cuda::prefix_xor_states(type_ptr, site_ptr, length, n_sites,
                                          output.mutable_data());
        }
        return output;
    }, py::arg("op_types"), py::arg("op_sites"), py::arg("n_sites"),
    "Return packed state immediately before every operator slice.\n\n"
    "The result has shape (M_total, ceil(n_sites/64)) and dtype uint64.");
}

}  // namespace qaqmc_cuda::bindings
