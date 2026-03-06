#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qaqmc_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qaqmc_cpp, m) {
    m.doc() = "C++ QAQMC core engine with pybind11 bindings";

#ifdef QAQMC_USE_OPENMP
    m.attr("has_openmp") = true;
    m.attr("omp_max_threads") = omp_get_max_threads();
#else
    m.attr("has_openmp") = false;
    m.attr("omp_max_threads") = 1;
#endif

    py::class_<QAQMCEngine>(m, "QAQMCEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr, int neighbor_cutoff,
                         bool precompute, int chunk_slices) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            return new QAQMCEngine(N, Omega, delta_min, delta_max, Rb, M,
                                    epsilon, seed, pos_ptr, pos_dim,
                                    neighbor_cutoff, precompute, chunk_slices);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon"), py::arg("seed"),
        py::arg("pos"), py::arg("neighbor_cutoff") = -1,
        py::arg("precompute") = true, py::arg("chunk_slices") = 0)

        .def("mc_step", &QAQMCEngine::mc_step,
             "Run one diagonal update + cluster update")

        .def("run", [](QAQMCEngine& self, int n_equil, int n_samples,
                       py::object progress_callback, int progress_every) {
            if (progress_every <= 0) progress_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            int M2 = self.get_M_total();

            // Allocate output numpy arrays
            py::array_t<int8_t> types_out({n_samples, M2});
            py::array_t<int32_t> sites_out({n_samples, M2});
            auto t_buf = types_out.mutable_unchecked<2>();
            auto s_buf = sites_out.mutable_unchecked<2>();

            for (int i = 0; i < n_samples; ++i) {
                self.mc_step();
                const auto& ot = self.get_op_types();
                const auto& os = self.get_op_sites();
                for (int p = 0; p < M2; ++p) {
                    t_buf(i, p) = static_cast<int8_t>(ot[p]);
                    s_buf(i, p) = static_cast<int32_t>(os[p]);
                }
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_samples))
                    progress_callback(i + 1, n_samples, "sample");
            }
            return py::make_tuple(types_out, sites_out);
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        "Run equilibration + sampling, returns (op_types, op_sites) numpy arrays")

        .def_property_readonly("N", &QAQMCEngine::get_N)
        .def_property_readonly("M", &QAQMCEngine::get_M)
        .def_property_readonly("M_total", &QAQMCEngine::get_M_total)

        .def_property_readonly("op_types", [](const QAQMCEngine& self) {
            const auto& v = self.get_op_types();
            return py::array_t<int32_t>(v.size(), v.data());
        })
        .def_property_readonly("op_sites", [](const QAQMCEngine& self) {
            const auto& v = self.get_op_sites();
            return py::array_t<int32_t>(v.size(), v.data());
        })
        .def_property_readonly("bond_sites", [](const QAQMCEngine& self) {
            const auto& v = self.get_bond_sites_flat();
            int n = (int)v.size() / 2;
            return py::array_t<int32_t>({n, 2}, v.data());
        })
        .def_property_readonly("delta_schedule", [](const QAQMCEngine& self) {
            const auto& v = self.get_delta_schedule();
            return py::array_t<double>(v.size(), v.data());
        })

        // Checkpoint support
        .def("get_rng_state", &QAQMCEngine::get_rng_state)
        .def("set_rng_state", &QAQMCEngine::set_rng_state)
        .def("set_op_string", [](QAQMCEngine& self,
                                 py::array_t<int32_t> types_arr,
                                 py::array_t<int32_t> sites_arr) {
            auto t = types_arr.request();
            auto s = sites_arr.request();
            self.set_op_string(static_cast<const int32_t*>(t.ptr),
                               static_cast<const int32_t*>(s.ptr),
                               (int)t.shape[0]);
        });
}
