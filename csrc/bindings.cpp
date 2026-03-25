#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qaqmc_core.hpp"
#include "sse_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(qaqmc_cpp, m) {
    m.doc() = "C++ QAQMC and SSE core engines with pybind11 bindings";

#ifdef QAQMC_USE_OPENMP
    m.attr("has_openmp") = true;
    m.attr("omp_max_threads") = omp_get_max_threads();
#else
    m.attr("has_openmp") = false;
    m.attr("omp_max_threads") = 1;
#endif

    // ── QAQMCEngine ──────────────────────────────────────────────────────────

    py::class_<QAQMCEngine>(m, "QAQMCEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr, int neighbor_cutoff,
                         bool precompute, int chunk_slices, int delta_groups) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            return new QAQMCEngine(N, Omega, delta_min, delta_max, Rb, M,
                                    epsilon, seed, pos_ptr, pos_dim,
                                    neighbor_cutoff, precompute, chunk_slices,
                                    delta_groups);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon"), py::arg("seed"),
        py::arg("pos"), py::arg("neighbor_cutoff") = -1,
        py::arg("precompute") = true, py::arg("chunk_slices") = 0,
        py::arg("delta_groups") = 0)

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
        })

        // Profiling
        .def_property_readonly("time_diag", &QAQMCEngine::get_time_diag)
        .def_property_readonly("time_clus", &QAQMCEngine::get_time_clus)
        .def_property_readonly("mc_steps", &QAQMCEngine::get_mc_steps)
        .def("reset_timers", &QAQMCEngine::reset_timers);

    // ── SSEEngine ─────────────────────────────────────────────────────────────

    py::class_<SSEEngine>(m, "SSEEngine",
        "Finite-temperature SSE QMC engine for the Rydberg Hamiltonian.\n\n"
        "Parameters\n----------\n"
        "N             : Number of atoms\n"
        "Omega         : Rabi frequency (energy scale)\n"
        "delta         : Global detuning\n"
        "Rb            : Blockade radius\n"
        "beta          : Inverse temperature\n"
        "epsilon       : Safety margin for alias table offset (default 0.01)\n"
        "seed          : RNG seed\n"
        "pos           : (N, d) float64 atom positions\n"
        "neighbor_cutoff: Keep only first N neighbor shells (−1 = all bonds)\n")
        .def(py::init([](int N, double Omega, double delta, double Rb,
                         double beta, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr,
                         int neighbor_cutoff) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            return new SSEEngine(N, Omega, delta, Rb, beta, epsilon, seed,
                                  pos_ptr, pos_dim, neighbor_cutoff);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta"), py::arg("Rb"),
        py::arg("beta"), py::arg("epsilon") = 0.01, py::arg("seed") = 42,
        py::arg("pos"), py::arg("neighbor_cutoff") = -1)

        .def("mc_step", &SSEEngine::mc_step,
             "Run one diagonal update + cluster update + adjust_M")

        .def("run", [](SSEEngine& self, int n_equil, int n_samples,
                       py::object progress_callback, int progress_every) {
            if (progress_every <= 0) progress_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            // Sampling: record per-step observables
            py::array_t<double>  energies(n_samples);
            py::array_t<double>  densities(n_samples);
            py::array_t<double>  mz(n_samples);
            py::array_t<int32_t> n_ops_arr(n_samples);

            auto e_buf = energies.mutable_unchecked<1>();
            auto d_buf = densities.mutable_unchecked<1>();
            auto m_buf = mz.mutable_unchecked<1>();
            auto n_buf = n_ops_arr.mutable_unchecked<1>();

            for (int i = 0; i < n_samples; ++i) {
                self.mc_step();
                e_buf(i) = self.measure_energy();
                d_buf(i) = self.measure_density();
                m_buf(i) = self.measure_mz();
                n_buf(i) = self.get_n_ops();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_samples))
                    progress_callback(i + 1, n_samples, "sample");
            }

            py::dict result;
            result["energies"]  = energies;
            result["densities"] = densities;
            result["mz"]        = mz;
            result["n_ops"]     = n_ops_arr;
            return result;
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        "Run equilibration + sampling.\n\n"
        "Returns\n-------\n"
        "dict with keys 'energies', 'densities', 'mz', 'n_ops' — each a 1D numpy array\n"
        "of length n_samples, one measurement per mc_step.")

        // Scalar properties
        .def_property_readonly("N",      &SSEEngine::get_N)
        .def_property_readonly("M",      &SSEEngine::get_M)
        .def_property_readonly("n_ops",  &SSEEngine::get_n_ops)
        .def_property_readonly("beta",   &SSEEngine::get_beta)
        .def_property_readonly("norm_N", &SSEEngine::get_norm_N)

        // Instant observable accessors
        .def("measure_energy",  &SSEEngine::measure_energy,
             "Current energy estimate: -n_ops/beta + sum_b C_b")
        .def("measure_density", &SSEEngine::measure_density,
             "Current Rydberg density: mean(state)")
        .def("measure_mz",      &SSEEngine::measure_mz,
             "Current staggered magnetization: (1/N) sum_i (-1)^i (n_i - 0.5)")

        // Array views
        .def_property_readonly("state", [](const SSEEngine& self) {
            const auto& v = self.get_state();
            return py::array_t<int32_t>(v.size(), v.data());
        }, "Current spin state (boundary at imaginary-time 0)")

        .def_property_readonly("op_types", [](const SSEEngine& self) {
            const auto& v = self.get_op_types();
            return py::array_t<int32_t>(v.size(), v.data());
        }, "Operator type array (length M): 0=identity, 1=diag-site, 2=diag-bond, -1=off-diag")

        .def_property_readonly("op_sites", [](const SSEEngine& self) {
            const auto& v = self.get_op_sites();
            return py::array_t<int32_t>(v.size(), v.data());
        }, "Operator site/bond index array (length M); -1 for identity slots")

        .def_property_readonly("bond_sites", [](const SSEEngine& self) {
            const auto& v = self.get_bond_sites_flat();
            int n = (int)v.size() / 2;
            return py::array_t<int32_t>({n, 2}, v.data());
        }, "(n_bonds, 2) bond endpoint indices")

        // Checkpoint
        .def("get_rng_state", &SSEEngine::get_rng_state,
             "Serialise RNG state to a string for checkpointing")
        .def("set_rng_state", &SSEEngine::set_rng_state,
             "Restore RNG state from a previously saved string")

        // Profiling
        .def_property_readonly("time_diag", &SSEEngine::get_time_diag,
                               "Cumulative wall-clock time in diagonal_update (s)")
        .def_property_readonly("time_clus", &SSEEngine::get_time_clus,
                               "Cumulative wall-clock time in cluster_update (s)")
        .def_property_readonly("mc_steps",  &SSEEngine::get_mc_steps,
                               "Total number of mc_step() calls since last reset")
        .def("reset_timers", &SSEEngine::reset_timers,
             "Reset profiling counters to zero");
}
