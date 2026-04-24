#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qaqmc_core.hpp"
#include "qaqmc_renyi_core.hpp"
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
                       py::object progress_callback, int progress_every,
                       int measure_every) {
            if (progress_every <= 0) progress_every = 1;
            if (measure_every <= 0) measure_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            int M2 = self.get_M_total();
            int total_steps = n_samples * measure_every;

            // Allocate output numpy arrays
            py::array_t<int8_t> types_out({n_samples, M2});
            py::array_t<int32_t> sites_out({n_samples, M2});
            auto t_buf = types_out.mutable_unchecked<2>();
            auto s_buf = sites_out.mutable_unchecked<2>();

            int sample_idx = 0;
            for (int i = 0; i < total_steps; ++i) {
                self.mc_step();
                if ((i + 1) % measure_every == 0) {
                    const auto& ot = self.get_op_types();
                    const auto& os = self.get_op_sites();
                    for (int p = 0; p < M2; ++p) {
                        t_buf(sample_idx, p) = static_cast<int8_t>(ot[p]);
                        s_buf(sample_idx, p) = static_cast<int32_t>(os[p]);
                    }
                    ++sample_idx;
                    if (has_cb && ((sample_idx % progress_every) == 0 || sample_idx == n_samples))
                        progress_callback(sample_idx, n_samples, "sample");
                }
            }
            return py::make_tuple(types_out, sites_out);
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        py::arg("measure_every") = 1,
        "Run equilibration + sampling, returns (op_types, op_sites) numpy arrays.\n"
        "measure_every: record one sample every this many MC steps (default 1).")

        .def_property_readonly("N", &QAQMCEngine::get_N)
        .def_property_readonly("M", &QAQMCEngine::get_M)
        .def_property_readonly("M_total", &QAQMCEngine::get_M_total)
        .def_property_readonly("n_loops",   &QAQMCEngine::get_n_loops)
        .def_property_readonly("n_strings", &QAQMCEngine::get_n_strings)

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

        // ── On-the-fly observable support ─────────────────────────────────────
        .def("set_bulk_sites", [](QAQMCEngine& self, py::list bulk_py) {
            std::vector<int> bulk;
            for (auto& x : bulk_py) bulk.push_back(x.cast<int>());
            self.set_bulk_sites(bulk);
        },
        py::arg("bulk_sites"),
        "Set interior (bulk) site indices for density computation; "
        "if empty, all sites are used")

        .def("set_observable_sites", [](QAQMCEngine& self,
                                        py::list loop_sets_py,
                                        py::list string_sets_py) {
            std::vector<std::vector<int>> loop_sets, string_sets;
            for (auto& item : loop_sets_py) {
                auto arr = item.cast<py::list>();
                std::vector<int> v;
                for (auto& x : arr) v.push_back(x.cast<int>());
                loop_sets.push_back(std::move(v));
            }
            for (auto& item : string_sets_py) {
                auto arr = item.cast<py::list>();
                std::vector<int> v;
                for (auto& x : arr) v.push_back(x.cast<int>());
                string_sets.push_back(std::move(v));
            }
            self.set_observable_sites(loop_sets, string_sets);
        },
        py::arg("loop_sets"), py::arg("string_sets"),
        "Set loop and string site index arrays for Z(l)/C_m(l) on-the-fly measurement")

        .def("run_onthefly", [](QAQMCEngine& self, int n_equil, int n_samples,
                                int me_density, int me_zl, int me_cml,
                                py::object progress_callback, int progress_every) {
            if (me_density  <= 0) me_density  = 1;
            if (me_zl       <= 0) me_zl       = 1;
            if (me_cml      <= 0) me_cml      = 1;
            if (progress_every <= 0) progress_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            // n_samples is defined for the finest observable (smallest interval).
            // Total steps = n_samples * min_interval.
            int min_me      = std::min({me_density, me_zl, me_cml});
            int total_steps = n_samples * min_me;
            int n_density   = total_steps / me_density;
            int n_zl        = total_steps / me_zl;
            int n_cml       = total_steps / me_cml;

            py::array_t<double> density_out(n_density);
            py::array_t<double> z_l_out(n_zl);
            py::array_t<double> c_m_l_out(n_cml);
            auto d_buf = density_out.mutable_unchecked<1>();
            auto z_buf = z_l_out.mutable_unchecked<1>();
            auto c_buf = c_m_l_out.mutable_unchecked<1>();

            int n_loops   = self.get_n_loops();
            int n_strings = self.get_n_strings();

            // Reallocate output arrays with per-copy second dimension
            py::array_t<double> z_l_out2({n_zl,  n_loops});
            py::array_t<double> c_m_l_out2({n_cml, n_strings});
            auto z_buf2 = z_l_out2.mutable_unchecked<2>();
            auto c_buf2 = c_m_l_out2.mutable_unchecked<2>();

            int idx_d = 0, idx_z = 0, idx_c = 0;
            for (int i = 0; i < total_steps; ++i) {
                self.mc_step();
                int step = i + 1;
                bool need_any = (step % me_density == 0) ||
                                (step % me_zl      == 0) ||
                                (step % me_cml     == 0);
                if (need_any) {
                    auto obs = self.measure_at_midpoint();
                    if (step % me_density == 0) d_buf(idx_d++) = obs.density;
                    if (step % me_zl == 0) {
                        for (int k = 0; k < n_loops; ++k)
                            z_buf2(idx_z, k) = obs.Z_l_copies[k];
                        ++idx_z;
                    }
                    if (step % me_cml == 0) {
                        for (int k = 0; k < n_strings; ++k)
                            c_buf2(idx_c, k) = obs.C_m_l_copies[k];
                        ++idx_c;
                    }
                }
                if (has_cb) {
                    int finest_done = step / min_me;
                    if ((step % min_me == 0) &&
                        (finest_done % progress_every == 0 || finest_done == n_samples))
                        progress_callback(finest_done, n_samples, "sample");
                }
            }

            py::dict result;
            result["density"] = density_out;
            result["Z_l"]     = z_l_out2;   // (n_zl, n_loops)
            result["C_m_l"]   = c_m_l_out2; // (n_cml, n_strings)
            return result;
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("me_density") = 1,
        py::arg("me_zl")      = 1,
        py::arg("me_cml")     = 1,
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        "On-the-fly symmetry-point sampling with per-observable measure intervals.\n"
        "n_samples refers to the finest (smallest) interval. Others get proportionally fewer.\n"
        "Returns dict: density (n_density,), Z_l (n_zl, n_loops), C_m_l (n_cml, n_strings).\n"
        "To compute |<Z(l)>|: np.abs(arr.mean(axis=0)).mean() — mean per copy, then |·|, then avg.")

        .def("run_profile", [](QAQMCEngine& self, int n_equil, int n_samples,
                               int me_density, int me_zl, int me_cml,
                               int profile_step, int batch_size,
                               py::object progress_callback, int progress_every) {
            if (me_density  <= 0) me_density  = 1;
            if (me_zl       <= 0) me_zl       = 1;
            if (me_cml      <= 0) me_cml      = 1;
            if (profile_step   <= 0) profile_step   = 10000;
            if (batch_size     <= 0) batch_size     = 1000;
            if (progress_every <= 0) progress_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            int M_total   = self.get_M_total();
            int n_points  = M_total / profile_step;
            int n_loops   = self.get_n_loops();
            int n_strings = self.get_n_strings();
            int min_me    = std::min({me_density, me_zl, me_cml});

            // n_samples is defined for the finest observable.
            // We group samples into batches; each batch stores the mean per copy per point.
            int total_steps = n_samples * min_me;
            int n_density   = total_steps / me_density;  // total density samples
            int n_batches_z = (total_steps / me_zl)  / batch_size;
            int n_batches_c = (total_steps / me_cml) / batch_size;
            if (n_batches_z < 1) n_batches_z = 1;
            if (n_batches_c < 1) n_batches_c = 1;

            // density: still per-sample (n_density, n_points)
            py::array_t<double> density_out({n_density, n_points});
            auto d_buf = density_out.mutable_unchecked<2>();

            // Z_l, C_m_l: batched per-copy  (n_batches, n_points, n_copies)
            py::array_t<double> z_l_out  ({n_batches_z, n_points, n_loops});
            py::array_t<double> c_m_l_out({n_batches_c, n_points, n_strings});
            auto z_buf = z_l_out.mutable_unchecked<3>();
            auto c_buf = c_m_l_out.mutable_unchecked<3>();

            // Initialize batch accumulators
            std::vector<std::vector<double>> z_acc(n_loops,   std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> c_acc(n_strings, std::vector<double>(n_points, 0.0));
            int idx_d = 0;
            int z_in_batch = 0, c_in_batch = 0;
            int batch_z = 0, batch_c = 0;

            for (int i = 0; i < total_steps; ++i) {
                self.mc_step();
                int step = i + 1;
                bool need_d = (step % me_density == 0);
                bool need_z = (step % me_zl      == 0);
                bool need_c = (step % me_cml     == 0);

                if (need_d || need_z || need_c) {
                    auto prof = self.measure_profile(profile_step);
                    if (need_d && idx_d < n_density) {
                        for (int k = 0; k < n_points; ++k) d_buf(idx_d, k) = prof.density[k];
                        ++idx_d;
                    }
                    if (need_z) {
                        for (int k = 0; k < n_loops; ++k)
                            for (int pt = 0; pt < n_points; ++pt)
                                z_acc[k][pt] += prof.Z_l_copies[k][pt];
                        ++z_in_batch;
                        if (z_in_batch >= batch_size && batch_z < n_batches_z) {
                            double inv = 1.0 / z_in_batch;
                            for (int k = 0; k < n_loops; ++k)
                                for (int pt = 0; pt < n_points; ++pt) {
                                    z_buf(batch_z, pt, k) = z_acc[k][pt] * inv;
                                    z_acc[k][pt] = 0.0;
                                }
                            ++batch_z;
                            z_in_batch = 0;
                        }
                    }
                    if (need_c) {
                        for (int k = 0; k < n_strings; ++k)
                            for (int pt = 0; pt < n_points; ++pt)
                                c_acc[k][pt] += prof.C_m_l_copies[k][pt];
                        ++c_in_batch;
                        if (c_in_batch >= batch_size && batch_c < n_batches_c) {
                            double inv = 1.0 / c_in_batch;
                            for (int k = 0; k < n_strings; ++k)
                                for (int pt = 0; pt < n_points; ++pt) {
                                    c_buf(batch_c, pt, k) = c_acc[k][pt] * inv;
                                    c_acc[k][pt] = 0.0;
                                }
                            ++batch_c;
                            c_in_batch = 0;
                        }
                    }
                }
                if (has_cb) {
                    int finest_done = step / min_me;
                    if ((step % min_me == 0) &&
                        (finest_done % progress_every == 0 || finest_done == n_samples))
                        progress_callback(finest_done, n_samples, "sample");
                }
            }
            // Flush remaining partial batch
            if (z_in_batch > 0 && batch_z < n_batches_z) {
                double inv = 1.0 / z_in_batch;
                for (int k = 0; k < n_loops; ++k)
                    for (int pt = 0; pt < n_points; ++pt)
                        z_buf(batch_z, pt, k) = z_acc[k][pt] * inv;
            }
            if (c_in_batch > 0 && batch_c < n_batches_c) {
                double inv = 1.0 / c_in_batch;
                for (int k = 0; k < n_strings; ++k)
                    for (int pt = 0; pt < n_points; ++pt)
                        c_buf(batch_c, pt, k) = c_acc[k][pt] * inv;
            }

            // p-index array
            py::array_t<int> p_indices(n_points);
            auto p_buf = p_indices.mutable_unchecked<1>();
            for (int k = 0; k < n_points; ++k)
                p_buf(k) = (k + 1) * profile_step - 1;

            py::dict result;
            result["density"]    = density_out;  // (n_density, n_points)
            result["Z_l"]        = z_l_out;       // (n_batches_z, n_points, n_loops)
            result["C_m_l"]      = c_m_l_out;     // (n_batches_c, n_points, n_strings)
            result["p_indices"]  = p_indices;
            result["batch_size"] = py::int_(batch_size);
            return result;
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("me_density") = 1,
        py::arg("me_zl")      = 1,
        py::arg("me_cml")     = 1,
        py::arg("profile_step")  = 10000,
        py::arg("batch_size")    = 1000,
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        "Asymmetric profile with batched per-copy storage.\n"
        "density: (n_density, n_points) per-sample.\n"
        "Z_l:     (n_batches, n_points, n_loops)   — batch means of per-copy signed products.\n"
        "C_m_l:   (n_batches, n_points, n_strings) — batch means of per-copy signed products.\n"
        "To get |<Z(l)>|(δ): np.abs(arr.mean(axis=0)).mean(axis=-1)  [mean over batches, then |·|, then avg copies].")

        // Profiling
        .def_property_readonly("time_diag", &QAQMCEngine::get_time_diag)
        .def_property_readonly("time_clus", &QAQMCEngine::get_time_clus)
        .def_property_readonly("mc_steps", &QAQMCEngine::get_mc_steps)
        .def("reset_timers", &QAQMCEngine::reset_timers);

    // ── QAQMCRenyiEngine ─────────────────────────────────────────────────────

    py::class_<QAQMCRenyiEngine>(m, "QAQMCRenyiEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr, int neighbor_cutoff,
                         int delta_groups) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = static_cast<int>(buf.shape[1]);
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            return new QAQMCRenyiEngine(
                N, Omega, delta_min, delta_max, Rb, M, epsilon, seed,
                pos_ptr, pos_dim, neighbor_cutoff, delta_groups);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon") = 0.01, py::arg("seed") = 42,
        py::arg("pos"), py::arg("neighbor_cutoff") = -1, py::arg("delta_groups") = 0)

        .def("mc_step", &QAQMCRenyiEngine::mc_step,
             "Run one two-replica QAQMC step in ratio-estimator mode.")
        .def("run_steps", &QAQMCRenyiEngine::run_steps, py::arg("n_steps"),
             "Run several mc_step() iterations.")
        .def("set_A_mask", [](QAQMCRenyiEngine& self, py::array_t<uint8_t> mask_arr) {
            auto buf = mask_arr.request();
            self.set_A_mask(static_cast<const uint8_t*>(buf.ptr), static_cast<int>(buf.shape[0]));
        }, py::arg("mask"))
        .def("set_topology_pair", [](QAQMCRenyiEngine& self,
                                     py::array_t<uint8_t> lower_mask_arr,
                                     py::array_t<uint8_t> upper_mask_arr,
                                     int diff_site) {
            auto lower = lower_mask_arr.request();
            auto upper = upper_mask_arr.request();
            self.set_topology_pair(
                static_cast<const uint8_t*>(lower.ptr),
                static_cast<const uint8_t*>(upper.ptr),
                static_cast<int>(lower.shape[0]),
                diff_site);
        }, py::arg("A_k"), py::arg("A_kp1"), py::arg("diff_site"))
        .def("reset_visit_counts", &QAQMCRenyiEngine::reset_visit_counts)
        .def("set_ensemble_ladder", [](QAQMCRenyiEngine& self,
                                       py::list masks_py,
                                       py::list neighbors_py,
                                       int initial_ensemble) {
            std::vector<std::vector<uint8_t>> masks;
            masks.reserve(py::len(masks_py));
            for (auto item : masks_py) {
                auto mask_arr = py::cast<py::array_t<uint8_t>>(item);
                auto buf = mask_arr.request();
                const auto* ptr = static_cast<const uint8_t*>(buf.ptr);
                masks.emplace_back(ptr, ptr + static_cast<int>(buf.shape[0]));
            }

            std::vector<std::vector<int>> neighbors;
            neighbors.reserve(py::len(neighbors_py));
            for (auto item : neighbors_py) {
                std::vector<int> row;
                for (auto nbr : py::cast<py::list>(item)) {
                    row.push_back(py::cast<int>(nbr));
                }
                neighbors.push_back(std::move(row));
            }
            self.set_ensemble_ladder(masks, neighbors, initial_ensemble);
        }, py::arg("masks"), py::arg("neighbors"), py::arg("initial_ensemble") = 0)
        .def("set_log_g", [](QAQMCRenyiEngine& self, py::array_t<double> log_g_arr) {
            auto buf = log_g_arr.request();
            const auto* ptr = static_cast<const double*>(buf.ptr);
            self.set_log_g(std::vector<double>(ptr, ptr + static_cast<int>(buf.shape[0])));
        }, py::arg("log_g"))
        .def("reset_visit_counts_ext", &QAQMCRenyiEngine::reset_visit_counts_ext)
        .def("reset_transition_counts", &QAQMCRenyiEngine::reset_transition_counts)
        .def("reset_collection_counts", &QAQMCRenyiEngine::reset_collection_counts)
        .def("topology_toggle", &QAQMCRenyiEngine::topology_toggle)
        .def("ensemble_switch", &QAQMCRenyiEngine::ensemble_switch)
        .def("log_weight_ratio_for_site", &QAQMCRenyiEngine::log_weight_ratio_for_site,
             py::arg("site"), py::arg("from_topology"), py::arg("to_topology"))
        .def("set_indicator_site", &QAQMCRenyiEngine::set_indicator_site, py::arg("site"))
        .def("reset_indicator", &QAQMCRenyiEngine::reset_indicator)
        .def("get_indicator_avg", &QAQMCRenyiEngine::get_indicator_avg)
        .def("current_indicator", &QAQMCRenyiEngine::current_indicator)
        .def("set_replica_op_string", [](QAQMCRenyiEngine& self, int replica,
                                         py::array_t<int32_t> types_arr,
                                         py::array_t<int32_t> sites_arr) {
            auto t = types_arr.request();
            auto s = sites_arr.request();
            self.set_replica_op_string(
                replica,
                static_cast<const int32_t*>(t.ptr),
                static_cast<const int32_t*>(s.ptr),
                static_cast<int>(t.shape[0]));
        }, py::arg("replica"), py::arg("types"), py::arg("sites"))
        .def("recompute_midpoint_states", &QAQMCRenyiEngine::recompute_midpoint_states)
        .def("get_site_paths", [](const QAQMCRenyiEngine& self, int site) {
            auto paths = self.get_site_paths(site);
            py::dict out;
            auto to_array = [](const std::vector<int32_t>& v) {
                py::array_t<int32_t> arr(v.size());
                auto buf = arr.mutable_unchecked<1>();
                for (ssize_t i = 0; i < static_cast<ssize_t>(v.size()); ++i) buf(i) = v[i];
                return arr;
            };
            out["channel_0"] = to_array(paths[0]);
            out["channel_1"] = to_array(paths[1]);
            out["replica_0"] = to_array(paths[2]);
            out["replica_1"] = to_array(paths[3]);
            return out;
        }, py::arg("site"))
        .def("get_state_at_M", [](const QAQMCRenyiEngine& self, int replica) {
            const auto& v = self.get_state_at_M(replica);
            return py::array_t<int32_t>(v.size(), v.data());
        }, py::arg("replica"))
        .def("get_op_types", [](const QAQMCRenyiEngine& self, int replica) {
            const auto& v = self.get_op_types(replica);
            return py::array_t<int32_t>(v.size(), v.data());
        }, py::arg("replica"))
        .def("get_op_sites", [](const QAQMCRenyiEngine& self, int replica) {
            const auto& v = self.get_op_sites(replica);
            return py::array_t<int32_t>(v.size(), v.data());
        }, py::arg("replica"))
        .def_property_readonly("A_mask", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_A_mask();
            return py::array_t<uint8_t>(v.size(), v.data());
        })
        .def("get_topology_mask", [](const QAQMCRenyiEngine& self, int topology) {
            const auto& v = self.get_topology_mask(topology);
            return py::array_t<uint8_t>(v.size(), v.data());
        }, py::arg("topology"))
        .def("get_visit_counts", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_visit_counts();
            return py::array_t<int64_t>({2}, v.data());
        })
        .def("get_visit_counts_ext", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_visit_counts_ext();
            return py::array_t<int64_t>(v.size(), v.data());
        })
        .def("get_transition_counts", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_transition_counts();
            int n = self.get_ensemble_count();
            return py::array_t<int64_t>({n, n}, v.data());
        })
        .def("get_collection_counts", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_collection_counts();
            int n = self.get_ensemble_count();
            return py::array_t<double>({n, n}, v.data());
        })
        .def_property_readonly("bond_sites", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_bond_sites_flat();
            int n = static_cast<int>(v.size()) / 2;
            return py::array_t<int32_t>({n, 2}, v.data());
        })
        .def_property_readonly("delta_schedule", [](const QAQMCRenyiEngine& self) {
            const auto& v = self.get_delta_schedule();
            return py::array_t<double>(v.size(), v.data());
        })
        .def_property_readonly("N", &QAQMCRenyiEngine::get_N)
        .def_property_readonly("M", &QAQMCRenyiEngine::get_M)
        .def_property_readonly("M_total", &QAQMCRenyiEngine::get_M_total)
        .def_property_readonly("indicator_site", &QAQMCRenyiEngine::get_indicator_site)
        .def_property_readonly("current_topology", &QAQMCRenyiEngine::get_current_topology)
        .def_property_readonly("current_ensemble", &QAQMCRenyiEngine::get_current_ensemble)
        .def_property_readonly("ensemble_count", &QAQMCRenyiEngine::get_ensemble_count)
        .def_property_readonly("mode", &QAQMCRenyiEngine::get_mode)
        .def_property_readonly("delta_groups", &QAQMCRenyiEngine::get_delta_groups)
        .def_property_readonly("diff_site", &QAQMCRenyiEngine::get_diff_site);

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
