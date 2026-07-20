#include "bindings.hpp"
#include "include/qaqmc_core.hpp"

#include <cstring>

void bind_qaqmc(py::module_& m) {
    // Allocation-free hooks for the large-M unit tests: exercise the exact
    // production slot arithmetic (delta schedule, packed64 event encoding)
    // at p ~ 1e11 without building an engine.
    m.def("_delta_at", &QAQMCEngine::delta_at_static,
          py::arg("p"), py::arg("M"), py::arg("delta_min"), py::arg("delta_max"));
    m.def("_packed64_roundtrip", [](std::int64_t p, int b, int e) {
        const std::int64_t packed = QAQMCEngine::pack_bond_entry(p, b, e);
        return py::make_tuple(packed,
                              QAQMCEngine::bond_entry_p(packed),
                              QAQMCEngine::bond_entry_b(packed),
                              QAQMCEngine::bond_entry_endpoint(packed));
    });
    m.attr("_packed64_slot_max") = py::int_(QAQMCEngine::kPackedSlotMax);
    m.attr("_packed64_bond_max") = py::int_(QAQMCEngine::kPackedBondMax);

    // ── QAQMCEngine ──────────────────────────────────────────────────────────

    py::class_<QAQMCEngine::HalfLineProposal>(m, "HalfLineProposal")
        .def_readonly("valid", &QAQMCEngine::HalfLineProposal::valid)
        .def_readonly("terminal_p", &QAQMCEngine::HalfLineProposal::terminal_p)
        .def_readonly("log_physical_ratio", &QAQMCEngine::HalfLineProposal::log_physical_ratio);

    py::class_<QAQMCModelData, std::shared_ptr<QAQMCModelData>>(
        m, "QAQMCModelData")
        .def_property_readonly("N", [](const QAQMCModelData& self) {
            return self.N;
        })
        .def_property_readonly("M", [](const QAQMCModelData& self) {
            return self.M;
        })
        .def_property_readonly("M_total", [](const QAQMCModelData& self) {
            return self.M_total;
        })
        .def_property_readonly("n_bonds", [](const QAQMCModelData& self) {
            return self.vij.n_bonds;
        })
        .def_property_readonly("logical_bytes", &QAQMCModelData::logical_bytes);

    py::class_<QAQMCEngine>(m, "QAQMCEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, std::int64_t M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr, int neighbor_cutoff,
                         int delta_groups, py::object box_vectors) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            int n_box; py::array_t<double> box_holder;
            const double* box_ptr = parse_box(box_vectors, n_box, box_holder);
            return new QAQMCEngine(N, Omega, delta_min, delta_max, Rb, M,
                                    epsilon, seed, pos_ptr, pos_dim,
                                    neighbor_cutoff, delta_groups, box_ptr, n_box);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon"), py::arg("seed"),
        py::arg("pos"), py::arg("neighbor_cutoff") = -1,
        py::arg("delta_groups") = 600, py::arg("box_vectors") = py::none())

        .def(py::init<std::shared_ptr<QAQMCModelData>, uint64_t>(),
             py::arg("model_data"), py::arg("seed"))

        .def("mc_step", &QAQMCEngine::mc_step,
             py::call_guard<py::gil_scoped_release>(),
             "Run one diagonal update + cluster update")

        .def("run", [](QAQMCEngine& self, int n_equil, int n_samples,
                       py::object progress_callback, int progress_every,
                       int measure_every) {
            if (progress_every <= 0) progress_every = 1;
            if (measure_every <= 0) measure_every = 1;
            const bool has_cb = !progress_callback.is_none();

            // Equilibration
            for (int i = 0; i < n_equil; ++i) {
                {
                    py::gil_scoped_release release;
                    self.mc_step();
                }
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            const py::ssize_t M2 = self.get_M_total();
            int total_steps = n_samples * measure_every;

            // Allocate output numpy arrays
            py::array_t<int8_t> types_out({(py::ssize_t)n_samples, M2});
            py::array_t<int32_t> sites_out({(py::ssize_t)n_samples, M2});
            auto t_buf = types_out.mutable_unchecked<2>();
            auto s_buf = sites_out.mutable_unchecked<2>();

            int sample_idx = 0;
            for (int i = 0; i < total_steps; ++i) {
                self.mc_step();
                if ((i + 1) % measure_every == 0) {
                    const auto& ot = self.get_op_types();
                    for (py::ssize_t p = 0; p < M2; ++p) {
                        t_buf(sample_idx, p) = static_cast<int8_t>(ot[p]);
                        s_buf(sample_idx, p) = static_cast<int32_t>(self.get_op_site(p));
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
        .def_property_readonly("delta_min", &QAQMCEngine::get_delta_min)
        .def_property_readonly("delta_max", &QAQMCEngine::get_delta_max)
        .def_property_readonly("epsilon", &QAQMCEngine::get_epsilon)
        .def_property_readonly("n_loops",   &QAQMCEngine::get_n_loops)
        .def_property_readonly("n_strings", &QAQMCEngine::get_n_strings)
        .def_property_readonly("n_loop_size_groups",   &QAQMCEngine::get_n_loop_size_groups)
        .def_property_readonly("n_string_size_groups", &QAQMCEngine::get_n_string_size_groups)

        .def_property_readonly("op_types", [](const QAQMCEngine& self) {
            const auto& v = self.get_op_types();
            py::array_t<int32_t> out(v.size());
            self.export_op_types(out.mutable_data(),
                                 static_cast<qaqmc_slot_t>(v.size()));
            return out;
        })
        .def_property_readonly("op_sites", [](const QAQMCEngine& self) {
            py::array_t<int32_t> out(self.get_M_total());
            self.export_op_sites(out.mutable_data(), self.get_M_total());
            return out;
        })
        .def_property_readonly("bond_sites", [](const QAQMCEngine& self) {
            const auto& v = self.get_bond_sites_flat();
            int n = (int)v.size() / 2;
            return py::array_t<int32_t>({n, 2}, v.data());
        })
        .def_property_readonly("delta_schedule", [](const QAQMCEngine& self) {
            const auto v = self.get_delta_schedule();
            py::array_t<double> out(v.size());
            std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
            return out;
        })
        .def_property_readonly("memory_breakdown", &QAQMCEngine::get_memory_breakdown)
        .def_property_readonly("compact_op_sites", &QAQMCEngine::uses_compact_op_sites)
        .def_property_readonly("compact_alias_indices", &QAQMCEngine::uses_compact_alias_indices)
        .def_property_readonly("model_data", &QAQMCEngine::get_model_data)
        .def_property_readonly("model_memory_bytes", &QAQMCEngine::get_model_memory_bytes)
        .def_property_readonly("model_use_count", &QAQMCEngine::get_model_use_count)
        .def_property("bond_event_storage",
                      &QAQMCEngine::get_bond_event_storage,
                      &QAQMCEngine::set_bond_event_storage)

#include "fragments/qaqmc_cuda_bridge.inc"

        // Checkpoint support
        .def("get_rng_state", &QAQMCEngine::get_rng_state)
        .def("set_rng_state", &QAQMCEngine::set_rng_state)
        .def("set_op_string", [](QAQMCEngine& self,
                                 py::array_t<int32_t, py::array::c_style |
                                                           py::array::forcecast> types_arr,
                                 py::array_t<int32_t, py::array::c_style |
                                                           py::array::forcecast> sites_arr) {
            auto t = types_arr.request();
            auto s = sites_arr.request();
            if (t.ndim != 1 || s.ndim != 1 || t.shape[0] != s.shape[0])
                throw std::invalid_argument(
                    "set_op_string: types/sites must be equal-length 1D arrays");
            self.set_op_string(static_cast<const int32_t*>(t.ptr),
                               static_cast<const int32_t*>(s.ptr),
                               (qaqmc_slot_t)t.shape[0]);
        })

        // ── Off-diagonal string (X_C) seam support — Phase A ───────────────────
        .def("set_string_sites", [](QAQMCEngine& self, py::list sites_py, std::int64_t m_star) {
            std::vector<int> sites;
            for (auto& x : sites_py) sites.push_back(x.cast<int>());
            self.set_string_sites(sites, m_star);
        },
        py::arg("sites"), py::arg("m_star"),
        "Configure the string site list C and cut position m_star; resets seam_mask to empty")
        .def("set_seam_mask", &QAQMCEngine::set_seam_mask, py::arg("mask"),
        "RAW seam-mask write (bit k <-> string_sites[k]); breaks per-site worldline "
        "closure when a bit changes -- only for callers restoring a recorded "
        "(op string, mask) pair. For sector resets use set_seam_mask_consistent.")
        .def("set_seam_mask_consistent", &QAQMCEngine::set_seam_mask_consistent, py::arg("mask"),
        "Set the seam mask AND repair per-site worldline closure (parity(sigma^x ops) "
        "== seam bit, from the fixed |0...0> tau boundaries) by toggling one single-site "
        "op per changed bit; decorrelate before measuring. Use for trajectory resets.")
        .def_property_readonly("seam_mask", &QAQMCEngine::get_seam_mask)
        .def_property_readonly("m_star", &QAQMCEngine::get_m_star)
        .def_property_readonly("string_sites", &QAQMCEngine::get_string_sites)
        .def_property_readonly("state_at_seam_minus", [](const QAQMCEngine& self) {
            return py::array_t<int32_t>(self.get_state_at_seam_minus().size(),
                                        self.get_state_at_seam_minus().data());
        })
        .def_property_readonly("state_at_seam_plus", [](const QAQMCEngine& self) {
            return py::array_t<int32_t>(self.get_state_at_seam_plus().size(),
                                        self.get_state_at_seam_plus().data());
        })
        .def("recompute_seam_snapshots", &QAQMCEngine::recompute_seam_snapshots,
             "Refresh state_at_seam_minus/plus from current op string without resampling")

        // ── Off-diagonal string (X_C) half-line topology move — Phase B ────────
        .def("build_half_line_proposal", &QAQMCEngine::build_half_line_proposal,
             py::arg("local_index"), py::arg("direction_right"),
             "Read-only: build the half-line proposal for toggling string_sites[local_index]")
        .def("commit_half_line_proposal", &QAQMCEngine::commit_half_line_proposal,
             py::arg("local_index"), py::arg("prop"),
             "Toggle the terminal operator + seam bit for a previously-built valid proposal")
        .def("attempt_string_toggle", &QAQMCEngine::attempt_string_toggle,
             py::arg("local_index"), py::arg("lambda_"),
             "One Metropolis attempt (random direction) for string_sites[local_index] at fixed lambda")
        .def("topology_sweep", &QAQMCEngine::topology_sweep, py::arg("lambda_"),
             "Random-permutation sweep: one attempt_string_toggle per string site")

// The remaining chained registrations are grouped by observable domain. They
// intentionally remain in this translation unit so the py::class_ fluent
// expression and its exact public API stay unchanged.
#include "fragments/qaqmc_profile.inc"
#include "fragments/qaqmc_dimer.inc"
#include "fragments/qaqmc_occupation_sf.inc"
#include "fragments/qaqmc_vbs_ss.inc"
