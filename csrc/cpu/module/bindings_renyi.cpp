#include "bindings.hpp"
#include "cpu/qaqmc_renyi_core.hpp"

void bind_qaqmc_renyi(py::module_& m) {
    // ── QAQMCRenyiEngine ─────────────────────────────────────────────────────

    py::class_<QAQMCRenyiEngine>(m, "QAQMCRenyiEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr, int neighbor_cutoff,
                         int delta_groups, py::object box_vectors) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = static_cast<int>(buf.shape[1]);
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            int n_box; py::array_t<double> box_holder;
            const double* box_ptr = parse_box(box_vectors, n_box, box_holder);
            return new QAQMCRenyiEngine(
                N, Omega, delta_min, delta_max, Rb, M, epsilon, seed,
                pos_ptr, pos_dim, neighbor_cutoff, delta_groups, box_ptr, n_box);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon") = 0.01, py::arg("seed") = 42,
        py::arg("pos"), py::arg("neighbor_cutoff") = -1, py::arg("delta_groups") = 0,
        py::arg("box_vectors") = py::none())

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
        // ── Single-bit-toggle primitives for Mode::Work ────────────────────
        .def("set_mode", [](QAQMCRenyiEngine& self, int m) {
            self.set_mode(static_cast<QAQMCRenyiEngine::Mode>(m));
        }, py::arg("mode"),
        "Set engine mode: 0 = PairToggle, 1 = Expanded, 2 = Work.")
        .def("log_weight_ratio_for_toggle",
             &QAQMCRenyiEngine::log_weight_ratio_for_toggle,
             py::arg("site"),
             "log[ Z(A_mask ^ {site}) / Z(A_mask) ] using current A_mask_ and op strings. "
             "Cost ~O(M_total). Used by the work engine for dynamic single-bit proposals.")
        .def("apply_single_bit_toggle",
             &QAQMCRenyiEngine::apply_single_bit_toggle,
             py::arg("site"),
             "Flip A_mask_[site] and reproject affected site ops at p >= M. "
             "Caller is responsible for the Metropolis accept/reject before calling.")
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
        .def("get_operator_counts", [](const QAQMCRenyiEngine& self) {
            auto v = self.get_operator_counts();
            py::array_t<int64_t> arr(3);
            auto out = arr.mutable_unchecked<1>();
            for (ssize_t i = 0; i < 3; ++i) out(i) = v[static_cast<size_t>(i)];
            return arr;
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
        .def_property_readonly("diff_site", &QAQMCRenyiEngine::get_diff_site)
        .def_property_readonly("time_diag", &QAQMCRenyiEngine::get_time_diag)
        .def_property_readonly("time_clus_build", &QAQMCRenyiEngine::get_time_clus_build)
        .def_property_readonly("time_clus_sweep", &QAQMCRenyiEngine::get_time_clus_sweep)
        .def_property_readonly("time_topology", &QAQMCRenyiEngine::get_time_topology)
        .def_property_readonly("time_ensemble", &QAQMCRenyiEngine::get_time_ensemble)
        .def_property_readonly("mc_steps", &QAQMCRenyiEngine::get_mc_steps)
        .def_property_readonly("diag_update_slots", &QAQMCRenyiEngine::get_diag_update_slots)
        .def_property_readonly("diag_proposal_attempts", &QAQMCRenyiEngine::get_diag_proposal_attempts)
        .def_property_readonly("diag_site_proposals", &QAQMCRenyiEngine::get_diag_site_proposals)
        .def_property_readonly("diag_bond_proposals", &QAQMCRenyiEngine::get_diag_bond_proposals)
        .def_property_readonly("diag_bond_accepts", &QAQMCRenyiEngine::get_diag_bond_accepts)
        .def("reset_timers", &QAQMCRenyiEngine::reset_timers);
}
