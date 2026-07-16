#include "bindings.hpp"
#include "include/qaqmc_renyi_work_core.hpp"

void bind_qaqmc_renyi_work(py::module_& m) {
    // ── QAQMCRenyiWorkEngine ─────────────────────────────────────────────────

    py::class_<QAQMCRenyiWorkEngine::WorkTrajectoryResult>(m, "WorkTrajectoryResult")
        .def_readonly("work",                  &QAQMCRenyiWorkEngine::WorkTrajectoryResult::work)
        .def_readonly("exp_minus_work",        &QAQMCRenyiWorkEngine::WorkTrajectoryResult::exp_minus_work)
        .def_readonly("final_swap_count",      &QAQMCRenyiWorkEngine::WorkTrajectoryResult::final_swap_count)
        .def_readonly("unjoined_at_end_count", &QAQMCRenyiWorkEngine::WorkTrajectoryResult::unjoined_at_end_count)
        .def_readonly("topology_attempts",     &QAQMCRenyiWorkEngine::WorkTrajectoryResult::topology_attempts)
        .def_readonly("topology_accepts",      &QAQMCRenyiWorkEngine::WorkTrajectoryResult::topology_accepts);

    py::class_<QAQMCRenyiWorkEngine::WorkRunResult>(m, "WorkRunResult")
        .def_readonly("mean_exp_minus_work",      &QAQMCRenyiWorkEngine::WorkRunResult::mean_exp_minus_work)
        .def_readonly("delta_s2",                 &QAQMCRenyiWorkEngine::WorkRunResult::delta_s2)
        .def_readonly("work_mean",                &QAQMCRenyiWorkEngine::WorkRunResult::work_mean)
        .def_readonly("work_var",                 &QAQMCRenyiWorkEngine::WorkRunResult::work_var)
        .def_readonly("trajectory_count",         &QAQMCRenyiWorkEngine::WorkRunResult::trajectory_count)
        .def_readonly("total_topology_attempts",  &QAQMCRenyiWorkEngine::WorkRunResult::total_topology_attempts)
        .def_readonly("total_topology_accepts",   &QAQMCRenyiWorkEngine::WorkRunResult::total_topology_accepts)
        .def_readonly("total_unjoined_at_end",    &QAQMCRenyiWorkEngine::WorkRunResult::total_unjoined_at_end)
        .def_property_readonly("work_samples", [](const QAQMCRenyiWorkEngine::WorkRunResult& self) {
            return py::array_t<double>(self.work_samples.size(), self.work_samples.data());
        })
        .def_property_readonly("final_swap_counts", [](const QAQMCRenyiWorkEngine::WorkRunResult& self) {
            return py::array_t<int32_t>(self.final_swap_counts.size(), self.final_swap_counts.data());
        })
        .def_property_readonly("unjoined_counts_per_traj", [](const QAQMCRenyiWorkEngine::WorkRunResult& self) {
            return py::array_t<int32_t>(self.unjoined_counts_per_traj.size(),
                                        self.unjoined_counts_per_traj.data());
        })
        .def_property_readonly("topology_attempts_per_traj", [](const QAQMCRenyiWorkEngine::WorkRunResult& self) {
            return py::array_t<int64_t>(self.topology_attempts_per_traj.size(),
                                        self.topology_attempts_per_traj.data());
        })
        .def_property_readonly("topology_accepts_per_traj", [](const QAQMCRenyiWorkEngine::WorkRunResult& self) {
            return py::array_t<int64_t>(self.topology_accepts_per_traj.size(),
                                        self.topology_accepts_per_traj.data());
        });

    py::class_<QAQMCRenyiWorkEngine>(m, "QAQMCRenyiWorkEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         py::array_t<double> pos_arr,
                         int neighbor_cutoff, int delta_groups,
                         py::object box_vectors) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            int n_box; py::array_t<double> box_holder;
            const double* box_ptr = parse_box(box_vectors, n_box, box_holder);
            return new QAQMCRenyiWorkEngine(N, Omega, delta_min, delta_max, Rb, M,
                                            epsilon, seed, pos_ptr, pos_dim,
                                            neighbor_cutoff, delta_groups,
                                            box_ptr, n_box);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta_min"), py::arg("delta_max"),
        py::arg("Rb"), py::arg("M"), py::arg("epsilon"), py::arg("seed"),
        py::arg("pos"), py::arg("neighbor_cutoff") = -1,
        py::arg("delta_groups") = 600, py::arg("box_vectors") = py::none())

        .def("set_region_pair", [](QAQMCRenyiWorkEngine& self,
                                   py::array_t<uint8_t> A_start, py::array_t<uint8_t> A_end) {
            auto s = A_start.request();
            auto e = A_end.request();
            if (s.shape[0] != e.shape[0])
                throw std::runtime_error("A_start and A_end must have the same length");
            self.set_region_pair(static_cast<const uint8_t*>(s.ptr),
                                 static_cast<const uint8_t*>(e.ptr),
                                 static_cast<int>(s.shape[0]));
        }, py::arg("A_start_mask"), py::arg("A_end_mask"),
        "Set nested region pair (A_start ⊆ A_end). Builds D = A_end \\ A_start, "
        "resets B = ∅, and sets backend A_mask = A_start.")

        .def("set_region", [](QAQMCRenyiWorkEngine& self, py::array_t<uint8_t> A_mask) {
            auto a = A_mask.request();
            self.set_region(static_cast<const uint8_t*>(a.ptr),
                            static_cast<int>(a.shape[0]));
        }, py::arg("A_mask"),
        "Convenience: equivalent to set_region_pair(zeros, A_mask). "
        "Corresponds to the paper's ∅→A case.")

        .def("set_cut", &QAQMCRenyiWorkEngine::set_cut, py::arg("m_star"),
             "Move the entanglement cut (swap boundary) to slice m_star in "
             "[0, M_total]; default M.  Vacuum-resets the operator strings — "
             "call before thermalize()/import_start_config().")
        .def("get_cut", &QAQMCRenyiWorkEngine::get_cut)

        .def("export_start_config", [](const QAQMCRenyiWorkEngine& self) {
            std::vector<int32_t> t0, s0, t1, s1;
            self.export_start_config(t0, s0, t1, s1);
            auto arr = [](const std::vector<int32_t>& v) {
                return py::array_t<int32_t>(v.size(), v.data());
            };
            return py::make_tuple(arr(t0), arr(s0), arr(t1), arr(s1));
        },
        "Warm-start export: (types0, sites0, types1, sites1) of the clean "
        "A_start-sector configuration (the checkpoint if one exists).")

        .def("import_start_config", [](QAQMCRenyiWorkEngine& self,
                                       py::array_t<int32_t> t0, py::array_t<int32_t> s0,
                                       py::array_t<int32_t> t1, py::array_t<int32_t> s1) {
            auto bt0 = t0.request(); auto bs0 = s0.request();
            auto bt1 = t1.request(); auto bs1 = s1.request();
            const auto len = bt0.shape[0];
            if (bs0.shape[0] != len || bt1.shape[0] != len || bs1.shape[0] != len)
                throw std::runtime_error("import_start_config: array length mismatch");
            self.import_start_config(
                static_cast<const int32_t*>(bt0.ptr), static_cast<const int32_t*>(bs0.ptr),
                static_cast<const int32_t*>(bt1.ptr), static_cast<const int32_t*>(bs1.ptr),
                static_cast<int>(len));
        }, py::arg("types0"), py::arg("sites0"), py::arg("types1"), py::arg("sites1"),
        "Warm-start import: install a previously exported A_start-sector "
        "configuration (set_region_pair with the same pair must be called "
        "first).  Seeds the checkpoint chain so thermalize() can be skipped.")

        .def("set_lambda_schedule", [](QAQMCRenyiWorkEngine& self,
                                       py::array_t<double> lambdas) {
            auto b = lambdas.request();
            std::vector<double> v(static_cast<const double*>(b.ptr),
                                  static_cast<const double*>(b.ptr) + b.shape[0]);
            self.set_lambda_schedule(v);
        }, py::arg("lambdas"),
        "Set forward λ schedule (must start at 0, end at 1, monotonic non-decreasing).")

        .def("set_sweeps_per_lambda", &QAQMCRenyiWorkEngine::set_sweeps_per_lambda,
             py::arg("n_topology_sweeps"), py::arg("n_qaqmc_sweeps"),
             "Set per-λ-step sweep counts. v1 paper default = (1, 1).")

        .def("thermalize", &QAQMCRenyiWorkEngine::thermalize, py::arg("n_steps"),
             "Thermalise in the start sector (backend A_mask = A_start, B = ∅).")

        .def("run_trajectory", &QAQMCRenyiWorkEngine::run_trajectory,
             "Run one trajectory; assumes engine is at (λ=0, B=∅, backend.A_mask=A_start).")

        .def("run_trajectories", &QAQMCRenyiWorkEngine::run_trajectories,
             py::arg("n_trajectories"), py::arg("decorrelation_steps"),
             "Run n trajectories with decorrelation_steps in the start sector between, "
             "and aggregate via log-sum-exp into ΔS2 = S2(A_end) - S2(A_start).")

        // Accessors
        .def_property_readonly("N", &QAQMCRenyiWorkEngine::get_N)
        .def_property_readonly("M_total", &QAQMCRenyiWorkEngine::get_M_total)
        .def_property_readonly("D_size", &QAQMCRenyiWorkEngine::get_D_size)
        .def_property_readonly("B_size", &QAQMCRenyiWorkEngine::get_B_size)
        .def_property_readonly("A_start_mask", [](const QAQMCRenyiWorkEngine& self) {
            const auto& v = self.get_A_start_mask();
            return py::array_t<uint8_t>(v.size(), v.data());
        })
        .def_property_readonly("A_end_mask", [](const QAQMCRenyiWorkEngine& self) {
            const auto& v = self.get_A_end_mask();
            return py::array_t<uint8_t>(v.size(), v.data());
        })
        .def_property_readonly("D_mask", [](const QAQMCRenyiWorkEngine& self) {
            const auto& v = self.get_D_mask();
            return py::array_t<uint8_t>(v.size(), v.data());
        })
        .def_property_readonly("B_mask", [](const QAQMCRenyiWorkEngine& self) {
            const auto& v = self.get_B_mask();
            return py::array_t<uint8_t>(v.size(), v.data());
        })
        .def_property_readonly("lambda_schedule", [](const QAQMCRenyiWorkEngine& self) {
            const auto& v = self.get_lambda_schedule();
            return py::array_t<double>(v.size(), v.data());
        })
        // Expose underlying QAQMCRenyiEngine backend for probe / diagnostic use.
        // py::return_value_policy::reference_internal keeps the backend alive
        // as long as the work engine is alive (no copy).
        .def_property_readonly("backend",
            [](QAQMCRenyiWorkEngine& self) -> QAQMCRenyiEngine& { return self.backend(); },
            py::return_value_policy::reference_internal,
            "Underlying QAQMCRenyiEngine (Mode::Work). Use sparingly — direct "
            "mutation can desync work-engine bookkeeping; OK for read-only "
            "timing of mc_step / log_weight_ratio_for_toggle.");
}
