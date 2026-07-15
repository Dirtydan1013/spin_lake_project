        // ── Dimer structure factor measurement ─────────────────────────────
        .def("set_dimer_sf_q_points", [](QAQMCEngine& self, py::array_t<double> q_arr) {
            auto buf = q_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("q_points must be a 2D array (n_q, pos_dim)");
            int n_q = (int)buf.shape[0];
            int pos_dim = (int)buf.shape[1];
            const double* p = static_cast<const double*>(buf.ptr);
            std::vector<std::vector<double>> q_points(n_q, std::vector<double>(pos_dim));
            for (int qi = 0; qi < n_q; ++qi)
                for (int d = 0; d < pos_dim; ++d)
                    q_points[qi][d] = p[qi * pos_dim + d];
            self.set_dimer_sf_q_points(q_points);
        },
        py::arg("q_points"),
        "Set q-points (n_q, pos_dim) and precompute cos/sin tables for "
        "Σ_i n_i exp(iq·r_i).")

        .def("set_dimer_sf_measure_deltas", [](QAQMCEngine& self, py::array_t<double> deltas_arr) {
            auto buf = deltas_arr.request();
            std::vector<double> deltas(static_cast<const double*>(buf.ptr),
                                       static_cast<const double*>(buf.ptr) + buf.shape[0]);
            self.set_dimer_sf_measure_deltas(deltas);
        },
        py::arg("deltas"),
        "Set target delta values for dimer SF measurements; engine snaps each "
        "to the nearest forward-ramp slice p ∈ [0, M).")

        .def("set_dimer_sf_measure_p_indices", [](QAQMCEngine& self, py::array_t<int> p_arr) {
            auto buf = p_arr.request();
            std::vector<int> p_idx(static_cast<const int*>(buf.ptr),
                                   static_cast<const int*>(buf.ptr) + buf.shape[0]);
            self.set_dimer_sf_measure_p_indices(p_idx);
        },
        py::arg("p_indices"),
        "Directly specify forward-ramp slice indices for dimer SF measurements.")

        .def("set_snapshot_point_indices", [](QAQMCEngine& self, py::array_t<int> idx_arr) {
            auto buf = idx_arr.request();
            std::vector<int> idx(static_cast<const int*>(buf.ptr),
                                 static_cast<const int*>(buf.ptr) + buf.shape[0]);
            self.set_snapshot_point_indices(idx);
        },
        py::arg("point_indices"),
        "Request full-state snapshots at the given profile-point indices "
        "(0-based into the n_points grid). Set empty to disable.")
        .def_property_readonly("n_snapshot_points", &QAQMCEngine::get_n_snapshot_points)
        .def_property_readonly("snapshot_point_indices", [](const QAQMCEngine& self) {
            const auto& v = self.get_snapshot_point_indices();
            return py::array_t<int>(v.size(), v.data());
        })
