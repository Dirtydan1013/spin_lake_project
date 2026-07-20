        // ── Occupation structure-factor matrix (sublattice-resolved) ────────
        .def("set_occ_sf_site_map", [](QAQMCEngine& self,
                                       py::array_t<double> cell_R,
                                       py::array_t<int> basis,
                                       py::array_t<int> in_bulk, int n_basis) {
            auto rb = cell_R.request();
            if (rb.ndim != 2) throw std::runtime_error("cell_R must be (N, pos_dim)");
            int N = (int)rb.shape[0]; int pd = (int)rb.shape[1];
            const double* rp = static_cast<const double*>(rb.ptr);
            std::vector<std::vector<double>> R(N, std::vector<double>(pd));
            for (int i = 0; i < N; ++i) for (int d = 0; d < pd; ++d) R[i][d] = rp[i*pd+d];
            auto bb = basis.request(); auto ib = in_bulk.request();
            std::vector<int> bas(static_cast<const int*>(bb.ptr),
                                 static_cast<const int*>(bb.ptr) + bb.shape[0]);
            std::vector<int> ibk(static_cast<const int*>(ib.ptr),
                                 static_cast<const int*>(ib.ptr) + ib.shape[0]);
            self.set_occ_sf_site_map(R, bas, ibk, n_basis);
        },
        py::arg("cell_R"), py::arg("basis"), py::arg("in_bulk_cell"), py::arg("n_basis"),
        "Set per-site cell Bravais position R, sublattice index α, and bulk-cell "
        "membership for the occupation SF matrix.")
        .def("set_occ_sf_q_points", [](QAQMCEngine& self, py::array_t<double> q_arr) {
            auto buf = q_arr.request();
            if (buf.ndim != 2) throw std::runtime_error("q_points must be (n_q, pos_dim)");
            int n_q = (int)buf.shape[0]; int pd = (int)buf.shape[1];
            const double* p = static_cast<const double*>(buf.ptr);
            std::vector<std::vector<double>> q(n_q, std::vector<double>(pd));
            for (int qi = 0; qi < n_q; ++qi) for (int d = 0; d < pd; ++d) q[qi][d] = p[qi*pd+d];
            self.set_occ_sf_q_points(q);
        },
        py::arg("q_points"),
        "Set q-points (n_q, pos_dim) for the occupation SF matrix (phases use cell R).")
        .def("set_occ_sf_point_indices", [](QAQMCEngine& self,
                py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> idx_arr) {
            auto buf = idx_arr.request();
            std::vector<qaqmc_slot_t> idx(
                static_cast<const std::int64_t*>(buf.ptr),
                static_cast<const std::int64_t*>(buf.ptr) + buf.shape[0]);
            self.set_occ_sf_point_indices(idx);
        },
        py::arg("point_indices"),
        "Profile-point indices at which to measure the occupation SF matrix.")
        .def("set_occ2_sf_site_map", [](QAQMCEngine& self,
                                        py::array_t<double> cell_R, py::array_t<int> basis, int n_basis) {
            auto rb = cell_R.request();
            int N = (int)rb.shape[0]; int pd = (int)rb.shape[1];
            const double* rp = static_cast<const double*>(rb.ptr);
            std::vector<std::vector<double>> R(N, std::vector<double>(pd));
            for (int i = 0; i < N; ++i) for (int d = 0; d < pd; ++d) R[i][d] = rp[i*pd+d];
            auto bb = basis.request();
            std::vector<int> bas(static_cast<const int*>(bb.ptr),
                                 static_cast<const int*>(bb.ptr) + bb.shape[0]);
            self.set_occ2_sf_site_map(R, bas, n_basis);
        },
        py::arg("cell_R"), py::arg("basis"), py::arg("n_basis"),
        "Set the second (triangle-pair) unit-cell map for the occ-SF matrix "
        "(basis=-1 marks sites excluded from the triangle tiling).")
        .def_property_readonly("occ2_active", &QAQMCEngine::get_occ2_active)

        .def_property_readonly("n_occ_q_points",  &QAQMCEngine::get_n_occ_q_points)
        .def_property_readonly("occ_n_basis",     &QAQMCEngine::get_occ_n_basis)
        .def_property_readonly("n_occ_sf_points", &QAQMCEngine::get_n_occ_sf_points)
