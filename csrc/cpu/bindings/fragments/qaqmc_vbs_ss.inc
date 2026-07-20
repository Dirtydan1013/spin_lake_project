        // ── VBS / SS order parameters ───────────────────────────────────────
        .def("set_vbs_triangles", [](QAQMCEngine& self,
                                     py::array_t<int> corners, py::array_t<int> n1_parity,
                                     py::array_t<int> vbs_sign, py::array_t<int> ss_sign,
                                     int ref00, int ref10) {
            auto cb = corners.request();
            std::vector<int> cf(static_cast<const int*>(cb.ptr),
                                static_cast<const int*>(cb.ptr) + cb.size);
            auto pb = n1_parity.request();
            std::vector<int> par(static_cast<const int*>(pb.ptr),
                                 static_cast<const int*>(pb.ptr) + pb.shape[0]);
            auto vb = vbs_sign.request();
            std::vector<int> vs(static_cast<const int*>(vb.ptr),
                                static_cast<const int*>(vb.ptr) + vb.shape[0]);
            auto sb = ss_sign.request();
            std::vector<int> ss(static_cast<const int*>(sb.ptr),
                                static_cast<const int*>(sb.ptr) + sb.shape[0]);
            self.set_vbs_triangles(cf, par, vs, ss, ref00, ref10);
        },
        py::arg("corners"), py::arg("n1_parity"), py::arg("vbs_sign"), py::arg("ss_sign"),
        py::arg("ref00"), py::arg("ref10"),
        "Configure up-triangles for the VBS/SS order parameters (paper Eq. 5-6).")
        .def_property_readonly("n_vbs_triangles", &QAQMCEngine::get_n_vbs_triangles)

        .def_property_readonly("n_q_points",            &QAQMCEngine::get_n_q_points)
        .def_property_readonly("n_dimer_measure_points",&QAQMCEngine::get_n_dimer_measure_points)
        .def_property_readonly("dimer_p_indices", [](const QAQMCEngine& self) {
            const auto& v = self.get_dimer_p_indices();
            return py::array_t<std::int64_t>(v.size(), v.data());
        })
        .def_property_readonly("dimer_deltas_used", [](const QAQMCEngine& self) {
            const auto& v = self.get_dimer_deltas_used();
            return py::array_t<double>(v.size(), v.data());
        })

        .def("run_dimer_sf", [](QAQMCEngine& self, int n_equil, int n_samples,
                                int batch_size,
                                py::object progress_callback, int progress_every) {
            if (batch_size <= 0) batch_size = 1000;
            if (progress_every <= 0) progress_every = 1;
            const bool has_cb = !progress_callback.is_none();

            int n_p = self.get_n_dimer_measure_points();
            int n_q = self.get_n_q_points();
            if (n_p == 0 || n_q == 0)
                throw std::runtime_error("run_dimer_sf: set q-points + measure deltas first");

            // Equilibration.
            for (int i = 0; i < n_equil; ++i) {
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            int n_batches = n_samples / batch_size;
            if (n_batches < 1) n_batches = 1;

            // Outputs: batch-averaged.  Density per batch per measure point;
            // s_q components per batch per (measure_p, q).
            py::array_t<double> density_out({n_batches, n_p});
            py::array_t<double> s_re_out({n_batches, n_p, n_q});
            py::array_t<double> s_im_out({n_batches, n_p, n_q});
            py::array_t<double> s_a2_out({n_batches, n_p, n_q});
            auto d_buf  = density_out.mutable_unchecked<2>();
            auto re_buf = s_re_out.mutable_unchecked<3>();
            auto im_buf = s_im_out.mutable_unchecked<3>();
            auto a2_buf = s_a2_out.mutable_unchecked<3>();

            std::vector<double> dens_acc(n_p, 0.0);
            std::vector<double> re_acc(n_p * n_q, 0.0);
            std::vector<double> im_acc(n_p * n_q, 0.0);
            std::vector<double> a2_acc(n_p * n_q, 0.0);

            int in_batch = 0;
            int batch = 0;
            for (int s = 0; s < n_samples; ++s) {
                self.mc_step();
                auto sample = self.measure_dimer_sf();
                for (int pi = 0; pi < n_p; ++pi) dens_acc[pi] += sample.density[pi];
                for (size_t k = 0; k < re_acc.size(); ++k) {
                    re_acc[k] += sample.s_q_real[k];
                    im_acc[k] += sample.s_q_imag[k];
                    a2_acc[k] += sample.s_q_abs2[k];
                }
                ++in_batch;

                if (in_batch >= batch_size && batch < n_batches) {
                    double inv = 1.0 / in_batch;
                    for (int pi = 0; pi < n_p; ++pi) {
                        d_buf(batch, pi) = dens_acc[pi] * inv;
                        dens_acc[pi] = 0.0;
                        for (int qi = 0; qi < n_q; ++qi) {
                            size_t k = (size_t)pi * n_q + qi;
                            re_buf(batch, pi, qi) = re_acc[k] * inv;
                            im_buf(batch, pi, qi) = im_acc[k] * inv;
                            a2_buf(batch, pi, qi) = a2_acc[k] * inv;
                            re_acc[k] = 0.0;
                            im_acc[k] = 0.0;
                            a2_acc[k] = 0.0;
                        }
                    }
                    ++batch;
                    in_batch = 0;
                }

                if (has_cb && (((s + 1) % progress_every) == 0 || (s + 1) == n_samples))
                    progress_callback(s + 1, n_samples, "sample");
            }

            // Flush partial last batch (if any) into the final slot.
            if (in_batch > 0 && batch < n_batches) {
                double inv = 1.0 / in_batch;
                for (int pi = 0; pi < n_p; ++pi) {
                    d_buf(batch, pi) = dens_acc[pi] * inv;
                    for (int qi = 0; qi < n_q; ++qi) {
                        size_t k = (size_t)pi * n_q + qi;
                        re_buf(batch, pi, qi) = re_acc[k] * inv;
                        im_buf(batch, pi, qi) = im_acc[k] * inv;
                        a2_buf(batch, pi, qi) = a2_acc[k] * inv;
                    }
                }
            }

            // Also return the actual slice indices and matched delta values.
            const auto& p_vec = self.get_dimer_p_indices();
            const auto& d_vec = self.get_dimer_deltas_used();
            py::array_t<std::int64_t> p_idx_out(p_vec.size(), p_vec.data());
            py::array_t<double> delta_out(d_vec.size(), d_vec.data());

            py::dict result;
            result["density"]      = density_out;
            result["s_q_real"]     = s_re_out;
            result["s_q_imag"]     = s_im_out;
            result["s_q_abs2"]     = s_a2_out;
            result["p_indices"]    = p_idx_out;
            result["deltas_used"]  = delta_out;
            result["batch_size"]   = py::int_(batch_size);
            return result;
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("batch_size") = 1000,
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        "Run sampling and measure dimer structure factor at the pre-set "
        "(q-points × measure deltas).  Returns dict with batch-averaged "
        "<Re(s_q)>, <Im(s_q)>, <|s_q|²>, density per (batch, measure_p, q).\n"
        "The site sum in s_q = Σ_i n_i exp(iq·r_i) is restricted to "
        "bulk_sites (set via set_bulk_sites) if non-empty; otherwise all N "
        "sites are summed.  This matches the density convention so boundary "
        "atoms can be excluded the same way.\n"
        "Connected S_d(q) per measure_p: "
        "<|s_q|²> - |<s_q>|², then divide by N_d (caller).")

        // Profiling
        .def_property_readonly("time_diag", &QAQMCEngine::get_time_diag)
        .def_property_readonly("time_clus", &QAQMCEngine::get_time_clus)
        .def_property_readonly("mc_steps", &QAQMCEngine::get_mc_steps)
        .def("reset_timers", &QAQMCEngine::reset_timers);
}
