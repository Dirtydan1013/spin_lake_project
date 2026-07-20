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
            auto d_buf = density_out.mutable_unchecked<1>();

            int n_zg = self.get_n_loop_size_groups();
            int n_cg = self.get_n_string_size_groups();

            // Output arrays: second dim = size group (mean over copies done in C++)
            py::array_t<double> z_l_out2({n_zl,  n_zg});
            py::array_t<double> c_m_l_out2({n_cml, n_cg});
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
                        for (int g = 0; g < n_zg; ++g)
                            z_buf2(idx_z, g) = obs.Z_l_by_size[g];
                        ++idx_z;
                    }
                    if (step % me_cml == 0) {
                        for (int g = 0; g < n_cg; ++g)
                            c_buf2(idx_c, g) = obs.C_m_l_by_size[g];
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
            result["Z_l"]     = z_l_out2;   // (n_zl,  n_loop_size_groups)
            result["C_m_l"]   = c_m_l_out2; // (n_cml, n_string_size_groups)
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
        "Returns dict: density (n_density,), Z_l (n_zl, n_loop_size_groups), C_m_l (n_cml, n_string_size_groups).\n"
        "Z_l / C_m_l are mean over copies within each size group (signed; no |·|).")

        .def("run_profile", [](QAQMCEngine& self, int n_equil, int n_samples,
                               int me_density, int me_zl, int me_cml,
                               int profile_step, int batch_size,
                               py::object progress_callback, int progress_every,
                               int n_snapshots, int occ_nbatch) {
            if (me_density  <= 0) me_density  = 1;
            if (me_zl       <= 0) me_zl       = 1;
            if (me_cml      <= 0) me_cml      = 1;
            if (profile_step   <= 0) profile_step   = 10000;
            if (batch_size     <= 0) batch_size     = 1000;
            if (progress_every <= 0) progress_every = 1;
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

            const std::int64_t M_total = self.get_M_total();
            const py::ssize_t n_points = (py::ssize_t)(M_total / profile_step);
            int n_zg      = self.get_n_loop_size_groups();
            int n_cg      = self.get_n_string_size_groups();
            int n_q       = self.get_n_q_points();  // 0 if SF not configured
            int N         = self.get_N();
            int min_me    = std::min({me_density, me_zl, me_cml});

            // Snapshots: collect full state vectors at requested profile points,
            // up to n_snapshots configs (collected from the first samples).
            int n_snap_pts     = self.get_n_snapshot_points();   // 0 if not configured
            int n_snap_collect = (n_snap_pts > 0 && n_snapshots > 0) ? n_snapshots : 0;
            py::array_t<int8_t> snapshots_out(
                {std::max(n_snap_collect, 1), std::max(n_snap_pts, 1), N});
            auto snap_buf = snapshots_out.mutable_unchecked<3>();
            int snap_count = 0;

            // n_samples is defined for the finest observable.
            // We group samples into batches; each batch stores the mean per size group per point.
            int total_steps = n_samples * min_me;

            // ── Occupation-SF matrices: accumulate outer products s_α s*_β into
            //    `occ_nbatch` coarse super-bins (decoupled from batch_size to keep
            //    the 6×6-matrix storage small). Measured at occ-SF profile points.
            int n_occ_pt = self.get_n_occ_sf_points();
            int n_occ_q  = self.get_n_occ_q_points();
            int nb       = self.get_occ_n_basis();
            bool do_occ  = (n_occ_pt > 0 && n_occ_q > 0 && nb > 0 && occ_nbatch > 0);
            int occ_nb   = do_occ ? occ_nbatch : 1;
            int occ_pt_a = std::max(n_occ_pt, 1);
            int occ_q_a  = std::max(n_occ_q, 1);
            int nb_a     = std::max(nb, 1);
            // 5D matrix outputs (occ_nb, n_occ_pt, n_occ_q, nb, nb) + nprof (occ_nb, n_occ_pt, N)
            py::array_t<double> occ_full_re({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            py::array_t<double> occ_full_im({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            py::array_t<double> occ_bulk_re({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            py::array_t<double> occ_bulk_im({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            py::array_t<double> occ_nprof  ({occ_nb, occ_pt_a, N});
            auto ofr = occ_full_re.mutable_unchecked<5>();
            auto ofi = occ_full_im.mutable_unchecked<5>();
            auto obr = occ_bulk_re.mutable_unchecked<5>();
            auto obi = occ_bulk_im.mutable_unchecked<5>();
            auto onp = occ_nprof.mutable_unchecked<3>();
            // Flat super-bin accumulators
            size_t occ_msz = (size_t)occ_pt_a * occ_q_a * nb_a * nb_a;
            size_t occ_nsz = (size_t)occ_pt_a * N;
            std::vector<double> aFr(do_occ?occ_msz:0,0.0), aFi(do_occ?occ_msz:0,0.0);
            std::vector<double> aBr(do_occ?occ_msz:0,0.0), aBi(do_occ?occ_msz:0,0.0);
            std::vector<double> aN (do_occ?occ_nsz:0,0.0);
            // Second (triangle) unit-cell matrix: single version, same super-bins.
            bool do_occ2 = (do_occ && self.get_occ2_active());
            py::array_t<double> occ2_re({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            py::array_t<double> occ2_im({occ_nb, occ_pt_a, occ_q_a, nb_a, nb_a});
            auto o2r = occ2_re.mutable_unchecked<5>();
            auto o2i = occ2_im.mutable_unchecked<5>();
            std::vector<double> a2r(do_occ2?occ_msz:0,0.0), a2i(do_occ2?occ_msz:0,0.0);
            long occ_count = 0, occ_bin = 0;
            long occ_bin_size = do_occ ? std::max<long>(1, total_steps / occ_nb) : 1;

            py::ssize_t n_batches_d = (total_steps / me_density) / batch_size;
            py::ssize_t n_batches_z = (total_steps / me_zl)  / batch_size;
            py::ssize_t n_batches_c = (total_steps / me_cml) / batch_size;
            if (n_batches_d < 1) n_batches_d = 1;
            if (n_batches_z < 1) n_batches_z = 1;
            if (n_batches_c < 1) n_batches_c = 1;

            // density: batched per point (n_batches_d, n_points) — batch means,
            // same batch_size as Z_l/C_m_l (was per-sample; batched to save storage).
            py::array_t<double> density_out({n_batches_d, n_points});
            auto d_buf = density_out.mutable_unchecked<2>();

            // Z_l, C_m_l: batched per size group (n_batches, n_points, n_size_groups)
            py::array_t<double> z_l_out  ({n_batches_z, n_points, (py::ssize_t)n_zg});
            py::array_t<double> c_m_l_out({n_batches_c, n_points, (py::ssize_t)n_cg});
            auto z_buf = z_l_out.mutable_unchecked<3>();
            auto c_buf = c_m_l_out.mutable_unchecked<3>();

            // SF: batched per q-point (n_batches_sf, n_points, n_q).  Shares the
            // same cadence + batch_size as Z_l (both ride the same forward walk).
            // Allocate at least size 1 to keep numpy happy even if n_q==0.
            py::ssize_t n_q_alloc    = std::max(n_q, 1);
            py::ssize_t n_batches_sf = (n_q > 0) ? n_batches_z : (py::ssize_t)1;
            py::array_t<double> s_q_real_out({n_batches_sf, n_points, n_q_alloc});
            py::array_t<double> s_q_imag_out({n_batches_sf, n_points, n_q_alloc});
            py::array_t<double> s_q_abs1_out({n_batches_sf, n_points, n_q_alloc});
            py::array_t<double> s_q_abs2_out({n_batches_sf, n_points, n_q_alloc});
            py::array_t<double> s_q_abs3_out({n_batches_sf, n_points, n_q_alloc});
            py::array_t<double> s_q_abs4_out({n_batches_sf, n_points, n_q_alloc});
            auto sf_re_buf = s_q_real_out.mutable_unchecked<3>();
            auto sf_im_buf = s_q_imag_out.mutable_unchecked<3>();
            auto sf_a1_buf = s_q_abs1_out.mutable_unchecked<3>();
            auto sf_a2_buf = s_q_abs2_out.mutable_unchecked<3>();
            auto sf_a3_buf = s_q_abs3_out.mutable_unchecked<3>();
            auto sf_a4_buf = s_q_abs4_out.mutable_unchecked<3>();

            // VBS/SS: per-sample scalar at each profile point → batched means of
            // M_vbs, M_ss and their squares (ride the me_zl batch cadence).
            bool do_vbs = (self.get_n_vbs_triangles() > 0);
            py::array_t<double> mvbs_out ({do_vbs?n_batches_z:(py::ssize_t)1, n_points});
            py::array_t<double> mss_out  ({do_vbs?n_batches_z:(py::ssize_t)1, n_points});
            py::array_t<double> mvbs2_out({do_vbs?n_batches_z:(py::ssize_t)1, n_points});
            py::array_t<double> mss2_out ({do_vbs?n_batches_z:(py::ssize_t)1, n_points});
            auto mvb_buf  = mvbs_out.mutable_unchecked<2>();
            auto mss_buf  = mss_out.mutable_unchecked<2>();
            auto mvb2_buf = mvbs2_out.mutable_unchecked<2>();
            auto mss2_buf = mss2_out.mutable_unchecked<2>();
            std::vector<double> mvb_acc (n_points, 0.0), mss_acc (n_points, 0.0);
            std::vector<double> mvb2_acc(n_points, 0.0), mss2_acc(n_points, 0.0);

            // Batch accumulators: [n_groups][n_points]
            std::vector<std::vector<double>> z_acc(n_zg, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> c_acc(n_cg, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_re_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_im_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_a1_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_a2_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_a3_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<std::vector<double>> sf_a4_acc(n_q, std::vector<double>(n_points, 0.0));
            std::vector<double> d_acc(n_points, 0.0);
            int d_in_batch = 0, batch_d = 0;
            int z_in_batch = 0, c_in_batch = 0;
            int batch_z = 0, batch_c = 0;

            // occ-SF super-bin flush: write current accumulators (mean over occ_count
            // occ-samples) into super-bin occ_bin, then reset.
            auto occ_flush = [&]() {
                if (!do_occ || occ_count <= 0 || occ_bin >= occ_nb) return;
                double inv = 1.0 / (double)occ_count;
                for (int pt = 0; pt < n_occ_pt; ++pt)
                  for (int qi = 0; qi < n_occ_q; ++qi)
                    for (int a = 0; a < nb; ++a)
                      for (int b = 0; b < nb; ++b) {
                        size_t k = (((size_t)pt*occ_q_a + qi)*nb_a + a)*nb_a + b;
                        ofr(occ_bin,pt,qi,a,b) = aFr[k]*inv; aFr[k]=0.0;
                        ofi(occ_bin,pt,qi,a,b) = aFi[k]*inv; aFi[k]=0.0;
                        obr(occ_bin,pt,qi,a,b) = aBr[k]*inv; aBr[k]=0.0;
                        obi(occ_bin,pt,qi,a,b) = aBi[k]*inv; aBi[k]=0.0;
                        if (do_occ2) { o2r(occ_bin,pt,qi,a,b)=a2r[k]*inv; a2r[k]=0.0;
                                       o2i(occ_bin,pt,qi,a,b)=a2i[k]*inv; a2i[k]=0.0; }
                      }
                for (int pt = 0; pt < n_occ_pt; ++pt)
                  for (int i = 0; i < N; ++i) {
                    size_t k = (size_t)pt*N + i;
                    onp(occ_bin,pt,i) = aN[k]*inv; aN[k]=0.0;
                  }
                ++occ_bin; occ_count = 0;
            };

            for (int i = 0; i < total_steps; ++i) {
                int step = i + 1;
                bool need_d = (step % me_density == 0);
                bool need_z = (step % me_zl      == 0);
                bool need_c = (step % me_cml     == 0);
                const bool need_any = need_d || need_z || need_c;

                // Fused step+measurement: mc_step_profiled captures the profile
                // during its diagonal sweep (the diagonal update never touches
                // off-diagonal ops, so the captured trajectory equals what
                // measure_profile would return for the PREVIOUS completed step
                // — a one-step lag with identical equilibrium statistics) and
                // saves a full O(M) measurement sweep per sample.
                QAQMCEngine::ProfileObservables prof;
                if (need_any) {
                    py::gil_scoped_release release;
                    prof = self.mc_step_profiled(profile_step);
                } else {
                    py::gil_scoped_release release;
                    self.mc_step();
                }

                if (need_any) {
                    // Collect full-state snapshots spread evenly across the run:
                    // the k-th snapshot is taken at step ≈ (k+1)/n_snap_collect of
                    // total_steps, so all are post-thermalization and mutually
                    // decorrelated (with n_snap_collect=1 → the final sample).
                    if (n_snap_collect > 0 && snap_count < n_snap_collect &&
                        (int64_t)step * n_snap_collect
                            >= (int64_t)(snap_count + 1) * total_steps) {
                        for (int k = 0; k < n_snap_pts; ++k)
                            for (int i = 0; i < N; ++i)
                                snap_buf(snap_count, k, i) = prof.snapshots[k][i];
                        ++snap_count;
                    }
                    // occ-SF: accumulate outer products s_α s*_β (full + bulk) and
                    // the per-site occupation, into the current super-bin.
                    if (do_occ) {
                        for (int pt = 0; pt < n_occ_pt; ++pt) {
                            const double* fr = prof.occ_s_full_re[pt].data();
                            const double* fi = prof.occ_s_full_im[pt].data();
                            const double* br = prof.occ_s_bulk_re[pt].data();
                            const double* bi = prof.occ_s_bulk_im[pt].data();
                            for (int qi = 0; qi < n_occ_q; ++qi) {
                                size_t vb = (size_t)qi * nb;
                                size_t mb = (((size_t)pt*occ_q_a + qi)*nb_a)*nb_a;
                                for (int a = 0; a < nb; ++a) {
                                    double far=fr[vb+a], fai=fi[vb+a], bar=br[vb+a], bai=bi[vb+a];
                                    for (int b = 0; b < nb; ++b) {
                                        size_t k = mb + (size_t)a*nb_a + b;
                                        double fbr=fr[vb+b], fbi=fi[vb+b];
                                        double bbr=br[vb+b], bbi=bi[vb+b];
                                        aFr[k] += far*fbr + fai*fbi;
                                        aFi[k] += fai*fbr - far*fbi;
                                        aBr[k] += bar*bbr + bai*bbi;
                                        aBi[k] += bai*bbr - bar*bbi;
                                    }
                                }
                            }
                            const int8_t* st = prof.occ_state[pt].data();
                            size_t nbase = (size_t)pt * N;
                            for (int i = 0; i < N; ++i) aN[nbase + i] += st[i];
                            if (do_occ2) {
                                const double* r2 = prof.occ2_s_re[pt].data();
                                const double* i2 = prof.occ2_s_im[pt].data();
                                for (int qi = 0; qi < n_occ_q; ++qi) {
                                    size_t vb = (size_t)qi * nb;
                                    size_t mb = (((size_t)pt*occ_q_a + qi)*nb_a)*nb_a;
                                    for (int a = 0; a < nb; ++a) {
                                        double ar=r2[vb+a], ai=i2[vb+a];
                                        for (int b = 0; b < nb; ++b) {
                                            size_t k = mb + (size_t)a*nb_a + b;
                                            double br=r2[vb+b], bi=i2[vb+b];
                                            a2r[k] += ar*br + ai*bi;
                                            a2i[k] += ai*br - ar*bi;
                                        }
                                    }
                                }
                            }
                        }
                        ++occ_count;
                        if (occ_count >= occ_bin_size && occ_bin < occ_nb - 1) occ_flush();
                    }
                    if (need_d) {
                        for (py::ssize_t k = 0; k < n_points; ++k) d_acc[k] += prof.density[k];
                        ++d_in_batch;
                        if (d_in_batch >= batch_size && batch_d < n_batches_d) {
                            double inv = 1.0 / d_in_batch;
                            for (py::ssize_t k = 0; k < n_points; ++k) {
                                d_buf(batch_d, k) = d_acc[k] * inv;
                                d_acc[k] = 0.0;
                            }
                            ++batch_d;
                            d_in_batch = 0;
                        }
                    }
                    if (need_z) {
                        for (int g = 0; g < n_zg; ++g)
                            for (py::ssize_t pt = 0; pt < n_points; ++pt)
                                z_acc[g][pt] += prof.Z_l_by_size[g][pt];
                        // VBS/SS rides same cadence as Z_l.
                        if (do_vbs)
                            for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                double mv = prof.M_vbs[pt], ms = prof.M_ss[pt];
                                mvb_acc[pt]  += mv;      mss_acc[pt]  += ms;
                                mvb2_acc[pt] += mv * mv; mss2_acc[pt] += ms * ms;
                            }
                        // SF rides same cadence as Z_l.
                        for (int qi = 0; qi < n_q; ++qi)
                            for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                sf_re_acc[qi][pt] += prof.s_q_real[qi][pt];
                                sf_im_acc[qi][pt] += prof.s_q_imag[qi][pt];
                                sf_a1_acc[qi][pt] += prof.s_q_abs1[qi][pt];
                                sf_a2_acc[qi][pt] += prof.s_q_abs2[qi][pt];
                                sf_a3_acc[qi][pt] += prof.s_q_abs3[qi][pt];
                                sf_a4_acc[qi][pt] += prof.s_q_abs4[qi][pt];
                            }
                        ++z_in_batch;
                        if (z_in_batch >= batch_size && batch_z < n_batches_z) {
                            double inv = 1.0 / z_in_batch;
                            for (int g = 0; g < n_zg; ++g)
                                for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                    z_buf(batch_z, pt, g) = z_acc[g][pt] * inv;
                                    z_acc[g][pt] = 0.0;
                                }
                            if (do_vbs)
                                for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                    mvb_buf (batch_z, pt) = mvb_acc[pt]  * inv;
                                    mss_buf (batch_z, pt) = mss_acc[pt]  * inv;
                                    mvb2_buf(batch_z, pt) = mvb2_acc[pt] * inv;
                                    mss2_buf(batch_z, pt) = mss2_acc[pt] * inv;
                                    mvb_acc[pt]=mss_acc[pt]=mvb2_acc[pt]=mss2_acc[pt]=0.0;
                                }
                            for (int qi = 0; qi < n_q; ++qi)
                                for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                    sf_re_buf(batch_z, pt, qi) = sf_re_acc[qi][pt] * inv;
                                    sf_im_buf(batch_z, pt, qi) = sf_im_acc[qi][pt] * inv;
                                    sf_a1_buf(batch_z, pt, qi) = sf_a1_acc[qi][pt] * inv;
                                    sf_a2_buf(batch_z, pt, qi) = sf_a2_acc[qi][pt] * inv;
                                    sf_a3_buf(batch_z, pt, qi) = sf_a3_acc[qi][pt] * inv;
                                    sf_a4_buf(batch_z, pt, qi) = sf_a4_acc[qi][pt] * inv;
                                    sf_re_acc[qi][pt] = 0.0;
                                    sf_im_acc[qi][pt] = 0.0;
                                    sf_a1_acc[qi][pt] = 0.0;
                                    sf_a2_acc[qi][pt] = 0.0;
                                    sf_a3_acc[qi][pt] = 0.0;
                                    sf_a4_acc[qi][pt] = 0.0;
                                }
                            ++batch_z;
                            z_in_batch = 0;
                        }
                    }
                    if (need_c) {
                        for (int g = 0; g < n_cg; ++g)
                            for (py::ssize_t pt = 0; pt < n_points; ++pt)
                                c_acc[g][pt] += prof.C_m_l_by_size[g][pt];
                        ++c_in_batch;
                        if (c_in_batch >= batch_size && batch_c < n_batches_c) {
                            double inv = 1.0 / c_in_batch;
                            for (int g = 0; g < n_cg; ++g)
                                for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                                    c_buf(batch_c, pt, g) = c_acc[g][pt] * inv;
                                    c_acc[g][pt] = 0.0;
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
            // Flush the last occ-SF super-bin (absorbs all remaining occ-samples).
            if (do_occ) occ_flush();
            // Flush remaining partial batch
            if (d_in_batch > 0 && batch_d < n_batches_d) {
                double inv = 1.0 / d_in_batch;
                for (py::ssize_t k = 0; k < n_points; ++k)
                    d_buf(batch_d, k) = d_acc[k] * inv;
            }
            if (z_in_batch > 0 && batch_z < n_batches_z) {
                double inv = 1.0 / z_in_batch;
                for (int g = 0; g < n_zg; ++g)
                    for (py::ssize_t pt = 0; pt < n_points; ++pt)
                        z_buf(batch_z, pt, g) = z_acc[g][pt] * inv;
                if (do_vbs)
                    for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                        mvb_buf (batch_z, pt) = mvb_acc[pt]  * inv;
                        mss_buf (batch_z, pt) = mss_acc[pt]  * inv;
                        mvb2_buf(batch_z, pt) = mvb2_acc[pt] * inv;
                        mss2_buf(batch_z, pt) = mss2_acc[pt] * inv;
                    }
                for (int qi = 0; qi < n_q; ++qi)
                    for (py::ssize_t pt = 0; pt < n_points; ++pt) {
                        sf_re_buf(batch_z, pt, qi) = sf_re_acc[qi][pt] * inv;
                        sf_im_buf(batch_z, pt, qi) = sf_im_acc[qi][pt] * inv;
                        sf_a1_buf(batch_z, pt, qi) = sf_a1_acc[qi][pt] * inv;
                        sf_a2_buf(batch_z, pt, qi) = sf_a2_acc[qi][pt] * inv;
                        sf_a3_buf(batch_z, pt, qi) = sf_a3_acc[qi][pt] * inv;
                        sf_a4_buf(batch_z, pt, qi) = sf_a4_acc[qi][pt] * inv;
                    }
            }
            if (c_in_batch > 0 && batch_c < n_batches_c) {
                double inv = 1.0 / c_in_batch;
                for (int g = 0; g < n_cg; ++g)
                    for (py::ssize_t pt = 0; pt < n_points; ++pt)
                        c_buf(batch_c, pt, g) = c_acc[g][pt] * inv;
            }

            // p-index array (slot indices — int64, they exceed int32 at large M)
            py::array_t<std::int64_t> p_indices(n_points);
            auto p_buf = p_indices.mutable_unchecked<1>();
            for (py::ssize_t k = 0; k < n_points; ++k)
                p_buf(k) = (std::int64_t)(k + 1) * profile_step - 1;

            py::dict result;
            result["density"]    = density_out;  // (n_density, n_points)
            result["Z_l"]        = z_l_out;       // (n_batches_z, n_points, n_loop_size_groups)
            result["C_m_l"]      = c_m_l_out;     // (n_batches_c, n_points, n_string_size_groups)
            result["p_indices"]  = p_indices;
            result["batch_size"] = py::int_(batch_size);
            if (n_q > 0) {
                result["s_q_real"] = s_q_real_out;  // (n_batches_z, n_points, n_q)
                result["s_q_imag"] = s_q_imag_out;
                result["s_q_abs1"] = s_q_abs1_out;
                result["s_q_abs2"] = s_q_abs2_out;
                result["s_q_abs3"] = s_q_abs3_out;
                result["s_q_abs4"] = s_q_abs4_out;
                result["n_q"]      = py::int_(n_q);
            }
            if (n_snap_collect > 0) {
                result["snapshots"] = snapshots_out;  // (n_snapshots, n_snap_pts, N) int8
                result["n_snapshots_collected"] = py::int_(snap_count);
            }
            if (do_vbs) {
                // (n_batches_z, n_points) batched means of M and M²
                result["M_vbs"]  = mvbs_out;
                result["M_ss"]   = mss_out;
                result["M_vbs2"] = mvbs2_out;
                result["M_ss2"]  = mss2_out;
            }
            if (do_occ) {
                // (occ_nbatch, n_occ_pt, n_occ_q, n_basis, n_basis) — unconnected ⟨s_α s*_β⟩
                result["occ_S_full_re"] = occ_full_re;
                result["occ_S_full_im"] = occ_full_im;
                result["occ_S_bulk_re"] = occ_bulk_re;
                result["occ_S_bulk_im"] = occ_bulk_im;
                result["occ_nprof"]     = occ_nprof;  // (occ_nbatch, n_occ_pt, N) ⟨n_i⟩ per super-bin
                result["occ_nbatch"]    = py::int_(occ_nb);
                if (do_occ2) {
                    result["occ2_S_re"] = occ2_re;    // (occ_nbatch, n_occ_pt, n_q, nb, nb) triangle unit cell
                    result["occ2_S_im"] = occ2_im;
                }
            }
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
        py::arg("n_snapshots")    = 0,
        py::arg("occ_nbatch")     = 0,
        "Asymmetric profile with batched per-size-group storage.\n"
        "density: (n_density, n_points) per-sample.\n"
        "Z_l:     (n_batches, n_points, n_loop_size_groups)   — batch means of size-group means (signed).\n"
        "C_m_l:   (n_batches, n_points, n_string_size_groups) — batch means of size-group means (signed).")
