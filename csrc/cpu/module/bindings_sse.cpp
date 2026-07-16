#include "bindings.hpp"
#include "cpu/sse_core.hpp"

void bind_sse(py::module_& m) {
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
                         int neighbor_cutoff, py::object box_vectors) {
            auto buf = pos_arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("pos must be a 2D array (N, dim)");
            int pos_dim = (int)buf.shape[1];
            const double* pos_ptr = static_cast<const double*>(buf.ptr);
            int n_box; py::array_t<double> box_holder;
            const double* box_ptr = parse_box(box_vectors, n_box, box_holder);
            return new SSEEngine(N, Omega, delta, Rb, beta, epsilon, seed,
                                  pos_ptr, pos_dim, neighbor_cutoff, box_ptr, n_box);
        }),
        py::arg("N"), py::arg("Omega"), py::arg("delta"), py::arg("Rb"),
        py::arg("beta"), py::arg("epsilon") = 0.01, py::arg("seed") = 42,
        py::arg("pos"), py::arg("neighbor_cutoff") = -1,
        py::arg("box_vectors") = py::none())

        .def("mc_step", &SSEEngine::mc_step,
             "Run one diagonal update + cluster update + adjust_M")

        .def("run", [](SSEEngine& self, int n_equil, int n_samples,
                       py::object progress_callback, int progress_every,
                       int n_snapshots, bool measure_chi_f) {
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

            // ── Diagonal observables (QAQMC-profile parity, all optional) ────
            const auto& dg = self.diag_obs;
            const int N = self.get_N();
            const int n_zg = dg.n_loop_groups();
            const int n_cg = dg.n_string_groups();
            const bool do_vbs  = dg.n_vbs_triangles() > 0;
            const bool do_bulk = dg.has_bulk();
            const bool do_occ  = dg.occ_ready() && n_samples > 0;
            const bool do_occ2 = do_occ && dg.occ2_active();
            const int n_q = dg.n_occ_q(), nb = dg.occ_n_basis();
            const int n_snap = (n_snapshots > 0 && n_samples > 0) ? n_snapshots : 0;

            py::array_t<double> z_l_out ({std::max(n_samples,1), std::max(n_zg,1)});
            py::array_t<double> c_m_out ({std::max(n_samples,1), std::max(n_cg,1)});
            py::array_t<double> mvbs_out(std::max(do_vbs ? n_samples : 1, 1));
            py::array_t<double> mss_out (std::max(do_vbs ? n_samples : 1, 1));
            py::array_t<double> dbulk_out(std::max(do_bulk ? n_samples : 1, 1));
            auto z_buf  = z_l_out.mutable_unchecked<2>();
            auto c_buf  = c_m_out.mutable_unchecked<2>();
            auto mv_buf = mvbs_out.mutable_unchecked<1>();
            auto ms_buf = mss_out.mutable_unchecked<1>();
            auto db_buf = dbulk_out.mutable_unchecked<1>();
            std::vector<double> z_row(std::max(n_zg, 1)), c_row(std::max(n_cg, 1));

            // occ-SF: mean over this run() call of s_a(q) s_b(q)* (UNCONNECTED,
            // unnormalised — one super-bin per call, same as one profile chunk)
            // plus the mean occupation profile for the connected subtraction.
            const size_t vlen = (size_t)std::max(n_q, 1) * std::max(nb, 1);
            const size_t mlen = vlen * std::max(nb, 1);
            std::vector<double> sfr(do_occ ? vlen : 0), sfi(do_occ ? vlen : 0);
            std::vector<double> sbr(do_occ ? vlen : 0), sbi(do_occ ? vlen : 0);
            std::vector<double> s2r(do_occ2 ? vlen : 0), s2i(do_occ2 ? vlen : 0);
            std::vector<double> aFr(do_occ ? mlen : 0, 0.0), aFi(do_occ ? mlen : 0, 0.0);
            std::vector<double> aBr(do_occ ? mlen : 0, 0.0), aBi(do_occ ? mlen : 0, 0.0);
            std::vector<double> a2r(do_occ2 ? mlen : 0, 0.0), a2i(do_occ2 ? mlen : 0, 0.0);
            std::vector<double> aN(do_occ ? (size_t)N : 0, 0.0);

            py::array_t<int8_t> snap_out({std::max(n_snap, 1), N});
            auto snap_buf = snap_out.mutable_unchecked<2>();
            int snap_count = 0;

            // chi_F (WLT) half-string sums, per sample
            py::array_t<double> chi_gl_out(std::max(measure_chi_f ? n_samples : 1, 1));
            py::array_t<double> chi_gr_out(std::max(measure_chi_f ? n_samples : 1, 1));
            auto gl_buf = chi_gl_out.mutable_unchecked<1>();
            auto gr_buf = chi_gr_out.mutable_unchecked<1>();

            for (int i = 0; i < n_samples; ++i) {
                self.mc_step();
                e_buf(i) = self.measure_energy();
                d_buf(i) = self.measure_density();
                m_buf(i) = self.measure_mz();
                n_buf(i) = self.get_n_ops();
                if (measure_chi_f) {
                    double gl, gr;
                    self.measure_chi_f_terms(gl, gr);
                    gl_buf(i) = gl; gr_buf(i) = gr;
                }

                const int32_t* st = self.get_state().data();
                if (n_zg > 0) {
                    dg.measure_loops(st, z_row.data());
                    for (int g = 0; g < n_zg; ++g) z_buf(i, g) = z_row[g];
                }
                if (n_cg > 0) {
                    dg.measure_strings(st, c_row.data());
                    for (int g = 0; g < n_cg; ++g) c_buf(i, g) = c_row[g];
                }
                if (do_vbs) {
                    double mv, ms;
                    dg.measure_vbs_ss(st, mv, ms);
                    mv_buf(i) = mv; ms_buf(i) = ms;
                }
                if (do_bulk) db_buf(i) = dg.measure_density_bulk(st);

                if (do_occ) {
                    dg.measure_occ_s(st, sfr.data(), sfi.data(), sbr.data(), sbi.data(),
                                     do_occ2 ? s2r.data() : nullptr,
                                     do_occ2 ? s2i.data() : nullptr);
                    for (int qi = 0; qi < n_q; ++qi) {
                        const size_t vb = (size_t)qi * nb;
                        const size_t mb = (size_t)qi * nb * nb;
                        for (int a = 0; a < nb; ++a) {
                            const double far = sfr[vb+a], fai = sfi[vb+a];
                            const double bar = sbr[vb+a], bai = sbi[vb+a];
                            for (int b = 0; b < nb; ++b) {
                                const size_t k = mb + (size_t)a * nb + b;
                                aFr[k] += far*sfr[vb+b] + fai*sfi[vb+b];
                                aFi[k] += fai*sfr[vb+b] - far*sfi[vb+b];
                                aBr[k] += bar*sbr[vb+b] + bai*sbi[vb+b];
                                aBi[k] += bai*sbr[vb+b] - bar*sbi[vb+b];
                                if (do_occ2) {
                                    a2r[k] += s2r[vb+a]*s2r[vb+b] + s2i[vb+a]*s2i[vb+b];
                                    a2i[k] += s2i[vb+a]*s2r[vb+b] - s2r[vb+a]*s2i[vb+b];
                                }
                            }
                        }
                    }
                    for (int s = 0; s < N; ++s) aN[s] += st[s];
                }

                // Snapshots spread evenly across the run (same rule as the
                // QAQMC profile bindings).
                if (n_snap > 0 && snap_count < n_snap &&
                    (int64_t)(i + 1) * n_snap >= (int64_t)(snap_count + 1) * n_samples) {
                    for (int s = 0; s < N; ++s) snap_buf(snap_count, s) = (int8_t)st[s];
                    ++snap_count;
                }

                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_samples))
                    progress_callback(i + 1, n_samples, "sample");
            }

            py::dict result;
            result["energies"]  = energies;
            result["densities"] = densities;
            result["mz"]        = mz;
            result["n_ops"]     = n_ops_arr;
            if (n_zg > 0)  result["Z_l"]   = z_l_out;
            if (n_cg > 0)  result["C_m_l"] = c_m_out;
            if (do_vbs) { result["M_vbs"] = mvbs_out; result["M_ss"] = mss_out; }
            if (do_bulk)   result["density_bulk"] = dbulk_out;
            if (n_snap > 0) result["snapshots"] = snap_out;
            if (measure_chi_f) {
                result["chi_gl"] = chi_gl_out;
                result["chi_gr"] = chi_gr_out;
            }
            if (do_occ && n_samples > 0) {
                const double inv = 1.0 / (double)n_samples;
                auto pack = [&](std::vector<double>& acc) {
                    py::array_t<double> out({std::max(n_q,1), nb, nb});
                    auto ob = out.mutable_unchecked<3>();
                    for (int qi = 0; qi < n_q; ++qi)
                        for (int a = 0; a < nb; ++a)
                            for (int b = 0; b < nb; ++b)
                                ob(qi, a, b) = acc[((size_t)qi*nb + a)*nb + b] * inv;
                    return out;
                };
                result["occ_S_full_re"] = pack(aFr);
                result["occ_S_full_im"] = pack(aFi);
                result["occ_S_bulk_re"] = pack(aBr);
                result["occ_S_bulk_im"] = pack(aBi);
                if (do_occ2) {
                    result["occ2_S_re"] = pack(a2r);
                    result["occ2_S_im"] = pack(a2i);
                }
                py::array_t<double> nprof(N);
                auto np_buf = nprof.mutable_unchecked<1>();
                for (int s = 0; s < N; ++s) np_buf(s) = aN[s] * inv;
                result["occ_nprof"] = nprof;
            }
            return result;
        },
        py::arg("n_equil"), py::arg("n_samples"),
        py::arg("progress_callback") = py::none(),
        py::arg("progress_every") = 1000,
        py::arg("n_snapshots") = 0,
        py::arg("measure_chi_f") = false,
        "Run equilibration + sampling.\n\n"
        "Returns\n-------\n"
        "dict with keys 'energies', 'densities', 'mz', 'n_ops' (1D per-sample\n"
        "arrays) plus, when configured via the set_* observable methods:\n"
        "'Z_l'/'C_m_l' (n_samples, n_size_groups), 'M_vbs'/'M_ss',\n"
        "'density_bulk', 'snapshots' (n_snapshots, N) int8, and one occ-SF\n"
        "super-bin per call: 'occ_S_{full,bulk}_{re,im}'/'occ2_S_{re,im}'\n"
        "(n_q, 6, 6) + 'occ_nprof' (N,).  With measure_chi_f=True also\n"
        "'chi_gl'/'chi_gr' (per-sample half-string sums of d ln W/d delta;\n"
        "chi_F = (<gl*gr> - <gl><gr>)/2, Wang-Liu-Troyer PRX 5, 031007).")

        // ── Diagonal observables (QAQMC-profile parity) ─────────────────────
        .def("set_bulk_sites", [](SSEEngine& self, const std::vector<int>& sites) {
            self.diag_obs.set_bulk_sites(sites);
        }, py::arg("bulk_sites"),
        "Restrict density_bulk / occ-SF bulk conventions to these sites")
        .def("set_observable_sites", [](SSEEngine& self,
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
            self.diag_obs.set_observable_sites(loop_sets, string_sets);
        }, py::arg("loop_sets"), py::arg("string_sets"),
        "Set loop/string site sets for Z_l / C_m_l (append A_v vertex sets to "
        "loop_sets — they form the trailing size group, as in the QAQMC profile)")
        .def("set_vbs_triangles", [](SSEEngine& self,
                                     py::array_t<int> corners, py::array_t<int> n1_parity,
                                     py::array_t<int> vbs_sign, py::array_t<int> ss_sign,
                                     int ref00, int ref10) {
            auto to_vec = [](py::array_t<int>& a) {
                auto b = a.request();
                const int* p = static_cast<const int*>(b.ptr);
                return std::vector<int>(p, p + b.shape[0]);
            };
            std::vector<int> c = to_vec(corners), par = to_vec(n1_parity);
            std::vector<int> vs = to_vec(vbs_sign), ss = to_vec(ss_sign);
            self.diag_obs.set_vbs_triangles(c, par, vs, ss, ref00, ref10);
        }, py::arg("corners_flat"), py::arg("n1_parity"), py::arg("vbs_sign"),
           py::arg("ss_sign"), py::arg("ref00"), py::arg("ref10"),
        "Set VBS/SS up-triangles (same construction as the QAQMC profile engine)")
        .def("set_occ_sf_site_map", [](SSEEngine& self,
                                       py::array_t<double> cell_R,
                                       py::array_t<int> basis,
                                       py::array_t<int> in_bulk, int n_basis) {
            auto rb = cell_R.request();
            if (rb.ndim != 2) throw std::runtime_error("cell_R must be (N, pos_dim)");
            int N = (int)rb.shape[0]; int pd = (int)rb.shape[1];
            const double* rp = static_cast<const double*>(rb.ptr);
            std::vector<std::vector<double>> R(N, std::vector<double>(pd));
            for (int i = 0; i < N; ++i)
                for (int d = 0; d < pd; ++d) R[i][d] = rp[i * pd + d];
            auto bb = basis.request(); auto ib = in_bulk.request();
            std::vector<int> bas(static_cast<const int*>(bb.ptr),
                                 static_cast<const int*>(bb.ptr) + bb.shape[0]);
            std::vector<int> ibk(static_cast<const int*>(ib.ptr),
                                 static_cast<const int*>(ib.ptr) + ib.shape[0]);
            self.diag_obs.set_occ_sf_site_map(R, bas, ibk, n_basis);
        }, py::arg("cell_R"), py::arg("basis"), py::arg("in_bulk_cell"), py::arg("n_basis"),
        "Set per-site cell Bravais position R, sublattice index α, and bulk-cell "
        "membership for the occupation SF matrix")
        .def("set_occ_sf_q_points", [](SSEEngine& self, py::array_t<double> q_arr) {
            auto buf = q_arr.request();
            if (buf.ndim != 2) throw std::runtime_error("q_points must be (n_q, dim)");
            int n_q = (int)buf.shape[0]; int pd = (int)buf.shape[1];
            const double* qp = static_cast<const double*>(buf.ptr);
            std::vector<std::vector<double>> q(n_q, std::vector<double>(pd));
            for (int qi = 0; qi < n_q; ++qi)
                for (int d = 0; d < pd; ++d) q[qi][d] = qp[qi * pd + d];
            self.diag_obs.set_occ_sf_q_points(q);
        }, py::arg("q_points"),
        "Set the occ-SF q grid (phases use the cell Bravais positions R)")
        .def("set_occ2_sf_site_map", [](SSEEngine& self,
                                        py::array_t<double> cell_R,
                                        py::array_t<int> basis, int n_basis) {
            auto rb = cell_R.request();
            if (rb.ndim != 2) throw std::runtime_error("cell_R must be (N, pos_dim)");
            int N = (int)rb.shape[0]; int pd = (int)rb.shape[1];
            const double* rp = static_cast<const double*>(rb.ptr);
            std::vector<std::vector<double>> R(N, std::vector<double>(pd));
            for (int i = 0; i < N; ++i)
                for (int d = 0; d < pd; ++d) R[i][d] = rp[i * pd + d];
            auto bb = basis.request();
            std::vector<int> bas(static_cast<const int*>(bb.ptr),
                                 static_cast<const int*>(bb.ptr) + bb.shape[0]);
            self.diag_obs.set_occ2_sf_site_map(R, bas, n_basis);
        }, py::arg("cell_R"), py::arg("basis"), py::arg("n_basis"),
        "Set the second (triangle-pair) occ-SF unit cell; call after set_occ_sf_q_points")
        .def_property_readonly("n_loop_size_groups", [](const SSEEngine& self) {
            return self.diag_obs.n_loop_groups();
        })
        .def_property_readonly("n_string_size_groups", [](const SSEEngine& self) {
            return self.diag_obs.n_string_groups();
        })

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
        .def("measure_chi_f_terms", [](SSEEngine& self) {
            double gl, gr;
            self.measure_chi_f_terms(gl, gr);
            return py::make_tuple(gl, gr);
        }, "(g_left, g_right): half-string sums of d ln W/d delta (chi_F terms)")

        // ── Off-diagonal string X_C (thermal string-work; QAQMC-parity API) ──
        .def("set_string_sites", [](SSEEngine& self, py::list sites_py, int m_star) {
            std::vector<int> sites;
            for (auto& x : sites_py) sites.push_back(x.cast<int>());
            self.set_string_sites(sites, m_star);
        },
        py::arg("sites"), py::arg("m_star") = 0,
        "Configure the string site list C and seam slot m_star (default 0 = "
        "tau=0 seam); resets seam_mask to empty")
        .def("set_seam_mask", &SSEEngine::set_seam_mask, py::arg("mask"),
        "Set which string_sites are seam-active (bit k <-> string_sites[k]); "
        "repairs worldline parity as needed")
        // Alias matching the QAQMCEngine name, so engine-agnostic drivers
        // (QAQMCStringWorkRydberg and its SSE subclass) can call one method
        // for a closure-safe sector reset on either engine.
        .def("set_seam_mask_consistent", &SSEEngine::set_seam_mask, py::arg("mask"),
        "Alias of set_seam_mask (which already repairs worldline parity)")
        .def_property_readonly("seam_mask", &SSEEngine::get_seam_mask)
        .def_property_readonly("m_star", &SSEEngine::get_m_star)
        .def_property_readonly("string_sites", &SSEEngine::get_string_sites)
        .def_property_readonly("state_at_seam_minus", [](const SSEEngine& self) {
            return py::array_t<int32_t>(self.get_state_at_seam_minus().size(),
                                        self.get_state_at_seam_minus().data());
        })
        .def_property_readonly("state_at_seam_plus", [](const SSEEngine& self) {
            return py::array_t<int32_t>(self.get_state_at_seam_plus().size(),
                                        self.get_state_at_seam_plus().data());
        })
        .def("recompute_seam_snapshots", &SSEEngine::recompute_seam_snapshots,
             "Refresh state_at_seam_minus/plus from current state_/op string")
        .def("attempt_string_toggle", &SSEEngine::attempt_string_toggle,
             py::arg("local_index"), py::arg("lambda_"),
             "One Metropolis attempt (random direction, periodic walk) for "
             "string_sites[local_index] at fixed lambda")
        .def("topology_sweep", &SSEEngine::topology_sweep, py::arg("lambda_"),
             "Refresh seam snapshots, then one attempt_string_toggle per "
             "string site in random order")
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

        .def("set_config", [](SSEEngine& self,
                              py::array_t<int32_t> state,
                              py::array_t<int32_t> op_types,
                              py::array_t<int32_t> op_sites) {
            auto sb = state.request();
            auto tb = op_types.request();
            auto ob = op_sites.request();
            if (sb.ndim != 1 || tb.ndim != 1 || ob.ndim != 1)
                throw std::runtime_error("set_config: arrays must be 1D");
            if (tb.shape[0] != ob.shape[0])
                throw std::runtime_error("set_config: op_types/op_sites length mismatch");
            self.set_config(static_cast<const int32_t*>(sb.ptr), (int)sb.shape[0],
                            static_cast<const int32_t*>(tb.ptr),
                            static_cast<const int32_t*>(ob.ptr), (int)tb.shape[0]);
        },
        py::arg("state"), py::arg("op_types"), py::arg("op_sites"),
        "Warm start: install a saved configuration (tau=0 spin state + operator\n"
        "string).  Combined with set_rng_state this resumes the exact chain;\n"
        "with a fresh seed it just skips thermalization.")

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
