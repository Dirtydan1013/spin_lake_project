#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "qaqmc_core.hpp"
#include "qaqmc_renyi_core.hpp"
#include "qaqmc_renyi_work_core.hpp"
#include "sse_core.hpp"

namespace py = pybind11;

// Parse an optional (n_box, dim) periodic box-vectors array into a raw pointer.
// Returns nullptr / n_box=0 when the argument is None (open boundary).  The
// backing array is kept alive in `holder`, which the caller must keep in scope
// until the engine constructor (which copies distances out) has run.
static const double* parse_box(py::object obj, int& n_box,
                               py::array_t<double>& holder) {
    n_box = 0;
    if (obj.is_none()) return nullptr;
    holder = obj.cast<py::array_t<double>>();
    auto buf = holder.request();
    if (buf.ndim != 2)
        throw std::runtime_error("box_vectors must be a 2D array (n_box, dim)");
    n_box = static_cast<int>(buf.shape[0]);
    return static_cast<const double*>(buf.ptr);
}

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

    py::class_<QAQMCEngine::HalfLineProposal>(m, "HalfLineProposal")
        .def_readonly("valid", &QAQMCEngine::HalfLineProposal::valid)
        .def_readonly("terminal_p", &QAQMCEngine::HalfLineProposal::terminal_p)
        .def_readonly("log_physical_ratio", &QAQMCEngine::HalfLineProposal::log_physical_ratio);

    py::class_<QAQMCEngine>(m, "QAQMCEngine")
        .def(py::init([](int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
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
        .def_property_readonly("n_loop_size_groups",   &QAQMCEngine::get_n_loop_size_groups)
        .def_property_readonly("n_string_size_groups", &QAQMCEngine::get_n_string_size_groups)

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

        // ── Off-diagonal string (X_C) seam support — Phase A ───────────────────
        .def("set_string_sites", [](QAQMCEngine& self, py::list sites_py, int m_star) {
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
                self.mc_step();
                if (has_cb && (((i + 1) % progress_every) == 0 || (i + 1) == n_equil))
                    progress_callback(i + 1, n_equil, "equil");
            }

            int M_total   = self.get_M_total();
            int n_points  = M_total / profile_step;
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

            int n_batches_d = (total_steps / me_density) / batch_size;
            int n_batches_z = (total_steps / me_zl)  / batch_size;
            int n_batches_c = (total_steps / me_cml) / batch_size;
            if (n_batches_d < 1) n_batches_d = 1;
            if (n_batches_z < 1) n_batches_z = 1;
            if (n_batches_c < 1) n_batches_c = 1;

            // density: batched per point (n_batches_d, n_points) — batch means,
            // same batch_size as Z_l/C_m_l (was per-sample; batched to save storage).
            py::array_t<double> density_out({n_batches_d, n_points});
            auto d_buf = density_out.mutable_unchecked<2>();

            // Z_l, C_m_l: batched per size group (n_batches, n_points, n_size_groups)
            py::array_t<double> z_l_out  ({n_batches_z, n_points, n_zg});
            py::array_t<double> c_m_l_out({n_batches_c, n_points, n_cg});
            auto z_buf = z_l_out.mutable_unchecked<3>();
            auto c_buf = c_m_l_out.mutable_unchecked<3>();

            // SF: batched per q-point (n_batches_sf, n_points, n_q).  Shares the
            // same cadence + batch_size as Z_l (both ride the same forward walk).
            // Allocate at least size 1 to keep numpy happy even if n_q==0.
            int n_q_alloc       = std::max(n_q, 1);
            int n_batches_sf    = (n_q > 0) ? n_batches_z : 1;
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
            py::array_t<double> mvbs_out ({do_vbs?n_batches_z:1, n_points});
            py::array_t<double> mss_out  ({do_vbs?n_batches_z:1, n_points});
            py::array_t<double> mvbs2_out({do_vbs?n_batches_z:1, n_points});
            py::array_t<double> mss2_out ({do_vbs?n_batches_z:1, n_points});
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
                    prof = self.mc_step_profiled(profile_step);
                } else {
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
                        for (int k = 0; k < n_points; ++k) d_acc[k] += prof.density[k];
                        ++d_in_batch;
                        if (d_in_batch >= batch_size && batch_d < n_batches_d) {
                            double inv = 1.0 / d_in_batch;
                            for (int k = 0; k < n_points; ++k) {
                                d_buf(batch_d, k) = d_acc[k] * inv;
                                d_acc[k] = 0.0;
                            }
                            ++batch_d;
                            d_in_batch = 0;
                        }
                    }
                    if (need_z) {
                        for (int g = 0; g < n_zg; ++g)
                            for (int pt = 0; pt < n_points; ++pt)
                                z_acc[g][pt] += prof.Z_l_by_size[g][pt];
                        // VBS/SS rides same cadence as Z_l.
                        if (do_vbs)
                            for (int pt = 0; pt < n_points; ++pt) {
                                double mv = prof.M_vbs[pt], ms = prof.M_ss[pt];
                                mvb_acc[pt]  += mv;      mss_acc[pt]  += ms;
                                mvb2_acc[pt] += mv * mv; mss2_acc[pt] += ms * ms;
                            }
                        // SF rides same cadence as Z_l.
                        for (int qi = 0; qi < n_q; ++qi)
                            for (int pt = 0; pt < n_points; ++pt) {
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
                                for (int pt = 0; pt < n_points; ++pt) {
                                    z_buf(batch_z, pt, g) = z_acc[g][pt] * inv;
                                    z_acc[g][pt] = 0.0;
                                }
                            if (do_vbs)
                                for (int pt = 0; pt < n_points; ++pt) {
                                    mvb_buf (batch_z, pt) = mvb_acc[pt]  * inv;
                                    mss_buf (batch_z, pt) = mss_acc[pt]  * inv;
                                    mvb2_buf(batch_z, pt) = mvb2_acc[pt] * inv;
                                    mss2_buf(batch_z, pt) = mss2_acc[pt] * inv;
                                    mvb_acc[pt]=mss_acc[pt]=mvb2_acc[pt]=mss2_acc[pt]=0.0;
                                }
                            for (int qi = 0; qi < n_q; ++qi)
                                for (int pt = 0; pt < n_points; ++pt) {
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
                            for (int pt = 0; pt < n_points; ++pt)
                                c_acc[g][pt] += prof.C_m_l_by_size[g][pt];
                        ++c_in_batch;
                        if (c_in_batch >= batch_size && batch_c < n_batches_c) {
                            double inv = 1.0 / c_in_batch;
                            for (int g = 0; g < n_cg; ++g)
                                for (int pt = 0; pt < n_points; ++pt) {
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
                for (int k = 0; k < n_points; ++k)
                    d_buf(batch_d, k) = d_acc[k] * inv;
            }
            if (z_in_batch > 0 && batch_z < n_batches_z) {
                double inv = 1.0 / z_in_batch;
                for (int g = 0; g < n_zg; ++g)
                    for (int pt = 0; pt < n_points; ++pt)
                        z_buf(batch_z, pt, g) = z_acc[g][pt] * inv;
                if (do_vbs)
                    for (int pt = 0; pt < n_points; ++pt) {
                        mvb_buf (batch_z, pt) = mvb_acc[pt]  * inv;
                        mss_buf (batch_z, pt) = mss_acc[pt]  * inv;
                        mvb2_buf(batch_z, pt) = mvb2_acc[pt] * inv;
                        mss2_buf(batch_z, pt) = mss2_acc[pt] * inv;
                    }
                for (int qi = 0; qi < n_q; ++qi)
                    for (int pt = 0; pt < n_points; ++pt) {
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
                    for (int pt = 0; pt < n_points; ++pt)
                        c_buf(batch_c, pt, g) = c_acc[g][pt] * inv;
            }

            // p-index array
            py::array_t<int> p_indices(n_points);
            auto p_buf = p_indices.mutable_unchecked<1>();
            for (int k = 0; k < n_points; ++k)
                p_buf(k) = (k + 1) * profile_step - 1;

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
        .def("set_occ_sf_point_indices", [](QAQMCEngine& self, py::array_t<int> idx_arr) {
            auto buf = idx_arr.request();
            std::vector<int> idx(static_cast<const int*>(buf.ptr),
                                 static_cast<const int*>(buf.ptr) + buf.shape[0]);
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
            return py::array_t<int>(v.size(), v.data());
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
            py::array_t<int>    p_idx_out(p_vec.size(), p_vec.data());
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
