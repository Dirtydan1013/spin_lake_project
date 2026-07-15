#include "bindings_cuda.cuh"
#include "qaqmc_cuda_diagonal.cuh"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace qaqmc_cuda::bindings {

void bind_diagonal_engine(py::module_& m) {
    py::class_<qaqmc_cuda::DiagonalEngine>(m, "DiagonalEngine")
        .def(py::init([](
                 int n_sites, int half_length,
                 double delta_min, double delta_max, double epsilon,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> bond_sites,
                 py::array_t<double, py::array::c_style | py::array::forcecast> bond_vij,
                 py::array_t<double, py::array::c_style | py::array::forcecast> inv_coord,
                 py::array_t<double, py::array::c_style | py::array::forcecast> alias_prob,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> alias_index,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> alias_loc_kind,
                 py::array_t<double, py::array::c_style | py::array::forcecast> bond_rmax,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> op_types,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> op_sites,
                 int device) {
            const auto bs = bond_sites.request();
            const auto bv = bond_vij.request();
            const auto ic = inv_coord.request();
            const auto ap = alias_prob.request();
            const auto ai = alias_index.request();
            const auto al = alias_loc_kind.request();
            const auto br = bond_rmax.request();
            const auto ot = op_types.request();
            const auto os = op_sites.request();
            if (bs.ndim != 2 || bs.shape[1] != 2)
                throw std::invalid_argument("bond_sites must have shape (n_bonds, 2)");
            const int n_bonds = static_cast<int>(bs.shape[0]);
            if (bv.ndim != 1 || bv.shape[0] != n_bonds)
                throw std::invalid_argument("bond_vij must have shape (n_bonds,)");
            if (ic.ndim != 1 || ic.shape[0] != n_sites)
                throw std::invalid_argument("inv_coord must have shape (n_sites,)");
            if (ap.ndim != 2) throw std::invalid_argument("alias_prob must be 2D");
            const int n_groups = static_cast<int>(ap.shape[0]);
            const int max_alias = static_cast<int>(ap.shape[1]);
            if (ai.ndim != 2 || al.ndim != 2 || ai.shape[0] != n_groups
                || al.shape[0] != n_groups || ai.shape[1] != max_alias
                || al.shape[1] != max_alias)
                throw std::invalid_argument("alias arrays must have the same 2D shape");
            if (br.ndim != 2 || br.shape[0] != n_groups || br.shape[1] != n_bonds)
                throw std::invalid_argument("bond_rmax must have shape (n_groups, n_bonds)");
            const py::ssize_t length = static_cast<py::ssize_t>(2) * half_length;
            if (ot.ndim != 1 || os.ndim != 1 || ot.shape[0] != length || os.shape[0] != length)
                throw std::invalid_argument("operator arrays must have length 2*half_length");

            return new qaqmc_cuda::DiagonalEngine(
                n_sites, half_length, delta_min, delta_max, epsilon,
                n_groups, max_alias, n_bonds,
                static_cast<const int32_t*>(bs.ptr), static_cast<const double*>(bv.ptr),
                static_cast<const double*>(ic.ptr), static_cast<const double*>(ap.ptr),
                static_cast<const int32_t*>(ai.ptr), static_cast<const int32_t*>(al.ptr),
                static_cast<const double*>(br.ptr), static_cast<const int32_t*>(ot.ptr),
                static_cast<const int32_t*>(os.ptr), device);
        }),
        py::arg("n_sites"), py::arg("half_length"),
        py::arg("delta_min"), py::arg("delta_max"), py::arg("epsilon"),
        py::arg("bond_sites"), py::arg("bond_vij"), py::arg("inv_coord"),
        py::arg("alias_prob"), py::arg("alias_index"), py::arg("alias_loc_kind"),
        py::arg("bond_rmax"), py::arg("op_types"), py::arg("op_sites"),
        py::arg("device") = 0)
        .def("diagonal_update", [](qaqmc_cuda::DiagonalEngine& self,
                                    uint64_t seed, uint64_t sweep_id) {
            qaqmc_cuda::DiagonalStats stats;
            {
                py::gil_scoped_release release;
                stats = self.diagonal_update(seed, sweep_id);
            }
            py::dict out;
            out["updated_slots"] = stats.updated_slots;
            out["proposal_attempts"] = stats.proposal_attempts;
            out["bond_proposals"] = stats.bond_proposals;
            out["bond_accepts"] = stats.bond_accepts;
            out["failed_slots"] = stats.failed_slots;
            out["elapsed_ms"] = stats.elapsed_ms;
            return out;
        }, py::arg("seed"), py::arg("sweep_id"))
        .def("build_events", [](qaqmc_cuda::DiagonalEngine& self, bool download) {
            qaqmc_cuda::EventStats stats;
            {
                py::gil_scoped_release release;
                stats = self.build_events();
            }
            py::dict out;
            out["site_events"] = stats.site_events;
            out["bond_events"] = stats.bond_events;
            out["elapsed_ms"] = stats.elapsed_ms;
            if (download) {
                py::array_t<uint64_t> site_keys(stats.site_events);
                py::array_t<uint32_t> site_values(stats.site_events);
                py::array_t<uint64_t> bond_keys(stats.bond_events);
                py::array_t<uint64_t> bond_values(stats.bond_events);
                py::array_t<int8_t> bond_spin(self.length());
                {
                    py::gil_scoped_release release;
                    self.get_site_events(site_keys.mutable_data(), site_values.mutable_data());
                    self.get_bond_events(bond_keys.mutable_data(), bond_values.mutable_data());
                    self.get_bond_spin(bond_spin.mutable_data());
                }
                out["site_keys"] = std::move(site_keys);
                out["site_values"] = std::move(site_values);
                out["bond_keys"] = std::move(bond_keys);
                out["bond_values"] = std::move(bond_values);
                out["bond_spin"] = std::move(bond_spin);
            }
            return out;
        }, py::arg("download") = false,
        "Build site/bond vertex-event streams sorted by (site,p).  Set download "
        "for white-box arrays; production keeps them device-resident.")
        .def("cluster_update", [](qaqmc_cuda::DiagonalEngine& self,
                                  uint64_t seed, uint64_t sweep_id) {
            qaqmc_cuda::ClusterStats stats;
            {
                py::gil_scoped_release release;
                stats = self.cluster_update(seed, sweep_id);
            }
            py::dict out;
            out["proposed_segments"] = stats.proposed_segments;
            out["accepted_segments"] = stats.accepted_segments;
            out["event_ms"] = stats.event_ms;
            out["sweep_ms"] = stats.sweep_ms;
            out["total_ms"] = stats.event_ms + stats.sweep_ms;
            return out;
        }, py::arg("seed"), py::arg("sweep_id"),
        "Build sorted events and run one open-boundary per-site cluster update.")
        .def("set_string_sites", [](
                 qaqmc_cuda::DiagonalEngine& self,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites,
                 int m_star) {
            const auto data = sites.request();
            if (data.ndim != 1)
                throw std::invalid_argument("string_sites must be one-dimensional");
            self.set_string_sites(static_cast<const int32_t*>(data.ptr),
                                  static_cast<int>(data.shape[0]), m_star);
        }, py::arg("string_sites"), py::arg("m_star"),
        "Configure a device-resident off-diagonal string seam.")
        .def("set_seam_mask_consistent",
             &qaqmc_cuda::DiagonalEngine::set_seam_mask_consistent,
             py::arg("mask"),
             "Set the seam mask and repair fixed-boundary worldline closure.")
        .def("half_line_proposal",
             [](qaqmc_cuda::DiagonalEngine& self, int local_index,
                bool direction_right) {
                 const auto proposal = self.half_line_proposal(
                     local_index, direction_right);
                 py::dict out;
                 out["valid"] = proposal.valid;
                 out["terminal_p"] = proposal.terminal_p;
                 out["log_physical_ratio"] = proposal.log_physical_ratio;
                 return out;
             }, py::arg("local_index"), py::arg("direction_right"),
             "Build a read-only CUDA half-line proposal for CPU exact comparison.")
        .def("topology_sweep", [](qaqmc_cuda::DiagonalEngine& self,
                                    double lambda, uint64_t seed,
                                    uint64_t sweep_id) {
            qaqmc_cuda::TopologyStats stats;
            {
                py::gil_scoped_release release;
                stats = self.topology_sweep(lambda, seed, sweep_id);
            }
            py::dict out;
            out["attempts"] = stats.attempts;
            out["accepts"] = stats.accepts;
            out["invalid"] = stats.invalid;
            out["active_count"] = stats.active_count;
            out["elapsed_ms"] = stats.elapsed_ms;
            return out;
        }, py::arg("lambda_"), py::arg("seed"), py::arg("sweep_id"),
        "Run one random-permutation half-line topology sweep on the GPU.")
        .def("save_checkpoint", &qaqmc_cuda::DiagonalEngine::save_checkpoint)
        .def("restore_checkpoint", &qaqmc_cuda::DiagonalEngine::restore_checkpoint)
        .def_property_readonly("has_checkpoint",
                               &qaqmc_cuda::DiagonalEngine::has_checkpoint)
        .def("profile_states", [](const qaqmc_cuda::DiagonalEngine& self,
                                   int profile_step) {
            if (profile_step <= 0
                || static_cast<std::size_t>(profile_step) > self.length())
                throw std::invalid_argument(
                    "profile_step must be in [1, operator length]");
            const py::ssize_t n_points = static_cast<py::ssize_t>(
                self.length() / static_cast<std::size_t>(profile_step));
            py::array_t<uint64_t> output({n_points,
                                          static_cast<py::ssize_t>(self.packed_words())});
            {
                py::gil_scoped_release release;
                self.get_profile_states(profile_step, output.mutable_data());
            }
            return output;
        }, py::arg("profile_step"),
        "Return packed propagated states after every profile_step slices.")
        .def("get_operator_string", [](const qaqmc_cuda::DiagonalEngine& self) {
            py::array_t<int32_t> types(self.length());
            py::array_t<int32_t> sites(self.length());
            {
                py::gil_scoped_release release;
                self.get_operator_string(types.mutable_data(), sites.mutable_data());
            }
            return py::make_tuple(types, sites);
        })
        .def("set_operator_string", [](
                 qaqmc_cuda::DiagonalEngine& self,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> types,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites) {
            const auto t = types.request();
            const auto s = sites.request();
            if (t.ndim != 1 || s.ndim != 1
                || static_cast<std::size_t>(t.shape[0]) != self.length()
                || static_cast<std::size_t>(s.shape[0]) != self.length())
                throw std::invalid_argument("operator arrays have the wrong length");
            py::gil_scoped_release release;
            self.set_operator_string(static_cast<const int32_t*>(t.ptr),
                                     static_cast<const int32_t*>(s.ptr));
        })
        .def_property_readonly("n_sites", &qaqmc_cuda::DiagonalEngine::n_sites)
        .def_property_readonly("half_length", &qaqmc_cuda::DiagonalEngine::half_length)
        .def_property_readonly("length", &qaqmc_cuda::DiagonalEngine::length)
        .def_property_readonly("packed_words", &qaqmc_cuda::DiagonalEngine::packed_words)
        .def_property_readonly("device_bytes", &qaqmc_cuda::DiagonalEngine::device_bytes)
        .def_property_readonly("seam_mask", &qaqmc_cuda::DiagonalEngine::seam_mask)
        .def_property_readonly("string_site_count",
                               &qaqmc_cuda::DiagonalEngine::string_site_count)
        .def_property_readonly("seam_cut", &qaqmc_cuda::DiagonalEngine::seam_cut);
}

}  // namespace qaqmc_cuda::bindings
