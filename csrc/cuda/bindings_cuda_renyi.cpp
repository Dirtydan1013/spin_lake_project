#include "bindings_cuda.cuh"
#include "qaqmc_cuda_renyi.cuh"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace qaqmc_cuda::bindings {

void bind_renyi_engine(py::module_& m) {
    py::class_<qaqmc_cuda::RenyiEngine>(m, "RenyiEngine")
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
            if (ap.ndim != 2)
                throw std::invalid_argument("alias_prob must be two-dimensional");
            const int n_groups = static_cast<int>(ap.shape[0]);
            const int max_alias = static_cast<int>(ap.shape[1]);
            if (ai.ndim != 2 || al.ndim != 2
                || ai.shape[0] != n_groups || ai.shape[1] != max_alias
                || al.shape[0] != n_groups || al.shape[1] != max_alias)
                throw std::invalid_argument("alias arrays must have identical shapes");
            if (br.ndim != 2 || br.shape[0] != n_groups || br.shape[1] != n_bonds)
                throw std::invalid_argument("bond_rmax has the wrong shape");
            const py::ssize_t length = static_cast<py::ssize_t>(2) * half_length;
            if (ot.ndim != 2 || os.ndim != 2 || ot.shape[0] != 2
                || os.shape[0] != 2 || ot.shape[1] != length || os.shape[1] != length)
                throw std::invalid_argument("operator arrays must have shape (2, 2*M)");
            return new qaqmc_cuda::RenyiEngine(
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
        .def("set_cut", &qaqmc_cuda::RenyiEngine::set_cut, py::arg("cut"))
        .def("set_mask", [](qaqmc_cuda::RenyiEngine& self,
                              py::array_t<uint8_t,
                                  py::array::c_style | py::array::forcecast> mask) {
            const auto data = mask.request();
            if (data.ndim != 1)
                throw std::invalid_argument("mask must be one-dimensional");
            self.set_mask(static_cast<const uint8_t*>(data.ptr),
                          static_cast<int>(data.shape[0]));
        }, py::arg("mask"))
        .def("get_mask", [](const qaqmc_cuda::RenyiEngine& self) {
            py::array_t<uint8_t> mask(self.n_sites());
            self.get_mask(mask.mutable_data(), self.n_sites());
            return mask;
        })
        .def("diagonal_update", [](qaqmc_cuda::RenyiEngine& self,
                                    uint64_t seed, uint64_t sweep_id) {
            const auto stats = self.diagonal_update(seed, sweep_id);
            py::dict out;
            out["updated_slots"] = stats.updated_slots;
            out["proposal_attempts"] = stats.proposal_attempts;
            out["bond_proposals"] = stats.bond_proposals;
            out["bond_accepts"] = stats.bond_accepts;
            out["failed_slots"] = stats.failed_slots;
            out["elapsed_ms"] = stats.elapsed_ms;
            return out;
        }, py::arg("seed"), py::arg("sweep_id"))
        .def("build_events", [](qaqmc_cuda::RenyiEngine& self, bool download) {
            const auto stats = self.build_events();
            py::dict out;
            out["site_events"] = stats.site_events;
            out["bond_events"] = stats.bond_events;
            out["elapsed_ms"] = stats.elapsed_ms;
            if (download) {
                py::array_t<uint64_t> site_keys(stats.site_events);
                py::array_t<uint32_t> site_values(stats.site_events);
                py::array_t<uint64_t> bond_keys(stats.bond_events);
                py::array_t<uint32_t> bond_values(stats.bond_events);
                py::array_t<int8_t> bond_spin(
                    static_cast<py::ssize_t>(2 * self.length()));
                self.get_site_events(site_keys.mutable_data(), site_values.mutable_data());
                self.get_bond_events(bond_keys.mutable_data(), bond_values.mutable_data());
                self.get_bond_spin(bond_spin.mutable_data());
                out["site_keys"] = std::move(site_keys);
                out["site_values"] = std::move(site_values);
                out["bond_keys"] = std::move(bond_keys);
                out["bond_values"] = std::move(bond_values);
                out["bond_spin"] = std::move(bond_spin);
            }
            return out;
        }, py::arg("download") = false)
        .def("cluster_update", [](qaqmc_cuda::RenyiEngine& self,
                                   uint64_t seed, uint64_t sweep_id) {
            const auto stats = self.cluster_update(seed, sweep_id);
            py::dict out;
            out["proposed_segments"] = stats.proposed_segments;
            out["accepted_segments"] = stats.accepted_segments;
            out["event_ms"] = stats.event_ms;
            out["sweep_ms"] = stats.sweep_ms;
            out["total_ms"] = stats.event_ms + stats.sweep_ms;
            return out;
        }, py::arg("seed"), py::arg("sweep_id"))
        .def("log_weight_ratio_for_toggle",
             [](qaqmc_cuda::RenyiEngine& self, int site) {
                 const auto ratio = self.log_weight_ratio_for_toggle(site);
                 py::dict out;
                 out["log_ratio"] = ratio.log_ratio;
                 out["current_valid"] = ratio.current_valid;
                 out["proposed_valid"] = ratio.proposed_valid;
                 return out;
             }, py::arg("site"),
             "Return the read-only QAQMC log-weight ratio for one Renyi mask bit.")
        .def("topology_sweep", [](
                 qaqmc_cuda::RenyiEngine& self,
                 py::array_t<int32_t,
                     py::array::c_style | py::array::forcecast> topology_sites,
                 double lambda, uint64_t seed, uint64_t sweep_id) {
            const auto data = topology_sites.request();
            if (data.ndim != 1)
                throw std::invalid_argument("topology_sites must be one-dimensional");
            const auto stats = self.topology_sweep(
                static_cast<const int32_t*>(data.ptr),
                static_cast<int>(data.shape[0]), lambda, seed, sweep_id);
            py::dict out;
            out["attempts"] = stats.attempts;
            out["accepts"] = stats.accepts;
            out["invalid"] = stats.invalid;
            out["active_count"] = stats.active_count;
            out["elapsed_ms"] = stats.elapsed_ms;
            return out;
        }, py::arg("topology_sites"), py::arg("lambda_"),
        py::arg("seed"), py::arg("sweep_id"))
        .def("save_checkpoint", &qaqmc_cuda::RenyiEngine::save_checkpoint,
             "Save both operator strings device-to-device; the mask is not included.")
        .def("restore_checkpoint", &qaqmc_cuda::RenyiEngine::restore_checkpoint,
             "Restore both operator strings; set the intended mask separately first.")
        .def_property_readonly("has_checkpoint",
                               &qaqmc_cuda::RenyiEngine::has_checkpoint)
        .def("get_operator_strings", [](const qaqmc_cuda::RenyiEngine& self) {
            py::array_t<int32_t> types({static_cast<py::ssize_t>(2),
                                        static_cast<py::ssize_t>(self.length())});
            py::array_t<int32_t> sites({static_cast<py::ssize_t>(2),
                                        static_cast<py::ssize_t>(self.length())});
            self.get_operator_strings(types.mutable_data(), sites.mutable_data());
            return py::make_tuple(types, sites);
        })
        .def("set_operator_strings", [](
                 qaqmc_cuda::RenyiEngine& self,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> types,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites) {
            const auto t = types.request();
            const auto s = sites.request();
            if (t.ndim != 2 || s.ndim != 2 || t.shape[0] != 2 || s.shape[0] != 2
                || static_cast<std::size_t>(t.shape[1]) != self.length()
                || static_cast<std::size_t>(s.shape[1]) != self.length())
                throw std::invalid_argument("operator arrays must have shape (2, length)");
            self.set_operator_strings(static_cast<const int32_t*>(t.ptr),
                                      static_cast<const int32_t*>(s.ptr));
        })
        .def_property_readonly("n_sites", &qaqmc_cuda::RenyiEngine::n_sites)
        .def_property_readonly("half_length", &qaqmc_cuda::RenyiEngine::half_length)
        .def_property_readonly("length", &qaqmc_cuda::RenyiEngine::length)
        .def_property_readonly("cut", &qaqmc_cuda::RenyiEngine::cut)
        .def_property_readonly("packed_words", &qaqmc_cuda::RenyiEngine::packed_words)
        .def_property_readonly("device_bytes", &qaqmc_cuda::RenyiEngine::device_bytes);
}

}  // namespace qaqmc_cuda::bindings
