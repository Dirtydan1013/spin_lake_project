#include "bindings_cuda.cuh"
#include "qaqmc_cuda_renyi.cuh"

#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace qaqmc_cuda::bindings {
namespace {

py::dict renyi_diagonal_stats_dict(const DiagonalStats& stats) {
    py::dict out;
    out["updated_slots"] = stats.updated_slots;
    out["proposal_attempts"] = stats.proposal_attempts;
    out["bond_proposals"] = stats.bond_proposals;
    out["bond_accepts"] = stats.bond_accepts;
    out["failed_slots"] = stats.failed_slots;
    out["elapsed_ms"] = stats.elapsed_ms;
    return out;
}

py::dict renyi_cluster_stats_dict(const ClusterStats& stats) {
    py::dict out;
    out["proposed_segments"] = stats.proposed_segments;
    out["accepted_segments"] = stats.accepted_segments;
    out["event_ms"] = stats.event_ms;
    out["sweep_ms"] = stats.sweep_ms;
    out["total_ms"] = stats.event_ms + stats.sweep_ms;
    return out;
}

py::dict renyi_topology_stats_dict(const TopologyStats& stats) {
    py::dict out;
    out["attempts"] = stats.attempts;
    out["accepts"] = stats.accepts;
    out["invalid"] = stats.invalid;
    out["active_count"] = stats.active_count;
    out["elapsed_ms"] = stats.elapsed_ms;
    return out;
}

void validate_renyi_rng_arrays(
    const py::buffer_info& seeds,
    const py::buffer_info& sweeps,
    int batch_size) {
    if (seeds.ndim != 1 || sweeps.ndim != 1
        || seeds.shape[0] != batch_size || sweeps.shape[0] != batch_size)
        throw std::invalid_argument(
            "seeds and sweep_ids must have shape (batch_size,)");
}

}  // namespace

void bind_batched_renyi_engine(py::module_& m) {
    py::class_<BatchedRenyiEngine>(m, "BatchedRenyiEngine")
        .def(py::init([](
                 int batch_size, int n_sites, int half_length,
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
            if (batch_size <= 0)
                throw std::invalid_argument("batch_size must be positive");
            if (bs.ndim != 2 || bs.shape[1] != 2)
                throw std::invalid_argument("bond_sites must have shape (n_bonds, 2)");
            const int n_bonds = static_cast<int>(bs.shape[0]);
            if (bv.ndim != 1 || bv.shape[0] != n_bonds)
                throw std::invalid_argument("bond_vij must have shape (n_bonds,)");
            if (ic.ndim != 1 || ic.shape[0] != n_sites)
                throw std::invalid_argument("inv_coord must have shape (n_sites,)");
            if (ap.ndim != 2)
                throw std::invalid_argument("alias_prob must be 2D");
            const int n_groups = static_cast<int>(ap.shape[0]);
            const int max_alias = static_cast<int>(ap.shape[1]);
            if (ai.ndim != 2 || al.ndim != 2
                || ai.shape[0] != n_groups || al.shape[0] != n_groups
                || ai.shape[1] != max_alias || al.shape[1] != max_alias)
                throw std::invalid_argument("alias arrays must have the same 2D shape");
            if (br.ndim != 2 || br.shape[0] != n_groups || br.shape[1] != n_bonds)
                throw std::invalid_argument(
                    "bond_rmax must have shape (n_groups, n_bonds)");
            const py::ssize_t length = static_cast<py::ssize_t>(2) * half_length;
            if (ot.ndim != 3 || os.ndim != 3
                || ot.shape[0] != batch_size || os.shape[0] != batch_size
                || ot.shape[1] != 2 || os.shape[1] != 2
                || ot.shape[2] != length || os.shape[2] != length)
                throw std::invalid_argument(
                    "operator arrays must have shape (batch_size, 2, 2*half_length)");
            return new BatchedRenyiEngine(
                batch_size, n_sites, half_length, delta_min, delta_max, epsilon,
                n_groups, max_alias, n_bonds,
                static_cast<const int32_t*>(bs.ptr), static_cast<const double*>(bv.ptr),
                static_cast<const double*>(ic.ptr), static_cast<const double*>(ap.ptr),
                static_cast<const int32_t*>(ai.ptr), static_cast<const int32_t*>(al.ptr),
                static_cast<const double*>(br.ptr), static_cast<const int32_t*>(ot.ptr),
                static_cast<const int32_t*>(os.ptr), device);
        }),
        py::arg("batch_size"), py::arg("n_sites"), py::arg("half_length"),
        py::arg("delta_min"), py::arg("delta_max"), py::arg("epsilon"),
        py::arg("bond_sites"), py::arg("bond_vij"), py::arg("inv_coord"),
        py::arg("alias_prob"), py::arg("alias_index"), py::arg("alias_loc_kind"),
        py::arg("bond_rmax"), py::arg("op_types"), py::arg("op_sites"),
        py::arg("device") = 0)
        .def("set_cut", &BatchedRenyiEngine::set_cut, py::arg("cut"))
        .def("set_masks", [](
                 BatchedRenyiEngine& self,
                 py::array_t<uint8_t, py::array::c_style | py::array::forcecast> masks) {
            const auto data = masks.request();
            if (data.ndim != 2 || data.shape[0] != self.batch_size()
                || data.shape[1] != self.n_sites())
                throw std::invalid_argument(
                    "masks must have shape (batch_size, n_sites)");
            py::gil_scoped_release release;
            self.set_masks(static_cast<const uint8_t*>(data.ptr));
        }, py::arg("masks"))
        .def("get_masks", [](const BatchedRenyiEngine& self) {
            py::array_t<uint8_t> masks({static_cast<py::ssize_t>(self.batch_size()),
                                        static_cast<py::ssize_t>(self.n_sites())});
            {
                py::gil_scoped_release release;
                self.get_masks(masks.mutable_data());
            }
            return masks;
        })
        .def("diagonal_update", [](
                 BatchedRenyiEngine& self,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> seeds,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> sweeps) {
            const auto seed_data = seeds.request();
            const auto sweep_data = sweeps.request();
            validate_renyi_rng_arrays(seed_data, sweep_data, self.batch_size());
            std::vector<DiagonalStats> stats;
            {
                py::gil_scoped_release release;
                stats = self.diagonal_update(
                    static_cast<const uint64_t*>(seed_data.ptr),
                    static_cast<const uint64_t*>(sweep_data.ptr));
            }
            py::list out;
            for (const auto& item : stats) out.append(renyi_diagonal_stats_dict(item));
            return out;
        }, py::arg("seeds"), py::arg("sweep_ids"))
        .def("cluster_update", [](
                 BatchedRenyiEngine& self,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> seeds,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> sweeps) {
            const auto seed_data = seeds.request();
            const auto sweep_data = sweeps.request();
            validate_renyi_rng_arrays(seed_data, sweep_data, self.batch_size());
            std::vector<ClusterStats> stats;
            {
                py::gil_scoped_release release;
                stats = self.cluster_update(
                    static_cast<const uint64_t*>(seed_data.ptr),
                    static_cast<const uint64_t*>(sweep_data.ptr));
            }
            py::list out;
            for (const auto& item : stats) out.append(renyi_cluster_stats_dict(item));
            return out;
        }, py::arg("seeds"), py::arg("sweep_ids"))
        .def("topology_sweep", [](
                 BatchedRenyiEngine& self,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites,
                 double lambda,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> seeds,
                 py::array_t<uint64_t, py::array::c_style | py::array::forcecast> sweeps) {
            const auto site_data = sites.request();
            const auto seed_data = seeds.request();
            const auto sweep_data = sweeps.request();
            if (site_data.ndim != 1)
                throw std::invalid_argument("topology_sites must be one-dimensional");
            validate_renyi_rng_arrays(seed_data, sweep_data, self.batch_size());
            std::vector<TopologyStats> stats;
            {
                py::gil_scoped_release release;
                stats = self.topology_sweep(
                    static_cast<const int32_t*>(site_data.ptr),
                    static_cast<int>(site_data.shape[0]), lambda,
                    static_cast<const uint64_t*>(seed_data.ptr),
                    static_cast<const uint64_t*>(sweep_data.ptr));
            }
            py::list out;
            for (const auto& item : stats) out.append(renyi_topology_stats_dict(item));
            return out;
        }, py::arg("topology_sites"), py::arg("lambda_"),
        py::arg("seeds"), py::arg("sweep_ids"))
        .def("save_checkpoint", &BatchedRenyiEngine::save_checkpoint)
        .def("restore_checkpoint", &BatchedRenyiEngine::restore_checkpoint)
        .def_property_readonly("has_checkpoint", &BatchedRenyiEngine::has_checkpoint)
        .def("get_operator_strings", [](const BatchedRenyiEngine& self) {
            py::array_t<int32_t> types({static_cast<py::ssize_t>(self.batch_size()),
                                        static_cast<py::ssize_t>(2),
                                        static_cast<py::ssize_t>(self.length())});
            py::array_t<int32_t> sites({static_cast<py::ssize_t>(self.batch_size()),
                                        static_cast<py::ssize_t>(2),
                                        static_cast<py::ssize_t>(self.length())});
            {
                py::gil_scoped_release release;
                self.get_operator_strings(types.mutable_data(), sites.mutable_data());
            }
            return py::make_tuple(types, sites);
        })
        .def("set_operator_strings", [](
                 BatchedRenyiEngine& self,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> types,
                 py::array_t<int32_t, py::array::c_style | py::array::forcecast> sites) {
            const auto t = types.request();
            const auto s = sites.request();
            if (t.ndim != 3 || s.ndim != 3
                || t.shape[0] != self.batch_size() || s.shape[0] != self.batch_size()
                || t.shape[1] != 2 || s.shape[1] != 2
                || static_cast<std::size_t>(t.shape[2]) != self.length()
                || static_cast<std::size_t>(s.shape[2]) != self.length())
                throw std::invalid_argument(
                    "operator arrays must have shape (batch_size, 2, length)");
            py::gil_scoped_release release;
            self.set_operator_strings(static_cast<const int32_t*>(t.ptr),
                                      static_cast<const int32_t*>(s.ptr));
        })
        .def_property_readonly("batch_size", &BatchedRenyiEngine::batch_size)
        .def_property_readonly("n_sites", &BatchedRenyiEngine::n_sites)
        .def_property_readonly("length", &BatchedRenyiEngine::length)
        .def_property_readonly("shared_model_bytes",
                               &BatchedRenyiEngine::shared_model_bytes)
        .def_property_readonly("device_bytes", &BatchedRenyiEngine::device_bytes);
}

}  // namespace qaqmc_cuda::bindings
