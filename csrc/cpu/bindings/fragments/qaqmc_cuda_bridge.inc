        // Read-only bridge consumed by the optional CUDA wrappers. The CPU
        // engine keeps compact alias indices internally; this export restores
        // the stable int32 NumPy schema only when a CUDA backend is created.
        .def("export_cuda_diagonal_data", [](const QAQMCEngine& self) {
            const auto model = self.get_model_data();
            const auto& group = model->grp_alias;
            const auto& vij = model->vij;
            const int n_groups = group.n_groups;
            const int max_alias = group.max_alias;
            const int n_bonds = vij.n_bonds;
            const std::size_t alias_count =
                static_cast<std::size_t>(n_groups) * max_alias;

            py::array_t<double> alias_prob({n_groups, max_alias});
            py::array_t<int32_t> alias_index({n_groups, max_alias});
            py::array_t<int32_t> alias_loc_kind({n_groups, max_alias});
            py::array_t<double> bond_rmax({n_groups, n_bonds});
            py::array_t<int32_t> bond_sites({n_bonds, 2});
            py::array_t<double> bond_vij(n_bonds);
            py::array_t<double> inv_coord(self.get_N());

            double* prob_out = alias_prob.mutable_data();
            int32_t* index_out = alias_index.mutable_data();
            int32_t* loc_out = alias_loc_kind.mutable_data();
            double* rmax_out = bond_rmax.mutable_data();
            int32_t* sites_out = bond_sites.mutable_data();
            double* vij_out = bond_vij.mutable_data();
            double* coord_out = inv_coord.mutable_data();
            {
                py::gil_scoped_release release;
                std::memcpy(prob_out, group.alias_prob.data(),
                            alias_count * sizeof(double));
                for (std::size_t flat = 0; flat < alias_count; ++flat) {
                    const int index = static_cast<int>(flat % max_alias);
                    index_out[flat] = group.alias_at(flat);
                    loc_out[flat] = index < self.get_N()
                        ? (index << 1)
                        : (((index - self.get_N()) << 1) | 1);
                }
                if (n_bonds > 0) {
                    for (int g = 0; g < n_groups; ++g) {
                        std::memcpy(
                            rmax_out + static_cast<std::size_t>(g) * n_bonds,
                            group.bond_W_rmax_all.data()
                                + static_cast<std::size_t>(g)
                                    * group.n_bonds_pad,
                            static_cast<std::size_t>(n_bonds)
                                * sizeof(double));
                    }
                }
                for (std::size_t i = 0; i < vij.bond_sites_flat.size(); ++i) {
                    sites_out[i] = static_cast<int32_t>(vij.bond_sites_flat[i]);
                }
                if (n_bonds > 0) {
                    std::memcpy(vij_out, vij.vij_list.data(),
                                vij.vij_list.size() * sizeof(double));
                }
                std::memcpy(coord_out, vij.inv_coord.data(),
                            vij.inv_coord.size() * sizeof(double));
            }

            py::dict out;
            out["alias_prob"] = std::move(alias_prob);
            out["alias_index"] = std::move(alias_index);
            out["alias_loc_kind"] = std::move(alias_loc_kind);
            out["bond_rmax"] = std::move(bond_rmax);
            out["bond_sites"] = std::move(bond_sites);
            out["bond_vij"] = std::move(bond_vij);
            out["inv_coord"] = std::move(inv_coord);
            return out;
        },
        "Export grouped alias and Rydberg model arrays for constructing the "
        "optional CUDA backend. This is a one-time compatibility copy, not a "
        "per-transition API.")
