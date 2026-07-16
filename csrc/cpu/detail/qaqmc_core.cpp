#include "../include/qaqmc_core.hpp"
#include <cstring>
#include <cassert>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// ═════════════════════════════════════════════════════════════════════════════
// V_ij builder
// ═════════════════════════════════════════════════════════════════════════════

// Minimum-image squared distance under periodic boundaries.  `box` holds
// `n_box` supercell vectors (each `pos_dim` long, row-major); the nearest
// periodic image is found by searching integer combinations in {-1,0,1}^n_box
// (enough for the mildly-skewed triangular / linear supercells used here).
// n_box == 0 reproduces the open (direct Euclidean) distance.
static double rydberg_min_image_dist2(const double* diff, int pos_dim,
                                      const double* box, int n_box) {
    if (n_box <= 0) {
        double d2 = 0.0;
        for (int d = 0; d < pos_dim; ++d) d2 += diff[d] * diff[d];
        return d2;
    }
    int n_combos = 1;
    for (int k = 0; k < n_box; ++k) n_combos *= 3;
    double best = 1e300;
    for (int c = 0; c < n_combos; ++c) {
        double shifted[3];
        for (int d = 0; d < pos_dim; ++d) shifted[d] = diff[d];
        int tmp = c;
        for (int k = 0; k < n_box; ++k) {
            int coef = (tmp % 3) - 1;   // -1, 0, +1
            tmp /= 3;
            if (coef)
                for (int d = 0; d < pos_dim; ++d)
                    shifted[d] -= coef * box[k * pos_dim + d];
        }
        double d2 = 0.0;
        for (int d = 0; d < pos_dim; ++d) d2 += shifted[d] * shifted[d];
        if (d2 < best) best = d2;
    }
    return best;
}

RydbergVij build_rydberg_vij(int N, double Omega, double Rb,
                              const double* pos, int pos_dim,
                              int neighbor_cutoff,
                              const double* box, int n_box) {
    // Step 1: Compute all pairwise distances (minimum-image if box given)
    std::vector<double> all_dists;
    all_dists.reserve(N * (N - 1) / 2);
    double diff[3];
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            for (int d = 0; d < pos_dim; ++d)
                diff[d] = pos[i * pos_dim + d] - pos[j * pos_dim + d];
            all_dists.push_back(
                std::sqrt(rydberg_min_image_dist2(diff, pos_dim, box, n_box)));
        }
    }

    // Step 2: Determine cutoff distance from neighbor shells
    double cutoff_dist = -1.0;  // -1 means no cutoff
    if (neighbor_cutoff > 0 && !all_dists.empty()) {
        std::vector<double> sorted_dists = all_dists;
        std::sort(sorted_dists.begin(), sorted_dists.end());

        // Group into shells using relative tolerance
        const double tol = 1e-6;
        std::vector<double> shells;
        shells.push_back(sorted_dists[0]);
        for (size_t k = 1; k < sorted_dists.size(); ++k) {
            double ref = std::max(shells.back(), 1e-15);
            if (std::abs(sorted_dists[k] - shells.back()) / ref > tol) {
                shells.push_back(sorted_dists[k]);
            }
        }

        int shell_idx = std::min(neighbor_cutoff, (int)shells.size()) - 1;
        cutoff_dist = shells[shell_idx] * (1.0 + tol);
    }

    // Step 3: Build bond list with filtering
    RydbergVij res;
    int dist_idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dist = all_dists[dist_idx++];
            if (cutoff_dist > 0.0 && dist > cutoff_dist)
                continue;
            double ratio = Rb / dist;
            double vij = Omega * ratio * ratio * ratio * ratio * ratio * ratio;
            res.bonds_i.push_back(i);
            res.bonds_j.push_back(j);
            res.vij_list.push_back(vij);
        }
    }
    res.n_bonds = (int)res.bonds_i.size();

    // Step 2: Compute per-site coordination number z_eff[i]
    res.coord_number.assign(N, 0);
    for (int b = 0; b < res.n_bonds; ++b) {
        res.coord_number[res.bonds_i[b]]++;
        res.coord_number[res.bonds_j[b]]++;
    }
    // Reciprocal coordination (delta_i = delta * inv_coord[i]) so the hot
    // paths trade a division for a multiply.
    res.inv_coord.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        if (res.coord_number[i] > 0) res.inv_coord[i] = 1.0 / res.coord_number[i];
    }

    int n_bonds_pad = std::max(res.n_bonds, 1);
    res.bond_sites_flat.resize(n_bonds_pad * 2, 0);
    for (int b = 0; b < res.n_bonds; ++b) {
        res.bond_sites_flat[b * 2 + 0] = res.bonds_i[b];
        res.bond_sites_flat[b * 2 + 1] = res.bonds_j[b];
    }
    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// Alias Table builder
// ═════════════════════════════════════════════════════════════════════════════

AliasTable build_qaqmc_alias_tables(int M_total, int N, int n_bonds,
                                     double Omega,
                                     const double* delta_sched,
                                     const double* bond_vij,
                                     const int* bond_si, const int* bond_sj,
                                     const int* coord_number,
                                     double epsilon,
                                     int p_start, int p_end) {
    if (p_end < 0) p_end = M_total;
    int p_count = p_end - p_start;
    AliasTable res;
    int max_alias = N + n_bonds;
    int n_bonds_pad = std::max(n_bonds, 1);
    res.max_alias = max_alias;
    res.n_bonds_pad = n_bonds_pad;

    // Allocate flat arrays — sized for p_count slices, not M_total
    res.bond_W_all.assign(p_count * n_bonds_pad * 4, 0.0);
    res.bond_W_max_all.assign(p_count * n_bonds_pad, 0.0);
    res.bond_W_rmax_all.assign(p_count * n_bonds_pad, 0.0);
    res.n_alias_all.resize(p_count, 0);
    res.alias_prob_all.assign(p_count * max_alias, 0.0);
    res.alias_idx_all.assign(p_count * max_alias, 0);
    res.op_map_kind_all.assign(p_count * max_alias, 0);
    res.op_map_loc_all.assign(p_count * max_alias, 0);

    // Each time slice is completely independent → OpenMP parallel for
#ifdef QAQMC_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int pi = 0; pi < p_count; ++pi) {
        int p = p_start + pi;  // actual slice index
        // Thread-local temporary buffers (each thread gets its own copy)
        std::vector<double> weights(max_alias);
        std::vector<int> op_kind(max_alias);
        std::vector<int> op_loc(max_alias);
        std::vector<double> prob_arr(max_alias);
        std::vector<int64_t> alias_arr(max_alias);
        std::vector<int> small_buf, large_buf;

        double delta = delta_sched[p];

        int n_a = 0;

        // Site operators
        for (int i = 0; i < N; ++i) {
            weights[n_a] = Omega / 2.0;
            op_kind[n_a] = 0;
            op_loc[n_a] = i;
            n_a++;
        }

        // Bond operators: Step 3+4 — asymmetric delta + dynamic C_ij
        for (int b = 0; b < n_bonds; ++b) {
            double vij = bond_vij[b];
            int si = bond_si[b];
            int sj = bond_sj[b];
            double inv_zi = (coord_number[si] > 0) ? 1.0 / coord_number[si] : 0.0;
            double inv_zj = (coord_number[sj] > 0) ? 1.0 / coord_number[sj] : 0.0;
            double delta_i = delta * inv_zi;
            double delta_j = delta * inv_zj;

            double W[4], bmax;
            QAQMCEngine::compute_bond_W_inline(delta_i, delta_j, vij, epsilon, W, bmax);

            int base = (pi * n_bonds_pad + b) * 4;
            res.bond_W_all[base + 0] = W[0];
            res.bond_W_all[base + 1] = W[1];
            res.bond_W_all[base + 2] = W[2];
            res.bond_W_all[base + 3] = W[3];

            res.bond_W_max_all[pi * n_bonds_pad + b] = bmax;
            res.bond_W_rmax_all[pi * n_bonds_pad + b] = (bmax > 0.0) ? 1.0 / bmax : 0.0;

            weights[n_a] = bmax;
            op_kind[n_a] = 1;
            op_loc[n_a] = b;
            n_a++;
        }

        res.n_alias_all[pi] = n_a;

        // Copy op_map
        for (int i = 0; i < n_a; ++i) {
            res.op_map_kind_all[pi * max_alias + i] = op_kind[i];
            res.op_map_loc_all[pi * max_alias + i] = op_loc[i];
        }

        // Build alias table (Vose's algorithm)
        double total = 0.0;
        for (int i = 0; i < n_a; ++i) total += weights[i];

        for (int i = 0; i < n_a; ++i) {
            prob_arr[i] = weights[i] * n_a / total;
            alias_arr[i] = i;
        }

        small_buf.clear();
        large_buf.clear();
        for (int i = 0; i < n_a; ++i) {
            if (prob_arr[i] < 1.0)
                small_buf.push_back(i);
            else
                large_buf.push_back(i);
        }

        while (!small_buf.empty() && !large_buf.empty()) {
            int s = small_buf.back(); small_buf.pop_back();
            int l = large_buf.back(); large_buf.pop_back();
            alias_arr[s] = l;
            prob_arr[l] -= (1.0 - prob_arr[s]);
            if (prob_arr[l] < 1.0)
                small_buf.push_back(l);
            else
                large_buf.push_back(l);
        }

        for (int i = 0; i < n_a; ++i) {
            res.alias_prob_all[pi * max_alias + i] = prob_arr[i];
            res.alias_idx_all[pi * max_alias + i] = alias_arr[i];
        }
    }

    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// QAQMCEngine
// ═════════════════════════════════════════════════════════════════════════════

QAQMCEngine::QAQMCEngine(int N, double Omega, double delta_min, double delta_max,
                          double Rb, int M, double epsilon, uint64_t seed,
                          const double* pos, int pos_dim,
                          int neighbor_cutoff, int delta_groups,
                          const double* box, int n_box)
    : N_(N), M_(M), M_total_(2 * M),
      Omega_(Omega), Rb_(Rb), delta_min_(delta_min), delta_max_(delta_max),
      epsilon_(epsilon), delta_groups_(delta_groups),
      rng_(seed), model_(std::make_shared<QAQMCModelData>())
{
    if (delta_groups_ <= 0) {
        throw std::invalid_argument(
            "QAQMCEngine: delta_groups must be > 0 (the engine only supports the grouped "
            "alias-table path; the legacy precompute/chunk/on-the-fly modes were removed).");
    }
    site_W_ = Omega / 2.0;
    site_W_max_ = Omega / 2.0;

    model_->N = N_;
    model_->M = M_;
    model_->M_total = M_total_;
    model_->Omega = Omega_;
    model_->Rb = Rb_;
    model_->delta_min = delta_min_;
    model_->delta_max = delta_max_;
    model_->epsilon = epsilon_;
    model_->delta_groups = delta_groups_;

    // Build V_ij (optional neighbor cutoff; optional periodic box)
    model_->vij = build_rydberg_vij(N, Omega, Rb, pos, pos_dim, neighbor_cutoff, box, n_box);

    // Cache positions for downstream observables (dimer structure factor etc.)
    model_->pos_dim = pos_dim;
    model_->pos_flat.assign(pos, pos + N * pos_dim);

    // Build grouped alias tables
    {
        int G = delta_groups_;
        int n_bonds = model_->vij.n_bonds;
        int n_bonds_pad = std::max(n_bonds, 1);
        int max_alias = N_ + n_bonds;
        const int* bond_si = model_->vij.bonds_i.data();
        const int* bond_sj = model_->vij.bonds_j.data();
        const double* bond_vij = model_->vij.vij_list.data();
        const double* inv_coord = model_->vij.inv_coord.data();

        model_->grp_alias.n_groups = G;
        model_->grp_alias.max_alias = max_alias;
        model_->grp_alias.n_bonds_pad = n_bonds_pad;
        // (slice -> group is computed incrementally in diagonal_update; no
        //  4-byte-per-slice map is materialised.)

        // For each group, find the range of delta values and compute
        // envelope W_max (upper bound) for rejection sampling.
        model_->grp_alias.bond_W_rmax_all.assign(G * n_bonds_pad, 0.0);
        model_->grp_alias.n_alias_all.resize(G, 0);
        const size_t alias_size = static_cast<size_t>(G) * max_alias;
        model_->grp_alias.alias_prob.assign(alias_size, 0.0);
        model_->grp_alias.alias_u16 = max_alias <=
            static_cast<int>(std::numeric_limits<uint16_t>::max()) + 1;
        if (model_->grp_alias.alias_u16) {
            model_->grp_alias.alias_idx16.assign(alias_size, 0);
            model_->grp_alias.alias_idx32.clear();
        } else {
            model_->grp_alias.alias_idx32.assign(alias_size, 0);
            model_->grp_alias.alias_idx16.clear();
        }

#ifdef QAQMC_USE_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int g = 0; g < G; ++g) {
            // Find slice range for this group
            int p_lo = (int)((int64_t)g * M_total_ / G);
            int p_hi = (int)((int64_t)(g + 1) * M_total_ / G);
            if (p_hi > M_total_) p_hi = M_total_;

            // Compute envelope W_max for each bond: max over all delta in group
            // Sample boundary + midpoint deltas for envelope
            std::vector<double> sample_deltas;
            sample_deltas.push_back(delta_at(p_lo));
            sample_deltas.push_back(delta_at(std::min(p_hi - 1, M_total_ - 1)));
            int p_mid = (p_lo + p_hi) / 2;
            if (p_mid < M_total_) sample_deltas.push_back(delta_at(p_mid));

            std::vector<double> env_W_max(n_bonds, 0.0);
            for (double delta : sample_deltas) {
                for (int b = 0; b < n_bonds; ++b) {
                    double di = delta * inv_coord[bond_si[b]];
                    double dj = delta * inv_coord[bond_sj[b]];
                    double W[4], wmax;
                    compute_bond_W_inline(di, dj, bond_vij[b], epsilon_, W, wmax);
                    if (wmax > env_W_max[b]) env_W_max[b] = wmax;
                }
            }

            // Store envelope W_max (+ reciprocal for the acceptance test)
            for (int b = 0; b < n_bonds; ++b) {
                model_->grp_alias.bond_W_rmax_all[g * n_bonds_pad + b] =
                    (env_W_max[b] > 0.0) ? 1.0 / env_W_max[b] : 0.0;
            }

            // Build alias table for this group
            std::vector<double> weights(max_alias);
            int n_a = 0;

            for (int i = 0; i < N_; ++i) {
                weights[n_a] = Omega / 2.0;
                n_a++;
            }
            for (int b = 0; b < n_bonds; ++b) {
                weights[n_a] = env_W_max[b];
                n_a++;
            }

            model_->grp_alias.n_alias_all[g] = n_a;

            // Vose's alias method
            double total = 0.0;
            for (int i = 0; i < n_a; ++i) total += weights[i];

            std::vector<double> prob_arr(n_a);
            std::vector<int32_t> alias_arr(n_a);
            for (int i = 0; i < n_a; ++i) {
                prob_arr[i] = weights[i] * n_a / total;
                alias_arr[i] = i;
            }

            std::vector<int> small_buf, large_buf;
            for (int i = 0; i < n_a; ++i) {
                if (prob_arr[i] < 1.0) small_buf.push_back(i);
                else large_buf.push_back(i);
            }
            while (!small_buf.empty() && !large_buf.empty()) {
                int s = small_buf.back(); small_buf.pop_back();
                int l = large_buf.back(); large_buf.pop_back();
                alias_arr[s] = l;
                prob_arr[l] -= (1.0 - prob_arr[s]);
                if (prob_arr[l] < 1.0) small_buf.push_back(l);
                else large_buf.push_back(l);
            }

            const size_t base = static_cast<size_t>(g) * max_alias;
            for (int i = 0; i < n_a; ++i) {
                model_->grp_alias.alias_prob[base + i] = prob_arr[i];
                if (model_->grp_alias.alias_u16)
                    model_->grp_alias.alias_idx16[base + i] = static_cast<uint16_t>(alias_arr[i]);
                else
                    model_->grp_alias.alias_idx32[base + i] = static_cast<uint32_t>(alias_arr[i]);
            }
        }
    }

    // Initialize operator string: all diagonal site ops
    op_types_.assign(M_total_, 1);
    const uint64_t max_location = static_cast<uint64_t>(
        std::max(N_, std::max(model_->vij.n_bonds, 1)) - 1);
    op_sites_u16_ = max_location <= std::numeric_limits<uint16_t>::max();
    if (op_sites_u16_) op_sites16_.assign(M_total_, 0);
    else               op_sites32_.assign(M_total_, 0);

    // State at midpoint for on-the-fly observables
    state_at_M_.assign(N_, 0);

    // Vertex-list scratch arrays
    site_op_count_.resize(N_, 0);
    site_op_head_.resize(N_, 0);
    site_bond_count_.resize(N_, 0);
    site_bond_head_.resize(N_, 0);
    bond_spin_.resize(M_total_, 0);
    spin_now_.resize(N_, 0);
}

QAQMCEngine::QAQMCEngine(std::shared_ptr<QAQMCModelData> model_data,
                         uint64_t seed)
    : N_(model_data ? model_data->N : 0),
      M_(model_data ? model_data->M : 0),
      M_total_(model_data ? model_data->M_total : 0),
      Omega_(model_data ? model_data->Omega : 0.0),
      Rb_(model_data ? model_data->Rb : 0.0),
      delta_min_(model_data ? model_data->delta_min : 0.0),
      delta_max_(model_data ? model_data->delta_max : 0.0),
      site_W_(model_data ? model_data->Omega / 2.0 : 0.0),
      site_W_max_(model_data ? model_data->Omega / 2.0 : 0.0),
      epsilon_(model_data ? model_data->epsilon : 0.0),
      delta_groups_(model_data ? model_data->delta_groups : 0),
      rng_(seed), model_(std::move(model_data)) {
    if (!model_ || N_ <= 0 || M_ <= 0 || M_total_ != 2 * M_
        || delta_groups_ <= 0) {
        throw std::invalid_argument(
            "QAQMCEngine: invalid or incomplete shared model data");
    }

    op_types_.assign(M_total_, 1);
    const uint64_t max_location = static_cast<uint64_t>(
        std::max(N_, std::max(model_->vij.n_bonds, 1)) - 1);
    op_sites_u16_ = max_location <= std::numeric_limits<uint16_t>::max();
    if (op_sites_u16_) op_sites16_.assign(M_total_, 0);
    else               op_sites32_.assign(M_total_, 0);

    state_at_M_.assign(N_, 0);
    site_op_count_.resize(N_, 0);
    site_op_head_.resize(N_, 0);
    site_bond_count_.resize(N_, 0);
    site_bond_head_.resize(N_, 0);
    bond_spin_.resize(M_total_, 0);
    spin_now_.resize(N_, 0);
}

uint64_t QAQMCModelData::logical_bytes() const {
    uint64_t bytes = 0;
    auto add = [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        bytes += static_cast<uint64_t>(values.size()) * sizeof(T);
    };
    add(vij.bonds_i);
    add(vij.bonds_j);
    add(vij.vij_list);
    add(vij.bond_sites_flat);
    add(vij.coord_number);
    add(vij.inv_coord);
    add(pos_flat);
    add(grp_alias.bond_W_rmax_all);
    add(grp_alias.n_alias_all);
    add(grp_alias.alias_prob);
    add(grp_alias.alias_idx16);
    add(grp_alias.alias_idx32);
    return bytes;
}

// ─── Checkpoint helpers ──────────────────────────────────────────────────────

std::vector<double> QAQMCEngine::get_delta_schedule() const {
    std::vector<double> out(M_total_);
    for (int p = 0; p < M_total_; ++p) out[p] = delta_at(p);
    return out;
}

void QAQMCEngine::export_op_types(int32_t* out, int len) const {
    if (len != M_total_)
        throw std::invalid_argument("export_op_types: output length mismatch");
    for (int p = 0; p < M_total_; ++p)
        out[p] = static_cast<int32_t>(op_types_[p]);
}

void QAQMCEngine::export_op_sites(int32_t* out, int len) const {
    if (len != M_total_)
        throw std::invalid_argument("export_op_sites: output length mismatch");
    for (int p = 0; p < M_total_; ++p)
        out[p] = static_cast<int32_t>(op_site_at(p));
}

void QAQMCEngine::set_bond_event_storage(const std::string& mode) {
    BondEventStorage requested;
    if (mode == "packed64") requested = BondEventStorage::Packed64;
    else if (mode == "p_bond16") {
        if (model_->vij.n_bonds > std::numeric_limits<uint16_t>::max()) {
            throw std::invalid_argument(
                "p_bond16 requires at most 65535 bonds");
        }
        requested = BondEventStorage::PositionBond16;
    }
    else if (mode == "p_only32") requested = BondEventStorage::Position32;
    else {
        throw std::invalid_argument(
            "bond event storage must be 'packed64', 'p_bond16', or "
            "'p_only32'");
    }
    if (requested == bond_event_storage_) return;
    bond_event_storage_ = requested;
    // Release the inactive O(M) scratch immediately. Counts stay valid, but
    // the next cluster step must refill the newly selected representation.
    std::vector<int64_t>().swap(site_bond_list_);
    std::vector<uint32_t>().swap(site_bond_p_list_);
    std::vector<uint16_t>().swap(site_bond_b16_list_);
}

std::string QAQMCEngine::get_bond_event_storage() const {
    if (bond_event_storage_ == BondEventStorage::Packed64) return "packed64";
    if (bond_event_storage_ == BondEventStorage::PositionBond16)
        return "p_bond16";
    return "p_only32";
}

std::map<std::string, uint64_t> QAQMCEngine::get_memory_breakdown() const {
    std::map<std::string, uint64_t> out;
    auto record = [&](const std::string& name, uint64_t size_bytes,
                      uint64_t capacity_bytes) {
        out[name + "_size_bytes"] = size_bytes;
        out[name + "_capacity_bytes"] = capacity_bytes;
    };
    auto vec_bytes = [&](const std::string& name, const auto& v) {
        using T = typename std::decay_t<decltype(v)>::value_type;
        record(name, static_cast<uint64_t>(v.size()) * sizeof(T),
               static_cast<uint64_t>(v.capacity()) * sizeof(T));
    };

    vec_bytes("alias_prob", model_->grp_alias.alias_prob);
    vec_bytes("alias_idx16", model_->grp_alias.alias_idx16);
    vec_bytes("alias_idx32", model_->grp_alias.alias_idx32);
    vec_bytes("bond_W_rmax", model_->grp_alias.bond_W_rmax_all);
    vec_bytes("n_alias", model_->grp_alias.n_alias_all);
    vec_bytes("op_types", op_types_);
    vec_bytes("op_sites16", op_sites16_);
    vec_bytes("op_sites32", op_sites32_);
    vec_bytes("bond_spin", bond_spin_);
    vec_bytes("site_op_list", site_op_list_);
    vec_bytes("site_bond_list", site_bond_list_);
    vec_bytes("site_bond_p_list", site_bond_p_list_);
    vec_bytes("site_bond_b16_list", site_bond_b16_list_);
    vec_bytes("site_op_count", site_op_count_);
    vec_bytes("site_op_head", site_op_head_);
    vec_bytes("site_bond_count", site_bond_count_);
    vec_bytes("site_bond_head", site_bond_head_);
    vec_bytes("spin_now", spin_now_);
    vec_bytes("seg_flipped", seg_flipped_);
    record("delta_schedule", 0, 0);

    uint64_t total_size = 0, total_capacity = 0;
    for (const auto& kv : out) {
        const std::string& key = kv.first;
        if (key.size() >= 11 && key.compare(key.size() - 11, 11, "_size_bytes") == 0)
            total_size += kv.second;
        else if (key.size() >= 15 &&
                 key.compare(key.size() - 15, 15, "_capacity_bytes") == 0)
            total_capacity += kv.second;
    }
    out["total_size_bytes"] = total_size;
    out["total_capacity_bytes"] = total_capacity;
    const uint64_t shared_table_size =
        out["alias_prob_size_bytes"]
        + out["alias_idx16_size_bytes"]
        + out["alias_idx32_size_bytes"]
        + out["bond_W_rmax_size_bytes"]
        + out["n_alias_size_bytes"];
    const uint64_t shared_table_capacity =
        out["alias_prob_capacity_bytes"]
        + out["alias_idx16_capacity_bytes"]
        + out["alias_idx32_capacity_bytes"]
        + out["bond_W_rmax_capacity_bytes"]
        + out["n_alias_capacity_bytes"];
    out["shared_table_size_bytes"] = shared_table_size;
    out["shared_table_capacity_bytes"] = shared_table_capacity;
    out["per_chain_size_bytes"] = total_size - shared_table_size;
    out["per_chain_capacity_bytes"] = total_capacity - shared_table_capacity;
    out["compact_op_sites"] = op_sites_u16_ ? 1 : 0;
    out["compact_alias_indices"] = model_->grp_alias.alias_u16 ? 1 : 0;
    out["shared_model_logical_bytes"] = model_->logical_bytes();
    out["shared_model_use_count"] = static_cast<uint64_t>(model_.use_count());
    out["bond_event_position_only"] =
        bond_event_storage_ == BondEventStorage::Position32 ? 1 : 0;
    out["bond_event_position_bond16"] =
        bond_event_storage_ == BondEventStorage::PositionBond16 ? 1 : 0;
    uint64_t site_ops = 0, bond_ops = 0, offdiag_ops = 0;
    for (const OpType type : op_types_) {
        if (type == 2) ++bond_ops;
        else {
            ++site_ops;
            if (type == -1) ++offdiag_ops;
        }
    }
    out["operator_slots"] = static_cast<uint64_t>(M_total_);
    out["site_operator_count"] = site_ops;
    out["bond_operator_count"] = bond_ops;
    out["offdiag_operator_count"] = offdiag_ops;
    return out;
}

std::string QAQMCEngine::get_rng_state() const {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
}

void QAQMCEngine::set_rng_state(const std::string& state_str) {
    std::istringstream iss(state_str);
    iss >> rng_;
}

void QAQMCEngine::set_op_string(const int32_t* types, const int32_t* sites, int len) {
    if (len != M_total_)
        throw std::invalid_argument("set_op_string: operator string length mismatch");
    for (int p = 0; p < len; ++p) {
        const int type = types[p];
        const int site = sites[p];
        if (type != -1 && type != 1 && type != 2)
            throw std::invalid_argument("set_op_string: invalid operator type");
        const int limit = (type == 2) ? model_->vij.n_bonds : N_;
        if (site < 0 || site >= limit)
            throw std::invalid_argument("set_op_string: operator location out of range");
    }
    for (int p = 0; p < len; ++p) {
        const int type = types[p];
        op_types_[p] = static_cast<OpType>(type);
        set_op_site_at(p, sites[p]);
    }
    // Externally-replaced ops invalidate the counts fused into the last
    // diagonal sweep; build_vertex_lists will re-count.
    vertex_counts_valid_ = false;
}

// Off-diagonal string (X_C) seam/half-line/topology_sweep methods
// (set_string_sites, recompute_seam_snapshots, build_half_line_proposal,
// commit_half_line_proposal, attempt_string_toggle, topology_sweep) live in
// qaqmc_off_diagonal_core.cpp -- see qaqmc_off_diagonal_core.hpp for why.

// ─── On-the-fly observable helpers ───────────────────────────────────────────

void QAQMCEngine::set_observable_sites(
        const std::vector<std::vector<int>>& loop_sets,
        const std::vector<std::vector<int>>& string_sets) {
    loop_site_sets_ = loop_sets;
    string_site_sets_ = string_sets;

    auto build_groups = [](const std::vector<std::vector<int>>& sets,
                           std::vector<int>& group_of,
                           std::vector<int>& group_n_copies) {
        group_of.assign(sets.size(), 0);
        group_n_copies.clear();
        std::vector<size_t> size_at_group; // set length identifying each group, in order
        for (size_t k = 0; k < sets.size(); ++k) {
            size_t sz = sets[k].size();
            int g = -1;
            for (size_t i = 0; i < size_at_group.size(); ++i) {
                if (size_at_group[i] == sz) { g = (int)i; break; }
            }
            if (g < 0) {
                g = (int)size_at_group.size();
                size_at_group.push_back(sz);
                group_n_copies.push_back(0);
            }
            group_of[k] = g;
            group_n_copies[g] += 1;
        }
    };
    build_groups(loop_site_sets_,   loop_group_of_,   loop_group_n_copies_);
    build_groups(string_site_sets_, string_group_of_, string_group_n_copies_);
}

void QAQMCEngine::set_bulk_sites(const std::vector<int>& bulk_sites) {
    bulk_sites_ = bulk_sites;
}

QAQMCEngine::MidpointObservables QAQMCEngine::measure_at_midpoint() const {
    MidpointObservables obs;

    // Density: average n_i over bulk sites (or all sites if bulk not set)
    if (!bulk_sites_.empty()) {
        int total = 0;
        for (int s : bulk_sites_) total += state_at_M_[s];
        obs.density = (double)total / (double)bulk_sites_.size();
    } else {
        int total = 0;
        for (int i = 0; i < N_; ++i) total += state_at_M_[i];
        obs.density = (double)total / N_;
    }

    // Z(l): mean over copies within each size group (signed, no |·|).
    const int n_zg = (int)loop_group_n_copies_.size();
    obs.Z_l_by_size.assign(n_zg, 0.0);
    for (size_t k = 0; k < loop_site_sets_.size(); ++k) {
        double prod = 1.0;
        for (int s : loop_site_sets_[k]) prod *= (1 - 2 * state_at_M_[s]);
        obs.Z_l_by_size[loop_group_of_[k]] += prod;
    }
    for (int g = 0; g < n_zg; ++g) {
        if (loop_group_n_copies_[g] > 0)
            obs.Z_l_by_size[g] /= (double)loop_group_n_copies_[g];
    }

    // C_m(l): mean over copies within each size group.
    const int n_cg = (int)string_group_n_copies_.size();
    obs.C_m_l_by_size.assign(n_cg, 0.0);
    for (size_t k = 0; k < string_site_sets_.size(); ++k) {
        double prod = 1.0;
        for (int s : string_site_sets_[k]) prod *= (1 - 2 * state_at_M_[s]);
        obs.C_m_l_by_size[string_group_of_[k]] += prod;
    }
    for (int g = 0; g < n_cg; ++g) {
        if (string_group_n_copies_[g] > 0)
            obs.C_m_l_by_size[g] /= (double)string_group_n_copies_[g];
    }

    return obs;
}

// ─── Asymmetric profile measurement ──────────────────────────────────────────
//
// Split into profile_alloc / profile_measure_point so the same measurement
// code serves both the standalone measure_profile() sweep and the fused
// capture inside diagonal_update_profiled().

void QAQMCEngine::profile_alloc(ProfileObservables& prof, int n_points) const {
    prof.n_points = n_points;
    prof.density.assign(n_points, 0.0);
    const int n_zg = (int)loop_group_n_copies_.size();
    const int n_cg = (int)string_group_n_copies_.size();
    const int n_q  = (int)dimer_q_points_.size();
    prof.Z_l_by_size.assign(n_zg,   std::vector<double>(n_points, 0.0));
    prof.C_m_l_by_size.assign(n_cg, std::vector<double>(n_points, 0.0));
    if (n_q > 0) {
        prof.s_q_real.assign(n_q, std::vector<double>(n_points, 0.0));
        prof.s_q_imag.assign(n_q, std::vector<double>(n_points, 0.0));
        prof.s_q_abs1.assign(n_q, std::vector<double>(n_points, 0.0));
        prof.s_q_abs2.assign(n_q, std::vector<double>(n_points, 0.0));
        prof.s_q_abs3.assign(n_q, std::vector<double>(n_points, 0.0));
        prof.s_q_abs4.assign(n_q, std::vector<double>(n_points, 0.0));
    }
    const int n_snap = (int)snapshot_point_indices_.size();
    if (n_snap > 0) prof.snapshots.assign(n_snap, std::vector<int8_t>(N_, 0));

    // VBS/SS order parameters at every profile point (if triangles configured).
    if (vbs_n_tri_ > 0) {
        prof.M_vbs.assign(n_points, 0.0);
        prof.M_ss.assign(n_points, 0.0);
    }

    // Occupation-SF bookkeeping: at requested points compute s_{q,α} (full+bulk).
    const int n_occ_pt = (int)occ_sf_point_indices_.size();
    const int n_occ_q  = (int)occ_q_points_.size();
    const int n_basis  = occ_n_basis_;
    if (n_occ_pt > 0 && n_occ_q > 0 && n_basis > 0) {
        const size_t vlen = (size_t)n_occ_q * n_basis;
        prof.occ_s_full_re.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
        prof.occ_s_full_im.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
        prof.occ_s_bulk_re.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
        prof.occ_s_bulk_im.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
        prof.occ_state.assign(n_occ_pt, std::vector<int8_t>(N_, 0));
        if (occ2_active_) {
            prof.occ2_s_re.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
            prof.occ2_s_im.assign(n_occ_pt, std::vector<double>(vlen, 0.0));
        }
    }
}

void QAQMCEngine::profile_measure_point(const std::vector<int32_t>& state, int out_idx,
                                        ProfileObservables& prof,
                                        size_t& next_snap, size_t& next_occ) const {
    const int n_zg = (int)loop_group_n_copies_.size();
    const int n_cg = (int)string_group_n_copies_.size();
    const int n_q  = (int)dimer_q_points_.size();
    const int n_occ_q = (int)occ_q_points_.size();
    const int n_basis = occ_n_basis_;

    // Density (bulk if set, else all sites)
    if (!bulk_sites_.empty()) {
        double s = 0.0;
        for (int si : bulk_sites_) s += state[si];
        prof.density[out_idx] = s / (double)bulk_sites_.size();
    } else {
        double s = 0.0;
        for (int i = 0; i < N_; ++i) s += state[i];
        prof.density[out_idx] = s / N_;
    }

    // Z_l: mean over copies per size group (signed, no |·|)
    for (size_t k = 0; k < loop_site_sets_.size(); ++k) {
        double prod = 1.0;
        for (int s : loop_site_sets_[k]) prod *= (1 - 2 * state[s]);
        prof.Z_l_by_size[loop_group_of_[k]][out_idx] += prod;
    }
    for (int g = 0; g < n_zg; ++g) {
        if (loop_group_n_copies_[g] > 0)
            prof.Z_l_by_size[g][out_idx] /= (double)loop_group_n_copies_[g];
    }

    // C_m_l: mean over copies per size group
    for (size_t k = 0; k < string_site_sets_.size(); ++k) {
        double prod = 1.0;
        for (int s : string_site_sets_[k]) prod *= (1 - 2 * state[s]);
        prof.C_m_l_by_size[string_group_of_[k]][out_idx] += prod;
    }
    for (int g = 0; g < n_cg; ++g) {
        if (string_group_n_copies_[g] > 0)
            prof.C_m_l_by_size[g][out_idx] /= (double)string_group_n_copies_[g];
    }

    // Dimer structure factor s_q = Σ_{i ∈ bulk} n_i e^{i q·r_i}.
    // Uses bulk_sites_ if non-empty (same convention as density),
    // else all N sites.  Same snapshot timing as density/Z/C — i.e.,
    // after applying op at slot p (asymmetric profile observable at
    // δ_sched_[p+1] in the dimer "before-op" convention).
    if (n_q > 0) {
        for (int qi = 0; qi < n_q; ++qi) {
            double sx = 0.0, sy = 0.0;
            const double* cosq = &dimer_phase_cos_[(size_t)qi * N_];
            const double* sinq = &dimer_phase_sin_[(size_t)qi * N_];
            if (!bulk_sites_.empty()) {
                for (int i : bulk_sites_) {
                    if (state[i]) { sx += cosq[i]; sy += sinq[i]; }
                }
            } else {
                for (int i = 0; i < N_; ++i) {
                    if (state[i]) { sx += cosq[i]; sy += sinq[i]; }
                }
            }
            double abs2 = sx * sx + sy * sy;
            double abs1 = std::sqrt(abs2);
            prof.s_q_real[qi][out_idx] = sx;
            prof.s_q_imag[qi][out_idx] = sy;
            prof.s_q_abs1[qi][out_idx] = abs1;
            prof.s_q_abs2[qi][out_idx] = abs2;
            prof.s_q_abs3[qi][out_idx] = abs2 * abs1;
            prof.s_q_abs4[qi][out_idx] = abs2 * abs2;
        }
    }

    // Snapshot the full state at requested profile points (all N sites,
    // same "after op p" convention as density/SF above).
    if (next_snap < snapshot_point_indices_.size()
        && snapshot_point_indices_[next_snap] == out_idx) {
        std::vector<int8_t>& snap = prof.snapshots[next_snap];
        for (int i = 0; i < N_; ++i) snap[i] = (int8_t)state[i];
        ++next_snap;
    }

    // Occupation-SF: at requested points, compute the sublattice-resolved
    // Fourier vectors s_{q,α} = Σ_{i:α(i)=α} n_i e^{i q·R_cell(i)} for the
    // full lattice and the bulk-complete-cell subset; also stash state.
    if (next_occ < occ_sf_point_indices_.size()
        && occ_sf_point_indices_[next_occ] == out_idx) {
        std::vector<double>& sfr = prof.occ_s_full_re[next_occ];
        std::vector<double>& sfi = prof.occ_s_full_im[next_occ];
        std::vector<double>& sbr = prof.occ_s_bulk_re[next_occ];
        std::vector<double>& sbi = prof.occ_s_bulk_im[next_occ];
        for (int qi = 0; qi < n_occ_q; ++qi) {
            const double* cosq = &occ_phase_cos_[(size_t)qi * N_];
            const double* sinq = &occ_phase_sin_[(size_t)qi * N_];
            const size_t base = (size_t)qi * n_basis;
            for (int i = 0; i < N_; ++i) {
                if (!state[i]) continue;
                int a = occ_site_basis_[i];
                double c = cosq[i], s = sinq[i];
                sfr[base + a] += c; sfi[base + a] += s;
                if (occ_site_in_bulk_[i]) { sbr[base + a] += c; sbi[base + a] += s; }
            }
        }
        // Second unit cell (triangle-pair): s_{q,α} over triangle cells.
        if (occ2_active_) {
            std::vector<double>& s2r = prof.occ2_s_re[next_occ];
            std::vector<double>& s2i = prof.occ2_s_im[next_occ];
            for (int qi = 0; qi < n_occ_q; ++qi) {
                const double* cosq = &occ2_phase_cos_[(size_t)qi * N_];
                const double* sinq = &occ2_phase_sin_[(size_t)qi * N_];
                const size_t base = (size_t)qi * n_basis;
                for (int i = 0; i < N_; ++i) {
                    if (!state[i]) continue;
                    int a = occ2_site_basis_[i];
                    if (a < 0) continue;
                    s2r[base + a] += cosq[i];
                    s2i[base + a] += sinq[i];
                }
            }
        }
        std::vector<int8_t>& st = prof.occ_state[next_occ];
        for (int i = 0; i < N_; ++i) st[i] = (int8_t)state[i];
        ++next_occ;
    }

    // VBS/SS order parameters (diagonal, at every profile point).
    if (vbs_n_tri_ > 0) {
        auto tri_state = [&](int t) -> int {
            const int* cc = &vbs_tri_corners_[3 * t];
            return 4 * state[cc[0]] + 2 * state[cc[1]] + state[cc[2]];
        };
        int s00 = tri_state(vbs_ref00_);
        int s10 = tri_state(vbs_ref10_);
        double g = (s10 == s00) ? 1.0 : -1.0;
        double mv = 0.0, ms = 0.0;
        for (int t = 0; t < vbs_n_tri_; ++t) {
            int s = tri_state(t);
            double u = (vbs_n1_parity_[t] == 0)
                           ? ((s == s00) ? 1.0 : -1.0)
                           : (((s == s10) ? 1.0 : -1.0) * g);
            mv += vbs_sign_[t] * u;
            ms += ss_sign_[t]  * u;
        }
        double inv = 1.0 / (double)vbs_n_tri_;
        prof.M_vbs[out_idx] = mv * inv;
        prof.M_ss[out_idx]  = ms * inv;
    }
}

QAQMCEngine::ProfileObservables QAQMCEngine::measure_profile(int profile_step) const {
    if (profile_step <= 0) profile_step = 1;
    int n_points = M_total_ / profile_step;

    ProfileObservables prof;
    profile_alloc(prof, n_points);
    size_t next_snap = 0;  // running pointer into sorted snapshot_point_indices_
    size_t next_occ = 0;   // running pointer into sorted occ_sf_point_indices_

    // Propagate state from left boundary |0...0>
    std::vector<int32_t> state(N_, 0);
    int out_idx = 0;

    for (int p = 0; p < M_total_ && out_idx < n_points; ++p) {
        // Off-diagonal operator: flip the site
        if (op_types_[p] == -1) state[op_site_at(p)] ^= 1;

        if ((p + 1) % profile_step == 0) {
            profile_measure_point(state, out_idx, prof, next_snap, next_occ);
            ++out_idx;
        }
    }

    return prof;
}

void QAQMCEngine::set_snapshot_point_indices(const std::vector<int>& point_indices) {
    snapshot_point_indices_ = point_indices;
    std::sort(snapshot_point_indices_.begin(), snapshot_point_indices_.end());
    snapshot_point_indices_.erase(
        std::unique(snapshot_point_indices_.begin(), snapshot_point_indices_.end()),
        snapshot_point_indices_.end());
    for (int idx : snapshot_point_indices_) {
        if (idx < 0) {
            throw std::runtime_error(
                "set_snapshot_point_indices: profile-point index must be >= 0");
        }
    }
}

// ─── Dimer (density-density) structure factor ───────────────────────────────

void QAQMCEngine::set_dimer_sf_q_points(
        const std::vector<std::vector<double>>& q_points) {
    if (model_->pos_flat.empty()) {
        throw std::runtime_error("set_dimer_sf_q_points: site positions not set");
    }
    const int n_q = static_cast<int>(q_points.size());
    dimer_q_points_ = q_points;
    dimer_phase_cos_.assign((size_t)n_q * (size_t)N_, 0.0);
    dimer_phase_sin_.assign((size_t)n_q * (size_t)N_, 0.0);

    for (int qi = 0; qi < n_q; ++qi) {
        const auto& q = q_points[qi];
        if ((int)q.size() != model_->pos_dim) {
            throw std::runtime_error(
                "set_dimer_sf_q_points: q vector dim mismatch with site pos_dim");
        }
        for (int i = 0; i < N_; ++i) {
            double qr = 0.0;
            for (int d = 0; d < model_->pos_dim; ++d) {
                qr += q[d] * model_->pos_flat[i * model_->pos_dim + d];
            }
            dimer_phase_cos_[(size_t)qi * N_ + i] = std::cos(qr);
            dimer_phase_sin_[(size_t)qi * N_ + i] = std::sin(qr);
        }
    }
}

// ─── Sublattice-resolved occupation structure factor ────────────────────────

void QAQMCEngine::set_occ_sf_site_map(
        const std::vector<std::vector<double>>& site_cell_R,
        const std::vector<int>& site_basis,
        const std::vector<int>& site_in_bulk_cell,
        int n_basis) {
    if ((int)site_cell_R.size() != N_ || (int)site_basis.size() != N_ ||
        (int)site_in_bulk_cell.size() != N_) {
        throw std::runtime_error("set_occ_sf_site_map: arrays must have length N");
    }
    occ_n_basis_ = n_basis;
    occ_site_basis_ = site_basis;
    occ_site_in_bulk_.assign(N_, 0);
    for (int i = 0; i < N_; ++i) occ_site_in_bulk_[i] = (int8_t)(site_in_bulk_cell[i] ? 1 : 0);
    // Stash cell positions for phase-table build when q-points are set.
    occ_cell_R_flat_.assign((size_t)N_ * model_->pos_dim, 0.0);
    for (int i = 0; i < N_; ++i) {
        if ((int)site_cell_R[i].size() != model_->pos_dim)
            throw std::runtime_error("set_occ_sf_site_map: cell_R dim mismatch");
        for (int d = 0; d < model_->pos_dim; ++d)
            occ_cell_R_flat_[(size_t)i * model_->pos_dim + d] = site_cell_R[i][d];
    }
}

void QAQMCEngine::set_occ_sf_q_points(
        const std::vector<std::vector<double>>& q_points) {
    if (occ_cell_R_flat_.empty()) {
        throw std::runtime_error("set_occ_sf_q_points: call set_occ_sf_site_map first");
    }
    const int n_q = static_cast<int>(q_points.size());
    occ_q_points_ = q_points;
    occ_phase_cos_.assign((size_t)n_q * (size_t)N_, 0.0);
    occ_phase_sin_.assign((size_t)n_q * (size_t)N_, 0.0);
    for (int qi = 0; qi < n_q; ++qi) {
        const auto& q = q_points[qi];
        if ((int)q.size() != model_->pos_dim)
            throw std::runtime_error("set_occ_sf_q_points: q dim mismatch with pos_dim");
        for (int i = 0; i < N_; ++i) {
            double qr = 0.0;  // phase uses the CELL Bravais position R, not r_i
            for (int d = 0; d < model_->pos_dim; ++d)
                qr += q[d] * occ_cell_R_flat_[(size_t)i * model_->pos_dim + d];
            occ_phase_cos_[(size_t)qi * N_ + i] = std::cos(qr);
            occ_phase_sin_[(size_t)qi * N_ + i] = std::sin(qr);
        }
    }
}

void QAQMCEngine::set_occ_sf_point_indices(const std::vector<int>& point_indices) {
    occ_sf_point_indices_ = point_indices;
    std::sort(occ_sf_point_indices_.begin(), occ_sf_point_indices_.end());
    occ_sf_point_indices_.erase(
        std::unique(occ_sf_point_indices_.begin(), occ_sf_point_indices_.end()),
        occ_sf_point_indices_.end());
    for (int idx : occ_sf_point_indices_)
        if (idx < 0) throw std::runtime_error("set_occ_sf_point_indices: index < 0");
}

void QAQMCEngine::set_occ2_sf_site_map(
        const std::vector<std::vector<double>>& site_cell_R,
        const std::vector<int>& site_basis, int n_basis) {
    if ((int)site_cell_R.size() != N_ || (int)site_basis.size() != N_)
        throw std::runtime_error("set_occ2_sf_site_map: arrays must have length N");
    if (occ_q_points_.empty())
        throw std::runtime_error("set_occ2_sf_site_map: call set_occ_sf_q_points first");
    if (n_basis != occ_n_basis_)
        throw std::runtime_error("set_occ2_sf_site_map: n_basis must match occ n_basis");
    occ2_site_basis_ = site_basis;
    const int n_q = (int)occ_q_points_.size();
    occ2_phase_cos_.assign((size_t)n_q * N_, 0.0);
    occ2_phase_sin_.assign((size_t)n_q * N_, 0.0);
    for (int qi = 0; qi < n_q; ++qi) {
        const auto& q = occ_q_points_[qi];
        for (int i = 0; i < N_; ++i) {
            if (site_basis[i] < 0) continue;      // excluded site
            double qr = 0.0;
            for (int d = 0; d < model_->pos_dim; ++d)
                qr += q[d] * site_cell_R[i][d];
            occ2_phase_cos_[(size_t)qi * N_ + i] = std::cos(qr);
            occ2_phase_sin_[(size_t)qi * N_ + i] = std::sin(qr);
        }
    }
    occ2_active_ = true;
}

void QAQMCEngine::set_vbs_triangles(const std::vector<int>& corners_flat,
                                    const std::vector<int>& n1_parity,
                                    const std::vector<int>& vbs_sign,
                                    const std::vector<int>& ss_sign,
                                    int ref00, int ref10) {
    int n_tri = (int)n1_parity.size();
    if ((int)corners_flat.size() != 3 * n_tri || (int)vbs_sign.size() != n_tri ||
        (int)ss_sign.size() != n_tri)
        throw std::runtime_error("set_vbs_triangles: array length mismatch");
    vbs_n_tri_ = n_tri;
    vbs_tri_corners_ = corners_flat;
    vbs_n1_parity_.assign(n_tri, 0);
    for (int t = 0; t < n_tri; ++t) vbs_n1_parity_[t] = (int8_t)(n1_parity[t] & 1);
    vbs_sign_.assign(n_tri, 1.0);
    ss_sign_.assign(n_tri, 1.0);
    for (int t = 0; t < n_tri; ++t) { vbs_sign_[t] = vbs_sign[t]; ss_sign_[t] = ss_sign[t]; }
    vbs_ref00_ = ref00; vbs_ref10_ = ref10;
}

void QAQMCEngine::set_dimer_sf_measure_p_indices(
        const std::vector<int>& p_indices) {
    dimer_p_indices_.clear();
    dimer_deltas_used_.clear();
    for (int p : p_indices) {
        if (p < 0 || p >= M_) {
            throw std::runtime_error(
                "set_dimer_sf_measure_p_indices: p must be in forward ramp [0, M)");
        }
        dimer_p_indices_.push_back(p);
    }
    // Sort ascending so propagation in measure_dimer_sf is a single forward sweep.
    std::sort(dimer_p_indices_.begin(), dimer_p_indices_.end());
    for (int p : dimer_p_indices_) {
        dimer_deltas_used_.push_back(delta_at(p));
    }
}

void QAQMCEngine::set_dimer_sf_measure_deltas(
        const std::vector<double>& target_deltas) {
    std::vector<int> p_idx;
    p_idx.reserve(target_deltas.size());
    for (double target : target_deltas) {
        // Forward ramp only: argmin over p ∈ [0, M).
        int best_p = 0;
        double best_diff = std::abs(delta_at(0) - target);
        for (int p = 1; p < M_; ++p) {
            double d = std::abs(delta_at(p) - target);
            if (d < best_diff) { best_diff = d; best_p = p; }
        }
        p_idx.push_back(best_p);
    }
    set_dimer_sf_measure_p_indices(p_idx);
}

QAQMCEngine::DimerSFSample QAQMCEngine::measure_dimer_sf() const {
    const int n_p = static_cast<int>(dimer_p_indices_.size());
    const int n_q = static_cast<int>(dimer_q_points_.size());
    DimerSFSample out;
    out.density.assign(n_p, 0.0);
    out.s_q_real.assign((size_t)n_p * (size_t)n_q, 0.0);
    out.s_q_imag.assign((size_t)n_p * (size_t)n_q, 0.0);
    out.s_q_abs2.assign((size_t)n_p * (size_t)n_q, 0.0);
    if (n_p == 0 || n_q == 0) return out;

    // Walk forward from p=0, flipping offdiag ops; at each target p, snapshot s_q.
    std::vector<int32_t> state(N_, 0);
    size_t next_idx = 0;
    const int target_max = dimer_p_indices_.back();

    for (int p = 0; p <= target_max; ++p) {
        // If this p is a measurement point, snapshot BEFORE applying op at p.
        // (state[i] here is n_i at "just after slice p-1 finished" = entering p.
        //  Equivalent to "state at p" in the trace boundary convention.)
        while (next_idx < dimer_p_indices_.size()
               && dimer_p_indices_[next_idx] == p) {
            // density (bulk if set, else all sites)
            const std::vector<int>& site_list =
                bulk_sites_.empty() ? std::vector<int>() : bulk_sites_;
            double dens = 0.0;
            if (!bulk_sites_.empty()) {
                for (int s : bulk_sites_) dens += state[s];
                out.density[next_idx] = dens / (double)bulk_sites_.size();
            } else {
                for (int i = 0; i < N_; ++i) dens += state[i];
                out.density[next_idx] = dens / (double)N_;
            }
            // s_q = Σ_{i ∈ bulk} n_i e^{i q·r_i} for each q.
            // If bulk_sites_ is non-empty, restrict the sum to the bulk
            // (matches the density convention above), so boundary atoms can
            // be excluded the same way as for ⟨n⟩.  Otherwise sum over all N.
            for (int qi = 0; qi < n_q; ++qi) {
                double sx = 0.0, sy = 0.0;
                const double* cosq = &dimer_phase_cos_[(size_t)qi * N_];
                const double* sinq = &dimer_phase_sin_[(size_t)qi * N_];
                if (!bulk_sites_.empty()) {
                    for (int i : bulk_sites_) {
                        if (state[i]) { sx += cosq[i]; sy += sinq[i]; }
                    }
                } else {
                    for (int i = 0; i < N_; ++i) {
                        if (state[i]) { sx += cosq[i]; sy += sinq[i]; }
                    }
                }
                size_t flat = (size_t)next_idx * (size_t)n_q + (size_t)qi;
                out.s_q_real[flat] = sx;
                out.s_q_imag[flat] = sy;
                out.s_q_abs2[flat] = sx * sx + sy * sy;
            }
            ++next_idx;
        }
        // Apply op at slot p to advance state for the next iteration.
        if (p < M_total_ && op_types_[p] == -1) {
            state[op_site_at(p)] ^= 1;
        }
    }
    (void)0;  // silence unused-var warnings if any
    return out;
}

// ─── Diagonal Update ─────────────────────────────────────────────────────────
//
// diagonal_update_profiled fuses three formerly-separate O(M) passes into the
// one forward sweep it already does:
//   1. the diagonal op re-insertion itself,
//   2. the per-site op/bond counting that build_vertex_lists' first pass used
//      to redo (counts are valid because each slot's op is final once this
//      sweep passes it, and cluster_update's 1 <-> -1 toggles are
//      count-neutral),
//   3. (optional, prof != nullptr) the asymmetric-profile measurement that
//      run_profile used to obtain via a separate measure_profile() sweep.
//      The diagonal update never changes off-diagonal ops, so the state
//      trajectory here is IDENTICAL to the one measure_profile would walk
//      after the previous cluster_update: capturing during this sweep yields
//      exactly the sample of the previous completed MC step.

void QAQMCEngine::diagonal_update() {
    diagonal_update_profiled(nullptr, 0);
}

void QAQMCEngine::diagonal_update_profiled(ProfileObservables* prof, int profile_step) {
    std::vector<int32_t> state(N_, 0); // boundary |0...0>
    const int* bond_sites = model_->vij.bond_sites_flat.data();

    // Grouped alias table path: shared per-group alias table for proposal,
    // real per-slice delta for rejection against the group envelope.
    int max_alias = model_->grp_alias.max_alias;
    int n_bonds_pad = model_->grp_alias.n_bonds_pad;

    // Incremental group tracker: reproduces g = floor(p*G/M_total) exactly
    // (g increments when p reaches ceil((g+1)*M_total/G)) without the
    // 4-byte-per-slice slice_to_group array.
    const int G = model_->grp_alias.n_groups;
    int g = 0;
    int64_t g_next = ((int64_t)M_total_ + G - 1) / G;

    // Fused vertex-list counting (build_vertex_lists' former first pass).
    std::fill(site_op_count_.begin(), site_op_count_.end(), 0);
    std::fill(site_bond_count_.begin(), site_bond_count_.end(), 0);

    // Fused profile capture bookkeeping.
    int out_idx = 0;
    int n_points = 0;
    size_t next_snap = 0, next_occ = 0;
    if (prof != nullptr) {
        if (profile_step <= 0) profile_step = 1;
        n_points = M_total_ / profile_step;
        profile_alloc(*prof, n_points);
    }

    for (int p = 0; p < M_total_; ++p) {
        // Capture state at symmetry point
        if (p == M_) std::memcpy(state_at_M_.data(), state.data(), N_ * sizeof(int32_t));

        // Off-diagonal string seam: no-op unless p == m_star_. See
        // QAQMCOffDiagonalCore::on_diagonal_slice (qaqmc_off_diagonal_core.hpp)
        // for the seam capture/XOR convention.
        off_diag_.on_diagonal_slice(p, state);

        while (p >= g_next) {
            ++g;
            g_next = ((int64_t)(g + 1) * M_total_ + G - 1) / G;
        }

        int ot = op_types_[p];
        if (ot == -1) {
            int s = op_site_at(p);
            state[s] ^= 1;
            site_op_count_[s]++;
        } else if (ot == 1 || ot == 2) {
            int n_alias_g = model_->grp_alias.n_alias_all[g];
            const size_t alias_base = static_cast<size_t>(g) * max_alias;
            const double* alias_prob = model_->grp_alias.alias_prob.data() + alias_base;
            const double* rmax_g = &model_->grp_alias.bond_W_rmax_all[(size_t)g * n_bonds_pad];
            // delta depends only on the slice: hoist out of the rejection loop.
            const double delta = delta_at(p);

            bool inserted = false;
            while (!inserted) {
                int i = randint(rng_, n_alias_g);
                int idx = (uniform01(rng_) < alias_prob[i])
                    ? i : model_->grp_alias.alias_at(alias_base + i);

                if (idx < N_) {
                    // Site op: always accept
                    op_types_[p] = 1;
                    set_op_site_at(p, idx);
                    site_op_count_[idx]++;
                    inserted = true;
                } else {
                    // Bond op: rejection with real per-slice weights
                    int b = idx - N_;
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    int w_idx = state[si] * 2 + state[sj];

                    // Compute actual weight for this slice's delta
                    double di = delta * model_->vij.inv_coord[si];
                    double dj = delta * model_->vij.inv_coord[sj];
                    double W[4], wmax_actual;
                    compute_bond_W_inline(di, dj, model_->vij.vij_list[b], epsilon_, W, wmax_actual);

                    // Reject against group envelope W_max (precomputed reciprocal;
                    // rmax == 0 when the envelope is <= 0, making the test false).
                    if (uniform01(rng_) < W[w_idx] * rmax_g[b]) {
                        op_types_[p] = 2;
                        set_op_site_at(p, b);
                        site_bond_count_[si]++;
                        site_bond_count_[sj]++;
                        inserted = true;
                    }
                }
            }
        }

        // Fused profile measurement: same "after applying op at slot p"
        // convention as measure_profile().
        if (prof != nullptr && out_idx < n_points && (p + 1) % profile_step == 0) {
            profile_measure_point(state, out_idx, *prof, next_snap, next_occ);
            ++out_idx;
        }
    }
    vertex_counts_valid_ = true;
}

// ─── build_vertex_lists  —  O(M + N) counting sort ──────────────────────────
//
// The counting pass is normally fused into diagonal_update_profiled (the op
// at each slot is final once the diagonal sweep passes it, and the cluster
// update's 1 <-> -1 toggles are count-neutral), so this only re-counts when
// the operator string was replaced externally (set_op_string).  The fill
// pass here is additionally fused with the former cluster Phase B: it
// propagates spin_now_ and records bond_spin_[p] in the same sweep.

template <typename T>
static void resize_scratch_with_bounded_headroom(std::vector<T>& v,
                                                 size_t required) {
    // std::vector::resize commonly doubles capacity when a stationary event
    // count fluctuates just above the previous high-water mark.  For an O(M)
    // bond-event list that can retain almost a second full copy forever.
    // These lists are scratch and overwritten immediately, so allocate a
    // small explicit cushion without copying the old contents.
    const size_t growth_headroom = std::max<size_t>(1024, required / 64); // ~1.6%
    const size_t shrink_slack = std::max<size_t>(4096, required / 8);     // 12.5%
    const bool must_grow = required > v.capacity();
    const bool materially_oversized = v.capacity() > required + shrink_slack;
    if (must_grow || materially_oversized) {
        std::vector<T> replacement;
        replacement.reserve(required + growth_headroom);
        replacement.resize(required);
        v.swap(replacement);
    } else {
        v.resize(required);
    }
}

void QAQMCEngine::build_vertex_lists() {
    const int M = M_total_, N = N_;
    const int* bond_sites = model_->vij.bond_sites_flat.data();

    if (!vertex_counts_valid_) {
        std::fill(site_op_count_.begin(), site_op_count_.end(), 0);
        std::fill(site_bond_count_.begin(), site_bond_count_.end(), 0);
        for (int p = 0; p < M; ++p) {
            int ot = op_types_[p];
            if (ot == 1 || ot == -1) {
                site_op_count_[op_site_at(p)]++;
            } else if (ot == 2) {
                int b  = op_site_at(p);
                site_bond_count_[bond_sites[b * 2 + 0]]++;
                site_bond_count_[bond_sites[b * 2 + 1]]++;
            }
        }
    }

    site_op_head_[0] = 0;
    site_bond_head_[0] = 0;
    for (int i = 1; i < N; ++i) {
        site_op_head_[i]   = site_op_head_[i-1]   + site_op_count_[i-1];
        site_bond_head_[i] = site_bond_head_[i-1]  + site_bond_count_[i-1];
    }
    int total_sops  = site_op_head_[N-1]   + site_op_count_[N-1];
    int total_bents = site_bond_head_[N-1] + site_bond_count_[N-1];

    resize_scratch_with_bounded_headroom(site_op_list_,
                                         static_cast<size_t>(total_sops));
    if (bond_event_storage_ == BondEventStorage::Packed64) {
        resize_scratch_with_bounded_headroom(site_bond_list_,
                                             static_cast<size_t>(total_bents));
        std::vector<uint32_t>().swap(site_bond_p_list_);
        std::vector<uint16_t>().swap(site_bond_b16_list_);
    } else {
        resize_scratch_with_bounded_headroom(site_bond_p_list_,
                                             static_cast<size_t>(total_bents));
        std::vector<int64_t>().swap(site_bond_list_);
        if (bond_event_storage_ == BondEventStorage::PositionBond16) {
            resize_scratch_with_bounded_headroom(
                site_bond_b16_list_, static_cast<size_t>(total_bents));
        } else {
            std::vector<uint16_t>().swap(site_bond_b16_list_);
        }
    }

    std::vector<int32_t> cur_op(N, 0);
    std::vector<int32_t> cur_bond(N, 0);

    // Fill pass + former Phase B (bond_spin_ from |0...0> propagation).
    std::fill(spin_now_.begin(), spin_now_.end(), 0);

    for (int p = 0; p < M; ++p) {
        // Off-diagonal string seam: same convention as diagonal_update(), so
        // bond_spin_[p] for p >= m_star_ reflects the post-seam occupation.
        off_diag_.on_cluster_slice(p, spin_now_);

        int ot = op_types_[p];
        if (ot == 1 || ot == -1) {
            int s = op_site_at(p);
            site_op_list_[site_op_head_[s] + cur_op[s]++] = p;
            if (ot == -1) spin_now_[s] ^= 1;
        } else if (ot == 2) {
            int b  = op_site_at(p);
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            bond_spin_[p] = spin_now_[si] * 2 + spin_now_[sj];
            if (bond_event_storage_ == BondEventStorage::Packed64) {
                site_bond_list_[site_bond_head_[si] + cur_bond[si]++] =
                    pack_bond_entry(p, b, 0);
                site_bond_list_[site_bond_head_[sj] + cur_bond[sj]++] =
                    pack_bond_entry(p, b, 1);
            } else {
                const int si_index = site_bond_head_[si] + cur_bond[si]++;
                const int sj_index = site_bond_head_[sj] + cur_bond[sj]++;
                site_bond_p_list_[si_index] = static_cast<uint32_t>(p);
                site_bond_p_list_[sj_index] = static_cast<uint32_t>(p);
                if (bond_event_storage_ == BondEventStorage::PositionBond16) {
                    site_bond_b16_list_[si_index] = static_cast<uint16_t>(b);
                    site_bond_b16_list_[sj_index] = static_cast<uint16_t>(b);
                }
            }
        }
    }
}

// ─── Cluster Update  —  O(M) via vertex lists ──────────────────────────────
//
// QAQMC uses OPEN boundary conditions:
//   - boundary state is |0...0> at both tau=0 and tau=M_total_
//   - segments touching the boundary (first/last per site) are FROZEN
//
// Phase A: build per-site vertex lists
// Phase B: single O(M) pass to record bond_spin_[p]
// Phase C: per-site segment Metropolis using per-site bond-op lists
// Phase D: reassign op_types using segment flip decisions

void QAQMCEngine::cluster_update() {
    if (M_total_ == 0) return;

    const int* bond_sites = model_->vij.bond_sites_flat.data();
    const int  M = M_total_, N = N_;

    // ── Phase A (+ fused former Phase B: bond_spin_ fill) ───────────────────
    build_vertex_lists();

    // ── Phase C: per-site segment processing ─────────────────────────────────

    // Metropolis accept test for flipping the segment: accumulates the plain
    // product of weight ratios Π w_new/w_old (no std::log in the hot loop —
    // exactly the same accept probability as the previous log-sum form).
    // Zero weights are tallied separately so "excluded → allowed" (inf) and
    // "allowed → excluded" (0) factors can't produce inf*0 = NaN; the running
    // product is renormalised by 1e±100 so long segments can't over/underflow.
    auto decode_event = [&](int site_i, int absolute_index,
                            int& p, int& b, int& ep) {
        if (bond_event_storage_ == BondEventStorage::Packed64) {
            const int64_t event = site_bond_list_[absolute_index];
            p = bond_entry_p(event);
            b = bond_entry_b(event);
            ep = bond_entry_endpoint(event);
        } else {
            p = static_cast<int>(site_bond_p_list_[absolute_index]);
            b = bond_event_storage_ == BondEventStorage::PositionBond16
                ? static_cast<int>(site_bond_b16_list_[absolute_index])
                : op_site_at(p);
            ep = bond_sites[b * 2] == site_i ? 0 : 1;
        }
    };

    auto accept_for_range = [&](int site_i, int bop_base,
                                int j_begin, int j_end) -> bool {
        double ratio = 1.0;
        int shift = 0;      // ratio_total = ratio * 1e100^shift
        int inf_ct = 0;     // factors with w_old == 0 < w_new  (force-accept)
        int zero_ct = 0;    // factors with w_new == 0 < w_old  (force-reject)
        for (int j = j_begin; j < j_end; ++j) {
            int p, b, ep;
            decode_event(site_i, bop_base + j, p, b, ep);
            const int si = bond_sites[b * 2 + 0];
            const int sj = bond_sites[b * 2 + 1];
            const int w_idx = bond_spin_[p];
            const int new_idx = w_idx ^ (ep == 0 ? 2 : 1);

            double delta = delta_at(p);
            double di = delta * model_->vij.inv_coord[si];
            double dj = delta * model_->vij.inv_coord[sj];
            double W[4], wmax;
            compute_bond_W_inline(di, dj, model_->vij.vij_list[b], epsilon_, W, wmax);
            const double w_old = W[w_idx];
            const double w_new = W[new_idx];

            if (w_new > 1e-300) {
                if (w_old > 1e-300) {
                    ratio *= w_new / w_old;
                    if (ratio > 1e100)       { ratio *= 1e-100; ++shift; }
                    else if (ratio < 1e-100) { ratio *= 1e100;  --shift; }
                } else {
                    ++inf_ct;
                }
            } else if (w_old > 1e-300) {
                ++zero_ct;
            }
            // w_new == w_old == 0: neutral factor (matches the old
            // (-1e30) - (-1e30) = 0 convention).
        }
        if (inf_ct != zero_ct) return inf_ct > zero_ct;
        if (shift >= 1) return true;
        if (shift <= -2) return false;
        const double r = (shift == -1) ? ratio * 1e-100 : ratio;
        return (r >= 1.0) || (uniform01(rng_) < r);
    };

    // Helper: flip site_i's bit in bond_spin_[p] for range [j_begin, j_end)
    auto flip_bond_range = [&](int site_i, int bop_base,
                               int j_begin, int j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            int p, b, ep;
            decode_event(site_i, bop_base + j, p, b, ep);
            bond_spin_[p] ^= (ep == 0 ? 2 : 1);
        }
    };

    auto upper_bound_idx = [&](int bop_base, int n_bops, int val) -> int {
        if (bond_event_storage_ == BondEventStorage::Packed64) {
            const int64_t* base = site_bond_list_.data() + bop_base;
            // Largest packed value with entry_p == val.
            const int64_t key =
                (static_cast<int64_t>(val) << 32) | 0xFFFFFFFFll;
            return static_cast<int>(
                std::upper_bound(base, base + n_bops, key) - base);
        }
        const uint32_t* base = site_bond_p_list_.data() + bop_base;
        return static_cast<int>(std::upper_bound(
            base, base + n_bops, static_cast<uint32_t>(val)) - base);
    };

    for (int site_i = 0; site_i < N; ++site_i) {
        int n_sops   = site_op_count_[site_i];
        int sop_base = site_op_head_[site_i];
        int n_bops   = site_bond_count_[site_i];
        int bop_base = site_bond_head_[site_i];

        if (n_sops == 0) continue;  // no single-site ops → nothing to flip

        // Segments: between consecutive single-site ops, plus boundary sentinels.
        // With sentinels at -1 and M_total_, there are (n_sops + 1) segments.
        // Boundary segments (first and last) are FROZEN.
        // So we process segments 1..n_sops-1 (i.e., between op[k] and op[k+1]).

        seg_flipped_.assign(n_sops + 1, 0);  // n_sops+1 segments total

        for (int seg = 1; seg < n_sops; ++seg) {
            // Segment between single-site op (seg-1) and op (seg)
            int p_start = site_op_list_[sop_base + seg - 1];
            int p_end   = site_op_list_[sop_base + seg];

            // Bond ops in (p_start, p_end] — no wrapping needed (open BC)
            int j0 = upper_bound_idx(bop_base, n_bops, p_start);
            int j1 = upper_bound_idx(bop_base, n_bops, p_end);

            if (accept_for_range(site_i, bop_base, j0, j1)) {
                flip_bond_range(site_i, bop_base, j0, j1);
                seg_flipped_[seg] = 1;
            }
        }

        // ── Phase D: reassign op_types for this site's single-site ops ───────
        // seg_flipped_[0] = 0 (frozen boundary), seg_flipped_[n_sops] = 0 (frozen boundary)
        // For op k (0-indexed): seg_before = k, seg_after = k+1
        for (int k = 0; k < n_sops; ++k) {
            int seg_before = k;       // segment ending at op k
            int seg_after  = k + 1;   // segment starting at op k
            bool flip_xor = seg_flipped_[seg_before] != seg_flipped_[seg_after];
            if (flip_xor) {
                int p_op = site_op_list_[sop_base + k];
                op_types_[p_op] = (op_types_[p_op] == 1) ? -1 : 1;
            }
        }
    }
}

// ─── mc_step ─────────────────────────────────────────────────────────────────

#include <chrono>

void QAQMCEngine::mc_step() {
    auto t0 = std::chrono::high_resolution_clock::now();
    diagonal_update();
    auto t1 = std::chrono::high_resolution_clock::now();
    cluster_update();
    auto t2 = std::chrono::high_resolution_clock::now();

    time_diag_ += std::chrono::duration<double>(t1 - t0).count();
    time_clus_ += std::chrono::duration<double>(t2 - t1).count();
    mc_steps_++;
}

QAQMCEngine::ProfileObservables QAQMCEngine::mc_step_profiled(int profile_step) {
    // Fused mc_step + profile measurement.  The capture rides the diagonal
    // sweep, and since diagonal_update never touches off-diagonal ops, the
    // measured trajectory is exactly the configuration left by the PREVIOUS
    // completed mc_step — i.e. this returns the sample that
    //   mc_step(); measure_profile(profile_step);
    // of the previous iteration would have produced, at the cost of zero
    // extra O(M) passes.  (One-step lag; statistically identical for
    // equilibrium sampling.)
    ProfileObservables prof;
    auto t0 = std::chrono::high_resolution_clock::now();
    diagonal_update_profiled(&prof, profile_step);
    auto t1 = std::chrono::high_resolution_clock::now();
    cluster_update();
    auto t2 = std::chrono::high_resolution_clock::now();

    time_diag_ += std::chrono::duration<double>(t1 - t0).count();
    time_clus_ += std::chrono::duration<double>(t2 - t1).count();
    mc_steps_++;
    return prof;
}
