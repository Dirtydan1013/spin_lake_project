#include "qaqmc_core.hpp"
#include <cstring>
#include <cassert>
#include <sstream>
#include <stdexcept>

// ─── Helper: uniform random [0, 1) ──────────────────────────────────────────
static inline double uniform01(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}
static inline int randint(std::mt19937_64& rng, int n) {
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(rng);
}

// ═════════════════════════════════════════════════════════════════════════════
// V_ij builder
// ═════════════════════════════════════════════════════════════════════════════

RydbergVij build_rydberg_vij(int N, double Omega, double Rb,
                              const double* pos, int pos_dim,
                              int neighbor_cutoff) {
    // Step 1: Compute all pairwise distances
    std::vector<double> all_dists;
    all_dists.reserve(N * (N - 1) / 2);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dist2 = 0.0;
            for (int d = 0; d < pos_dim; ++d) {
                double diff = pos[i * pos_dim + d] - pos[j * pos_dim + d];
                dist2 += diff * diff;
            }
            all_dists.push_back(std::sqrt(dist2));
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
            double delta_i = (coord_number[si] > 0) ? delta / coord_number[si] : 0.0;
            double delta_j = (coord_number[sj] > 0) ? delta / coord_number[sj] : 0.0;

            double W[4], bmax;
            QAQMCEngine::compute_bond_W_inline(delta_i, delta_j, vij, epsilon, W, bmax);

            int base = (pi * n_bonds_pad + b) * 4;
            res.bond_W_all[base + 0] = W[0];
            res.bond_W_all[base + 1] = W[1];
            res.bond_W_all[base + 2] = W[2];
            res.bond_W_all[base + 3] = W[3];

            res.bond_W_max_all[pi * n_bonds_pad + b] = bmax;

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
                          int neighbor_cutoff, int delta_groups)
    : N_(N), M_(M), M_total_(2 * M),
      Omega_(Omega), Rb_(Rb), delta_min_(delta_min), delta_max_(delta_max),
      epsilon_(epsilon), delta_groups_(delta_groups),
      rng_(seed)
{
    if (delta_groups_ <= 0) {
        throw std::invalid_argument(
            "QAQMCEngine: delta_groups must be > 0 (the engine only supports the grouped "
            "alias-table path; the legacy precompute/chunk/on-the-fly modes were removed).");
    }
    site_W_ = Omega / 2.0;
    site_W_max_ = Omega / 2.0;

    // Build V_ij (with optional neighbor cutoff)
    vij_ = build_rydberg_vij(N, Omega, Rb, pos, pos_dim, neighbor_cutoff);

    // Cache positions for downstream observables (dimer structure factor etc.)
    pos_dim_ = pos_dim;
    pos_flat_.assign(pos, pos + N * pos_dim);

    // Build delta schedule: delta_min -> delta_max -> delta_min
    delta_sched_.resize(M_total_);
    for (int p = 0; p < M_; ++p) {
        delta_sched_[p] = delta_min + (delta_max - delta_min) * ((double)p / M_);
    }
    for (int p = M_; p < M_total_; ++p) {
        delta_sched_[p] = delta_max - (delta_max - delta_min) * ((double)(p - M_) / M_);
    }

    // Build grouped alias tables
    {
        int G = delta_groups_;
        int n_bonds = vij_.n_bonds;
        int n_bonds_pad = std::max(n_bonds, 1);
        int max_alias = N_ + n_bonds;
        const int* bond_si = vij_.bonds_i.data();
        const int* bond_sj = vij_.bonds_j.data();
        const double* bond_vij = vij_.vij_list.data();
        const int* coord_num = vij_.coord_number.data();

        grp_alias_.n_groups = G;
        grp_alias_.max_alias = max_alias;
        grp_alias_.n_bonds_pad = n_bonds_pad;
        grp_alias_.slice_to_group.resize(M_total_);

        // Assign each slice to a group (uniform partition)
        for (int p = 0; p < M_total_; ++p) {
            int g = (int)((int64_t)p * G / M_total_);
            if (g >= G) g = G - 1;
            grp_alias_.slice_to_group[p] = g;
        }

        // For each group, find the range of delta values and compute
        // envelope W_max (upper bound) for rejection sampling.
        grp_alias_.bond_W_max_all.assign(G * n_bonds_pad, 0.0);
        grp_alias_.n_alias_all.resize(G, 0);
        grp_alias_.alias_prob_all.assign(G * max_alias, 0.0);
        grp_alias_.alias_idx_all.assign(G * max_alias, 0);
        grp_alias_.op_map_kind_all.assign(G * max_alias, 0);
        grp_alias_.op_map_loc_all.assign(G * max_alias, 0);

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
            sample_deltas.push_back(delta_sched_[p_lo]);
            sample_deltas.push_back(delta_sched_[std::min(p_hi - 1, M_total_ - 1)]);
            int p_mid = (p_lo + p_hi) / 2;
            if (p_mid < M_total_) sample_deltas.push_back(delta_sched_[p_mid]);

            std::vector<double> env_W_max(n_bonds, 0.0);
            for (double delta : sample_deltas) {
                for (int b = 0; b < n_bonds; ++b) {
                    double di = (coord_num[bond_si[b]] > 0) ? delta / coord_num[bond_si[b]] : 0.0;
                    double dj = (coord_num[bond_sj[b]] > 0) ? delta / coord_num[bond_sj[b]] : 0.0;
                    double W[4], wmax;
                    compute_bond_W_inline(di, dj, bond_vij[b], epsilon_, W, wmax);
                    if (wmax > env_W_max[b]) env_W_max[b] = wmax;
                }
            }

            // Store envelope W_max
            for (int b = 0; b < n_bonds; ++b) {
                grp_alias_.bond_W_max_all[g * n_bonds_pad + b] = env_W_max[b];
            }

            // Build alias table for this group
            std::vector<double> weights(max_alias);
            std::vector<int> op_kind(max_alias);
            std::vector<int> op_loc(max_alias);
            int n_a = 0;

            for (int i = 0; i < N_; ++i) {
                weights[n_a] = Omega / 2.0;
                op_kind[n_a] = 0;
                op_loc[n_a] = i;
                n_a++;
            }
            for (int b = 0; b < n_bonds; ++b) {
                weights[n_a] = env_W_max[b];
                op_kind[n_a] = 1;
                op_loc[n_a] = b;
                n_a++;
            }

            grp_alias_.n_alias_all[g] = n_a;
            for (int i = 0; i < n_a; ++i) {
                grp_alias_.op_map_kind_all[g * max_alias + i] = op_kind[i];
                grp_alias_.op_map_loc_all[g * max_alias + i] = op_loc[i];
            }

            // Vose's alias method
            double total = 0.0;
            for (int i = 0; i < n_a; ++i) total += weights[i];

            std::vector<double> prob_arr(n_a);
            std::vector<int64_t> alias_arr(n_a);
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

            for (int i = 0; i < n_a; ++i) {
                grp_alias_.alias_prob_all[g * max_alias + i] = prob_arr[i];
                grp_alias_.alias_idx_all[g * max_alias + i] = alias_arr[i];
            }
        }
    }

    // Initialize operator string: all diagonal site ops
    op_types_.assign(M_total_, 1);
    op_sites_.assign(M_total_, 0);

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

// ─── Checkpoint helpers ──────────────────────────────────────────────────────

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
    if (len != M_total_) return;
    std::memcpy(op_types_.data(), types, len * sizeof(int32_t));
    std::memcpy(op_sites_.data(), sites, len * sizeof(int32_t));
}

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

QAQMCEngine::ProfileObservables QAQMCEngine::measure_profile(int profile_step) const {
    if (profile_step <= 0) profile_step = 1;
    int n_points = M_total_ / profile_step;

    ProfileObservables prof;
    prof.n_points = n_points;
    prof.density.resize(n_points, 0.0);
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
    // Snapshot bookkeeping: copy full state at requested profile points.
    const int n_snap = (int)snapshot_point_indices_.size();
    if (n_snap > 0) prof.snapshots.assign(n_snap, std::vector<int8_t>(N_, 0));
    size_t next_snap = 0;  // running pointer into sorted snapshot_point_indices_

    // Propagate state from left boundary |0...0>
    std::vector<int32_t> state(N_, 0);
    int out_idx = 0;

    for (int p = 0; p < M_total_ && out_idx < n_points; ++p) {
        // Off-diagonal operator: flip the site
        if (op_types_[p] == -1) state[op_sites_[p]] ^= 1;

        if ((p + 1) % profile_step == 0) {
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
    if (pos_flat_.empty()) {
        throw std::runtime_error("set_dimer_sf_q_points: site positions not set");
    }
    const int n_q = static_cast<int>(q_points.size());
    dimer_q_points_ = q_points;
    dimer_phase_cos_.assign((size_t)n_q * (size_t)N_, 0.0);
    dimer_phase_sin_.assign((size_t)n_q * (size_t)N_, 0.0);

    for (int qi = 0; qi < n_q; ++qi) {
        const auto& q = q_points[qi];
        if ((int)q.size() != pos_dim_) {
            throw std::runtime_error(
                "set_dimer_sf_q_points: q vector dim mismatch with site pos_dim");
        }
        for (int i = 0; i < N_; ++i) {
            double qr = 0.0;
            for (int d = 0; d < pos_dim_; ++d) {
                qr += q[d] * pos_flat_[i * pos_dim_ + d];
            }
            dimer_phase_cos_[(size_t)qi * N_ + i] = std::cos(qr);
            dimer_phase_sin_[(size_t)qi * N_ + i] = std::sin(qr);
        }
    }
}

void QAQMCEngine::set_dimer_sf_measure_p_indices(
        const std::vector<int>& p_indices) {
    dimer_p_indices_.clear();
    dimer_deltas_used_.clear();
    if (delta_sched_.empty()) {
        throw std::runtime_error("set_dimer_sf_measure_p_indices: delta schedule not built");
    }
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
        dimer_deltas_used_.push_back(delta_sched_[p]);
    }
}

void QAQMCEngine::set_dimer_sf_measure_deltas(
        const std::vector<double>& target_deltas) {
    if (delta_sched_.empty()) {
        throw std::runtime_error("set_dimer_sf_measure_deltas: delta schedule not built");
    }
    std::vector<int> p_idx;
    p_idx.reserve(target_deltas.size());
    for (double target : target_deltas) {
        // Forward ramp only: argmin over p ∈ [0, M).
        int best_p = 0;
        double best_diff = std::abs(delta_sched_[0] - target);
        for (int p = 1; p < M_; ++p) {
            double d = std::abs(delta_sched_[p] - target);
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
            state[op_sites_[p]] ^= 1;
        }
    }
    (void)0;  // silence unused-var warnings if any
    return out;
}

// ─── Diagonal Update ─────────────────────────────────────────────────────────

void QAQMCEngine::diagonal_update() {
    std::vector<int32_t> state(N_, 0); // boundary |0...0>
    const int* bond_sites = vij_.bond_sites_flat.data();

    // Grouped alias table path: shared per-group alias table for proposal,
    // real per-slice delta for rejection against the group envelope.
    int max_alias = grp_alias_.max_alias;
    int n_bonds_pad = grp_alias_.n_bonds_pad;

    for (int p = 0; p < M_total_; ++p) {
        // Capture state at symmetry point
        if (p == M_) std::memcpy(state_at_M_.data(), state.data(), N_ * sizeof(int32_t));

        int ot = op_types_[p];
        if (ot == -1) {
            state[op_sites_[p]] ^= 1;
        } else if (ot == 1 || ot == 2) {
            int g = grp_alias_.slice_to_group[p];
            int n_alias_g = grp_alias_.n_alias_all[g];

            bool inserted = false;
            while (!inserted) {
                int i = randint(rng_, n_alias_g);
                int idx;
                if (uniform01(rng_) < grp_alias_.alias_prob_all[g * max_alias + i])
                    idx = i;
                else
                    idx = (int)grp_alias_.alias_idx_all[g * max_alias + i];

                int kind = grp_alias_.op_map_kind_all[g * max_alias + idx];
                int loc  = grp_alias_.op_map_loc_all[g * max_alias + idx];

                if (kind == 0) {
                    // Site op: always accept
                    op_types_[p] = 1;
                    op_sites_[p] = loc;
                    inserted = true;
                } else {
                    // Bond op: rejection with real per-slice weights
                    int b = loc;
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    int w_idx = state[si] * 2 + state[sj];

                    // Compute actual weight for this slice's delta
                    double delta = delta_sched_[p];
                    double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
                    double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
                    double W[4], wmax_actual;
                    compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax_actual);

                    // Reject against group envelope W_max
                    double w_max_env = grp_alias_.bond_W_max_all[g * n_bonds_pad + b];
                    if (w_max_env > 0.0 && uniform01(rng_) < W[w_idx] / w_max_env) {
                        op_types_[p] = 2;
                        op_sites_[p] = b;
                        inserted = true;
                    }
                }
            }
        }
    }
}

// ─── build_vertex_lists  —  O(M + N) two-pass counting sort ─────────────────

void QAQMCEngine::build_vertex_lists() {
    const int M = M_total_, N = N_;
    const int* bond_sites = vij_.bond_sites_flat.data();

    std::fill(site_op_count_.begin(), site_op_count_.end(), 0);
    std::fill(site_bond_count_.begin(), site_bond_count_.end(), 0);

    for (int p = 0; p < M; ++p) {
        int ot = op_types_[p];
        if (ot == 1 || ot == -1) {
            site_op_count_[op_sites_[p]]++;
        } else if (ot == 2) {
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            site_bond_count_[si]++;
            site_bond_count_[sj]++;
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

    site_op_list_.resize(total_sops);
    site_bond_list_.resize(total_bents);

    std::vector<int32_t> cur_op(N, 0);
    std::vector<int32_t> cur_bond(N, 0);

    for (int p = 0; p < M; ++p) {
        int ot = op_types_[p];
        if (ot == 1 || ot == -1) {
            int s = op_sites_[p];
            site_op_list_[site_op_head_[s] + cur_op[s]++] = p;
        } else if (ot == 2) {
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            site_bond_list_[site_bond_head_[si] + cur_bond[si]++] = p;
            site_bond_list_[site_bond_head_[sj] + cur_bond[sj]++] = p;
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

    const int* bond_sites = vij_.bond_sites_flat.data();
    const int  M = M_total_, N = N_;

    // ── Phase A ──────────────────────────────────────────────────────────────
    build_vertex_lists();

    // ── Phase B: compute bond_spin_[p] ───────────────────────────────────────
    //   Propagate from |0...0> boundary at tau=0
    std::fill(spin_now_.begin(), spin_now_.end(), 0);  // |0...0>
    for (int p = 0; p < M; ++p) {
        int ot = op_types_[p];
        if (ot == 2) {
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            bond_spin_[p] = spin_now_[si] * 2 + spin_now_[sj];
        } else if (ot == -1) {
            spin_now_[op_sites_[p]] ^= 1;
        }
    }

    // ── Phase C: per-site segment processing ─────────────────────────────────

    // Helper: compute log(W_new/W_old) for bond ops in [j_begin, j_end)
    auto lr_for_range = [&](int site_i, int bop_base, int j_begin, int j_end) -> double {
        double lr = 0.0;
        for (int j = j_begin; j < j_end; ++j) {
            int p  = site_bond_list_[bop_base + j];
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            int w_idx = bond_spin_[p];
            int ni = w_idx >> 1, nj = w_idx & 1;

            double w_old, w_new;
            {
                double delta = delta_sched_[p];
                double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
                double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
                double W[4], wmax;
                compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
                w_old = W[w_idx];
                if (si == site_i)
                    w_new = W[(1 - ni) * 2 + nj];
                else
                    w_new = W[ni * 2 + (1 - nj)];
            }

            lr += ((w_new > 1e-300) ? std::log(w_new) : -1e30)
                - ((w_old > 1e-300) ? std::log(w_old) : -1e30);
        }
        return lr;
    };

    // Helper: flip site_i bit in bond_spin_[p] for range [j_begin, j_end)
    auto flip_bond_range = [&](int site_i, int bop_base, int j_begin, int j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            int p  = site_bond_list_[bop_base + j];
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            if (si == site_i)
                bond_spin_[p] ^= 2;
            else
                bond_spin_[p] ^= 1;
        }
    };

    auto upper_bound_idx = [&](int bop_base, int n_bops, int val) -> int {
        const int32_t* base = site_bond_list_.data() + bop_base;
        return (int)(std::upper_bound(base, base + n_bops, val) - base);
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

            double lr = lr_for_range(site_i, bop_base, j0, j1);
            bool do_flip = (lr >= 0.0) || (uniform01(rng_) < std::exp(lr));

            if (do_flip) {
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