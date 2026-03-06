#include "qaqmc_core.hpp"
#include <cstring>
#include <cassert>
#include <sstream>

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
                          int neighbor_cutoff, bool precompute,
                          int chunk_slices)
    : N_(N), M_(M), M_total_(2 * M),
      Omega_(Omega), Rb_(Rb), delta_min_(delta_min), delta_max_(delta_max),
      epsilon_(epsilon), precompute_(precompute), chunk_slices_(chunk_slices),
      rng_(seed)
{
    site_W_ = Omega / 2.0;
    site_W_max_ = Omega / 2.0;

    // Build V_ij (with optional neighbor cutoff)
    vij_ = build_rydberg_vij(N, Omega, Rb, pos, pos_dim, neighbor_cutoff);

    // Build delta schedule: delta_min -> delta_max -> delta_min
    delta_sched_.resize(M_total_);
    for (int p = 0; p < M_; ++p) {
        delta_sched_[p] = delta_min + (delta_max - delta_min) * ((double)p / M_);
    }
    for (int p = M_; p < M_total_; ++p) {
        delta_sched_[p] = delta_max - (delta_max - delta_min) * ((double)(p - M_) / M_);
    }

    // Build alias tables:
    // - precompute && chunk_slices==0: full precompute (original)
    // - precompute && chunk_slices>0:  chunked, built on demand in diagonal_update
    // - !precompute:                   on-the-fly (no table)
    if (precompute_ && chunk_slices_ <= 0) {
        alias_ = build_qaqmc_alias_tables(M_total_, N_, vij_.n_bonds, Omega_,
                                           delta_sched_.data(), vij_.vij_list.data(),
                                           vij_.bonds_i.data(), vij_.bonds_j.data(),
                                           vij_.coord_number.data(),
                                           epsilon);
    } else {
        // Minimal init for alias_ so accessors don't crash
        alias_.n_bonds_pad = std::max(vij_.n_bonds, 1);
        alias_.max_alias = N_ + vij_.n_bonds;
    }

    // Initialize operator string: all diagonal site ops
    op_types_.assign(M_total_, 1);
    op_sites_.assign(M_total_, 0);
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

// ─── Diagonal Update ─────────────────────────────────────────────────────────

void QAQMCEngine::diagonal_update() {
    std::vector<int32_t> state(N_, 0); // boundary |0...0>
    const int* bond_sites = vij_.bond_sites_flat.data();
    int n_bonds = vij_.n_bonds;

    if (precompute_ && chunk_slices_ <= 0) {
        // ── Full precomputed alias table path (original) ──
        int n_bonds_pad = alias_.n_bonds_pad;
        int max_alias = alias_.max_alias;

        for (int p = 0; p < M_total_; ++p) {
            int ot = op_types_[p];
            if (ot == -1) {
                state[op_sites_[p]] ^= 1;
            } else if (ot == 1 || ot == 2) {
                bool inserted = false;
                while (!inserted) {
                    int n_alias_p = alias_.n_alias_all[p];
                    int i = randint(rng_, n_alias_p);
                    int idx;
                    if (uniform01(rng_) < alias_.alias_prob_all[p * max_alias + i])
                        idx = i;
                    else
                        idx = (int)alias_.alias_idx_all[p * max_alias + i];

                    int kind = alias_.op_map_kind_all[p * max_alias + idx];
                    int loc  = alias_.op_map_loc_all[p * max_alias + idx];

                    if (kind == 0) {
                        op_types_[p] = 1;
                        op_sites_[p] = loc;
                        inserted = true;
                    } else {
                        int b = loc;
                        int si = bond_sites[b * 2 + 0];
                        int sj = bond_sites[b * 2 + 1];
                        int w_idx = state[si] * 2 + state[sj];
                        double w_actual = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + w_idx];
                        double w_max = alias_.bond_W_max_all[p * n_bonds_pad + b];
                        if (w_max > 0.0 && uniform01(rng_) < w_actual / w_max) {
                            op_types_[p] = 2;
                            op_sites_[p] = b;
                            inserted = true;
                        }
                    }
                }
            }
        }
    } else if (precompute_ && chunk_slices_ > 0) {
        // ── Chunked precompute path ──
        // Build alias tables for each chunk, use them, then free.
        for (int chunk_start = 0; chunk_start < M_total_; chunk_start += chunk_slices_) {
            int chunk_end = std::min(chunk_start + chunk_slices_, M_total_);

            // Build alias table for this chunk only
            AliasTable chunk_alias = build_qaqmc_alias_tables(
                M_total_, N_, n_bonds, Omega_,
                delta_sched_.data(), vij_.vij_list.data(),
                vij_.bonds_i.data(), vij_.bonds_j.data(),
                vij_.coord_number.data(), epsilon_,
                chunk_start, chunk_end);

            int n_bonds_pad = chunk_alias.n_bonds_pad;
            int max_alias = chunk_alias.max_alias;

            for (int p = chunk_start; p < chunk_end; ++p) {
                int pi = p - chunk_start;  // local index into chunk_alias
                int ot = op_types_[p];
                if (ot == -1) {
                    state[op_sites_[p]] ^= 1;
                } else if (ot == 1 || ot == 2) {
                    bool inserted = false;
                    while (!inserted) {
                        int n_alias_p = chunk_alias.n_alias_all[pi];
                        int i = randint(rng_, n_alias_p);
                        int idx;
                        if (uniform01(rng_) < chunk_alias.alias_prob_all[pi * max_alias + i])
                            idx = i;
                        else
                            idx = (int)chunk_alias.alias_idx_all[pi * max_alias + i];

                        int kind = chunk_alias.op_map_kind_all[pi * max_alias + idx];
                        int loc  = chunk_alias.op_map_loc_all[pi * max_alias + idx];

                        if (kind == 0) {
                            op_types_[p] = 1;
                            op_sites_[p] = loc;
                            inserted = true;
                        } else {
                            int b = loc;
                            int si = bond_sites[b * 2 + 0];
                            int sj = bond_sites[b * 2 + 1];
                            int w_idx = state[si] * 2 + state[sj];
                            double w_actual = chunk_alias.bond_W_all[(pi * n_bonds_pad + b) * 4 + w_idx];
                            double w_max = chunk_alias.bond_W_max_all[pi * n_bonds_pad + b];
                            if (w_max > 0.0 && uniform01(rng_) < w_actual / w_max) {
                                op_types_[p] = 2;
                                op_sites_[p] = b;
                                inserted = true;
                            }
                        }
                    }
                }
            }
            // chunk_alias goes out of scope here → memory freed
        }
    } else {
        // ── On-the-fly path (no precomputed tables) ──
        std::vector<double> cum_weights(N_ + n_bonds);

        for (int p = 0; p < M_total_; ++p) {
            int ot = op_types_[p];
            if (ot == -1) {
                state[op_sites_[p]] ^= 1;
            } else if (ot == 1 || ot == 2) {
                double delta = delta_sched_[p];

                // Build cumulative weight array for this time slice
                double running = 0.0;
                for (int i = 0; i < N_; ++i) {
                    running += site_W_;
                    cum_weights[i] = running;
                }
                for (int b = 0; b < n_bonds; ++b) {
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
                    double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
                    double W[4], wmax;
                    compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
                    running += wmax;
                    cum_weights[N_ + b] = running;
                }
                double total_weight = running;

                bool inserted = false;
                while (!inserted) {
                    double r = uniform01(rng_) * total_weight;
                    int idx = (int)(std::lower_bound(cum_weights.begin(),
                                                     cum_weights.begin() + N_ + n_bonds, r)
                                    - cum_weights.begin());
                    if (idx < N_) {
                        op_types_[p] = 1;
                        op_sites_[p] = idx;
                        inserted = true;
                    } else {
                        int b = idx - N_;
                        int si = bond_sites[b * 2 + 0];
                        int sj = bond_sites[b * 2 + 1];
                        int w_idx = state[si] * 2 + state[sj];
                        double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
                        double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
                        double W[4], wmax;
                        compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
                        if (wmax > 0.0 && uniform01(rng_) < W[w_idx] / wmax) {
                            op_types_[p] = 2;
                            op_sites_[p] = b;
                            inserted = true;
                        }
                    }
                }
            }
        }
    }
}

// ─── Cluster Update ──────────────────────────────────────────────────────────

void QAQMCEngine::cluster_update() {
    if (M_total_ == 0) return;

    const int* bond_sites = vij_.bond_sites_flat.data();

    // Build state_at: (M_total_ x N_), row-major
    std::vector<int32_t> state_at(M_total_ * N_, 0);
    std::vector<int32_t> cur(N_, 0); // boundary |0...0>

    for (int p = 0; p < M_total_; ++p) {
        for (int s = 0; s < N_; ++s)
            state_at[p * N_ + s] = cur[s];
        if (op_types_[p] == -1)
            cur[op_sites_[p]] ^= 1;
    }

    // Per-site segment processing
    std::vector<int32_t> site_ops(M_total_ + 2);

    for (int site_i = 0; site_i < N_; ++site_i) {
        int n_sops = 0;
        site_ops[n_sops++] = -1; // left boundary sentinel

        for (int p = 0; p < M_total_; ++p) {
            int ot = op_types_[p];
            if ((ot == 1 || ot == -1) && op_sites_[p] == site_i) {
                site_ops[n_sops++] = p;
            }
        }

        site_ops[n_sops++] = M_total_; // right boundary sentinel

        for (int seg = 0; seg < n_sops - 1; ++seg) {
            int p_start = site_ops[seg];
            int p_end   = site_ops[seg + 1];

            // Freeze segments touching the boundary
            if (p_start == -1 || p_end == M_total_)
                continue;

            // Compute log weight ratio for flipping this segment
            double log_w_old = 0.0;
            double log_w_new = 0.0;

            for (int p = std::max(0, p_start + 1); p < std::min(M_total_, p_end + 1); ++p) {
                if (op_types_[p] == 2) {
                    int b = op_sites_[p];
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    if (si == site_i || sj == site_i) {
                        int ni = state_at[p * N_ + si];
                        int nj = state_at[p * N_ + sj];

                        double w_old, w_new;
                        if (precompute_ && chunk_slices_ <= 0) {
                            // Full precompute: read from stored table
                            int n_bonds_pad = alias_.n_bonds_pad;
                            w_old = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + ni * 2 + nj];
                            if (si == site_i)
                                w_new = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + (1 - ni) * 2 + nj];
                            else
                                w_new = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + ni * 2 + (1 - nj)];
                        } else {
                            double delta = delta_sched_[p];
                            double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
                            double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
                            double W[4], wmax;
                            compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
                            w_old = W[ni * 2 + nj];
                            if (si == site_i)
                                w_new = W[(1 - ni) * 2 + nj];
                            else
                                w_new = W[ni * 2 + (1 - nj)];
                        }

                        log_w_old += (w_old > 1e-300) ? std::log(w_old) : -1e30;
                        log_w_new += (w_new > 1e-300) ? std::log(w_new) : -1e30;
                    }
                }
            }

            double log_ratio = log_w_new - log_w_old;
            bool do_flip = (log_ratio >= 0.0) || (uniform01(rng_) < std::exp(log_ratio));

            if (do_flip) {
                // Flip segment in state_at
                for (int p = std::max(0, p_start + 1); p < std::min(M_total_, p_end + 1); ++p) {
                    state_at[p * N_ + site_i] ^= 1;
                }
            }
        }
    }

    // Reassign op_types based on the updated state_at
    for (int p = 0; p < M_total_; ++p) {
        int ot = op_types_[p];
        if (ot == 1 || ot == -1) {
            int site = op_sites_[p];
            int n_before = state_at[p * N_ + site];
            int n_after;
            if (p < M_total_ - 1)
                n_after = state_at[(p + 1) * N_ + site];
            else
                n_after = cur[site];
            op_types_[p] = (n_before == n_after) ? 1 : -1;
        }
    }
}

// ─── mc_step ─────────────────────────────────────────────────────────────────

void QAQMCEngine::mc_step() {
    diagonal_update();
    cluster_update();
}