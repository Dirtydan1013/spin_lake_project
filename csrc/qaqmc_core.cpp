#include "qaqmc_core.hpp"
#include <cstring>
#include <cassert>

// ─── Helper: uniform random [0, 1) ──────────────────────────────────────────
static inline double uniform01(std::mt19937_64& rng) {
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
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
                              const double* pos, int pos_dim) {
    RydbergVij res;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dist2 = 0.0;
            for (int d = 0; d < pos_dim; ++d) {
                double diff = pos[i * pos_dim + d] - pos[j * pos_dim + d];
                dist2 += diff * diff;
            }
            double dist = std::sqrt(dist2);
            double ratio = Rb / dist;
            double vij = Omega * ratio * ratio * ratio * ratio * ratio * ratio; // (Rb/r)^6
            res.bonds_i.push_back(i);
            res.bonds_j.push_back(j);
            res.vij_list.push_back(vij);
        }
    }
    res.n_bonds = (int)res.bonds_i.size();
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
                                     double epsilon) {
    AliasTable res;
    int max_alias = N + n_bonds;
    int n_bonds_pad = std::max(n_bonds, 1);
    res.max_alias = max_alias;
    res.n_bonds_pad = n_bonds_pad;

    // Allocate flat arrays
    res.bond_W_all.assign(M_total * n_bonds_pad * 4, 0.0);
    res.bond_W_max_all.assign(M_total * n_bonds_pad, 0.0);
    res.n_alias_all.resize(M_total, 0);
    res.alias_prob_all.assign(M_total * max_alias, 0.0);
    res.alias_idx_all.assign(M_total * max_alias, 0);
    res.op_map_kind_all.assign(M_total * max_alias, 0);
    res.op_map_loc_all.assign(M_total * max_alias, 0);

    // Each time slice p is completely independent → OpenMP parallel for
#ifdef QAQMC_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < M_total; ++p) {
        // Thread-local temporary buffers (each thread gets its own copy)
        std::vector<double> weights(max_alias);
        std::vector<int> op_kind(max_alias);
        std::vector<int> op_loc(max_alias);
        std::vector<double> prob_arr(max_alias);
        std::vector<int64_t> alias_arr(max_alias);
        std::vector<int> small_buf, large_buf;

        double delta = delta_sched[p];
        double delta_b = (N > 1) ? delta / (N - 1) : delta;

        int n_a = 0;

        // Site operators
        for (int i = 0; i < N; ++i) {
            weights[n_a] = Omega / 2.0;
            op_kind[n_a] = 0;
            op_loc[n_a] = i;
            n_a++;
        }

        // Bond operators
        for (int b = 0; b < n_bonds; ++b) {
            double vij = bond_vij[b];
            double two_db_vij = 2.0 * delta_b - vij;
            double m1 = std::min({0.0, delta_b, two_db_vij});
            double m2 = std::min(delta_b, two_db_vij);
            double cij = std::abs(m1) + epsilon * std::abs(m2);

            int base = (p * n_bonds_pad + b) * 4;
            res.bond_W_all[base + 0] = cij;
            res.bond_W_all[base + 1] = delta_b + cij;
            res.bond_W_all[base + 2] = delta_b + cij;
            res.bond_W_all[base + 3] = -vij + 2.0 * delta_b + cij;

            double bmax = std::max({res.bond_W_all[base + 0],
                                    res.bond_W_all[base + 1],
                                    res.bond_W_all[base + 2],
                                    res.bond_W_all[base + 3]});
            res.bond_W_max_all[p * n_bonds_pad + b] = bmax;

            weights[n_a] = bmax;
            op_kind[n_a] = 1;
            op_loc[n_a] = b;
            n_a++;
        }

        res.n_alias_all[p] = n_a;

        // Copy op_map
        for (int i = 0; i < n_a; ++i) {
            res.op_map_kind_all[p * max_alias + i] = op_kind[i];
            res.op_map_loc_all[p * max_alias + i] = op_loc[i];
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
            res.alias_prob_all[p * max_alias + i] = prob_arr[i];
            res.alias_idx_all[p * max_alias + i] = alias_arr[i];
        }
    }

    return res;
}

// ═════════════════════════════════════════════════════════════════════════════
// QAQMCEngine
// ═════════════════════════════════════════════════════════════════════════════

QAQMCEngine::QAQMCEngine(int N, double Omega, double delta_min, double delta_max,
                          double Rb, int M, double epsilon, uint64_t seed,
                          const double* pos, int pos_dim)
    : N_(N), M_(M), M_total_(2 * M),
      Omega_(Omega), Rb_(Rb), delta_min_(delta_min), delta_max_(delta_max),
      rng_(seed)
{
    site_W_ = Omega / 2.0;
    site_W_max_ = Omega / 2.0;

    // Build V_ij
    vij_ = build_rydberg_vij(N, Omega, Rb, pos, pos_dim);

    // Build delta schedule: delta_min -> delta_max -> delta_min
    delta_sched_.resize(M_total_);
    for (int p = 0; p < M_; ++p) {
        delta_sched_[p] = delta_min + (delta_max - delta_min) * ((double)p / M_);
    }
    for (int p = M_; p < M_total_; ++p) {
        delta_sched_[p] = delta_max - (delta_max - delta_min) * ((double)(p - M_) / M_);
    }

    // Build alias tables
    alias_ = build_qaqmc_alias_tables(M_total_, N_, vij_.n_bonds, Omega_,
                                       delta_sched_.data(), vij_.vij_list.data(),
                                       epsilon);

    // Initialize operator string: all diagonal site ops
    op_types_.assign(M_total_, 1);
    op_sites_.assign(M_total_, 0);
}

// ─── Diagonal Update ─────────────────────────────────────────────────────────

void QAQMCEngine::diagonal_update() {
    std::vector<int32_t> state(N_, 0); // boundary |0...0>

    const int* bond_sites = vij_.bond_sites_flat.data();
    int n_bonds_pad = alias_.n_bonds_pad;
    int max_alias = alias_.max_alias;

    for (int p = 0; p < M_total_; ++p) {
        int ot = op_types_[p];

        if (ot == -1) {
            // Off-diagonal: propagate
            state[op_sites_[p]] ^= 1;
        } else if (ot == 1 || ot == 2) {
            // Diagonal: remove and re-sample
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
                    // Site op
                    op_types_[p] = 1;
                    op_sites_[p] = loc;
                    inserted = true;
                } else {
                    // Bond op
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
}

// ─── Cluster Update ──────────────────────────────────────────────────────────

void QAQMCEngine::cluster_update() {
    if (M_total_ == 0) return;

    const int* bond_sites = vij_.bond_sites_flat.data();
    int n_bonds_pad = alias_.n_bonds_pad;

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
                        double w_old = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + ni * 2 + nj];
                        double w_new;
                        if (si == site_i)
                            w_new = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + (1 - ni) * 2 + nj];
                        else
                            w_new = alias_.bond_W_all[(p * n_bonds_pad + b) * 4 + ni * 2 + (1 - nj)];
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