#include "qaqmc_renyi_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

inline double renyi_u01(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

inline int renyi_randi(std::mt19937_64& rng, int n) {
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(rng);
}

}  // namespace

QAQMCRenyiEngine::QAQMCRenyiEngine(int N, double Omega, double delta_min, double delta_max,
                                   double Rb, int M, double epsilon, uint64_t seed,
                                   const double* pos, int pos_dim, int neighbor_cutoff,
                                   int delta_groups)
    : N_(N),
      M_(M),
      M_total_(2 * M),
      Omega_(Omega),
      Rb_(Rb),
      delta_min_(delta_min),
      delta_max_(delta_max),
      epsilon_(epsilon),
      delta_groups_(delta_groups),
      rngs_{std::mt19937_64(seed), std::mt19937_64(seed + 0x9e3779b97f4a7c15ULL)} {
    if (delta_groups_ < 0) delta_groups_ = 0;
    if (M_total_ > 0 && delta_groups_ > M_total_) delta_groups_ = M_total_;

    vij_ = build_rydberg_vij(N, Omega, Rb, pos, pos_dim, neighbor_cutoff);

    delta_sched_.resize(M_total_);
    for (int p = 0; p < M_; ++p) {
        delta_sched_[p] = delta_min_ + (delta_max_ - delta_min_) * (static_cast<double>(p) / M_);
    }
    for (int p = M_; p < M_total_; ++p) {
        delta_sched_[p] = delta_max_ - (delta_max_ - delta_min_) * (static_cast<double>(p - M_) / M_);
    }

    if (delta_groups_ > 0) {
        alias_.n_bonds_pad = std::max(vij_.n_bonds, 1);
        alias_.max_alias = N_ + vij_.n_bonds;
        n_bonds_pad_ = alias_.n_bonds_pad;
        max_alias_ = alias_.max_alias;
        build_grouped_alias_tables();
    } else {
        alias_ = build_qaqmc_alias_tables(
            M_total_, N_, vij_.n_bonds, Omega_, delta_sched_.data(), vij_.vij_list.data(),
            vij_.bonds_i.data(), vij_.bonds_j.data(), vij_.coord_number.data(), epsilon_);
        n_bonds_pad_ = alias_.n_bonds_pad;
        max_alias_ = alias_.max_alias;
    }

    for (auto& replica : replicas_) {
        replica.op_types.assign(M_total_, 1);
        replica.op_sites.assign(M_total_, 0);
        replica.state_at_M.assign(N_, 0);
    }
    A_mask_.assign(N_, 0);
    A_masks_[0] = A_mask_;
    A_masks_[1] = A_mask_;
    // Per-site RNG streams seeded deterministically from the master seed; used
    // by the parallel cluster_update so multiple threads don't race on a shared
    // generator.  SplitMix-style mixing keeps streams nominally independent.
    site_rngs_.resize(N_);
    for (int s = 0; s < N_; ++s) {
        uint64_t z = seed + 0xD2B74407B1CE6E93ULL * static_cast<uint64_t>(s + 1);
        z ^= z >> 33;
        z *= 0xFF51AFD7ED558CCDULL;
        z ^= z >> 33;
        z *= 0xC4CEB9FE1A85EC53ULL;
        z ^= z >> 33;
        site_rngs_[s].seed(z);
    }

    // Compute bond-graph coloring once so cluster_update can dispatch each
    // colour class in parallel without races on per-bond spin-cache writes.
    compute_site_coloring();

    const int ch_sites = 2 * N_;
    ch_site_op_count_.assign(ch_sites, 0);
    ch_site_op_head_.assign(ch_sites, 0);
    ch_site_bond_count_.assign(ch_sites, 0);
    ch_site_bond_head_.assign(ch_sites, 0);
    bond_spin_by_replica_.assign(static_cast<size_t>(2) * M_total_, 0);
}

void QAQMCRenyiEngine::compute_site_coloring() {
    // Build undirected adjacency from the active bond list.
    std::vector<std::vector<int>> adj(N_);
    const int* bond_sites = vij_.bond_sites_flat.data();
    for (int b = 0; b < vij_.n_bonds; ++b) {
        int si = bond_sites[b * 2 + 0];
        int sj = bond_sites[b * 2 + 1];
        if (si == sj) continue;
        adj[si].push_back(sj);
        adj[sj].push_back(si);
    }
    // Greedy first-fit colouring.  Order: descending degree gives near-optimal
    // colour count on irregular graphs.
    std::vector<int> order(N_);
    for (int s = 0; s < N_; ++s) order[s] = s;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return adj[a].size() > adj[b].size();
    });

    std::vector<int> color_of(N_, -1);
    int max_color = -1;
    std::vector<int> used;
    for (int s : order) {
        used.assign(adj[s].size() + 1, 0);
        for (int n : adj[s]) {
            int c = color_of[n];
            if (c >= 0 && c < static_cast<int>(used.size())) used[c] = 1;
        }
        int c = 0;
        while (c < static_cast<int>(used.size()) && used[c]) ++c;
        color_of[s] = c;
        if (c > max_color) max_color = c;
    }
    color_groups_.assign(max_color + 1, {});
    for (int s = 0; s < N_; ++s) {
        color_groups_[color_of[s]].push_back(s);
    }
}

void QAQMCRenyiEngine::build_grouped_alias_tables() {
    const int G = delta_groups_;
    const int n_bonds = vij_.n_bonds;
    const int n_bonds_pad = std::max(n_bonds, 1);
    const int max_alias = N_ + n_bonds;
    const int* bond_si = vij_.bonds_i.data();
    const int* bond_sj = vij_.bonds_j.data();
    const int* coord_num = vij_.coord_number.data();

    grp_alias_.n_groups = G;
    grp_alias_.max_alias = max_alias;
    grp_alias_.n_bonds_pad = n_bonds_pad;
    grp_alias_.slice_to_group.resize(M_total_);

    for (int p = 0; p < M_total_; ++p) {
        int g = static_cast<int>((static_cast<int64_t>(p) * G) / M_total_);
        if (g >= G) g = G - 1;
        grp_alias_.slice_to_group[p] = g;
    }

    grp_alias_.bond_W_max_all.assign(G * n_bonds_pad, 0.0);
    grp_alias_.n_alias_all.assign(G, 0);
    grp_alias_.alias_prob_all.assign(G * max_alias, 0.0);
    grp_alias_.alias_idx_all.assign(G * max_alias, 0);
    grp_alias_.op_map_kind_all.assign(G * max_alias, 0);
    grp_alias_.op_map_loc_all.assign(G * max_alias, 0);

#ifdef QAQMC_USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int g = 0; g < G; ++g) {
        int p_lo = static_cast<int>((static_cast<int64_t>(g) * M_total_) / G);
        int p_hi = static_cast<int>((static_cast<int64_t>(g + 1) * M_total_) / G);
        if (p_hi > M_total_) p_hi = M_total_;

        std::vector<double> env_W_max(n_bonds, 0.0);
        for (int p = p_lo; p < p_hi; ++p) {
            const double delta = delta_sched_[p];
            for (int b = 0; b < n_bonds; ++b) {
                const int si = bond_si[b];
                const int sj = bond_sj[b];
                const double di = (coord_num[si] > 0) ? delta / coord_num[si] : 0.0;
                const double dj = (coord_num[sj] > 0) ? delta / coord_num[sj] : 0.0;
                double W[4], wmax;
                compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
                if (wmax > env_W_max[b]) env_W_max[b] = wmax;
            }
        }

        for (int b = 0; b < n_bonds; ++b) {
            grp_alias_.bond_W_max_all[g * n_bonds_pad + b] = env_W_max[b];
        }

        std::vector<double> weights(max_alias);
        std::vector<int> op_kind(max_alias);
        std::vector<int> op_loc(max_alias);
        int n_a = 0;

        for (int i = 0; i < N_; ++i) {
            weights[n_a] = Omega_ / 2.0;
            op_kind[n_a] = 0;
            op_loc[n_a] = i;
            ++n_a;
        }
        for (int b = 0; b < n_bonds; ++b) {
            weights[n_a] = env_W_max[b];
            op_kind[n_a] = 1;
            op_loc[n_a] = b;
            ++n_a;
        }

        grp_alias_.n_alias_all[g] = n_a;
        for (int i = 0; i < n_a; ++i) {
            grp_alias_.op_map_kind_all[g * max_alias + i] = op_kind[i];
            grp_alias_.op_map_loc_all[g * max_alias + i] = op_loc[i];
        }

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

int QAQMCRenyiEngine::replica_for_with_mask(int channel, int site, int p,
                                            const std::vector<uint8_t>& mask) const {
    if (p < M_) return channel;
    return mask[site] ? (1 - channel) : channel;
}

int QAQMCRenyiEngine::channel_for_actual_with_mask(int replica, int site, int p,
                                                   const std::vector<uint8_t>& mask) const {
    if (p < M_) return replica;
    return mask[site] ? (1 - replica) : replica;
}

int QAQMCRenyiEngine::replica_for(int channel, int site, int p) const {
    return replica_for_with_mask(channel, site, p, A_mask_);
}

int QAQMCRenyiEngine::channel_for_actual(int replica, int site, int p) const {
    return channel_for_actual_with_mask(replica, site, p, A_mask_);
}

void QAQMCRenyiEngine::set_A_mask(const uint8_t* mask, int len) {
    if (len != N_) {
        throw std::runtime_error("A_mask length mismatch");
    }
    A_mask_.assign(mask, mask + len);
    A_masks_[0] = A_mask_;
    A_masks_[1] = A_mask_;
    cur_topology_ = 0;
    diff_site_ = -1;
    mode_ = Mode::PairToggle;
    reset_visit_counts();
    recompute_midpoint_states();
}

void QAQMCRenyiEngine::set_topology_pair(const uint8_t* A_k, const uint8_t* A_kp1, int len, int diff_site) {
    if (len != N_) {
        throw std::runtime_error("topology-pair mask length mismatch");
    }
    if (diff_site < 0 || diff_site >= N_) {
        throw std::runtime_error("diff_site out of range");
    }

    A_masks_[0].assign(A_k, A_k + len);
    A_masks_[1].assign(A_kp1, A_kp1 + len);

    int n_diff = 0;
    for (int site = 0; site < N_; ++site) {
        if (A_masks_[0][site] != A_masks_[1][site]) {
            ++n_diff;
            if (site != diff_site) {
                throw std::runtime_error("topology pair must differ only at diff_site");
            }
        }
    }
    if (n_diff != 1) {
        throw std::runtime_error("topology pair must differ at exactly one site");
    }

    diff_site_ = diff_site;
    cur_topology_ = 0;
    A_mask_ = A_masks_[0];
    mode_ = Mode::PairToggle;
    reset_visit_counts();
    recompute_midpoint_states();
}

void QAQMCRenyiEngine::reset_visit_counts() {
    visit_count_[0] = 0;
    visit_count_[1] = 0;
}

void QAQMCRenyiEngine::set_ensemble_ladder(const std::vector<std::vector<uint8_t>>& masks,
                                           const std::vector<std::vector<int>>& neighbors,
                                           int initial_ensemble) {
    if (masks.empty()) {
        throw std::runtime_error("ensemble ladder must contain at least one mask");
    }
    if (masks.size() != neighbors.size()) {
        throw std::runtime_error("masks and neighbors must have the same length");
    }
    const int n_ens = static_cast<int>(masks.size());
    if (initial_ensemble < 0 || initial_ensemble >= n_ens) {
        throw std::runtime_error("initial_ensemble out of range");
    }
    for (const auto& mask : masks) {
        if (static_cast<int>(mask.size()) != N_) {
            throw std::runtime_error("each ensemble mask must have length N");
        }
    }
    for (const auto& row : neighbors) {
        for (int nbr : row) {
            if (nbr < 0 || nbr >= n_ens) {
                throw std::runtime_error("ensemble neighbor index out of range");
            }
        }
    }

    ensembles_.clear();
    ensembles_.reserve(n_ens);
    for (int l = 0; l < n_ens; ++l) {
        Ensemble e;
        e.A_mask = masks[l];
        e.size = 0;
        for (uint8_t v : e.A_mask) {
            if (v) ++e.size;
        }
        ensembles_.push_back(std::move(e));
    }
    ens_neighbors_ = neighbors;

    cur_ens_ = initial_ensemble;
    A_mask_ = ensembles_[initial_ensemble].A_mask;
    A_masks_[0] = A_mask_;
    A_masks_[1] = A_mask_;
    diff_site_ = -1;
    cur_topology_ = 0;
    mode_ = Mode::Expanded;

    visit_count_ext_.assign(n_ens, 0);
    transition_count_.assign(static_cast<size_t>(n_ens) * n_ens, 0);
    collection_count_.assign(static_cast<size_t>(n_ens) * n_ens, 0.0);
    log_g_.assign(n_ens, 0.0);

    OffdiagPaths paths;
    build_offdiag_paths(A_mask_, paths);
    reproject_site_ops_for_mask_with_paths(A_mask_, paths);
    recompute_midpoint_states_from_ops();
}

void QAQMCRenyiEngine::set_log_g(const std::vector<double>& log_g) {
    if (ensembles_.empty()) {
        throw std::runtime_error("set_ensemble_ladder must be called before set_log_g");
    }
    if (log_g.size() != ensembles_.size()) {
        throw std::runtime_error("log_g length mismatch");
    }
    log_g_ = log_g;
}

void QAQMCRenyiEngine::reset_visit_counts_ext() {
    std::fill(visit_count_ext_.begin(), visit_count_ext_.end(), 0);
}

void QAQMCRenyiEngine::reset_transition_counts() {
    std::fill(transition_count_.begin(), transition_count_.end(), 0);
}

void QAQMCRenyiEngine::reset_collection_counts() {
    std::fill(collection_count_.begin(), collection_count_.end(), 0.0);
}

int QAQMCRenyiEngine::diff_site_between_masks(const std::vector<uint8_t>& from_mask,
                                              const std::vector<uint8_t>& to_mask) const {
    int diff = -1;
    int count = 0;
    for (int s = 0; s < N_; ++s) {
        if (from_mask[s] != to_mask[s]) {
            diff = s;
            ++count;
        }
    }
    return (count == 1) ? diff : -1;
}

double QAQMCRenyiEngine::log_weight_ratio_between_masks(int site,
                                                        const std::vector<uint8_t>& from_mask,
                                                        const std::vector<uint8_t>& to_mask) const {
    OffdiagPaths paths_from;
    OffdiagPaths paths_to;
    build_offdiag_paths(from_mask, paths_from);
    build_offdiag_paths(to_mask, paths_to);
    double log_from = log_weight_for_site_with_paths(site, from_mask, paths_from);
    double log_to = log_weight_for_site_with_paths(site, to_mask, paths_to);
    return log_to - log_from;
}

void QAQMCRenyiEngine::ensemble_switch() {
    if (ensembles_.empty()) {
        return;
    }
    const int n_ens = static_cast<int>(ensembles_.size());
    const int from_ens = cur_ens_;
    const auto& nbrs = ens_neighbors_[from_ens];

    if (nbrs.empty()) {
        collection_count_[static_cast<size_t>(from_ens) * n_ens + from_ens] += 1.0;
        transition_count_[static_cast<size_t>(from_ens) * n_ens + from_ens]++;
        return;
    }

    const auto& mask_from = ensembles_[from_ens].A_mask;
    build_offdiag_paths(mask_from, paths_scratch_from_);

    const double propose_prob = 1.0 / static_cast<double>(nbrs.size());
    std::vector<double> accept_probs(nbrs.size(), 0.0);
    std::vector<int> diff_sites(nbrs.size(), -1);
    double off_diag_weight = 0.0;
    paths_scratch_targets_.resize(nbrs.size());

    for (size_t i = 0; i < nbrs.size(); ++i) {
        const int to_ens = nbrs[i];
        const auto& mask_to = ensembles_[to_ens].A_mask;
        const int diff = diff_site_between_masks(mask_from, mask_to);
        diff_sites[i] = diff;
        build_offdiag_paths(mask_to, paths_scratch_targets_[i]);

        double log_ratio = 0.0;
        bool feasible = true;

        if (diff >= 0) {
            const double log_from = log_weight_for_site_with_paths(diff, mask_from, paths_scratch_from_);
            const double log_to = log_weight_for_site_with_paths(diff, mask_to, paths_scratch_targets_[i]);
            if (log_to <= -1e29) {
                feasible = false;
            } else if (log_from <= -1e29) {
                log_ratio = 1e30;  // from unreachable ⇒ force accept
            } else {
                log_ratio = log_to - log_from;
            }
        } else {
            for (int s = 0; s < N_; ++s) {
                if (mask_from[s] == mask_to[s]) continue;
                const double log_from = log_weight_for_site_with_paths(s, mask_from, paths_scratch_from_);
                const double log_to = log_weight_for_site_with_paths(s, mask_to, paths_scratch_targets_[i]);
                if (log_to <= -1e29) {
                    feasible = false;
                    break;
                }
                if (log_from <= -1e29) {
                    log_ratio = 1e30;
                    break;
                }
                log_ratio += (log_to - log_from);
            }
        }

        double a = 0.0;
        if (feasible) {
            const double n_from = static_cast<double>(nbrs.size());
            const double n_to = static_cast<double>(ens_neighbors_[to_ens].size());
            const double log_prop = (n_to > 0.0) ? std::log(n_from / n_to) : 0.0;
            const double log_a = log_ratio + log_g_[to_ens] - log_g_[from_ens] + log_prop;
            if (log_a >= 0.0) {
                a = 1.0;
            } else if (log_a <= -700.0) {
                a = 0.0;
            } else {
                a = std::exp(log_a);
            }
        }
        accept_probs[i] = a;
        collection_count_[static_cast<size_t>(from_ens) * n_ens + to_ens] += propose_prob * a;
        off_diag_weight += propose_prob * a;
    }
    const double self_weight = 1.0 - off_diag_weight;
    collection_count_[static_cast<size_t>(from_ens) * n_ens + from_ens] +=
        (self_weight > 0.0 ? self_weight : 0.0);

    const int idx = renyi_randi(rngs_[0], static_cast<int>(nbrs.size()));
    const int proposed = nbrs[idx];
    const double a = accept_probs[idx];
    const bool accept = (a >= 1.0) || (renyi_u01(rngs_[0]) < a);

    int to_ens = from_ens;
    if (accept) {
        to_ens = proposed;
        cur_ens_ = proposed;
        A_mask_ = ensembles_[proposed].A_mask;
        const int diff = diff_sites[idx];
        if (diff >= 0) {
            reproject_site_ops_at_site_with_paths(diff, A_mask_, paths_scratch_targets_[idx]);
        } else {
            reproject_site_ops_for_mask_with_paths(A_mask_, paths_scratch_targets_[idx]);
        }
    }
    transition_count_[static_cast<size_t>(from_ens) * n_ens + to_ens]++;
}

void QAQMCRenyiEngine::set_indicator_site(int site) {
    if (site < 0 || site >= N_) {
        throw std::runtime_error("indicator_site out of range");
    }
    indicator_site_ = site;
}

void QAQMCRenyiEngine::reset_indicator() {
    indicator_sum_ = 0;
    indicator_count_ = 0;
}

double QAQMCRenyiEngine::get_indicator_avg() const {
    if (indicator_count_ == 0) return 0.0;
    return static_cast<double>(indicator_sum_) / static_cast<double>(indicator_count_);
}

int QAQMCRenyiEngine::current_indicator() const {
    if (indicator_site_ < 0) return 0;
    return replicas_[0].state_at_M[indicator_site_] == replicas_[1].state_at_M[indicator_site_] ? 1 : 0;
}

void QAQMCRenyiEngine::set_replica_op_string(int replica, const int32_t* types, const int32_t* sites, int len) {
    if (replica < 0 || replica > 1) {
        throw std::runtime_error("replica index out of range");
    }
    if (len != M_total_) {
        throw std::runtime_error("operator string length mismatch");
    }
    std::memcpy(replicas_[replica].op_types.data(), types, len * sizeof(int32_t));
    std::memcpy(replicas_[replica].op_sites.data(), sites, len * sizeof(int32_t));
}

void QAQMCRenyiEngine::recompute_midpoint_states() {
    recompute_midpoint_states_from_ops();
}

void QAQMCRenyiEngine::recompute_midpoint_states_from_ops() {
    for (auto& replica : replicas_) {
        std::fill(replica.state_at_M.begin(), replica.state_at_M.end(), 0);
    }
    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_; ++p) {
            if (replicas_[replica].op_types[p] == -1) {
                int site = replicas_[replica].op_sites[p];
                if (site >= 0 && site < N_) {
                    replicas_[replica].state_at_M[site] ^= 1;
                }
            }
        }
    }
}

void QAQMCRenyiEngine::build_offdiag_paths(const std::vector<uint8_t>& mask,
                                           OffdiagPaths& paths) const {
    const int ch_sites = 2 * N_;
    if (static_cast<int>(paths.count.size()) != ch_sites) {
        paths.count.assign(ch_sites, 0);
        paths.head.assign(ch_sites, 0);
    } else {
        std::fill(paths.count.begin(), paths.count.end(), 0);
        std::fill(paths.head.begin(), paths.head.end(), 0);
    }

    auto cs_idx = [&](int channel, int site) { return channel * N_ + site; };
    for (int p = 0; p < M_total_; ++p) {
        for (int replica = 0; replica < 2; ++replica) {
            if (replicas_[replica].op_types[p] != -1) continue;
            int site = replicas_[replica].op_sites[p];
            int channel = channel_for_actual_with_mask(replica, site, p, mask);
            paths.count[cs_idx(channel, site)]++;
        }
    }

    int total = 0;
    for (int idx = 0; idx < ch_sites; ++idx) {
        paths.head[idx] = total;
        total += paths.count[idx];
    }
    paths.list.assign(total, 0);

    std::vector<int32_t> cursor(ch_sites, 0);
    for (int p = 0; p < M_total_; ++p) {
        for (int replica = 0; replica < 2; ++replica) {
            if (replicas_[replica].op_types[p] != -1) continue;
            int site = replicas_[replica].op_sites[p];
            int channel = channel_for_actual_with_mask(replica, site, p, mask);
            int idx = cs_idx(channel, site);
            paths.list[paths.head[idx] + cursor[idx]++] = p;
        }
    }
}

int QAQMCRenyiEngine::occupancy_from_paths(const OffdiagPaths& paths,
                                           int channel, int site, int p) const {
    int idx = channel * N_ + site;
    int begin = paths.head[idx];
    int end = begin + paths.count[idx];
    const auto* first = paths.list.data() + begin;
    const auto* last = paths.list.data() + end;
    return static_cast<int>(std::lower_bound(first, last, p) - first) & 1;
}

double QAQMCRenyiEngine::log_weight_for_site_with_paths(
    int site,
    const std::vector<uint8_t>& mask,
    const OffdiagPaths& paths) const {
    if (site < 0 || site >= N_) {
        throw std::runtime_error("site out of range");
    }
    for (int channel = 0; channel < 2; ++channel) {
        if (occupancy_from_paths(paths, channel, site, M_total_) != 0) {
            return -1e30;
        }
    }

    const int* bond_sites = vij_.bond_sites_flat.data();
    double log_weight = 0.0;
    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_total_; ++p) {
            if (replicas_[replica].op_types[p] != 2) continue;
            int b = replicas_[replica].op_sites[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            if (si != site && sj != site) continue;

            int c_i = channel_for_actual_with_mask(replica, si, p, mask);
            int c_j = channel_for_actual_with_mask(replica, sj, p, mask);
            int n_i = occupancy_from_paths(paths, c_i, si, p);
            int n_j = occupancy_from_paths(paths, c_j, sj, p);
            double w = actual_bond_weight(p, b, n_i * 2 + n_j);
            if (w <= 1e-300) {
                return -1e30;
            }
            log_weight += std::log(w);
        }
    }
    return log_weight;
}

void QAQMCRenyiEngine::reproject_site_ops_for_mask_with_paths(
    const std::vector<uint8_t>& mask,
    const OffdiagPaths& paths) {
    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_total_; ++p) {
            int& ot = replicas_[replica].op_types[p];
            if (ot != 1 && ot != -1) continue;
            int site = replicas_[replica].op_sites[p];
            int channel = channel_for_actual_with_mask(replica, site, p, mask);
            int before = occupancy_from_paths(paths, channel, site, p);
            int after = occupancy_from_paths(paths, channel, site, p + 1);
            ot = (before == after) ? 1 : -1;
        }
    }
}

void QAQMCRenyiEngine::reproject_site_ops_at_site_with_paths(
    int diff_site,
    const std::vector<uint8_t>& mask,
    const OffdiagPaths& paths) {
    if (diff_site < 0 || diff_site >= N_) return;
    for (int replica = 0; replica < 2; ++replica) {
        for (int p = M_; p < M_total_; ++p) {
            int& ot = replicas_[replica].op_types[p];
            if (ot != 1 && ot != -1) continue;
            if (replicas_[replica].op_sites[p] != diff_site) continue;
            int channel = channel_for_actual_with_mask(replica, diff_site, p, mask);
            int before = occupancy_from_paths(paths, channel, diff_site, p);
            int after = occupancy_from_paths(paths, channel, diff_site, p + 1);
            ot = (before == after) ? 1 : -1;
        }
    }
}

double QAQMCRenyiEngine::actual_bond_weight(int p, int b, int w_idx) const {
    if (delta_groups_ <= 0 && !alias_.bond_W_all.empty()) {
        return alias_.bond_W_all[(p * n_bonds_pad_ + b) * 4 + w_idx];
    }

    const int* bond_sites = vij_.bond_sites_flat.data();
    const int si = bond_sites[b * 2 + 0];
    const int sj = bond_sites[b * 2 + 1];
    const double delta = delta_sched_[p];
    const double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
    const double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;
    double W[4], wmax;
    compute_bond_W_inline(di, dj, vij_.vij_list[b], epsilon_, W, wmax);
    return W[w_idx];
}

double QAQMCRenyiEngine::log_weight_ratio_for_site(int site, int from_topology, int to_topology) const {
    if (from_topology < 0 || from_topology > 1 || to_topology < 0 || to_topology > 1) {
        throw std::runtime_error("topology index out of range");
    }
    OffdiagPaths paths_from;
    OffdiagPaths paths_to;
    build_offdiag_paths(A_masks_[from_topology], paths_from);
    build_offdiag_paths(A_masks_[to_topology], paths_to);
    double log_from = log_weight_for_site_with_paths(site, A_masks_[from_topology], paths_from);
    double log_to = log_weight_for_site_with_paths(site, A_masks_[to_topology], paths_to);
    return log_to - log_from;
}

std::array<std::vector<int32_t>, 4> QAQMCRenyiEngine::get_site_paths(int site) const {
    if (site < 0 || site >= N_) {
        throw std::runtime_error("site out of range");
    }
    std::array<std::vector<int32_t>, 4> paths;
    for (auto& v : paths) v.resize(M_total_ + 1, 0);

    OffdiagPaths channel_paths;
    build_offdiag_paths(A_mask_, channel_paths);
    for (int p = 0; p <= M_total_; ++p) {
        paths[0][p] = occupancy_from_paths(channel_paths, 0, site, p);
        paths[1][p] = occupancy_from_paths(channel_paths, 1, site, p);
    }

    int value0 = 0;
    int value1 = 0;
    paths[2][0] = 0;
    paths[3][0] = 0;
    for (int p = 0; p < M_total_; ++p) {
        if (replicas_[0].op_types[p] == -1 && replicas_[0].op_sites[p] == site) value0 ^= 1;
        if (replicas_[1].op_types[p] == -1 && replicas_[1].op_sites[p] == site) value1 ^= 1;
        paths[2][p + 1] = value0;
        paths[3][p + 1] = value1;
    }
    return paths;
}

void QAQMCRenyiEngine::accumulate_indicator() {
    if (indicator_site_ < 0) return;
    indicator_sum_ += current_indicator();
    indicator_count_++;
}

void QAQMCRenyiEngine::topology_toggle() {
    if (diff_site_ < 0 || A_masks_[0] == A_masks_[1]) {
        return;
    }

    const int proposed = 1 - cur_topology_;
    const auto& mask_to = A_masks_[proposed];

    build_offdiag_paths(A_mask_, paths_scratch_from_);
    build_offdiag_paths(mask_to, paths_scratch_to_);

    const double log_from = log_weight_for_site_with_paths(diff_site_, A_mask_, paths_scratch_from_);
    const double log_to = log_weight_for_site_with_paths(diff_site_, mask_to, paths_scratch_to_);

    const double log_ratio = log_to - log_from;
    double accept_prob = 0.0;
    if (log_ratio >= 0.0) {
        const double inv_ratio = std::exp(-log_ratio);
        accept_prob = 1.0 / (1.0 + inv_ratio);
    } else {
        const double ratio = std::exp(log_ratio);
        accept_prob = ratio / (1.0 + ratio);
    }

    const bool accept = renyi_u01(rngs_[0]) < accept_prob;
    if (!accept) {
        return;
    }

    cur_topology_ = proposed;
    A_mask_ = mask_to;
    reproject_site_ops_at_site_with_paths(diff_site_, A_mask_, paths_scratch_to_);
}

void QAQMCRenyiEngine::diagonal_update() {
    const int* bond_sites = vij_.bond_sites_flat.data();
    std::vector<int32_t> channel_state(2 * N_, 0);
    auto ch_idx = [&](int channel, int site) {
        return channel * N_ + site;
    };

    for (int p = 0; p < M_total_; ++p) {
        std::array<int32_t, 2> next_types = {
            replicas_[0].op_types[p],
            replicas_[1].op_types[p],
        };
        std::array<int32_t, 2> next_sites = {
            replicas_[0].op_sites[p],
            replicas_[1].op_sites[p],
        };

        for (int replica = 0; replica < 2; ++replica) {
            std::mt19937_64& rng = rngs_[replica];
            int ot = next_types[replica];
            if (ot == -1) {
                continue;
            }
            if (ot != 1 && ot != 2) continue;

            bool inserted = false;
            while (!inserted) {
                int kind = 0;
                int loc = 0;
                int group = -1;
                if (delta_groups_ > 0) {
                    group = grp_alias_.slice_to_group[p];
                    int n_alias_g = grp_alias_.n_alias_all[group];
                    int i = renyi_randi(rng, n_alias_g);
                    int idx = (renyi_u01(rng) < grp_alias_.alias_prob_all[group * max_alias_ + i])
                        ? i
                        : static_cast<int>(grp_alias_.alias_idx_all[group * max_alias_ + i]);
                    kind = grp_alias_.op_map_kind_all[group * max_alias_ + idx];
                    loc = grp_alias_.op_map_loc_all[group * max_alias_ + idx];
                } else {
                    int n_alias_p = alias_.n_alias_all[p];
                    int i = renyi_randi(rng, n_alias_p);
                    int idx = (renyi_u01(rng) < alias_.alias_prob_all[p * max_alias_ + i])
                        ? i
                        : static_cast<int>(alias_.alias_idx_all[p * max_alias_ + i]);
                    kind = alias_.op_map_kind_all[p * max_alias_ + idx];
                    loc = alias_.op_map_loc_all[p * max_alias_ + idx];
                }

                if (kind == 0) {
                    next_types[replica] = 1;
                    next_sites[replica] = loc;
                    inserted = true;
                } else {
                    int b = loc;
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    int c_i = channel_for_actual(replica, si, p);
                    int c_j = channel_for_actual(replica, sj, p);
                    int w_idx = channel_state[ch_idx(c_i, si)] * 2 + channel_state[ch_idx(c_j, sj)];
                    double w_actual = actual_bond_weight(p, b, w_idx);
                    double w_max = (delta_groups_ > 0)
                        ? grp_alias_.bond_W_max_all[group * n_bonds_pad_ + b]
                        : alias_.bond_W_max_all[p * n_bonds_pad_ + b];
                    if (w_max > 0.0 && renyi_u01(rng) < w_actual / w_max) {
                        next_types[replica] = 2;
                        next_sites[replica] = b;
                        inserted = true;
                    }
                }
            }
        }

        for (int replica = 0; replica < 2; ++replica) {
            replicas_[replica].op_types[p] = next_types[replica];
            replicas_[replica].op_sites[p] = next_sites[replica];
        }

        for (int replica = 0; replica < 2; ++replica) {
            if (next_types[replica] != -1) continue;
            int site = next_sites[replica];
            int channel = channel_for_actual(replica, site, p);
            channel_state[ch_idx(channel, site)] ^= 1;
        }
    }
}

void QAQMCRenyiEngine::build_channel_vertex_lists() {
    const int ch_sites = 2 * N_;
    if (static_cast<int>(ch_site_op_count_.size()) != ch_sites) {
        ch_site_op_count_.assign(ch_sites, 0);
        ch_site_op_head_.assign(ch_sites, 0);
        ch_site_bond_count_.assign(ch_sites, 0);
        ch_site_bond_head_.assign(ch_sites, 0);
    } else {
        std::fill(ch_site_op_count_.begin(), ch_site_op_count_.end(), 0);
        std::fill(ch_site_bond_count_.begin(), ch_site_bond_count_.end(), 0);
    }

    auto cs_idx = [&](int channel, int site) { return channel * N_ + site; };
    const int* bond_sites = vij_.bond_sites_flat.data();

    for (int p = 0; p < M_total_; ++p) {
        for (int replica = 0; replica < 2; ++replica) {
            int ot = replicas_[replica].op_types[p];
            if (ot == 1 || ot == -1) {
                int site = replicas_[replica].op_sites[p];
                int channel = channel_for_actual(replica, site, p);
                ch_site_op_count_[cs_idx(channel, site)]++;
            } else if (ot == 2) {
                int b = replicas_[replica].op_sites[p];
                int si = bond_sites[b * 2 + 0];
                int sj = bond_sites[b * 2 + 1];
                int c_i = channel_for_actual(replica, si, p);
                int c_j = channel_for_actual(replica, sj, p);
                ch_site_bond_count_[cs_idx(c_i, si)]++;
                ch_site_bond_count_[cs_idx(c_j, sj)]++;
            }
        }
    }

    int total_ops = 0;
    int total_bonds = 0;
    for (int idx = 0; idx < ch_sites; ++idx) {
        ch_site_op_head_[idx] = total_ops;
        total_ops += ch_site_op_count_[idx];
        ch_site_bond_head_[idx] = total_bonds;
        total_bonds += ch_site_bond_count_[idx];
    }
    ch_site_op_list_.assign(total_ops, SiteEvent{});
    ch_site_bond_list_.assign(total_bonds, BondEvent{});

    std::vector<int32_t> op_cursor(ch_sites, 0);
    std::vector<int32_t> bond_cursor(ch_sites, 0);
    for (int p = 0; p < M_total_; ++p) {
        for (int replica = 0; replica < 2; ++replica) {
            int ot = replicas_[replica].op_types[p];
            if (ot == 1 || ot == -1) {
                int site = replicas_[replica].op_sites[p];
                int channel = channel_for_actual(replica, site, p);
                int idx = cs_idx(channel, site);
                ch_site_op_list_[ch_site_op_head_[idx] + op_cursor[idx]++] =
                    SiteEvent{static_cast<int32_t>(p), static_cast<int8_t>(replica)};
            } else if (ot == 2) {
                int b = replicas_[replica].op_sites[p];
                int si = bond_sites[b * 2 + 0];
                int sj = bond_sites[b * 2 + 1];
                int c_i = channel_for_actual(replica, si, p);
                int c_j = channel_for_actual(replica, sj, p);
                int idx_i = cs_idx(c_i, si);
                int idx_j = cs_idx(c_j, sj);
                ch_site_bond_list_[ch_site_bond_head_[idx_i] + bond_cursor[idx_i]++] =
                    BondEvent{static_cast<int32_t>(p), static_cast<int8_t>(replica),
                              static_cast<int32_t>(b), static_cast<int8_t>(0)};
                ch_site_bond_list_[ch_site_bond_head_[idx_j] + bond_cursor[idx_j]++] =
                    BondEvent{static_cast<int32_t>(p), static_cast<int8_t>(replica),
                              static_cast<int32_t>(b), static_cast<int8_t>(1)};
            }
        }
    }
}

void QAQMCRenyiEngine::build_bond_spins_from_ops() {
    if (static_cast<int>(bond_spin_by_replica_.size()) != 2 * M_total_) {
        bond_spin_by_replica_.assign(static_cast<size_t>(2) * M_total_, 0);
    } else {
        std::fill(bond_spin_by_replica_.begin(), bond_spin_by_replica_.end(), 0);
    }
    std::vector<int32_t> channel_state(2 * N_, 0);
    auto ch_idx = [&](int channel, int site) { return channel * N_ + site; };
    const int* bond_sites = vij_.bond_sites_flat.data();

    for (int p = 0; p < M_total_; ++p) {
        for (int replica = 0; replica < 2; ++replica) {
            int ot = replicas_[replica].op_types[p];
            if (ot == 2) {
                int b = replicas_[replica].op_sites[p];
                int si = bond_sites[b * 2 + 0];
                int sj = bond_sites[b * 2 + 1];
                int c_i = channel_for_actual(replica, si, p);
                int c_j = channel_for_actual(replica, sj, p);
                bond_spin_by_replica_[replica * M_total_ + p] =
                    channel_state[ch_idx(c_i, si)] * 2 + channel_state[ch_idx(c_j, sj)];
            } else if (ot == -1) {
                int site = replicas_[replica].op_sites[p];
                int channel = channel_for_actual(replica, site, p);
                channel_state[ch_idx(channel, site)] ^= 1;
            }
        }
    }
}

void QAQMCRenyiEngine::cluster_update() {
    const int* bond_sites = vij_.bond_sites_flat.data();
    auto cs_idx = [&](int channel, int site) { return channel * N_ + site; };
    auto upper_bound_event = [&](const std::vector<BondEvent>& events,
                                 int begin, int end, int val) {
        return static_cast<int>(std::upper_bound(
            events.begin() + begin,
            events.begin() + end,
            val,
            [](int value, const BondEvent& event) { return value < event.p; }
        ) - events.begin());
    };

    build_channel_vertex_lists();
    build_bond_spins_from_ops();

    // Per-colour parallel sweep: sites in the same colour share no bond, so
    // their bond-spin writes cannot race.  Within one site the segment
    // Metropolis remains sequential.
    for (const auto& color_sites : color_groups_) {
        const int n_in_color = static_cast<int>(color_sites.size());
#ifdef QAQMC_USE_OPENMP
        #pragma omp parallel for schedule(static) if (n_in_color > 1)
#endif
        for (int idx = 0; idx < n_in_color; ++idx) {
            const int site = color_sites[idx];
            std::mt19937_64& rng = site_rngs_[site];

            for (int channel = 0; channel < 2; ++channel) {
                const int op_idx = cs_idx(channel, site);
                const int n_sops = ch_site_op_count_[op_idx];
                if (n_sops < 2) continue;
                const int op_base = ch_site_op_head_[op_idx];
                const int bond_base = ch_site_bond_head_[op_idx];
                const int n_bops = ch_site_bond_count_[op_idx];
                const int bond_end = bond_base + n_bops;

                std::vector<int8_t> seg_flipped(n_sops + 1, 0);

                for (int seg = 1; seg < n_sops; ++seg) {
                    int p_start = ch_site_op_list_[op_base + seg - 1].p;
                    int p_end = ch_site_op_list_[op_base + seg].p;
                    double log_ratio = 0.0;

                    int j0 = upper_bound_event(ch_site_bond_list_, bond_base, bond_end, p_start);
                    int j1 = upper_bound_event(ch_site_bond_list_, bond_base, bond_end, p_end);
                    for (int j = j0; j < j1; ++j) {
                        const BondEvent& event = ch_site_bond_list_[j];
                        int p = event.p;
                        int replica = event.replica;
                        int b = event.bond;
                        int si = bond_sites[b * 2 + 0];
                        int sj = bond_sites[b * 2 + 1];
                        int w_idx = bond_spin_by_replica_[replica * M_total_ + p];
                        int n_i = w_idx >> 1;
                        int n_j = w_idx & 1;
                        double w_old = actual_bond_weight(p, b, w_idx);

                        int new_n_i = n_i;
                        int new_n_j = n_j;
                        if (event.endpoint == 0 && si == site) new_n_i ^= 1;
                        if (event.endpoint == 1 && sj == site) new_n_j ^= 1;
                        double w_new = actual_bond_weight(p, b, new_n_i * 2 + new_n_j);

                        log_ratio += ((w_new > 1e-300) ? std::log(w_new) : -1e30)
                                   - ((w_old > 1e-300) ? std::log(w_old) : -1e30);
                    }

                    bool do_flip = (log_ratio >= 0.0) || (renyi_u01(rng) < std::exp(log_ratio));
                    if (!do_flip) continue;

                    for (int j = j0; j < j1; ++j) {
                        const BondEvent& event = ch_site_bond_list_[j];
                        int& w_idx = bond_spin_by_replica_[event.replica * M_total_ + event.p];
                        w_idx ^= (event.endpoint == 0) ? 2 : 1;
                    }
                    seg_flipped[seg] = 1;
                }

                for (int k = 0; k < n_sops; ++k) {
                    bool flip_xor = seg_flipped[k] != seg_flipped[k + 1];
                    if (!flip_xor) continue;
                    const SiteEvent& event = ch_site_op_list_[op_base + k];
                    int& ot = replicas_[event.replica].op_types[event.p];
                    ot = (ot == 1) ? -1 : 1;
                }
            }
        }
    }

    recompute_midpoint_states_from_ops();
}

void QAQMCRenyiEngine::mc_step() {
    diagonal_update();
    cluster_update();
    if (mode_ == Mode::Expanded) {
        ensemble_switch();
        if (!visit_count_ext_.empty()) {
            visit_count_ext_[cur_ens_]++;
        }
    } else {
        topology_toggle();
        visit_count_[cur_topology_]++;
    }
    accumulate_indicator();
}

void QAQMCRenyiEngine::run_steps(int n_steps) {
    for (int i = 0; i < n_steps; ++i) {
        mc_step();
    }
}
