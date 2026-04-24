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

    auto occ = build_channel_occupancies();
    reproject_site_ops_for_current_topology(occ);
    update_midpoint_from_channels(occ);
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
    auto occ_from = build_channel_occupancies(from_mask);
    auto occ_to = build_channel_occupancies(to_mask);
    double log_from = log_weight_for_site_with_mask(site, from_mask, occ_from);
    double log_to = log_weight_for_site_with_mask(site, to_mask, occ_to);
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

    // Pre-compute occ and log weights for the current topology once.
    auto occ_from = build_channel_occupancies(ensembles_[from_ens].A_mask);

    const double propose_prob = 1.0 / static_cast<double>(nbrs.size());
    std::vector<double> accept_probs(nbrs.size(), 0.0);
    double off_diag_weight = 0.0;

    for (size_t i = 0; i < nbrs.size(); ++i) {
        const int to_ens = nbrs[i];
        double log_ratio = 0.0;
        bool feasible = true;

        auto occ_to = build_channel_occupancies(ensembles_[to_ens].A_mask);
        for (int s = 0; s < N_; ++s) {
            if (ensembles_[from_ens].A_mask[s] == ensembles_[to_ens].A_mask[s]) {
                continue;
            }
            const double log_from = log_weight_for_site_with_mask(
                s, ensembles_[from_ens].A_mask, occ_from);
            const double log_to = log_weight_for_site_with_mask(
                s, ensembles_[to_ens].A_mask, occ_to);
            if (log_to <= -1e29) {
                feasible = false;
                break;
            }
            if (log_from <= -1e29) {
                log_ratio = 1e30;  // from-weight zero ⇒ from-ensemble unreachable, force accept
                break;
            }
            log_ratio += (log_to - log_from);
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
        auto occ = build_channel_occupancies();
        reproject_site_ops_for_current_topology(occ);
        update_midpoint_from_channels(occ);
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

std::vector<int32_t> QAQMCRenyiEngine::build_channel_occupancies(const std::vector<uint8_t>& mask) const {
    std::vector<int32_t> occ(2 * (M_total_ + 1) * N_, 0);
    auto idx = [&](int channel, int p, int site) {
        return ((channel * (M_total_ + 1) + p) * N_) + site;
    };

    for (int channel = 0; channel < 2; ++channel) {
        for (int site = 0; site < N_; ++site) {
            int value = 0;
            occ[idx(channel, 0, site)] = 0;
            for (int p = 0; p < M_total_; ++p) {
                int replica = replica_for_with_mask(channel, site, p, mask);
                if (replicas_[replica].op_types[p] == -1 && replicas_[replica].op_sites[p] == site) {
                    value ^= 1;
                }
                occ[idx(channel, p + 1, site)] = value;
            }
        }
    }
    return occ;
}

std::vector<int32_t> QAQMCRenyiEngine::build_channel_occupancies() const {
    return build_channel_occupancies(A_mask_);
}

void QAQMCRenyiEngine::update_midpoint_from_channels(const std::vector<int32_t>& occ) {
    auto idx = [&](int channel, int p, int site) {
        return ((channel * (M_total_ + 1) + p) * N_) + site;
    };
    for (int replica = 0; replica < 2; ++replica) {
        for (int site = 0; site < N_; ++site) {
            replicas_[replica].state_at_M[site] = occ[idx(replica, M_, site)];
        }
    }
}

void QAQMCRenyiEngine::recompute_midpoint_states() {
    auto occ = build_channel_occupancies();
    update_midpoint_from_channels(occ);
}

void QAQMCRenyiEngine::reproject_site_ops_for_current_topology(const std::vector<int32_t>& occ) {
    auto idx = [&](int channel, int p, int site) {
        return ((channel * (M_total_ + 1) + p) * N_) + site;
    };

    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_total_; ++p) {
            int& ot = replicas_[replica].op_types[p];
            if (ot != 1 && ot != -1) continue;
            int site = replicas_[replica].op_sites[p];
            int channel = channel_for_actual(replica, site, p);
            int before = occ[idx(channel, p, site)];
            int after = occ[idx(channel, p + 1, site)];
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

double QAQMCRenyiEngine::log_weight_for_site_with_mask(int site, const std::vector<uint8_t>& mask,
                                                       const std::vector<int32_t>& occ) const {
    if (site < 0 || site >= N_) {
        throw std::runtime_error("site out of range");
    }

    auto idx = [&](int channel, int p, int site_idx) {
        return ((channel * (M_total_ + 1) + p) * N_) + site_idx;
    };

    for (int channel = 0; channel < 2; ++channel) {
        if (occ[idx(channel, M_total_, site)] != 0) {
            return -1e30;
        }
    }

    const int* bond_sites = vij_.bond_sites_flat.data();
    double log_weight = 0.0;

    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_total_; ++p) {
            const int ot = replicas_[replica].op_types[p];
            if (ot != 2) continue;

            int b = replicas_[replica].op_sites[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            if (si != site && sj != site) continue;

            int c_i = channel_for_actual_with_mask(replica, si, p, mask);
            int c_j = channel_for_actual_with_mask(replica, sj, p, mask);
            int n_i = occ[idx(c_i, p, si)];
            int n_j = occ[idx(c_j, p, sj)];
            int w_idx = n_i * 2 + n_j;

            double w = actual_bond_weight(p, b, w_idx);

            if (w <= 1e-300) {
                return -1e30;
            }
            log_weight += std::log(w);
        }
    }

    return log_weight;
}

double QAQMCRenyiEngine::log_weight_ratio_for_site(int site, int from_topology, int to_topology) const {
    if (from_topology < 0 || from_topology > 1 || to_topology < 0 || to_topology > 1) {
        throw std::runtime_error("topology index out of range");
    }
    auto occ_from = build_channel_occupancies(A_masks_[from_topology]);
    auto occ_to = build_channel_occupancies(A_masks_[to_topology]);
    double log_from = log_weight_for_site_with_mask(site, A_masks_[from_topology], occ_from);
    double log_to = log_weight_for_site_with_mask(site, A_masks_[to_topology], occ_to);
    return log_to - log_from;
}

std::array<std::vector<int32_t>, 4> QAQMCRenyiEngine::get_site_paths(int site) const {
    if (site < 0 || site >= N_) {
        throw std::runtime_error("site out of range");
    }
    auto occ = build_channel_occupancies();
    auto idx = [&](int channel, int p, int site_idx) {
        return ((channel * (M_total_ + 1) + p) * N_) + site_idx;
    };

    std::array<std::vector<int32_t>, 4> paths;
    for (auto& v : paths) v.resize(M_total_ + 1, 0);

    for (int p = 0; p <= M_total_; ++p) {
        paths[0][p] = occ[idx(0, p, site)];
        paths[1][p] = occ[idx(1, p, site)];
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
    const double log_ratio = log_weight_ratio_for_site(diff_site_, cur_topology_, proposed);
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
    A_mask_ = A_masks_[cur_topology_];
    auto occ = build_channel_occupancies();
    reproject_site_ops_for_current_topology(occ);
    update_midpoint_from_channels(occ);
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

void QAQMCRenyiEngine::cluster_update() {
    auto occ = build_channel_occupancies();
    auto occ_idx = [&](int channel, int p, int site) {
        return ((channel * (M_total_ + 1) + p) * N_) + site;
    };
    const int* bond_sites = vij_.bond_sites_flat.data();

    for (int site = 0; site < N_; ++site) {
        for (int channel = 0; channel < 2; ++channel) {
            std::vector<int> site_ops;
            site_ops.reserve(M_total_ + 2);
            site_ops.push_back(-1);
            for (int p = 0; p < M_total_; ++p) {
                int replica = replica_for(channel, site, p);
                int ot = replicas_[replica].op_types[p];
                if ((ot == 1 || ot == -1) && replicas_[replica].op_sites[p] == site) {
                    site_ops.push_back(p);
                }
            }
            site_ops.push_back(M_total_);

            if (site_ops.size() <= 3) continue;

            for (size_t seg = 2; seg + 1 < site_ops.size(); ++seg) {
                int p_start = site_ops[seg - 1];
                int p_end = site_ops[seg];
                double log_ratio = 0.0;

                for (int p = std::max(0, p_start + 1); p <= std::min(M_total_ - 1, p_end); ++p) {
                    int replica = replica_for(channel, site, p);
                    if (replicas_[replica].op_types[p] != 2) continue;

                    int b = replicas_[replica].op_sites[p];
                    int si = bond_sites[b * 2 + 0];
                    int sj = bond_sites[b * 2 + 1];
                    if (si != site && sj != site) continue;

                    int c_i = channel_for_actual(replica, si, p);
                    int c_j = channel_for_actual(replica, sj, p);
                    int n_i = occ[occ_idx(c_i, p, si)];
                    int n_j = occ[occ_idx(c_j, p, sj)];
                    double w_old = actual_bond_weight(p, b, n_i * 2 + n_j);

                    int new_n_i = n_i;
                    int new_n_j = n_j;
                    if (si == site && c_i == channel) new_n_i ^= 1;
                    if (sj == site && c_j == channel) new_n_j ^= 1;
                    double w_new = actual_bond_weight(p, b, new_n_i * 2 + new_n_j);

                    log_ratio += ((w_new > 1e-300) ? std::log(w_new) : -1e30)
                               - ((w_old > 1e-300) ? std::log(w_old) : -1e30);
                }

                bool do_flip = (log_ratio >= 0.0) || (renyi_u01(rngs_[channel]) < std::exp(log_ratio));
                if (!do_flip) continue;

                for (int p = std::max(0, p_start + 1); p <= std::min(M_total_ - 1, p_end); ++p) {
                    occ[occ_idx(channel, p, site)] ^= 1;
                }
            }
        }
    }

    for (int replica = 0; replica < 2; ++replica) {
        for (int p = 0; p < M_total_; ++p) {
            int ot = replicas_[replica].op_types[p];
            if (ot != 1 && ot != -1) continue;
            int site = replicas_[replica].op_sites[p];
            int channel = channel_for_actual(replica, site, p);
            int before = occ[occ_idx(channel, p, site)];
            int after = occ[occ_idx(channel, p + 1, site)];
            replicas_[replica].op_types[p] = (before == after) ? 1 : -1;
        }
    }

    update_midpoint_from_channels(occ);
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
