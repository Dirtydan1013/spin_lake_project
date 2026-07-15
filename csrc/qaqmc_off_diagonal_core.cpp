#include "qaqmc_core.hpp"
#include <cstring>
#include <stdexcept>

// ─── QAQMCOffDiagonalCore ────────────────────────────────────────────────────
//
// This engine never caches a persistent worldline/leg representation:
// QAQMCEngine::diagonal_update()/cluster_update() always re-derive occupation
// from op_types_/op_sites_ (+ our seam_mask_/m_star_) from scratch on every
// call. Consequently a half-line "flip legs along the path" is a no-op at
// commit time -- only the terminal single-site operator's type actually
// needs to change; the walk itself exists purely to (a) find that terminal
// and (b) accumulate the physical weight ratio of the bond operators it
// passes.

void QAQMCOffDiagonalCore::set_string_sites(const QAQMCEngine& eng,
                                            const std::vector<int>& sites, int m_star) {
    if (m_star < 0 || m_star >= eng.M_total_) {
        throw std::invalid_argument("m_star must satisfy 0 <= m_star < M_total");
    }
    if (sites.size() > 64) {
        throw std::invalid_argument("string_sites size must be <= 64 (fits in seam_mask_ bits)");
    }
    for (int s : sites) {
        if (s < 0 || s >= eng.N_) {
            throw std::invalid_argument("string_sites entries must be valid site indices");
        }
    }
    string_sites_ = sites;
    m_star_ = m_star;
    seam_mask_ = 0;
    state_at_seam_minus_.assign(eng.N_, 0);
    state_at_seam_plus_.assign(eng.N_, 0);
    recompute_seam_snapshots(eng);
}

void QAQMCOffDiagonalCore::set_seam_mask_consistent(QAQMCEngine& eng, uint64_t mask) {
    seam_mask_ = mask;
    if (m_star_ < 0) return;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        const int site = string_sites_[k];
        int parity = 0;
        int first_pm = -1, first_steal = -1;
        for (int p = 0; p < eng.M_total_; ++p) {
            const int ot = eng.op_types_[p];
            if (ot == -1 && eng.op_site_at(p) == site) parity ^= 1;
            if (first_pm < 0 && (ot == 1 || ot == -1) && eng.op_site_at(p) == site)
                first_pm = p;
            if (first_steal < 0 && (ot == 2 || (ot == 1 && eng.op_site_at(p) != site)))
                first_steal = p;
        }
        const int want = (int)((mask >> k) & 1ULL);
        if (parity == want) continue;
        if (first_pm >= 0) {
            eng.op_types_[first_pm] = (eng.op_types_[first_pm] == 1) ? -1 : 1;
        } else if (first_steal >= 0) {
            // No single-site op on this site (e.g. a fresh engine): repurpose
            // a diagonal slot as the parity-fixing sigma^x. The resulting
            // configuration is valid but atypical -- caller must decorrelate
            // before measuring, same contract as the SSE core.
            eng.op_types_[first_steal] = -1;
            eng.set_op_site_at(first_steal, site);
            eng.vertex_counts_valid_ = false;
        } else {
            throw std::runtime_error("set_seam_mask_consistent: cannot repair "
                                     "worldline closure (no repurposable operator)");
        }
    }
    recompute_seam_snapshots(eng);
}

void QAQMCOffDiagonalCore::recompute_seam_snapshots(const QAQMCEngine& eng) {
    if (m_star_ < 0) return;
    std::vector<int32_t> state(eng.N_, 0);
    for (int p = 0; p < m_star_; ++p) {
        if (eng.op_types_[p] == -1) state[eng.op_site_at(p)] ^= 1;
    }
    state_at_seam_minus_ = state;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) state[string_sites_[k]] ^= 1;
    }
    state_at_seam_plus_ = state;
}

void QAQMCOffDiagonalCore::on_diagonal_slice(int p, std::vector<int32_t>& state) {
    if (p != m_star_) return;
    state_at_seam_minus_ = state;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) state[string_sites_[k]] ^= 1;
    }
    state_at_seam_plus_ = state;
}

void QAQMCOffDiagonalCore::on_cluster_slice(int p, std::vector<int32_t>& spin_now) const {
    if (p != m_star_) return;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) spin_now[string_sites_[k]] ^= 1;
    }
}

QAQMCOffDiagonalCore::HalfLineProposal QAQMCOffDiagonalCore::build_half_line_proposal(
        const QAQMCEngine& eng, int local_index, bool direction_right) const {
    HalfLineProposal out;
    if (m_star_ < 0 || local_index < 0 || local_index >= (int)string_sites_.size()) {
        return out;
    }
    const int site = string_sites_[local_index];
    const int* bond_sites = eng.model_->vij.bond_sites_flat.data();

    // local_state tracks state_before(p) for whichever p is currently being
    // examined (see derivation in the doc / spec): it starts at the known
    // seam snapshot and is updated by undoing/applying single-site flips for
    // sites OTHER than `site` as the walk proceeds (site's own occupation is
    // constant between the seam and the terminal by construction: no other
    // single-site op for `site` can occur before the terminal is reached).
    std::vector<int32_t> local_state = direction_right ? state_at_seam_plus_ : state_at_seam_minus_;

    // The physical ratio Π w_new/w_old over touching bonds is accumulated as
    // a plain product (renormalised by 1e±100 so long walks can't over/
    // underflow); a single std::log at the terminal converts it back to the
    // log_physical_ratio the callers expect.  This replaces two std::log
    // calls per touching bond op with one division.
    double ratio = 1.0;
    int shift = 0;  // ratio_total = ratio * 1e100^shift
    static const double LOG_1E100 = std::log(1e100);

    // Returns false iff the bond op's old or flipped weight is non-positive
    // (invalid proposal -> caller must reject).  Non-touching bonds
    // contribute nothing and are always "valid".
    auto bond_ratio_touching_site = [&](int p) -> bool {
        int b = eng.op_site_at(p);
        int si = bond_sites[b * 2 + 0];
        int sj = bond_sites[b * 2 + 1];
        if (si != site && sj != site) return true;
        double delta = eng.delta_at(p);
        double di = delta * eng.model_->vij.inv_coord[si];
        double dj = delta * eng.model_->vij.inv_coord[sj];
        double W[4], wmax;
        QAQMCEngine::compute_bond_W_inline(di, dj, eng.model_->vij.vij_list[b], eng.epsilon_, W, wmax);
        int ni = local_state[si], nj = local_state[sj];
        double w_old = W[ni * 2 + nj];
        double w_new = (si == site) ? W[(1 - ni) * 2 + nj] : W[ni * 2 + (1 - nj)];
        if (w_old <= 0.0 || w_new <= 0.0) return false;
        ratio *= w_new / w_old;
        if (ratio > 1e100)       { ratio *= 1e-100; ++shift; }
        else if (ratio < 1e-100) { ratio *= 1e100;  --shift; }
        return true;
    };

    if (direction_right) {
        for (int p = m_star_; p < eng.M_total_; ++p) {
            int ot = eng.op_types_[p];
            if ((ot == 1 || ot == -1) && eng.op_site_at(p) == site) {
                out.terminal_p = p;
                out.valid = true;
                out.log_physical_ratio = std::log(ratio) + shift * LOG_1E100;
                return out;
            }
            if (ot == 2) {
                if (!bond_ratio_touching_site(p)) return out;
            } else if (ot == -1) {
                local_state[eng.op_site_at(p)] ^= 1;
            }
        }
        return out; // hit the right boundary without a terminal: invalid
    } else {
        for (int p = m_star_ - 1; p >= 0; --p) {
            int ot = eng.op_types_[p];
            if ((ot == 1 || ot == -1) && eng.op_site_at(p) == site) {
                out.terminal_p = p;
                out.valid = true;
                out.log_physical_ratio = std::log(ratio) + shift * LOG_1E100;
                return out;
            }
            if (ot == 2) {
                if (!bond_ratio_touching_site(p)) return out;
            } else if (ot == -1) {
                local_state[eng.op_site_at(p)] ^= 1;
            }
        }
        return out; // hit the left boundary without a terminal: invalid
    }
}

void QAQMCOffDiagonalCore::commit_half_line_proposal(QAQMCEngine& eng, int local_index,
                                                     const HalfLineProposal& prop) {
    if (!prop.valid) return;
    eng.op_types_[prop.terminal_p] = (eng.op_types_[prop.terminal_p] == 1) ? -1 : 1;
    seam_mask_ ^= (1ULL << local_index);
    // Keep the cached seam snapshots in sync for any later
    // build_half_line_proposal() in the same topology_sweep() (the next
    // diagonal_update() refreshes them from scratch). Which snapshot changes
    // depends on which side of the seam the terminal sits: a right-walk
    // terminal (p >= m_star_) alters the worldline only after the seam, so
    // n^+ flips and n^- is untouched; a left-walk terminal (p < m_star_)
    // flips the propagated n^-, and since the seam bit b flips too,
    // n^+ = n^- xor b is unchanged.
    const int site = string_sites_[local_index];
    if (prop.terminal_p >= m_star_) state_at_seam_plus_[site]  ^= 1;
    else                            state_at_seam_minus_[site] ^= 1;
}

bool QAQMCOffDiagonalCore::attempt_string_toggle(QAQMCEngine& eng, int local_index, double lambda) {
    if (lambda <= 0.0 || lambda >= 1.0) return false;

    bool direction_right = uniform01(eng.rng_) < 0.5;
    HalfLineProposal prop = build_half_line_proposal(eng, local_index, direction_right);
    if (!prop.valid) return false;

    bool active = (seam_mask_ >> local_index) & 1ULL;
    double log_odds = std::log(lambda) - std::log1p(-lambda);
    double log_topology_ratio = active ? -log_odds : log_odds;
    double log_accept = prop.log_physical_ratio + log_topology_ratio;

    double log_u = std::log(uniform01(eng.rng_));
    if (log_u >= std::min(0.0, log_accept)) return false;

    commit_half_line_proposal(eng, local_index, prop);
    return true;
}

void QAQMCOffDiagonalCore::topology_sweep(QAQMCEngine& eng, double lambda) {
    int L = (int)string_sites_.size();
    if (L == 0) return;
    // cluster_update() can flip segments spanning m_star_ (changing the seam
    // occupation) and only diagonal_update() refreshes the snapshots; since
    // mc_step() runs diagonal -> cluster, the cache is usually stale on
    // entry here. Rebuild it (O(M), no RNG). Same guard as the SSE core.
    recompute_seam_snapshots(eng);
    std::vector<int> order(L);
    std::iota(order.begin(), order.end(), 0);
    for (int i = L - 1; i > 0; --i) {
        int j = randint(eng.rng_, i + 1);
        std::swap(order[i], order[j]);
    }
    for (int idx : order) {
        attempt_string_toggle(eng, idx, lambda);
    }
}
