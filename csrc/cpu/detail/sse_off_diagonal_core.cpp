#include "../include/sse_core.hpp"
#include <numeric>
#include <stdexcept>
#include <cmath>

// ─── SSEOffDiagonalCore ──────────────────────────────────────────────────────
//
// Like the QAQMC engine, SSE never caches a persistent worldline: state_/
// spin_now_/bond_spin_ are re-derived from op_types_/op_sites_ (+ the seam)
// every sweep, so a half-line commit only toggles the terminal op's type
// (plus state_[site] when the flipped segment wraps tau = 0); the walk exists
// to find the terminal and to accumulate the bond-weight ratio.

void SSEOffDiagonalCore::set_string_sites(const SSEEngine& eng,
                                          const std::vector<int>& sites,
                                          int m_star) {
    if (m_star < 0 || m_star >= eng.M_) {
        throw std::invalid_argument("m_star must satisfy 0 <= m_star < M "
                                    "(m_star = 0 recommended: M grows)");
    }
    if (sites.size() > 64) {
        throw std::invalid_argument("string_sites size must be <= 64");
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

void SSEOffDiagonalCore::recompute_seam_snapshots(const SSEEngine& eng) {
    if (m_star_ < 0) return;
    std::vector<int32_t> state(eng.state_);
    for (int p = 0; p < m_star_; ++p) {
        if (eng.op_types_[p] == -1) state[eng.op_sites_[p]] ^= 1;
    }
    state_at_seam_minus_ = state;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) state[string_sites_[k]] ^= 1;
    }
    state_at_seam_plus_ = state;
}

void SSEOffDiagonalCore::set_seam_mask_consistent(SSEEngine& eng, uint64_t mask) {
    seam_mask_ = mask;
    if (m_star_ < 0) return;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        const int site = string_sites_[k];
        int parity = 0;
        int first_pm = -1, first_id = -1;
        for (int p = 0; p < eng.M_; ++p) {
            const int ot = eng.op_types_[p];
            if (ot == -1 && eng.op_sites_[p] == site) parity ^= 1;
            if (first_pm < 0 && (ot == 1 || ot == -1) && eng.op_sites_[p] == site)
                first_pm = p;
            if (first_id < 0 && ot == 0) first_id = p;
        }
        const int want = (int)((mask >> k) & 1ULL);
        if (parity == want) continue;
        if (first_pm >= 0) {
            eng.op_types_[first_pm] = (eng.op_types_[first_pm] == 1) ? -1 : 1;
        } else if (first_id >= 0) {
            eng.op_types_[first_id] = -1;
            eng.op_sites_[first_id] = site;
            eng.n_ops_++;
            eng.vertex_counts_valid_ = false;
        } else {
            throw std::runtime_error("set_seam_mask: cannot repair worldline "
                                     "parity (operator string full)");
        }
    }
    recompute_seam_snapshots(eng);
}

void SSEOffDiagonalCore::on_diagonal_slice(int p, std::vector<int32_t>& state) {
    if (p != m_star_) return;
    state_at_seam_minus_ = state;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) state[string_sites_[k]] ^= 1;
    }
    state_at_seam_plus_ = state;
}

void SSEOffDiagonalCore::on_fill_slice(int p, std::vector<int32_t>& spin_now) const {
    if (p != m_star_) return;
    for (size_t k = 0; k < string_sites_.size(); ++k) {
        if ((seam_mask_ >> k) & 1ULL) spin_now[string_sites_[k]] ^= 1;
    }
}

SSEOffDiagonalCore::HalfLineProposal SSEOffDiagonalCore::build_half_line_proposal(
        const SSEEngine& eng, int local_index, bool direction_right) const {
    HalfLineProposal out;
    if (m_star_ < 0 || local_index < 0 ||
        local_index >= (int)string_sites_.size()) {
        return out;
    }
    const int site = string_sites_[local_index];
    const int M = eng.M_;
    const int* bond_sites = eng.vij_.bond_sites_flat.data();

    // Occupation snapshot on the walk's starting side of the seam; updated
    // by other sites' -1 flips as the walk proceeds.  `site`'s own value is
    // constant until the terminal by construction.
    std::vector<int32_t> local_state =
        direction_right ? state_at_seam_plus_ : state_at_seam_minus_;

    double ratio = 1.0;
    int shift = 0;  // ratio_total = ratio * 1e100^shift
    static const double LOG_1E100 = std::log(1e100);

    auto bond_ratio_touching_site = [&](int p) -> bool {
        int b = eng.op_sites_[p];
        int si = bond_sites[b * 2 + 0];
        int sj = bond_sites[b * 2 + 1];
        if (si != site && sj != site) return true;
        int ni = local_state[si], nj = local_state[sj];
        const double* W = &eng.bond_W_[b * 4];
        double w_old = W[ni * 2 + nj];
        double w_new = (si == site) ? W[(1 - ni) * 2 + nj]
                                    : W[ni * 2 + (1 - nj)];
        if (w_old <= 0.0 || w_new <= 0.0) return false;
        ratio *= w_new / w_old;
        if (ratio > 1e100)       { ratio *= 1e-100; ++shift; }
        else if (ratio < 1e-100) { ratio *= 1e100;  --shift; }
        return true;
    };

    // Periodic walk over all M slots starting next to the seam; `wrapped`
    // becomes true once the walk crosses tau = 0 (slot M-1 -> 0 going right,
    // slot 0 -> M-1 going left).
    for (int step = 0; step < M; ++step) {
        int p;
        bool wrapped_here;
        if (direction_right) {
            int q = m_star_ + step;
            wrapped_here = (q >= M);
            p = wrapped_here ? q - M : q;
        } else {
            int q = m_star_ - 1 - step;
            wrapped_here = (q < 0);
            p = wrapped_here ? q + M : q;
        }
        int ot = eng.op_types_[p];
        if ((ot == 1 || ot == -1) && eng.op_sites_[p] == site) {
            out.terminal_p = p;
            out.valid = true;
            out.wrapped = wrapped_here;
            out.log_physical_ratio = std::log(ratio) + shift * LOG_1E100;
            return out;
        }
        if (ot == 2) {
            if (!bond_ratio_touching_site(p)) return out;
        } else if (ot == -1) {
            local_state[eng.op_sites_[p]] ^= 1;
        }
    }
    return out;  // no single-site op for `site` anywhere: invalid
}

void SSEOffDiagonalCore::commit_half_line_proposal(SSEEngine& eng, int local_index,
                                                   bool direction_right,
                                                   const HalfLineProposal& prop) {
    if (!prop.valid) return;
    eng.op_types_[prop.terminal_p] =
        (eng.op_types_[prop.terminal_p] == 1) ? -1 : 1;
    seam_mask_ ^= (1ULL << local_index);
    const int site = string_sites_[local_index];
    // Which snapshot the flipped segment touches (see header): the segment
    // extends from the seam to the terminal on the walk's side.
    if (direction_right) state_at_seam_plus_[site]  ^= 1;
    else                 state_at_seam_minus_[site] ^= 1;
    // A wrapped segment contains tau = 0.
    if (prop.wrapped) eng.state_[site] ^= 1;
}

bool SSEOffDiagonalCore::attempt_string_toggle(SSEEngine& eng, int local_index,
                                               double lambda) {
    if (lambda <= 0.0 || lambda >= 1.0) return false;

    bool direction_right = uniform01(eng.rng_) < 0.5;
    HalfLineProposal prop = build_half_line_proposal(eng, local_index,
                                                     direction_right);
    if (!prop.valid) return false;

    bool active = (seam_mask_ >> local_index) & 1ULL;
    double log_odds = std::log(lambda) - std::log1p(-lambda);
    double log_topology_ratio = active ? -log_odds : log_odds;
    double log_accept = prop.log_physical_ratio + log_topology_ratio;

    double log_u = std::log(uniform01(eng.rng_));
    if (log_u >= std::min(0.0, log_accept)) return false;

    commit_half_line_proposal(eng, local_index, direction_right, prop);
    return true;
}

void SSEOffDiagonalCore::topology_sweep(SSEEngine& eng, double lambda) {
    int L = (int)string_sites_.size();
    if (L == 0) return;
    // The cluster update may have flipped segments spanning the seam since
    // the last diagonal pass refreshed the snapshots: rebuild them (O(M)).
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
