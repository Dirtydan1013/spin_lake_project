#include "sse_core.hpp"
#include <cstring>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

// RNG: uniform01 / randint from qaqmc_core.hpp (Lemire bounded int + top-53-bit
// uniform) — same helpers as the QAQMC engines, no per-call distribution objects.

// ─── Packed bond-op vertex entries: [ p : 32 ][ b : 31 ][ endpoint : 1 ] ─────

static inline int64_t sse_pack_bond_entry(int p, int b, int endpoint) {
    return (static_cast<int64_t>(p) << 32)
         | (static_cast<int64_t>(static_cast<uint32_t>(b)) << 1)
         | static_cast<int64_t>(endpoint & 1);
}
static inline int sse_entry_p(int64_t e)        { return static_cast<int>(e >> 32); }
static inline int sse_entry_b(int64_t e)        { return static_cast<int>((e >> 1) & 0x7FFFFFFF); }
static inline int sse_entry_endpoint(int64_t e) { return static_cast<int>(e & 1); }

// ═════════════════════════════════════════════════════════════════════════════
// Constructor
// ═════════════════════════════════════════════════════════════════════════════

SSEEngine::SSEEngine(int N, double Omega, double delta, double Rb,
                     double beta, double epsilon, uint64_t seed,
                     const double* pos, int pos_dim,
                     int neighbor_cutoff,
                     const double* box, int n_box)
    : N_(N), M_(20), n_ops_(0),
      Omega_(Omega), delta_(delta), Rb_(Rb), beta_(beta), epsilon_(epsilon),
      norm_N_(0.0), energy_shift_(0.0),
      rng_(seed)
{
    diag_obs.init(N);

    // ── Build V_ij (optional neighbor cutoff; optional periodic box) ───────────
    vij_ = build_rydberg_vij(N, Omega, Rb, pos, pos_dim, neighbor_cutoff, box, n_box);
    int n_bonds     = vij_.n_bonds;
    int n_bonds_pad = std::max(n_bonds, 1);

    // ── Compute bond weights ──────────────────────────────────────────────────
    bond_W_.assign(n_bonds_pad * 4, 0.0);
    bond_W_rmax_.assign(n_bonds_pad, 0.0);
    energy_shift_ = 0.0;

    std::vector<double> weights;
    std::vector<int>    op_kind, op_loc;
    weights.reserve(N + n_bonds);
    op_kind.reserve(N + n_bonds);
    op_loc.reserve(N + n_bonds);

    // Site operators: W = Omega / 2
    double site_W = Omega / 2.0;
    for (int i = 0; i < N; ++i) {
        weights.push_back(site_W);
        op_kind.push_back(0);
        op_loc.push_back(i);
    }
    norm_N_ = N * site_W;

    // Bond operators
    for (int b = 0; b < n_bonds; ++b) {
        int si = vij_.bonds_i[b];
        int sj = vij_.bonds_j[b];
        double vij = vij_.vij_list[b];

        // Asymmetric delta per endpoint (consistent with QAQMC convention)
        double di = (vij_.coord_number[si] > 0) ? delta / vij_.coord_number[si] : 0.0;
        double dj = (vij_.coord_number[sj] > 0) ? delta / vij_.coord_number[sj] : 0.0;

        // SSE-specific cij: raw0 = 0 is deliberately excluded from m_abs so
        // that the epsilon safety margin stays effective.  This guarantees
        // W[|11>] > 0, which prevents zero weights in the cluster update.
        //
        // Raw diagonal elements of (-H_bond):
        //   raw0 = 0            (|00>)
        //   raw1 = dj           (|01>)
        //   raw2 = di           (|10>)
        //   raw3 = -vij+di+dj   (|11>)
        //
        // cij = max(0, -min(raws)) + epsilon * min(|raw1|,|raw2|,|raw3|)
        double raw1 = dj;
        double raw2 = di;
        double raw3 = -vij + di + dj;
        double m_min = std::min({0.0, raw1, raw2, raw3});
        double m_nz  = std::min({std::abs(raw1), std::abs(raw2), std::abs(raw3)});
        double cij   = (m_min < 0.0 ? -m_min : 0.0) + epsilon * m_nz;

        double W[4], bmax;
        W[0] = 0.0  + cij;   // |00>
        W[1] = raw1 + cij;   // |01>
        W[2] = raw2 + cij;   // |10>
        W[3] = raw3 + cij;   // |11>
        bmax = std::max({W[0], W[1], W[2], W[3]});

        bond_W_[b * 4 + 0] = W[0];
        bond_W_[b * 4 + 1] = W[1];
        bond_W_[b * 4 + 2] = W[2];
        bond_W_[b * 4 + 3] = W[3];
        bond_W_rmax_[b] = (bmax > 0.0) ? 1.0 / bmax : 0.0;

        energy_shift_ += W[0];  // W[0] == cij (bond offset constant)
        norm_N_        += bmax;

        weights.push_back(bmax);
        op_kind.push_back(1);
        op_loc.push_back(b);
    }

    // Site operators (sigma_x) are inserted as type-1 (diagonal) in the
    // diagonal update, but sigma_x has zero diagonal matrix elements.
    // The type-1 site ops inflate n_ops by N*Omega/2*beta on average,
    // shifting the energy by -N*Omega/2.  Correct for this here.
    energy_shift_ += N * Omega / 2.0;

    beta_norm_     = beta_ * norm_N_;
    inv_beta_norm_ = (beta_norm_ > 0.0) ? 1.0 / beta_norm_ : 0.0;

    // ── Build alias table (Vose's algorithm) ─────────────────────────────────
    n_alias_ = (int)weights.size();
    std::vector<double>  prob(n_alias_);
    std::vector<int32_t> alias(n_alias_);
    for (int i = 0; i < n_alias_; ++i) {
        prob[i]  = weights[i] * n_alias_ / norm_N_;
        alias[i] = i;
    }
    std::vector<int> small, large;
    for (int i = 0; i < n_alias_; ++i) {
        if (prob[i] < 1.0) small.push_back(i);
        else               large.push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        int s = small.back(); small.pop_back();
        int l = large.back(); large.pop_back();
        alias[s]  = l;
        prob[l]  -= (1.0 - prob[s]);
        if (prob[l] < 1.0) small.push_back(l);
        else               large.push_back(l);
    }
    alias_entries_.resize(n_alias_);
    for (int i = 0; i < n_alias_; ++i) {
        alias_entries_[i].prob     = prob[i];
        alias_entries_[i].alias    = alias[i];
        alias_entries_[i].loc_kind = (op_loc[i] << 1) | op_kind[i];
    }

    // ── Initial spin state (random) ───────────────────────────────────────────
    std::uniform_int_distribution<int> coin(0, 1);
    state_.resize(N);
    for (int i = 0; i < N; ++i) state_[i] = coin(rng_);

    // ── Operator string: all identity ─────────────────────────────────────────
    op_types_.assign(M_, 0);
    op_sites_.assign(M_, -1);

    // ── Vertex-list scratch arrays ─────────────────────────────────────────
    site_op_count_.resize(N, 0);
    site_op_head_.resize(N, 0);
    site_bond_count_.resize(N, 0);
    site_bond_head_.resize(N, 0);
    bond_spin_.resize(M_, 0);
    spin_now_.resize(N, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Checkpoint / warm start
// ═════════════════════════════════════════════════════════════════════════════

std::string SSEEngine::get_rng_state() const {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
}

void SSEEngine::set_rng_state(const std::string& s) {
    std::istringstream iss(s);
    iss >> rng_;
}

void SSEEngine::set_config(const int32_t* state, int n_state,
                           const int32_t* types, const int32_t* sites, int len) {
    if (n_state != N_)
        throw std::runtime_error("set_config: state length != N");
    if (len <= 0)
        throw std::runtime_error("set_config: empty operator string");
    for (int i = 0; i < N_; ++i)
        if (state[i] != 0 && state[i] != 1)
            throw std::runtime_error("set_config: state entries must be 0/1");

    int n_ops = 0;
    for (int p = 0; p < len; ++p) {
        int ot = types[p];
        if (ot == 0) continue;
        if (ot == 1 || ot == -1) {
            if (sites[p] < 0 || sites[p] >= N_)
                throw std::runtime_error("set_config: site index out of range");
        } else if (ot == 2) {
            if (sites[p] < 0 || sites[p] >= vij_.n_bonds)
                throw std::runtime_error("set_config: bond index out of range");
        } else {
            throw std::runtime_error("set_config: op type must be in {-1,0,1,2}");
        }
        ++n_ops;
    }

    std::copy(state, state + N_, state_.begin());
    op_types_.assign(types, types + len);
    op_sites_.assign(sites, sites + len);
    M_ = len;
    n_ops_ = n_ops;
    bond_spin_.assign(M_, 0);
    vertex_counts_valid_ = false;
}

// ═════════════════════════════════════════════════════════════════════════════
// Observables
// ═════════════════════════════════════════════════════════════════════════════

double SSEEngine::measure_energy() const {
    // E = -<n_ops> / beta  +  sum_b C_b
    // where C_b = W[b, |00>] = bond_W_[b*4+0]  (the offset constant)
    return -static_cast<double>(n_ops_) / beta_ + energy_shift_;
}

double SSEEngine::measure_density() const {
    double total = 0.0;
    for (int i = 0; i < N_; ++i) total += state_[i];
    return total / N_;
}

double SSEEngine::measure_mz() const {
    // Staggered mz: (1/N) * sum_i (-1)^i * (n_i - 0.5)
    double mz = 0.0;
    for (int i = 0; i < N_; ++i) {
        double phase = (i % 2 == 0) ? 1.0 : -1.0;
        mz += phase * (state_[i] - 0.5);
    }
    return mz / N_;
}

// ═════════════════════════════════════════════════════════════════════════════
// adjust_M_if_needed
// ═════════════════════════════════════════════════════════════════════════════

void SSEEngine::adjust_M_if_needed() {
    int new_M = static_cast<int>(n_ops_ * 1.33);
    if (new_M > M_) {
        op_types_.resize(new_M, 0);
        op_sites_.resize(new_M, -1);
        bond_spin_.resize(new_M, 0);
        M_ = new_M;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Diagonal Update  (+ fused vertex-list counting)
// ═════════════════════════════════════════════════════════════════════════════
//
// For each position p in the operator string:
//   type -1 : off-diagonal — propagate state (no change to op)
//   type 1,2: diagonal     — try to REMOVE with prob (M-n+1) / (beta*normN)
//   type 0  : identity     — try to INSERT a random operator
//
// Insertion is a two-step process:
//   1. Sample category (site vs bond) from alias table — O(1)
//   2. For bond ops: acceptance/rejection based on state-specific weight
//
// The per-site op/bond counting that build_vertex_lists' first pass used to
// do is fused here: the op at each slot is final once this sweep passes it,
// and cluster_update's 1 <-> -1 toggles are count-neutral.

void SSEEngine::diagonal_update() {
    const int* bond_sites = vij_.bond_sites_flat.data();
    const AliasEntry* tab = alias_entries_.data();

    std::fill(site_op_count_.begin(), site_op_count_.end(), 0);
    std::fill(site_bond_count_.begin(), site_bond_count_.end(), 0);

    for (int p = 0; p < M_; ++p) {
        int ot = op_types_[p];

        if (ot == -1) {
            // ── Off-diagonal: propagate state ─────────────────────────────
            int s = op_sites_[p];
            state_[s] ^= 1;
            site_op_count_[s]++;

        } else if (ot == 1 || ot == 2) {
            // ── Diagonal: try removal ─────────────────────────────────────
            // accept prob = min(1, (M-n+1) / (beta*normN)); the clipping is
            // implicit in the u < prob comparison.
            double prob_remove = static_cast<double>(M_ - n_ops_ + 1)
                                 * inv_beta_norm_;
            if (uniform01(rng_) < prob_remove) {
                op_types_[p] = 0;
                op_sites_[p] = -1;
                n_ops_--;
            } else if (ot == 1) {
                site_op_count_[op_sites_[p]]++;
            } else {
                int b = op_sites_[p];
                site_bond_count_[bond_sites[b * 2 + 0]]++;
                site_bond_count_[bond_sites[b * 2 + 1]]++;
            }

        } else {
            // ot == 0: identity — try insertion
            if (n_ops_ == M_) continue;  // buffer full (should not normally occur)

            // accept prob = min(1, beta*normN / (M-n)):
            //   u < bn/(M-n)  ⟺  u*(M-n) < bn   (division-free, same test)
            if (uniform01(rng_) * static_cast<double>(M_ - n_ops_) < beta_norm_) {
                // Sample from alias table
                int i   = randint(rng_, n_alias_);
                const AliasEntry& e = tab[i];
                int idx = (uniform01(rng_) < e.prob) ? i : (int)e.alias;

                int lk  = tab[idx].loc_kind;
                int loc = lk >> 1;

                if ((lk & 1) == 0) {
                    // Single-site diagonal op
                    op_types_[p] = 1;
                    op_sites_[p] = loc;
                    site_op_count_[loc]++;
                    n_ops_++;
                } else {
                    // Bond diagonal op — additional acceptance/rejection
                    // against the precomputed reciprocal envelope (rmax == 0
                    // when W_max <= 0, making the test false).
                    int b      = loc;
                    int si     = bond_sites[b * 2 + 0];
                    int sj     = bond_sites[b * 2 + 1];
                    int w_idx  = state_[si] * 2 + state_[sj];
                    if (uniform01(rng_) < bond_W_[b * 4 + w_idx] * bond_W_rmax_[b]) {
                        op_types_[p] = 2;
                        op_sites_[p] = b;
                        site_bond_count_[si]++;
                        site_bond_count_[sj]++;
                        n_ops_++;
                    }
                }
            }
        }
    }
    vertex_counts_valid_ = true;
}

// ═════════════════════════════════════════════════════════════════════════════
// build_vertex_lists  —  O(M + N) counting sort
// ═════════════════════════════════════════════════════════════════════════════
//
// The counting pass is normally fused into diagonal_update (the op at each
// slot is final once the diagonal sweep passes it, and the cluster update's
// 1 <-> -1 toggles are count-neutral), so this only re-counts when the
// configuration was replaced externally (set_config).  The fill pass is
// additionally fused with the former cluster Phase B: it propagates
// spin_now_ from state_ (tau=0 boundary) and records bond_spin_[p] in the
// same sweep.

void SSEEngine::build_vertex_lists() {
    const int M = M_, N = N_;
    const int* bond_sites = vij_.bond_sites_flat.data();

    if (!vertex_counts_valid_) {
        std::fill(site_op_count_.begin(), site_op_count_.end(), 0);
        std::fill(site_bond_count_.begin(), site_bond_count_.end(), 0);
        for (int p = 0; p < M; ++p) {
            int ot = op_types_[p];
            if (ot == 1 || ot == -1) {
                site_op_count_[op_sites_[p]]++;
            } else if (ot == 2) {
                int b = op_sites_[p];
                site_bond_count_[bond_sites[b * 2 + 0]]++;
                site_bond_count_[bond_sites[b * 2 + 1]]++;
            }
        }
    }

    // Prefix sums for heads
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

    // Fill pass + former Phase B (bond_spin_ from tau=0 state_ propagation).
    std::copy(state_.begin(), state_.end(), spin_now_.begin());

    for (int p = 0; p < M; ++p) {
        int ot = op_types_[p];
        if (ot == 1 || ot == -1) {
            int s = op_sites_[p];
            site_op_list_[site_op_head_[s] + cur_op[s]++] = p;
            if (ot == -1) spin_now_[s] ^= 1;
        } else if (ot == 2) {
            int b  = op_sites_[p];
            int si = bond_sites[b * 2 + 0];
            int sj = bond_sites[b * 2 + 1];
            bond_spin_[p] = (int8_t)(spin_now_[si] * 2 + spin_now_[sj]);
            site_bond_list_[site_bond_head_[si] + cur_bond[si]++] = sse_pack_bond_entry(p, b, 0);
            site_bond_list_[site_bond_head_[sj] + cur_bond[sj]++] = sse_pack_bond_entry(p, b, 1);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Cluster (Line) Update  —  O(M) via vertex lists
// ═════════════════════════════════════════════════════════════════════════════
//
// Phase A: build per-site vertex lists (+ fused bond_spin_ fill)
// Phase C: per-site segment Metropolis using only per-site bond-op lists
// Phase D: reassign op_types using segment flip decisions (O(n_ops_total))
//
// PERIODIC boundary conditions: segments wrap across tau = 0, and a flipped
// wrapping segment (or a site with no single-site ops) also flips state_.

void SSEEngine::cluster_update() {
    if (M_ == 0) return;

    const int N = N_;

    // ── Phase A (+ fused former Phase B: bond_spin_ fill) ────────────────────
    build_vertex_lists();

    // Metropolis accumulator for flipping one segment: plain product of
    // weight ratios Π w_new/w_old (no std::log in the hot loop — the same
    // accept probability as the previous log-sum form).  Zero weights are
    // tallied separately so "excluded → allowed" (inf) and "allowed →
    // excluded" (0) factors can't produce inf*0 = NaN; the running product is
    // renormalised by 1e±100 so long segments can't over/underflow.  A
    // wrapping segment accumulates two ranges (tail + head) before the single
    // accept decision.
    struct RatioAcc {
        double ratio = 1.0;
        int shift = 0;      // ratio_total = ratio * 1e100^shift
        int inf_ct = 0;     // factors with w_old == 0 < w_new  (force-accept)
        int zero_ct = 0;    // factors with w_new == 0 < w_old  (force-reject)
    };

    auto accum_range = [&](RatioAcc& a, int bop_base, int j_begin, int j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            const int64_t e = site_bond_list_[bop_base + j];
            const int p  = sse_entry_p(e);
            const int b  = sse_entry_b(e);
            const int ep = sse_entry_endpoint(e);
            const int w_idx   = bond_spin_[p];
            const int new_idx = w_idx ^ (ep == 0 ? 2 : 1);
            const double w_old = bond_W_[b * 4 + w_idx];
            const double w_new = bond_W_[b * 4 + new_idx];

            if (w_new > 1e-300) {
                if (w_old > 1e-300) {
                    a.ratio *= w_new / w_old;
                    if (a.ratio > 1e100)       { a.ratio *= 1e-100; ++a.shift; }
                    else if (a.ratio < 1e-100) { a.ratio *= 1e100;  --a.shift; }
                } else {
                    ++a.inf_ct;
                }
            } else if (w_old > 1e-300) {
                ++a.zero_ct;
            }
            // w_new == w_old == 0: neutral factor (matches the old
            // (-1e30) - (-1e30) = 0 convention).
        }
    };

    auto accept = [&](const RatioAcc& a) -> bool {
        if (a.inf_ct != a.zero_ct) return a.inf_ct > a.zero_ct;
        if (a.shift >= 1) return true;
        if (a.shift <= -2) return false;
        const double r = (a.shift == -1) ? a.ratio * 1e-100 : a.ratio;
        return (r >= 1.0) || (uniform01(rng_) < r);
    };

    // Helper: flip this site's bit in bond_spin_[p] for range [j_begin, j_end)
    auto flip_bond_range = [&](int bop_base, int j_begin, int j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            const int64_t e = site_bond_list_[bop_base + j];
            bond_spin_[sse_entry_p(e)] ^= (sse_entry_endpoint(e) == 0 ? 2 : 1);
        }
    };

    // Helper: first index j in the site's packed list with entry_p > val.
    // Entries are packed [p:32][b:31][endpoint:1] in ascending p order; the
    // key below is the largest packed value with entry_p == val.
    auto upper_bound_idx = [&](int bop_base, int n_bops, int val) -> int {
        const int64_t* base = site_bond_list_.data() + bop_base;
        const int64_t key = (static_cast<int64_t>(val) << 32) | 0xFFFFFFFFll;
        return (int)(std::upper_bound(base, base + n_bops, key) - base);
    };

    for (int site_i = 0; site_i < N; ++site_i) {
        int n_sops   = site_op_count_[site_i];
        int sop_base = site_op_head_[site_i];
        int n_bops   = site_bond_count_[site_i];
        int bop_base = site_bond_head_[site_i];

        if (n_sops == 0) {
            // No single-site ops: try flipping entire worldline
            RatioAcc a;
            accum_range(a, bop_base, 0, n_bops);
            if (accept(a)) {
                flip_bond_range(bop_base, 0, n_bops);
                state_[site_i] ^= 1;
            }
            continue;
        }

        // Resize seg_flipped scratch
        seg_flipped_.assign(n_sops, 0);

        // Process each segment between consecutive single-site ops (cyclic)
        for (int seg = 0; seg < n_sops; ++seg) {
            int p_start = site_op_list_[sop_base + seg];
            int p_end   = site_op_list_[sop_base + (seg + 1) % n_sops];

            bool wraps = (p_end <= p_start);  // includes full-circle if n_sops==1

            if (!wraps) {
                // Non-wrapping segment: bond ops with position in (p_start, p_end]
                int j0 = upper_bound_idx(bop_base, n_bops, p_start);
                int j1 = upper_bound_idx(bop_base, n_bops, p_end);
                RatioAcc a;
                accum_range(a, bop_base, j0, j1);
                if (accept(a)) {
                    flip_bond_range(bop_base, j0, j1);
                    seg_flipped_[seg] = 1;
                }
            } else {
                // Wrapping segment: (p_start, M-1] + [0, p_end]
                int j0a = upper_bound_idx(bop_base, n_bops, p_start);
                int j1b = upper_bound_idx(bop_base, n_bops, p_end);
                RatioAcc a;
                accum_range(a, bop_base, j0a, n_bops);  // tail
                accum_range(a, bop_base, 0,   j1b);     // head
                if (accept(a)) {
                    flip_bond_range(bop_base, j0a, n_bops);
                    flip_bond_range(bop_base, 0,   j1b);
                    state_[site_i] ^= 1;  // wrapping always contains tau=0
                    seg_flipped_[seg] = 1;
                }
            }
        }

        // ── Phase D: reassign op_types for this site's single-site ops ───────
        // new_type toggles from orig iff the two adjacent segments have
        // different flip status.  Segment k lies between op k and op (k+1), so:
        //   - segment BEFORE op k = segment (k-1+n_sops)%n_sops
        //   - segment AFTER  op k = segment k
        for (int k = 0; k < n_sops; ++k) {
            int seg_before = (k - 1 + n_sops) % n_sops;
            int seg_after  = k;
            bool flip_xor  = seg_flipped_[seg_before] != seg_flipped_[seg_after];
            if (flip_xor) {
                int p_op = site_op_list_[sop_base + k];
                op_types_[p_op] = (op_types_[p_op] == 1) ? -1 : 1;
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// mc_step
// ═════════════════════════════════════════════════════════════════════════════

void SSEEngine::mc_step() {
    auto t0 = std::chrono::high_resolution_clock::now();
    diagonal_update();
    auto t1 = std::chrono::high_resolution_clock::now();
    cluster_update();
    auto t2 = std::chrono::high_resolution_clock::now();
    adjust_M_if_needed();

    time_diag_ += std::chrono::duration<double>(t1 - t0).count();
    time_clus_ += std::chrono::duration<double>(t2 - t1).count();
    mc_steps_++;
}
