#pragma once
#include <vector>
#include <cstdint>

// ─── QAQMCEngine off-diagonal string (X_C) component ────────────────────────
//
// Genuinely separate class from QAQMCEngine (the "diagonal" engine declared
// in qaqmc_core.hpp): every seam/half-line/topology_sweep member and method
// for measuring an off-diagonal string X_C = prod_{i in C} sigma_i^x via the
// nonequilibrium-work method lives here, not in qaqmc_core.hpp waiting to be
// borrowed. See document/QAQMC_LineCluster_Jarzynski_String_Implementation.md
// for the physics and specs/csrc/qaqmc_core.md for the full contract.
//
// QAQMCEngine holds one QAQMCOffDiagonalCore as a private member (`off_diag_`)
// and forwards its public API with one-line pass-throughs, so the Python-
// facing surface (qaqmc_cpp.QAQMCEngine.set_string_sites(...), etc.) is
// unchanged. Methods here take the owning QAQMCEngine explicitly (there is no
// implicit `this` into a QAQMCEngine, since this is a different class) because
// they need read -- and, for commit/toggle, write -- access to its operator
// string, bond tables, and RNG; that access is granted via
// `friend class QAQMCOffDiagonalCore;` declared inside QAQMCEngine.
class QAQMCEngine;

class QAQMCOffDiagonalCore {
public:
    struct HalfLineProposal {
        bool valid = false;
        std::int64_t terminal_p = -1; // operator-string position to toggle on commit
        double log_physical_ratio = 0.0;
    };

    // Configure the site list C and the imaginary-time cut position m_star;
    // resets the seam mask to empty (B = empty set) and recomputes the seam
    // snapshots from eng's current operator string. The cut sits *before*
    // operator index m_star in the 0-indexed operator string: occupation is
    // propagated through operators 0..m_star-1 unmodified (n^-), then XORed
    // by the active seam bits, then propagation continues through operators
    // m_star..M_total-1 (n^+ = n^- xor b).
    void set_string_sites(const QAQMCEngine& eng, const std::vector<int>& sites,
                          std::int64_t m_star);

    // Recompute state_at_seam_minus_/plus_ from eng's CURRENT op_types_/
    // op_sites_/seam_mask_ without resampling any operators (read-only pass,
    // no RNG use) -- unlike QAQMCEngine::diagonal_update() (via
    // on_diagonal_slice), which also refreshes these as a side effect of
    // resampling. Needed after externally restoring an operator string via
    // QAQMCEngine::set_op_string() when the seam snapshots must be back in
    // sync before the next build_half_line_proposal() call.
    void recompute_seam_snapshots(const QAQMCEngine& eng);

    // Read-only: build the proposal for toggling string_sites_[local_index],
    // walking right (direction_right=true, towards increasing p) or left
    // (false, towards decreasing p). Does not mutate any state.
    // valid=false means the walk reached the p=0/M_total_ boundary before
    // any single-site operator for that site (R_boundary=0 for the fixed
    // |0...0> boundary condition of this engine -> always reject).
    HalfLineProposal build_half_line_proposal(const QAQMCEngine& eng, int local_index,
                                              bool direction_right) const;

    // Commit a previously-built, currently-valid proposal for local_index:
    // toggle the terminal operator's type (1 <-> -1) in eng and the seam
    // mask bit here. Caller must not have mutated eng's op_types_/op_sites_/
    // this object's seam_mask_ between build and commit.
    void commit_half_line_proposal(QAQMCEngine& eng, int local_index, const HalfLineProposal& prop);

    // One Metropolis attempt for string_sites_[local_index] at fixed lambda:
    // random direction, accept with min(1, (lambda/(1-lambda))^{+-1} * R),
    // commit if accepted. lambda must satisfy 0 < lambda < 1 (endpoints are
    // the caller's responsibility). Returns whether the toggle was accepted.
    bool attempt_string_toggle(QAQMCEngine& eng, int local_index, double lambda);

    // One sweep: random permutation over all string_sites_, one
    // attempt_string_toggle each, at fixed lambda.
    void topology_sweep(QAQMCEngine& eng, double lambda);

    // ── Seam drag (cut-position Jarzynski) ───────────────────────────────
    // Drag the cut from m_star_ to m_new (either direction) and return the
    // exact log weight ratio ln[W_{m_new}(sigma)/W_{m_star_}(sigma)] of the
    // CURRENT configuration (-inf if a crossed bond op has zero weight in
    // the target frame -- an exact-zero trajectory, not an error). Moving
    // the cut past slot p changes only that operator's frame: single-site
    // ops have constant weight (ratio 1, type -1 just flips the n^- book-
    // keeping), bond ops not touching an ACTIVE seam site are frame-blind,
    // and bond ops touching one contribute W[n^-]/W[n^+] at the slot's own
    // delta_at(p). Consumes no RNG; never mutates eng. Rebuilds the seam
    // snapshots on entry (cluster_update stales them; same guard as
    // topology_sweep) and leaves m_star_/snapshots consistent at m_new.
    double seam_drag_to(const QAQMCEngine& eng, std::int64_t m_new);

    // Rao-Blackwellized single-rung ratio for dragging the cut one slot
    // right (crossing slot p = m_star_) or left (p = m_star_ - 1): the
    // conditional expectation of W_target(sigma)/W_current(sigma) over the
    // diagonal-op choice at the crossing slot given the state trajectory,
    //
    //   E[ratio | state] = 1                          (flip slot: frame-blind)
    //                    = Lambda_tgt(s)/Lambda_cur(s) (diagonal slot)
    //   Lambda(s) = N * site_W + sum_b W_b(s)   at the slot's own delta_at(p),
    //
    // which is exactly the stationary conditional of diagonal_update's
    // alias-proposal + envelope-rejection sampling. Averaging this over
    // equilibrium samples of Z_X(m_star_) estimates Z_X(m+-1)/Z_X(m) with
    // BOUNDED per-sample values: the raw per-config ratio is heavy-tailed
    // (a crossed bond op can sit at W ~ epsilon*m_abs in one frame and O(1)
    // in the other, ratio ~ 1/epsilon), which is why plain Jarzynski dragging
    // shows one-sided bias at reachable sample sizes; the RB form replaces
    // the whale by a ratio of two O(N) sums. Expectation is unchanged
    // (Rao-Blackwell). Rebuilds the seam snapshots on entry; consumes no
    // RNG; never mutates eng or this object's m_star_/mask.
    double seam_rung_rb_ratio(const QAQMCEngine& eng, bool right);

    // Block generalization: read-only RB log-ratio for dragging the cut from
    // m_star_ to m_new (either direction) in ONE rung. Given the worldline,
    // the diagonal-op choices at the crossed slots are conditionally
    // independent, so the RB conditional of the k-slot block weight ratio
    // factorizes exactly into the per-slot Lambda ratios:
    //
    //   E[W_tgt/W_cur | worldline] = prod_{p crossed, diagonal} Lambda_tgt(s_p)/Lambda_cur(s_p)
    //
    // (flip slots contribute no factor; their sigma^x just advances the local
    // n^- bookkeeping). Averaging exp(return) over equilibrium samples of
    // Z_X(m_star_) estimates Z_X(m_new)/Z_X(m_star_). Per-slot log
    // contributions are O(L_C/N) but NOT independent along the block: the
    // worldline's imaginary-time correlations make the rung log-sd grow
    // ~linearly (not sqrt) in k beyond a correlation crossover, so cost/error
    // improves with k only up to that point -- calibrate k so the rung log-sd
    // stays <~0.3 (docs/design/seam_drag_curve.md SS7; kagome 4x4/M=4096:
    // k ~ 64). Does NOT move the cut; mutates only the snapshot cache (entry
    // recompute). No RNG.
    double seam_rb_log_ratio_to(const QAQMCEngine& eng, std::int64_t m_new);

    // Re-anchor the cut at m_new WITHOUT work accumulation: m_star_ = m_new
    // plus a snapshot recompute; seam_mask_ untouched (closure parity does
    // not reference the cut position, so no repair is needed). The current
    // configuration is a sample of the OLD ensemble -- callers must
    // decorrelate before measuring, same contract as set_seam_mask_consistent.
    void seam_set_position(const QAQMCEngine& eng, std::int64_t m_new);

    // ── Hooks called from QAQMCEngine::diagonal_update()/cluster_update() ──
    // Both are no-ops unless p == m_star_ (including when no string is
    // configured: m_star_ == -1 can never equal a valid p >= 0).
    // diagonal_update() needs the full pre/post snapshot (consumed later by
    // build_half_line_proposal); cluster_update()'s spin_now_ is a transient
    // scratch array that only needs the in-place XOR, no snapshot.
    void on_diagonal_slice(std::int64_t p, std::vector<int32_t>& state);
    void on_cluster_slice(std::int64_t p, std::vector<int32_t>& spin_now) const;

    // Set the seam mask AND repair per-site worldline closure. The fixed
    // |0...0> boundary at BOTH tau ends imposes, per string site,
    // parity(sigma^x ops) == seam bit; every update (diagonal, cluster,
    // half-line toggle) preserves it, but writing the mask directly breaks
    // it whenever a bit changes -- the raw setter below leaves the engine
    // sampling an unphysical sector that no amount of decorrelation can
    // leave. Use this for any sector reset (trajectory restarts,
    // thermalize); repairs by toggling the site's first single-site op
    // (repurposing a diagonal slot if it has none), then refreshes the
    // seam snapshots.
    void set_seam_mask_consistent(QAQMCEngine& eng, uint64_t mask);

    // ── Accessors ────────────────────────────────────────────────────────
    // Raw mask write, no closure repair: bit k of mask <-> string_sites_[k].
    // Only valid when the caller separately guarantees closure (e.g. paired
    // with set_op_string of a config recorded at this mask).
    void set_seam_mask(uint64_t mask) { seam_mask_ = mask; }
    uint64_t get_seam_mask() const { return seam_mask_; }
    const std::vector<int>& get_string_sites() const { return string_sites_; }
    std::int64_t get_m_star() const { return m_star_; }

    // Pre-/post-seam spin state at m_star_. Both are full-length (N)
    // snapshots; only string_sites_ entries can differ between the two.
    // Unset (empty) if no string is configured.
    const std::vector<int32_t>& get_state_at_seam_minus() const { return state_at_seam_minus_; }
    const std::vector<int32_t>& get_state_at_seam_plus()  const { return state_at_seam_plus_; }

private:
    std::vector<int> string_sites_;            // physical site ids in C (size L_C)
    std::int64_t m_star_{-1};                  // cut position; -1 = no string configured
    uint64_t seam_mask_{0};                    // bit k <-> string_sites_[k]
    std::vector<int32_t> state_at_seam_minus_; // n^- at m_star_ (pre-seam)
    std::vector<int32_t> state_at_seam_plus_;  // n^+ at m_star_ (post-seam)
};
