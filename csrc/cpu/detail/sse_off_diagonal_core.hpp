#pragma once
#include <vector>
#include <cstdint>

// ─── SSEEngine off-diagonal string (X_C) component ───────────────────────────
//
// Thermal-trace port of QAQMCOffDiagonalCore (qaqmc_off_diagonal_core.hpp):
// measures O_C = Tr[X_C e^{-beta H}] / Tr[e^{-beta H}] for
// X_C = prod_{i in C} sigma_i^x via the same seam + half-line + lambda
// topology-interpolation (Jarzynski work) method.  The seam sits before
// operator slot m_star: occupation propagated through slots 0..m_star-1 is
// n^-, XORed by the active seam bits to n^+ = n^- xor b, then propagation
// continues through m_star..M-1 and wraps to tau = 0 (state_).
//
// Differences from the QAQMC (projector) version, all due to the PERIODIC
// imaginary-time boundary of the trace:
//   * Worldline closure: for site i the parity of type -1 ops XOR the seam
//     bit must be even, so a seam-bit toggle always pairs with one terminal
//     single-site op type toggle (1 <-> -1) -- same as QAQMC -- but the
//     half-line walk WRAPS around tau = 0 instead of failing at a boundary.
//     A wrapped commit additionally flips state_[site] (the flipped segment
//     contains tau = 0).  The only invalid proposal is a site with no
//     single-site operator anywhere in the string.
//   * Seam snapshot bookkeeping on commit (exact, derived from which side of
//     the seam the flipped segment touches):
//         right walk: n^+ flips             -> state_at_seam_plus_[site] ^= 1
//         left  walk: n^- flips (via the toggled terminal) -> minus ^= 1
//         wrapped     (either direction)    -> also state_[site] ^= 1
//   * Bond weights are beta-independent: read straight from bond_W_ (no
//     delta schedule).
//
// SSEEngine holds one SSEOffDiagonalCore (`off_diag_`) and forwards the API;
// hooks in diagonal_update()/build_vertex_lists() apply the seam XOR at
// p == m_star_ so state_/spin_now_/bond_spin_ caches stay consistent.
// topology_sweep() recomputes the seam snapshots first: the cluster update
// may have flipped segments spanning the seam since the last diagonal pass.
class SSEEngine;

class SSEOffDiagonalCore {
public:
    struct HalfLineProposal {
        bool valid = false;
        int terminal_p = -1;          // slot whose op type toggles on commit
        bool wrapped = false;         // walk crossed tau = 0
        double log_physical_ratio = 0.0;
    };

    // Configure the site list C and the seam slot m_star (0 <= m_star < M);
    // resets the seam mask to empty and recomputes the seam snapshots.
    // m_star = 0 (seam at tau = 0) is the natural default for the trace.
    void set_string_sites(const SSEEngine& eng, const std::vector<int>& sites,
                          int m_star);

    // Rebuild state_at_seam_minus_/plus_ from eng's CURRENT state_/op string
    // (read-only pass, no RNG use).
    void recompute_seam_snapshots(const SSEEngine& eng);

    // Read-only: proposal for toggling string_sites_[local_index], walking
    // right (towards increasing slot, wrapping M-1 -> 0) or left.
    HalfLineProposal build_half_line_proposal(const SSEEngine& eng,
                                              int local_index,
                                              bool direction_right) const;

    // Commit a valid proposal: toggle the terminal op type in eng, the seam
    // bit and the side-dependent snapshot here, and state_ if wrapped.
    void commit_half_line_proposal(SSEEngine& eng, int local_index,
                                   bool direction_right,
                                   const HalfLineProposal& prop);

    // One Metropolis attempt at fixed lambda (0 < lambda < 1); returns
    // whether the toggle was accepted.
    bool attempt_string_toggle(SSEEngine& eng, int local_index, double lambda);

    // One sweep: refresh seam snapshots, then one attempt per site of C in
    // random order at fixed lambda.
    void topology_sweep(SSEEngine& eng, double lambda);

    // ── Hooks (called from SSEEngine's sweeps; no-ops when p != m_star_) ──
    void on_diagonal_slice(int p, std::vector<int32_t>& state);
    void on_fill_slice(int p, std::vector<int32_t>& spin_now) const;

    // Set the seam mask AND repair worldline closure: unlike the projector
    // engine (open tau boundaries: any mask is valid), the trace requires
    // parity(type -1 ops on site) XOR seam bit == even per string site.  For
    // each mismatch this toggles the type of the site's first single-site op
    // (weight-neutral: both types carry Omega/2), or, if the site has no
    // single-site op at all, converts the first identity slot into a type -1
    // op as a valid positive-weight seed (caller must re-equilibrate — the
    // trajectory reset path always does).  Recomputes the seam snapshots.
    void set_seam_mask_consistent(SSEEngine& eng, uint64_t mask);

    // ── Accessors ─────────────────────────────────────────────────────────
    void set_seam_mask(uint64_t mask) { seam_mask_ = mask; }
    uint64_t get_seam_mask() const { return seam_mask_; }
    const std::vector<int>& get_string_sites() const { return string_sites_; }
    int get_m_star() const { return m_star_; }
    const std::vector<int32_t>& get_state_at_seam_minus() const { return state_at_seam_minus_; }
    const std::vector<int32_t>& get_state_at_seam_plus()  const { return state_at_seam_plus_; }

private:
    std::vector<int> string_sites_;            // physical site ids in C
    int m_star_{-1};                           // seam slot; -1 = unconfigured
    uint64_t seam_mask_{0};                    // bit k <-> string_sites_[k]
    std::vector<int32_t> state_at_seam_minus_; // n^- at m_star_
    std::vector<int32_t> state_at_seam_plus_;  // n^+ = n^- xor b
};
