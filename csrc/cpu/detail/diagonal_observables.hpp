#pragma once
// Diagonal (basis-state) observables measured on an arbitrary spin state.
//
// Mirrors the conventions of QAQMCEngine's profile observable set (Z_l/C_m_l
// size groups, A_v riding as extra loop sets, VBS/SS reference-triangle gauge,
// sublattice-resolved occupation-SF s vectors) so the SSE engine can measure
// the exact same quantities at its fixed (delta, beta) point.  Implemented as
// a standalone header used by SSEEngine only — the validated QAQMC production
// path is untouched.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

class DiagonalObservables {
public:
    void init(int N) { N_ = N; }
    int N() const { return N_; }

    // ── Geometry setters (same semantics as QAQMCEngine) ─────────────────────
    void set_bulk_sites(const std::vector<int>& sites) { bulk_sites_ = sites; }

    void set_observable_sites(const std::vector<std::vector<int>>& loop_sets,
                              const std::vector<std::vector<int>>& string_sets) {
        loop_sets_ = loop_sets;
        string_sets_ = string_sets;
        build_groups(loop_sets_, loop_group_of_, loop_group_n_);
        build_groups(string_sets_, string_group_of_, string_group_n_);
    }

    void set_vbs_triangles(const std::vector<int>& corners_flat,
                           const std::vector<int>& n1_parity,
                           const std::vector<int>& vbs_sign,
                           const std::vector<int>& ss_sign,
                           int ref00, int ref10) {
        int n_tri = (int)n1_parity.size();
        if ((int)corners_flat.size() != 3 * n_tri ||
            (int)vbs_sign.size() != n_tri || (int)ss_sign.size() != n_tri)
            throw std::runtime_error("set_vbs_triangles: array length mismatch");
        vbs_n_tri_ = n_tri;
        vbs_corners_ = corners_flat;
        vbs_par_.assign(n_tri, 0);
        vbs_sign_.assign(n_tri, 1.0);
        ss_sign_.assign(n_tri, 1.0);
        for (int t = 0; t < n_tri; ++t) {
            vbs_par_[t] = (int8_t)(n1_parity[t] & 1);
            vbs_sign_[t] = vbs_sign[t];
            ss_sign_[t] = ss_sign[t];
        }
        vbs_ref00_ = ref00;
        vbs_ref10_ = ref10;
    }

    void set_occ_sf_site_map(const std::vector<std::vector<double>>& site_cell_R,
                             const std::vector<int>& site_basis,
                             const std::vector<int>& site_in_bulk_cell,
                             int n_basis) {
        if ((int)site_cell_R.size() != N_ || (int)site_basis.size() != N_ ||
            (int)site_in_bulk_cell.size() != N_)
            throw std::runtime_error("set_occ_sf_site_map: arrays must have length N");
        occ_n_basis_ = n_basis;
        occ_basis_ = site_basis;
        occ_in_bulk_.assign(N_, 0);
        for (int i = 0; i < N_; ++i)
            occ_in_bulk_[i] = (int8_t)(site_in_bulk_cell[i] ? 1 : 0);
        occ_dim_ = site_cell_R.empty() ? 0 : (int)site_cell_R[0].size();
        occ_cell_R_flat_.assign((size_t)N_ * occ_dim_, 0.0);
        for (int i = 0; i < N_; ++i) {
            if ((int)site_cell_R[i].size() != occ_dim_)
                throw std::runtime_error("set_occ_sf_site_map: cell_R dim mismatch");
            for (int d = 0; d < occ_dim_; ++d)
                occ_cell_R_flat_[(size_t)i * occ_dim_ + d] = site_cell_R[i][d];
        }
    }

    void set_occ_sf_q_points(const std::vector<std::vector<double>>& q_points) {
        if (occ_cell_R_flat_.empty())
            throw std::runtime_error("set_occ_sf_q_points: call set_occ_sf_site_map first");
        const int n_q = (int)q_points.size();
        occ_q_points_ = q_points;
        occ_cos_.assign((size_t)n_q * N_, 0.0);
        occ_sin_.assign((size_t)n_q * N_, 0.0);
        for (int qi = 0; qi < n_q; ++qi) {
            const auto& q = q_points[qi];
            if ((int)q.size() != occ_dim_)
                throw std::runtime_error("set_occ_sf_q_points: q dim mismatch");
            for (int i = 0; i < N_; ++i) {
                double qr = 0.0;  // phase uses the CELL Bravais position R
                for (int d = 0; d < occ_dim_; ++d)
                    qr += q[d] * occ_cell_R_flat_[(size_t)i * occ_dim_ + d];
                occ_cos_[(size_t)qi * N_ + i] = std::cos(qr);
                occ_sin_[(size_t)qi * N_ + i] = std::sin(qr);
            }
        }
    }

    void set_occ2_sf_site_map(const std::vector<std::vector<double>>& site_cell_R,
                              const std::vector<int>& site_basis, int n_basis) {
        if ((int)site_cell_R.size() != N_ || (int)site_basis.size() != N_)
            throw std::runtime_error("set_occ2_sf_site_map: arrays must have length N");
        if (occ_q_points_.empty())
            throw std::runtime_error("set_occ2_sf_site_map: call set_occ_sf_q_points first");
        if (n_basis != occ_n_basis_)
            throw std::runtime_error("set_occ2_sf_site_map: n_basis must match");
        occ2_basis_ = site_basis;
        const int n_q = (int)occ_q_points_.size();
        occ2_cos_.assign((size_t)n_q * N_, 0.0);
        occ2_sin_.assign((size_t)n_q * N_, 0.0);
        for (int qi = 0; qi < n_q; ++qi) {
            const auto& q = occ_q_points_[qi];
            for (int i = 0; i < N_; ++i) {
                if (site_basis[i] < 0) continue;   // excluded site
                double qr = 0.0;
                for (int d = 0; d < occ_dim_; ++d)
                    qr += q[d] * site_cell_R[i][d];
                occ2_cos_[(size_t)qi * N_ + i] = std::cos(qr);
                occ2_sin_[(size_t)qi * N_ + i] = std::sin(qr);
            }
        }
        occ2_active_ = true;
    }

    // ── Configuration queries ─────────────────────────────────────────────────
    int n_loop_groups()   const { return (int)loop_group_n_.size(); }
    int n_string_groups() const { return (int)string_group_n_.size(); }
    int n_vbs_triangles() const { return vbs_n_tri_; }
    int n_occ_q()         const { return (int)occ_q_points_.size(); }
    int occ_n_basis()     const { return occ_n_basis_; }
    bool occ_ready()      const { return n_occ_q() > 0 && occ_n_basis_ > 0; }
    bool occ2_active()    const { return occ2_active_; }
    bool has_bulk()       const { return !bulk_sites_.empty(); }

    // ── Measurement on a basis state (length-N array of 0/1) ─────────────────
    double measure_density_bulk(const int32_t* state) const {
        if (bulk_sites_.empty()) {
            double s = 0.0;
            for (int i = 0; i < N_; ++i) s += state[i];
            return s / N_;
        }
        double s = 0.0;
        for (int i : bulk_sites_) s += state[i];
        return s / (double)bulk_sites_.size();
    }

    // Signed copy-mean per size group; out has n_loop_groups() entries.
    void measure_loops(const int32_t* state, double* out) const {
        measure_products(state, loop_sets_, loop_group_of_, loop_group_n_, out);
    }

    void measure_strings(const int32_t* state, double* out) const {
        measure_products(state, string_sets_, string_group_of_, string_group_n_, out);
    }

    void measure_vbs_ss(const int32_t* state, double& mvbs, double& mss) const {
        mvbs = mss = 0.0;
        if (vbs_n_tri_ <= 0) return;
        auto tri_state = [&](int t) -> int {
            const int* cc = &vbs_corners_[3 * t];
            return 4 * state[cc[0]] + 2 * state[cc[1]] + state[cc[2]];
        };
        int s00 = tri_state(vbs_ref00_);
        int s10 = tri_state(vbs_ref10_);
        double g = (s10 == s00) ? 1.0 : -1.0;
        double mv = 0.0, ms = 0.0;
        for (int t = 0; t < vbs_n_tri_; ++t) {
            int s = tri_state(t);
            double u = (vbs_par_[t] == 0)
                           ? ((s == s00) ? 1.0 : -1.0)
                           : (((s == s10) ? 1.0 : -1.0) * g);
            mv += vbs_sign_[t] * u;
            ms += ss_sign_[t] * u;
        }
        double inv = 1.0 / (double)vbs_n_tri_;
        mvbs = mv * inv;
        mss = ms * inv;
    }

    // Fourier vectors s_{q,α} = Σ_{i: α(i)=α, n_i=1} e^{i q·R_cell(i)}.
    // Each out array must hold n_occ_q()*occ_n_basis() doubles (zeroed here).
    // occ2 outputs may be nullptr when occ2 is not active.
    void measure_occ_s(const int32_t* state,
                       double* full_re, double* full_im,
                       double* bulk_re, double* bulk_im,
                       double* o2_re, double* o2_im) const {
        const int n_q = n_occ_q();
        const int nb = occ_n_basis_;
        const size_t vlen = (size_t)n_q * nb;
        std::fill(full_re, full_re + vlen, 0.0);
        std::fill(full_im, full_im + vlen, 0.0);
        std::fill(bulk_re, bulk_re + vlen, 0.0);
        std::fill(bulk_im, bulk_im + vlen, 0.0);
        if (occ2_active_ && o2_re) {
            std::fill(o2_re, o2_re + vlen, 0.0);
            std::fill(o2_im, o2_im + vlen, 0.0);
        }
        for (int qi = 0; qi < n_q; ++qi) {
            const double* cosq = &occ_cos_[(size_t)qi * N_];
            const double* sinq = &occ_sin_[(size_t)qi * N_];
            const size_t base = (size_t)qi * nb;
            for (int i = 0; i < N_; ++i) {
                if (!state[i]) continue;
                int a = occ_basis_[i];
                double c = cosq[i], s = sinq[i];
                full_re[base + a] += c;
                full_im[base + a] += s;
                if (occ_in_bulk_[i]) {
                    bulk_re[base + a] += c;
                    bulk_im[base + a] += s;
                }
            }
        }
        if (occ2_active_ && o2_re) {
            for (int qi = 0; qi < n_q; ++qi) {
                const double* cosq = &occ2_cos_[(size_t)qi * N_];
                const double* sinq = &occ2_sin_[(size_t)qi * N_];
                const size_t base = (size_t)qi * nb;
                for (int i = 0; i < N_; ++i) {
                    if (!state[i]) continue;
                    int a = occ2_basis_[i];
                    if (a < 0) continue;
                    o2_re[base + a] += cosq[i];
                    o2_im[base + a] += sinq[i];
                }
            }
        }
    }

private:
    static void build_groups(const std::vector<std::vector<int>>& sets,
                             std::vector<int>& group_of,
                             std::vector<int>& group_n) {
        // Group by first occurrence of set length (same rule as QAQMCEngine,
        // so A_v vertex sets appended after the loops form the trailing group).
        group_of.assign(sets.size(), 0);
        group_n.clear();
        std::vector<size_t> size_at_group;
        for (size_t k = 0; k < sets.size(); ++k) {
            size_t sz = sets[k].size();
            int g = -1;
            for (size_t i = 0; i < size_at_group.size(); ++i)
                if (size_at_group[i] == sz) { g = (int)i; break; }
            if (g < 0) {
                g = (int)size_at_group.size();
                size_at_group.push_back(sz);
                group_n.push_back(0);
            }
            group_of[k] = g;
            group_n[g] += 1;
        }
    }

    void measure_products(const int32_t* state,
                          const std::vector<std::vector<int>>& sets,
                          const std::vector<int>& group_of,
                          const std::vector<int>& group_n,
                          double* out) const {
        const int n_g = (int)group_n.size();
        for (int g = 0; g < n_g; ++g) out[g] = 0.0;
        for (size_t k = 0; k < sets.size(); ++k) {
            double prod = 1.0;
            for (int s : sets[k]) prod *= (1 - 2 * state[s]);
            out[group_of[k]] += prod;
        }
        for (int g = 0; g < n_g; ++g)
            if (group_n[g] > 0) out[g] /= (double)group_n[g];
    }

    int N_{0};
    std::vector<int> bulk_sites_;
    std::vector<std::vector<int>> loop_sets_, string_sets_;
    std::vector<int> loop_group_of_, loop_group_n_;
    std::vector<int> string_group_of_, string_group_n_;

    int vbs_n_tri_{0};
    std::vector<int> vbs_corners_;
    std::vector<int8_t> vbs_par_;
    std::vector<double> vbs_sign_, ss_sign_;
    int vbs_ref00_{0}, vbs_ref10_{0};

    int occ_n_basis_{0}, occ_dim_{0};
    std::vector<int> occ_basis_, occ2_basis_;
    std::vector<int8_t> occ_in_bulk_;
    std::vector<double> occ_cell_R_flat_;
    std::vector<std::vector<double>> occ_q_points_;
    std::vector<double> occ_cos_, occ_sin_, occ2_cos_, occ2_sin_;
    bool occ2_active_{false};
};
