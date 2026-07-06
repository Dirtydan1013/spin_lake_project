#include "qaqmc_renyi_work_core.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ─── ctor ────────────────────────────────────────────────────────────────────

QAQMCRenyiWorkEngine::QAQMCRenyiWorkEngine(
    int N, double Omega, double delta_min, double delta_max,
    double Rb, int M, double epsilon, uint64_t seed,
    const double* pos, int pos_dim,
    int neighbor_cutoff, int delta_groups)
    : N_(N),
      // Backend gets its own derived seed so its internal RNGs are decoupled
      // from this engine's topology-toggle/permutation RNG.
      backend_(N, Omega, delta_min, delta_max, Rb, M, epsilon,
               seed ^ 0x9E3779B97F4A7C15ULL,
               pos, pos_dim, neighbor_cutoff, delta_groups),
      rng_(seed),
      A_start_mask_(N, 0),
      A_end_mask_(N, 0),
      D_mask_(N, 0),
      B_mask_(N, 0)
{
    backend_.set_A_mask_for_work(A_start_mask_.data(), N_);
}

// ─── set_region_pair / set_region ────────────────────────────────────────────

void QAQMCRenyiWorkEngine::set_region_pair(
    const uint8_t* A_start_mask, const uint8_t* A_end_mask, int len)
{
    if (len != N_) {
        throw std::runtime_error("set_region_pair: mask length must equal N");
    }
    A_start_mask_.assign(A_start_mask, A_start_mask + len);
    A_end_mask_  .assign(A_end_mask,   A_end_mask   + len);

    // Validate nested: A_start ⊆ A_end.
    for (int i = 0; i < N_; ++i) {
        if (A_start_mask_[i] && !A_end_mask_[i]) {
            throw std::runtime_error(
                "set_region_pair: A_start must be a subset of A_end (nested pairs only)");
        }
    }

    D_mask_.assign(N_, 0);
    D_sites_.clear();
    for (int i = 0; i < N_; ++i) {
        if (A_end_mask_[i] && !A_start_mask_[i]) {
            D_mask_[i] = 1;
            D_sites_.push_back(i);
        }
    }

    B_mask_.assign(N_, 0);
    // A new region pair invalidates any prior checkpoint (which was specific
    // to the previous A_start sector); vacuum-reset the backend.
    backend_.invalidate_op_string_checkpoint();
    backend_.set_A_mask_for_work(A_start_mask_.data(), N_);
}

void QAQMCRenyiWorkEngine::set_region(const uint8_t* A_mask, int len)
{
    if (len != N_) {
        throw std::runtime_error("set_region: mask length must equal N");
    }
    std::vector<uint8_t> zeros(N_, 0);
    set_region_pair(zeros.data(), A_mask, len);
}

// ─── set_lambda_schedule ─────────────────────────────────────────────────────

void QAQMCRenyiWorkEngine::set_lambda_schedule(const std::vector<double>& lambdas) {
    if (lambdas.size() < 2) {
        throw std::runtime_error("lambda_schedule must contain at least two points");
    }
    for (size_t i = 1; i < lambdas.size(); ++i) {
        if (lambdas[i] < lambdas[i - 1]) {
            throw std::runtime_error("lambda_schedule must be non-decreasing");
        }
    }
    if (std::abs(lambdas.front()) > 0.0) {
        throw std::runtime_error("lambda_schedule must start at 0.0");
    }
    if (std::abs(lambdas.back() - 1.0) > 0.0) {
        throw std::runtime_error("lambda_schedule must end at 1.0");
    }
    lambda_schedule_ = lambdas;
}

void QAQMCRenyiWorkEngine::set_cut(int m_star) {
    backend_.set_cut(m_star);
    // A configuration equilibrated under the old cut is generally invalid
    // under the new channel mapping: reset to the trivially-valid vacuum in
    // the A_start sector and drop any saved checkpoint.
    B_mask_.assign(N_, 0);
    backend_.invalidate_op_string_checkpoint();
    backend_.set_A_mask_for_work(A_start_mask_.data(), N_);
}

void QAQMCRenyiWorkEngine::export_start_config(
    std::vector<int32_t>& types0, std::vector<int32_t>& sites0,
    std::vector<int32_t>& types1, std::vector<int32_t>& sites1) const {
    if (backend_.has_op_string_checkpoint()) {
        // The checkpoint is refreshed at each trajectory start and is always
        // a clean A_start-sector configuration.
        types0 = backend_.get_checkpoint_op_types(0);
        sites0 = backend_.get_checkpoint_op_sites(0);
        types1 = backend_.get_checkpoint_op_types(1);
        sites1 = backend_.get_checkpoint_op_sites(1);
    } else {
        types0 = backend_.get_op_types(0);
        sites0 = backend_.get_op_sites(0);
        types1 = backend_.get_op_types(1);
        sites1 = backend_.get_op_sites(1);
    }
}

void QAQMCRenyiWorkEngine::import_start_config(
    const int32_t* types0, const int32_t* sites0,
    const int32_t* types1, const int32_t* sites1, int len) {
    if (len != 2 * backend_.get_M()) {
        throw std::runtime_error(
            "import_start_config: op-string length must equal M_total");
    }
    // Walk the backend to the A_start sector (mask + mode), then install the
    // configuration and seed the checkpoint chain so run_trajectories'
    // reset_to_start_sector restores it without any thermalization.
    B_mask_.assign(N_, 0);
    backend_.set_A_mask(A_start_mask_.data(), N_);
    backend_.set_mode(QAQMCRenyiEngine::Mode::Work);
    backend_.set_replica_op_string(0, types0, sites0, len);
    backend_.set_replica_op_string(1, types1, sites1, len);
    backend_.recompute_midpoint_states();
    backend_.save_op_string_checkpoint();
}

void QAQMCRenyiWorkEngine::set_sweeps_per_lambda(int n_topology_sweeps, int n_qaqmc_sweeps) {
    if (n_topology_sweeps < 0 || n_qaqmc_sweeps < 0) {
        throw std::runtime_error("sweeps_per_lambda must be non-negative");
    }
    n_topology_sweeps_per_lambda_ = n_topology_sweeps;
    n_qaqmc_sweeps_per_lambda_    = n_qaqmc_sweeps;
}

// ─── Accessors ───────────────────────────────────────────────────────────────

int QAQMCRenyiWorkEngine::get_B_size() const {
    int s = 0;
    for (auto b : B_mask_) s += b;
    return s;
}

// ─── Internal helpers ────────────────────────────────────────────────────────

void QAQMCRenyiWorkEngine::reset_to_start_sector() {
    B_mask_.assign(N_, 0);
    if (backend_.has_op_string_checkpoint()) {
        // Checkpoint-chain path: restore the previously-saved A_start config
        // instead of clearing ops to vacuum.  We still have to walk the mask
        // back to A_start (the previous trajectory ended at A_end), but skip
        // the op-string clear (which set_A_mask_for_work does).  set_A_mask
        // only writes A_mask_ / A_masks_[0,1] / mode flags and recomputes
        // state_at_M from current ops; we then immediately overwrite ops with
        // the checkpoint, so the stale state_at_M is fine — restore() will
        // recompute it from the checkpoint ops.
        backend_.set_A_mask(A_start_mask_.data(), N_);
        backend_.set_mode(QAQMCRenyiEngine::Mode::Work);
        backend_.restore_op_string_checkpoint();
    } else {
        // First call (or right after set_region_pair): vacuum reset.
        backend_.set_A_mask_for_work(A_start_mask_.data(), N_);
    }
}

void QAQMCRenyiWorkEngine::accumulate_work(double lambda_old, double lambda_new) {
    // Δw = -[ log g(λ_new, B) - log g(λ_old, B) ]  with g = λ^|B| (1-λ)^(|D|-|B|)
    const int b = get_B_size();
    const int d = static_cast<int>(D_sites_.size());
    if (d == 0) return;  // trivial case, work = 0

    // Δ_join: b * (log λ_new - log λ_old)
    // Handle endpoint: λ_old = 0 only happens at the very first step where
    // invariant guarantees b = 0, so this term is 0 either way.  We still need
    // to avoid log(0) explicitly so we branch on b first.
    double d_join = 0.0;
    if (b > 0) {
        // b > 0 implies we've already inserted at least one site, so the
        // previous lambda where this happened was already > 0, hence lambda_old
        // > 0 by the invariant.  Defensive guard anyway.
        if (lambda_old <= 0.0 || lambda_new <= 0.0) {
            // pathological — caller violated invariant
            throw std::runtime_error(
                "accumulate_work: λ ≤ 0 with |B| > 0 violates invariant");
        }
        d_join = static_cast<double>(b) * (std::log(lambda_new) - std::log(lambda_old));
    }

    // Δ_split: (d - b) * (log(1-λ_new) - log(1-λ_old))
    // Endpoint: if λ_new == 1 and b < d, apply paper's "multiply by 1" prescription:
    // treat the (1-λ)-factor as 1 (Δ_split contribution = 0) and record diagnostic.
    double d_split = 0.0;
    const int unjoined = d - b;
    if (unjoined > 0) {
        if (lambda_new >= 1.0) {
            // "multiply by 1" prescription
            unjoined_at_end_current_ += unjoined;
            // d_split stays 0
        } else {
            d_split = static_cast<double>(unjoined)
                    * (std::log1p(-lambda_new) - std::log1p(-lambda_old));
        }
    }

    w_current_ += -(d_join + d_split);
}

bool QAQMCRenyiWorkEngine::propose_toggle_site(int site, double lambda) {
    topo_attempts_current_++;

    // backend's A_mask_ currently reflects A_start ∪ B; toggling site
    // toggles the corresponding bit in both backend and B_mask_.
    const bool is_insert = (B_mask_[site] == 0);

    // log-ratio of QAQMC topology weight  Z(A_mask_^{site}) / Z(A_mask_)
    const double log_ratio_qaqmc = backend_.log_weight_ratio_for_toggle(site);
    if (log_ratio_qaqmc <= -1e29) {
        // spin/path mismatch → reject
        return false;
    }

    // log-ratio of g(λ)-factor for single-bit toggle
    //   insert: g_new/g_old = λ / (1-λ)
    //   remove: g_new/g_old = (1-λ) / λ
    double log_ratio_lambda;
    if (lambda <= 0.0 || lambda >= 1.0) {
        // Defensive: schedule interior should never pass endpoint λ to propose.
        // At endpoints the only legal topology is fixed; reject silently.
        return false;
    }
    if (is_insert) {
        log_ratio_lambda = std::log(lambda) - std::log1p(-lambda);
    } else {
        log_ratio_lambda = std::log1p(-lambda) - std::log(lambda);
    }

    const double log_accept = log_ratio_qaqmc + log_ratio_lambda;
    double accept_prob;
    if (log_accept >= 0.0) {
        accept_prob = 1.0;
    } else {
        accept_prob = std::exp(log_accept);
    }

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double r = u01(rng_);
    if (r >= accept_prob) {
        return false;
    }

    // Accept: apply the bit flip in both backend and B_mask_.
    backend_.apply_single_bit_toggle(site);
    B_mask_[site] ^= 1;
    topo_accepts_current_++;
    return true;
}

void QAQMCRenyiWorkEngine::topology_sweep_random_permutation(double lambda) {
    if (D_sites_.empty()) return;

    std::vector<int> order = D_sites_;
    std::shuffle(order.begin(), order.end(), rng_);
    for (int site : order) {
        propose_toggle_site(site, lambda);
    }
}

void QAQMCRenyiWorkEngine::qaqmc_sweep_current_topology() {
    // backend is in Mode::Work so mc_step() is just diagonal + cluster update
    // at the current A_mask_ (= A_start ∪ B).
    backend_.mc_step();
}

// ─── Execution: thermalize, run_trajectory, run_trajectories ────────────────

void QAQMCRenyiWorkEngine::thermalize(int n_steps) {
    reset_to_start_sector();
    for (int i = 0; i < n_steps; ++i) {
        backend_.mc_step();
    }
    // Seed the checkpoint chain.  Subsequent reset_to_start_sector() calls
    // (from run_trajectories) will restore this configuration instead of
    // vacuum-resetting, so decorrelation_steps only needs to break local
    // correlations rather than do a full thermalization.
    backend_.save_op_string_checkpoint();
}

QAQMCRenyiWorkEngine::WorkTrajectoryResult QAQMCRenyiWorkEngine::run_trajectory() {
    if (lambda_schedule_.size() < 2) {
        throw std::runtime_error("run_trajectory: lambda_schedule not set");
    }
    if (A_end_mask_.empty()) {
        throw std::runtime_error("run_trajectory: region pair not set");
    }

    w_current_                = 0.0;
    unjoined_at_end_current_  = 0;
    topo_attempts_current_    = 0;
    topo_accepts_current_     = 0;

    // Trivial case: D = ∅ → ΔS2 = 0.
    if (D_sites_.empty()) {
        WorkTrajectoryResult res;
        res.work = 0.0;
        res.exp_minus_work = 1.0;
        res.final_swap_count = 0;
        res.unjoined_at_end_count = 0;
        return res;
    }

    for (size_t m = 0; m + 1 < lambda_schedule_.size(); ++m) {
        const double lam_old = lambda_schedule_[m];
        const double lam_new = lambda_schedule_[m + 1];

        // 1. accumulate work BEFORE updating λ, using current B
        accumulate_work(lam_old, lam_new);

        // 2. update current λ (implicit; lambda_new is now "current" for the sweeps)
        //    only matters as argument to topology_sweep / propose_toggle_site.

        // 3. topology sweep(s) at lambda_new — only on D
        for (int k = 0; k < n_topology_sweeps_per_lambda_; ++k) {
            topology_sweep_random_permutation(lam_new);
        }

        // 4. QAQMC sweep(s) at the current backend mask (= A_start ∪ B)
        for (int k = 0; k < n_qaqmc_sweeps_per_lambda_; ++k) {
            qaqmc_sweep_current_topology();
        }
    }

    WorkTrajectoryResult res;
    res.work = w_current_;
    res.exp_minus_work = std::exp(-w_current_);
    res.final_swap_count = get_B_size();
    res.unjoined_at_end_count = unjoined_at_end_current_;
    res.topology_attempts = topo_attempts_current_;
    res.topology_accepts  = topo_accepts_current_;
    return res;
}

QAQMCRenyiWorkEngine::WorkRunResult
QAQMCRenyiWorkEngine::run_trajectories(int n_trajectories, int decorrelation_steps)
{
    if (n_trajectories < 0) {
        throw std::runtime_error("run_trajectories: n_trajectories must be non-negative");
    }
    if (decorrelation_steps < 0) {
        throw std::runtime_error("run_trajectories: decorrelation_steps must be non-negative");
    }

    WorkRunResult out;
    const size_t nt = static_cast<size_t>(n_trajectories);
    out.work_samples.reserve(nt);
    out.final_swap_counts.reserve(nt);
    out.unjoined_counts_per_traj.reserve(nt);
    out.topology_attempts_per_traj.reserve(nt);
    out.topology_accepts_per_traj.reserve(nt);

    // Trivial short-circuit: D = ∅ → ΔS2 = 0 regardless of trajectory count.
    if (D_sites_.empty()) {
        out.mean_exp_minus_work = 1.0;
        out.delta_s2 = 0.0;
        out.trajectory_count = static_cast<int64_t>(std::max(0, n_trajectories));
        return out;
    }

    // log-sum-exp accumulator for exp(-w):  M = max(-w_k);   sum = Σ exp(-w_k - M)
    double w_sum = 0.0;
    double w_sq_sum = 0.0;
    double lse_max = -std::numeric_limits<double>::infinity();
    double lse_sum_exp = 0.0;

    for (int t = 0; t < n_trajectories; ++t) {
        // 1) reset to start sector.  With checkpoint chain on (which thermalize
        //    sets up), this restores the saved A_start config in O(M) ops
        //    rather than vacuum-resetting and re-thermalizing from scratch.
        reset_to_start_sector();

        // 2) decorrelate at λ = 0 in the start sector.  Short here, since the
        //    restored checkpoint is already at A_start equilibrium; we only
        //    need to break local correlations.
        for (int s = 0; s < decorrelation_steps; ++s) {
            backend_.mc_step();
        }

        // 3) refresh checkpoint with the post-decorr config so the chain
        //    advances forward in A_start sector between trajectories (paper's
        //    "save → re-equilibrate → save" cycle).
        backend_.save_op_string_checkpoint();

        // 4) run one trajectory
        WorkTrajectoryResult r = run_trajectory();

        // 5) accumulate diagnostics
        out.work_samples.push_back(r.work);
        out.final_swap_counts.push_back(static_cast<int32_t>(r.final_swap_count));
        out.unjoined_counts_per_traj.push_back(static_cast<int32_t>(r.unjoined_at_end_count));
        out.topology_attempts_per_traj.push_back(r.topology_attempts);
        out.topology_accepts_per_traj.push_back(r.topology_accepts);
        w_sum    += r.work;
        w_sq_sum += r.work * r.work;
        out.total_topology_attempts += r.topology_attempts;
        out.total_topology_accepts  += r.topology_accepts;
        out.total_unjoined_at_end   += r.unjoined_at_end_count;

        // log-sum-exp update with x_k = -w_k
        const double x = -r.work;
        if (x > lse_max) {
            const double old_max = lse_max;
            lse_max = x;
            if (std::isfinite(old_max)) {
                lse_sum_exp = lse_sum_exp * std::exp(old_max - lse_max) + 1.0;
            } else {
                lse_sum_exp = 1.0;
            }
        } else {
            lse_sum_exp += std::exp(x - lse_max);
        }
    }

    const int64_t n = static_cast<int64_t>(n_trajectories);
    out.trajectory_count = n;
    if (n > 0) {
        out.work_mean = w_sum / static_cast<double>(n);
        const double mean = out.work_mean;
        if (n > 1) {
            out.work_var = (w_sq_sum - n * mean * mean) / static_cast<double>(n - 1);
        } else {
            out.work_var = 0.0;
        }
        // log <exp(-w)> = lse_max + log(lse_sum_exp / n) = lse_max + log(lse_sum_exp) - log(n)
        const double log_mean_exp_minus_work = lse_max + std::log(lse_sum_exp) - std::log(static_cast<double>(n));
        out.mean_exp_minus_work = std::exp(log_mean_exp_minus_work);
        out.delta_s2 = -log_mean_exp_minus_work;
    } else {
        out.mean_exp_minus_work = 1.0;
        out.delta_s2 = 0.0;
    }
    return out;
}
