#pragma once
#include "qaqmc_renyi_core.hpp"

#include <cstdint>
#include <random>
#include <vector>

// ─── QAQMCRenyiWorkEngine ────────────────────────────────────────────────────
//
// Two-replica QAQMC nonequilibrium-work Renyi engine.  Drives a backend
// `QAQMCRenyiEngine` in `Mode::Work` along a finite-time λ protocol that
// interpolates between two nested swap-region masks `(A_start, A_end)`,
// and returns a Jarzynski estimate of
//
//     ΔS2 = S2(A_end) − S2(A_start) = − log < exp(−w) >.
//
// When `A_start = ∅` this reduces to the standard `S2(A_end)` estimator from
// D'Emidio (arXiv:2402.05439).  When the pair is non-empty-nested (e.g.
// `(A, A∪B)`), the same trajectory measures the rung `S2(A∪B) − S2(A)`,
// matching the KP ratio-ladder interface.

class QAQMCRenyiWorkEngine {
public:
    struct WorkTrajectoryResult {
        double   work{0.0};                  // dimensionless work w along the trajectory
        double   exp_minus_work{1.0};        // exp(-w)
        int      final_swap_count{0};        // |B| at end of trajectory
        int      unjoined_at_end_count{0};   // (|D| − |B|) at λ=1, see paper "multiply by 1"
        int64_t  topology_attempts{0};
        int64_t  topology_accepts{0};
    };

    struct WorkRunResult {
        double   mean_exp_minus_work{1.0};   // = Z_{A_end} / Z_{A_start}
        double   delta_s2{0.0};              // = -log mean_exp_minus_work
        double   work_mean{0.0};
        double   work_var{0.0};
        int64_t  trajectory_count{0};
        int64_t  total_topology_attempts{0};
        int64_t  total_topology_accepts{0};
        int64_t  total_unjoined_at_end{0};
        // Per-trajectory diagnostics (all sized = trajectory_count after run):
        std::vector<double>  work_samples;                    // raw w per traj
        std::vector<int32_t> final_swap_counts;               // |B| at end per traj
        std::vector<int32_t> unjoined_counts_per_traj;        // |D|-|B| per traj
        std::vector<int64_t> topology_attempts_per_traj;
        std::vector<int64_t> topology_accepts_per_traj;
    };

    QAQMCRenyiWorkEngine(int N, double Omega, double delta_min, double delta_max,
                         double Rb, int M, double epsilon, uint64_t seed,
                         const double* pos, int pos_dim,
                         int neighbor_cutoff = -1, int delta_groups = 600);

    // ── Configuration ─────────────────────────────────────────────────────
    // Set nested region pair (A_start ⊆ A_end); builds D = A_end \ A_start
    // and initialises backend A_mask_ = A_start, B = ∅.
    void set_region_pair(const uint8_t* A_start_mask, const uint8_t* A_end_mask, int len);

    // Convenience: equivalent to set_region_pair(zeros, A_mask).
    void set_region(const uint8_t* A_mask, int len);

    // λ schedule (must be monotonically non-decreasing, start at 0, end at 1).
    void set_lambda_schedule(const std::vector<double>& lambdas);

    // Move the entanglement cut (swap boundary) to slice m_star in
    // [0, M_total]; default is M (the ramp turning point).  Vacuum-resets the
    // operator strings (a previous configuration is generally invalid under
    // the new channel mapping) and invalidates any warm-start checkpoint —
    // call BEFORE thermalize()/import_start_config().
    void set_cut(int m_star);
    int get_cut() const { return backend_.get_cut(); }

    // Per-λ-step counts.  v1 defaults to 1 each, aligned with paper.
    void set_sweeps_per_lambda(int n_topology_sweeps, int n_qaqmc_sweeps);

    // ── Warm-start support ────────────────────────────────────────────────
    // export_start_config copies the A_start-sector configuration (the saved
    // checkpoint if one exists — always a clean A_start config; otherwise the
    // live op strings) for both replicas.  import_start_config installs a
    // previously exported configuration: requires set_region_pair to have
    // been called with the SAME region pair first; it recomputes the midpoint
    // states and seeds the checkpoint chain, so thermalize() can be skipped.
    void export_start_config(std::vector<int32_t>& types0, std::vector<int32_t>& sites0,
                             std::vector<int32_t>& types1, std::vector<int32_t>& sites1) const;
    void import_start_config(const int32_t* types0, const int32_t* sites0,
                             const int32_t* types1, const int32_t* sites1, int len);

    // ── Execution ─────────────────────────────────────────────────────────
    // Thermalise in the start sector (backend A_mask_ = A_start, B = ∅, λ = 0).
    void thermalize(int n_steps);

    // Run one trajectory from current state to λ = 1.  Caller is responsible
    // for ensuring the engine starts in (λ=0, B=∅, backend.A_mask_=A_start).
    WorkTrajectoryResult run_trajectory();

    // Run n_trajectories trajectories, with `decorrelation_steps` of backend
    // mc_step() in the start sector between consecutive trajectories.  Computes
    // ΔS2 via log-sum-exp aggregation of exp(-w).
    WorkRunResult run_trajectories(int n_trajectories, int decorrelation_steps);

    // ── Accessors ─────────────────────────────────────────────────────────
    int get_N() const { return N_; }
    int get_M_total() const { return backend_.get_M_total(); }
    const std::vector<uint8_t>& get_A_start_mask() const { return A_start_mask_; }
    const std::vector<uint8_t>& get_A_end_mask()   const { return A_end_mask_;   }
    const std::vector<uint8_t>& get_D_mask()       const { return D_mask_;       }
    const std::vector<uint8_t>& get_B_mask()       const { return B_mask_;       }
    const std::vector<double>&  get_lambda_schedule() const { return lambda_schedule_; }
    int get_D_size() const { return static_cast<int>(D_sites_.size()); }
    int get_B_size() const;
    QAQMCRenyiEngine& backend() { return backend_; }
    const QAQMCRenyiEngine& backend() const { return backend_; }

private:
    int N_;
    QAQMCRenyiEngine backend_;
    std::mt19937_64 rng_;

    std::vector<uint8_t> A_start_mask_;
    std::vector<uint8_t> A_end_mask_;
    std::vector<uint8_t> D_mask_;
    std::vector<int>     D_sites_;
    std::vector<uint8_t> B_mask_;

    std::vector<double> lambda_schedule_;
    int n_topology_sweeps_per_lambda_{1};
    int n_qaqmc_sweeps_per_lambda_{1};

    // Trajectory-local accumulators
    double  w_current_{0.0};
    int     unjoined_at_end_current_{0};
    int64_t topo_attempts_current_{0};
    int64_t topo_accepts_current_{0};

    // ── Internal helpers ──────────────────────────────────────────────────
    void   reset_to_start_sector();
    void   accumulate_work(double lambda_old, double lambda_new);
    void   topology_sweep_random_permutation(double lambda);
    bool   propose_toggle_site(int site, double lambda);
    void   qaqmc_sweep_current_topology();

    static double log_safe(double x) {  // log with -inf for non-positive
        return x > 0.0 ? std::log(x) : -1e30;
    }
};
