# QAQMC Renyi Work Engine Code Review

Date: 2026-05-24

Scope: current working-tree implementation of `QAQMCRenyiWorkEngine`, including
`csrc/qaqmc_renyi_work_core.*`, `csrc/qaqmc_renyi_core.*`,
`csrc/bindings.cpp`, `src/engines/qaqmc_renyi_work.py`, and
`tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py`.

## Summary

The new Renyi work engine builds successfully, but it should not be merged yet.
There are two correctness risks in how the work engine resets and drives the
backend topology, and the new integration test is currently not collected by
pytest.

## Verification Performed

Build commands:

```bash
cmake -S . -B build
cmake --build build -j 4
```

Result: build passed.

Test collection command:

```bash
/home/tohenry20109/miniconda3/envs/qaqmc/bin/python \
  -m pytest --collect-only \
  tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py -q
```

Result:

```text
no tests collected
```

Direct script command:

```bash
/home/tohenry20109/miniconda3/envs/qaqmc/bin/python \
  tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py
```

Result:

```text
ModuleNotFoundError: No module named 'src'
```

## Findings

### 1. High: reset to start sector does not reproject operators

Location:

- `csrc/qaqmc_renyi_work_core.cpp`
- `QAQMCRenyiWorkEngine::reset_to_start_sector()`
- `csrc/qaqmc_renyi_core.cpp`
- `QAQMCRenyiEngine::set_A_mask()`

Current behavior:

```cpp
void QAQMCRenyiWorkEngine::reset_to_start_sector() {
    B_mask_.assign(N_, 0);
    backend_.set_mode(QAQMCRenyiEngine::Mode::Work);
    backend_.set_A_mask(A_start_mask_.data(), N_);
}
```

`set_A_mask()` changes `A_mask_` and recomputes midpoint states, but it does not
reproject all site operators under the new mask. This is risky because a
trajectory may end with backend mask `A_start | B`. The next trajectory resets
the mask to `A_start`, but the operator string may still contain site-op
projection choices associated with the previous topology, especially for
`p >= M`.

Impact:

The next decorrelation steps or trajectory may run under inconsistent boundary
conditions. This is a correctness issue for the Jarzynski estimator.

Recommended fix:

Add a backend API specifically for work-mode reset, for example:

```cpp
void QAQMCRenyiEngine::set_A_mask_for_work(const uint8_t* mask, int len) {
    if (len != N_) throw std::runtime_error("A_mask length mismatch");
    A_mask_.assign(mask, mask + len);
    A_masks_[0] = A_mask_;
    A_masks_[1] = A_mask_;
    cur_topology_ = 0;
    diff_site_ = -1;
    mode_ = Mode::Work;

    OffdiagPaths paths;
    build_offdiag_paths(A_mask_, paths);
    reproject_site_ops_for_mask_with_paths(A_mask_, paths);
    recompute_midpoint_states();
}
```

Then use this API from constructor, `set_region_pair()`, and
`reset_to_start_sector()`.

### 2. High: Mode::Work is overwritten by set_A_mask()

Location:

- `csrc/qaqmc_renyi_work_core.cpp`
- `QAQMCRenyiWorkEngine` constructor
- `QAQMCRenyiWorkEngine::set_region_pair()`
- `QAQMCRenyiWorkEngine::reset_to_start_sector()`
- `csrc/qaqmc_renyi_core.cpp`
- `QAQMCRenyiEngine::set_A_mask()`

Current behavior:

The work engine often does:

```cpp
backend_.set_mode(QAQMCRenyiEngine::Mode::Work);
backend_.set_A_mask(A_start_mask_.data(), N_);
```

But `set_A_mask()` contains:

```cpp
mode_ = Mode::PairToggle;
```

So the final backend mode is `PairToggle`, not `Work`.

Impact:

The backend does not actually remain in `Mode::Work`. In many cases
`topology_toggle()` may no-op because `diff_site_ = -1`, but the engine is still
not following the intended mode contract. This also makes future changes brittle.

Recommended fix:

Do not use the generic `set_A_mask()` for work-mode state changes. Use a
dedicated work-mode reset API that sets the mask and leaves `mode_ = Mode::Work`.

If a minimal patch is desired first, reverse the order:

```cpp
backend_.set_A_mask(A_start_mask_.data(), N_);
backend_.set_mode(QAQMCRenyiEngine::Mode::Work);
```

However, this alone does not fix the missing reproject issue in finding 1.

### 3. High: new integration tests are not collected by pytest

Location:

- `tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py`

Current behavior:

The test functions are named:

```python
def _test_empty_to_A_matches_ed():
def _test_nested_pair_matches_ed():
def _test_ladder_sum_matches_end_to_end():
```

pytest does not collect functions beginning with `_test_`.

Impact:

The new integration test file provides no CI protection. A full pytest run can
pass without exercising the new Renyi work engine at all.

Recommended fix:

Rename the functions to:

```python
def test_empty_to_A_matches_ed():
def test_nested_pair_matches_ed():
def test_ladder_sum_matches_end_to_end():
```

Then run:

```bash
/home/tohenry20109/miniconda3/envs/qaqmc/bin/python \
  -m pytest tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py -q
```

### 4. Medium: direct execution of the integration script has wrong sys.path

Location:

- `tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py`

Current behavior:

The file contains:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
```

For this path:

```text
tests/engines/integration/test_qaqmc_renyi_work_vs_ed.py
```

three `dirname` calls resolve to `tests/`, not the repository root. Direct script
execution therefore fails with:

```text
ModuleNotFoundError: No module named 'src'
```

Recommended fix:

Use four `dirname` calls, or preferably avoid manual `sys.path` surgery and run
through pytest from the repository root. If script-style execution is desired:

```python
repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, repo_root)
```

### 5. Medium: run_trajectories accepts negative inputs silently

Location:

- `csrc/qaqmc_renyi_work_core.cpp`
- `QAQMCRenyiWorkEngine::run_trajectories(int n_trajectories, int decorrelation_steps)`

Current behavior:

Negative `n_trajectories` leads to a negative `trajectory_count` in the returned
result. Negative `decorrelation_steps` silently behaves like zero decorrelation
steps because the loop condition never runs.

Impact:

This is an API boundary issue. It can hide caller bugs and produce misleading
diagnostics.

Recommended fix:

Reject invalid inputs:

```cpp
if (n_trajectories < 0) {
    throw std::runtime_error("n_trajectories must be non-negative");
}
if (decorrelation_steps < 0) {
    throw std::runtime_error("decorrelation_steps must be non-negative");
}
```

## Suggested Fix Order

1. Add a work-mode backend mask reset/reproject API.
2. Replace work engine calls to `set_A_mask()` with that API.
3. Ensure backend mode is `Mode::Work` after constructor, `set_region_pair()`,
   `thermalize()`, and `reset_to_start_sector()`.
4. Rename `_test_*` functions to `test_*`.
5. Fix the test file's direct-execution import path, or remove script-style
   `main()` if pytest-only execution is intended.
6. Add negative-input tests for `run_trajectories()`.

## Merge Recommendation

Do not merge yet. The implementation compiles, but the work-mode state reset is
not robust and the tests are currently not active under pytest.
