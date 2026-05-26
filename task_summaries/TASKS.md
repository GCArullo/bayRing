# Task Notes

## 2026-05-04 Start-Time Range Runs

### Goal
- Allow one bayRing invocation to repeat the same fit over a user-defined range/list of `t-start` values.
- Store each fit in a dedicated results subtree so plots, sampler outputs, posterior files, mismatch files, and copied config are not overwritten.

### Tasks
- [x] Trace the existing single-start inference, post-processing, and output-directory lifecycle.
- [x] Add parsing/validation for scalar and ranged/list start-time input.
- [x] Add per-start output directories while preserving the existing scalar `t-start` behavior.
- [x] Ensure post-processing and plot paths point at the active start-time directory.
- [x] Add focused tests or syntax checks that exercise parsing and routing.
- [x] Update user-facing docs/help text.

### Verification
- [x] Run targeted tests covering config parsing and source structure.
- [x] Run the full test suite.
- [x] Run syntax checks for edited modules.
- [x] Run `git diff --check`.

### Outcome
- `t-start` now accepts scalar values, comma/list values, and inclusive `start:stop:step` ranges.
- Scalar runs keep the original `outdir` layout.
- Multi-start runs write each active fit to `outdir/t_start_<value>M/`, with the usual `Algorithm`, `Peak_quantities`, and `Plots` tree under each start-time directory.

### Follow-up: shared scan files
- [x] Move start-time-independent files such as `git_info.txt` and the copied config to the base `outdir` for multi-start scans.
- [x] Keep start-time-dependent products under each `t_start_<value>M` subdirectory.
- [x] Add tests covering shared-file placement.

### Follow-up Verification
- [x] Run targeted start-time and source-structure tests.
- [x] Run syntax checks for edited modules.
- [x] Run `git diff --check`.
- [x] Run the full test suite.

### Follow-up Outcome
- Multi-start scans now call shared output setup once at the base `outdir`.
- Per-start subdirectories create only result directories and start-time-dependent outputs.
- Scalar runs keep the previous behavior where shared metadata and result files live in the same `outdir`.

## 2026-05-03 Merge Conflict Resolution

### Goal
- Merge `origin/main` into `codex/add-kerr-model-injection-support` and resolve conflicts.

### Conflicted files
- [x] `bayRing/bayRing.py`
- [x] `bayRing/postprocess.py`

### Verification
- [x] Confirm no conflict markers remain.
- [x] Run syntax parsing for `bayRing/bayRing.py` and `bayRing/postprocess.py`.
- [x] Run `git diff --check`.
- [x] Run `python -m pytest tests/test_source_structure.py tests/test_inference.py -q`.

### Outcome
- Kept the `origin/main` mismatch/SNR postprocessing flow.
- Preserved injected truth overlays for nested-sampler corner plots.
