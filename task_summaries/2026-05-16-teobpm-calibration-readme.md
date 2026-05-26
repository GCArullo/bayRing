# 2026-05-16 TEOBPM Calibration README

## Goal
- Write one student-facing calibration README explaining the TEOBPM workflow as four rerunnable stages:
  - local fits;
  - local-fit coefficient and mismatch plots;
  - global fits;
  - global-fit coefficient and mismatch plots.
- Document selectable catalog subsets, binary type, template, local algorithm, and global polynomial degree.
- Make the documented commands executable against the current CLI.

## Tasks
- [x] Read existing workflow notes, reports, docs, tests, and CLI implementation.
- [x] Delegate CLI and workflow-default inspection to sub-agents.
- [x] Add missing workflow CLI knobs needed by the README:
  - `--sxs-ids`, `--sxs-id-file`, and `--random-fraction` for catalog selection;
  - `equal-mass-spinning` and `spinning` family choices;
  - default mixed modes `32,43`;
  - `--max-polynomial-degree` for generic global fits;
  - `plot-local-fits` for local coefficient and mismatch diagnostics;
  - all-core default `run-local-fits` worker count;
  - progress-bar status prints in `run-local-fits`.
- [x] Write `scripts/teobpm/teobpm_calibration_readme.md`.
- [x] Move the README under `scripts/teobpm/` and keep Sphinx focused on package docs.
- [x] Regenerate Sphinx HTML locally.
- [x] Run focused calibration and source-structure tests.
- [x] Move the workflow module from `bayRing/teobpm_calibration.py` to `scripts/teobpm/teobpm_calibration.py`.
- [x] Move the README from `docs/` to `scripts/teobpm/`.
- [x] Update the console-script entry point, Appendix A wrappers, and tests for the new module path.
- [x] Support nested-sampler local products by treating `Algorithm/posterior.dat` as the completion marker and collecting posterior medians when `point_estimates.dat` is absent.
- [x] Move the reconstructed nonspinning TEOBPM-paper catalog into the repo under `scripts/teobpm/catalogs/arxiv_1904_09550_nonspinning.csv`.

## Verification
- [x] `python -m pytest tests/test_teobpm_calibration.py -q`
- [x] `python -m compileall scripts/teobpm/teobpm_calibration.py scripts/teobpm/run_appendix_a_local_fits.py scripts/teobpm/run_appendix_a_global_mismatch.py`
- [x] `make html` in `docs/`
- [x] `python -m pytest tests/test_teobpm_calibration.py tests/test_source_structure.py -q`
- [x] `git diff --check`
- [x] `python -m scripts.teobpm.teobpm_calibration --help`

## Outcome
- The README gives a single campaign setup block and four clean command stages.
- Local execution skips existing simulation/mode outputs unless `--force` is passed.
- The documented default uses all available higher modes and enables mode mixing for `32` and `43`.
- Higher-level spin campaigns are guarded by lower-level fit-file existence checks in the README commands.
- The code now exposes the catalog and global-degree controls described by the README.
- The calibration workflow module now lives at `scripts/teobpm/teobpm_calibration.py`.
- The moved README now lives at `scripts/teobpm/teobpm_calibration_readme.md`.
- Local-fit skip and collection now support both minimization-style `point_estimates.dat` files and nested-sampler `posterior.dat` files.

## Notes
- The reconstructed nonspinning TEOBPM-paper catalog is now tracked at `scripts/teobpm/catalogs/arxiv_1904_09550_nonspinning.csv`; the remaining reconstructed spin catalog CSVs are still referenced from `/private/tmp/teobpm_current_spinning_refit/catalogs/`.
- The generic `spinning` global fit still uses the nonspinning base file as the actual coefficient baseline; the README also checks the equal-mass fit before starting the generic spinning stage to enforce the requested run order.
