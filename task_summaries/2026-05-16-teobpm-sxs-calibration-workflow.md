# 2026-05-16 TEOBPM SXS Calibration Workflow

## Goal
- Build an automatic workflow to calibrate the TEOBPM model against non-eccentric, non-precessing SXS simulations.
- Support two calibration families:
  - nonspinning;
  - aligned-spin, always bootstrapped from the nonspinning workflow and then extended hierarchically.
- The end goal is a stand-alone post-merger TEOB model in pyRing that can run
  parameter estimation on real detector data with no NR inputs.
- Local SXS fits may read NR-derived quantities while constructing calibration
  labels, but those quantities are not allowed to remain runtime inputs for the
  final pyRing model. Every NR input used by local fits must ultimately be
  reproduced internally by pyRing global fits once those global fits have been
  constructed.

## Repositories
- bayRing branch: `codex/teobpm-sxs-calibration-workflow`.
- pyRing temporary worktree: `/private/tmp/pyring_teobpm_sxs_calibration` was removed after the pyRing commit.
- pyRing branch: `codex/teobpm-sxs-calibration-workflow`, based on `origin/generalise_NR_informed_models`.
- pyRing implementation commit: `4be098605908629758edbb980d93097f027b0fa4`.
- pyRing Appendix A ingestion commit: `2a482cb484724df51aa2b16f3ca57a00bd7ce59c`, pushed to `origin/codex/teobpm-sxs-calibration-workflow`.
- Complete local campaign artifacts are mirrored inside this checkout under `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/`. This directory is excluded through `.git/info/exclude` and must not be tracked.

## Work Packages

### WP1 Dataset Discovery And Splitting
- [x] Define a machine-readable calibration configuration.
- [x] Query available SXS metadata for the chosen subset.
- [x] Filter to non-eccentric, non-precessing simulations.
- [x] Split simulations into train and validation sets.
- [x] Use metadata quality to reject or down-rank simulations.
- [x] Persist manifests for reproducibility.
- [ ] Review the ringdown-quality score plots before final dataset construction:
  choose the low-quality simulation cut, likely by excluding the poor-score tail
  visible in the plots above, then decide the held-out testing fraction versus
  the training fraction.

### WP2 Local Fits
- [x] Generate per-simulation bayRing/pyRing fit inputs.
- [x] Index local TEOBPM fit jobs for training and validation splits.
- [x] Add split-aware local-fit execution with per-job stdout/stderr logs.
- [x] Store local-fit parameters with uncertainty and quality diagnostics.
- [x] Track missing point estimates and failed collection without stopping the entire campaign.

### WP3 Global Fits
- [x] Fit global functions with the smallest adequate basis.
- [x] Add model-selection criteria that penalize unnecessary coefficients.
- [x] Fit merger quantities and relative mode phases.
- [x] Package global coefficients into one pyRing-ingestable file.
- [ ] Audit all NR-derived local-fit inputs and promote each required
  production quantity to an internal pyRing global fit.

### WP4 Validation
- [x] Evaluate global fits on the validation set.
- [x] Ingest available mismatch diagnostics from local bayRing outputs.
- [x] Produce publication-quality parameter-space scatter plots.
- [x] Produce parameter-space mismatch plots and mismatch histograms when mismatch values are supplied.
- [x] Add standard mismatch-comparison plots for fixed-coefficient campaigns:
  - spinning rows are shown over `nu`-`chi_eff`;
  - nonspinning rows are shown against `nu`.
- [ ] Run production global-fit waveform mismatches against held-out SXS simulations.

### WP5 Reporting
- [x] Generate a LaTeX report from the campaign artifacts.
- [x] Compile the report locally.
- [x] Show or reference the compiled PDF.

### WP6 pyRing Integration
- [x] Implement TEOBPM mode-mixing support following the SEOBNRv5-style spherical/spheroidal mixing prescription.
- [x] Add single-file global-fit ingestion.
- [x] Add direct ingestion of Appendix A calibration files, including nonspinning `nu` fits and equal-mass aligned-spin `S_hat` fits.
- [x] Add tests covering ingestion.
- [ ] Add full mixed-mode waveform tests after Cython is available in the pyRing build environment.
- [ ] Make the production TEOBPM path stand-alone for real-data parameter
  estimation: coefficients, merger quantities, mode phases, time shifts, and
  any ansatz-specific peak data must come from pyRing internals or a packaged
  global-fit payload, not from NR files.

### WP7 Stand-Alone Real-Data Readiness
- [ ] Inventory every `[NR-data] properties-file`, `fits-file`, local-fit
  column, and parent-mode value read by TEOBPM calibration runs.
- [ ] Classify each item as one of:
  - calibration-only diagnostic;
  - local-fit label that needs a global fit;
  - runtime parameter to be sampled in real-data PE.
- [ ] Construct and validate global fits for all local-fit labels needed at
  runtime, including RatExp peak quantities and higher-mode phase/time
  quantities where applicable.
- [ ] Add a pyRing real-data smoke test that evaluates TEOBPM with no NR
  waveform path, no SXS properties file, and no per-simulation local-fit
  products.
- [ ] Document which remaining options are calibration-only and should not
  appear in production real-data PE configs.

### WP8 Appendix A Reproduction
- [x] Download the arXiv:2604.15431 source and identify Appendix A plot/data definitions.
- [x] Clone public `nc_ringdown` metadata and recover the exact Appendix A SXS ID lists.
- [x] Add `appendix-a-prepare`, `appendix-a-collect`, `appendix-a-global-fit`, and `appendix-a-plot` commands.
- [x] Run real SXS downloads/local fits for the 56-job Appendix A campaign under `/private/tmp/teobpm_appendix_a_campaign`.
- [x] Construct Appendix A global polynomial fits from completed jobs.
- [x] Generate coefficient/residual panels and parameter-space/histogram mismatch diagnostics.
- [x] Overlay published HypTan "Past Fits" coefficient curves from the reproduction notebooks.
- [x] Compare the Appendix A nonspinning new HypTan/RatExp global fits against current TEOB HypTan fits and Rao/Carullo RatExp references.
- [ ] Retry failed long-running SXS IDs and rebuild the final Appendix A fits from the complete 56-job set.
- [ ] Run fixed-coefficient global/past waveform mismatch campaigns to reproduce the full local/global/past mismatch comparison panels.

## Decisions
- Nonspinning fits are the base layer.
- Aligned-spin fits must start by loading the nonspinning coefficients, then add spin-dependent correction terms only where the validation diagnostics require them.
- The first implementation should be a robust orchestrated workflow with dry-run support, because full SXS downloads and production local fits are expensive.
- The calibration workflow is allowed to use NR data to generate training
  labels. The production pyRing TEOBPM workflow is not: any NR quantity read
  during local fits is a debt item until it is supplied by an internal global
  fit or is proven to be calibration-only.
- TEOBPM local-fit construction follows pyRing's current time convention:
  `tref` is the `(2,2)` peak, `22` starts at `t-start = 0`, and higher modes
  start at `t_peak_22 + DeltaT_lm`.
- Waveform mismatch evaluation for local-fit plots and global-fit checks is a
  separate step after coefficient construction and must start at the `(2,2)`
  peak (`t-start = 0`, `tref = peak22`), even when higher modes are empty before
  their internal `DeltaT_lm`.
- Local-fit execution should use parallel job batches that respect mode-mixing
  dependencies: `22` first, then `21,33,32,31,44,42,41,55`, then `43` after a
  second `fill-hm-inputs` pass has injected the completed `33` parent values.
- Appendix A uses the public `nc_ringdown/src/data/SXS_Parameters.csv` metadata. Equal-mass/aligned-spin fits use `S_hat`; nonspinning fits use `nu`.
- Appendix A polynomial degrees follow the reproduction notebooks: nonspinning targets are linear, equal-mass RatExp targets are cubic, equal-mass HypTan targets are cubic except `c3p`, which is quartic.
- The nonspinning comparison command treats current TEOB HypTan as the hard-coded nonspinning formulas. For RatExp it defaults to the current Rao/Carullo `nu_emrg_bmrg` order-3 fit file used by `config_TEOBPM_RatExp_global.ini`, with `emrg = Heff_til` and `bmrg = b_massless_EOB`, matching bayRing metadata plumbing. The one-dimensional Appendix notebook Rao/SXS `nu` fit is still available with `--rao-reference appendix-nu`.

## Verification Log
- [x] Run bayRing focused tests: `python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py tests/test_inference.py -q` (`38 passed, 6 skipped`).
- [x] Run bayRing source-structure tests: `python -m pytest tests/test_teobpm_calibration.py tests/test_source_structure.py -q` (`396 passed, 4 warnings`).
- [x] Run Appendix A workflow unit tests: `python -m pytest tests/test_teobpm_calibration.py -q` (`13 passed`).
- [x] Run pyRing focused tests: `python -m pytest pyRing/tests/test_teobpm_calibration.py -q`.
- [x] Regenerate bayRing documentation: `make html` in `docs/`.
- [x] Regenerate pyRing Sphinx HTML directly in the temporary worktree.
- [x] Compile the calibration report: `reports/teobpm_sxs_calibration_report.pdf`.
- [x] Regenerate the report preview: `reports/teobpm_sxs_calibration_report.pdf.png`.
- [x] Run `git diff --check` for bayRing and pyRing.
- [x] Smoke-run SXS:0305 HypTan and RatExp local fits. RatExp required `c2A_22-min = 0` to avoid the negative-branch complex-to-double failure in the current TEOBPM Cython path.
- [x] Run the Appendix A 56-job campaign:
  - completed: 38;
  - failed after manual termination of long-running child processes: 18;
  - collected rows: 152;
  - collected mismatch rows: 222.
- [x] Rebuild Appendix A plots after fixing the mismatch collector to ignore `Optimal_SNR_*.txt` values.
- [x] Generate Appendix A nonspinning comparison artifacts:
  - current Rao/Carullo reference: `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/current_rao`;
  - one-dimensional Appendix `nu` Rao reference: `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_nu_rao`.
- [x] Re-run Appendix A collection/global-fit/comparison on 2026-05-16 from the completed local fits:
  - jobs collected: 56 total, 38 completed, 18 missing point-estimate files;
  - coefficient rows: 152;
  - mismatch rows: 222;
  - rebuilt global-fit file: `/private/tmp/teobpm_appendix_a_campaign/appendix_a_global_fits.json`;
  - regenerated current TEOB/Rao comparison plots under `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/current_rao`;
  - regenerated Appendix `nu` Rao comparison plots under `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_nu_rao`.
- [x] Add and smoke-run the standard mismatch comparison plotter:
  - command: `python -m bayRing.teobpm_calibration plot-mismatch-comparison`;
  - local-mismatch sanity plots: `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/local_mismatch_parameter_space`;
  - output files include `teobpm_mismatch_comparison_nonspinning.png`, `teobpm_mismatch_comparison_spinning.png`, point CSV, summary CSV, and summary JSON.
- [x] Run updated bayRing focused/source-structure tests: `python -m pytest tests/test_teobpm_calibration.py tests/test_source_structure.py -q` (`411 passed, 4 warnings`).
- [x] Regenerate bayRing documentation after adding the mismatch-comparison command: `make html` in `docs/`.
- [x] Confirm pyRing TEOBPM example-config convention (`fix-t = 0` at the
  `(2,2)` peak) and update bayRing campaign prep/diagnostics accordingly.
- [x] Reprepare current SXS refit campaign definitions for `HypTan` and
  `RatExp` across nonspinning, equal-mass spinning, and generic aligned-spin
  families under `local_artifacts/teobpm_current_spinning_refit/`.
- [x] Re-run focused TEOBPM/template tests after the time-reference update:
  `python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py -q`
  (`33 passed`).
- [x] Regenerate bayRing documentation after the time-reference update:
  `make html` in `docs/`.
- [x] Export Appendix A nonspinning fits to the pyRing-ingestable schema `bayRing.teobpm.global-fit.v1`:
  - `/private/tmp/teobpm_appendix_a_campaign/pyRing_calibration/teobpm_appendix_a_nonspinning_hyptan_global_fit.json`;
  - `/private/tmp/teobpm_appendix_a_campaign/pyRing_calibration/teobpm_appendix_a_nonspinning_ratexp_global_fit.json`.
- [x] Push pyRing branch update for Appendix A ingestion:
  - commit: `2a482cb484724df51aa2b16f3ca57a00bd7ce59c`;
  - pushed branch: `origin/codex/teobpm-sxs-calibration-workflow`;
  - pyRing now accepts `bayRing.teobpm.appendix-a.v1` files directly through `teob-calibration-file`, selecting `nu` for nonspinning binaries and `S_hat` for the equal-mass aligned-spin sequence.
- [x] Mirror portable Appendix A campaign outputs into the bayRing checkout without tracking them:
  - local artifact root: `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign`;
  - copied metadata/configs/tables: `campaign_config.json`, `appendix_a_manifest.csv`, `local_fit_index.csv`, `run_local_fits.sh`, run/collection summaries, `local_fit_summary.csv`, `appendix_a_local_fit_summary.csv`, `local_fit_collection_failures.csv`, and `mismatch_summary.csv`;
  - copied fit files: `appendix_a_global_fits.json` and `pyRing_calibration/*.json`;
  - copied plots: `appendix_a_plots/`, including current-Rao, Appendix-`nu` Rao, and local-mismatch parameter-space diagnostics;
  - copied rerun configs: `local_fit_configs/`;
  - copied raw per-job outputs: `local_fits/`, about 4.7 GB.

## Residual Risks
- The current calibration artifacts still include NR-derived local-fit inputs.
  They are valid as calibration labels, but the TEOBPM model is not fully
  stand-alone for real-data pyRing PE until all required labels are replaced by
  pyRing global fits or removed from the runtime path.
- The Appendix A run is a partial real campaign, not the final complete reproduction. Failed IDs are listed in `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/local_fit_collection_failures.csv` and in the original scratch copy at `/private/tmp/teobpm_appendix_a_campaign/local_fit_collection_failures.csv`.
- The complete local Appendix A handoff is now available at `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/`, including the raw 4.7 GB `local_fits/` tree. The original `/private/tmp/teobpm_appendix_a_campaign/` copy was left in place as a scratch/cache copy.
- The validation plots currently use collected coefficient residuals and any mismatch values emitted by bayRing local runs; full held-out waveform mismatch campaigns still need to be executed.
- Appendix A coefficient plots compare against published HypTan past-fit curves, but the full mismatch comparison against old TEOB/global/past waveforms still needs fixed-coefficient mismatch jobs.
- The new mismatch-comparison plotter is ready for fixed-coefficient new-global/current-TEOBPM mismatch tables, but those tables are not present in the current handoff. The plots generated in `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/appendix_a_plots/local_mismatch_parameter_space` use the collected local-fit mismatch column only and should not be interpreted as a new-global-vs-current-TEOBPM waveform comparison.
- The current Rao/Carullo `nu_emrg_bmrg` order-3 reference produces very large RatExp coefficient outliers on this partial Appendix A SXS nonspinning subset. The command clips those markers in the plot for readability and writes exact values to the point-level CSV. This should be checked against the intended current pyRing/bayRing normalization semantics before making a science claim.
- pyRing `make html` could not complete its Makefile build-ext pre-step because Cython is missing in the active environment; direct Sphinx HTML generation completed with pre-existing autodoc import warnings tied to packaged data/LFS files.
- The pyRing TEOBPM Cython extension was not rebuilt in this environment for the same missing-Cython reason.

## 2026-05-16 Late Iteration Wrap-Down

- Report written: `reports/teobpm_appendix_a_global_fit_wrapdown_2026-05-16.md`.
- bayRing now separates `TEOB-template = HypTan|RatExp` from `TEOB-calibration = qc|noncirc`.
- pyRing worktree `/private/tmp/pyring_teobpm_mode_mixing` now separates `teob-template = HypTan|RatExp` from `teob-calibration = qc|noncirc`; it is a detached worktree with uncommitted edits and is used through `PYTHONPATH` for local bayRing runs.
- Appendix A fixed-coefficient mismatch configs under `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1` are qc-only and use `method = Minimization`.
- Eccentric/noncircular variables are excluded from the noneccentric/qc Appendix workflow: `ecc`, `e0`, `eccentricity`, `bmrg`, `Emrg`, and `Jmrg`.
- Standard mismatch plots were corrected and regenerated so nonspinning rows use `nu` and spinning rows use `nu`-`chi_eff`.
- Current fixed-coefficient mismatch campaign status:
  - 76 jobs total;
  - 59 point-estimate outputs present;
  - 30 wide new-vs-existing rows collected;
  - 59 mismatch values collected;
  - 17 equal-mass spinning jobs still missing.
- Valid current mismatch plots:
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_nonspinning.png`;
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning.png`;
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning_chi_eff.png`.
- Remaining SXS blocker: incomplete cached equal-mass spinning SXS waveform levels resolve through the read-only user cache. A writable mirror exists at `/private/tmp/teobpm_sxs_cache_rw`, and configs now have `download = 1`, but the sandboxed direct download failed on DNS and the escalated direct download attempt was rejected by the environment.
