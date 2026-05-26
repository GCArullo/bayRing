# TEOBPM SXS Calibration Workflow Handoff

## Current State
- bayRing branch: `codex/teobpm-sxs-calibration-workflow`.
- pyRing branch: `codex/teobpm-sxs-calibration-workflow`, based on `origin/generalise_NR_informed_models`.
- pyRing commit containing the mode-mixing/global-fit ingestion work: `4be098605908629758edbb980d93097f027b0fa4`.
- pyRing commit adding direct Appendix A `bayRing.teobpm.appendix-a.v1` ingestion, including equal-mass aligned-spin `S_hat` fits: `2a482cb484724df51aa2b16f3ca57a00bd7ce59c`.
- The pyRing branch has been pushed to `origin/codex/teobpm-sxs-calibration-workflow`.
- The temporary pyRing worktrees used for this task have been removed, so the branch is not locked by a private worktree.
- Pre-existing untracked files in bayRing before this work included `TASKS.md` and `config_files/plot_ERC.py`; they were left untouched.
- Complete Appendix A campaign artifacts in the bayRing checkout: `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/`.
- That local artifact directory is excluded through `.git/info/exclude`; do not stage or track it.
- Appendix A plot directory: `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/appendix_a_plots`.
- Appendix A fit file, now readable directly by the updated pyRing branch: `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/appendix_a_global_fits.json`.
- pyRing-format nonspinning exports remain under `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/pyRing_calibration/`.
- To use the combined Appendix A file in pyRing, set `teob-calibration-file = /Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/appendix_a_global_fits.json` and set `teob-template = HypTan` or `teob-template = RatExp`; the updated pyRing branch chooses the nonspinning or equal-mass aligned-spin subfit from the sample masses/spins.
- Raw per-job Appendix A local-fit outputs are included at `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/local_fits/` and are about 4.7 GB. The original `/private/tmp/teobpm_appendix_a_campaign/` copy was left in place as scratch/cache provenance.
- Public `nc_ringdown` clone used for Appendix A metadata: `/private/tmp/nc_ringdown`.

## Implemented In bayRing
- Added `bayRing/teobpm_calibration.py`.
- Added CLI entry point in `pyproject.toml`:
  - `bayRing-teobpm-calibrate = "bayRing.teobpm_calibration:main"`
- Added campaign commands:
  - `prepare`: loads SXS metadata, filters non-eccentric/non-precessing simulations, requires a nonspinning base fit for aligned-spin campaigns, writes manifests, per-mode configs, `local_fit_index.csv`, and `run_local_fits.sh`.
  - `run-local-fits`: executes indexed jobs, supports `--split all|training|validation`, `--workers`, `--force`, `--timeout`, and custom `--bayring-executable`.
  - `collect-local-fits`: parses `Algorithm/point_estimates.dat`, `Algorithm/Mismatch/*.txt`, metadata peak quantities, and relative phases using `phi_mrg_22` as reference.
  - `global-fit`: constructs schema `bayRing.teobpm.global-fit.v1`; aligned-spin fits load the nonspinning fit and add spin-correction terms.
  - `validate`: evaluates validation rows, filters to `split=validation` when present, writes prediction residuals, residual plots, mismatch histograms, and parameter-space mismatch plots when mismatch values exist.
  - `report`: renders a LaTeX report from campaign summaries and includes validation figures when available.
  - `appendix-a-prepare`: prepares the quasi-circular Appendix A SXS ID subsets from `nc_ringdown` metadata.
  - `appendix-a-collect`: collects Appendix A local point estimates and merges the Appendix A manifest metadata.
  - `appendix-a-global-fit`: constructs Appendix A polynomial fits with the degrees used in the reproduction notebooks.
  - `appendix-a-plot`: produces one coefficient/residual panel per Appendix A coefficient, mismatch-vs-parameter plots, and mismatch histograms.
  - `appendix-a-compare-nonspinning`: compares the nonspinning Appendix A new HypTan/RatExp global fits against current TEOB HypTan and Rao/Carullo RatExp references.
  - `plot-mismatch-comparison`: standard plotting command for fixed-coefficient mismatch tables. It accepts repeated `--mismatch-table`/`--label` inputs or wide tables with columns such as `new_global_mismatch` and `existing_teobpm_mismatch`; spinning rows are plotted over `nu`-`chi_eff`, and nonspinning rows are plotted against `nu`.
- Added documentation:
  - `docs/teobpm_calibration.rst`
  - `docs/index.rst` toctree entry
- Added tests:
  - `tests/test_teobpm_calibration.py`
- Added report artifacts:
  - `reports/teobpm_sxs_calibration_report.tex`
  - `reports/teobpm_sxs_calibration_report.pdf`
  - `reports/teobpm_sxs_calibration_report.pdf.png`

## Implemented In pyRing
- Added a pure Python calibration loader for the new global-fit schema.
- Added opt-in TEOBPM mode mixing following the spherical/spheroidal mixing prescription needed for the Cheung et al. formalism.
- Added pyRing configuration plumbing for a single TEOB calibration file.
- Added direct Appendix A calibration ingestion:
  - accepts schema `bayRing.teobpm.appendix-a.v1`;
  - evaluates nonspinning fits with `nu`;
  - evaluates equal-mass aligned-spin fits with `S_hat`;
  - requires `teob-template` to choose `HypTan` or `RatExp` when the file contains both.
- Added pyRing tests for ingestion.
- Committed pyRing work as `4be098605908629758edbb980d93097f027b0fa4` and `2a482cb484724df51aa2b16f3ca57a00bd7ce59c`.

## Verification Already Run
- `python -m pytest tests/test_teobpm_calibration.py -q`
  - `13 passed`
- `python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py tests/test_inference.py -q`
  - `38 passed, 6 skipped`
- `python -m pytest tests/test_teobpm_calibration.py tests/test_source_structure.py -q`
  - `396 passed, 4 warnings`
- `python -m compileall bayRing/teobpm_calibration.py`
- `python -m bayRing.teobpm_calibration --help`
- `python -m bayRing.teobpm_calibration run-local-fits --help`
- `python -m bayRing.teobpm_calibration collect-local-fits --help`
- `make html` in `docs/`
- `python -m bayRing.teobpm_calibration plot-mismatch-comparison --help`
- `python -m bayRing.teobpm_calibration plot-mismatch-comparison --mismatch-table /private/tmp/teobpm_appendix_a_campaign/appendix_a_local_fit_summary.csv --label "Appendix A local fits" --output-dir /private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/local_mismatch_parameter_space`
- `pdflatex -interaction=nonstopmode teobpm_sxs_calibration_report.tex` in `reports/`
- `qlmanage -t -s 1200 -o reports reports/teobpm_sxs_calibration_report.pdf`
- `git diff --check`
- pyRing focused tests passed in the temporary worktree before it was removed.
- In `/private/tmp/pyring_teobpm_spin_format`, before removing the worktree:
  - `python -m pytest pyRing/tests/test_teobpm_calibration.py -q` (`7 passed`);
  - `python -m compileall pyRing/teobpm_calibration.py pyRing/pyRing.py pyRing/inject_signal.py`;
  - loaded `/private/tmp/teobpm_appendix_a_campaign/appendix_a_global_fits.json` and evaluated both equal-mass spinning `HypTan` and nonspinning `RatExp` coefficients through `teobpm_calibration.evaluate_calibration_payload`;
  - `git diff --check`.

## Appendix A Campaign Results
- Prepared 56 jobs:
  - equal-mass/aligned-spin HypTan: 12;
  - equal-mass/aligned-spin RatExp: 12;
  - nonspinning HypTan: 16;
  - nonspinning RatExp: 16.
- Real SXS downloads were performed through the `sxs` package and cached under the user SXS cache.
- Completed jobs: 38.
- Failed jobs: 18. The failures are jobs manually terminated after long stalls with no new point-estimate/mismatch files:
  - equal-mass: `SXS:BBH:1477`, `SXS:BBH:0156` for both HypTan and RatExp;
  - nonspinning HypTan: `SXS:BBH:0007`, `0169`, `0030`, `0167`, `0295`, `0063`, `0300`;
  - nonspinning RatExp: `SXS:BBH:0007`, `0169`, `0030`, `0167`, `0295`, `0063`, `0300`.
- Collection after the parser fix:
  - coefficient rows: 152;
  - mismatch rows: 222;
  - mismatch range in collected local fits: about `6.4e-7` to `3.1e-5`.
- The mismatch parser was corrected so `Optimal_SNR_*.txt` files no longer contaminate the representative mismatch.
- Generated Appendix A figures:
  - `c3A_qc_fit_sxs_chi.png`, `c3p_qc_fit_sxs_chi.png`, `c4p_qc_fit_sxs_chi.png`;
  - `c2A_fit_sxs_chi.png`, `c3A_fit_sxs_chi.png`, `c2p_fit_sxs_chi.png`, `c3p_fit_sxs_chi.png`, `c4p_fit_sxs_chi.png`;
  - corresponding `_nu.png` panels;
  - `appendix_a_mismatch_equal_mass.png`, `appendix_a_mismatch_nonspinning.png`;
  - `appendix_a_mismatch_histogram_equal_mass.png`, `appendix_a_mismatch_histogram_nonspinning.png`.
- The HypTan coefficient panels overlay the published "Past Fits" curves from the Appendix A notebooks. The RatExp panels do not have a published past-fit overlay in those notebooks.
- Generated nonspinning comparison outputs:
  - `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/current_rao/` compares RatExp against the current Rao/Carullo `src/data/bayesian_fit_coefficients/order_fits_nu_emrg_bmrg_3.csv` file used by bayRing's current global RatExp example.
  - `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_nu_rao/` compares RatExp against the one-dimensional Appendix notebook SXS `nu` reference `src/data/fits/nc_fits_sxs_non-spinning/order_fits_nu_1.csv`.
- Both folders contain coefficient panels, point-level prediction/residual CSVs, grid/curve CSVs, and residual-summary CSV/PNG.
- Current Rao/Carullo `nu_emrg_bmrg` predictions show large outliers on the partial SXS nonspinning subset; the plot clips those markers for readability and the exact values are in the point CSV.
- Generated standard local-mismatch parameter-space sanity plots under `/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/local_mismatch_parameter_space/`:
  - `teobpm_mismatch_comparison_nonspinning.png`;
  - `teobpm_mismatch_comparison_spinning.png`;
  - `teobpm_mismatch_comparison_points.csv`;
  - `teobpm_mismatch_comparison_summary.csv`;
  - `teobpm_mismatch_comparison_summary.json`.
  These use collected local-fit mismatches because fixed-coefficient new-global/current-TEOBPM waveform mismatch tables have not yet been produced.
- Complete local copies of the campaign outputs listed above are in `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/`.
- The copied campaign artifacts include:
  - metadata/config/provenance: `campaign_config.json`, `appendix_a_manifest.csv`, `local_fit_index.csv`, `run_local_fits.sh`, and `local_fit_configs/`;
  - run and collection tables: `local_fit_run_summary.*`, `local_fit_collection_summary.json`, `local_fit_collection_failures.csv`, `local_fit_summary.csv`, `appendix_a_local_fit_summary.csv`, `appendix_a_collection_summary.json`, and `mismatch_summary.csv`;
  - fit files: `appendix_a_global_fits.json`, `pyRing_calibration/teobpm_appendix_a_nonspinning_hyptan_global_fit.json`, and `pyRing_calibration/teobpm_appendix_a_nonspinning_ratexp_global_fit.json`;
  - generated figures and comparison CSV/JSON files under `appendix_a_plots/`;
  - raw per-job run products under `local_fits/`.

## Remaining Work To Complete The Workflow

### Target Architecture
- The end goal is a stand-alone post-merger TEOB model in pyRing that can run
  parameter estimation on real detector data with no NR inputs.
- NR data are allowed in this bayRing workflow only as calibration data: local
  SXS fits can read NR peak quantities, phase references, parent-mode values,
  and local coefficients to build training labels.
- Anything read from NR during local fits must either be marked
  calibration-only or become obtainable inside pyRing through a global fit
  before the final real-data workflow is considered complete.
- A production pyRing TEOBPM run should not require an SXS waveform path,
  `[NR-data] properties-file`, per-simulation local-fit products, or an
  externally hand-filled table of NR merger quantities.

### 0. Immediate Wrap-Down State From 2026-05-16
- Detailed report: `reports/teobpm_appendix_a_global_fit_wrapdown_2026-05-16.md`.
- The latest fixed-coefficient Appendix A mismatch campaign is `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1`.
- The campaign is configured as qc-only, `Minimization`, and currently `download = 1`.
- A writable SXS cache mirror is available at `/private/tmp/teobpm_sxs_cache_rw`.
- 59 point-estimate outputs have been collected from 76 jobs; 17 equal-mass spinning jobs still need rerun after the SXS cache/download issue is solved.
- The current standard plots are:
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_nonspinning.png`;
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning.png`;
  - `/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning_chi_eff.png`.
- The plot summary JSON confirms: `Nonspinning mismatch plots use nu. Spinning mismatch plots use nu-chi_eff.`
- The pyRing side of the ansatz/calibration split is in `/private/tmp/pyring_teobpm_mode_mixing` and still needs to be committed or moved to the target pyRing branch.

### 1. Complete Appendix A Reproduction
- Retry the 18 failed jobs individually, preferably with `--workers 1`, longer timeout, and a profiler or live process sample to determine whether they are genuinely slow or stuck inside SXS supersession/waveform loading.
- Re-run:
  - `python -m bayRing.teobpm_calibration appendix-a-collect --campaign-dir /private/tmp/teobpm_appendix_a_campaign`
  - `python -m bayRing.teobpm_calibration appendix-a-global-fit --campaign-dir /private/tmp/teobpm_appendix_a_campaign`
  - `env MPLCONFIGDIR=/private/tmp/teobpm_appendix_a_smoke/mpl python -m bayRing.teobpm_calibration appendix-a-plot --campaign-dir /private/tmp/teobpm_appendix_a_campaign`
- After any retry campaign, mirror the updated outputs back into `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_appendix_a_campaign/` and keep that directory untracked.
- Add fixed-coefficient mismatch campaigns for:
  - local-fit waveforms already produced;
  - new Appendix A global fits from `appendix_a_global_fits.json`;
  - published TEOB/Past Fits for HypTan.
- Reproduce the paper's combined mismatch panels with local/global/past markers once those fixed-coefficient mismatch outputs exist. Feed the resulting tables into `plot-mismatch-comparison` so the standard nonspinning `nu` and spinning `nu`-`chi_eff` views are generated automatically.
- Audit the current Rao/Carullo `nu_emrg_bmrg` normalization path before using the current-Rao comparison scientifically. The comparison command follows bayRing's metadata mapping (`emrg = Heff_til`, `bmrg = b_massless_EOB`) and the pyRing-style normalized polynomial evaluation, but it yields very large RatExp outliers for some high-mass-ratio SXS Appendix points.

### 2. Production SXS Campaign Smoke Test
- Prepare a small nonspinning campaign with a real SXS catalog query outside the Appendix A fixed-ID path.
- Use 1-2 simulations and 1-2 modes first.
- Confirm generated configs download expected SXS waveform data.
- Confirm `Algorithm/point_estimates.dat` contains TEOBPM parameters with the expected names for `HypTan` and `RatExp`.
- Check whether RatExp local configs need an explicit `[NR-data] properties-file` for merger quantities in all real SXS cases.

### 3. Full Nonspinning Calibration
- Run the nonspinning SXS subset over the requested mode list.
- Inspect `local_fit_collection_failures.csv`.
- Decide whether low-quality or failed modes should be dropped globally or re-run with adjusted windows.
- Build `teobpm_global_fit.json`.
- Inspect BIC-selected bases for physical plausibility.
- Compare fitted nonspinning functions against existing quasi-circular TEOBPM fits where available.

### 4. Full Aligned-Spin Calibration
- Start only after the nonspinning global fit is final.
- Prepare the aligned-spin campaign with `--base-nonspinning-fit`.
- Keep the nonspinning terms fixed and fit only spin corrections.
- Inspect whether the minimal basis selected by BIC is stable under validation splits.
- Check whether additional terms are justified by residual structure; do not add terms only to improve training residuals.

### 5. Validation Waveform Mismatches
- The current validation step plots coefficient residuals and any mismatch values found in bayRing output.
- The remaining science validation is to run held-out SXS simulations through the global-fit TEOBPM waveform and compute actual waveform mismatches.
- Decide whether to run those mismatches from bayRing or pyRing:
  - bayRing already has local mismatch machinery, but the new single-file global-fit schema is pyRing-side.
  - pyRing now ingests both `bayRing.teobpm.global-fit.v1` and `bayRing.teobpm.appendix-a.v1` on `codex/teobpm-sxs-calibration-workflow`, but the Cython extension needs to be rebuilt before full waveform tests are reliable.
- Add an explicit validation command once the execution path is chosen, instead of overloading coefficient residual validation.

### 6. Stand-Alone pyRing Readiness
- Audit the local-fit summary schema, generated configs, and pyRing TEOBPM
  constructor arguments for every quantity currently sourced from NR files.
- For each required quantity, identify the calibration coordinates and build a
  corresponding global fit. This includes ansatz-specific peak data such as the
  RatExp peak-amplitude derivatives if they remain part of the production
  waveform.
- Add tests that evaluate global-fit TEOBPM waveforms from binary parameters
  alone, with no `properties-file` and no per-simulation local-fit table.
- Update the pyRing configuration docs so real-data examples use only sampled
  PE parameters plus packaged/internal TEOBPM global fits.

### 7. pyRing Build And Tests
- Install or activate an environment with Cython.
- Rebuild the pyRing Cython extension.
- Run the pyRing TEOBPM tests after rebuilding.
- Add a mixed-mode waveform regression test that verifies the calibration-file path and mode-mixing path together.
- Regenerate pyRing docs with the normal Makefile path after Cython is available.

### 8. Report Finalisation
- Re-run the report command against a completed campaign directory, not the static template.
- Include the actual validation figures generated by the completed campaign.
- Add a table of final global-fit mode-target pairs and selected basis terms.
- Add failure accounting: number of SXS simulations rejected by eccentricity, transverse spin, quality, missing modes, local-fit failure, and validation failure.

### 9. Pre-merge Hygiene
- Re-run the bayRing checks listed above.
- Re-run pyRing checks after Cython rebuild.
- Review untracked files carefully before staging because `TASKS.md` and `config_files/plot_ERC.py` pre-existed this task.
- Stage only the workflow, documentation, tests, task summaries, and report artifacts.
- Do not stage generated LaTeX sidecars; only keep `.tex`, `.pdf`, and `.pdf.png`.
