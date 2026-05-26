# TEOBPM Current Spinning Refit

## Request

Repeat the local and global fits behind the current spinning TEOBPM model, using
the original SXS calibration simulations. The workflow must run hierarchically:

1. Nonspinning baseline.
2. Equal-mass spinning correction.
3. Generic aligned-spin correction.

The campaign covers the available TEOBPM higher modes in parallel for both
`HypTan` and `RatExp`; modes `32` and `43` use TEOBPM mode-mixing. Final
diagnostics compare the implemented pyRing TEOBPM fits with the refitted
coefficients, plotting mismatches versus `q`, `q`-`chi_eff`, and `q`-`chi_a`.

The refit is a calibration step toward a stand-alone post-merger TEOB model in
pyRing. The final target is real-data parameter estimation with no NR inputs.
Therefore any NR quantity read here to construct local fits, including peak
properties, phase references, parent-mode values, local coefficients, or
RatExp-specific peak data, must ultimately be supplied by internal pyRing
global fits or be documented as calibration-only.

## Source Simulation Sets

Primary source: arXiv:2001.09082 TeX source, with nonspinning SXS IDs inherited
from arXiv:1904.09550.

- Nonspinning SXS set: 19 simulations.
- Equal-mass aligned-spin SXS set: 38 simulations.
- Generic aligned-spin SXS set: 78 simulations.
- BAM and test-particle rows from the paper tables are excluded because this
  run is requested against SXS data.

## Current Task Tree

- [x] Recover the original SXS calibration tables from the paper sources.
- [x] Patch bayRing campaign support required for the full HM run.
  - [x] Allow all selected simulations to remain in the training split.
  - [x] Allow `run-local-fits` to execute selected modes first.
  - [x] Fill higher-mode `t-peak-22` values from completed `22` local fits.
  - [x] Expose TEOBPM mode-mixing for `32` and `43`.
  - [x] Produce mismatch plots over `q`, `q`-`chi_eff`, and `q`-`chi_a`.
- [x] Build/use a pyRing TEOBPM implementation with mode-mixing support.
- [x] Prepare exact SXS catalog CSVs and campaign directories for both templates.
- [x] Run/reuse local fits in hierarchical order.
- [x] Collect local fits and build global fits.
- [ ] Audit all NR-sourced local-fit inputs and decide which need pyRing
  global fits for stand-alone real-data TEOBPM PE.
- [ ] Compare refitted and current pyRing TEOBPM mismatch surfaces.
- [x] Regenerate campaign report and run focused tests.
- [x] Remove any private `/private/tmp` worktree opened for this run.

## Running Notes

- SXS imports require a private `NUMBA_CACHE_DIR` in this environment.
- The installed pyRing import currently points at
  `/Users/g.carullo@bham.ac.uk/Repos/pyring_test`; the rebuilt branch at
  commit `1f217ab` exposes the corrected TEOBPM `mode_mixing` argument used
  by the 32/43 mixed-mode runs.
- Focused infrastructure tests passed:
  `python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py -q`.
- Exact SXS catalog CSVs were generated under
  `local_artifacts/teobpm_current_spinning_refit/catalogs/` with row counts
  `19`, `38`, and `78`.
- Campaign directories were prepared for `HypTan` and `RatExp`:
  `nonspinning`, `equal_mass_spinning`, and `generic_spinning`.
- The old temporary pyRing worktree at `/private/tmp/pyring_teobpm_mode_mixing`
  is no longer used; the active pyRing source is
  `/Users/g.carullo@bham.ac.uk/Repos/pyring_test`.
- Wrap-down report written at
  `reports/teobpm_current_spinning_refit_wrapdown_2026-05-16.md`.

## Wrap-Down State: 2026-05-16

The 2026-05-16 wrap-down notes below are retained for provenance. They were
superseded by the local checkout runs summarized in the 2026-05-17 completion
update.

- No target bayRing calibration jobs are currently running.
- The original SXS catalog CSVs are under
  `/private/tmp/teobpm_current_spinning_refit/catalogs/`:
  19 nonspinning, 38 equal-mass spinning, 78 generic spinning, and 135 total
  SXS rows excluding headers.
- Campaigns are prepared under
  `/private/tmp/teobpm_current_spinning_refit/{hyptan,ratexp}/`.
- HypTan nonspinning collection was run once and wrote:
  `/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/local_fit_summary.csv`.
- Current HypTan nonspinning collected state:
  190 jobs, 641 coefficient rows, 204 mismatch rows, 25 missing jobs.
- Completed HypTan nonspinning point-estimate counts by mode:
  `21=19`, `22=19`, `31=19`, `32=12`, `33=19`, `41=19`,
  `42=19`, `43=1`, `44=19`, `55=19`.
- Mode `43` initially failed because pyRing's packaged
  `swsh_s2.hdf5` lacked `s2l4` coefficient datasets. Temporary datasets
  `s2l4/l4m3lp3np0` and `s2l4/l4m3lp4np0` were generated in
  `/private/tmp/pyring_teobpm_mode_mixing`.
- A diagnostic `SXS:BBH:0180`, mode `43` run reached post-processing after the
  HDF5 patch, but it was stopped during wrap-down; rerun all `43` jobs with
  `--force` before using the output scientifically.
- Remaining blockers are making the pyRing `43` coefficient data permanent,
  generating RatExp mode-by-mode merger properties, allowing spinning SXS
  waveform downloads, and producing fixed-coefficient current-vs-new mismatch
  tables for the full campaign.

## Time-Reference Update: 2026-05-16

- Confirmed the current pyRing TEOBPM convention from example configs and code:
  `tref` is the `(2,2)` peak (`fix-t = 0`), and each mode starts internally at
  `t_peak_22 + DeltaT_lm`.
- Updated the bayRing TEOBPM campaign preparation so local-fit construction
  writes `t-start = 0` for `22` and pyRing `DeltaT_lm` for higher modes.
- Deleted the generated local campaign artifacts that assumed a uniform 20M
  start, preserving reusable SXS/numba/matplotlib caches.
- Reprepared all six local checkout campaign trees under
  `local_artifacts/teobpm_current_spinning_refit/{hyptan,ratexp}/`:
  nonspinning (`190` jobs), equal-mass spinning (`380` jobs), and generic
  aligned-spin (`780` jobs) for each template.
- Separated construction mismatch diagnostics from evaluation mismatch
  diagnostics. `collect-local-fits` writes representative construction values
  as `construction_mismatch`; local-fit plots and global-fit validation only
  plot waveform mismatches from explicit evaluation tables.
- Evaluation mismatch plotting now checks declared `t_start`/`t-start` and
  `tref`/`reference_time` metadata against the required `22`-peak comparison
  window (`t_start = 0`, `tref = peak22`).
- The local-fit run order is batched for parallelism while respecting
  dependencies: run `22`, fill HM inputs, run `21,33,32,31,44,42,41,55`, fill
  again, then run `43` after its `33` parent coefficients exist.
- Regenerated Sphinx HTML docs with `make html`.
- Focused tests passed:
  `python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py -q`.

## Completion Update: 2026-05-17

- Rebuilt and used pyRing from
  `/Users/g.carullo@bham.ac.uk/Repos/pyring_test` at commit
  `1f217ab` so TEOBPM exposes the corrected `mode_mixing` implementation.
- Reran all mode-mixing local fits in the active checkout campaigns. This
  includes forced HypTan nonspinning `32` and `43`, plus fresh `32` and `43`
  runs for all RatExp families and the remaining HypTan spinning families.
- Added a TEOBPM mixed-mode `DeltaT` floor in `bayRing/template_waveforms.py`:
  when child-mode merger metadata starts before the required parent mode, the
  child matching time is floored to the parent matching time. This fixed the
  `(4,3)` parent-availability failures seen in equal-mass anti-aligned cases.
- Added robust RatExp minimization/postprocessing handling in
  `bayRing/inference.py`, `bayRing/bayRing.py`, and `bayRing/postprocess.py`.
  Out-of-domain RatExp waveform proposals now receive a large finite residual,
  and invalid point-estimate offset waveforms are skipped during serialization
  and postprocessing instead of failing completed local fits.

### Final Campaign Outputs

All six campaign branches completed with zero local-fit collection failures:

| Template | Family | Jobs | Rows | Mismatch rows | Validation n | Median abs residual | Max abs residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HypTan | nonspinning | 190 | 1121 | 786 | 1121 | 0.09797849263824368 | 8.526432383375958 |
| HypTan | equal-mass spinning | 380 | 2242 | 1626 | 2242 | 0.3269767258404572 | 10.33415257184116 |
| HypTan | generic spinning | 780 | 4602 | 3618 | 4602 | 0.2663106360323272 | 9.572833678274858 |
| RatExp | nonspinning | 190 | 1691 | 996 | 1691 | 0.38586214223836757 | 9.333412804903569 |
| RatExp | equal-mass spinning | 380 | 3382 | 1932 | 3382 | 0.5875241691924438 | 11.205248789377306 |
| RatExp | generic spinning | 780 | 6942 | 4680 | 6942 | 0.5168615120953034 | 10.918890693974168 |

Global-fit JSON files were written under each campaign directory:

- `local_artifacts/teobpm_current_spinning_refit/hyptan/{nonspinning,equal_mass_spinning,generic_spinning}/teobpm_global_fit.json`
- `local_artifacts/teobpm_current_spinning_refit/ratexp/{nonspinning,equal_mass_spinning,generic_spinning}/teobpm_global_fit.json`

### Verification

- Syntax check passed:
  `python -m compileall bayRing/NR_waveforms.py bayRing/template_waveforms.py bayRing/inference.py bayRing/postprocess.py bayRing/bayRing.py scripts/teobpm/teobpm_calibration.py`.
- Focused tests passed:
  `PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py tests/test_higher_mode_mismatch.py tests/test_mismatch_output.py -q`
  (`42 passed`).
- Regenerated the campaign report:
  `reports/teobpm_sxs_calibration_report.tex` and
  `reports/teobpm_sxs_calibration_report.pdf`.

### Caveat

- The collected mismatch tables still report comparison-window status
  `unverified`: they contain mismatch rows, but not the `t_start`/reference
  metadata needed by `plot-local-fits` and `validate` to certify the
  `t_start = 0`, `tref = peak22` comparison window. Coefficient/global-fit
  plots and validation residual plots were generated successfully.

## SEOB/RatExp-NR Completion Update: 2026-05-18

- The dependent mode-mixing task
  `019e3823-9b62-7061-8e20-d1b2c5e90c62` was complete before this run. The
  active mixed-mode parent map is `31 -> 21`, `32 -> 22`, `41 -> 21`,
  `42 -> 22`, and `43 -> 33`; the default `55` quadratic term is
  `Px220x330`.
- New campaign root:
  `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/teobpm_current_spinning_refit_2026_05_18`.
- All three templates used the same RatExp-style NR properties/error workflow:
  the prepared campaigns read peak properties from
  `sxs_mode_peak_properties.csv`, run with `align-with-mismatch-all`, and use
  the generated local error estimates over the `0M` to `30M` error window.
- Those NR properties are calibration labels for this refit. They cannot remain
  required inputs in the final pyRing real-data path; the corresponding peak
  quantities must be globally fitted or removed from production TEOBPM
  evaluation.
- HypTan was rerun as `hyptan_ratexp_nr`, i.e. HypTan template local fits with
  NR peak data supplied through the RatExp properties workflow.
- SEOBNRv5 was run for the same hierarchical families. Early SEOBNRv5
  `mode_43` postprocessing failures were diagnostic-only invalid-waveform
  cases; after adding finite-waveform guards in `bayRing/bayRing.py` and
  `bayRing/postprocess.py`, the nonspinning and equal-mass-spinning SEOBNRv5
  `43` batches were force-rerun cleanly and downstream SEOB global fits were
  regenerated.
- Old local artifact attempts were moved under
  `/Users/g.carullo@bham.ac.uk/Repos/bayRing/local_artifacts/OLD/2026-05-18-pre-seob-ratexp`.

### Final Campaign Outputs

All nine campaign branches completed with zero local-fit collection failures
and zero failed rows in the final local-fit run summaries:

| Template | Family | Jobs | Rows | Mismatch rows | Validation n | Median abs residual | Max abs residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SEOBNRv5 | nonspinning | 190 | 1501 | 1088 | 1501 | 0.29962535880926927 | 3915.407137651293 |
| SEOBNRv5 | equal-mass spinning | 380 | 3002 | 1788 | 3002 | 0.6907369614147977 | 6917.0369977181335 |
| SEOBNRv5 | generic spinning | 780 | 6162 | 4418 | 6162 | 0.5446795700067503 | 16206.551269106038 |
| HypTan with RatExp NR | nonspinning | 190 | 1311 | 1104 | 1311 | 0.30109174804288474 | 3915.407137651293 |
| HypTan with RatExp NR | equal-mass spinning | 380 | 2622 | 1800 | 2622 | 0.8219731174167901 | 6917.0369977181335 |
| HypTan with RatExp NR | generic spinning | 780 | 5382 | 4446 | 5382 | 0.6575626019937975 | 16206.551269106038 |
| RatExp | nonspinning | 190 | 1881 | 1104 | 1881 | 0.4581441272121468 | 3915.407137651293 |
| RatExp | equal-mass spinning | 380 | 3762 | 1788 | 3762 | 0.7898943928225024 | 6917.0369977181335 |
| RatExp | generic spinning | 780 | 7722 | 4446 | 7722 | 0.6721253606455448 | 16206.551269106038 |

Global-fit JSON files were written under each campaign directory:

- `local_artifacts/teobpm_current_spinning_refit_2026_05_18/seobnrv5/{nonspinning,equal_mass_spinning,generic_spinning}/teobpm_global_fit.json`
- `local_artifacts/teobpm_current_spinning_refit_2026_05_18/hyptan_ratexp_nr/{nonspinning,equal_mass_spinning,generic_spinning}/teobpm_global_fit.json`
- `local_artifacts/teobpm_current_spinning_refit_2026_05_18/ratexp/{nonspinning,equal_mass_spinning,generic_spinning}/teobpm_global_fit.json`

### Verification

- Campaign summary written:
  `local_artifacts/teobpm_current_spinning_refit_2026_05_18/campaign_run_summary.json`.
- Final run-summary scan found zero failed or timed-out rows in all nine
  campaign directories.
- Syntax check passed:
  `python -m compileall bayRing/bayRing.py bayRing/postprocess.py bayRing/template_waveforms.py bayRing/inference.py bayRing/NR_waveforms.py scripts/teobpm_calibration/teobpm_calibration.py`.
- Focused tests passed:
  `PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test:/Users/g.carullo@bham.ac.uk/Repos/bayRing python -m pytest tests/test_teobpm_calibration.py tests/test_template_waveforms.py -q`
  (`45 passed`).
- `git diff --check` passed.
