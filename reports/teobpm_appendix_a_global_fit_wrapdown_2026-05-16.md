# TEOBPM Appendix A Global-Fit Wrap-Down

Date: 2026-05-16

## Scope

This report wraps down the current TEOBPM Appendix A/global-fit workflow iteration. The active goals were:

- perform the Appendix A global fits locally;
- compare new global-fit mismatches against existing TEOBPM global-fit mismatches;
- make the mismatch plots standard in the workflow;
- plot nonspinning mismatch versus `nu`;
- plot spinning mismatch over `nu` and `chi_eff`;
- make global-fit bayRing mismatch runs use `Minimization`, not nested sampling;
- keep the workflow restricted to noneccentric data, where `ecc`, `e0`, `eccentricity`, `bmrg`, `Emrg`, and `Jmrg` are treated as eccentric/noncircular variables;
- restructure pyRing and bayRing so `HypTan`/`RatExp` are independent from `qc`/`noncirc`.

## Code State

bayRing branch:

```text
codex/teobpm-sxs-calibration-workflow
```

pyRing worktree used for the cross-repo changes:

```text
/private/tmp/pyring_teobpm_mode_mixing
```

The pyRing worktree is detached and has uncommitted edits. It was used through:

```text
PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing
```

The bayRing code now has:

- `TEOB-calibration = qc|noncirc` as an axis separate from `TEOB-template = HypTan|RatExp`;
- Appendix A fixed-coefficient mismatch configs written with `method = Minimization`;
- Appendix A fixed-coefficient mismatch configs written with `TEOB-calibration = qc`;
- validation that noneccentric/qc workflows reject noncircular fit variables including `ecc`, `e0`, `eccentricity`, `bmrg`, `Emrg`, and `Jmrg`;
- standard mismatch plots where nonspinning rows use `nu` and spinning rows use `nu`-`chi_eff`;
- collection of wide new-vs-existing mismatch tables into the standard point table and summary plots;
- timeout handling in local-fit campaign execution so timed-out jobs are reported rather than crashing the campaign.

The pyRing worktree now has:

- `teob-calibration = qc|noncirc`;
- `teob-template = HypTan|RatExp` independent from the calibration family;
- global-fit metadata validation for noncircular variables;
- ansatz-named TEOBPM amplitude paths for `HypTan` and `RatExp`;
- calibration-file evaluation that does not inject noncircular zeroes into qc payloads.

## Appendix A Fit And Mismatch Artifacts

Primary Appendix A coefficient-fit campaign:

```text
/private/tmp/teobpm_appendix_a_campaign
```

Regenerated Appendix A global-fit file:

```text
/private/tmp/teobpm_appendix_a_campaign/appendix_a_global_fits.json
```

Nonspinning Appendix A coefficient comparison plots:

```text
/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_a_nonspinning_hyptan_comparison.png
/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_a_nonspinning_ratexp_comparison.png
/private/tmp/teobpm_appendix_a_campaign/appendix_a_plots/appendix_a_nonspinning_comparison_residual_summary.png
```

Fixed-coefficient new-vs-existing mismatch campaign:

```text
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1
```

Important generated tables:

```text
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/local_fit_run_summary.csv
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/appendix_a_global_mismatch_run_points.csv
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/appendix_a_new_vs_existing_mismatch_points.csv
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_points.csv
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_summary.csv
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_summary.json
```

Current standard mismatch plots:

```text
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_nonspinning.png
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning.png
/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/new_vs_existing_plots/teobpm_mismatch_comparison_spinning_chi_eff.png
```

The regenerated plot summary says:

```text
Nonspinning mismatch plots use nu. Spinning mismatch plots use nu-chi_eff.
```

## Run Accounting

The fixed-coefficient mismatch campaign has 76 jobs total.

The first minimization pass with 300 second job timeout produced:

```json
{
  "jobs": 76,
  "completed": 48,
  "timeout": 24,
  "failed": 0,
  "skipped_existing": 4
}
```

A targeted 900 second rerun produced:

```json
{
  "jobs": 76,
  "skipped_existing": 52,
  "completed": 7,
  "failed": 17,
  "timeout": 0
}
```

Combined point-estimate outputs currently present:

```text
59
```

The final collected wide mismatch table has:

```text
30 rows
59 mismatch values
18 nonspinning rows
12 spinning rows
15 HypTan rows
15 RatExp rows
```

The mismatch values in the collected table span approximately:

```text
1.17e-05 to 1.14
```

The high upper value should be treated as a diagnostic requiring follow-up, not as a final science result, because 17 equal-mass spinning jobs are still missing.

## SXS Status

The original SXS decoder/import error path was addressed in `bayRing/NR_waveforms.py` by keeping the SXS/Numba compatibility patches in the bayRing import path.

The remaining SXS issue is different. The equal-mass spinning failures are caused by incomplete cached waveform levels resolving through symlinks into:

```text
/Users/g.carullo@bham.ac.uk/.sxs/cache
```

That cache directory is readable but not writable from this environment. For `SXS:BBH:0148`, for example, the cache contains only:

```text
Lev3:metadata.json
Lev4:metadata.json
```

and not the corresponding waveform HDF5 data needed by `sxs.load(.../Lev3|Lev4, extrapolation_order=2)`. The loader then fails before setting `t_NR`.

A writable cache mirror was created:

```text
/private/tmp/teobpm_sxs_cache_rw
```

The fixed-coefficient mismatch configs have been switched from:

```text
download          = 0
```

to:

```text
download          = 1
```

The next attempt should run with:

```text
SXSCACHEDIR=/private/tmp/teobpm_sxs_cache_rw
SXSCONFIGDIR=/private/tmp/teobpm_sxs_config_rw
```

A direct `sxs.load` download attempt from the sandbox failed on DNS resolution. An escalated direct SXS download command was rejected by the environment, so I did not start another long run.

## Verification Run In This Iteration

bayRing:

```text
python -m compileall bayRing/template_waveforms.py bayRing/teobpm_calibration.py bayRing/initialise.py bayRing/bayRing.py bayRing/NR_waveforms.py
python -m pytest tests/test_template_waveforms.py tests/test_teobpm_calibration.py -q
python -m pytest tests/test_inference.py -q
python -m pytest tests/test_teobpm_calibration.py -q
make html
python -m compileall bayRing/teobpm_calibration.py
git diff --check
```

Observed passing results included:

```text
tests/test_template_waveforms.py tests/test_teobpm_calibration.py: 26 passed
tests/test_inference.py: 21 passed, 6 skipped
tests/test_teobpm_calibration.py: 18 passed
docs make html: passed
git diff --check: passed
```

pyRing worktree:

```text
python setup.py build_ext --inplace
python -m compileall pyRing/teobpm_calibration.py pyRing/pyRing.py pyRing/inject_signal.py pyRing/initialise.py
python -m pytest pyRing/tests/test_waveform.py pyRing/tests/test_teobpm_calibration.py -q
git diff --check
```

Observed passing result:

```text
19 passed
```

Before writing this report, I also checked for lingering bayRing/SXS processes. No active target process remained after stopping the stale `/private/tmp/teobpm_current_spinning_refit` batch.

## Next Steps To Complete The Task

1. Finish the SXS cache/download repair.

   Use the writable cache mirror and run a single direct SXS load first:

   ```bash
   env NUMBA_CACHE_DIR=/private/tmp/teobpm_sxs_inspect_numba \
       SXSCACHEDIR=/private/tmp/teobpm_sxs_cache_rw \
       SXSCONFIGDIR=/private/tmp/teobpm_sxs_config_rw \
       python - <<'PY'
   import sxs
   sim = sxs.load("SXS:BBH:0148/Lev4", download=True, extrapolation_order=2,
                  auto_supersede=False, ignore_deprecation=True)
   print(len(sim.h.t))
   PY
   ```

   If this still fails, inspect the exact `sxscatalogdata` asset URL generation in the installed `sxs` package. Do not bypass the SXS load path by dropping cases.

2. Rerun only the missing equal-mass spinning mismatch jobs.

   Use the existing campaign, not a new one:

   ```bash
   env PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing \
       NUMBA_CACHE_DIR=/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/numba_cache \
       MPLCONFIGDIR=/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/mpl \
       SXSCACHEDIR=/private/tmp/teobpm_sxs_cache_rw \
       SXSCONFIGDIR=/private/tmp/teobpm_sxs_config_rw \
       python -m bayRing.teobpm_calibration run-local-fits \
         --campaign-dir /private/tmp/teobpm_appendix_a_global_mismatch_qc_min1 \
         --workers 4 \
         --timeout 1800
   ```

   Completed outputs will be skipped. The current missing set is the 17 failed jobs in `local_fit_run_summary.csv`.

3. Recollect and regenerate the standard mismatch plots.

   ```bash
   env PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing \
       MPLCONFIGDIR=/private/tmp/teobpm_appendix_a_global_mismatch_qc_min1/mpl \
       python -m bayRing.teobpm_calibration appendix-a-collect-global-mismatch \
         --campaign-dir /private/tmp/teobpm_appendix_a_global_mismatch_qc_min1
   ```

   Confirm the summary JSON still says:

   ```text
   Nonspinning mismatch plots use nu. Spinning mismatch plots use nu-chi_eff.
   ```

4. Re-run the focused tests after the final SXS/mismatch completion.

   ```bash
   python -m pytest tests/test_template_waveforms.py tests/test_teobpm_calibration.py tests/test_inference.py -q
   git diff --check
   ```

   In the pyRing worktree:

   ```bash
   python -m pytest pyRing/tests/test_waveform.py pyRing/tests/test_teobpm_calibration.py -q
   git diff --check
   ```

5. Decide how to land the pyRing edits.

   The current pyRing changes live in:

   ```text
   /private/tmp/pyring_teobpm_mode_mixing
   ```

   They need to be committed or moved to the intended pyRing branch. Until then, bayRing mismatch runs that need the new `calibration` keyword must keep:

   ```text
   PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing
   ```

6. Clean up scratch artifacts after final collection.

   Keep the final tables and plots. Remove stale intermediate plot files that are not referenced by `teobpm_mismatch_comparison_summary.json`, especially the old `chi_a` projection, after the final rerun confirms it is no longer generated.

7. Prepare the merge.

   Review the bayRing untracked files before staging. Some files predate this iteration. Stage only the workflow code, docs, tests, reports, and task summaries that belong to this work.
