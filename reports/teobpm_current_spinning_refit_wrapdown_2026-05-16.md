# TEOBPM Current Spinning Refit Wrap-Down

Date: 2026-05-16

## Scope

This report wraps down the in-progress SXS refit of the current spinning TEOBPM
model. The requested production workflow is not complete. The infrastructure is
in place, the original SXS calibration sets have been reconstructed, and the
first HypTan nonspinning local-fit campaign is partially complete.

The requested final science products still need to be generated:

- local and global fits for `HypTan` and `RatExp`;
- hierarchy `nonspinning -> equal-mass spinning -> generic spinning`;
- all available TEOBPM higher modes;
- mode-mixing for `32` and `43`;
- mismatch plots comparing implemented pyRing values against the new fits over
  `q`, `q`-`chi_eff`, and `q`-`chi_a`.

## Main Artifacts

Campaign root:

```text
/private/tmp/teobpm_current_spinning_refit
```

Original SXS catalog CSVs:

```text
/private/tmp/teobpm_current_spinning_refit/catalogs/nonspinning.csv
/private/tmp/teobpm_current_spinning_refit/catalogs/equal_mass_spinning.csv
/private/tmp/teobpm_current_spinning_refit/catalogs/generic_spinning.csv
/private/tmp/teobpm_current_spinning_refit/catalogs/all_sxs_postpeak.csv
```

Rows excluding headers:

```text
nonspinning: 19
equal_mass_spinning: 38
generic_spinning: 78
all_sxs_postpeak: 135
```

Prepared campaign directories:

```text
/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning
/private/tmp/teobpm_current_spinning_refit/hyptan/equal_mass_spinning
/private/tmp/teobpm_current_spinning_refit/hyptan/generic_spinning
/private/tmp/teobpm_current_spinning_refit/ratexp/nonspinning
/private/tmp/teobpm_current_spinning_refit/ratexp/equal_mass_spinning
/private/tmp/teobpm_current_spinning_refit/ratexp/generic_spinning
```

Temporary pyRing worktree used for TEOBPM mode-mixing:

```text
/private/tmp/pyring_teobpm_mode_mixing
```

Use it through:

```bash
PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing
```

Do not remove that worktree until the pyRing-side data/code changes needed for
the production rerun have been committed or copied elsewhere.

## Code And Workflow State

The bayRing branch is:

```text
codex/teobpm-sxs-calibration-workflow
```

The active branch already contains the calibration workflow code. The current
working tree has no tracked-code diff from `HEAD`; relevant local notes/reports
are untracked under `task_summaries/` and `reports/`.

The workflow currently supports:

- full-training runs with validation fraction zero;
- selected-mode local-fit reruns;
- higher-mode config filling from completed `22` fits;
- fixed parent phase and coefficients for mixed `32` and `43` local fits;
- TEOBPM mode-mixing plumbing from bayRing into pyRing;
- SXS cache priming and bounded lower-resolution fallback;
- q-based nonspinning mismatch plots and spinning `q`-`chi_eff` plus
  `q`-`chi_a` plots;
- streaming child bayRing logs instead of capturing all child output in memory.

## Current Run State

Collected current HypTan nonspinning local fits:

```text
/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/local_fit_summary.csv
/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/mismatch_summary.csv
/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/local_fit_collection_failures.csv
/private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/local_fit_collection_summary.json
```

Collection summary:

```json
{
  "jobs": 190,
  "rows": 641,
  "mismatch_rows": 204,
  "failures": 25
}
```

Completed point-estimate files by mode:

```text
21: 19/19
22: 19/19
31: 19/19
32: 12/19
33: 19/19
41: 19/19
42: 19/19
43: 1/19
44: 19/19
55: 19/19
```

The non-mixed HypTan nonspinning modes are complete. Mode `32` was interrupted
with 12 completed points. Mode `43` initially failed because the pyRing
mode-mixing coefficient file did not contain the `s2l4` datasets needed by
`(4,3)`.

I added the missing temporary pyRing HDF5 datasets:

```text
pyRing/data/NR_data/Kerr_BH/swsh_s2.hdf5:s2l4/l4m3lp3np0
pyRing/data/NR_data/Kerr_BH/swsh_s2.hdf5:s2l4/l4m3lp4np0
```

A single diagnostic `SXS:BBH:0180`, mode `43` run then passed the
missing-coefficient stage and reached post-processing. It was stopped during
wrap-down at the user's request, so the lone `43` output should be rerun with
`--force` before using it in a global fit.

No target bayRing calibration process is currently running.

## Immediate Commands To Resume

Use this environment prefix:

```bash
env PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing \
    NUMBA_CACHE_DIR=/private/tmp/teobpm_current_spinning_refit/numba_cache \
    MPLCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/mpl \
    SXSCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/sxs \
    OMP_NUM_THREADS=1
```

Finish the remaining HypTan nonspinning `32` jobs:

```bash
env PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing \
    NUMBA_CACHE_DIR=/private/tmp/teobpm_current_spinning_refit/numba_cache \
    MPLCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/mpl \
    SXSCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/sxs \
    OMP_NUM_THREADS=1 \
    python -u -m bayRing.teobpm_calibration run-local-fits \
      --campaign-dir /private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning \
      --workers 4 \
      --modes 32 \
      --timeout 1800 \
      --bayring-executable "python -u -m bayRing.bayRing"
```

Rerun all HypTan nonspinning `43` jobs after preserving the pyRing `s2l4`
coefficient fix:

```bash
env PYTHONPATH=/private/tmp/pyring_teobpm_mode_mixing \
    NUMBA_CACHE_DIR=/private/tmp/teobpm_current_spinning_refit/numba_cache \
    MPLCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/mpl \
    SXSCONFIGDIR=/private/tmp/teobpm_current_spinning_refit/sxs \
    OMP_NUM_THREADS=1 \
    python -u -m bayRing.teobpm_calibration run-local-fits \
      --campaign-dir /private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning \
      --workers 4 \
      --modes 43 \
      --force \
      --timeout 1800 \
      --bayring-executable "python -u -m bayRing.bayRing"
```

Collect and build the first global fit:

```bash
python -u -m bayRing.teobpm_calibration collect-local-fits \
  --campaign-dir /private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning

python -u -m bayRing.teobpm_calibration global-fit \
  --family nonspinning \
  --template HypTan \
  --teob-calibration qc \
  --local-fit-table /private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/local_fit_summary.csv \
  --output-file /private/tmp/teobpm_current_spinning_refit/hyptan/nonspinning/teobpm_global_fit.json
```

## Blockers And Required Fixes

1. Make the pyRing `43` mode-mixing coefficient fix permanent.

   The packaged file only had `s2l2` and `s2l3` groups. Mode `43` needs the
   `s2l4` group for `((4,3),(3,3,0))` and `((4,3),(4,3,0))`. The temporary HDF5
   patch unblocks local execution, but it is not a reviewed committed data
   update.

2. Decide how to provide RatExp merger quantities for all SXS higher modes.

   `RatExp` requires `TEOB-merger-data = 1`. The prepared RatExp configs do not
   yet point to a full mode-by-mode properties file. Before running RatExp
   production fits, generate or load `A_peak_lm`, `omg_peak_lm`, and
   `A_peaklmdotdot` for all requested modes and expose them through
   `NR_waveforms.read_NR_metadata`.

3. Let equal-mass and generic spinning campaigns download missing SXS waveforms.

   The nonspinning cache is mostly available. The spinning sets will require SXS
   downloads into the private campaign cache. If sandbox DNS blocks this, rerun
   the same command with escalated execution as required by the environment.

4. Implement or finish fixed-coefficient global mismatch evaluation for this
   full spinning campaign.

   The plotting command can already compare wide tables, but the full campaign
   still needs a command/table producer that evaluates:

   - current implemented pyRing TEOBPM coefficients;
   - new global-fit coefficients;
   - mode-mixed `32` and `43` with parent phases and parent coefficients fixed
     appropriately.

## Remaining End-To-End Plan

1. Complete HypTan nonspinning local fits for modes `32` and `43`.
2. Collect HypTan nonspinning local fits and build the nonspinning global fit.
3. Run HypTan equal-mass spinning local fits using the nonspinning global fit as
   the base.
4. Build the HypTan equal-mass aligned-spin global fit.
5. Run HypTan generic spinning local fits using the equal-mass/global hierarchy.
6. Build the HypTan generic spinning global fit.
7. Generate the RatExp mode-by-mode merger properties file.
8. Repeat steps 1-6 for RatExp.
9. Produce fixed-coefficient mismatch tables for current pyRing vs new global
   fits for every template, family, and available mode.
10. Generate the requested plots:
    - nonspinning: mismatch vs `q`;
    - spinning: mismatch over `q` and `chi_eff`;
    - spinning: mismatch over `q` and `chi_a`.
11. Run focused tests.
12. Regenerate documentation only if documentation files are changed in the next
    iteration.
13. Remove `/private/tmp/pyring_teobpm_mode_mixing` once its pyRing changes are
    safely committed or otherwise preserved.

