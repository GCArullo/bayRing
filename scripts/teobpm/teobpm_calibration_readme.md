# TEOBPM Calibration README

This workflow calibrates TEOBPM against SXS simulations in four separate stages:
local fits, local-fit plots, global fits, and global-fit plots. Every stage is
safe to rerun. Local jobs are skipped when
`Algorithm/point_estimates.dat` already exists for minimization-style runs, or
when `Algorithm/posterior.dat` already exists for nested-sampler runs. Pass
`--force` only when you intentionally want to recompute an existing simulation
and mode.

The default higher-mode set is
`22,21,33,32,31,44,43,42,41,55`. Mode mixing is enabled by default for `32`
and `43`; those modes need completed parent-mode fits before they are run.
`run-local-fits` uses all available CPU cores by default and prints one status
line per job plus a campaign progress bar.

The installed command is `bayRing-teobpm-calibrate`. From a source checkout the
same CLI can also be run as `python -m scripts.teobpm.teobpm_calibration`.

## 0. Choose The Campaign

Set these variables once. Use `HypTan` or `RatExp` for the template, and use
`Minimization` or `Nested-sampler` for the local-fit algorithm.

```bash
export ROOT=runs/teobpm_calibration
export TEMPLATE=RatExp
export BINARY_TYPE=nonspinning        # nonspinning, equal-mass-spinning, spinning
export LOCAL_ALGORITHM=Minimization
export MAX_POLYNOMIAL_DEGREE=3
export MODES=22,21,33,32,31,44,43,42,41,55
export MIXED_MODES=32,43
export RANDOM_FRACTION=1.0
export CAMPAIGN_DIR="${ROOT}/${TEMPLATE}/${BINARY_TYPE}"
export GLOBAL_FIT="${CAMPAIGN_DIR}/teobpm_global_fit.json"
```

Choose exactly one catalog selection style.

For the original TEOBPM-paper SXS subsets, use the reconstructed paper catalog
CSV for the selected binary type. In the current scratch workflow these were:

```bash
export CATALOG_FILE=scripts/teobpm/catalogs/arxiv_1904_09550_nonspinning.csv
# export CATALOG_FILE=/private/tmp/teobpm_current_spinning_refit/catalogs/equal_mass_spinning.csv
# export CATALOG_FILE=/private/tmp/teobpm_current_spinning_refit/catalogs/generic_spinning.csv
```

For a custom SXS list, either set comma-separated IDs or point to a text file:

```bash
export SXS_IDS="SXS:BBH:0180,SXS:BBH:0007"
export SXS_ID_FILE=
```

For a random fraction of the filtered SXS catalog, leave `SXS_IDS`,
`SXS_ID_FILE`, and `CATALOG_FILE` empty, then set for example:

```bash
export RANDOM_FRACTION=0.10
```

Build reusable catalog and hierarchy arguments:

```bash
CATALOG_ARGS=()
[ -n "${CATALOG_FILE:-}" ] && CATALOG_ARGS+=(--catalog-file "$CATALOG_FILE")
[ -n "${SXS_IDS:-}" ] && CATALOG_ARGS+=(--sxs-ids "$SXS_IDS")
[ -n "${SXS_ID_FILE:-}" ] && CATALOG_ARGS+=(--sxs-id-file "$SXS_ID_FILE")

PREPARE_BASE_ARGS=()
GLOBAL_BASE_ARGS=()
if [ "$BINARY_TYPE" != "nonspinning" ]; then
  export BASE_NONSPINNING_FIT="${ROOT}/${TEMPLATE}/nonspinning/teobpm_global_fit.json"
  test -s "$BASE_NONSPINNING_FIT"
  PREPARE_BASE_ARGS+=(--base-nonspinning-fit "$BASE_NONSPINNING_FIT")
  GLOBAL_BASE_ARGS+=(--base-nonspinning-file "$BASE_NONSPINNING_FIT")
fi

if [ "$BINARY_TYPE" = "spinning" ]; then
  export BASE_EQUAL_MASS_FIT="${ROOT}/${TEMPLATE}/equal-mass-spinning/teobpm_global_fit.json"
  test -s "$BASE_EQUAL_MASS_FIT"
fi
```

Run the hierarchy in order:

1. `nonspinning`
2. `equal-mass-spinning`
3. `spinning`

Do not start a higher level until the lower-level JSON checked above exists.

## 1. Local Fits

Prepare the campaign:

```bash
bayRing-teobpm-calibrate prepare \
  --family "$BINARY_TYPE" \
  --output-dir "$CAMPAIGN_DIR" \
  "${CATALOG_ARGS[@]}" \
  "${PREPARE_BASE_ARGS[@]}" \
  --modes "$MODES" \
  --mode-mixing-modes "$MIXED_MODES" \
  --template "$TEMPLATE" \
  --teob-calibration qc \
  --method "$LOCAL_ALGORITHM" \
  --random-fraction "$RANDOM_FRACTION" \
  --validation-fraction 0.33
```

Run non-mixed modes first:

```bash
OMP_NUM_THREADS=1 bayRing-teobpm-calibrate run-local-fits \
  --campaign-dir "$CAMPAIGN_DIR" \
  --modes 22,21,33,31,44,42,41,55
```

Fill higher-mode inputs from completed parent fits:

```bash
bayRing-teobpm-calibrate fill-hm-inputs \
  --campaign-dir "$CAMPAIGN_DIR" \
  --mode-mixing-modes "$MIXED_MODES"
```

Run mixed modes:

```bash
OMP_NUM_THREADS=1 bayRing-teobpm-calibrate run-local-fits \
  --campaign-dir "$CAMPAIGN_DIR" \
  --modes "$MIXED_MODES"
```

Collect local coefficients, mismatch files, and failures:

```bash
bayRing-teobpm-calibrate collect-local-fits \
  --campaign-dir "$CAMPAIGN_DIR" \
  --split all
```

Inspect `local_fit_collection_failures.csv` before moving on.

## 2. Local-Fit Plots

Plot local coefficients and local mismatch diagnostics:

```bash
bayRing-teobpm-calibrate plot-local-fits \
  --local-fit-table "$CAMPAIGN_DIR/local_fit_summary.csv" \
  --output-dir "$CAMPAIGN_DIR/local_fit_plots" \
  --family auto
```

The coefficient plots go in `local_fit_plots/coefficients/`. Mismatch plots and
tables go in `local_fit_plots/mismatches/` when the collected table contains a
`mismatch` column.

## 3. Global Fits

Build the global fit from the training rows in `local_fit_summary.csv`:

```bash
bayRing-teobpm-calibrate global-fit \
  --family "$BINARY_TYPE" \
  --template "$TEMPLATE" \
  --teob-calibration qc \
  "${GLOBAL_BASE_ARGS[@]}" \
  --max-polynomial-degree "$MAX_POLYNOMIAL_DEGREE" \
  --local-fit-table "$CAMPAIGN_DIR/local_fit_summary.csv" \
  --output-file "$GLOBAL_FIT"
```

For spin stages, the fitted terms are spin corrections on top of the
nonspinning baseline. Increase `MAX_POLYNOMIAL_DEGREE` only after checking the
validation residuals; the selected basis is chosen with BIC among the allowed
terms.

## 4. Global-Fit Plots

Plot global coefficient residuals on the validation split. If the validation
table contains mismatch values, this command also writes the standard mismatch
views:

```bash
bayRing-teobpm-calibrate validate \
  --global-fit-file "$GLOBAL_FIT" \
  --validation-table "$CAMPAIGN_DIR/local_fit_summary.csv" \
  --output-dir "$CAMPAIGN_DIR/global_fit_plots"
```

For fixed-coefficient waveform mismatch campaigns, compare one or more
mismatch tables with:

```bash
bayRing-teobpm-calibrate plot-mismatch-comparison \
  --mismatch-table "$CAMPAIGN_DIR/global_mismatch_summary.csv" \
  --label "New global fit" \
  --output-dir "$CAMPAIGN_DIR/global_mismatch_plots" \
  --family auto
```

Nonspinning mismatch plots use `nu`. Spinning mismatch plots use `nu` and
`chi_eff`.
