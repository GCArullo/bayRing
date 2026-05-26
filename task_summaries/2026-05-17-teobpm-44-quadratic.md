# TEOBPM 44 Quadratic Contribution

## Request

Investigate whether adding a quadratic contribution to the TEOBPM `(4,4)` mode
removes residual `220x220` content and improves mismatches on a few SXS
simulations.

## First Iteration Implementation

- Added an opt-in `TEOB-quadratic-44` branch to `bayRing.template_waveforms`.
- The branch adds a tapered `220x220` sum-frequency QNM to TEOBPM `(4,4)`.
- The sampled parameters are:
  - `ln_A_sum_440_220_220`
  - `phi_sum_440_220_220`
- Added fixed window controls:
  - `TEOB-quadratic-44-window-start`
  - `TEOB-quadratic-44-window-width`
- Registered priors in TEOBPM inference and passed the new options through
  bayRing and injection construction.
- Updated Sphinx docs and regenerated HTML locally.

## Current h22-Squared Strategy

The current implementation replaces the free-amplitude constant-QNM branch with

```text
h_quad(t) = k(af) * W(t) * h_22(t)^2
```

where `h_22(t)` is a TEOBPM parent `(2,2)` template evaluated with the same
time grid and remnant metadata as the target `(4,4)` model. The branch no
longer samples `ln_A_sum_440_220_220` or `phi_sum_440_220_220`.

The new option `TEOB-quadratic-44-ratio-fit` selects the perturbative coupling
ratio:

- `khera-total`: complex polynomial fit for the nonprecessing total
  `(220+,220+) -> (4,4)` coupling, using `R++ + R+-` for the tabulated
  identical-parent channel.
- `khera-r++`: complex polynomial fit for the direct `R++` channel only.
- `redondo-yuste`: real amplitude-only spin fit.

`khera-total` is the default because it gives the complex phase and includes the
reflection-symmetric total channel. The Redondo-Yuste option is useful as a
real-amplitude comparison, but it does not set a perturbative phase.

For local TEOBPM fits, the parent `(2,2)` coefficients must be fixed in
`[Priors]`, e.g. `fix-phi_mrg_22`, `fix-c3A_22`, `fix-c3p_22`, and
`fix-c4p_22` for `HypTan`. `RatExp`/`SEOBNRv5` additionally require their
parent `(2,2)` coefficients. The calibration config generator now copies
available parent `(2,2)` point estimates into generated `(4,4)` configs when
`TEOB-quadratic-44 = 1`.

## Tested Treatments

- Direct residual projection on existing local fits.
- Full TEOB coefficient re-optimization plus the quadratic complex amplitude.
- `mismatch_fc_edits`-style treatment with TEOB coefficients fixed to the
  existing local-fit values and only the quadratic complex amplitude fit.
- A two-coefficient time-dependent envelope was tested in projection. It often
  improves raw residual power, but it adds two extra complex degrees of freedom
  and was not promoted to the main model in this iteration.

## Simulation Checks

Tested HypTan nonspinning `(4,4)` local fits:

| SXS ID | Best treatment in these runs | Median real mismatch | Median imag mismatch | L2 residual |
| --- | --- | ---: | ---: | ---: |
| `SXS_BBH_0180` | fixed TEOB + quadratic | `8.7166e-4` | `7.8773e-4` | `0.0321468` |
| `SXS_BBH_0169` | fixed TEOB + quadratic | `2.3122e-2` | `2.0653e-2` | `0.0804043` |
| `SXS_BBH_0297` | free TEOB + quadratic | `6.0044e-5` | `6.0838e-5` | `0.00749233` |

Baseline mismatches for the same simulations were:

| SXS ID | Median real mismatch | Median imag mismatch | L2 residual |
| --- | ---: | ---: | ---: |
| `SXS_BBH_0180` | `2.0557e-3` | `1.8659e-3` | `0.0419633` |
| `SXS_BBH_0169` | `3.4956e-2` | `3.1288e-2` | `0.106139` |
| `SXS_BBH_0297` | `1.3771e-4` | `1.4041e-4` | `0.0106161` |

The weighted residual correlation with the fitted quadratic basis drops from
`0.715`, `0.181`, and `0.167` in the baseline fits to approximately `1e-5`,
`1e-5`, and `1e-6` after the quadratic term is included.

## Artifacts

- Summary CSV:
  `local_artifacts/teobpm_44_quadratic/summary/teobpm_44_quadratic_summary.csv`
- Residual comparison plots:
  `local_artifacts/teobpm_44_quadratic/summary/*_residual_comparison.png`
- Residual-projection plots:
  `local_artifacts/teobpm_44_quadratic/simple_grid/diagnostics/*_residual_quadratic_projection.png`
- Rerun configs:
  `local_artifacts/teobpm_44_quadratic/{simple_grid,fixed_teob}/configs/`

## Verification

- `PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest tests/test_template_waveforms.py tests/test_inference.py tests/test_source_structure.py -q`
  passed with `353 passed, 6 skipped`.
- `python -m compileall bayRing/template_waveforms.py bayRing/inference.py bayRing/initialise.py bayRing/NR_waveforms.py bayRing/bayRing.py`
  passed.
- `make html` in `docs/` succeeded.

## h22-Squared Verification

- Focused tests passed:
  `PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest tests/test_template_waveforms.py::test_teobpm_quadratic_44_uses_parent_22_template_and_qqnm_ratio tests/test_template_waveforms.py::test_teobpm_quadratic_44_rejects_other_modes tests/test_inference.py::test_teobpm_quadratic_44_uses_fixed_parent_22_without_free_quadratic_amplitude -q`
- Touched-module tests passed:
  `PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest tests/test_template_waveforms.py tests/test_inference.py tests/test_teobpm_calibration.py::test_fill_higher_mode_inputs_adds_t_peak_and_parent_fixes -q`
- Compile check passed:
  `python -m compileall bayRing/template_waveforms.py bayRing/inference.py bayRing/initialise.py bayRing/NR_waveforms.py bayRing/bayRing.py scripts/teobpm_calibration/teobpm_calibration.py tests/test_template_waveforms.py tests/test_inference.py`
- Documentation rebuild passed:
  `make html` in `docs/`.
- A broader calibration test run had one unrelated failure:
  `tests/test_teobpm_calibration.py::test_prepare_parser_defaults_to_mixed_higher_modes_and_sxs_id_subset`.
  The expected default list omits additional mixed modes that are present in
  the existing `TEOB_MODE_MIXING_PARENTS` default.

## SXS 0305 h22-Squared Comparison

The first quick check used saved paper-case runs with `t-start = 15M`. That was
not the intended start time for this comparison. The corrected run starts at the
mode peak:

```text
t-start = DeltaT_44 = 3.3994562800917265M
```

The corrected comparison uses SXS:BBH:0305, mode `(4,4)`, `HypTan`,
`TEOB-merger-data = 0`, `t-end = 100M`, no 441 overtone, and
`constant-0.0001` error. The old constant-QNM quadratic was run from a temporary
`git archive HEAD` export so that the current working tree did not need to be
reverted.

| Case | Residual L2 | Median real mismatch | Median imag mismatch |
| --- | ---: | ---: | ---: |
| baseline | `3.01872e-2` | `1.26410e-3` | `1.35610e-3` |
| old constant-QNM quadratic | `2.08641e-2` | `6.67918e-4` | `6.61195e-4` |
| new `h_22(t)^2`, `khera-total` | `7.90875e-2` | `7.71451e-3` | `1.02079e-2` |

At the mode peak, the old free-amplitude constant-QNM quadratic improves over
the baseline. The new fixed-ratio `h_22(t)^2` strategy performs worse than the
baseline in this setup and rails multiple 44 coefficients.

Artifacts:

- New configs and run outputs:
  `local_artifacts/teobpm_44_strategy_compare_0305_modepeak/`
- Summary CSV:
  `local_artifacts/teobpm_44_strategy_compare_0305_modepeak/comparison_summary.csv`
- Bar plot:
  `local_artifacts/teobpm_44_strategy_compare_0305_modepeak/comparison_metrics.png`

## SXS 0305 Prior and Window Iteration

Widening the priors by itself did not explain the poor fixed-ratio
`h_22(t)^2` result. With `TEOB-quadratic-44-window-start = 0M` and
`TEOB-quadratic-44-window-width = 10M`, both widened and ultra-widened priors
left the residual near `7.74e-2`, still much worse than the baseline and old
constant-QNM strategy.

The main improvement came from slowing the quadratic window turn-on while
keeping the run start at the mode peak, `DeltaT_44 = 3.3994562800917265M`.
The best local run in this scan used

```text
TEOB-quadratic-44-window-start = 0.0
TEOB-quadratic-44-window-width = 25.0
```

| Case | Residual L2 | Median real mismatch | Median imag mismatch | Railing flags |
| --- | ---: | ---: | ---: | --- |
| baseline | `3.01872e-2` | `1.26410e-3` | `1.35610e-3` | none |
| old constant-QNM quadratic | `2.08641e-2` | `6.67918e-4` | `6.61195e-4` | `c4p_44_low` |
| new `h_22(t)^2`, width `10M` | `7.90875e-2` | `7.71451e-3` | `1.02079e-2` | multiple |
| new `h_22(t)^2`, width `25M` | `8.99516e-3` | `1.67188e-4` | `1.81891e-4` | `c3p_44_low` |
| new `h_22(t)^2`, width `30M` | `1.67034e-2` | `3.71782e-4` | `3.46583e-4` | `c3p_44_low` |

The `25M` window reduces the residual L2 by about `70%` relative to the
baseline and about `57%` relative to the old constant-QNM quadratic. The
amplitude reconstruction captures the mode-peak start, the decay, and the
late-time plateau much better than the original `10M` window, which overshoots
badly just after the peak.

The best point estimates are

```text
phi_mrg_44 = 5.315579131229112
c3A_44     = 13.170128535548574
c3p_44     = 6.1108372683005125
c4p_44     = 33.6040269956074
```

The sampler railing diagnostic still flags `c3p_44_low`, so the result should
not be treated as proof that the local parameter support is fully resolved.
However, because the point estimate is not on the original tight bound and the
ultra-wide prior run with a `10M` window still failed, the dominant issue in
this SXS:BBH:0305 test is the window placement/width rather than the original
prior range.

Artifacts:

- Iteration summary:
  `local_artifacts/teobpm_44_strategy_iter_0305_modepeak/iteration_summary.csv`
- Amplitude and residual morphology plot:
  `local_artifacts/teobpm_44_strategy_iter_0305_modepeak/amplitude_morphology_best.png`
- L2 ranking plot:
  `local_artifacts/teobpm_44_strategy_iter_0305_modepeak/iteration_l2_ranking.png`
- Best run output:
  `local_artifacts/teobpm_44_strategy_iter_0305_modepeak/h22sq_win_start_0_width_25/`

## SXS 0305 Standard Nested Run

A standard bayRing nested-sampler run was completed for SXS:BBH:0305 using
the tuned `h_22(t)^2` quadratic strategy from the mode peak:

```text
t-start = 3.3994562800917265
TEOB-quadratic-44-window-start = 0.0
TEOB-quadratic-44-window-width = 25.0
TEOB-quadratic-44-ratio-fit = khera-total
nlive = 128
maxmcmc = 128
```

The run output is
`local_artifacts/teobpm_44_strategy_standard_0305_modepeak/h22sq_win_start_0_width_25_nested/`.

Key diagnostics:

| Quantity | Value |
| --- | ---: |
| logZ | `-1132.910217299494` |
| residual L2 | `8.8983445555440129e-3` |
| NR-error L2 | `1.3644074623035433e-2` |
| median real mismatch | `1.7134356483594626e-4` |
| median imag mismatch | `1.8647462423049266e-4` |

Posterior medians and 90% intervals:

```text
phi_mrg_44 = 5.315556770061 +0.001550851012 -0.001452766663
c3A_44     = 16.323386850481 +0.107659487357 -0.147473685257
c3p_44     = 6.101150980029 +0.088495985286 -0.080891397091
c4p_44     = 33.599659146124 +0.250107750627 -0.230130330319
```

The prior-railing diagnostic still flags `c3p_44_low`. The posterior support is
not located near the numerical lower bound, so this is a diagnostic warning to
inspect rather than evidence that the reported median is itself pinned at the
prior boundary.

## SXS 0306 and 0255 Standard Nested Runs

Standard bayRing nested-sampler runs were completed for SXS:BBH:0306 and
SXS:BBH:0255 using the same tuned `h_22(t)^2` quadratic strategy as the
SXS:BBH:0305 standard run:

```text
TEOB-quadratic-44-window-start = 0.0
TEOB-quadratic-44-window-width = 25.0
TEOB-quadratic-44-ratio-fit = khera-total
nlive = 128
maxmcmc = 128
```

The runs start at the `(4,4)` mode peak:

| SXS ID | `t-start = DeltaT_44` |
| --- | ---: |
| `SXS_BBH_0306` | `3.598609602066972` |
| `SXS_BBH_0255` | `1.1998105334760112` |

Run outputs:

- `local_artifacts/teobpm_44_strategy_standard_0306_0255_modepeak/SXS_BBH_0306_h22sq_win_start_0_width_25_nested/`
- `local_artifacts/teobpm_44_strategy_standard_0306_0255_modepeak/SXS_BBH_0255_h22sq_win_start_0_width_25_nested/`

After inspecting the SXS:BBH:0255 start-time waveform mismatch, the original
SXS:BBH:0255 run above should be treated as a diagnostic only. Its config used
`TEOB-merger-data = 0` and did not pass the SXS peak-properties file, so TEOBPM
used its internal peak fits rather than the NR merger data. The corrected
SXS:BBH:0255 rerun is:

```text
local_artifacts/teobpm_44_strategy_standard_0306_0255_modepeak/SXS_BBH_0255_h22sq_win_start_0_width_25_nested_merger_data/
```

with

```text
TEOB-merger-data = 1
properties-file = local_artifacts/teobpm_current_spinning_refit_2026_05_18/sxs_mode_peak_properties.csv
```

The corrected metadata includes both the target `(4,4)` merger quantities and
the parent `(2,2)` quantities needed by the `h_22(t)^2` quadratic branch. For
SXS:BBH:0255 it uses `A_peak_22 = 0.3452752550582185`,
`omega_peak_22 = 0.3916545355982634`, `A_peak_44 = 0.02180922868624563`,
`omega_peak_44 = 0.8194053206052558`, and
`DeltaT_44 = 1.1998105334760112`.

Comparison against the older local `(4,4)` fits from
`local_artifacts/teobpm_current_spinning_refit_2026_05_18/`:

| SXS ID | Case | Residual L2 | NR-error L2 | Median real mismatch | Median imag mismatch | Railing flags |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `SXS_BBH_0306` | old local fit | `1.483357549806895e-1` | `1.9908006789411854e-3` | `7.228359684986407e-2` | `8.103574828542637e-2` | `c3A_44_up` |
| `SXS_BBH_0306` | new `h_22(t)^2`, width `25M` | `1.610559162548427e-2` | `1.3598240419605594e-2` | `2.118382392074514e-4` | `1.612163433976077e-4` | `c3p_44_low` |
| `SXS_BBH_0255` | old local fit | `1.1658359186996566e-1` | `5.8489968786977674e-2` | `1.500836575310005e-2` | `6.11491408351571e-2` | none |
| `SXS_BBH_0255` | new `h_22(t)^2`, width `25M`, no NR merger data | `5.443618122986016e-2` | `1.3940612052588428e-2` | `4.418045026469142e-3` | `4.706834491215228e-3` | `c3p_44_low` |
| `SXS_BBH_0255` | new `h_22(t)^2`, width `25M`, NR merger data | `5.7387699532906004e-2` | `1.3940612052588428e-2` | `4.573002012455696e-3` | `4.662972750359318e-3` | `c3p_44_low` |

The L2 residual decreases by a factor `9.21` for SXS:BBH:0306 and by a factor
`2.03` for the corrected SXS:BBH:0255 rerun relative to those older local
fits.

Additional diagnostics:

| SXS ID | logZ | Final KS test |
| --- | ---: | --- |
| `SXS_BBH_0306` | `-4446.242984754591` | `D = 0.0271`, `p = 4.45e-4` |
| `SXS_BBH_0255`, no NR merger data | `-47478.435075433226` | `D = 0.0228`, `p = 1.32e-2` |
| `SXS_BBH_0255`, NR merger data | `-53204.21596694976` | `D = 0.0426`, `p = 6.3e-8` |

Posterior medians and 90% intervals:

```text
SXS_BBH_0306
phi_mrg_44 = 3.158409648106 +0.001559898895 -0.001601222300
c3A_44     = 17.860684316756 +0.000091676517 -0.000024140435
c3p_44     = 6.540909754973 +0.060052417232 -0.059016548597
c4p_44     = 36.106558733905 +0.194317075758 -0.204029974379

SXS_BBH_0255
phi_mrg_44 = 5.065098455313 +0.001206318784 -0.001332110822
c3A_44     = 19.118343910679 +0.049806036805 -0.048553967647
c3p_44     = 4.168056943216 +0.038362869322 -0.039854247775
c4p_44     = 15.902537419143 +0.112802607954 -0.115129969124

SXS_BBH_0255, corrected NR merger data
phi_mrg_44 = 5.218147514290 +0.001220549674 -0.001250743637
c3A_44     = 19.121903045247 +0.052231579626 -0.054091404777
c3p_44     = 4.297379926498 +0.029700141866 -0.033679937632
c4p_44     = 10.980863136305 +0.102379395246 -0.109789459374
```

These nested runs saved posterior products successfully. CPNest reported final
KS warnings and then printed `no prior samples found` /
`no MCMC samples found` after posterior saving. The main posterior, evidence,
waveform reconstruction, residual reconstruction, and corner products are
present in both output directories.

Visual inspection of `Waveform_reconstruction.pdf` shows that SXS:BBH:0306
captures the amplitude peak, decay, and plateau well. SXS:BBH:0255 captures
the same broad morphology, but the late-time NR amplitude has oscillatory
structure that the smooth TEOBPM reconstruction does not follow, consistent
with its larger residual L2.

In the corrected SXS:BBH:0255 rerun, the first available NR sample after the
requested `t-start = DeltaT_44` is at `t - t_peak_22 = 1.2997940620298323`.
There the NR amplitude is `0.021808530808678145`, while the reconstructed
model amplitude is `0.02765742187266521`. The merger data are therefore now
being passed correctly, but the start-time amplitude mismatch remains. The
likely cause is the quadratic window convention: `TEOB-quadratic-44-window-start
= 0` is measured relative to the `(2,2)` peak, so the quadratic contribution is
already nonzero at the `(4,4)` peak. A delayed window with start near
`DeltaT_44`, or a window defined relative to the target-mode peak, should be
tested before treating the SXS:BBH:0255 result as production quality.

## Quadratic-44 Window Sampling

The quadratic-44 window can now be varied in local fits. The fixed config
values remain the fallback:

```text
TEOB-quadratic-44-window-start
TEOB-quadratic-44-window-width
TEOB-quadratic-44-window-steepness = 1.0
```

If any of the following priors are supplied, they are sampled as local-fit
parameters:

```text
quad44_window_delay
quad44_window_width
quad44_window_steepness
```

The delay is measured relative to the target `(4,4)` peak, so the actual taper
start is `DeltaT_44 + quad44_window_delay`. The steepness parameter preserves
the half-cosine result at `quad44_window_steepness = 1`; larger values make the
central turn-on sharper without changing the start and end points. These
targets are also collected by `bayRing-teobpm-calibrate collect-local-fits` and
accepted by `global-fit`, so global fits can be constructed for them once enough
local training rows exist.

Focused verification passed:

```text
PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest \
  tests/test_template_waveforms.py::test_teobpm_quadratic_44_uses_parent_22_template_and_qqnm_ratio \
  tests/test_template_waveforms.py::test_teobpm_quadratic_44_samples_target_peak_window_parameters \
  tests/test_template_waveforms.py::test_teobpm_quadratic_44_window_uses_global_fit_metadata \
  tests/test_inference.py::test_teobpm_quadratic_44_uses_fixed_parent_22_without_free_quadratic_amplitude \
  tests/test_inference.py::test_teobpm_quadratic_44_samples_requested_window_parameters \
  tests/test_teobpm_calibration.py::test_collect_and_global_fit_include_quadratic_44_window_targets -q
```

Touched-module verification passed with `59 passed`, and `make html` in
`docs/` succeeded.

A broad standard nested run was started for SXS:BBH:0255 with these sampled
window parameters:

```text
local_artifacts/teobpm_44_strategy_standard_0306_0255_modepeak/SXS_BBH_0255_h22sq_sampled_window_nested_merger_data.ini
```

It used NR merger data and sampled
`quad44_window_delay in [0, 40]`, `quad44_window_width in [1, 80]`, and
`quad44_window_steepness in [0.2, 10]`. The blind 7D nested run was stopped
after several thousand iterations because it remained far below the
fixed-window likelihood scale; it produced only checkpoint/prior files, not a
posterior result. A practical production rerun should first use a seeded
minimization or tighter window priors around the expected 44-peak turn-on, then
start the nested sampler from that better-localised support.

The sampled-window config was then tightened to fix the taper end at `20M`
after the target `(4,4)` peak:

```text
TEOB-quadratic-44-window-end = 20.0
quad44_window_delay in [0, 19]
quad44_window_steepness in [0.2, 10]
```

With `TEOB-quadratic-44-window-end >= 0`, `quad44_window_width` is no longer
sampled or fixed. The code derives the taper width as
`TEOB-quadratic-44-window-end - quad44_window_delay` in the target-peak
convention, while retaining the legacy fixed-width step behavior when the fixed
end option is disabled. The updated SXS:BBH:0255 config parses with NR merger
data enabled, no width prior, and the delay prior capped below the fixed end.

Verification for the fixed-end edit passed:

```text
python -m compileall bayRing/template_waveforms.py bayRing/inference.py \
  bayRing/initialise.py bayRing/bayRing.py bayRing/NR_waveforms.py
PYTHONPATH=/Users/g.carullo@bham.ac.uk/Repos/pyring_test python -m pytest \
  tests/test_template_waveforms.py tests/test_inference.py \
  tests/test_injection.py tests/test_start_time_ranges.py -q
make html
```

The touched-module test run reported `80 passed`, and the docs build succeeded.

## Caveats

- The window placement was tuned per simulation in this exploratory pass.
  A production calibration should either fit or calibrate that placement, or
  choose a conservative global rule.
- The q=2 case improves but remains poor relative to its very small NR error
  estimate.
- The unweighted residual can retain power from other modeling errors; the
  fitted `220x220` component is eliminated in the weighted likelihood metric.
- The current h22-squared implementation has been unit-tested, plumbed through
  config generation, and tested on SXS:BBH:0305 from the mode peak. I have not
  rerun the full multi-SXS comparison table with this new strategy in this
  iteration.
- The original SXS:BBH:0305 mode-peak `h_22(t)^2` run with a `10M` window had
  multiple railed 44 coefficients and did not fit this case well. The tuned
  `25M` window is a substantial improvement, but its `c3p_44_low` diagnostic
  remains a warning sign.
