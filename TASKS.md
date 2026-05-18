# Task Notes

## 2026-05-03 Issue 15 Mass Correction

### Goal
- [x] Implement the Christodoulou remnant-mass correction requested in issue 15.
- [x] Apply the correction path to all waveform models that use remnant mass and spin.
- [x] Produce an example plot demonstrating the effect.
- [x] Open a pull request against `main`.

### Context
- Issue 15 asks to account for the Christodoulou mass correction described in arXiv:2411.11269.
- This worktree was created from `origin/main` because the repository has no `master` branch.

### Implementation Plan
- [x] Inspect the existing remnant mass/spin plumbing.
- [x] Add shared correction utilities instead of duplicating formulas model by model.
- [x] Thread the corrected mass/spin through every template path that uses Kerr QNM frequencies.
- [x] Add focused tests for the correction and a model-level regression.

### Verification
- [x] Run the relevant test suite.
- [x] Run an example plotting script and check the output image.

### Outcome
- SXS waveform times are multiplied by the metadata total Christodoulou mass before peak finding, fit-window construction, NR frequency computation, and model evaluation.
- SXS peak-frequency metadata is divided by the same factor, and peak second-derivative metadata by the factor squared.
- The shared corrected NR time axis feeds all templates, including damped sinusoids, Kerr, Kerr plus damped sinusoids, KerrBinary, and TEOBPM.
- Draft PR opened: https://github.com/GCArullo/bayRing/pull/25.

## 2026-05-04 Linear Inversion Comparison

### Goal
- [x] Compare `origin/main` against `codex/issue-15-mass-correction` for SXS:BBH:0305.
- [x] Use Kerr 220 at late times (`t-start=30M`, `t-end=80M`) with linear inversion.
- [x] Use Kerr 220 through 227 from the peak (`t-start=0M`, `t-end=80M`) with linear inversion.
- [x] Compare L2 residuals and default time-domain mismatch outputs.

### Run Context
- Main worktree: `/private/tmp/bayring-issue-15-main`.
- Corrected worktree: `/private/tmp/bayring-issue-15`.
- Run outputs: `/private/tmp/bayring-issue15-runs`.
- A temporary pyRing overlay/shim in `/private/tmp/bayring-issue15-runs` was used for both branches because the local pyRing extension stack mixed stale `NR_amp` and renamed `KerrBH` keywords. The shim maps `qnm_interpolants` to `interpolants` and drops unused `t_ref`, equally in all comparison runs.

### Results
- 220 late-time L2 residual: main `0.009898087028597275`, corrected `0.009937299663441606` (`+0.396164%`, worse).
- 220 late-time 50% time-domain mismatch: real `1.928208951573751e-05` -> `1.9542606114919536e-05` (`+1.351081%`, worse); imaginary `3.652592502556562e-05` -> `3.7009132859089355e-05` (`+1.322917%`, worse).
- 220-227 peak-start L2 residual: main `0.013135685355992806`, corrected `0.013103190788017945` (`-0.247376%`, better).
- 220-227 peak-start 50% time-domain mismatch: real `3.653466058572974e-07` -> `3.610680171517444e-07` (`-1.171104%`, better); imaginary `3.792331662033632e-07` -> `3.7457580492894493e-07` (`-1.228100%`, better).

### Conclusion
- The correction does not improve every tested configuration.
- It slightly worsens the single-mode late-time 220 fit for SXS:BBH:0305.
- It slightly improves the peak-starting 220-227 overtone fit, which is the configuration more directly sensitive to accumulated peak-time and QNM phase alignment.
