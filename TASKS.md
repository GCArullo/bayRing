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
