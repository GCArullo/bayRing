# Task Notes

## 2026-05-03 SXS Precessing Catalog Support

### Goal
- Address GitHub issue #4 for SXS precessing NR data reading.
- Add a quick example configuration for a precessing SXS catalog simulation.

### Implementation Tasks
- [x] Inspect existing markdown guidance and config-file examples.
- [x] Read issue #4 and identify the SXS NR metadata path.
- [x] Preserve the existing aligned-spin interface by storing `chi1` and `chi2` as signed projections along the SXS reference orbital-frequency direction.
- [x] Compute `tilt1` and `tilt2` from the full SXS reference spin vectors.
- [x] Use the remnant spin-vector magnitude for the final Kerr spin `af`.
- [x] Add a quick SXS precessing config file using `SXS:BBH:1389`.
- [x] Document the new config file in `config_files/README.md`.

### Verification
- [x] `python -m pytest tests/test_NR_waveforms.py -q`
- [x] `python -m pytest tests/test_source_structure.py tests/test_template_waveforms.py tests/test_NR_waveforms.py -q`
- [x] `python -m pytest -q`
- [x] `git diff --check`
