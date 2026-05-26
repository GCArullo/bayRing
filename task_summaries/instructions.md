# SXS Ringdown-Quality Selector

This implementation follows the local algorithm request in `/Users/g.carullo@bham.ac.uk/Downloads/instructions.md`.

Key constraints implemented:

- select a single common simulation set across all requested modes;
- restrict the SXS catalog to non-deprecated, nonprecessing, non-eccentric BBH simulations;
- score only the fixed `[0,30]M` post-merger window using the `ell=2` norm peak as `t=0`;
- use mode-by-mode resolution, extrapolation-order, signal-to-error floor, and nonprecessing symmetry metrics;
- use default metric weights `0.7, 0.1, 0.1, 0.1`;
- convert raw mode scores to within-mode percentiles before simulation-level aggregation;
- avoid QNM, remnant-mass, remnant-spin, or ringdown-consistency fitting diagnostics.

Run:

```bash
python -m scripts.ringdown_quality.cli run --config configs/default.yml
```
