# Changelog

## 0.2.0 (2026-09-13)

User-facing layer for practitioners, on top of the unchanged engine.

- `Targets`: mean, volatility and correlations from whatever the user has
  (own assumptions, a published table, a history at any frequency), with
  skewness and kurtosis given, estimated, or defaulting to normal.
- `FleishmanMarket` and `CVineMarket`: `fit`, `simulate`, `simulate_paths`,
  `diagnostics`, `plot_exceedance`, `save`/`load` (JSON).
- `CVineMarket` families: `'auto'` (from the history), `'gaussian'`, or
  chosen per pair, mixtures included.
- Dynamics for path simulation: `'ar1'` (Appendix A moment transfer) and
  `'ar1-garch'` (via the optional `arch` package, `pip install cvinemarketgen[garch]`).
- `Paths` container; standalone functions `fit_johnson_su`, `fit_fleishman`,
  `exceedance_curve`, `classify_pair`, `select_family`.
- Exact normal marginal when skewness is 0 and kurtosis 3.
- Seven tutorial notebooks under `examples/`.

## 0.1.0 (2026-09-13)

First release, the code of the thesis chapter *Vine-Copula Based Financial
Market Simulation with Moment and Tail Dependence Targeting*.

- `FleishmanGenerator`: Fleishman cubic transform, Vale-Maurelli intermediate
  correlations, accept-reject simulation.
- `CVineGenerator`: copula family selection from exceedance correlations
  (Algorithm 3), correlation-targeting calibration variable by variable
  (Algorithm 4), scenario generation with accept-reject (Algorithm 5),
  synthetic markets from known families.
- `CopulaTools`: two-component mixture copulas, NCS copulas, tail-dependence
  catalogs, BIC selection, exceedance correlation.
- `MomentMatch`: Fleishman and Johnson SU moment fitting.
- `data`: six point-in-time market factors built from Fama-French, FRED and
  Yahoo Finance at run time.
- Two executed example notebooks (LTCMA targeting; macro factors).
- Compatible with pyvinecopulib 0.6 and 0.7, pandas 1.4 to 3.0, SciPy 1.7 to 1.17.
