# Changelog

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
