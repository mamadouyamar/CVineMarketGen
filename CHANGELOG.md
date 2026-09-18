# Changelog

## 0.4.0 (unreleased)

- Block filters: `BlockVECM` (cointegration rank by the Johansen trace test, lags
  by BIC) and `BlockVAR` filter a block of variables jointly; a tuple key in
  `dynamics` declares the block, `dynamics={('DGS10', 'DFII10'): 'vecm', 'SPY': 'ar1-garch'}`;
  the block's standardized innovations form its residual layer and its paths come
  back in levels from the last observed rows. Optional dependency `statsmodels`
  (`pip install cvinemarketgen[blocks]`).
- `load_fred_monthly`: monthly means of daily FRED series.
- Notebook 10, a VECM block on the nominal and real 10-year yields.

## 0.3.0 (2026-09-17)

Per-asset dynamics chosen from the data, and assets mapped on simulated factors.

- `dynamics='auto'`: for each asset, a constant or AR(1) mean with a constant,
  GARCH, GJR or EGARCH variance (`GarchFamily`, orders up to 2) or a Gaussian
  hidden Markov model with 2 or 3 regimes (`GaussianHMM`, estimated by
  GenHMM1d, now a dependency); i.i.d. tests on the residual layer (Ljung-Box, Ljung-Box on the
  squares, ARCH-LM), BIC among the candidates that pass, parametric-bootstrap
  Cramér-von Mises goodness-of-fit test of the winner (`select_dynamics`,
  `iid_tests`, `gof_bootstrap`); `cv.dynamics_report` and `cv.dynamics.candidates`.
- A dict of specs per asset (`dynamics={'SPY': 'ar1-gjr(1,1)', 'TLT': 'hmm(2)'}`),
  `AssetDynamics`, JSON round trip of every model.
- `FactorModel`: assets regressed on factors (Newey-West t-statistics, R2),
  `simulate` from factor scenarios or paths with or without a Johnson SU
  residual, `implied_mean` for a view on the factors, save/load.
- Seventh factor, the small-cap premium (SMB); `load_etf_monthly`.
- GARCH-family log-likelihoods and BICs reported on the data's scale (arch fits
  returns times 100), so that they are comparable with the HMM's.
- Fixes: the Johnson SU moment fit is multi-start with a moment check (a
  heavy-tailed residual could end degenerate); the residual-layer kurtosis is
  floored just above the Johnson SU boundary (near-normal residuals left
  `simulate` without an acceptable draw).
- Notebooks 08 (assets on macro factors) and 09 (HMM dynamics); 07 now selects
  the dynamics; 06 with seven factors.

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
