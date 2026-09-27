# Changelog

## 0.4.0 (unreleased)

- The specification layer (notebook 03): every factor and asset is described by
  what is known and the rest filled by a stated rule. `FactorModel.exposures`
  accepts given betas (`{factor: value}`, the remainder regressed) and pinned
  allowed sets (`{factor: 'fit' | value}`) besides the allowed-set list;
  `beta_source`. `FactorModel.add_asset` / `add_assets`: assets without a history
  from mean, volatility, exposures and residual shape. `Targets.add_factor`: a
  factor without a history from mean, volatility and a few correlations, the
  others by maximum-determinant completion (`complete_correlation`), attached to
  the C-vine by a Gaussian conditional draw. `with_targets(pair_correlations=...)`:
  a correlation between two assets through their residual correlation, with the
  eligible range `rho_sys +/- sqrt((1 - R2_i)(1 - R2_j))`, a positive
  semidefiniteness check and `pair_report`; correlated residuals drawn from a
  Gaussian copula with Johnson SU marginals (`johnson_su_from_normal`).
  `FactorModel(factor_targets=...)`, `spec_report`, `implied_covariance`,
  save/load version 2.
- Dynamics selection by goodness of fit: `select_dynamics(criterion='gof')`
  (default) runs the parametric-bootstrap Cramér-von Mises test on every
  candidate and keeps the lowest BIC among those with a p-value of at least 5
  percent, the rule of the master's thesis; the Ljung-Box and ARCH-LM tests are
  reported as diagnostics (`criterion='iid'` keeps the 0.3.0 rule). For the
  GARCH family the bootstrap uses a Johnson SU innovation fitted to the
  residual (`gof_bootstrap(innovation='jsu')`, refitting model and marginal on
  each simulated series), which tests what the generator simulates; the Gaussian
  version rejected every GARCH whose only defect was a skewed error. `johnson_su_cdf`.
- Family selection, first tree on the copula scale: the exceedance profile that
  classifies (and now tests) each first-tree pair is computed on the
  pseudo-observations through the normal quantile, as the deeper trees already
  did, instead of on the standardized returns, whose skewness was read as copula
  asymmetry (SPY against EFA: 0.19 on returns, 0.10 on normal scores).
  `classify_pair`, `select_family`, `tail_asymmetry_test` take `copula_scale`.
- Family selection: a symmetry test before the classification of Algorithm 3.
  The left-minus-right mean of a pair's exceedance profile is bootstrapped; a
  pair whose asymmetry is not distinguishable from sampling noise gets the
  Gaussian or the Student t copula by BIC (the Student t is a new candidate and
  a new family in the `families` dict, `('student', 0, rho, df)`); asymmetric
  pairs follow the paper's rule. On Gaussian pairs of 260 observations the old
  rule chose a tail-dependent family or a mixture most of the time.
  `CVineMarket(symmetry_test=True)` (default), `cv.classification`,
  `tail_asymmetry_test`, `select_family(symmetry_test=...)`.
- Block filters: `BlockVECM` (cointegration rank by the Johansen trace test, lags
  by BIC) and `BlockVAR` filter a block of variables jointly; a tuple key in
  `dynamics` declares the block, `dynamics={('DGS10', 'DFII10'): 'vecm', 'SPY': 'ar1-garch'}`;
  the block's standardized innovations form its residual layer and its paths come
  back in levels from the last observed rows. Optional dependency `statsmodels`
  (`pip install cvinemarketgen[blocks]`).
- `load_fred_monthly`: monthly means of daily FRED series.
- The structural layer: `Structural`, a child variable on parent variables plus its
  own innovation, `'ecm(p1, p2)'` in levels or `'linear(p1, p2)'` in returns;
  `AssetDynamics` orders parents before children (`order`, cycles refused) and
  hands the child the parents' simulated paths.
- The yield curve: `NelsonSiegel` (factors by least squares per date for a fixed
  or searched decay, curves at any maturity from factors or factor paths) and
  `PCACurve` as the benchmark; pricing from the factors, `bond_price`,
  `par_yield`, `zero_return`, `constant_maturity_return` and `curve_returns`
  (returns of constant-maturity or zero-coupon bonds along simulated paths).
- Fix: the Johnson SU re-fit on a drawn cross-section (Step 2 of Algorithm 5)
  could fail silently for a fat-tailed target and fill the column with NaN;
  `simulate` and `simulate_paths` now fall back to Nelder-Mead and then to the
  calibrated parameters.
- `FactorModel.with_targets(table)`: a copy of the model whose listed assets hit
  a target mean (through the alpha) and volatility (through the residual scale,
  feasible when the target is at least the systematic volatility), from a
  dictionary or a spreadsheet table in annual terms; `target_report` per asset.
- `CVineMarket(symmetry_level=0.90)`: the confidence level of the symmetry test.
- `CVineMarket(families='auto', overrides={(a, b): spec})`: automatic selection
  with named pairs imposed by the user and kept through the calibration.
- The symmetry test's window is -1 to 1 standard deviations (the classification's
  -0.5 to 0.5 gave 22 percent false asymmetries at a nominal 10 on Gaussian
  pairs of 230 months; -1 to 1 gives 8).
- `FactorModel(..., exposures={asset: [factors]})`: a priori exposures, each asset
  regressed on the factors its characteristics justify, the other betas zero.
- Calibration speed: closed-form inverse h-functions for the Gaussian and the
  Student t copulas in the vine sampler (pyvinecopulib's Gaussian inverse is a
  hundred times slower than Clayton's and dominated the calibration once the
  symmetry test made most edges Gaussian or Student); Student t fitted by tau
  inversion; vectorized bootstrap in the symmetry test.
- The Johnson SU kurtosis floor (`3.1 + 2 skew^2`) now also applies to the
  targets of a `CVineMarket` without dynamics (a long-short factor such as SMB
  has kurtosis near 3), on a copy of the targets, the exact normal excepted;
  the engine's marginal fit uses the multi-start `fit_johnson_su`.
- `exclude=[...]` on the markets: periods left out of the residual layer before
  the marginals and the vine are fitted (known one-off interventions, such as
  the pandemic months for the unemployment rate); saved with the model.
- `load_fred_monthly` refetches when the cache starts after the requested
  `start`, and merges new series into the cache instead of overwriting it.
- Every notebook states the mathematics of each step before the code cell that
  runs it: the object, its formula with the symbols defined, the estimator or
  algorithm, and how to read the output.
- `Bridge`: a quarterly series (GDP growth) on the quarterly means of monthly
  parents plus its lags, applied to simulated monthly paths with a Johnson SU
  quarterly innovation; `growth_to_level`, `recession_probability`;
  `load_fred_quarterly`.
- `price_level`, `yoy` and `deflate`: price levels, year-on-year rates and real
  returns from a column of monthly inflation along the paths.
- Notebooks 10 (a VECM block on the nominal and real 10-year yields), 11
  (USDCAD on the rate differential and oil), 12 (the Treasury curve as
  Nelson-Siegel factors filtered as a block, fixed income priced along the
  paths) and 13 (CPI inflation as a child of oil, unemployment and
  expectations; price levels, an oil scenario, real returns) and 14 (quarterly
  GDP from the monthly paths through a bridge equation, recession probabilities).

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
