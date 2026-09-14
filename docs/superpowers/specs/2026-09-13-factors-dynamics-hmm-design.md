# CVineMarketGen 0.3.0: factor mapping, dynamics selection, HMM dynamics, and the companion paper

Approved in conversation on 2026-09-13, section by section. Decisions taken:
standalone working paper (practitioner, SSRN-style) outside the repo; residual
of the factor regression simulated by default with an option to drop it;
selection by i.i.d. tests first and BIC as tie-breaker, per asset; mean models
constant or AR(1), variance models constant, GARCH(p,q), GJR(p,q), EGARCH(p,q)
over a (p,q) grid; the HMM enters as a dynamics model through its Rosenblatt
residuals, reimplemented in the package and checked against GenHMM1d; the
parametric-bootstrap Cramér-von Mises goodness-of-fit test of GenHMM1d is
ported and applied to every selected model; the small-cap premium is the
Fama-French U.S. SMB factor; the beta example uses ten U.S.-listed ETFs.

## Goal

Article 3 of the thesis introduces the generator and applies it to capital
market assumptions. This release, and the paper that goes with it, show how
the same generator serves two other models: (i) assets rebuilt from simulated
macro factors through their betas, and (ii) daily paths whose serial dynamics
are chosen per asset among GARCH-type and hidden Markov models by a test-based
rule. The article-3 engine (`moment_match.py`, `copulas.py`, `cvine.py`,
`fleishman.py`) is unchanged.

## Package changes

### `cvinemarketgen/data.py` (extended)

* `FACTORS` gains `'Small cap'` (seventh factor). `load_factor_data` reads SMB
  from the Fama-French `F-F_Research_Data_Factors` monthly file (percent,
  divided by 100) and appends it as the last column. Cache format unchanged
  except for the new column; a cache without it is refreshed automatically.
* `load_etf_monthly(tickers, start='2006-03', cache='data/etf_cache.csv')`:
  monthly simple returns of the tickers from `yahoo_monthly_adjclose`, one
  column per ticker, `PeriodIndex('M')`. Default tickers of the example:
  `['SPY', 'IWM', 'EFA', 'EEM', 'TLT', 'TIP', 'LQD', 'HYG', 'GLD', 'VNQ']`.

### `cvinemarketgen/factors.py` (new)

```
FactorModel(asset_returns, factor_returns, nw_lags=None)
  .fit() -> self
  .alpha (Series), .beta (DataFrame assets x factors), .tstat (DataFrame, alpha and betas),
  .r2 (Series), .resid (DataFrame), .resid_vol (Series), .resid_params (DataFrame, Johnson SU per asset)
  .report (DataFrame: alpha, betas, t-statistics, R2, residual vol)
  .implied_mean(factor_mean) -> Series          # alpha + beta @ factor_mean
  .simulate(F, residuals=True, seed=None)        # F: DataFrame (n x K) -> DataFrame (n x assets); Paths -> Paths
  .save(path) / FactorModel.load(path)           # JSON
```

* Alignment on the intersection of the two indexes; factor columns matched by
  name, in any order.
* OLS per asset with intercept. Newey-West standard errors with
  `nw_lags = floor(4 (n/100)^(2/9))` by default.
* Residuals: Johnson SU fitted with `fit_johnson_su(skew, kurt, mean=0, vol=resid_vol)`;
  drawn i.i.d. per asset, independent across assets and of the factors.
* `simulate`: `alpha + F @ beta.T (+ eps)`. A `Paths` input returns a `Paths`
  with the assets as columns and the same `(n_paths, horizon)`.

### `cvinemarketgen/dynamics.py` (extended)

Univariate models, one per asset, sharing one interface:

```
Model.fit(y: Series) -> self
Model.filter(y) -> Series z          # standardized (GARCH family) or Rosenblatt normal-score (HMM) residuals
Model.unfilter(z: ndarray (n_paths, horizon)) -> ndarray returns, from the last observed state
Model.simulate(n, seed) -> ndarray   # z ~ N(0,1) through unfilter; used by the bootstrap
Model.loglik, Model.n_params, Model.bic, Model.name  # e.g. 'AR(1)-GJR(1,2)', 'HMM(2)'
Model.to_dict() / Model.from_dict(d)
```

* `GarchFamily(mean='const'|'ar1', vol='const'|'garch'|'gjr'|'egarch', p=1, q=1)`
  through `arch.arch_model` (`mean='Constant'` or `'AR', lags=1`;
  `vol='Constant'`, `'GARCH'`, `'GARCH'` with `o=1` for GJR, `'EGARCH'`;
  `dist='normal'`, `rescale=False`, returns scaled by 100 as today).
  `unfilter` runs the variance recursion of the chosen model from the last
  residual and variance, vectorised over paths. `EGARCH` recursion in log
  variance with the standardized residual as in `arch`.
* The existing `AR1` (constant variance, `transfer_targets` of Appendix A) and
  `AR1GARCH` classes are kept as thin wrappers so `dynamics='ar1'` and
  `'ar1-garch'` behave as in 0.2.0 (same numbers, same JSON).
* `AssetDynamics`: the per-asset container used by the markets.
  `fit(history)`, `filter(history) -> DataFrame`, `unfilter(Z (n_paths, horizon, N))`,
  `models` (dict asset -> Model), `report` (DataFrame from the selector, or
  built from the models when they were given), `to_dict/from_dict`.

### `cvinemarketgen/hmm.py` (new)

`GaussianHMM(n_states=2, n_init=10, max_iter=500, tol=1e-8, seed=0)`, same interface as above.

* Parameters: means `mu_k`, standard deviations `sigma_k`, transition matrix
  `Q`, initial distribution `eta0` (estimated). Estimation by EM
  (forward-backward in scaled form), initialised from percentile splits of the
  sample and `n_init - 1` random perturbations; keep the best likelihood.
* `filter`: forward recursion; predictive weights `w_t = eta_{t-1} Q`;
  `u_t = sum_k w_{t,k} Phi((y_t - mu_k)/sigma_k)`; `z_t = Phi^{-1}(u_t)`;
  `eta_t ∝ w_t ⊙ phi_k(y_t)`. Stores `eta_T` (last filtered probabilities)
  and the smoothed probabilities for plots.
* `unfilter`: exact inverse of `filter`: from `eta_T`, for each period
  `w = eta Q`, `y = F_w^{-1}(Phi(z))` by bracketing and Brent's method on the
  mixture CDF, then the forward update. No regime is drawn; the predictive
  mixture already integrates it.
* `n_params = K(K-1) + 2K + (K-1)`; `loglik` from the forward recursion.
* `regimes` property: DataFrame of filtered and smoothed probabilities.

### `cvinemarketgen/selection.py` (new)

```
ljung_box(x, lags=20) -> (stat, pvalue)
arch_lm(z, lags=20) -> (stat, pvalue)
iid_tests(z, lags=20) -> dict: lb_z, lb_z2, arch_lm (p-values), passed (all > alpha is decided by the caller)
cvm_statistic(u) -> float                       # 1/(12n) + sum (u_(i) - (i-0.5)/n)^2
gof_bootstrap(model, y, B=100, seed=0) -> dict: stat, pvalue, stats (B values)
select_dynamics(returns, candidates=('const','garch','gjr','egarch','hmm'), means=('const','ar1'),
                pq=(1, 2), states=(2, 3), alpha=0.05, lags=20, gof=True, B=100, verbose=True)
    -> AssetDynamics with .report
```

* Candidates per asset: every (mean, vol, p, q) with p and q in `range(1, max+1)`
  (`vol='const'` has no orders), plus `HMM(K)` for K in `states`.
  Default: 2 x (1 + 3 x 4) + 2 = 28 models.
* Rule: compute `iid_tests` on each candidate's z; a candidate passes when the
  three p-values exceed `alpha`. Among those that pass, lowest BIC. If none
  passes, lowest BIC overall and `warnings.warn` naming the asset.
* `gof=True`: `gof_bootstrap` on the selected model only. The bootstrap
  simulates `n` observations from the fitted model (`Model.simulate`), refits
  the same specification, recomputes the statistic on the refit's `u`.
  Sequential; the report prints progress when `verbose`.
* `report` columns: `asset, model, mean, vol, p, q, states, n_params, loglik,
  bic, lb_z, lb_z2, arch_lm, passed, cvm, gof_pvalue, n_passed, n_candidates`.
  `AssetDynamics.candidates` keeps the full candidate table per asset.

### `cvinemarketgen/markets.py` (extended)

* `dynamics=` accepts: `None`, `'ar1'`, `'ar1-garch'` (unchanged), `'auto'`
  (runs `select_dynamics` on the history), a dict `{asset: spec}` with spec a
  string `'const-garch(1,1)'`, `'ar1-gjr(1,2)'`, `'ar1'`, `'hmm(3)'`, or an
  `AssetDynamics`. `dynamics_kwargs` forwards options to `select_dynamics`.
* `cv.dynamics_report` returns `AssetDynamics.report`.
* `'auto'`, dict and `AssetDynamics` require history-based targets (same rule
  as `'ar1-garch'` today). JSON save/load covers every model.
* The `Fleishman` market gets the same argument handling.

### `cvinemarketgen/__init__.py`

Exports `FactorModel`, `GaussianHMM`, `AssetDynamics`, `select_dynamics`,
`iid_tests`, `gof_bootstrap`, `load_etf_monthly`. Version `0.3.0`.

## Tests

* `tests/test_factors.py`: betas recovered within 2 standard errors on
  simulated data (n = 240, 3 factors); `simulate` shapes for DataFrame and
  `Paths`; `residuals=False` reproduces `alpha + F @ beta.T` exactly; save/load
  round trip.
* `tests/test_dynamics.py`: (i) on 2,500 simulated AR(1)-GARCH(1,1) days the
  selector picks a GARCH-family model that passes the tests; (ii) on 2,500
  simulated 2-state HMM days it picks `HMM(2)` or `HMM(3)`; (iii)
  `unfilter(filter(y))` reproduces `y` to 1e-10 for one path of every model
  class; (iv) `GaussianHMM` on a fixed series matches GenHMM1d's `EstHMMGen`
  (`mu`, `sigma`, `Q`, `U`, `cvm`) to 1e-4, using a stored reference output
  in `tests/data/genhmm1d_reference.json` produced once from GenHMM1d;
  (v) `ljung_box` and `arch_lm` agree with statsmodels on a fixed vector
  (reference values stored, statsmodels not a dependency).
* `arch` is added to the `test` extra; CI unchanged otherwise.

## Notebooks (`examples/`)

* `06_macro_factors.ipynb`: updated to seven factors.
* `07_daily_paths_for_backtesting.ipynb`: `dynamics='auto'`, the selection
  report, the goodness-of-fit p-values, paths, comparison with 0.2.0's fixed
  AR(1)-GARCH(1,1).
* `08_assets_on_macro_factors.ipynb` (new): load ETFs and factors, fit the
  factor model, read the report, factor targets from the history and from a
  projection, implied expected returns, asset scenarios with and without
  residuals, asset paths from factor paths.
* `09_hmm_dynamics.ipynb` (new, short): one ETF, `GaussianHMM(2)` and `(3)`,
  regime probabilities, Rosenblatt residuals and their tests, the bootstrap
  goodness-of-fit test, a simulated path against the history.

Each notebook: one heading per action, one short cell, print and inspect,
executed before commit, Colab badge. Figures and tables of the paper are
written by an export script outside the repo (`OldVersion/article4_export.py`)
that reruns the relevant cells and saves to `OldVersion/article4_figs/`.

## Docs

User guide: sections "Choosing the dynamics" (candidates, rule, report,
forcing a model, cost) and "Assets on factors" (fit, report, projections,
simulate). API page: new modules. Method page: the two new steps mapped to
functions. Changelog entry for 0.3.0. No release, no Zenodo version while the
repo is private.

## The paper (`OldVersion/article-4.txt`)

Thesis preamble and macros, English, 8 to 12 pages, notation of article 3
(`\theta_{ik|1:k-1}`, `\boldsymbol{\theta}_i`, threshold `z` for exceedance
curves; to avoid a clash the standardized residual is written `\varepsilon_t`
and the Rosenblatt uniform `v_t`).

1. Introduction: what article 3 does, what this paper adds, the two use cases.
2. The generator in one page: marginals, family selection, calibration,
   accept-reject, with references to the algorithms of article 3.
3. Macro factors and asset betas: factor construction (seven factors, DTS for
   the credit leg, small-cap premium), factor targets from history or from a
   projection, the regression, the simulation map, the residual option, the
   implied expected return.
4. Daily dynamics: the candidate set, standardized and Rosenblatt residuals,
   the three i.i.d. tests, the parametric-bootstrap Cramér-von Mises test, the
   selection rule, path simulation from the last observed state, and a remark
   on what stays constant (the copula) and what moves (the marginal
   variances and regimes).
5. Examples: (a) seven factors and ten ETFs, tables of betas and of implied
   means, exceedance curves of factor pairs, one-year asset distributions with
   and without residuals; (b) daily backtesting on four ETFs, the selection
   table, the goodness-of-fit p-values, volatility clustering of simulated
   paths against the history, one-year terminal distributions.
6. Conclusion.

## Implementation order

1. `data.py` (SMB, ETF loader) and `factors.py` with tests; notebook 08.
2. `dynamics.py` generalisation (`GarchFamily`, `AssetDynamics`) and
   `selection.py` without the HMM; tests (i), (iii), (v).
3. `hmm.py`, the bootstrap test, the HMM in the selector; tests (ii), (iv);
   notebook 09.
4. Notebooks 06 and 07; docs; version 0.3.0.
5. Export script and the paper.
