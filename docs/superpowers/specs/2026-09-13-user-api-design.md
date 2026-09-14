# CVineMarketGen user-facing API — design

Approved in conversation on 2026-09-13 ("go"), with the three choices: JSON
for save/load; mixtures allowed without history (expert use); `arch` optional.

## Goal

Make the package usable by practitioners who arrive with their own inputs,
at any frequency, and want simulated returns or paths, without knowing the
paper's pipeline. The paper's classes (`MomentMatch`, `CopulaTools`,
`CVineGenerator`, `FleishmanGenerator`) stay as the engine, unchanged in
behaviour; the new layer sits on top.

## Use cases

| Case | Inputs | Skew / kurt | Families | Dynamics |
|---|---|---|---|---|
| A | mean, vol, corr | default 0 / 3, or given | Gaussian (default) or per-pair choice | none |
| B | mean, vol, corr + history | from history | from history ('auto') | none or 'ar1' (Appendix A transfer) |
| C | history only (any frequency, e.g. daily) | from sample | from sample | none, 'ar1', 'ar1-garch' |
| D | mean, vol, corr, skew, kurt | given | per-pair choice | none |

Units are the user's; no annualization anywhere. Kurtosis is raw (normal = 3).

## Objects

### `Targets` (`cvinemarketgen/targets.py`)

```
Targets(mean, vol, corr, skew=None, kurt=None, history=None, assets=None)
Targets.from_history(returns, assets=None)
Targets.from_ltcma(table, assets=None, history=None, mean_col='Arithmetic Mean', vol_col='Volatility')
.assets, .mean, .vol, .skew, .kurt (Series), .corr (DataFrame), .history (DataFrame or None)
.moments (DataFrame), .summary() (prints moments + corr), .freq (inferred from history index or None)
.layer ('returns' | 'residuals'), .to_setup_frame(), .to_dict() / .from_dict()
```
Validation: assets consistent across inputs; vol > 0; corr symmetric with unit
diagonal, projected to PD with a warning if needed; kurt >= skew^2 + 1.
Missing skew/kurt with no history -> 0 and 3 with a note in `summary()`.

### `FleishmanMarket(targets)` (`cvinemarketgen/markets.py`)

`fit()` -> self; `.coefficients`, `.intermediate_corr`, `.infeasible_pairs`;
`simulate(n, seed=None, corr_tol=0.05)` -> DataFrame; `simulate_paths(...)`;
`diagnostics(X)`; `plot_exceedance(X, ...)`; `save/load`.

### `CVineMarket(targets, central=None, families='auto', mixtures=True, n_opt=10000, corr_tol=0.05, tol_func=1e-6, dynamics=None)`

- `central`: asset name of the C-vine root; default the first asset.
- `families`: `'auto'` (Algorithm 3 on the history; requires history),
  `'gaussian'` (Gaussian pair copulas everywhere, initialised at the partial
  correlations of the target matrix), or a dict `{(asset_i, asset_k): spec}`
  with `spec = (family, rotation)` or `(family, rotation, theta)` or
  `('mixture', [(fam1, rot1), (fam2, rot2)], w)`; unspecified pairs are Gaussian.
- `fit()` -> self; `.marginals` (Johnson SU table), `.edges` (DataFrame: tree,
  edge, family, rotation, parameters, status), `.engine` (CVineGenerator),
  `.fit_results`, `.vine_results`.
- `simulate(n, seed=None)` -> DataFrame of returns (accept-reject of Algorithm 5).
- `simulate_paths(n_paths, horizon, seed=None)` -> `Paths`.
- `diagnostics(X)` -> `Diagnostics` (`.moments` DataFrame, `.corr_error` DataFrame, `.summary()`).
- `plot_exceedance(X, base=None, pairs=None, zlim=1.0, ax=None)`.
- `save(path)` JSON; `CVineMarket.load(path)`.

Engine extension (small, documented): `CVineGenerator.load_data_and_setup`
accepts `higher_moments` (skew/kurt table) and `historical_data=None`;
`CVineGenerator.fit_results_from_spec(spec)` builds the dict that
`run_vine_optimization` expects from `make_vine_spec` output (empty fallback lists).

### Dynamics (`cvinemarketgen/dynamics.py`)

- `AR1`: per-asset OLS `y_t = a + b y_{t-1} + e_t`; `filter(returns)` ->
  residuals; `unfilter(eps_paths, y0)` -> returns; targets transfer of
  Appendix A (`transfer_targets(targets)`) for case B.
- `AR1GARCH`: AR(1)-GARCH(1,1) per asset via `arch` (optional); `filter` ->
  standardized residuals; `unfilter` rebuilds variance and returns from the
  last observed state. Supported with history-based targets only (case C).
- `CVineMarket(..., dynamics='ar1' | 'ar1-garch' | None)`: the generator is
  fitted on the residual layer; `simulate_paths` filters back.

### `Paths` (`cvinemarketgen/paths.py`)

`.array` (n_paths, horizon, N), `.assets`, `.to_frame()` (long: path, t, assets),
`.wide(asset)` (horizon x n_paths), `.cumulative()` (Paths of cumulative returns),
`.terminal()` (DataFrame n_paths x N).

### Functions (`cvinemarketgen/functions.py`)

`fit_johnson_su(skew, kurt)`, `johnson_su_moments(params)`,
`fit_fleishman(skew, kurt)`, `exceedance_curve(x, y, z=None)`,
`classify_pair(x, y, target_corr=None)`, `select_family(x, y, mixtures=True)`.

## Notebooks (`examples/`)

01_getting_started, 02_bivariate_copulas_and_tails, 03_fleishman_generator,
04_cvine_step_by_step, 05_ltcma_targeting, 06_macro_factors,
07_daily_paths_for_backtesting (daily ETF history from Yahoo, AR(1)-GARCH,
paths). Each: one heading per action, one short cell, print and inspect.

## Docs and README

README leads with case A in ten lines; docs get a "User guide" page (cases A
to D, dynamics, paths, save/load) and the API page lists the new layer first.

## Status (2026-09-13, end of day)

Implemented and committed: `targets.py`, `markets.py`, `dynamics.py`,
`paths.py`, `functions.py`; engine extensions (`higher_moments`, optional
history, `fit_results_from_spec`, exact normal marginal); `load_daily_returns`;
18 tests on pyvinecopulib 0.6/0.7; seven executed tutorial notebooks; user
guide, API page, README rewritten; version 0.2.0. Deviations from the plan:
`simulate_paths` draws without accept-reject by default (a small pool cannot
meet a correlation tolerance); with normal marginals the on-draw refit only
matches mean and volatility (skewness/kurtosis errors of order 3e-2 at n=20000).
