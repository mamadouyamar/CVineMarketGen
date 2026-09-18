# Block filters: VAR and VECM as residual layers (CVineMarketGen 0.4, notebook 10, paper-blocks)

## Goal

A block of variables observed in levels (yields, inflation, breakevens) is filtered
jointly by a vector autoregression or a vector error-correction model; its
standardized innovations join the residual layer of the market, the vine estimates
their dependence together with everything else, and paths are rebuilt in levels
from the last observed values. This is the "block filter" of the principle paper.

## Model (`cvinemarketgen/blocks.py`, new)

`BlockVECM(rank='auto', lags=1, deterministic='co', max_lags=4)` and `BlockVAR(lags='auto', max_lags=4)`,
estimated with statsmodels (optional extra `blocks = ["statsmodels"]`, imported inside functions).

- `fit(X: DataFrame)`: `X` as given (levels for VECM, levels or returns for VAR).
  VECM: rank by the Johansen trace test at 5 percent when `'auto'` (`select_coint_rank`),
  lags of the differences by BIC over `1..max_lags` when `'auto'` (`select_order`).
  Attributes: `columns`, `rank`, `lags`, `alpha` (loading matrix), `beta` (cointegrating
  vectors), `Gamma` (short-run matrices), `const`, `sigma` (innovation standard deviations),
  `loglik`, `n_params`, `bic`, `n_obs`, `state` (the last `lags + 1` rows of `X`).
- `filter() -> DataFrame`: standardized innovations `e_{j,t} / sigma_j`, one column per
  variable, index of the fitted sample minus the first `lags + 1` rows.
- `filter_new(X_new) -> ndarray (n, p)`: continues from `state`.
- `unfilter(Z: (n_paths, horizon, p)) -> (n_paths, horizon, p)`: the recursion in levels
  from `state`, `Delta x_t = c + Pi x_{t-1} + sum Gamma_i Delta x_{t-i} + diag(sigma) z_t`;
  returns levels. Exact inverse of `filter_new`.
- `simulate(n, seed)`: Gaussian innovations through `unfilter`, for the bootstrap.
- `name`: `'VECM(r=1, q=2)'` or `'VAR(q)'`; `kind = 'vecm' | 'var'`; `spec`; `to_dict/from_dict`.

## Container (`dynamics.py`, extended)

`AssetDynamics` accepts tuple keys: `{('DGS2', 'DGS10'): 'vecm', 'SPY': 'ar1-garch'}`.
`parse_spec` maps `'vecm'`, `'vecm(r=1,q=2)'`, `'var'`, `'var(q=2)'` to block models.
`fit` fits each block on its columns; `filter` concatenates block and univariate
residual columns in the history's column order; `unfilter` slices `Z` by column;
`report` has one row per variable (block rows share the model name and its BIC);
JSON keys of blocks are `'DGS2|DGS10'`. `select_dynamics` is unchanged (univariate
candidates only); blocks are declared by the user.

## Markets

`CVineMarket(targets, dynamics={...})` works as today; with a block, `simulate_paths`
returns the block's variables in levels and the others in returns, and
`Paths.cumulative()` documents that it applies to return columns only. `to_dict`
round trip. `Targets.from_history` on levels is allowed (targets of the residual
layer only are used).

## Data (`data.py`)

`load_fred_monthly(series_ids, start, cache='data/fred_cache.csv')`: monthly averages
of daily FRED series, one column per id, `PeriodIndex('M')`, cached and git-ignored.

## Tests (`tests/test_blocks.py`)

- Simulated cointegrated system (p = 3, rank 1, known `alpha`, `beta`): rank 1 found,
  `beta` recovered up to scale, `filter_new` is the inverse of `unfilter` to 1e-9, `n_params`
  and `bic` finite, JSON round trip reproduces `unfilter`.
- VAR on returns: coefficients recovered on simulated data.
- Container with a block and a univariate model: shapes, report rows, order of columns.
- Market: `CVineMarket(t, dynamics={('a','b','c'): 'vecm', 'd': 'ar1'})` fits, saves,
  loads, and `simulate_paths` returns finite levels for the block.

## Notebook 10 and paper-blocks

Monthly FRED: 2-year and 10-year Treasury yields and the 10-year breakeven (levels),
plus SPY monthly returns from the ETF loader. Johansen rank, the VECM, the residual
layer and its i.i.d. tests, the vine on the four residuals (families from the history),
paths of the yields in levels from the last curve with SPY returns alongside, and the
comparison with three univariate AR(1) filters (the vine then carries the levels'
dependence badly). Export script `paper_blocks_export.py` writes the tables of the
paper into `overleaf-path-simulation/figs/blocks/`.

## Not in scope

Automatic discovery of blocks; VECM with exogenous regressors (that is the structural
layer, notebook 11); GARCH on block innovations.
