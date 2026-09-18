# The structural layer: a variable driven by others plus its own innovation (0.4.0, notebook 11, paper-structural)

## Goal

A child variable follows a structural equation on parent variables of the same market
and its own innovation. The innovation is the child's residual layer; the vine joins it
to the parents' innovations and to the assets; on a path the parents are simulated first
and the child is rebuilt from them and from its own draw. It is the structural filter of
the principle paper, and the factor map of paper-factors is its static case. Example:
the USDCAD exchange rate on the U.S. minus Canada 10-year yield differential and the
oil price.

## Model (`cvinemarketgen/structural.py`, new)

`Structural(parents, form='ecm', lags=1)`, estimated by least squares (no new dependency).

- `form='ecm'` (child and parents in levels, integrated): the unrestricted single-equation
  error-correction model
  `Delta s_t = c + a s_{t-1} + b' z_{t-1} + gamma' Delta z_t + sum_{i=1}^{lags} phi_i Delta s_{t-i} + sigma eps_t`,
  with `kappa = a` (adjustment, expected negative), long-run relation `theta = -b / a`,
  contemporaneous response `gamma`. Attributes after `fit(s: Series, Z: DataFrame)`:
  `parents`, `kappa`, `theta` (Series over parents), `gamma` (Series), `phi` (list),
  `const`, `sigma`, `half_life` (months, `log(0.5) / log(1 + kappa)`), `r2`, `n_obs`,
  `loglik` (Gaussian), `n_params`, `bic`, `state` (last `lags + 1` values of `s` and the
  last row of `Z`), `tstat` (Series, plain OLS).
- `form='linear'` (child and parents in returns, or stationary): `s_t = c + gamma' z_t + sum phi_i s_{t-i} + sigma eps_t`,
  the factor model with an autoregressive term; `lags=0` gives the static factor map.
- `filter() -> Series` of standardized residuals on the fitted sample.
- `filter_new(s_new, Z_new) -> ndarray`: continues from `state`.
- `unfilter(z: (n_paths, horizon), Zpath: (n_paths, horizon, k)) -> (n_paths, horizon)`:
  the child's path given the parents' simulated paths, from `state`. Exact inverse of
  `filter_new`.
- `simulate(n, seed, Zpath)`: Gaussian innovations through `unfilter` (bootstrap use).
- `name` (`'ECM(USDCAD | rate_diff, log_oil)'`), `spec`, `kind = 'structural'`, `to_dict/from_dict`.

## Container (`dynamics.py`, extended)

- `parse_spec('ecm(rate_diff, log_oil)')` and `'linear(rate_diff, log_oil; lags=1)'`.
- `AssetDynamics.fit`: a `Structural` model is fitted on `history[key]` with
  `history[parents]`; parents must be columns of the history; a dependency order is
  computed (parents before children, cycles raise); `order` attribute lists the keys in
  that order.
- `unfilter(Z)`: models are run in `order`; a child receives the already-simulated paths
  of its parents (levels or returns as the parents' models produce them).
- `filter`: the child's standardized residual column, as for any model. `report` row
  per variable. JSON: `kind='structural'` with the parents' names; `from_dict` restores
  the order.
- Markets: no change beyond docstrings; `Targets.from_history` on the full history.

## Data

FRED monthly (`load_fred_monthly`): `DEXCAUS` (CAD per USD), `DGS10`, `IRLTLT01CAM156N`
(Canada 10-year), `DCOILWTICO` (WTI). Derived columns in the notebook: `log_fx`,
`rate_diff = DGS10 - CA10Y`, `log_oil`. Sample 2003-01 to date, 284 months.

## Tests (`tests/test_structural.py`)

- Simulated parents (two random walks) and a child from a known ECM
  (`kappa = -0.1`, `theta = (0.5, -0.3)`, `gamma = (0.2, 0.1)`): `kappa`, `theta`, `gamma`
  recovered within tolerance; `filter_new` inverts `unfilter` to 1e-9 given the parents'
  paths; JSON round trip.
- `form='linear'`, `lags=0`: coefficients equal the OLS betas of the factor model.
- Container: `{('rd', 'oil'): 'vecm(r=0,q=1)', 'fx': 'ecm(rd, oil)', 'spy': 'ar1-garch'}`
  fits, `order` puts the block before `fx`, `filter` has the history's columns, `unfilter`
  is finite and the child's first simulated value moves with its parents; a cycle
  (`'a': 'ecm(b)', 'b': 'ecm(a)'`) raises `ValueError`.
- Market with block, child and asset: fit, `simulate_paths` finite, save/load, identical
  paths after reload.

## Notebook 11 and paper-structural

The parents as a block `'vecm(r=0,q=1)'` on `(rate_diff, log_oil)` (a VAR in differences,
levels wander), the child `'ecm(rate_diff, log_oil)'` on `log_fx`, SPY alongside; the
ECM table with t-statistics, half-life and long-run elasticities; the residual tests;
the vine on the four residuals (the child's innovation against oil's and the rate
differential's innovations); 24-month paths of USDCAD from the last observed values,
conditional on the parents; the same with a shocked parent path (oil down 30 percent
over a year, the child's response through the ECM); and the comparison with an AR(1)
filter on `log_fx` alone. Export script `paper_structural_export.py` into `figs/structural/`.

## Not in scope

Exogenous regressors inside a VECM block; instrumenting for simultaneity (parents are
taken as weakly exogenous, stated as an assumption); more than one lag of the parents'
differences (`gamma` is contemporaneous only).
