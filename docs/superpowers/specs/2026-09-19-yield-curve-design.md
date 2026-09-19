# The yield curve and fixed-income pricing (0.4.0, notebook 12, paper-yieldcurve)

## Goal

Simulate the Treasury curve and price fixed income from it. The curve is reduced to
three factors, level, slope and curvature, by the dynamic Nelson-Siegel model of
Diebold and Li (2006), with principal components as the benchmark; the factors are
three observed monthly series, filtered as one block (notebook 10) whose innovations
join the residual layer next to the assets; simulated factor paths give simulated
curves through the loadings, and curves give discount factors, bond prices and
fixed-income returns. Maturities are deterministic functions of the factors, so
they are not variables of the market: the block simulates the factors, functions
map them to curves and prices.

## Module (`cvinemarketgen/yieldcurve.py`, new; numpy and pandas only)

`NelsonSiegel(lam=0.7308)`, maturities in years, yields in percent.
- `loadings(tau) -> ndarray (len(tau), 3)`: `(1, (1 - e^{-lam tau}) / (lam tau), (1 - e^{-lam tau}) / (lam tau) - e^{-lam tau})`.
- `fit(yields: DataFrame)`: columns are maturities (floats or strings like `'DGS10'` mapped
  by a `maturities` argument); one least-squares regression per date gives the factors;
  attributes `factors` (DataFrame level, slope, curvature), `tau`, `rmse` (Series by
  maturity, in percent), `lam`. `lam='auto'` picks the decay on a grid `[0.2, 2.0]` that
  minimises the total RMSE.
- `curve(factors, tau=None) -> DataFrame or Paths`: yields at `tau` (default: the fitted
  maturities) from a DataFrame of factors, or from a `Paths` of factors (a `Paths` of
  yields with the maturities as assets).
- `to_dict / from_dict`.

`PCACurve(n_components=3)`: same interface (`fit`, `factors`, `curve`, `explained` as a
Series of variance shares); the loadings are the eigenvectors of the covariance of the
yields' levels, the scores the centred projections; `curve` adds the mean back.

Pricing functions (module level, continuous compounding, yields in percent):
- `discount(y, tau) -> exp(-tau * y / 100)`.
- `zero_price(ns_factors_row, tau)` through the NS curve at any `tau`.
- `bond_price(factors_row, coupon, maturity, freq=2, ns=ns) -> price per 100 of face`,
  cash flows discounted on the NS curve.
- `zero_return(factors_t, factors_t1, tau, dt=1/12)`: total return of a zero of maturity
  `tau` bought at `t` and sold at `t + dt` with maturity `tau - dt`, `P_{t+dt}(tau - dt) / P_t(tau) - 1`.
- `constant_maturity_return(factors_t, factors_t1, tau, coupon=None, dt=1/12)`: the
  return of a par bond of maturity `tau` issued at `t` (coupon = its par yield) and sold
  at `t + dt` on the new curve, accrued coupon included: the return of a constant-maturity
  index.
- `curve_returns(P: Paths of factors, tau, kind='zero'|'par', dt=1/12) -> Paths` of monthly
  returns per maturity along simulated factor paths, from the last observed curve.

## Integration

No change to `AssetDynamics` or the markets: the three factors are columns of the
history and a block `'vecm'` or `'var'`; the notebook adds SPY as an asset. The
duration-scaled factors of notebook 06 (Real premia, Inflation, Credit) are the
approximation that `curve_returns` makes exact.

## Data

`load_fred_monthly(['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30'], start='1993-10')`,
396 months (DGS20 has a gap in 1987-1993, hence the start); maturities 1, 2, 3, 5, 7,
10, 20, 30 years.

## Tests (`tests/test_yieldcurve.py`)

- Loadings: shape, `level` loading 1, slope loading -> 1 at `tau -> 0`, curvature -> 0 at both ends.
- Synthetic curves from known factors (plus 2 bp noise): `fit` recovers the factors to
  1e-2, `rmse` below 3 bp, `curve(factors)` reproduces the yields; `lam='auto'` returns a
  value in the grid.
- PCA on the same synthetic set: `explained` sums to 1 and the first three components
  explain above 99 percent; `curve(factors)` reproduces the yields to 1e-6.
- Pricing: a par bond on a flat curve prices at 100 (to 1e-8 with continuous
  compounding adjusted par coupon); `zero_return` on an unchanged flat curve equals
  `exp(y dt / 100) - 1`; `constant_maturity_return` on an unchanged curve equals the
  carry plus roll-down, positive on an upward-sloping curve; `curve_returns` on a
  `Paths` of constant factors gives the same number on every path.
- Round trip `to_dict / from_dict` of `NelsonSiegel`.

## Notebook 12 and paper-yieldcurve

Load the eight maturities since 1993; fit Nelson-Siegel with `lam = 0.7308` and show
the loadings, the factors against the 10-year yield and the 1-to-10 slope, the RMSE by
maturity; PCA benchmark (variance explained, loadings shaped as level, slope,
curvature). The factors as a block next to SPY: rank test, `'vecm'`, the residual tests
(curvature keeps some autocorrelation with one lag: report), the innovation
correlations, the vine's families. Paths: 1,000 paths of 24 months of the factors from
the last curve, the simulated curves at months 1, 12, 24 (fan of the 10-year and of the
2s10s spread), and fixed income: the 2, 5, 10 and 30-year constant-maturity returns
along the paths, their annualized mean, volatility and correlation with SPY, against
the history's constant-maturity returns built the same way on the historical curves.
Comparison with the duration proxy `-D * change of yield + carry` of notebook 06.
Export script `paper_yieldcurve_export.py` into `figs/yieldcurve/`.

## Not in scope

Arbitrage-free Nelson-Siegel (the Christensen-Diebold-Rudebusch adjustment); Svensson's
fourth factor; state-space estimation of the factors; credit curves.
