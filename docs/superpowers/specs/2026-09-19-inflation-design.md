# Inflation as a Child of the Market: Design

**Date:** 2026-09-19. **Status:** approved in chat (user: "go").

## Goal

Notebook 13 and the short paper `paper-inflation.tex`: U.S. CPI inflation
enters the market as a structural child of economic parents, its paths give
price levels and real asset returns, and an oil scenario shows the pass-through.
No new equation code: the `Structural` layer of notebook 11 in its `linear` form
on the monthly inflation rate.

## Data (FRED monthly, `load_fred_monthly`, since 1993-10 to match SPY)

- `CPIAUCSL` (headline CPI, SA) and `CPILFESL` (core CPI, SA): the children are
  `pi = 1200 * diff(log CPI)` and `core = 1200 * diff(log core CPI)`, month over
  month annualized, percent.
- Parents: `d_log_oil = 100 * diff(log DCOILWTICO)` (percent change of WTI),
  `unrate` (UNRATE, percent), `mich` (MICH, Michigan one-year expected
  inflation, percent).
- Asset alongside: SPY monthly return from `load_etf_monthly(['SPY'],
  start='1993-10', cache='data/etf_cache_spy.csv')`.

Probe (1990-2026): headline m/m annualized mean 2.6, sd 3.8, autocorrelation
0.49; `linear(d_log_oil, unrate, mich; lags=3)` has R2 0.55, gamma (0.18,
-0.18, 0.83), Ljung-Box p 0.54 on the residual, ARCH p 0.05.

## The market

```python
dynamics = {('d_log_oil', 'unrate', 'mich'): 'vecm',                 # rank by the test
            'pi':   'linear(d_log_oil, unrate, mich; lags=3)',
            'core': 'linear(d_log_oil, unrate, mich; lags=3)',
            'SPY':  'ar1-garch'}
```

The block is simulated first; both children are rebuilt from the parents' paths
and their own draws; the vine ties the six residual columns (three block
innovations, two children's innovations, SPY's). The two children share parents
but not innovations: their residual correlation is what the vine carries.

## Package additions (`paths.py`, tests in `tests/test_paths.py`)

- `price_level(P, column, base=100.0, scale=1200.0)`: `Paths`-shaped array
  `(n_paths, horizon)` of the price index along each path,
  `base * exp(cumsum(pi / scale))`, for a column of annualized monthly log
  inflation in percent (`scale=1200`); `scale=100` for a monthly rate in percent.
- `deflate(P, asset, inflation, scale=1200.0)`: real per-period returns of a
  return column, `(1 + r) / exp(pi / scale) - 1`, as a `Paths` with one column
  `f'{asset} (real)'`.
- `yoy(P, column, history, scale=1200.0)`: year-on-year inflation along the paths
  (percent), joining the last 11 observed months of the history for the first
  11 simulated months: `100 * (exp(sum of the last 12 monthly log changes / scale) - 1)`.

## Notebook 13 (`examples/13_inflation.ipynb`, about 5 minutes)

1. Load and build the six series; plot headline, core and expectations.
2. The equation for headline: coefficients, t-statistics, R2, sigma; reading
   (oil pass-through per 10 percent, the unemployment slope, the weight of
   expectations, the lags). The same for core in a compact table.
3. The children's residual layer: i.i.d. tests, moments, the correlation between
   headline and core innovations and with the parents' and SPY's.
4. The market: Johansen table for the block, fit, `dynamics_report`, order, edges.
5. Paths: 1,000 x 24 months; fans of monthly and year-on-year headline inflation
   and of the price level; quantiles at 1, 12, 24 months.
6. Oil scenario: oil up 50 percent over six months (drift on the oil innovation),
   response of headline and core inflation and of the price level, baseline vs
   scenario medians.
7. Real returns: SPY nominal vs real cumulative return at 12 and 24 months.
8. Alternative: `'pi': 'ar1'` (a univariate filter): residual tests, residual
   correlations, year-on-year dispersion at 24 months, and the oil scenario has
   no channel.
9. Save; where to go next (TIPS with the real curve of notebook 12, GDP).

## Paper (`paper-inflation.tex`, chapter conventions)

Introduction (inflation is what real returns are measured against; a market
generator that draws it from its own history alone ignores what moves it); the
generator in brief (shared section); the inflation equation as a child
(Phillips-curve reading: Gordon's triangle model, Stock-Watson on the
unforecastability of inflation, Blanchard 2016 on the flattening; the linear
form with lags, the residual layer); parents as a block and the simulation
order; price levels, year-on-year rates and real returns from the paths (the
three identities); example (all tables); conclusion (ARCH in the innovation,
TIPS, GDP next). Export script `paper_inflation_export.py` -> `figs/inflation/`.

## Out of scope

Inflation-linked bonds and the real curve; seasonal adjustment (the series are
SA); regime models of inflation; monetary-policy rules (the fed funds rate is
not a parent).
