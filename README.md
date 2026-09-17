# CVineMarketGen — simulated asset returns that match your assumptions, tails included

[![docs](https://readthedocs.org/projects/cvinemarketgen/badge/?version=latest)](https://cvinemarketgen.readthedocs.io/en/latest/) [![tests](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml/badge.svg)](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22740671.svg)](https://doi.org/10.5281/zenodo.22740671)

You have expected returns, volatilities and correlations for a set of assets,
perhaps skewness and kurtosis, perhaps a history at some frequency. You want
simulated returns, or paths, that match them. CVineMarketGen does that with
two generators:

- **Fleishman + Vale-Maurelli**: first four moments and the correlation matrix,
  Gaussian dependence. Fast and exact on correlations, no tail dependence.
- **C-vine copula with Johnson SU marginals**: the same targets, plus the tail
  dependence and the sign-changing conditional correlations observed in
  historical data, through pair copulas selected from the data (mixtures
  included), or chosen by you.

It is the code of the thesis chapter *Vine-Copula Based Financial Market
Simulation with Moment and Tail Dependence Targeting*, with a layer on top for
practitioners. Documentation: [cvinemarketgen.readthedocs.io](https://cvinemarketgen.readthedocs.io/en/latest/).

## Installation

```sh
pip install git+https://github.com/mamadouyamar/CVineMarketGen.git
pip install arch          # optional, for GARCH-family dynamics on daily data
```

The HMM dynamics are estimated by [GenHMM1d](https://github.com/mamadouyamar/GenHMM1d),
installed with the package.

Python 3.8 or later; numpy, scipy, pandas, matplotlib, requests and
[pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) (0.6 and 0.7 supported).

## Ten lines

```python
from cvinemarketgen import Targets, CVineMarket

t = Targets(mean={'Equity': 0.07, 'Bonds': 0.03, 'Gold': 0.04},
            vol={'Equity': 0.16, 'Bonds': 0.05, 'Gold': 0.15},
            corr=[[1, -0.1, 0.05], [-0.1, 1, 0.3], [0.05, 0.3, 1]],
            assets=['Equity', 'Bonds', 'Gold'])
cv = CVineMarket(t, families='gaussian').fit()
X = cv.simulate(25000, seed=1)            # DataFrame, one row per scenario
cv.diagnostics(X).summary()               # target vs simulated moments and correlations
P = cv.simulate_paths(1000, 10, seed=1)   # 1000 paths of 10 periods
```

Units are yours: annual targets give annual returns, daily give daily.

## What you have, what you write

| You have | Targets | Families |
|---|---|---|
| mean, vol, corr | `Targets(mean, vol, corr, assets)` | `'gaussian'`, or your own per pair |
| … plus skewness and kurtosis | `Targets(..., skew=..., kurt=...)` | same |
| a published table (CSV, Excel, DataFrame) | `Targets.from_ltcma(table, assets)` | same |
| a table plus a history | `Targets.from_ltcma(table, assets, history=returns)` | `'auto'`: selected from the history |
| a history only, any frequency | `Targets.from_history(returns)` | `'auto'` |
| daily data, want daily paths | `Targets.from_history(daily)` | `'auto'` with `dynamics='auto'` |
| assets and factors | `Targets.from_history(factors)` | `'auto'`, then `FactorModel(assets, factors)` |

Families chosen by hand, one entry per pair, the rest Gaussian:

```python
families = {('Bonds', 'Equity'): ('mixture', [('clayton', 270), ('gumbel', 0)], 0.6),   # sign-changing dependence
            ('Gold', 'Equity'): ('gumbel', 180)}                                         # lower-tail dependence
cv = CVineMarket(t, families=families).fit()
```

## From a history, with dynamics, to daily paths

```python
from cvinemarketgen import Targets, CVineMarket, load_daily_returns

daily = load_daily_returns(['SPY', 'TLT', 'GLD', 'DBC'], start='2010-01-01')   # or your own DataFrame
t = Targets.from_history(daily)
cv = CVineMarket(t, central='SPY', families='auto', dynamics='auto').fit()
cv.dynamics_report                         # per asset: the model chosen, BIC, i.i.d. tests, bootstrap goodness of fit
cv.edges                                   # the copula selected for each pair
P = cv.simulate_paths(n_paths=1000, horizon=252, seed=1)
P.to_frame()                               # long table: path, day, one column per asset
P.terminal().quantile([0.05, 0.5, 0.95])   # one-year cumulative return
cv.save('model.json'); cv = CVineMarket.load('model.json')
```

`dynamics='auto'` chooses, for each asset, among a constant or AR(1) mean
with a constant, GARCH, GJR or EGARCH variance and a Gaussian hidden Markov
model with two or three regimes: the residual layer of each candidate is
tested for independence (Ljung-Box, Ljung-Box on the squares, ARCH-LM), the
lowest BIC among the candidates that pass wins, and the winner gets a
parametric-bootstrap Cramér-von Mises test. `dynamics='ar1-garch'` keeps the
fixed AR(1)-GARCH(1,1) of version 0.2; a dict fixes the model per asset,
`dynamics={'SPY': 'ar1-gjr(1,1)', 'TLT': 'hmm(2)'}`.

## Assets on factors

```python
from cvinemarketgen import FactorModel, load_factor_data, load_etf_monthly

factors = load_factor_data()                                   # seven monthly macro factors
etfs = load_etf_monthly(['SPY', 'IWM', 'TLT', 'TIP', 'HYG'])   # or your own asset returns
fm = FactorModel(etfs, factors).fit()
fm.report                                                      # alpha, betas, t-statistics, R2
F = CVineMarket(Targets.from_history(factors), central='Equity DM', families='auto').fit().simulate(25000, seed=1)
X = fm.simulate(F)                                             # asset scenarios through the betas, plus a residual
fm.implied_mean(view)                                          # expected asset returns for a view on the factors
```

## Inspecting a fitted market

```python
cv.edges           # family, rotation, calibrated parameters per edge
cv.marginals       # Johnson SU parameters per asset
d = cv.diagnostics(X); d.moments; d.corr_error
cv.plot_exceedance(X)                      # exceedance-correlation curves, history overlaid
```

Pieces on their own: `fit_johnson_su`, `fit_fleishman`, `exceedance_curve`,
`classify_pair`, `select_family`.

## Notebooks

Nine tutorials in [`examples/`](examples/), each one small step at a time,
all executed, each with a Colab badge. Start with the first.

| | Notebook | What it shows |
|---|---|---|
| 01 | [`01_getting_started`](examples/01_getting_started.ipynb) | load a table, build targets, simulate with both generators, inspect, paths, save |
| 02 | [`02_bivariate_copulas_and_tails`](examples/02_bivariate_copulas_and_tails.ipynb) | a marginal, a pair, its exceedance curve, classification, family selection, a mixture |
| 03 | [`03_fleishman_generator`](examples/03_fleishman_generator.ipynb) | the Fleishman generator, every output inspected, and what it cannot do |
| 04 | [`04_cvine_step_by_step`](examples/04_cvine_step_by_step.ipynb) | the four algorithms of the paper one at a time, on a synthetic history with known families |
| 05 | [`05_ltcma_targeting`](examples/05_ltcma_targeting.ipynb) | complete workflow with J.P. Morgan's 2024 assumptions as targets |
| 06 | [`06_macro_factors`](examples/06_macro_factors.ipynb) | seven macro factors built from public market data, targets from the sample |
| 07 | [`07_daily_paths_for_backtesting`](examples/07_daily_paths_for_backtesting.ipynb) | daily ETF returns, dynamics chosen per asset (GARCH family or HMM) by i.i.d. tests, BIC and a bootstrap test, 1,000 one-year paths |
| 08 | [`08_assets_on_macro_factors`](examples/08_assets_on_macro_factors.ipynb) | ten ETFs regressed on the seven factors, a view on the factors turned into asset distributions and paths |
| 09 | [`09_hmm_dynamics`](examples/09_hmm_dynamics.ipynb) | Gaussian HMM for one asset: regimes, the Rosenblatt residual layer, the bootstrap goodness-of-fit test, a simulated year |

## Data

`data/jpm_ltcma_2024.csv` holds the arithmetic mean, volatility and correlation
matrix of 59 asset classes from J.P. Morgan's 2024 Long-Term Capital Market
Assumptions (USD), as published in the public report. Nothing else is shipped:
notebooks 06 to 09 download their series at run time (Fama-French, FRED,
Yahoo Finance) into a git-ignored cache. The paper's own historical data
(Finaeon/GFD) is licensed and not included.

## Method

The C-vine generator follows the paper: Johnson SU marginals fitted to four
moments; for every pair of the vine, an exceedance-correlation curve
classified into a tail signature, the admissible copula families fitted by
maximum likelihood and the best BIC kept, mixtures for sign-changing
dependence; the copula parameters then re-calibrated, variable by variable, so
that the simulated correlations match the targets; scenarios drawn with an
accept-reject on moments and correlations. Version 0.3 adds, on top of the
unchanged engine, the per-asset choice of the daily dynamics (GARCH family, or
a Gaussian hidden Markov model estimated by GenHMM1d) and the factor model. The docs' [Method](https://cvinemarketgen.readthedocs.io/en/latest/method.html)
page maps each algorithm to a function.

## Citation

```
Thioub, M. Y. (2026). CVineMarketGen: C-vine copula financial market generator with
moment and tail dependence targeting (Version 0.3.0) [Computer software]. Zenodo.
https://doi.org/10.5281/zenodo.22804158
```

The DOI `10.5281/zenodo.22804158` identifies version 0.3.0; `10.5281/zenodo.22740671`
always resolves to the latest version. `CITATION.cff` holds the metadata; GitHub's
"Cite this repository" button formats it.

## References

- Thesis, article 3: *Vine-Copula Based Financial Market Simulation with Moment
  and Tail Dependence Targeting*.
- Fleishman, A. I. (1978). A method for simulating non-normal distributions. *Psychometrika*, 43, 521–532.
- Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal distributions. *Psychometrika*, 48, 465–471.
- Johnson, N. L. (1949). Systems of frequency curves generated by methods of translation. *Biometrika*, 36, 149–176.
- Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*. Springer.
- Nasri, B. R. (2020). On non-central squared copulas. *Statistics & Probability Letters*, 162.
- Pan, Y., Nieto-Barajas, L. E., & Craiu, R. V. (2024). Four-corner tail dependence of copulas.
- Longin, F., & Solnik, B. (2001). Extreme correlation of international equity markets. *Journal of Finance*, 56, 649–676.
- Ben Dor, A., Dynkin, L., Hyman, J., Houweling, P., van Leeuwen, E., & Penninga, O. (2007). DTS (Duration Times Spread). *Journal of Portfolio Management*, 33(2), 77–100.

## Contributing

Please report bugs to mamadou.yamar.thioub@gmail.com and mamadou.yamar.thioub@hec.ca with:
* a clear and descriptive title;
* the exact steps necessary to reproduce the problem;
* your environment (`pip freeze` output, Python version);
* a minimal code example.
* Thank you in advance !

## Contact

Mamadou Yamar Thioub — [@MamadouYamar](https://twitter.com/MamadouYamar) —
mamadou.yamar.thioub@gmail.com and mamadou-yamar.thioub@hec.ca
