# CVineMarketGen — simulated asset returns that match your assumptions, tails included

[![docs](https://readthedocs.org/projects/cvinemarketgen/badge/?version=latest)](https://cvinemarketgen.readthedocs.io/en/latest/) [![tests](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml/badge.svg)](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

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
pip install arch          # optional, for GARCH dynamics on daily data
```

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
| daily data, want daily paths | `Targets.from_history(daily)` | `'auto'` with `dynamics='ar1-garch'` |

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
cv = CVineMarket(t, central='SPY', families='auto', dynamics='ar1-garch').fit()
cv.edges                                   # the copula selected for each pair
P = cv.simulate_paths(n_paths=1000, horizon=252, seed=1)
P.to_frame()                               # long table: path, day, one column per asset
P.terminal().quantile([0.05, 0.5, 0.95])   # one-year cumulative return
cv.save('model.json'); cv = CVineMarket.load('model.json')
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

Seven tutorials in [`examples/`](examples/), each one small step at a time,
all executed, each with a Colab badge. Start with the first.

| | Notebook | What it shows |
|---|---|---|
| 01 | [`01_getting_started`](examples/01_getting_started.ipynb) | load a table, build targets, simulate with both generators, inspect, paths, save |
| 02 | [`02_bivariate_copulas_and_tails`](examples/02_bivariate_copulas_and_tails.ipynb) | a marginal, a pair, its exceedance curve, classification, family selection, a mixture |
| 03 | [`03_fleishman_generator`](examples/03_fleishman_generator.ipynb) | the Fleishman generator, every output inspected, and what it cannot do |
| 04 | [`04_cvine_step_by_step`](examples/04_cvine_step_by_step.ipynb) | the four algorithms of the paper one at a time, on a synthetic history with known families |
| 05 | [`05_ltcma_targeting`](examples/05_ltcma_targeting.ipynb) | complete workflow with J.P. Morgan's 2024 assumptions as targets |
| 06 | [`06_macro_factors`](examples/06_macro_factors.ipynb) | six macro factors built from public market data, targets from the sample |
| 07 | [`07_daily_paths_for_backtesting`](examples/07_daily_paths_for_backtesting.ipynb) | daily ETF returns, AR(1)-GARCH dynamics, 1,000 one-year paths |

## Data

`data/jpm_ltcma_2024.csv` holds the arithmetic mean, volatility and correlation
matrix of 59 asset classes from J.P. Morgan's 2024 Long-Term Capital Market
Assumptions (USD), as published in the public report. Nothing else is shipped:
notebooks 06 and 07 download their series at run time (Fama-French, FRED,
Yahoo Finance) into a git-ignored cache. The paper's own historical data
(Finaeon/GFD) is licensed and not included.

## Method

The C-vine generator follows the paper: Johnson SU marginals fitted to four
moments; for every pair of the vine, an exceedance-correlation curve
classified into a tail signature, the admissible copula families fitted by
maximum likelihood and the best BIC kept, mixtures for sign-changing
dependence; the copula parameters then re-calibrated, variable by variable, so
that the simulated correlations match the targets; scenarios drawn with an
accept-reject on moments and correlations. The docs' [Method](https://cvinemarketgen.readthedocs.io/en/latest/method.html)
page maps each algorithm to a function.

## Citation

```
Thioub, M. Y. (2026). CVineMarketGen: C-vine copula financial market generator with
moment and tail dependence targeting (Version 0.2.0) [Computer software].
https://github.com/mamadouyamar/CVineMarketGen
```

`CITATION.cff` holds the metadata; GitHub's "Cite this repository" button formats it.

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

Please report bugs to mamadou.yamar.thioub@hec.ca with:
* a clear and descriptive title;
* the exact steps necessary to reproduce the problem;
* your environment (`pip freeze` output, Python version);
* a minimal code example.

## Contact

Mamadou Yamar Thioub — [@MamadouYamar](https://twitter.com/MamadouYamar) —
mamadou-yamar.thioub@hec.ca
