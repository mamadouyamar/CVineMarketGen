# CVineMarketGen — C-vine copula financial market generator with moment and tail dependence targeting

[![tests](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml/badge.svg)](https://github.com/mamadouyamar/CVineMarketGen/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

[![Example 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamadouyamar/CVineMarketGen/blob/main/examples/example_ltcma.ipynb) Example 1, LTCMA targeting &nbsp;&nbsp; [![Example 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mamadouyamar/CVineMarketGen/blob/main/examples/example_factors.ipynb) Example 2, macro factors

CVineMarketGen simulates multivariate asset returns whose first four moments,
pairwise correlations and tail dependence match prescribed targets, such as
long-term capital market assumptions (LTCMAs). It implements the two generators
of the third article of the thesis, *Vine-Copula Based Financial Market
Simulation with Moment and Tail Dependence Targeting*:

- the **Fleishman + Vale-Maurelli** generator (first four moments and the
  correlation matrix, Gaussian dependence);
- the **C-vine copula** generator with Johnson SU marginals (first four moments,
  correlation matrix, and the tail dependence and sign-changing conditional
  correlations observed in historical data), with copula families selected from
  the exceedance-correlation curve of each pair, including two-component mixture
  copulas for non-monotone dependence.

> ### 📓 Start with the worked examples
> **[`examples/example_ltcma.ipynb`](examples/example_ltcma.ipynb)** ([▶ Colab](https://colab.research.google.com/github/mamadouyamar/CVineMarketGen/blob/main/examples/example_ltcma.ipynb)): 4 asset classes,
> a synthetic history **simulated from a C-vine with known families**, the families
> **selected back**, the vine **calibrated to J.P. Morgan's 2024 targets** and a scenario
> matrix **simulated**; the Fleishman generator on the same targets; exceedance-correlation
> curves compared. About 3 to 4 minutes.
>
> **[`examples/example_factors.ipynb`](examples/example_factors.ipynb)** ([▶ Colab](https://colab.research.google.com/github/mamadouyamar/CVineMarketGen/blob/main/examples/example_factors.ipynb)): 6 factor excess
> returns built at run time from point-in-time market data (Fama-French, FRED, Yahoo
> Finance), with **no target other than the history**; same pipeline, curves compared with
> the actual factor history. About 3 minutes.
>
> Both are seeded and reproducible.

## Model classes

| Component | Simulate | Estimate / calibrate | Module |
|---|---|---|---|
| Fleishman cubic transform + Vale-Maurelli intermediate correlations (paper, Section 3) | `FleishmanGenerator.simulate` | `FleishmanGenerator.fit` | `cvinemarketgen.fleishman` |
| Johnson SU marginals fitted to four moments (Section 4.3) | `MomentMatch.sim_JSU_with_U` | `MomentMatch.find_params_for_moments_matching_JSU` | `cvinemarketgen.moment_match` |
| Bivariate building blocks: mixtures, NCS copulas, h-functions, tail-dependence catalogs, BIC selection (Appendix C) | `CopulaTools.simulate_mixture` | `CopulaTools.est_mixture_MLE`, `CopulaTools.select_best_preselected_bivariate_copula` | `cvinemarketgen.copulas` |
| C-vine generator: family selection (Algorithm 3), correlation-targeting calibration (Algorithm 4), scenario generation (Algorithm 5) | `CVineGenerator.simulate_known_vine`, `CVineGenerator.run_multi_year_simulation` | `CVineGenerator.fit_and_structure_CVine`, `CVineGenerator.run_vine_optimization` | `cvinemarketgen.cvine` |
| Point-in-time market factor data (6 monthly factor excess returns, downloaded and cached) | — | `load_factor_data`, `factor_targets` | `cvinemarketgen.data` |

`CVineGenerator` inherits from `MomentMatch` and `CopulaTools`, so one object
gives access to everything. Method names follow the paper's algorithms.

## Installation

```sh
pip install git+https://github.com/mamadouyamar/CVineMarketGen.git
```

Requires Python ≥ 3.8 with numpy, scipy, pandas, matplotlib, requests and
[pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib) (tested with 0.6.1 and 0.7.6, pandas 1.4 and 3.0, SciPy 1.7 and 1.17).
To run the smoke tests from a clone: `pip install -e ".[test]"` then `pytest -q`.

## Quick start

### Fleishman + Vale-Maurelli generator

Targets: mean, volatility, skewness, excess kurtosis per asset, and a correlation
matrix. `fit` solves the Fleishman system per asset and the Vale-Maurelli equation
per pair (and reports the pairs without a real root, if any); `simulate` draws the
Gaussian vector, transforms it, and keeps the draw only if it meets the tolerances
of the paper.

```python
import numpy as np, pandas as pd
from cvinemarketgen import FleishmanGenerator

assets = ['U.S. Large Cap', 'U.S. Long Treasuries', 'Commodities', 'Gold']
ltcma = pd.read_csv('data/jpm_ltcma_2024.csv', index_col='Assets').loc[assets]
skewness = pd.Series([-0.56, 0.07, -0.51, 0.21], index=assets)     # from historical data
exkurt   = pd.Series([0.79, 1.06, 1.69, 0.64], index=assets)       # kurtosis - 3

fg = FleishmanGenerator()
out = fg.fit(ltcma['Arithmetic Mean'], ltcma['Volatility'], skewness, exkurt, ltcma[assets])
print(out['coef'][['a', 'b', 'c', 'd']].astype(float).round(4))
print('pairs without a real root:', len(out['infeasible_pairs']))
sim, err, _ = fg.simulate(n=25000, corr_tol=2e-2, seed=1)         # 25000 x 4 DataFrame
```

Output:

```
                           a       b       c       d
U.S. Large Cap        0.0864  0.9500 -0.0864  0.0140
U.S. Long Treasuries -0.0097  0.8987  0.0097  0.0327
Commodities           0.0679  0.8713 -0.0679  0.0398
Gold                 -0.0311  0.9371  0.0311  0.0202
pairs without a real root: 0
try 1: err mean/vol=3.60e-06  skew/kurt=0.000  corr=0.0157  accepted=True
```

### C-vine generator

`run_complete_simulation` chains the four algorithms: targets and Johnson SU
marginals, family selection and vine fit on the historical returns, calibration
of the copula parameters to the target correlations, scenario generation with
accept-reject. Here the "historical" returns are simulated from a known vine.

```python
from cvinemarketgen import CVineGenerator

gen = CVineGenerator(tol_opt=1e-10, n_samples=10000,
                     use_ncs_on_firsttree=False, use_ncs_on_deepertrees=False,
                     use_mixture_on_firsttree=True, use_mixture_on_deepertrees=False,
                     force_try_ncscopula=False, tol_for_optimization_func=1e-6)

# true families of the synthetic history, indexed (i, k) as theta_{ik|1:k-1} in the paper
edges = {(2, 1): ('mixture', [('clayton', 270, 0.66), ('gumbel', 0, 1.84)], 0.70),
         (3, 1): ('gumbel', 180, 1.38), (4, 1): ('gaussian', 0, 0.05),
         (3, 2): ('gaussian', 0, -0.15), (4, 2): ('gaussian', 0, 0.30), (4, 3): ('gaussian', 0, 0.20)}
jsu = {'U.S. Large Cap': (2.570, 2.179, 3.540, 2.646), 'U.S. Long Treasuries': (-0.096, -0.095, 2.279, 2.062),
       'Commodities': (0.611, 0.589, 2.092, 1.774), 'Gold': (-0.523, -0.516, 2.893, 2.675)}
spec = gen.make_vine_spec(assets, edges)
hist = gen.simulate_known_vine(spec, jsu, ltcma['Arithmetic Mean'], ltcma['Volatility'], assets, n=3000, seed=1)

np.random.seed(1)
res = gen.run_complete_simulation(ltcma=ltcma[['Arithmetic Mean', 'Volatility'] + assets],
                                  historical_data=hist, asset_order=assets,
                                  n_year=1, n_per_year=25000, corr_tol=5e-2)
print(gen.selected_edge_table(assets, res['CVinefitresults'], res['vine_results']).to_string(index=False))
sim = res['simulated_years'][0]      # 25000 x 4 DataFrame; max |corr error| 2.5 pp, moments to 1e-6
```

Selected families and calibrated parameters (true families in `edges` above; the near-independent Gold edge, true Gaussian with $\rho = 0.05$, is not identifiable from its exceedance curve and receives a mixture, at the right level):

```
 tree                                                      edge           selected family                   parameters
    1                     U.S. Long Treasuries , U.S. Large Cap  clayton 270° + gumbel 0° w=0.73, theta=(0.646, 1.937)
    1                              Commodities , U.S. Large Cap               gumbel 180°                  theta=1.379
    1                                     Gold , U.S. Large Cap clayton 180° + gumbel 90° w=0.21, theta=(0.598, 1.032)
    2       Commodities , U.S. Long Treasuries | U.S. Large Cap               gaussian 0°                 theta=-0.204
    2              Gold , U.S. Long Treasuries | U.S. Large Cap               gaussian 0°                  theta=0.319
    3 Gold , Commodities | U.S. Large Cap, U.S. Long Treasuries               gaussian 0°                  theta=0.514
```

The exceedance-correlation figures of the notebooks show what the correlation
table cannot: the C-vine generator reproduces the sign change of the conditional
correlation on the mixture edge, the Fleishman generator does not.

Runtime on a laptop: 3 to 4 minutes for `example_ltcma.ipynb` and
about 3 minutes for `example_factors.ipynb`, dominated by the correlation-targeting optimizer
(`n_samples` draws per objective evaluation; the paper used 20000 and a
tolerance of 2e-2, the notebooks 10000 and 5e-2).

## Data

`data/jpm_ltcma_2024.csv` holds the arithmetic mean, volatility and correlation
matrix of 59 asset classes from J.P. Morgan's 2024 Long-Term Capital Market
Assumptions (USD), as published in the public report. No historical return
series is shipped: the paper's skewness, kurtosis and copula-family selection
used licensed Finaeon/GFD data, which is why `example_ltcma.ipynb` simulates its own
history.

`example_factors.ipynb` downloads its six factors at run time with `load_factor_data` and
caches them in `data/factors_cache.csv` (ignored by git). All are monthly excess
returns from point-in-time market series:

| Factor | Construction | Source |
|---|---|---|
| Equity DM | developed-market excess return, Mkt-RF | Fama-French, Developed 3 factors |
| Equity EM | emerging minus developed excess return | Fama-French, Emerging 5 factors minus Developed |
| Real premia | long TIPS / short cash: carry − 8 × change of the 10y TIPS real yield (DFII10) | FRED |
| Inflation | long TIPS / short nominal: carry + 8 × change of the 10y breakeven (T10YIE) | FRED |
| Credit | long Baa corporates / short Treasuries: carry − 10 × change of the Baa − 10y spread (BAA10Y) | FRED |
| Commodity | DBC commodity ETF total return minus the risk-free rate | Yahoo Finance |

The yield-based factors are duration-scaled proxies (Duration-Times-Spread
decomposition of Ben Dor et al., 2007, for the credit leg); the durations are
constants in `cvinemarketgen.data.DURATIONS`. Terms of use: the Fama-French
library is free for research (cite it); the Treasury-derived FRED series are
public domain; Moody's series is a third-party series on FRED, fine to download
for research, not to redistribute as a file; Yahoo Finance data is downloaded
for research use. This is why the cache file is not committed.

## Citation

If you use this package, please cite the thesis chapter and the software
(`CITATION.cff` holds the metadata; GitHub's "Cite this repository" button
formats it):

```
Thioub, M. Y. (2026). CVineMarketGen: C-vine copula financial market generator with
moment and tail dependence targeting (Version 0.1.0) [Computer software].
https://github.com/mamadouyamar/CVineMarketGen
```

## References

- Thesis, article 3: *Vine-Copula Based Financial Market Simulation with Moment
  and Tail Dependence Targeting* (methodology, Sections 3 to 5; algorithms,
  Appendix C).
- Fleishman, A. I. (1978). A method for simulating non-normal distributions.
  *Psychometrika*, 43, 521–532.
- Vale, C. D., & Maurelli, V. A. (1983). Simulating multivariate nonnormal
  distributions. *Psychometrika*, 48, 465–471.
- Johnson, N. L. (1949). Systems of frequency curves generated by methods of
  translation. *Biometrika*, 36, 149–176.
- Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*. Springer.
- Nasri, B. R. (2020). On non-central squared copulas. *Statistics & Probability
  Letters*, 162.
- Pan, Y., Nieto-Barajas, L. E., & Craiu, R. V. (2024). Four-corner tail
  dependence of copulas.
- Ben Dor, A., Dynkin, L., Hyman, J., Houweling, P., van Leeuwen, E., & Penninga, O.
  (2007). DTS (Duration Times Spread). *Journal of Portfolio Management*, 33(2), 77–100.
- Fama, E. F., & French, K. R. Data Library, Dartmouth College.
- Longin, F., & Solnik, B. (2001). Extreme correlation of international equity
  markets. *Journal of Finance*, 56, 649–676.

## Contributing

Please report bugs to mamadou.yamar.thioub@hec.ca with:
* a clear and descriptive title;
* the exact steps necessary to reproduce the problem;
* your environment (`pip freeze` output, Python version);
* a minimal code example.

## Contact

Mamadou Yamar Thioub — [@MamadouYamar](https://twitter.com/MamadouYamar) —
mamadou-yamar.thioub@hec.ca
