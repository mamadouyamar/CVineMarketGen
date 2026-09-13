# CVineMarketGen — design

Python package for the third thesis article, "Vine-Copula Based Financial
Market Simulation with Moment and Tail Dependence Targeting", in the style of
the GenHMM1d repository (https://github.com/mamadouyamar/GenHMM1d).

Approved in conversation on 2026-09-13.

## Decisions

- Code base: `OldVersion/momentmatchingscript.py`, the code of record for the
  paper. Code is relocated into modules, not rewritten.
- Name: `CVineMarketGen`, import as `cvinemarketgen`.
- Data shipped: J.P. Morgan 2024 LTCMA targets (mean, volatility, correlation
  matrix) as a CSV. No Finaeon/GFD historical returns: the "historical" series
  of the example are simulated from a C-vine with known families.
- Example size: 4 assets (U.S. Large Cap, U.S. Long Treasuries, Commodities,
  Gold), runtime target under 5 minutes on Colab.

## Layout

```
CVineMarketGen/
├── README.md
├── LICENSE.txt                 MIT
├── setup.py
├── examples.ipynb
├── data/jpm_ltcma_2024.csv     mean, volatility, correlation matrix, 11 assets
└── cvinemarketgen/
    ├── __init__.py
    ├── moment_match.py         class MomentMatch      (Fleishman, Vale-Maurelli, JSU fitting)
    ├── copulas.py              class CopulaTools      (NCS, mixtures, h-functions, exceedance correlation)
    ├── cvine.py                class CVineGenerator   (selection, calibration, simulation)
    └── fleishman.py            class FleishmanGenerator (Section 3 method, from fleishman_comparison.py)
```

## Mapping from the script

| Script (OldVersion/momentmatchingscript.py) | Package |
|---|---|
| `class MomentMatch` | `moment_match.MomentMatch` |
| `class DistributionManager` (JSU, NCS, mixtures, h-functions, exceedance correlation, selection, simulation) | split: JSU fitting into `moment_match.MomentMatch`; the rest into `copulas.CopulaTools` |
| `class FinancialSimulationSystem` | `cvine.CVineGenerator` |
| `OldVersion/fleishman_comparison.py` (Fleishman + Vale-Maurelli + accept-reject) | `fleishman.FleishmanGenerator` |

Three deliberate changes:

1. `load_data_and_setup` takes a DataFrame of targets and a DataFrame of
   returns instead of Excel paths.
2. The main-block plotting and pickle logic is dropped.
3. A helper `simulate_known_vine` is added to generate synthetic returns from
   chosen families, for the example.

Method names inside the classes are kept, so the paper's algorithms and the
code keep matching.

## Notebook (examples.ipynb)

One markdown cell plus one code cell per step, seeded and timed:

1. Setup: imports, Colab install fallback, constants `N_SAMPLES`, `SEED`, the
   4 assets and their LTCMA targets from the CSV.
2. Synthetic market: families for the 6 edges, including a Clayton 270° +
   Gumbel 0° mixture for the equity/bond pair; simulate "historical" returns
   with JSU marginals.
3. Family selection (Algorithm 3): true vs selected family per edge.
4. Calibration and simulation (Algorithms 4 and 5) targeting the LTCMA
   correlations: target vs simulated moments and correlations.
5. Fleishman benchmark on the same targets, same printout.
6. Conditional correlation figure: historical (synthetic), C-vine, Fleishman.

## README

Same sections as GenHMM1d: description with chapter reference, Colab badge,
table of model classes with simulate/estimate entry points, installation via
`pip install git+...`, quick start snippets with real printed outputs,
references, contributing, contact.

## Out of scope

No tests folder, no parity notebook, no GitHub push. The notebook is run once
end to end before hand-over so printed outputs are real.

## Addendum (2026-09-13): second example, macro factors

The notebook has two examples, each following steps 2 to 6 above.

1. **LTCMA targeting**: 4 asset classes (U.S. Large Cap, U.S. Long
   Treasuries, Commodities, Gold), targets from `data/jpm_ltcma_2024.csv`.
2. **Macro factors**: 6 factors (Equity DM, Equity EM, Real Premia, Inflation,
   Credit, Commodity), Equity DM as central node. No published LTCMA exists
   for factors, so the targets (mean, volatility, correlation matrix) and the
   true families of the synthetic market are illustrative constants set in
   one cell, stored in `data/macro_factors_targets.csv` for the README.
   Mixtures enabled on the first tree only, to keep runtime under 5 minutes.

## Addendum 2 (2026-09-13): Example 2 uses point-in-time market data

Supersedes the illustrative targets above. Example 2 has no target other than
the history: six monthly factor excess returns are downloaded at run time by
`cvinemarketgen.data.load_factor_data` (Fama-French Developed and Emerging
market factors; FRED T10YIE, DFII10, BAA10Y as duration-scaled long/short
proxies with durations 8, 8, 10; Yahoo Finance DBC minus RF), cached in
`data/factors_cache.csv` (git-ignored). Targets = sample mean, volatility and
correlation matrix (`factor_targets`); skewness, kurtosis and family selection
from the same sample. Sample April 2006 to the latest month, 244 months at the
time of writing. `data/macro_factors_targets.csv` was removed.

## Addendum 3 (2026-09-13): one notebook per example

`examples.ipynb` was split into `example_ltcma.ipynb` (Example 1) and
`example_factors.ipynb` (Example 2), each with its own Colab badge; the README
links to both.

## Addendum 4 (2026-09-13): packaging hygiene borrowed from pyvinecopulib

`pyproject.toml` replaces `setup.py`; `CITATION.cff` added (Zenodo DOI to be
minted by the author from a GitHub release); `tests/test_smoke.py` (8 tests, no
network) with a GitHub Actions workflow on Python 3.10 and 3.12; notebooks moved
to `examples/` with badges and paths updated and re-executed from there.

## Addendum 5 (2026-09-13): compatibility with the current scientific stack

The first CI run failed on pyvinecopulib 0.7.6 / pandas 3.0 / SciPy 1.17.
Fixes, all behaviour-preserving (both notebooks give identical numbers on the
old and new stacks): Bicop parameters passed as (k, 1) arrays via
`bicop_params`; parameter-bound accessors via `bicop_bounds` (method in 0.6,
property in 0.7); DataFrames converted to arrays before pyvinecopulib calls;
`Distr` column created as object dtype; monthly aggregation via groupby on
periods instead of `resample('M')`; statsmodels' `mvnormcdf` replaced by
SciPy's `multivariate_normal.cdf` (identical values) and statsmodels dropped
from the dependencies.

## Addendum 6 (2026-09-13): documentation site

Sphinx site under `docs/` (furo theme, myst-nb for the executed notebooks,
autodoc/napoleon for the API), built by Read the Docs from
`.readthedocs.yaml`; numpydoc docstrings written for the public entry points
of the four classes; `CHANGELOG.md` added; `doc` extra in `pyproject.toml`.
