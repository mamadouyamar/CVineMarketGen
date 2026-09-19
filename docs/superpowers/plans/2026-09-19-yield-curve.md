# Yield Curve and Fixed-Income Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the Treasury curve to Nelson-Siegel factors (PCA as benchmark), simulate the factors as a block of the market, map simulated factors to curves and price fixed income along the paths; notebook 12 and the short paper `paper-yieldcurve.tex`.

**Architecture:** New module `yieldcurve.py` with `NelsonSiegel`, `PCACurve` and pricing functions (numpy and pandas only); no change to the container or the markets, the factors are a block; the notebook and the export script use notebook 10's block machinery.

**Tech Stack:** numpy, pandas, pytest, nbformat/nbconvert, statsmodels (optional, for the block in the notebook), FRED data through `load_fred_monthly`.

**Spec:** `docs/superpowers/specs/2026-09-19-yield-curve-design.md`

## Global Constraints

- Python >= 3.8; tests on the anaconda interpreter and the durable venv311 (`/Users/mamadouthioub/Desktop/CopulaGenerator/venv311/bin/python -m pytest tests -q`); CI runs bare `pytest` with the `test,garch,blocks` extras.
- Yields in percent, maturities in years, continuous compounding; `lam = 0.7308` per year by default (Diebold-Li's 0.0609 per month).
- No data file committed; the FRED cache is git-ignored. The article-3 engine is untouched. Commit locally only.

---

### Task 1: `NelsonSiegel`, `PCACurve` and pricing in `yieldcurve.py`

**Files:**
- Create: `cvinemarketgen/yieldcurve.py`
- Create: `tests/test_yieldcurve.py`
- Modify: `cvinemarketgen/__init__.py` (export `NelsonSiegel`, `PCACurve`, `curve_returns`, `bond_price`, `par_yield`)

**Interfaces:**
- Produces: `NelsonSiegel(lam=0.7308)` with `loadings(tau)`, `fit(yields, maturities=None)`, `factors` (DataFrame level, slope, curvature), `tau`, `columns`, `rmse` (Series), `lam`, `curve(factors, tau=None)` (DataFrame -> DataFrame, Paths -> Paths), `to_dict/from_dict`; `PCACurve(n_components=3)` with `fit`, `factors`, `explained`, `curve(factors)`; functions `yield_at(F, tau, ns)`, `discount(y, tau)`, `bond_price(F, coupon, maturity, ns, freq=2)`, `par_yield(F, maturity, ns, freq=2)`, `zero_return(F0, F1, tau, ns, dt=1/12)`, `constant_maturity_return(F0, F1, tau, ns, dt=1/12, freq=2)`, `curve_returns(P, tau, ns, F0, kind='par', dt=1/12, freq=2)` -> `Paths`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_yieldcurve.py
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.yieldcurve import (NelsonSiegel, PCACurve, yield_at, discount, bond_price, par_yield,
                                        zero_return, constant_maturity_return, curve_returns)
from cvinemarketgen.paths import Paths

TAU = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


def _synthetic(n=300, seed=0, noise=0.0002):
    rng = np.random.default_rng(seed)
    level = 4.0 + np.cumsum(0.1 * rng.standard_normal(n)); slope = -1.5 + np.cumsum(0.1 * rng.standard_normal(n)); curv = -2.0 + np.cumsum(0.15 * rng.standard_normal(n))
    F = pd.DataFrame({'level': level, 'slope': slope, 'curvature': curv}, index=pd.period_range('2000-01', periods=n, freq='M'))
    ns = NelsonSiegel(0.7308)
    Y = pd.DataFrame(F.values @ ns.loadings(TAU).T + noise * 100 * rng.standard_normal((n, len(TAU))), index=F.index, columns=[str(t) for t in TAU])
    return F, Y


def test_loadings_shape_and_limits():
    ns = NelsonSiegel(0.7308)
    L = ns.loadings(TAU)
    assert L.shape == (8, 3) and np.allclose(L[:, 0], 1.0)
    small = ns.loadings([1e-6])[0]
    assert abs(small[1] - 1.0) < 1e-4 and abs(small[2]) < 1e-4
    big = ns.loadings([1000.0])[0]
    assert abs(big[1]) < 1e-2 and abs(big[2]) < 1e-2


def test_fit_recovers_factors_and_curve():
    F, Y = _synthetic()
    ns = NelsonSiegel(0.7308).fit(Y)
    assert list(ns.factors.columns) == ['level', 'slope', 'curvature'] and ns.tau == TAU and ns.columns == list(Y.columns)
    assert np.abs(ns.factors.values - F.values).max() < 1e-1 and (ns.rmse < 0.03).all()
    back = ns.curve(ns.factors)
    assert list(back.columns) == list(Y.columns) and np.abs(back.values - Y.values).max() < 0.1
    assert ns.curve(ns.factors.iloc[:5], tau=[4.0, 15.0]).shape == (5, 2)
    auto = NelsonSiegel('auto').fit(Y)
    assert 0.2 <= auto.lam <= 2.0
    ns2 = NelsonSiegel.from_dict(ns.to_dict())
    assert ns2.lam == ns.lam and ns2.tau == ns.tau and np.allclose(ns2.curve(ns.factors.iloc[:3]).values, back.values[:3])


def test_pca_curve():
    F, Y = _synthetic()
    pc = PCACurve(3).fit(Y)
    assert list(pc.factors.columns) == ['pc1', 'pc2', 'pc3'] and abs(pc.explained.sum() - 1) < 1e-9 or pc.explained.sum() < 1
    assert pc.explained.iloc[:3].sum() > 0.99
    assert np.abs(pc.curve(pc.factors).values - Y.values).max() < 0.05


def test_pricing_on_a_flat_curve():
    ns = NelsonSiegel(0.7308)
    F = np.array([[5.0, 0.0, 0.0]])                        # flat curve at 5 percent
    assert np.allclose(yield_at(F, [1.0, 10.0], ns), 5.0)
    assert np.allclose(discount(5.0, 2.0), np.exp(-0.10))
    c = par_yield(F, 10.0, ns, freq=2)
    assert np.allclose(bond_price(F, c, 10.0, ns, freq=2), 100.0, atol=1e-8)
    r0 = zero_return(F, F, 10.0, ns, dt=1 / 12)
    assert np.allclose(r0, np.exp(0.05 / 12) - 1, atol=1e-10)
    F_up = np.array([[5.0, -1.0, 0.0]])                    # upward sloping: 1-year below 10-year
    r = constant_maturity_return(F_up, F_up, 10.0, ns, dt=1 / 12)
    assert r > 0 and r > constant_maturity_return(F, F, 10.0, ns, dt=1 / 12) - 1e-12


def test_curve_returns_along_paths():
    ns = NelsonSiegel(0.7308)
    F0 = np.array([4.0, -1.0, -2.0])
    P = Paths(np.tile(F0, (5, 12, 1)), ['level', 'slope', 'curvature'])
    R = curve_returns(P, [2.0, 10.0], ns, F0, kind='par')
    assert isinstance(R, Paths) and R.assets == ['2y', '10y'] and R.array.shape == (5, 12, 2)
    assert np.allclose(R.array[0], R.array[4]) and np.allclose(R.array[:, 0, :], R.array[:, 5, :])
    Rz = curve_returns(P, [2.0], ns, F0, kind='zero')
    assert np.allclose(Rz.array[:, 0, 0], zero_return(F0[None], F0[None], 2.0, ns)[0])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_yieldcurve.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cvinemarketgen.yieldcurve'`.

- [ ] **Step 3: Implement `cvinemarketgen/yieldcurve.py`**

```python
# -*- coding: utf-8 -*-
"""
The yield curve as three factors and fixed income priced from it.

Nelson-Siegel: y_t(tau) = L_t + S_t (1 - e^{-lam tau}) / (lam tau) + C_t ((1 - e^{-lam tau}) / (lam tau) - e^{-lam tau}),
the factors estimated by least squares date by date (Diebold and Li, 2006), with
principal components as the benchmark. The factors are three observed series that
the market filters as a block; curves at any maturity, discount factors, bond
prices and the returns of zero-coupon or constant-maturity par bonds follow from
simulated factor paths. Yields in percent, maturities in years, continuous
compounding.
"""
import numpy as np
import pandas as pd

from .paths import Paths

FACTOR_NAMES = ['level', 'slope', 'curvature']


class NelsonSiegel:
    """Nelson-Siegel curve with decay ``lam`` per year (``'auto'``: grid search on the total RMSE)."""

    def __init__(self, lam=0.7308):
        self.lam = lam
        self.tau = None
        self.columns = None
        self.factors = None
        self.rmse = None

    @staticmethod
    def _loadings(tau, lam):
        tau = np.asarray(tau, float)
        x = lam * tau
        f1 = (1.0 - np.exp(-x)) / x
        return np.column_stack([np.ones_like(tau), f1, f1 - np.exp(-x)])

    def loadings(self, tau):
        """Loadings ``(len(tau), 3)`` of level, slope and curvature at maturities ``tau`` (years)."""
        return self._loadings(tau, self.lam)

    @staticmethod
    def _tau_of(columns, maturities):
        if maturities is not None:
            return [float(maturities[c]) for c in columns]
        out = []
        for c in columns:
            s = str(c).upper().replace('DGS', '').replace('Y', '')
            out.append(float(s) / (12.0 if str(c).upper().endswith('MO') else 1.0) if not str(c).upper().endswith('MO') else float(s.replace('MO', '')) / 12.0)
        return out

    def fit(self, yields, maturities=None):
        """Factors by least squares per date; ``maturities`` maps column names to years when the names are not numbers."""
        Y = pd.DataFrame(yields).astype(float)
        self.columns = list(Y.columns)
        self.tau = self._tau_of(self.columns, maturities)
        if self.lam == 'auto':
            grid = np.linspace(0.2, 2.0, 37)
            tot = []
            for g in grid:
                L = self._loadings(self.tau, g)
                F = np.linalg.lstsq(L, Y.values.T, rcond=None)[0]
                tot.append(np.sqrt(((Y.values - F.T @ L.T) ** 2).mean()))
            self.lam = float(grid[int(np.argmin(tot))])
        L = self.loadings(self.tau)
        F = np.linalg.lstsq(L, Y.values.T, rcond=None)[0].T
        self.factors = pd.DataFrame(F, index=Y.index, columns=FACTOR_NAMES)
        fit = F @ L.T
        self.rmse = pd.Series(np.sqrt(((Y.values - fit) ** 2).mean(axis=0)), index=self.columns)
        return self

    def curve(self, factors, tau=None):
        """Yields at ``tau`` (default: the fitted maturities) from factors: DataFrame -> DataFrame, Paths -> Paths."""
        t = self.tau if tau is None else [float(x) for x in tau]
        names = self.columns if tau is None else [f'{x:g}y' for x in t]
        L = self.loadings(t)
        if isinstance(factors, Paths):
            return Paths(factors.array @ L.T, names, layer='yields')
        F = pd.DataFrame(factors)
        return pd.DataFrame(F.values @ L.T, index=F.index, columns=names)

    def to_dict(self):
        return {'lam': self.lam, 'tau': self.tau, 'columns': self.columns,
                'rmse': None if self.rmse is None else self.rmse.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['lam'])
        m.tau, m.columns = d['tau'], d['columns']
        if d.get('rmse') is not None:
            m.rmse = pd.Series(d['rmse'], index=m.columns)
        return m


class PCACurve:
    """Principal components of the yields' levels: the benchmark to Nelson-Siegel, same interface."""

    def __init__(self, n_components=3):
        self.n = int(n_components)
        self.columns = self.mean = self.vectors = self.factors = self.explained = None

    def fit(self, yields):
        Y = pd.DataFrame(yields).astype(float)
        self.columns = list(Y.columns)
        self.mean = Y.values.mean(axis=0)
        w, v = np.linalg.eigh(np.cov(Y.values.T))
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        v = v * np.sign(v[-1, :])                           # last maturity loads positively: level up means yields up
        self.explained = pd.Series(w / w.sum(), index=[f'pc{i + 1}' for i in range(len(w))])
        self.vectors = v[:, :self.n]
        self.factors = pd.DataFrame((Y.values - self.mean) @ self.vectors, index=Y.index, columns=[f'pc{i + 1}' for i in range(self.n)])
        return self

    def curve(self, factors):
        """Yields at the fitted maturities from component scores (DataFrame or Paths)."""
        if isinstance(factors, Paths):
            return Paths(factors.array @ self.vectors.T + self.mean, self.columns, layer='yields')
        F = pd.DataFrame(factors)
        return pd.DataFrame(F.values @ self.vectors.T + self.mean, index=F.index, columns=self.columns)


# ---- pricing ---------------------------------------------------------------------
def yield_at(F, tau, ns):
    """Yields (percent) at maturities ``tau`` from factor rows ``F`` of shape ``(..., 3)``: shape ``(..., len(tau))``."""
    F = np.asarray(F, float)
    return F @ ns.loadings(np.atleast_1d(tau)).T


def discount(y, tau):
    """Discount factor ``exp(-tau y / 100)`` for a yield in percent and a maturity in years."""
    return np.exp(-np.asarray(tau, float) * np.asarray(y, float) / 100.0)


def _cash_times(maturity, freq, dt=0.0):
    k = np.arange(1, int(round(maturity * freq)) + 1)
    return k / freq - dt


def bond_price(F, coupon, maturity, ns, freq=2):
    """Price per 100 of face of a bond paying ``coupon`` percent a year ``freq`` times a year, on the curve of ``F``."""
    t = _cash_times(maturity, freq)
    P = discount(yield_at(F, t, ns), t)
    return (coupon / freq) * P.sum(axis=-1) + 100.0 * P[..., -1]


def par_yield(F, maturity, ns, freq=2):
    """Coupon (percent a year) that prices a bond of ``maturity`` at par on the curve of ``F``."""
    t = _cash_times(maturity, freq)
    P = discount(yield_at(F, t, ns), t)
    return freq * 100.0 * (1.0 - P[..., -1]) / P.sum(axis=-1)


def zero_return(F0, F1, tau, ns, dt=1.0 / 12):
    """Return of a zero of maturity ``tau`` bought on curve ``F0`` and sold ``dt`` years later on curve ``F1``."""
    p0 = discount(yield_at(F0, [tau], ns)[..., 0], tau)
    p1 = discount(yield_at(F1, [tau - dt], ns)[..., 0], tau - dt)
    return p1 / p0 - 1.0


def constant_maturity_return(F0, F1, tau, ns, dt=1.0 / 12, freq=2):
    """
    Return of a par bond of maturity ``tau`` issued on curve ``F0`` (coupon = its par
    yield) and valued ``dt`` years later on curve ``F1`` with its remaining cash flows
    (dirty price, no coupon paid within ``dt``): carry, roll-down and price change.
    """
    c = par_yield(F0, tau, ns, freq)
    t = _cash_times(tau, freq, dt)
    P = discount(yield_at(F1, t, ns), t)
    price1 = (np.asarray(c)[..., None] / freq * P).sum(axis=-1) + 100.0 * P[..., -1]
    return price1 / 100.0 - 1.0


def curve_returns(P, tau, ns, F0, kind='par', dt=1.0 / 12, freq=2):
    """
    Monthly returns of bonds of maturities ``tau`` along simulated factor paths ``P``
    (a ``Paths`` of level, slope, curvature), starting from the last observed factors
    ``F0``; ``kind`` ``'par'`` (constant-maturity par bond) or ``'zero'``. Returns a ``Paths``.
    """
    A = P.array
    prev = np.concatenate([np.tile(np.asarray(F0, float), (A.shape[0], 1, 1)), A[:, :-1]], axis=1)
    out = np.empty((A.shape[0], A.shape[1], len(tau)))
    for j, t in enumerate(tau):
        f = constant_maturity_return if kind == 'par' else zero_return
        out[:, :, j] = f(prev, A, float(t), ns, dt, freq) if kind == 'par' else f(prev, A, float(t), ns, dt)
    return Paths(out, [f'{float(t):g}y' for t in tau], layer='returns')
```

`__init__.py`: `from .yieldcurve import NelsonSiegel, PCACurve, curve_returns, bond_price, par_yield` and the five names in `__all__`.

- [ ] **Step 4: Run the tests on both stacks**

Run: `python -m pytest tests/test_yieldcurve.py -q` and the venv311 one.
Expected: 5 passed on both.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/yieldcurve.py cvinemarketgen/__init__.py tests/test_yieldcurve.py
git commit -m "feat: Nelson-Siegel and PCA curve factors, bond pricing and curve returns"
```

---

### Task 2: Notebook 12, the curve as a block and fixed income along the paths

**Files:**
- Create: `examples/12_yield_curve_fixed_income.ipynb`
- Modify: `README.md` (row 12, "Eleven" -> "Twelve"), `docs/examples.rst`

**Interfaces:**
- Consumes: `load_fred_monthly`, `load_etf_monthly`, `Targets`, `CVineMarket`, `NelsonSiegel`, `PCACurve`, `curve_returns`, `iid_tests`.

- [ ] **Step 1: Build the cells** (import cell of notebook 11)

1. md: title "12 — The yield curve and fixed income"; three sentences: the curve reduced to level, slope and curvature; the factors filtered as a block; simulated curves priced into bond returns; runtime about 5 minutes.
2. md "Load the curve" / code: `ids = ['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']`; `Y = load_fred_monthly(ids, start='1993-10', cache=...)`; `tau = {c: float(c[3:]) for c in ids}`; plot of the 1, 10 and 30-year yields.
3. md "Nelson-Siegel factors" / code: `ns = NelsonSiegel(0.7308).fit(Y, maturities=tau)`; loadings plot on `np.linspace(0.25, 30, 100)`; `ns.factors.tail()`; `ns.rmse.mul(100).round(1).rename('RMSE (bp)')`; plot of level vs DGS10 and slope vs DGS1 − DGS10.
4. md "PCA benchmark" / code: `pc = PCACurve(3).fit(Y)`; `pc.explained.round(4).head(4)`; loadings plot of the three vectors against maturities; sentence: shapes are level, slope, curvature.
5. md "The factors as a block, SPY alongside" / code: `spy = load_etf_monthly(['SPY'], cache=...)`; `data = ns.factors.join(spy, how='inner')`; Johansen table (q = 1, 2) as in notebook 10; `t = Targets.from_history(data)`; `cv = CVineMarket(t, central='SPY', families='auto', dynamics={('level', 'slope', 'curvature'): 'vecm(q=1)', 'SPY': 'ar1-garch'}).fit()`; `cv.dynamics_report`; residual tests per factor; `cv.edges`.
6. md "Simulated curves" / code: `P = cv.simulate_paths(1000, 24, seed=1)`; `PF = Paths(P.array[:, :, [P.assets.index(c) for c in ['level', 'slope', 'curvature']]], ['level', 'slope', 'curvature'])`; `C = ns.curve(PF)`; fan of the 10-year yield and of the 2s10s spread; table of the curve's quantiles at months 1, 12, 24 for 2, 10, 30 years.
7. md "Fixed income along the paths" / code: `F0 = ns.factors.iloc[-1].values`; `R = curve_returns(PF, [2, 5, 10, 30], ns, F0, kind='par')`; `R.summary()` annualized; `R.terminal().quantile([0.05, 0.5, 0.95])`; correlation of the monthly 10-year return with SPY's simulated return.
8. md "Against the history and the duration proxy" / code: historical constant-maturity returns from consecutive fitted factors: `Fh = ns.factors.values; Rh = pd.DataFrame({f'{t}y': constant_maturity_return(Fh[:-1], Fh[1:], float(t), ns) for t in [2, 5, 10, 30]}, index=ns.factors.index[1:])`; annualized mean and vol of `Rh` next to `R.summary()`; the duration proxy for the 10-year, `carry - D * change of the 10-year yield` with `D = 8`, against `Rh['10y']`: correlation and RMSE.
9. md "Save" / code: `cv.save('curve_market.json')`; `json.dump(ns.to_dict(), open('curve_ns.json', 'w'))`.
10. md closing: any curve with a parametric loading fits; credit curves and the arbitrage-free adjustment are next.

- [ ] **Step 2: Execute and inspect** (`jupyter nbconvert --execute --inplace`, timeout 1800): no error cell; RMSE under 15 bp; PCA three components above 99 percent; the block's rank 1; the 10-year historical return correlated above 0.95 with the duration proxy; simulated annualized volatilities of the same order as the historical ones.

- [ ] **Step 3: README and docs** (row 12 "the Treasury curve as Nelson-Siegel factors filtered as a block, simulated curves, constant-maturity bond returns", counts to "Twelve", toctree `examples/12_yield_curve_fixed_income`).

- [ ] **Step 4: Commit**

```bash
git add examples/12_yield_curve_fixed_income.ipynb README.md docs/examples.rst
git commit -m "docs: notebook 12, the yield curve as a block and fixed income along the paths"
```

---

### Task 3: Docs, changelog, export script and the paper

**Files:**
- Modify: `docs/userguide.rst` (section "The yield curve and fixed income" after "The structural layer"), `docs/api.rst` (section "Yield curve", `cvinemarketgen.yieldcurve`), `docs/method.rst` (bullet), `CHANGELOG.md` (bullets), `README.md` (paragraph)
- Create: `OldVersion/overleaf-path-simulation/paper_yieldcurve_export.py`, `figs/yieldcurve/`
- Modify: `OldVersion/overleaf-path-simulation/paper-yieldcurve.tex`

- [ ] **Step 1: User guide**

```rst
The yield curve and fixed income
--------------------------------

.. code-block:: python

   Y = load_fred_monthly(['DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30'], start='1993-10')
   ns = NelsonSiegel(0.7308).fit(Y, maturities={c: float(c[3:]) for c in Y.columns})
   data = ns.factors.join(spy_returns, how='inner')
   cv = CVineMarket(Targets.from_history(data), central='SPY', families='auto',
                    dynamics={('level', 'slope', 'curvature'): 'vecm', 'SPY': 'ar1-garch'}).fit()
   P = cv.simulate_paths(1000, 24, seed=1)
   PF = Paths(P.array[:, :, :3], ['level', 'slope', 'curvature'])
   curves = ns.curve(PF)                                       # yields at the fitted maturities
   R = curve_returns(PF, [2, 10, 30], ns, ns.factors.iloc[-1].values)   # constant-maturity bond returns

The curve is three observed factors, level, slope and curvature, estimated by
least squares per date on the Nelson-Siegel loadings (``PCACurve`` is the
benchmark). They enter the market as a block like any other variables in
levels; simulated factors give curves at any maturity through ``curve``, and
``bond_price``, ``par_yield`` and ``curve_returns`` price zero-coupon and
constant-maturity par bonds along the paths.
```

- [ ] **Step 2: API, method, changelog, README** (`.. automodule:: cvinemarketgen.yieldcurve` `:members: NelsonSiegel, PCACurve, yield_at, discount, bond_price, par_yield, zero_return, constant_maturity_return, curve_returns`; method bullet "Yield curve: Nelson-Siegel factors as a block, deterministic curves and bond returns from the paths"; changelog bullets for `yieldcurve` and notebook 12; README paragraph with the four lines of the user guide's example).

- [ ] **Step 3: Export script** `paper_yieldcurve_export.py` (pattern of `paper_structural_export.py`, `OUT = figs/yieldcurve`): `tab_yields_moments`, `tab_ns_rmse` (RMSE by maturity in bp, for `lam = 0.7308` and `lam = 'auto'`), `tab_factor_moments`, `tab_pca_explained`, `tab_johansen`, `tab_block_iid`, `tab_residual_corr` (innovations of the three factors and SPY), `tab_families`, `tab_curve_quantiles` (2, 10, 30 years at months 1, 12, 24), `tab_bond_returns` (annualized mean, volatility, skewness, kurtosis of the 2, 5, 10, 30-year constant-maturity returns: history and paths), `tab_terminal` (one-year and two-year cumulative returns' quantiles), `tab_proxy` (correlation and RMSE of the duration proxy against the exact 10-year return), figures `fig_loadings.png` (NS and PCA), `fig_factors.png`, `fig_fan_10y.png`, `fig_fan_spread.png`, `fig_bond_returns.png` (median cumulative return of the four maturities), `curve_market.json`.

- [ ] **Step 4: Write `paper-yieldcurve.tex`** in the style of the other short papers: introduction (a curve is a function of a few factors; pricing needs a curve, not a yield; the factors as a block, the curve as a deterministic child); the generator in brief; level, slope and curvature: PCA and Nelson-Siegel (both models with equations, the estimation per date, the decay, the fit); dynamic Nelson-Siegel as a filter (the block on the factors, rank and lags, the residual layer, what the vine carries: the correlated innovations); from simulated curves to bond prices and index returns (the pricing formulas, the two return definitions, the duration proxy as their first-order approximation); example (the tables); conclusion (arbitrage-free adjustment, credit curves, the proxy retired).

- [ ] **Step 5: Compile, run the suites, commit, re-zip, stage**

```bash
git add docs/userguide.rst docs/api.rst docs/method.rst CHANGELOG.md README.md
git commit -m "docs: the yield curve in the user guide, API, method, changelog and README"
```

---

## Self-review

- **Spec coverage:** NelsonSiegel (loadings, fit, auto lam, curve for DataFrame and Paths, JSON), PCACurve, all pricing functions (Task 1); the block integration, notebook, comparison with the history and the proxy (Task 2); docs, export, paper (Task 3).
- **Placeholders:** none.
- **Type consistency:** `curve_returns(P, tau, ns, F0, kind, dt, freq)` matches its use in the notebook and tests; `constant_maturity_return(F0, F1, tau, ns, dt, freq)` and `zero_return(F0, F1, tau, ns, dt)` signatures match `curve_returns`' dispatch; `yield_at` returns `(..., len(tau))` and `discount` broadcasts.
