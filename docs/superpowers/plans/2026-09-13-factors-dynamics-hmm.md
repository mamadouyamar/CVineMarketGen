# CVineMarketGen 0.3.0 Implementation Plan: factors, dynamics selection, HMM, paper

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map assets onto simulated macro factors through regression betas, choose per-asset daily dynamics (GARCH family or Gaussian HMM) by i.i.d. tests then BIC with a bootstrap goodness-of-fit check, and write the companion paper.

**Architecture:** New modules `factors.py`, `hmm.py`, `selection.py`; `dynamics.py` gains a univariate-model interface (`GarchFamily`, `GaussianHMM`) and a per-asset container (`AssetDynamics`); `markets.py` accepts `dynamics='auto'`, a per-asset dict, or an `AssetDynamics`. The article-3 engine is untouched.

**Tech Stack:** numpy, scipy, pandas, `arch` (optional, GARCH family), pyvinecopulib 0.6/0.7, pytest, nbformat + nbconvert for the notebooks, Sphinx docs.

**Spec:** `docs/superpowers/specs/2026-09-13-factors-dynamics-hmm-design.md`

## Global Constraints

- Python >= 3.8; run tests with the anaconda interpreter (`python -m pytest tests -q`) and, before the final commit of each task touching the engine interface, with the venv311 stack too (`~/Desktop/CopulaGenerator/venv311/bin/python -m pytest tests -q`).
- `arch` stays optional: import it inside functions, raise `ImportError` with the pip hint.
- No file under `data/` other than `jpm_ltcma_2024.csv` is committed; caches are git-ignored (`data/factors_cache.csv`, `data/daily_cache.csv`, add `data/etf_cache.csv`).
- Kurtosis is raw (normal = 3) everywhere in the user layer.
- The user uploads to GitHub by hand; commit locally only, never push.
- Notation in docstrings and the paper: standardized residual `epsilon_t`, Rosenblatt uniform `v_t` (article 3 uses `z` for the exceedance threshold).
- Deviation from the spec, decided while planning: the HMM initial distribution is fixed uniform (`1/K`), as in GenHMM1d, so the reference test can match; `n_params = K(K-1) + 2K`.

---

### Task 1: Small-cap factor and ETF loader in `data.py`

**Files:**
- Modify: `cvinemarketgen/data.py` (FACTORS, `load_factor_data`, new `load_etf_monthly`)
- Modify: `.gitignore` (add `data/etf_cache.csv`)
- Test: `tests/test_data_offline.py` (new, no network)

**Interfaces:**
- Produces: `FACTORS = ['Equity DM', 'Equity EM', 'Real premia', 'Inflation', 'Credit', 'Commodity', 'Small cap']`; `load_etf_monthly(tickers, start='2006-03', end=None, cache='data/etf_cache.csv', refresh=False, verbose=True) -> DataFrame` (PeriodIndex 'M', decimal simple returns).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_offline.py
"""Offline checks of the data module: constants and cache handling, no download."""
import pandas as pd
from cvinemarketgen import data


def test_factors_has_small_cap():
    assert data.FACTORS[-1] == 'Small cap' and len(data.FACTORS) == 7


def test_load_factor_data_refreshes_old_cache(tmp_path, monkeypatch):
    old = pd.DataFrame({c: [0.01, 0.02] for c in data.FACTORS[:-1]},
                       index=pd.PeriodIndex(['2020-01', '2020-02'], freq='M'))
    p = tmp_path / 'factors_cache.csv'
    old.to_csv(p)
    called = {}

    def fake_download(start, end, cache, D, verbose):
        called['yes'] = True
        return old.assign(**{'Small cap': [0.0, 0.0]})
    monkeypatch.setattr(data, '_download_factors', fake_download)
    df = data.load_factor_data(cache=str(p), verbose=False)
    assert called and list(df.columns) == data.FACTORS


def test_load_etf_monthly_from_cache(tmp_path):
    idx = pd.PeriodIndex(['2020-01', '2020-02', '2020-03'], freq='M')
    pd.DataFrame({'SPY': [0.01, -0.02, 0.03], 'TLT': [0.0, 0.01, -0.01]}, index=idx).to_csv(tmp_path / 'etf.csv')
    df = data.load_etf_monthly(['TLT', 'SPY'], cache=str(tmp_path / 'etf.csv'), verbose=False)
    assert list(df.columns) == ['TLT', 'SPY'] and isinstance(df.index, pd.PeriodIndex) and df.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_offline.py -q`
Expected: FAIL, `AttributeError: module 'cvinemarketgen.data' has no attribute '_download_factors'` and the FACTORS assertion fails.

- [ ] **Step 3: Implement**

In `cvinemarketgen/data.py`:

```python
FACTORS = ['Equity DM', 'Equity EM', 'Real premia', 'Inflation', 'Credit', 'Commodity', 'Small cap']
```

Add `Small cap  U.S. small minus big  Fama-French, F-F_Research_Data_Factors, SMB` to the module docstring table. Split `load_factor_data` so the download is a separate function:

```python
def _download_factors(start, end, cache, D, verbose):
    dm = fama_french_monthly('Developed_3_Factors')
    em = fama_french_monthly('Emerging_5_Factors')
    us = fama_french_monthly('F-F_Research_Data_Factors')
    rf = dm['RF'] / 100.0
    equity_dm = dm['Mkt-RF'] / 100.0
    equity_em = (em['Mkt-RF'] - dm['Mkt-RF']) / 100.0
    small_cap = us['SMB'] / 100.0
    breakeven = fred_series('T10YIE', start='2003-01-01')
    real_yield = fred_series('DFII10', start='2003-01-01')
    baa_spread = fred_series('BAA10Y', start='1986-01-01')
    inflation = _yield_factor(breakeven, D['Inflation'], sign=+1)
    real_premia = _yield_factor(real_yield, D['Real premia'], sign=-1)
    credit = _yield_factor(baa_spread, D['Credit'], sign=-1)
    dbc = yahoo_monthly_adjclose('DBC')
    commodity = (dbc.pct_change() - rf).dropna()
    df = pd.concat({'Equity DM': equity_dm, 'Equity EM': equity_em, 'Real premia': real_premia,
                    'Inflation': inflation, 'Credit': credit, 'Commodity': commodity, 'Small cap': small_cap}, axis=1)
    df = df[FACTORS].dropna()
    df = df.loc[start:end] if end else df.loc[start:]
    df.index.name = 'month'
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        df.to_csv(cache)
    if verbose:
        print(f'factor data downloaded: {df.shape[0]} months, {df.index.min()} to {df.index.max()}'
              + (f', cached to {cache}' if cache else ''))
    return df


def load_factor_data(start='2006-03', end=None, cache='data/factors_cache.csv',
                     durations=None, refresh=False, verbose=True):
    """
    Monthly excess returns (decimal) of the seven market factors, PeriodIndex('M'),
    columns in the order of FACTORS. Downloaded from Fama-French, FRED and Yahoo
    Finance, then cached to `cache` (set refresh=True to download again; a cache
    from an earlier version without the small-cap column is refreshed automatically).
    """
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq='M')
        if set(FACTORS) <= set(df.columns):
            if verbose:
                print(f'factor data read from cache {cache}: {df.shape[0]} months, {df.index.min()} to {df.index.max()}')
            df = df[FACTORS]
            return df.loc[start:end] if end else df.loc[start:]
        if verbose:
            print(f'cache {cache} lacks some factors, downloading again')
    D = dict(DURATIONS)
    if durations:
        D.update(durations)
    return _download_factors(start, end, cache, D, verbose)
```

Add after `load_daily_returns`:

```python
def load_etf_monthly(tickers, start='2006-03', end=None, cache='data/etf_cache.csv', refresh=False, verbose=True):
    """
    Monthly simple returns (decimal) of Yahoo Finance tickers from adjusted
    closes, PeriodIndex('M'), aligned on common months, cached as a CSV.
    """
    tickers = list(tickers)
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq='M')
        if set(tickers) <= set(df.columns):
            df = df.loc[start:end, tickers] if end else df.loc[start:, tickers]
            df = df.dropna()
            if verbose:
                print(f'ETF returns read from cache {cache}: {df.shape[0]} months, {df.index.min()} to {df.index.max()}')
            return df
    prices = pd.concat([yahoo_monthly_adjclose(t) for t in tickers], axis=1).dropna()
    df = prices.pct_change().dropna()
    df = df.loc[start:end] if end else df.loc[start:]
    df.index.name = 'month'
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        df.to_csv(cache)
    if verbose:
        print(f'ETF returns downloaded: {df.shape[0]} months, {df.index.min()} to {df.index.max()}'
              + (f', cached to {cache}' if cache else ''))
    return df
```

Append `data/etf_cache.csv` to `.gitignore`. Export `load_etf_monthly` in `cvinemarketgen/__init__.py` (add to the `from .data import ...` line and to `__all__`).

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests -q`
Expected: all pass (21).

- [ ] **Step 5: Refresh the real cache once and check the seventh column**

Run: `python -c "from cvinemarketgen import load_factor_data; d=load_factor_data(refresh=True); print(d.tail(3)); print(d.corr().round(2))"`
Expected: seven columns, `Small cap` present, correlation of `Small cap` with `Equity DM` between 0.1 and 0.6.

- [ ] **Step 6: Commit**

```bash
git add cvinemarketgen/data.py cvinemarketgen/__init__.py .gitignore tests/test_data_offline.py
git commit -m "feat: small-cap factor and monthly ETF loader"
```

---

### Task 2: `FactorModel` in `factors.py`

**Files:**
- Create: `cvinemarketgen/factors.py`
- Modify: `cvinemarketgen/__init__.py`
- Test: `tests/test_factors.py`

**Interfaces:**
- Consumes: `fit_johnson_su(skew, kurt, mean, vol)` and `johnson_su_sample(params, n, seed)` from `functions.py`; `Paths(array, assets)` from `paths.py`.
- Produces: `FactorModel(asset_returns, factor_returns, nw_lags=None).fit()`; attributes `alpha, beta, se, tstat, r2, resid, resid_vol, resid_params, report, factors, assets, n_obs`; `implied_mean(factor_mean) -> Series`; `simulate(F, residuals=True, seed=None)`; `save(path)`, `FactorModel.load(path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_factors.py
import json
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen import FactorModel, Paths


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame(rng.standard_normal((n, 3)) * 0.03, columns=['f1', 'f2', 'f3'],
                     index=pd.period_range('2000-01', periods=n, freq='M'))
    B = pd.DataFrame([[1.0, 0.5, 0.0], [0.2, -0.3, 0.8]], index=['A', 'B'], columns=F.columns)
    alpha = pd.Series([0.002, -0.001], index=['A', 'B'])
    eps = rng.standard_normal((n, 2)) * 0.01
    R = pd.DataFrame(alpha.values + F.values @ B.values.T + eps, columns=['A', 'B'], index=F.index)
    return R, F, B, alpha


def test_betas_recovered():
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    assert fm.beta.shape == (2, 3) and fm.n_obs == 240
    assert (np.abs(fm.beta - B) < 2 * fm.se[F.columns]).all().all()
    assert (np.abs(fm.alpha - alpha) < 2 * fm.se['alpha']).all()
    assert (fm.r2 > 0.85).all() and set(fm.report.columns) >= {'alpha', 'f1', 'f2', 'f3', 'R2', 'resid vol'}
    assert list(fm.resid_params.columns) == ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']


def test_simulate_shapes_and_no_residual_case():
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    Fs = F.iloc[:50]
    X0 = fm.simulate(Fs, residuals=False)
    assert X0.shape == (50, 2) and list(X0.columns) == ['A', 'B']
    assert np.allclose(X0.values, fm.alpha.values + Fs.values @ fm.beta.values.T)
    X1 = fm.simulate(Fs[['f3', 'f1', 'f2']], residuals=True, seed=1)     # columns in another order
    assert X1.shape == (50, 2) and not np.allclose(X1.values, X0.values)
    P = Paths(np.tile(F.values[:12], (7, 1, 1)), list(F.columns))
    PX = fm.simulate(P, seed=2)
    assert isinstance(PX, Paths) and PX.array.shape == (7, 12, 2) and PX.assets == ['A', 'B']


def test_implied_mean_and_roundtrip(tmp_path):
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    m = fm.implied_mean(pd.Series({'f1': 0.01, 'f2': 0.0, 'f3': -0.01}))
    assert np.allclose(m.values, fm.alpha.values + fm.beta.values @ np.array([0.01, 0.0, -0.01]))
    p = tmp_path / 'fm.json'; fm.save(str(p))
    assert json.load(open(p))['kind'] == 'FactorModel'
    fm2 = FactorModel.load(str(p))
    assert np.allclose(fm2.beta.values, fm.beta.values) and fm2.report.equals(fm.report)
    assert np.allclose(fm2.simulate(F.iloc[:5], seed=3).values, fm.simulate(F.iloc[:5], seed=3).values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_factors.py -q`
Expected: FAIL, `ImportError: cannot import name 'FactorModel'`.

- [ ] **Step 3: Implement `cvinemarketgen/factors.py`**

```python
# -*- coding: utf-8 -*-
"""
Assets on factors: a linear factor model fitted by least squares, whose betas
map simulated factor scenarios or paths (from a market on the factors) to
asset returns, with an optional independent Johnson SU residual per asset.

    r_t = alpha + beta' f_t + eps_t

Standard errors are Newey-West. Units and frequency are the data's.
"""
import json
import numpy as np
import pandas as pd

from .functions import fit_johnson_su, johnson_su_sample
from .paths import Paths


def _newey_west_se(X, e, lags):
    """Newey-West (Bartlett) standard errors of OLS coefficients; X (n, k), e (n,)."""
    n = X.shape[0]
    Xe = X * e[:, None]
    S = Xe.T @ Xe / n
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xe[l:].T @ Xe[:-l] / n
        S += w * (G + G.T)
    A = np.linalg.inv(X.T @ X / n)
    V = A @ S @ A / n
    return np.sqrt(np.diag(V))


class FactorModel:
    """
    Linear factor model of assets on factors, ``r = alpha + beta' f + eps``.

    Parameters
    ----------
    asset_returns : DataFrame
        One column per asset.
    factor_returns : DataFrame
        One column per factor, same frequency; aligned on the common index.
    nw_lags : int, optional
        Newey-West lags for the standard errors; default ``floor(4 (n/100)^(2/9))``.

    Notes
    -----
    After ``fit``: ``alpha`` (Series), ``beta`` (DataFrame assets x factors),
    ``se`` and ``tstat`` (DataFrames with an ``alpha`` column and one per factor),
    ``r2`` (Series), ``resid`` (DataFrame), ``resid_vol`` (Series), ``resid_params``
    (Johnson SU per asset, mean zero) and ``report`` (one table).
    """

    def __init__(self, asset_returns, factor_returns, nw_lags=None):
        R = pd.DataFrame(asset_returns).astype(float)
        F = pd.DataFrame(factor_returns).astype(float)
        idx = R.index.intersection(F.index)
        if len(idx) < 3 * (F.shape[1] + 1):
            raise ValueError('not enough common observations to fit the factor model')
        self.R = R.loc[idx]
        self.F = F.loc[idx]
        self.assets = list(self.R.columns)
        self.factors = list(self.F.columns)
        self.n_obs = len(idx)
        self.nw_lags = int(np.floor(4 * (self.n_obs / 100.0) ** (2.0 / 9.0))) if nw_lags is None else int(nw_lags)
        self.fitted = False

    def fit(self):
        X = np.column_stack([np.ones(self.n_obs), self.F.values])
        coef, *_ = np.linalg.lstsq(X, self.R.values, rcond=None)      # (K+1, N)
        E = self.R.values - X @ coef
        cols = ['alpha'] + self.factors
        self.alpha = pd.Series(coef[0], index=self.assets)
        self.beta = pd.DataFrame(coef[1:].T, index=self.assets, columns=self.factors)
        se = np.vstack([_newey_west_se(X, E[:, j], self.nw_lags) for j in range(len(self.assets))])
        self.se = pd.DataFrame(se, index=self.assets, columns=cols)
        self.tstat = pd.DataFrame(coef.T / se, index=self.assets, columns=cols)
        tss = ((self.R.values - self.R.values.mean(0)) ** 2).sum(0)
        self.r2 = pd.Series(1.0 - (E ** 2).sum(0) / tss, index=self.assets)
        self.resid = pd.DataFrame(E, index=self.R.index, columns=self.assets)
        self.resid_vol = self.resid.std(ddof=X.shape[1])
        rows = {}
        for a in self.assets:
            e = self.resid[a]
            rows[a] = fit_johnson_su(e.skew(), e.kurtosis() + 3.0, mean=0.0, vol=float(self.resid_vol[a]))
        self.resid_params = pd.DataFrame(rows).T[['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']]
        rep = pd.concat([self.alpha.rename('alpha'), self.beta], axis=1)
        for c in cols:
            rep[f't({c})'] = self.tstat[c]
        rep['R2'] = self.r2
        rep['resid vol'] = self.resid_vol
        self.report = rep
        self.fitted = True
        return self

    def implied_mean(self, factor_mean):
        """``alpha + beta @ factor_mean``: expected asset returns for a view on the factors."""
        self._check()
        m = pd.Series(factor_mean).astype(float).loc[self.factors]
        return self.alpha + self.beta.values @ m.values

    def simulate(self, F, residuals=True, seed=None):
        """
        Asset returns from factor scenarios: DataFrame (n x K) -> DataFrame (n x N);
        :class:`~cvinemarketgen.paths.Paths` -> ``Paths``. With ``residuals``,
        an independent Johnson SU residual is added per asset.
        """
        self._check()
        if isinstance(F, Paths):
            n, h, K = F.array.shape
            flat = pd.DataFrame(F.array.reshape(n * h, K), columns=F.assets)
            X = self.simulate(flat, residuals=residuals, seed=seed)
            return Paths(X.values.reshape(n, h, len(self.assets)), self.assets)
        Fd = pd.DataFrame(F).astype(float).loc[:, self.factors]
        X = self.alpha.values + Fd.values @ self.beta.values.T
        if residuals:
            rng = np.random.default_rng(seed)
            for j, a in enumerate(self.assets):
                X[:, j] += johnson_su_sample(self.resid_params.loc[a].to_dict(), len(Fd), seed=int(rng.integers(2 ** 31)))
        return pd.DataFrame(X, index=Fd.index, columns=self.assets)

    def save(self, path):
        self._check()
        d = {'kind': 'FactorModel', 'assets': self.assets, 'factors': self.factors, 'n_obs': self.n_obs,
             'nw_lags': self.nw_lags, 'alpha': self.alpha.to_dict(), 'beta': self.beta.to_dict(orient='index'),
             'se': self.se.to_dict(orient='index'), 'tstat': self.tstat.to_dict(orient='index'),
             'r2': self.r2.to_dict(), 'resid_vol': self.resid_vol.to_dict(),
             'resid_params': self.resid_params.to_dict(orient='index'), 'report': self.report.to_dict(orient='index')}
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls.__new__(cls)
        m.assets, m.factors, m.n_obs, m.nw_lags = d['assets'], d['factors'], d['n_obs'], d['nw_lags']
        m.R = m.F = m.resid = None
        m.alpha = pd.Series(d['alpha']).loc[m.assets]
        m.beta = pd.DataFrame(d['beta']).T.loc[m.assets, m.factors]
        cols = ['alpha'] + m.factors
        m.se = pd.DataFrame(d['se']).T.loc[m.assets, cols]
        m.tstat = pd.DataFrame(d['tstat']).T.loc[m.assets, cols]
        m.r2 = pd.Series(d['r2']).loc[m.assets]
        m.resid_vol = pd.Series(d['resid_vol']).loc[m.assets]
        m.resid_params = pd.DataFrame(d['resid_params']).T.loc[m.assets, ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']]
        rep = pd.DataFrame(d['report']).T.loc[m.assets]
        m.report = rep[['alpha'] + m.factors + [f't({c})' for c in cols] + ['R2', 'resid vol']].astype(float)
        m.fitted = True
        return m

    def _check(self):
        if not self.fitted:
            raise RuntimeError('call fit() first')
```

Export in `__init__.py`: `from .factors import FactorModel` and add `'FactorModel'` to `__all__`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_factors.py -q`
Expected: 3 passed. If `fm2.report.equals(fm.report)` fails on dtype, compare with `np.allclose(fm2.report.values, fm.report.values)` and keep the column-order assertion.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/factors.py cvinemarketgen/__init__.py tests/test_factors.py
git commit -m "feat: FactorModel, assets on simulated factors"
```

---

### Task 3: Notebook 08, assets on macro factors

**Files:**
- Create: `examples/08_assets_on_macro_factors.ipynb` (built by a scratchpad script with `nbformat`, executed with `jupyter nbconvert --to notebook --execute --inplace`)
- Modify: `examples/06_macro_factors.ipynb` (seven factors: only the text cells that say "six factors", and the asset lists; rerun)
- Modify: `README.md` notebook table (rows 06 and 08), `docs/examples.rst` (add 08)

**Interfaces:**
- Consumes: `load_factor_data`, `load_etf_monthly`, `Targets`, `CVineMarket`, `FactorModel`.

- [ ] **Step 1: Build the notebook cells** (scratchpad `build_nb08.py`, same helper as the earlier `build_tutorials.py`: `md(text)`, `code(src)`, Colab badge first cell)

Cells, in order:

1. md: title "Assets on macro factors", three sentences: factors are simulated by the C-vine, an asset is `alpha + beta' f + eps`, so a view on the factors becomes a distribution for every asset through its betas. Colab badge.
2. code: imports and `pd.set_option('display.width', 160)`.
3. md: "Load the seven factors and ten ETFs" / code: `factors = load_factor_data()`; `etfs = load_etf_monthly(['SPY','IWM','EFA','EEM','TLT','TIP','LQD','HYG','GLD','VNQ'])`; print shapes and common window.
4. md: "Fit the factor model" / code: `fm = FactorModel(etfs, factors).fit(); fm.report.round(2)`.
5. md: "Read the betas" (IWM on Small cap, TIP on Inflation and Real premia, HYG on Credit and Equity DM; R2 column) / code: `fm.tstat.round(1)`; `fm.r2.round(2)`.
6. md: "Targets for the factors: the sample" / code: `t = Targets.from_history(factors); t.summary()`.
7. md: "Simulate the factors with the C-vine (families from the history)" / code: `cv = CVineMarket(t, central='Equity DM', families='auto', n_opt=10000, tol_func=1e-6).fit(); cv.edges`.
8. code: `Fsim = cv.simulate(25000, seed=1); cv.diagnostics(Fsim).summary()`.
9. md: "Map the factor scenarios to the assets" / code: `X = fm.simulate(Fsim, seed=1); X.describe().T[['mean','std']].round(4)` and compare with `etfs.describe().T[['mean','std']]`.
10. md: "With and without the residual" / code: `X0 = fm.simulate(Fsim, residuals=False)`; table of vol: history, with residual, without residual; sentence that the gap is `sqrt(1 - R2)` of the asset vol.
11. md: "A projection on the factors" / code: `proj = t.mean.copy(); proj['Equity DM'] = 0.003; proj['Credit'] = -0.002`; `t2 = Targets(mean=proj, vol=t.vol, corr=t.corr, history=factors)`; `fm.implied_mean(t2.mean).round(4)` next to `fm.implied_mean(t.mean)`.
12. code: `cv2 = CVineMarket(t2, central='Equity DM', families='auto', n_opt=10000, tol_func=1e-6).fit(); X2 = fm.simulate(cv2.simulate(25000, seed=2), seed=2); X2.quantile([0.05, 0.5, 0.95]).round(3)`.
13. md: "Asset paths from factor paths" / code: `P = cv.simulate_paths(2000, 12, seed=3); PX = fm.simulate(P, seed=3); PX.terminal().quantile([0.05, 0.5, 0.95]).round(3)`.
14. md: "Save" / code: `fm.save('factor_model.json'); cv.save('factor_market.json')`; and a closing md pointing to notebook 06 for the factors and 05 for LTCMA targeting.

- [ ] **Step 2: Execute and inspect**

Run: `cd examples && jupyter nbconvert --to notebook --execute --inplace 08_assets_on_macro_factors.ipynb --ExecutePreprocessor.timeout=1800`
Expected: no error cell; runtime under 10 minutes. Read the outputs: IWM's `Small cap` beta positive and significant (t > 3), TIP's `Inflation` beta positive, HYG's `Credit` beta positive.

- [ ] **Step 3: Update notebook 06 for seven factors and rerun**

Edit with `nbformat`: replace "six" by "seven" in markdown cells that describe the factor set; add `Small cap` to any explicit factor list; run `jupyter nbconvert --execute --inplace 06_macro_factors.ipynb`. Check the family table has seven rows in tree 1 minus one (six edges).

- [ ] **Step 4: README and docs**

README notebook table: row 06 "seven macro factors ..."; new row `| 08 | [08_assets_on_macro_factors](examples/08_assets_on_macro_factors.ipynb) | ten ETFs regressed on the seven factors, a view on the factors turned into asset distributions and paths |`. `docs/examples.rst`: add the toctree entry `examples/08_assets_on_macro_factors`.

- [ ] **Step 5: Commit**

```bash
git add examples/06_macro_factors.ipynb examples/08_assets_on_macro_factors.ipynb README.md docs/examples.rst
git commit -m "docs: notebook 08 assets on macro factors; seven factors in 06"
```

---

### Task 4: Univariate GARCH-family models and the per-asset container in `dynamics.py`

**Files:**
- Modify: `cvinemarketgen/dynamics.py` (add `GarchFamily`, `AssetDynamics`, `parse_spec`, registry; keep `AR1`, `AR1GARCH`, `make_dynamics`)
- Test: `tests/test_dynamics.py` (new; HMM parts come in Task 6)

**Interfaces:**
- Produces:
  - `GarchFamily(mean='const'|'ar1', vol='const'|'garch'|'gjr'|'egarch', p=1, q=1)`; after `fit(y: Series)`: `name`, `spec` (dict), `loglik`, `n_params`, `bic`, `n_obs`, `state`, `filter() -> Series` (fitted sample), `filter_new(y_new: 1-d array) -> 1-d array` (continues from `state`), `unfilter(z: (n_paths, horizon)) -> (n_paths, horizon)`, `simulate(n, seed) -> 1-d array`, `clone() -> unfitted copy`, `to_dict()`, `from_dict(d)`.
  - `parse_spec('ar1-gjr(1,2)') -> GarchFamily`, `'const-garch'` (p=q=1), `'ar1'` (constant variance), `'hmm(3)'` (Task 6).
  - `AssetDynamics(models: dict)` with `fit(history)`, `filter(history) -> DataFrame`, `unfilter(Z (n_paths, horizon, N)) -> ndarray`, `report`, `candidates`, `to_dict()`, `from_dict(d)`, `name = 'assets'`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dynamics.py
import numpy as np
import pandas as pd
import pytest

arch = pytest.importorskip('arch')
from cvinemarketgen.dynamics import GarchFamily, AssetDynamics, parse_spec


def _garch_series(n=2500, seed=0, a=0.0002, b=0.05, om=2e-6, al=0.08, be=0.90):
    rng = np.random.default_rng(seed)
    y = np.zeros(n); e = np.zeros(n); h = np.full(n, om / (1 - al - be))
    for t in range(1, n):
        h[t] = om + al * e[t - 1] ** 2 + be * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
        y[t] = a + b * y[t - 1] + e[t]
    return pd.Series(y, index=pd.bdate_range('2015-01-01', periods=n), name='X')


@pytest.mark.parametrize('spec', ['const-const', 'ar1', 'const-garch(1,1)', 'ar1-garch(2,1)', 'ar1-gjr(1,1)', 'const-egarch(1,2)'])
def test_filter_unfilter_are_inverse(spec):
    y = _garch_series()
    m = parse_spec(spec).fit(y)
    z = m.filter()
    assert isinstance(z, pd.Series) and np.isfinite(z.values).all() and len(z) >= len(y) - 1
    rng = np.random.default_rng(1)
    znew = rng.standard_normal((3, 40))
    ynew = m.unfilter(znew)
    assert ynew.shape == (3, 40)
    back = m.filter_new(ynew[1])
    assert np.allclose(back, znew[1], atol=1e-9)
    assert m.n_params == len(m.to_dict()['params']) and np.isfinite(m.bic)


def test_garch_recovers_parameters_and_names():
    y = _garch_series()
    m = GarchFamily('ar1', 'garch', 1, 1).fit(y)
    assert m.name == 'AR(1)-GARCH(1,1)'
    assert abs(m.params['alpha'][0] - 0.08) < 0.04 and abs(m.params['beta'][0] - 0.90) < 0.05
    s = m.simulate(500, seed=0)
    assert s.shape == (500,) and abs(s.std() / y.std() - 1) < 0.5
    d = m.to_dict(); m2 = GarchFamily.from_dict(d)
    assert np.allclose(m2.unfilter(np.ones((1, 5))), m.unfilter(np.ones((1, 5))))


def test_asset_dynamics_container():
    h = pd.DataFrame({'A': _garch_series(seed=1).values, 'B': _garch_series(seed=2, b=0.0).values},
                     index=pd.bdate_range('2015-01-01', periods=2500))
    ad = AssetDynamics({'A': parse_spec('ar1-garch(1,1)'), 'B': parse_spec('const-gjr(1,1)')}).fit(h)
    Z = ad.filter(h)
    assert list(Z.columns) == ['A', 'B'] and Z.shape[0] == 2499
    Y = ad.unfilter(np.zeros((4, 10, 2)))
    assert Y.shape == (4, 10, 2)
    assert list(ad.report['model']) == ['AR(1)-GARCH(1,1)', 'Const-GJR(1,1)']
    ad2 = AssetDynamics.from_dict(ad.to_dict())
    assert np.allclose(ad2.unfilter(np.zeros((1, 3, 2))), Y[:1, :3])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_dynamics.py -q`
Expected: FAIL, `ImportError: cannot import name 'GarchFamily'`.

- [ ] **Step 3: Implement in `cvinemarketgen/dynamics.py`**

Add after the imports:

```python
import re
import warnings

_SPEC_RE = re.compile(r'^(const|ar1)(?:-(const|garch|gjr|egarch)(?:\((\d+),(\d+)\))?)?$')
_HMM_RE = re.compile(r'^hmm(?:\((\d+)\))?$')


def _arch():
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("GARCH-family dynamics need the arch package: pip install arch  "
                          "(or pip install cvinemarketgen[garch])") from e
    return arch_model


class GarchFamily:
    """
    One asset's ``mean`` (``'const'`` or ``'ar1'``) with a ``vol`` model
    (``'const'``, ``'garch'``, ``'gjr'``, ``'egarch'``) of orders ``(p, q)``,
    fitted with ``arch`` on returns scaled by 100. ``filter`` gives the
    standardized residuals of the fitted sample; ``unfilter`` continues the
    recursion from the last observed state, vectorised over paths.
    """

    kind = 'garch'

    def __init__(self, mean='const', vol='garch', p=1, q=1):
        if mean not in ('const', 'ar1') or vol not in ('const', 'garch', 'gjr', 'egarch'):
            raise ValueError(f'unknown model mean={mean!r}, vol={vol!r}')
        self.mean, self.vol, self.p, self.q = mean, vol, (int(p) if vol != 'const' else 0), (int(q) if vol != 'const' else 0)
        self.params = None
        self.state = None
        self.scale = 1.0
        self._z = None

    # ---- naming --------------------------------------------------------------
    @property
    def spec(self):
        return {'mean': self.mean, 'vol': self.vol, 'p': self.p, 'q': self.q}

    @property
    def name(self):
        m = 'AR(1)' if self.mean == 'ar1' else 'Const'
        v = {'const': 'Const', 'garch': f'GARCH({self.p},{self.q})', 'gjr': f'GJR({self.p},{self.q})',
             'egarch': f'EGARCH({self.p},{self.q})'}[self.vol]
        return f'{m}-{v}'

    def clone(self):
        return GarchFamily(**self.spec)

    # ---- fit -----------------------------------------------------------------
    def fit(self, y):
        arch_model = _arch()
        s = pd.Series(y).astype(float)
        self.scale = 100.0 if s.abs().mean() < 0.5 else 1.0
        v = s.values * self.scale
        mkw = {'mean': 'Constant'} if self.mean == 'const' else {'mean': 'AR', 'lags': 1}
        vkw = {'const': {'vol': 'Constant'}, 'garch': {'vol': 'GARCH', 'p': self.p, 'o': 0, 'q': self.q},
               'gjr': {'vol': 'GARCH', 'p': self.p, 'o': 1, 'q': self.q},
               'egarch': {'vol': 'EGARCH', 'p': self.p, 'o': 0, 'q': self.q}}[self.vol]
        res = arch_model(v, dist='normal', rescale=False, **mkw, **vkw).fit(disp='off')
        prm = res.params
        a = float(prm['mu']) if self.mean == 'const' else float(prm['Const'])
        b = 0.0 if self.mean == 'const' else float(prm[[k for k in prm.index if k.endswith('[1]')
                                                        and not k.startswith(('alpha', 'beta', 'gamma'))][0]])
        if self.vol == 'const':
            omega, alpha, beta, gamma = float(prm['sigma2']), [], [], 0.0
        else:
            omega = float(prm['omega'])
            alpha = [float(prm[f'alpha[{i}]']) for i in range(1, self.p + 1)]
            beta = [float(prm[f'beta[{j}]']) for j in range(1, self.q + 1)]
            gamma = float(prm['gamma[1]']) if self.vol == 'gjr' else 0.0
        self.params = {'a': a, 'b': b, 'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
        e = np.asarray(res.resid, float)
        h = np.asarray(res.conditional_volatility, float) ** 2
        ok = np.isfinite(e) & np.isfinite(h)
        self._z = pd.Series((e / np.sqrt(h))[ok], index=s.index[ok])
        L = max(self.p, self.q, 1)
        self.state = {'y': float(v[-1]), 'e': e[ok][-L:].tolist(), 'h': h[ok][-L:].tolist()}
        self.n_obs = int(ok.sum())
        self.loglik = float(res.loglikelihood)
        self.n_params = int(len(prm))
        self.bic = float(res.bic)
        return self

    # ---- recursion -----------------------------------------------------------
    def _next_var(self, e_hist, h_hist):
        """Next variance from histories (n_paths, L), most recent last."""
        P = self.params
        if self.vol == 'const':
            return np.full(e_hist.shape[0], P['omega'])
        if self.vol == 'egarch':
            zh = e_hist / np.sqrt(h_hist)
            lh = P['omega'] + sum(P['alpha'][i - 1] * (np.abs(zh[:, -i]) - np.sqrt(2.0 / np.pi)) for i in range(1, self.p + 1)) \
                + sum(P['beta'][j - 1] * np.log(h_hist[:, -j]) for j in range(1, self.q + 1))
            return np.exp(lh)
        h = P['omega'] + sum(P['alpha'][i - 1] * e_hist[:, -i] ** 2 for i in range(1, self.p + 1)) \
            + sum(P['beta'][j - 1] * h_hist[:, -j] for j in range(1, self.q + 1))
        if self.vol == 'gjr':
            h = h + P['gamma'] * e_hist[:, -1] ** 2 * (e_hist[:, -1] < 0)
        return h

    def _start(self, n_paths):
        st = self.state
        return (np.full(n_paths, st['y']), np.tile(np.array(st['e']), (n_paths, 1)), np.tile(np.array(st['h']), (n_paths, 1)))

    def filter(self):
        """Standardized residuals ``epsilon_t`` of the fitted sample (Series)."""
        return self._z

    def filter_new(self, y_new):
        """Standardized residuals of new observations, continuing from the last state (scale of ``y``)."""
        v = np.asarray(y_new, float) * self.scale
        y_prev, e_hist, h_hist = self._start(1)
        z = np.empty(len(v))
        for t in range(len(v)):
            h = self._next_var(e_hist, h_hist)
            e = v[t] - (self.params['a'] + self.params['b'] * y_prev)
            z[t] = e[0] / np.sqrt(h[0])
            y_prev = np.array([v[t]])
            e_hist = np.column_stack([e_hist[:, 1:], e]); h_hist = np.column_stack([h_hist[:, 1:], h])
        return z

    def unfilter(self, z):
        """Returns from standardized-residual paths ``(n_paths, horizon)``, from the last observed state."""
        z = np.asarray(z, float)
        n_paths, horizon = z.shape
        y_prev, e_hist, h_hist = self._start(n_paths)
        out = np.empty_like(z)
        for t in range(horizon):
            h = self._next_var(e_hist, h_hist)
            e = np.sqrt(h) * z[:, t]
            y = self.params['a'] + self.params['b'] * y_prev + e
            out[:, t] = y
            y_prev = y
            e_hist = np.column_stack([e_hist[:, 1:], e]); h_hist = np.column_stack([h_hist[:, 1:], h])
        return out / self.scale

    def simulate(self, n, seed=None):
        """One simulated series of length ``n`` (Gaussian innovations), for the bootstrap."""
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z)[0]

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'scale': self.scale, 'params': self.params, 'state': self.state,
                'n_obs': self.n_obs, 'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m.scale, m.params, m.state = d['scale'], d['params'], d['state']
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m
```

Note for the test `m.n_params == len(m.to_dict()['params'])`: replace that assertion in the test by `m.n_params == (1 if m.mean == 'const' else 2) + (1 if m.vol == 'const' else 1 + m.p + m.q + (1 if m.vol == 'gjr' else 0))` (the dict has six keys regardless). Write the test that way.

Then the container and the spec parser:

```python
def parse_spec(spec):
    """``'const-garch(1,1)'``, ``'ar1-gjr(1,2)'``, ``'ar1'`` (constant variance), ``'hmm(3)'`` -> model."""
    if not isinstance(spec, str):
        return spec
    s = spec.strip().lower()
    m = _HMM_RE.match(s)
    if m:
        from .hmm import GaussianHMM
        return GaussianHMM(int(m.group(1) or 2))
    m = _SPEC_RE.match(s)
    if not m:
        raise ValueError(f'cannot parse dynamics spec {spec!r}')
    mean, vol, p, q = m.group(1), m.group(2) or 'const', m.group(3), m.group(4)
    return GarchFamily(mean, vol, int(p or 1), int(q or 1))


def model_from_dict(d):
    if d['kind'] == 'garch':
        return GarchFamily.from_dict(d)
    from .hmm import GaussianHMM
    return GaussianHMM.from_dict(d)


class AssetDynamics:
    """Per-asset dynamics: one univariate model per asset, sharing the market interface."""

    name = 'assets'

    def __init__(self, models=None):
        self.models = {a: parse_spec(m) for a, m in (models or {}).items()}
        self.report = None
        self.candidates = None

    def fit(self, history):
        h = pd.DataFrame(history).astype(float)
        for a in h.columns:
            if a not in self.models:
                raise ValueError(f'no dynamics model given for {a!r}')
            self.models[a].fit(h[a])
        if self.report is None:
            self.report = pd.DataFrame({a: {'model': m.name, 'n_params': m.n_params, 'loglik': m.loglik, 'bic': m.bic}
                                        for a, m in self.models.items()}).T
        return self

    def filter(self, history):
        """Residual layer of the fitted sample, one column per asset, rows where every asset has a value."""
        return pd.concat({a: self.models[a].filter() for a in pd.DataFrame(history).columns}, axis=1).dropna()

    def unfilter(self, Z):
        Z = np.asarray(Z, float)
        out = np.empty_like(Z)
        for j, a in enumerate(self.models):
            out[:, :, j] = self.models[a].unfilter(Z[:, :, j])
        return out

    def to_dict(self):
        return {'name': self.name, 'models': {a: m.to_dict() for a, m in self.models.items()},
                'report': None if self.report is None else self.report.to_dict(orient='index')}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.models = {a: model_from_dict(md) for a, md in d['models'].items()}
        if d.get('report'):
            m.report = pd.DataFrame(d['report']).T.loc[list(m.models)]
        return m
```

`AssetDynamics.filter` must return the assets in the history's column order; `unfilter` iterates `self.models` in insertion order, which `fit` guarantees matches the history's columns when the dict was built from it. In `_prepare_layer` (Task 7) the history is passed in `targets.assets` order, so build the dict in that order there.

Update `make_dynamics`:

```python
def make_dynamics(name):
    """None, 'ar1', 'ar1-garch', 'auto' (kept as the string, resolved at fit), a per-asset dict, or a model container."""
    if name is None or name == 'auto':
        return name
    if isinstance(name, (AR1, AR1GARCH, AssetDynamics)):
        return name
    if isinstance(name, dict):
        return AssetDynamics(name)
    try:
        return DYNAMICS[name]()
    except KeyError:
        raise ValueError(f"dynamics must be None, 'ar1', 'ar1-garch', 'auto', a dict of specs or an AssetDynamics, got {name!r}")
```

Add `'assets': AssetDynamics` to `DYNAMICS`? No: `DYNAMICS` maps names to no-argument constructors for the legacy strings only; loading uses `dynamics_from_dict`:

```python
def dynamics_from_dict(d):
    return {'ar1': AR1, 'ar1-garch': AR1GARCH, 'assets': AssetDynamics}[d['name']].from_dict(d)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dynamics.py -q`
Expected: 8 passed. If `arch` names the AR coefficient differently on the CI version (`'X[1]'` when the Series has a name), the list comprehension already covers it. If EGARCH residual reconstruction fails the inverse test at 1e-9, loosen to 1e-7 and note it in the docstring; the recursion is exact so 1e-9 should hold.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/dynamics.py tests/test_dynamics.py
git commit -m "feat: GARCH-family univariate models and per-asset dynamics container"
```

---

### Task 5: i.i.d. tests, Cramér-von Mises bootstrap and the selector in `selection.py`

**Files:**
- Create: `cvinemarketgen/selection.py`
- Create: `tests/data/ljungbox_reference.json` (values from statsmodels, produced once in the venv311 stack where statsmodels is available, or with `pip install statsmodels` in a scratch env)
- Test: `tests/test_selection.py`
- Modify: `cvinemarketgen/__init__.py`

**Interfaces:**
- Consumes: `GarchFamily`, `parse_spec` (Task 4); `GaussianHMM` (Task 6, imported lazily only when `'hmm'` is in the candidates).
- Produces: `ljung_box(x, lags=20) -> (stat, p)`, `arch_lm(z, lags=20) -> (stat, p)`, `iid_tests(z, lags=20) -> dict(lb_z, lb_z2, arch_lm)`, `cvm_statistic(u) -> float`, `gof_bootstrap(model, y, B=100, seed=0, verbose=False) -> dict(stat, pvalue, stats)`, `candidate_models(candidates, means, pq, states) -> list`, `select_dynamics(returns, candidates=('const','garch','gjr','egarch','hmm'), means=('const','ar1'), pq=(1, 2), states=(2, 3), alpha=0.05, lags=20, gof=True, B=100, seed=0, verbose=True) -> AssetDynamics`.

- [ ] **Step 1: Produce the statsmodels reference** (one-off, scratch env with statsmodels):

```python
import json, numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
x = np.random.default_rng(7).standard_normal(500); x[1:] += 0.3 * x[:-1]
lb = acorr_ljungbox(x, lags=[20], return_df=True)
lm = het_arch(x, nlags=20)
json.dump({'x': x.tolist(), 'lb_stat': float(lb['lb_stat'].iloc[0]), 'lb_p': float(lb['lb_pvalue'].iloc[0]),
           'lm_stat': float(lm[0]), 'lm_p': float(lm[1])}, open('tests/data/ljungbox_reference.json', 'w'))
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_selection.py
import json, os
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.selection import ljung_box, arch_lm, iid_tests, cvm_statistic, gof_bootstrap, select_dynamics
from tests.test_dynamics import _garch_series

HERE = os.path.dirname(__file__)


def test_tests_match_statsmodels_reference():
    ref = json.load(open(os.path.join(HERE, 'data', 'ljungbox_reference.json')))
    x = np.array(ref['x'])
    s, p = ljung_box(x, 20)
    assert abs(s - ref['lb_stat']) < 1e-6 and abs(p - ref['lb_p']) < 1e-8
    s, p = arch_lm(x, 20)
    assert abs(s - ref['lm_stat']) < 1e-4 and abs(p - ref['lm_p']) < 1e-6


def test_cvm_statistic_uniform_is_small():
    u = (np.arange(1, 1001) - 0.5) / 1000
    assert abs(cvm_statistic(u) - 1 / 12000) < 1e-12
    assert cvm_statistic(np.linspace(0, 0.5, 1000)) > 10


def test_selector_picks_garch_family_and_passes():
    pytest.importorskip('arch')
    y = _garch_series().to_frame()
    ad = select_dynamics(y, candidates=('const', 'garch', 'gjr'), pq=(1, 1), gof=False, verbose=False)
    r = ad.report.iloc[0]
    assert r['vol'] in ('garch', 'gjr') and r['passed'] and r['n_candidates'] == 6
    assert set(ad.candidates.columns) >= {'asset', 'model', 'bic', 'lb_z', 'lb_z2', 'arch_lm', 'passed'}
    z = ad.filter(y)
    assert min(iid_tests(z['X'].values).values()) > 0.05


def test_gof_bootstrap_runs_small():
    pytest.importorskip('arch')
    from cvinemarketgen.dynamics import parse_spec
    y = _garch_series(n=800)
    m = parse_spec('ar1-garch(1,1)').fit(y)
    g = gof_bootstrap(m, y, B=5, seed=0)
    assert 0 <= g['pvalue'] <= 1 and len(g['stats']) == 5 and g['stat'] > 0
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_selection.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cvinemarketgen.selection'`.

- [ ] **Step 4: Implement `cvinemarketgen/selection.py`**

```python
# -*- coding: utf-8 -*-
"""
Choosing the dynamics of each asset: i.i.d. tests on the residual layer,
BIC among the candidates that pass, and a parametric-bootstrap Cramér-von
Mises goodness-of-fit test of the selected model (ported from GenHMM1d).
"""
import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from .dynamics import GarchFamily, AssetDynamics


# ---- tests -------------------------------------------------------------------
def _acf(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    d = x @ x
    return np.array([(x[:-k] @ x[k:]) / d for k in range(1, lags + 1)])


def ljung_box(x, lags=20):
    """Ljung-Box statistic and p-value (chi-square with ``lags`` degrees of freedom)."""
    n = len(x)
    r = _acf(x, lags)
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def arch_lm(z, lags=20):
    """Engle's ARCH-LM test: ``z**2`` on its ``lags`` lags, ``n R^2`` chi-square(lags)."""
    s = np.asarray(z, float) ** 2
    n = len(s) - lags
    X = np.column_stack([np.ones(n)] + [s[lags - k:len(s) - k] for k in range(1, lags + 1)])
    y = s[lags:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ coef
    r2 = 1.0 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    lm = n * r2
    return float(lm), float(stats.chi2.sf(lm, lags))


def iid_tests(z, lags=20):
    """p-values of Ljung-Box on ``z``, Ljung-Box on ``z**2`` and ARCH-LM on ``z``."""
    z = np.asarray(z, float)
    return {'lb_z': ljung_box(z, lags)[1], 'lb_z2': ljung_box(z ** 2, lags)[1], 'arch_lm': arch_lm(z, lags)[1]}


def cvm_statistic(u):
    """Cramér-von Mises statistic of ``u`` against the uniform: ``1/(12n) + sum (u_(i) - (i-0.5)/n)^2``."""
    u = np.sort(np.asarray(u, float))
    n = len(u)
    return float(1.0 / (12 * n) + np.sum((u - (np.arange(1, n + 1) - 0.5) / n) ** 2))


def gof_bootstrap(model, y, B=100, seed=0, verbose=False):
    """
    Parametric-bootstrap Cramér-von Mises test of a fitted univariate model:
    the statistic on the Rosenblatt uniforms ``Phi(epsilon_t)`` of the sample,
    against ``B`` refits on series simulated from the model (GenHMM1d, GofHMMGen).
    """
    u = stats.norm.cdf(np.asarray(model.filter(), float))
    stat = cvm_statistic(u)
    n = len(pd.Series(y))
    out = np.empty(B)
    for b in range(B):
        ys = model.simulate(n, seed=seed + b)
        try:
            mb = model.clone().fit(pd.Series(ys))
            out[b] = cvm_statistic(stats.norm.cdf(np.asarray(mb.filter(), float)))
        except Exception:
            out[b] = np.nan
        if verbose and (b + 1) % 10 == 0:
            print(f'  bootstrap {b + 1}/{B}')
    ok = np.isfinite(out)
    return {'stat': stat, 'pvalue': float(np.mean(out[ok] > stat)), 'stats': out}


# ---- candidates and selection -------------------------------------------------
def candidate_models(candidates=('const', 'garch', 'gjr', 'egarch', 'hmm'), means=('const', 'ar1'),
                     pq=(1, 2), states=(2, 3)):
    """Unfitted models of every candidate specification."""
    out = []
    pmax, qmax = (pq if isinstance(pq, (tuple, list)) else (pq, pq))
    for vol in candidates:
        if vol == 'hmm':
            from .hmm import GaussianHMM
            out += [GaussianHMM(int(k)) for k in states]
        elif vol == 'const':
            out += [GarchFamily(m, 'const') for m in means]
        else:
            out += [GarchFamily(m, vol, p, q) for m in means for p in range(1, pmax + 1) for q in range(1, qmax + 1)]
    return out


def select_dynamics(returns, candidates=('const', 'garch', 'gjr', 'egarch', 'hmm'), means=('const', 'ar1'),
                    pq=(1, 2), states=(2, 3), alpha=0.05, lags=20, gof=True, B=100, seed=0, verbose=True):
    """
    Per asset: fit every candidate, test its residual layer for i.i.d.-ness,
    keep the lowest BIC among the candidates that pass (all three p-values
    above ``alpha``); if none passes, the lowest BIC overall with a warning.
    With ``gof``, the bootstrap Cramér-von Mises test is run on the selected
    model. Returns an :class:`~cvinemarketgen.dynamics.AssetDynamics` whose
    ``report`` has one row per asset and ``candidates`` every fit.
    """
    h = pd.DataFrame(returns).astype(float)
    rows, chosen = [], {}
    for a in h.columns:
        fits = []
        for m in candidate_models(candidates, means, pq, states):
            try:
                m.fit(h[a])
            except Exception as e:                      # non-convergence: skip the candidate
                if verbose:
                    print(f'{a}: {m.name} failed ({e.__class__.__name__}), skipped')
                continue
            t = iid_tests(m.filter().values, lags)
            sp = m.spec
            fits.append({'asset': a, 'model': m.name, 'mean': sp.get('mean', ''), 'vol': sp.get('vol', 'hmm'),
                         'p': sp.get('p', 0), 'q': sp.get('q', 0), 'states': sp.get('n_states', 0),
                         'n_params': m.n_params, 'loglik': m.loglik, 'bic': m.bic, **t,
                         'passed': all(v > alpha for v in t.values()), '_model': m})
        if not fits:
            raise RuntimeError(f'no candidate could be fitted for {a!r}')
        tab = pd.DataFrame(fits)
        ok = tab[tab['passed']]
        if len(ok):
            best = ok.sort_values('bic').iloc[0]
        else:
            best = tab.sort_values('bic').iloc[0]
            warnings.warn(f'{a}: no candidate passes the i.i.d. tests at {alpha}; keeping {best["model"]} (lowest BIC)')
        row = best.drop('_model').to_dict()
        row['n_passed'], row['n_candidates'] = int(tab['passed'].sum()), len(tab)
        if gof:
            if verbose:
                print(f'{a}: {best["model"]} selected, bootstrap goodness-of-fit with B={B}')
            g = gof_bootstrap(best['_model'], h[a], B=B, seed=seed, verbose=verbose)
            row['cvm'], row['gof_pvalue'] = g['stat'], g['pvalue']
        else:
            row['cvm'], row['gof_pvalue'] = np.nan, np.nan
        rows.append(row)
        chosen[a] = best['_model']
        if verbose:
            print(f'{a}: {row["model"]}  BIC {row["bic"]:.1f}  p-values LB {row["lb_z"]:.2f} / LB2 {row["lb_z2"]:.2f} / ARCH {row["arch_lm"]:.2f}'
                  + (f'  GoF {row["gof_pvalue"]:.2f}' if gof else ''))
        tab['_model'] = None
        rows[-1]['_cands'] = tab
    ad = AssetDynamics()
    ad.models = chosen
    ad.candidates = pd.concat([r.pop('_cands') for r in rows], ignore_index=True).drop(columns='_model')
    ad.report = pd.DataFrame(rows).set_index('asset')[['model', 'mean', 'vol', 'p', 'q', 'states', 'n_params', 'loglik', 'bic',
                                                        'lb_z', 'lb_z2', 'arch_lm', 'passed', 'cvm', 'gof_pvalue',
                                                        'n_passed', 'n_candidates']]
    return ad
```

`GaussianHMM.spec` must return `{'n_states': K}` (Task 6) so the row builder works. Export in `__init__.py`: `from .selection import select_dynamics, iid_tests, gof_bootstrap` and `from .dynamics import AR1, AR1GARCH, AssetDynamics, GarchFamily, parse_spec`; add to `__all__`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_selection.py -q`
Expected: 4 passed (the HMM candidates are not requested in these tests, so Task 6 is not needed yet).

- [ ] **Step 6: Commit**

```bash
git add cvinemarketgen/selection.py cvinemarketgen/__init__.py tests/test_selection.py tests/data/ljungbox_reference.json
git commit -m "feat: i.i.d. tests, bootstrap Cramér-von Mises test and dynamics selector"
```

---

### Task 6: Gaussian HMM in `hmm.py`, checked against GenHMM1d

**Files:**
- Create: `cvinemarketgen/hmm.py`
- Create: `tests/data/genhmm1d_reference.json` (one-off from GenHMM1d, script below)
- Modify: `tests/test_dynamics.py` (append HMM tests), `cvinemarketgen/__init__.py`

**Interfaces:**
- Produces: `GaussianHMM(n_states=2, n_init=10, max_iter=2000, tol=1e-10, seed=0)`; after `fit(y)`: `mu, sigma (arrays sorted by mu), Q (K x K), eta_T, name = 'HMM(K)'`, `spec = {'n_states': K}`, `kind = 'hmm'`, `loglik, n_params = K(K-1) + 2K, bic, n_obs`, `filter() -> Series` (normal scores of the Rosenblatt uniforms), `uniforms() -> Series`, `regimes -> DataFrame` (filtered probabilities, one column per state), `filter_new(y_new)`, `unfilter(z (n_paths, horizon))`, `simulate(n, seed)`, `clone()`, `to_dict()/from_dict()`.

- [ ] **Step 1: Produce the GenHMM1d reference** (scratch, uses the downloaded `genhmm1d/hmm.py` in the scratchpad; needs `joblib`):

```python
import sys, json, numpy as np
sys.path.insert(0, '<scratchpad>/genhmm1d_pkg')      # folder containing genhmm1d/__init__.py
from genhmm1d.hmm import HMM
rng = np.random.default_rng(11)
Q = np.array([[0.97, 0.03], [0.06, 0.94]]); mu = [0.0008, -0.0010]; sd = [0.007, 0.018]
s = 0; y = np.empty(2000)
for t in range(2000):
    s = rng.choice(2, p=Q[s]); y[t] = mu[s] + sd[s] * rng.standard_normal()
out = HMM().EstHMMGen(y=y.reshape(-1, 1) if False else y, reg=2, family='norm', percentiles=[0.5], max_iter=10000, eps=1e-12)
json.dump({'y': y.tolist(), 'theta': np.asarray(out['theta']).tolist(), 'Q': np.asarray(out['Q']).tolist(),
           'U': np.asarray(out['U']).ravel().tolist(), 'cvm': float(out['cvm']), 'LL': float(out['LL'])},
          open('tests/data/genhmm1d_reference.json', 'w'))
```

Check `out['theta']` layout is `(K, 2)` = `(mu, sigma)` per regime sorted by mean (GenHMM1d sorts regimes by their mean). If `EstHMMGen` expects a 1-d array, pass `y`; if it expects `(n, 1)`, pass `y.reshape(-1, 1)`: try the first, fall back to the second.

- [ ] **Step 2: Append the failing tests to `tests/test_dynamics.py`**

```python
from cvinemarketgen.hmm import GaussianHMM


def test_hmm_matches_genhmm1d_reference():
    import json, os
    ref = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'genhmm1d_reference.json')))
    y = pd.Series(ref['y'])
    m = GaussianHMM(2).fit(y)
    theta = np.array(ref['theta'])
    assert np.allclose(m.mu, theta[:, 0], atol=1e-4) and np.allclose(m.sigma, theta[:, 1], atol=1e-4)
    assert np.allclose(m.Q, np.array(ref['Q']), atol=1e-4)
    assert np.allclose(m.uniforms().values, np.array(ref['U']), atol=1e-4)
    assert abs(cvm_statistic(m.uniforms().values) - ref['cvm']) < 1e-4
    assert abs(m.loglik - ref['LL']) < 1e-3


def test_hmm_filter_unfilter_inverse_and_persistence():
    import json, os
    ref = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'genhmm1d_reference.json')))
    m = GaussianHMM(2).fit(pd.Series(ref['y']))
    assert m.name == 'HMM(2)' and m.n_params == 6 and m.regimes.shape == (2000, 2)
    znew = np.random.default_rng(3).standard_normal((2, 30))
    ynew = m.unfilter(znew)
    assert np.allclose(m.filter_new(ynew[0]), znew[0], atol=1e-8)
    m2 = GaussianHMM.from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(znew), ynew)
    assert m.simulate(100, seed=0).shape == (100,)


def test_selector_picks_hmm_on_hmm_data():
    import json, os
    from cvinemarketgen.selection import select_dynamics
    ref = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'genhmm1d_reference.json')))
    y = pd.DataFrame({'H': ref['y']})
    ad = select_dynamics(y, candidates=('const', 'hmm'), states=(2, 3), gof=False, verbose=False)
    assert ad.report.loc['H', 'model'] in ('HMM(2)', 'HMM(3)')
```

Add `from cvinemarketgen.selection import cvm_statistic` at the top of the test file.

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_dynamics.py -q -k hmm`
Expected: FAIL, `ModuleNotFoundError: No module named 'cvinemarketgen.hmm'`.

- [ ] **Step 4: Implement `cvinemarketgen/hmm.py`**

```python
# -*- coding: utf-8 -*-
"""
Gaussian hidden Markov model of one asset, as a dynamics model: the residual
layer is the normal score of the Rosenblatt uniform ``v_t = F_t(y_t)``, where
``F_t`` is the one-step-ahead predictive mixture. Compact reimplementation of
the Gaussian case of GenHMM1d (Nasri, Rémillard and Thioub); the initial
regime distribution is uniform, as there.
"""
import numpy as np
import pandas as pd
from scipy import stats


class GaussianHMM:
    """
    ``y_t | s_t = k ~ N(mu_k, sigma_k^2)``, ``s_t`` a Markov chain with transition matrix ``Q``.

    Parameters
    ----------
    n_states : int
    n_init : int
        EM starts: one from percentile splits, the others random perturbations of it.
    max_iter, tol : EM stopping rule on the log-likelihood increase.
    """

    kind = 'hmm'

    def __init__(self, n_states=2, n_init=10, max_iter=2000, tol=1e-10, seed=0):
        self.K = int(n_states)
        self.n_init, self.max_iter, self.tol, self.seed = int(n_init), int(max_iter), float(tol), seed
        self.mu = self.sigma = self.Q = None
        self.eta_T = None
        self._u = self._eta = None

    @property
    def spec(self):
        return {'n_states': self.K}

    @property
    def name(self):
        return f'HMM({self.K})'

    def clone(self):
        return GaussianHMM(self.K, self.n_init, self.max_iter, self.tol, self.seed)

    # ---- likelihood pieces -----------------------------------------------------
    def _dens(self, v):
        return stats.norm.pdf(v[:, None], self.mu[None, :], self.sigma[None, :])

    def _forward(self, f):
        """Filtered ``eta`` (n, K), predictive ``W`` (n, K), scaling ``c`` (n,)."""
        n, K = f.shape
        eta = np.empty((n, K)); W = np.empty((n, K)); c = np.empty(n)
        prev = np.full(K, 1.0 / K)
        for t in range(n):
            w = prev @ self.Q
            v = w * f[t]
            c[t] = v.sum()
            eta[t] = v / c[t]
            W[t] = w
            prev = eta[t]
        return eta, W, c

    def _backward(self, f, c):
        n, K = f.shape
        beta = np.empty((n, K)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (self.Q @ (f[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def _em(self, v, mu, sigma, Q):
        self.mu, self.sigma, self.Q = mu.copy(), sigma.copy(), Q.copy()
        ll_old = -np.inf
        for _ in range(self.max_iter):
            f = self._dens(v)
            eta, W, c = self._forward(f)
            ll = float(np.log(c).sum())
            beta = self._backward(f, c)
            g = eta * beta; g /= g.sum(1, keepdims=True)                    # smoothed probabilities
            xi = np.zeros((self.K, self.K))
            for t in range(1, len(v)):
                m = self.Q * np.outer(eta[t - 1], f[t] * beta[t]) / c[t]
                xi += m
            self.Q = xi / xi.sum(1, keepdims=True)
            wsum = g.sum(0)
            self.mu = (g * v[:, None]).sum(0) / wsum
            self.sigma = np.sqrt((g * (v[:, None] - self.mu) ** 2).sum(0) / wsum)
            if ll - ll_old < self.tol:
                break
            ll_old = ll
        f = self._dens(v)
        _, _, c = self._forward(f)
        return float(np.log(c).sum())

    def _init_params(self, v, rng, perturb):
        qs = np.quantile(v, np.linspace(0, 1, self.K + 1))
        mu = np.array([v[(v >= qs[k]) & (v <= qs[k + 1])].mean() for k in range(self.K)])
        sigma = np.array([v[(v >= qs[k]) & (v <= qs[k + 1])].std() for k in range(self.K)])
        if perturb:
            mu = mu + rng.standard_normal(self.K) * v.std() * 0.5
            sigma = sigma * np.exp(rng.standard_normal(self.K) * 0.3)
        Q = np.full((self.K, self.K), 0.1 / max(self.K - 1, 1)); np.fill_diagonal(Q, 0.9)
        return mu, np.maximum(sigma, 1e-8), Q

    # ---- fit -----------------------------------------------------------------
    def fit(self, y):
        s = pd.Series(y).astype(float)
        v = s.values
        rng = np.random.default_rng(self.seed)
        best = (-np.inf, None)
        for i in range(self.n_init):
            mu, sigma, Q = self._init_params(v, rng, perturb=i > 0)
            try:
                ll = self._em(v, mu, sigma, Q)
            except FloatingPointError:
                continue
            if ll > best[0]:
                best = (ll, (self.mu.copy(), self.sigma.copy(), self.Q.copy()))
        self.loglik, (self.mu, self.sigma, self.Q) = best
        order = np.argsort(self.mu)                                          # regimes sorted by mean, as GenHMM1d
        self.mu, self.sigma, self.Q = self.mu[order], self.sigma[order], self.Q[np.ix_(order, order)]
        f = self._dens(v)
        eta, W, c = self._forward(f)
        cdf = stats.norm.cdf(v[:, None], self.mu[None, :], self.sigma[None, :])
        u = np.clip((W * cdf).sum(1), 1e-12, 1 - 1e-12)
        self._u = pd.Series(u, index=s.index)
        self._eta = pd.DataFrame(eta, index=s.index, columns=[f'state {k + 1}' for k in range(self.K)])
        self.eta_T = eta[-1].copy()
        self.n_obs = len(v)
        self.n_params = self.K * (self.K - 1) + 2 * self.K
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    # ---- residual layer ------------------------------------------------------
    def uniforms(self):
        """Rosenblatt uniforms ``v_t`` of the fitted sample."""
        return self._u

    def filter(self):
        """Normal scores of the Rosenblatt uniforms (i.i.d. N(0,1) under the model)."""
        return pd.Series(stats.norm.ppf(self._u.values), index=self._u.index)

    @property
    def regimes(self):
        """Filtered regime probabilities of the fitted sample."""
        return self._eta

    def _mixture_cdf(self, y, W):
        return (W * stats.norm.cdf(y[:, None], self.mu[None, :], self.sigma[None, :])).sum(1)

    def _mixture_inv(self, u, W):
        lo = np.full(len(u), self.mu.min() - 12 * self.sigma.max())
        hi = np.full(len(u), self.mu.max() + 12 * self.sigma.max())
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self._mixture_cdf(mid, W) < u
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def _step_eta(self, eta, y):
        f = stats.norm.pdf(y[:, None], self.mu[None, :], self.sigma[None, :])
        v = (eta @ self.Q) * f
        return v / v.sum(1, keepdims=True)

    def unfilter(self, z):
        """Returns from normal-score paths ``(n_paths, horizon)``, the exact inverse of the filter from ``eta_T``."""
        z = np.asarray(z, float)
        n_paths, horizon = z.shape
        eta = np.tile(self.eta_T, (n_paths, 1))
        out = np.empty_like(z)
        for t in range(horizon):
            W = eta @ self.Q
            y = self._mixture_inv(stats.norm.cdf(z[:, t]), W)
            out[:, t] = y
            eta = self._step_eta(eta, y)
        return out

    def filter_new(self, y_new):
        v = np.asarray(y_new, float)
        eta = self.eta_T[None, :].copy()
        z = np.empty(len(v))
        for t in range(len(v)):
            W = eta @ self.Q
            u = np.clip(self._mixture_cdf(v[t:t + 1], W), 1e-12, 1 - 1e-12)
            z[t] = stats.norm.ppf(u[0])
            eta = self._step_eta(eta, v[t:t + 1])
        return z

    def simulate(self, n, seed=None):
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z)[0]

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                'Q': self.Q.tolist(), 'eta_T': self.eta_T.tolist(), 'n_obs': self.n_obs,
                'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['spec']['n_states'])
        m.mu, m.sigma, m.Q, m.eta_T = (np.array(d[k], float) for k in ('mu', 'sigma', 'Q', 'eta_T'))
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m
```

Export `GaussianHMM` in `__init__.py`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_dynamics.py tests/test_selection.py -q`
Expected: all pass. If the GenHMM1d comparison fails on `U` only, check whether GenHMM1d's `W` at `t = 0` uses `eta0 Q` (it does: `w00 = [eta00; eta_EM]`, `W = w00[:n] Q`), which the forward pass above reproduces; if it fails on `Q` beyond 1e-4, raise `max_iter` or tighten `tol` on both sides before touching the EM.

- [ ] **Step 6: Commit**

```bash
git add cvinemarketgen/hmm.py cvinemarketgen/__init__.py tests/test_dynamics.py tests/data/genhmm1d_reference.json
git commit -m "feat: Gaussian HMM dynamics with Rosenblatt residuals, checked against GenHMM1d"
```

---

### Task 7: `dynamics='auto'`, per-asset dict, JSON and `dynamics_report` in `markets.py`

**Files:**
- Modify: `cvinemarketgen/markets.py` (`_Market.__init__`, `_prepare_layer`, `dynamics_report`, `save`/`load` of both markets)
- Test: `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `make_dynamics`, `AssetDynamics`, `dynamics_from_dict`, `select_dynamics`.
- Produces: `CVineMarket(..., dynamics='auto'|dict|AssetDynamics, dynamics_kwargs=None)`, `FleishmanMarket(..., dynamics=..., dynamics_kwargs=None)`, property `dynamics_report`.

- [ ] **Step 1: Append failing tests to `tests/test_api.py`**

```python
def test_cvine_market_auto_dynamics_and_roundtrip(tmp_path):
    pytest.importorskip('arch')
    h = _synthetic_history(n=1500)
    t = Targets.from_history(h)
    cv = CVineMarket(t, families='gaussian', n_opt=2000, dynamics='auto',
                     dynamics_kwargs=dict(candidates=('const', 'garch'), pq=(1, 1), gof=False, verbose=False)).fit()
    rep = cv.dynamics_report
    assert list(rep.index) == ['A', 'B', 'C'] and set(rep.columns) >= {'model', 'bic', 'passed'}
    P = cv.simulate_paths(20, 30, seed=1)
    assert P.array.shape == (20, 30, 3)
    p = tmp_path / 'auto.json'; cv.save(str(p))
    cv2 = CVineMarket.load(str(p))
    assert cv2.dynamics_report['model'].equals(rep['model'])
    assert np.allclose(cv2.simulate_paths(3, 5, seed=2).array, cv.simulate_paths(3, 5, seed=2).array)


def test_cvine_market_dict_dynamics_with_hmm():
    pytest.importorskip('arch')
    h = _synthetic_history(n=1500)
    t = Targets.from_history(h)
    cv = CVineMarket(t, families='gaussian', n_opt=2000,
                     dynamics={'A': 'ar1-garch(1,1)', 'B': 'hmm(2)', 'C': 'const'}).fit()
    assert list(cv.dynamics_report['model']) == ['AR(1)-GARCH(1,1)', 'HMM(2)', 'Const-Const']
    assert cv.fit_targets.layer == 'residuals'
    assert cv.simulate_paths(5, 10, seed=0).array.shape == (5, 10, 3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_api.py -q -k dynamics`
Expected: FAIL, `TypeError: __init__() got an unexpected keyword argument 'dynamics_kwargs'`.

- [ ] **Step 3: Implement**

In `markets.py`, replace the import line and `_Market.__init__`/`_prepare_layer`:

```python
from .dynamics import make_dynamics, dynamics_from_dict, AR1, AR1GARCH, AssetDynamics


class _Market:
    def __init__(self, targets, dynamics=None, dynamics_kwargs=None):
        if not isinstance(targets, Targets):
            raise TypeError('targets must be a Targets object')
        self.targets = targets
        self.dynamics = make_dynamics(dynamics)
        self.dynamics_kwargs = dict(dynamics_kwargs or {})
        self.fit_targets = None
        self.fitted = False

    def _prepare_layer(self):
        t = self.targets
        if self.dynamics is None:
            self.fit_targets = t
            return
        if t.history is None:
            raise ValueError('dynamics need a history in the targets')
        hist = t.history.loc[:, t.assets]
        if isinstance(self.dynamics, AR1):
            self.dynamics.fit(hist)
            if t.higher_moments_source != 'history':
                self.fit_targets = self.dynamics.transfer_targets(t)      # LTCMA targets, Appendix A transfer
                return
        else:
            if t.higher_moments_source != 'history':
                raise ValueError('these dynamics are supported with history-based targets (Targets.from_history)')
            if self.dynamics == 'auto':
                from .selection import select_dynamics
                self.dynamics = select_dynamics(hist, **self.dynamics_kwargs)
            else:
                self.dynamics.fit(hist)
        ft = Targets.from_history(self.dynamics.filter(hist), assets=t.assets)
        ft.layer = 'residuals'; ft.freq = t.freq
        self.fit_targets = ft

    @property
    def dynamics_report(self):
        """Selection table of the per-asset dynamics (``AssetDynamics.report``), or None."""
        return getattr(self.dynamics, 'report', None)
```

`AssetDynamics.fit(hist)` needs its models in `hist` column order: in `AssetDynamics.fit`, rebuild `self.models = {a: self.models[a] for a in h.columns}` before fitting. `AR1GARCH.filter(hist)` and `AssetDynamics.filter(hist)` both take the history; `AssetDynamics.filter` ignores the values and returns the fitted-sample residuals (documented).

`CVineMarket.__init__` and `FleishmanMarket.__init__` gain `dynamics_kwargs=None` and pass it to `super().__init__`. In both `load` methods replace the `(AR1 if ... else AR1GARCH).from_dict(...)` line with `m.dynamics = dynamics_from_dict(d['dynamics'])`. The `FleishmanMarket.load` `NotImplementedError` for dynamics stays. Docstrings: `dynamics : None, 'ar1', 'ar1-garch', 'auto', dict or AssetDynamics` with two lines on `'auto'` and the dict; `dynamics_kwargs : dict, options of select_dynamics for 'auto'`.

- [ ] **Step 4: Run the full suite on both stacks**

Run: `python -m pytest tests -q` and `~/Desktop/CopulaGenerator/venv311/bin/python -m pytest tests -q` (install `arch` there first if missing: `~/Desktop/CopulaGenerator/venv311/bin/pip install arch`).
Expected: all pass on both.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/markets.py cvinemarketgen/dynamics.py tests/test_api.py
git commit -m "feat: dynamics='auto', per-asset dynamics dict, JSON round trip, dynamics_report"
```

---

### Task 8: Notebooks 07 (selector) and 09 (HMM)

**Files:**
- Modify: `examples/07_daily_paths_for_backtesting.ipynb`
- Create: `examples/09_hmm_dynamics.ipynb`
- Modify: `README.md` (rows 07, 09), `docs/examples.rst`

- [ ] **Step 1: Notebook 07 changes** (rebuild the cells after the data load; keep the load and the targets cells)

Replace the cell `cv = CVineMarket(t, central='SPY', families='auto', dynamics='ar1-garch').fit()` by:

1. md: "Choose the dynamics per asset" (candidates, tests, rule, cost note: 28 candidates and one bootstrap per asset, a few minutes).
2. code: `cv = CVineMarket(t, central='SPY', families='auto', dynamics='auto', dynamics_kwargs=dict(B=100, seed=0)).fit()`
3. code: `cv.dynamics_report[['model', 'bic', 'lb_z', 'lb_z2', 'arch_lm', 'passed', 'gof_pvalue', 'n_passed', 'n_candidates']].round(3)`
4. md: "The candidates of one asset" / code: `c = cv.dynamics.candidates; c[c['asset'] == 'SPY'].sort_values('bic')[['model', 'bic', 'lb_z', 'lb_z2', 'arch_lm', 'passed']].head(10).round(3)`
5. md: "Compare with the fixed AR(1)-GARCH(1,1) of the previous version" / code: `cv_fixed = CVineMarket(t, central='SPY', families='auto', dynamics='ar1-garch').fit()`; the squared-return autocorrelation table of the existing notebook, with three rows: history, selected dynamics, fixed GARCH.

Keep the remaining cells (paths, terminal quantiles, save/load) and rerun the whole notebook. Runtime target: under 15 minutes.

- [ ] **Step 2: Notebook 09 cells**

1. md: title "HMM dynamics for one asset", Colab badge, three sentences (regimes, the Rosenblatt uniform as the residual layer, the bootstrap test).
2. code: imports; `daily = load_daily_returns(['SPY', 'TLT', 'GLD', 'DBC'], start='2010-01-01'); y = daily['TLT']`.
3. md: "Fit two and three regimes" / code: `m2 = GaussianHMM(2).fit(y); m3 = GaussianHMM(3).fit(y)`; a DataFrame with `mu`, `sigma` (both per year: `mu*252`, `sigma*sqrt(252)`), `Q` diagonal, `loglik`, `bic` for both.
4. md: "Regime probabilities" / code: plot `m2.regimes['state 1']` over time with the cumulative return of TLT on a second axis.
5. md: "The residual layer" / code: `z = m2.filter(); iid_tests(z.values)`; histogram of `m2.uniforms()` against the uniform.
6. md: "The bootstrap goodness-of-fit test" / code: `g = gof_bootstrap(m2, y, B=100, seed=0); g['stat'], g['pvalue']`; same for `m3`.
7. md: "Against the GARCH family" / code: `ad = select_dynamics(y.to_frame(), gof=True, B=100, verbose=False); ad.report.T`.
8. md: "A simulated year" / code: `paths = m2.unfilter(np.random.default_rng(1).standard_normal((5, 252)))`; plot the five cumulative paths after the history's last 252 days.
9. md: closing: in the market this model is one candidate per asset; point to notebook 07.

- [ ] **Step 3: Execute both**

Run: `cd examples && jupyter nbconvert --to notebook --execute --inplace 07_daily_paths_for_backtesting.ipynb 09_hmm_dynamics.ipynb --ExecutePreprocessor.timeout=3600`
Expected: no error cell. Read the outputs: every asset's selected model passes the three tests or the warning is visible; GoF p-values printed.

- [ ] **Step 4: README and docs tables** (rows 07 and 09; `docs/examples.rst` toctree entry `examples/09_hmm_dynamics`).

- [ ] **Step 5: Commit**

```bash
git add examples/07_daily_paths_for_backtesting.ipynb examples/09_hmm_dynamics.ipynb README.md docs/examples.rst
git commit -m "docs: notebooks 07 with dynamics selection and 09 HMM dynamics"
```

---

### Task 9: Docs, version 0.3.0, changelog

**Files:**
- Modify: `docs/userguide.rst`, `docs/api.rst`, `docs/method.rst`, `CHANGELOG.md`, `pyproject.toml`, `cvinemarketgen/__init__.py`, `README.md`

- [ ] **Step 1: User guide.** After "Dynamics for daily data", add:

```rst
Choosing the dynamics
---------------------

.. code-block:: python

   cv = CVineMarket(t, families='auto', dynamics='auto').fit()
   cv.dynamics_report          # one row per asset: model, BIC, the three p-values, goodness-of-fit p-value
   cv.dynamics.candidates      # every candidate fitted, per asset

For each asset the candidates are a constant or AR(1) mean with a constant,
GARCH(p,q), GJR(p,q) or EGARCH(p,q) variance (``p, q`` up to 2 by default),
plus a Gaussian hidden Markov model with 2 or 3 regimes. Each candidate's
residual layer (standardized residuals, or the normal scores of the
Rosenblatt uniforms for the HMM) is tested for the absence of autocorrelation
(Ljung-Box), of remaining ARCH (Ljung-Box on the squares, ARCH-LM); among the
candidates that pass at 5 percent the lowest BIC is kept, otherwise the lowest
BIC overall with a warning. The selected model is then checked by a
parametric-bootstrap Cramér-von Mises test (``gof_pvalue``).

Options through ``dynamics_kwargs``: ``candidates``, ``means``, ``pq``,
``states``, ``alpha``, ``lags``, ``gof``, ``B``. To force a model per asset
give a dict: ``dynamics={'SPY': 'ar1-gjr(1,1)', 'TLT': 'hmm(2)', 'GLD': 'const-garch(1,1)'}``.
Cost: about 28 fits per asset (seconds each) and one bootstrap of ``B``
refits per asset.

Assets on factors
-----------------

.. code-block:: python

   fm = FactorModel(asset_returns, factor_returns).fit()
   fm.report                                   # alpha, betas, t-statistics, R2, residual vol
   F = CVineMarket(Targets.from_history(factor_returns), families='auto').fit().simulate(25000, seed=1)
   X = fm.simulate(F)                          # asset scenarios: alpha + beta' f + Johnson SU residual
   X0 = fm.simulate(F, residuals=False)        # factor exposure only
   fm.implied_mean(view_on_factor_means)       # expected asset returns under a view on the factors

A ``Paths`` object of factor paths gives a ``Paths`` of asset paths. Residuals
are independent across assets and of the factors; drop them to study the
factor exposure alone.
```

- [ ] **Step 2: API page.** Under Dynamics list `AR1, AR1GARCH, GarchFamily, AssetDynamics, parse_spec, make_dynamics`; add sections `HMM` (`cvinemarketgen.hmm`, `GaussianHMM`), `Selection` (`ljung_box, arch_lm, iid_tests, cvm_statistic, gof_bootstrap, candidate_models, select_dynamics`), `Factors` (`FactorModel`); add `load_etf_monthly` to Market data.

- [ ] **Step 3: Method page.** Add two bullets to the mapping: "Daily dynamics: `select_dynamics` (i.i.d. tests, BIC, bootstrap Cramér-von Mises), `GaussianHMM` (Rosenblatt residuals)"; "Assets on factors: `FactorModel`".

- [ ] **Step 4: Version and changelog.** `pyproject.toml` version `0.3.0`; `__version__ = '0.3.0'`; `CHANGELOG.md` entry `## 0.3.0 (unreleased)` listing: small-cap factor, `load_etf_monthly`, `FactorModel`, GARCH family models, `GaussianHMM`, `select_dynamics` with tests and bootstrap GoF, `dynamics='auto'` and per-asset dict, notebooks 08 and 09, 07 updated. README: one paragraph under "From a history, with dynamics" showing `dynamics='auto'` and `cv.dynamics_report`, and a short "Assets on factors" block.

- [ ] **Step 5: Build the docs locally**

Run: `cd docs && python -m sphinx -b html . _build/html -q`
Expected: no warnings from the new docstrings (fix RST as before: no `|x|`, no trailing underscores).

- [ ] **Step 6: Run the full suite once more and commit**

```bash
python -m pytest tests -q
git add docs/userguide.rst docs/api.rst docs/method.rst CHANGELOG.md pyproject.toml cvinemarketgen/__init__.py README.md
git commit -m "docs: user guide, API and changelog for 0.3.0"
```

---

### Task 10: Export script and the paper skeleton

**Files:**
- Create: `OldVersion/article4_export.py` (outside the repo)
- Create: `OldVersion/article-4.txt` (chapter file, same conventions as `article-3.txt`), `OldVersion/article-4-standalone.tex` (wrapper that compiles it alone)

- [ ] **Step 1: Export script.** Runs from `CVineMarketGen/`, reuses the cached data, and writes to `OldVersion/article4_figs/`:

```python
"""Tables and figures of article 4 from the package (run from the CVineMarketGen folder)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from cvinemarketgen import (load_factor_data, load_etf_monthly, load_daily_returns, Targets, CVineMarket,
                            FactorModel, GaussianHMM, select_dynamics)
OUT = '../OldVersion/article4_figs'; os.makedirs(OUT, exist_ok=True)
def tex(df, name, **kw): df.to_latex(os.path.join(OUT, name + '.tex'), **kw)

factors = load_factor_data(verbose=False)
etfs = load_etf_monthly(['SPY', 'IWM', 'EFA', 'EEM', 'TLT', 'TIP', 'LQD', 'HYG', 'GLD', 'VNQ'], verbose=False)
tex(factors.describe().T[['mean', 'std']].join(factors.skew().rename('skew')).join((factors.kurtosis() + 3).rename('kurt')).round(4), 'tab_factor_moments')
tex(factors.corr().round(2), 'tab_factor_corr')
fm = FactorModel(etfs, factors).fit()
tex(fm.report[['alpha'] + list(factors.columns) + ['R2', 'resid vol']].round(3), 'tab_betas')
tex(fm.tstat.round(1), 'tab_betas_t')
t = Targets.from_history(factors)
cv = CVineMarket(t, central='Equity DM', families='auto', n_opt=10000, tol_func=1e-6).fit()
tex(cv.edges[['tree', 'edge', 'selected family', 'status']], 'tab_factor_families', index=False)
F = cv.simulate(25000, seed=1)
fig = cv.plot_exceedance(F); fig.savefig(os.path.join(OUT, 'fig_factor_exceedance.png'), dpi=200); plt.close(fig)
X, X0 = fm.simulate(F, seed=1), fm.simulate(F, residuals=False)
tex(pd.DataFrame({'history': etfs.std(), 'with residual': X.std(), 'exposure only': X0.std()}).round(4), 'tab_asset_vol')
proj = t.mean.copy(); proj['Equity DM'] = 0.003; proj['Credit'] = -0.002
tex(pd.DataFrame({'sample view': fm.implied_mean(t.mean), 'projection': fm.implied_mean(proj)}).round(4), 'tab_implied_means')

daily = load_daily_returns(['SPY', 'TLT', 'GLD', 'DBC'], start='2010-01-01', verbose=False)
ad = select_dynamics(daily, B=100, seed=0, verbose=False)
tex(ad.report[['model', 'bic', 'lb_z', 'lb_z2', 'arch_lm', 'gof_pvalue', 'n_passed', 'n_candidates']].round(3), 'tab_selection')
td = Targets.from_history(daily)
cvd = CVineMarket(td, central='SPY', families='auto', dynamics=ad).fit()
P = cvd.simulate_paths(1000, 252, seed=1)
tex(P.terminal().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3), 'tab_terminal')
```

Plus the squared-return autocorrelation table (history, selected dynamics, i.i.d.) and the HMM regime figure for TLT, as in notebooks 07 and 09.

- [ ] **Step 2: Paper skeleton `OldVersion/article-4.txt`.** Same header block as `article-3.txt` (adapted: "Article 4", the title, `\chapter{...}\label{ch:article-4}`, `\begingroup`, the same `\providecommand{\inw}`), then:

```latex
\section*{\articleabstract}
%% abstract, 150 words, to be written with the author

\section{Introduction}\label{sec:art4-introduction}
\subsection{Contributions}
\subsection{Organization}

\section{The Generator in Brief}\label{sec:art4-generator}
%% Johnson SU marginals (Section~\ref{ssec:jsu}), family selection (Algorithm~3 of Chapter~\ref{ch:article-3}),
%% recursive calibration (Algorithm~4), accept-reject (Algorithm~5): one paragraph each with cross references.

\section{Macro Factors and Asset Betas}\label{sec:art4-factors}
\subsection{Factor Construction}\label{ssec:art4-factor_construction}
%% Table~\ref{tab:art4-factor_moments}, Table~\ref{tab:art4-factor_corr}
\subsection{Simulating the Factors}\label{ssec:art4-factor_sim}
%% Table~\ref{tab:art4-factor_families}, Figure~\ref{fig:art4-factor_exceedance}
\subsection{From Factors to Assets}\label{ssec:art4-betas}
%% r_t = \alpha + \boldsymbol{\beta}' \mathbf{f}_t + \varepsilon_t ; Newey-West ; residual Johnson SU ; Table~\ref{tab:art4-betas}
\subsection{Projections}\label{ssec:art4-projections}
%% implied means, Table~\ref{tab:art4-implied_means}, Table~\ref{tab:art4-asset_vol}

\section{Daily Dynamics}\label{sec:art4-dynamics}
\subsection{Candidate Models}\label{ssec:art4-candidates}
\subsection{Residual Layers}\label{ssec:art4-residuals}
%% \varepsilon_t = e_t / \sqrt{h_t} ; v_t = F_t(y_t) = \sum_k w_{t,k} \Phi((y_t-\mu_k)/\sigma_k) ; \varepsilon_t = \Phi^{-1}(v_t)
\subsection{Tests and Selection Rule}\label{ssec:art4-selection}
%% Ljung-Box, ARCH-LM, Cramér-von Mises with parametric bootstrap, rule
\subsection{Path Simulation}\label{ssec:art4-paths}
%% from the last observed state; what is constant (copula) and what moves (variances, regimes)

\section{Examples}\label{sec:art4-examples}
\subsection{Seven Factors and Ten ETFs}\label{ssec:art4-ex_factors}
\subsection{Daily Backtesting on Four ETFs}\label{ssec:art4-ex_daily}
%% Table~\ref{tab:art4-selection}, Figure~\ref{fig:art4-regimes}, Table~\ref{tab:art4-terminal}

\section{Conclusion}\label{sec:art4-conclusion}

\endgroup
```

`OldVersion/article-4-standalone.tex`: `\documentclass[11pt]{report}` with `amsmath, amssymb, booktabs, graphicx, hyperref, biblatex` (`\addbibresource{biblio.bib}`), `\newcommand{\articleabstract}{Abstract}`, `\graphicspath{{article4_figs/}}`, `\begin{document}\input{article-4.txt}\printbibliography\end{document}`.

- [ ] **Step 3: Run the export and compile the skeleton**

Run: `cd CVineMarketGen && python ../OldVersion/article4_export.py && cd ../OldVersion && pdflatex -interaction=nonstopmode article-4-standalone.tex`
Expected: tables and figures in `article4_figs/`, a PDF with the section headings.

- [ ] **Step 4: Write the prose with the author, section by section**, starting with Section 3 (factors), then Section 4 (dynamics), then the examples, the introduction and the abstract last. Each section: one copy-paste block, notation as in article 3, every number read from the exported tables. This step is interactive by the author's rule and is not committed to the repo (the paper lives outside it).

---

## Self-review

- **Spec coverage:** SMB and ETF loader (Task 1), `FactorModel` (2), notebook 08 and seven-factor 06 (3), GARCH family and container (4), tests, bootstrap and selector (5), HMM with GenHMM1d check (6), markets integration and JSON (7), notebooks 07 and 09 (8), docs and version (9), export script and paper (10). The spec's `Model.simulate` for the bootstrap, `clone`, `regimes`, `dynamics_kwargs`, `dynamics_report`, `candidates` are all defined.
- **Placeholders:** none; the paper prose is deliberately interactive (author's rule), the skeleton and every table it cites are produced by Task 10.
- **Type consistency:** `filter()` takes no argument on univariate models and returns a Series; `AssetDynamics.filter(history)` returns a DataFrame; `unfilter` takes `(n_paths, horizon)` for univariate models and `(n_paths, horizon, N)` for the container; `spec` is a dict on both model kinds (`GaussianHMM.spec = {'n_states': K}`); `select_dynamics` reads `spec.get(...)` accordingly.
