# Block Filters (VAR and VECM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter a block of variables in levels jointly with a vector error-correction model (or a VAR), feed its standardized innovations to the residual layer of the market, and rebuild paths in levels from the last observed values; notebook 10 and the short paper `paper-blocks.tex`.

**Architecture:** New module `blocks.py` with `BlockVECM` and `BlockVAR` (statsmodels estimation, own recursion for `unfilter`/`filter_new`); `AssetDynamics` accepts tuple keys and dispatches columns to blocks; `parse_spec` understands `'vecm'`/`'var'`; markets need no change beyond docstrings. A FRED monthly loader supplies yields and breakevens.

**Tech Stack:** numpy, pandas, statsmodels (optional extra `blocks`), pytest, nbformat/nbconvert, the export-script pattern of `article4_export.py`.

**Spec:** `docs/superpowers/specs/2026-09-18-block-filters-design.md`

## Global Constraints

- Python >= 3.8; tests with the anaconda interpreter (`python -m pytest tests -q`, statsmodels 0.13) and the venv311 stack (`/private/tmp/claude-501/-Users-mamadouthioub-Desktop-CopulaGenerator/f3ee7c82-ad93-4569-91ff-1868c0c475a5/scratchpad/venv311/bin/pytest -q`, statsmodels 0.15, numpy 2, pandas 3). CI runs bare `pytest`, so `tests/__init__.py` stays.
- statsmodels is optional: imported inside functions, `ImportError` with the pip hint; tests `importorskip`.
- No data file committed except `data/jpm_ltcma_2024.csv`; add `data/fred_cache.csv` to `.gitignore`.
- The article-3 engine (`moment_match.py`, `copulas.py`, `cvine.py`, `fleishman.py`) is untouched.
- Commit locally only; the user uploads to GitHub by hand.
- Block residual layer: each innovation divided by its own standard deviation; the cross-correlation of innovations is left to the vine.
- Version becomes `0.4.0 (unreleased)` in the changelog only in Task 5.

---

### Task 1: `BlockVECM` and `BlockVAR` in `blocks.py`

**Files:**
- Create: `cvinemarketgen/blocks.py`
- Create: `tests/test_blocks.py`
- Modify: `pyproject.toml` (optional extra `blocks = ["statsmodels"]`)

**Interfaces:**
- Produces: `BlockVECM(rank='auto', lags='auto', deterministic='co', max_lags=4, signif=0.05)`, `BlockVAR(lags='auto', max_lags=4)`. After `fit(X: DataFrame)`: `columns` (list), `p`, `rank` (VECM), `lags`, `alpha` (p x r), `beta` (p x r), `Gamma` (list of q arrays p x p), `const` (p,), `sigma` (p,), `state` (ndarray (lags + 1, p) for VECM, (lags, p) for VAR), `n_obs`, `loglik`, `n_params`, `bic`, `name`, `spec`, `kind` (`'vecm'` or `'var'`), `filter() -> DataFrame`, `filter_new(X_new: (n, p)) -> ndarray (n, p)`, `unfilter(Z: (n_paths, horizon, p)) -> ndarray (n_paths, horizon, p)`, `simulate(n, seed) -> ndarray (n, p)`, `clone()`, `refit(X)`, `to_dict()`, `from_dict(d)`, `block_from_dict(d)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_blocks.py
import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip('statsmodels')
from cvinemarketgen.blocks import BlockVECM, BlockVAR, block_from_dict


def _cointegrated(n=800, seed=0):
    """x1 random walk, x2 = x1 + AR(1) noise, x3 random walk: rank 1, beta ~ (1, -1, 0)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 3)) * [0.2, 0.1, 0.3]
    x1 = np.cumsum(e[:, 0]); u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + e[t, 1]
    x3 = np.cumsum(e[:, 2])
    return pd.DataFrame({'a': x1, 'b': x1 + u, 'c': x3}, index=pd.period_range('1960-01', periods=n, freq='M'))


def test_vecm_finds_rank_and_cointegrating_vector():
    X = _cointegrated()
    m = BlockVECM().fit(X)
    assert m.rank == 1 and m.lags >= 1 and m.columns == ['a', 'b', 'c']
    b = m.beta[:, 0] / m.beta[0, 0]
    assert abs(b[1] + 1) < 0.05 and abs(b[2]) < 0.05
    assert m.name == f'VECM(r=1, q={m.lags})' and np.isfinite(m.bic) and m.n_params > 0
    z = m.filter()
    assert list(z.columns) == ['a', 'b', 'c'] and len(z) == len(X) - m.lags - 1
    assert np.allclose(z.std(ddof=1).values, 1.0, atol=1e-6)


def test_vecm_unfilter_is_inverse_of_filter_new_and_round_trips():
    X = _cointegrated()
    m = BlockVECM(rank=1, lags=2).fit(X)
    Z = np.random.default_rng(1).standard_normal((4, 30, 3))
    Y = m.unfilter(Z)
    assert Y.shape == (4, 30, 3) and np.isfinite(Y).all()
    assert np.allclose(m.filter_new(Y[2]), Z[2], atol=1e-9)
    m2 = block_from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(Z), Y) and m2.name == m.name and m2.columns == m.columns
    s = m.simulate(50, seed=0)
    assert s.shape == (50, 3)


def test_var_recovers_coefficients_on_returns():
    rng = np.random.default_rng(2); n = 2000
    A = np.array([[0.3, 0.1], [0.0, 0.2]]); X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = A @ X[t - 1] + rng.standard_normal(2) * 0.01
    m = BlockVAR(lags=1).fit(pd.DataFrame(X, columns=['u', 'v']))
    assert np.allclose(m.Gamma[0], A, atol=0.05) and m.name == 'VAR(1)' and m.state.shape == (1, 2)
    Z = np.zeros((1, 5, 2))
    Y = m.unfilter(Z)
    assert np.allclose(m.filter_new(Y[0]), 0.0, atol=1e-9)


def test_statsmodels_missing_message(monkeypatch):
    import cvinemarketgen.blocks as b
    def boom():
        raise ImportError('block dynamics need statsmodels: pip install statsmodels')
    monkeypatch.setattr(b, '_sm', boom)
    with pytest.raises(ImportError, match='statsmodels'):
        BlockVAR(lags=1).fit(pd.DataFrame({'u': [0.0, 1.0, 0.5, 0.2], 'v': [0.0, 0.5, 0.1, 0.3]}))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_blocks.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cvinemarketgen.blocks'`.

- [ ] **Step 3: Implement `cvinemarketgen/blocks.py`**

```python
# -*- coding: utf-8 -*-
"""
Block filters: a vector error-correction model or a vector autoregression on a
block of variables, whose standardized innovations form the block's residual
layer (one column per variable; their cross-correlation is left to the vine).
Paths are rebuilt by the recursion from the last observed rows, in the units of
the fitted series (levels for a VECM). Estimated with statsmodels, an optional
dependency (``pip install statsmodels`` or ``pip install cvinemarketgen[blocks]``).

    VECM:  Delta x_t = c + alpha beta' x_{t-1} + sum_{i=1}^{q} Gamma_i Delta x_{t-i} + diag(sigma) eps_t
    VAR:   x_t = c + sum_{i=1}^{q} Gamma_i x_{t-i} + diag(sigma) eps_t
"""
import numpy as np
import pandas as pd


def _sm():
    try:
        from statsmodels.tsa.vector_ar import vecm
        from statsmodels.tsa.api import VAR
    except ImportError as e:
        raise ImportError('block dynamics need statsmodels: pip install statsmodels '
                          '(or pip install cvinemarketgen[blocks])') from e
    return vecm, VAR


class _Block:
    """Shared pieces of the two block models."""

    kind = 'block'

    def __init__(self):
        self.columns = None
        self.Gamma = []
        self.const = None
        self.sigma = None
        self.state = None
        self._z = None
        self.n_obs = self.loglik = self.n_params = self.bic = None

    @property
    def p(self):
        return len(self.columns)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}{"" if self.sigma is None else ", fitted"})'

    def filter(self):
        """Standardized innovations of the fitted sample, one column per variable (DataFrame)."""
        return self._z

    def simulate(self, n, seed=None):
        """One simulated block of length ``n`` with Gaussian innovations, shape ``(n, p)``."""
        z = np.random.default_rng(seed).standard_normal((1, int(n), self.p))
        return self.unfilter(z)[0]

    def refit(self, X):
        """Fit a fresh copy on ``X`` (bootstrap refit)."""
        return self.clone().fit(X)

    def _base_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'columns': self.columns, 'lags': self.lags,
                'Gamma': [G.tolist() for G in self.Gamma], 'const': self.const.tolist(), 'sigma': self.sigma.tolist(),
                'state': self.state.tolist(), 'n_obs': self.n_obs, 'loglik': self.loglik,
                'n_params': self.n_params, 'bic': self.bic}

    def _load_base(self, d):
        self.columns, self.lags = list(d['columns']), int(d['lags'])
        self.Gamma = [np.array(G, float) for G in d['Gamma']]
        self.const, self.sigma, self.state = (np.array(d[k], float) for k in ('const', 'sigma', 'state'))
        self.n_obs, self.loglik, self.n_params, self.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']


class BlockVECM(_Block):
    """
    Vector error-correction model on a block observed in levels.

    Parameters
    ----------
    rank : int or 'auto'
        Cointegration rank; ``'auto'`` uses the Johansen trace test at ``signif``.
    lags : int or 'auto'
        Number ``q`` of lagged differences; ``'auto'`` picks the BIC choice in ``1..max_lags``.
    deterministic : 'co' (constant outside the cointegration relation) or 'n' (none).
    """

    kind = 'vecm'

    def __init__(self, rank='auto', lags='auto', deterministic='co', max_lags=4, signif=0.05):
        super().__init__()
        if deterministic not in ('co', 'n'):
            raise ValueError("deterministic must be 'co' or 'n'")
        self.rank_spec, self.lags_spec = rank, lags
        self.deterministic, self.max_lags, self.signif = deterministic, int(max_lags), float(signif)
        self.rank = self.lags = None
        self.alpha = self.beta = None

    @property
    def spec(self):
        return {'rank': self.rank_spec, 'lags': self.lags_spec, 'deterministic': self.deterministic,
                'max_lags': self.max_lags, 'signif': self.signif}

    @property
    def name(self):
        return 'VECM' if self.rank is None else f'VECM(r={self.rank}, q={self.lags})'

    def clone(self):
        return BlockVECM(**self.spec)

    def fit(self, X):
        vecm, _ = _sm()
        X = pd.DataFrame(X).astype(float)
        self.columns = list(X.columns)
        V = X.values
        q = self.lags_spec
        if q == 'auto':
            q = max(1, int(vecm.select_order(V, maxlags=self.max_lags, deterministic=self.deterministic).bic))
        q = int(q)
        r = self.rank_spec
        if r == 'auto':
            r = int(vecm.select_coint_rank(V, det_order=0, k_ar_diff=q, method='trace', signif=self.signif).rank)
        r = int(r)
        res = vecm.VECM(V, k_ar_diff=q, coint_rank=r, deterministic=self.deterministic).fit()
        self.rank, self.lags = r, q
        self.alpha = np.asarray(res.alpha, float).reshape(self.p, r)
        self.beta = np.asarray(res.beta, float).reshape(self.p, r)
        self.Gamma = [np.asarray(res.gamma[:, i * self.p:(i + 1) * self.p], float) for i in range(q)]
        self.const = np.asarray(res.det_coef, float).ravel() if self.deterministic == 'co' else np.zeros(self.p)
        E = np.asarray(res.resid, float)
        self.sigma = E.std(axis=0, ddof=1)
        self._z = pd.DataFrame(E / self.sigma, index=X.index[q + 1:], columns=self.columns)
        self.state = V[-(q + 1):].copy()                                    # last q + 1 levels, most recent last
        self.n_obs = int(res.nobs)
        self.loglik = float(res.llf)
        # loadings and cointegrating vectors (minus the r^2 normalisations), short-run matrices, constants, variances
        self.n_params = int(2 * self.p * r - r * r + self.p * self.p * q + (self.p if self.deterministic == 'co' else 0) + self.p)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    # ---- recursion ---------------------------------------------------------
    def _step(self, levels, z):
        """Next levels from ``levels`` (n_paths, q + 1, p), most recent last, and innovations ``z`` (n_paths, p)."""
        x_prev = levels[:, -1]
        d = np.diff(levels, axis=1)                                          # d[:, -1] is Delta x_{t-1}
        Pi = self.alpha @ self.beta.T
        dx = self.const + x_prev @ Pi.T + sum(d[:, -(i + 1)] @ self.Gamma[i].T for i in range(self.lags)) + z * self.sigma
        return x_prev + dx

    def unfilter(self, Z):
        """Levels from innovation paths ``(n_paths, horizon, p)``, from the last observed rows."""
        Z = np.asarray(Z, float)
        n, H, p = Z.shape
        levels = np.tile(self.state, (n, 1, 1))
        out = np.empty_like(Z)
        for t in range(H):
            x = self._step(levels, Z[:, t])
            out[:, t] = x
            levels = np.concatenate([levels[:, 1:], x[:, None, :]], axis=1)
        return out

    def filter_new(self, X_new):
        """Standardized innovations of new rows of levels ``(n, p)``, continuing from the last state."""
        V = np.asarray(X_new, float).reshape(-1, self.p)
        levels = self.state[None].copy()
        z = np.empty_like(V)
        for t in range(len(V)):
            pred = self._step(levels, np.zeros((1, self.p)))[0]
            z[t] = (V[t] - pred) / self.sigma
            levels = np.concatenate([levels[:, 1:], V[t][None, None, :]], axis=1)
        return z

    # ---- persistence -------------------------------------------------------
    def to_dict(self):
        d = self._base_dict()
        d.update({'rank': self.rank, 'alpha': self.alpha.tolist(), 'beta': self.beta.tolist()})
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m._load_base(d)
        m.rank = int(d['rank'])
        m.alpha, m.beta = np.array(d['alpha'], float).reshape(m.p, m.rank), np.array(d['beta'], float).reshape(m.p, m.rank)
        return m


class BlockVAR(_Block):
    """Vector autoregression on a block, in the units of the series given (levels or returns)."""

    kind = 'var'

    def __init__(self, lags='auto', max_lags=4):
        super().__init__()
        self.lags_spec, self.max_lags = lags, int(max_lags)
        self.lags = None

    @property
    def spec(self):
        return {'lags': self.lags_spec, 'max_lags': self.max_lags}

    @property
    def name(self):
        return 'VAR' if self.lags is None else f'VAR({self.lags})'

    def clone(self):
        return BlockVAR(**self.spec)

    def fit(self, X):
        _, VAR = _sm()
        X = pd.DataFrame(X).astype(float)
        self.columns = list(X.columns)
        V = X.values
        q = self.lags_spec
        if q == 'auto':
            q = max(1, int(VAR(V).select_order(self.max_lags).bic))
        q = int(q)
        res = VAR(V).fit(q, trend='c')
        self.lags = q
        self.Gamma = [np.asarray(res.coefs[i], float) for i in range(q)]
        self.const = np.asarray(res.intercept, float).ravel()
        E = np.asarray(res.resid, float)
        self.sigma = E.std(axis=0, ddof=1)
        self._z = pd.DataFrame(E / self.sigma, index=X.index[q:], columns=self.columns)
        self.state = V[-q:].copy()
        self.n_obs = int(res.nobs)
        self.loglik = float(res.llf)
        self.n_params = int(self.p * self.p * q + self.p + self.p)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    def _step(self, hist, z):
        """Next values from ``hist`` (n_paths, q, p), most recent last."""
        return self.const + sum(hist[:, -(i + 1)] @ self.Gamma[i].T for i in range(self.lags)) + z * self.sigma

    def unfilter(self, Z):
        Z = np.asarray(Z, float)
        n, H, p = Z.shape
        hist = np.tile(self.state, (n, 1, 1))
        out = np.empty_like(Z)
        for t in range(H):
            x = self._step(hist, Z[:, t])
            out[:, t] = x
            hist = np.concatenate([hist[:, 1:], x[:, None, :]], axis=1)
        return out

    def filter_new(self, X_new):
        V = np.asarray(X_new, float).reshape(-1, self.p)
        hist = self.state[None].copy()
        z = np.empty_like(V)
        for t in range(len(V)):
            z[t] = (V[t] - self._step(hist, np.zeros((1, self.p)))[0]) / self.sigma
            hist = np.concatenate([hist[:, 1:], V[t][None, None, :]], axis=1)
        return z

    def to_dict(self):
        return self._base_dict()

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m._load_base(d)
        return m


def block_from_dict(d):
    """Block model from its ``to_dict`` output."""
    return {'vecm': BlockVECM, 'var': BlockVAR}[d['kind']].from_dict(d)
```

`pyproject.toml`: in `[project.optional-dependencies]` add the line `blocks = ["statsmodels"]` after `garch = ["arch"]`.

Note on `VAR(V).select_order(maxlags)`: statsmodels returns a `LagOrderResults` whose `.bic` is the selected lag; on statsmodels 0.13 the same attribute exists.

- [ ] **Step 4: Run the tests on both stacks**

Run: `python -m pytest tests/test_blocks.py -q` and the venv311 `pytest tests/test_blocks.py -q`.
Expected: 4 passed on both. If `select_coint_rank` returns rank 2 on the simulated system with the anaconda statsmodels, raise `n` in `_cointegrated` to 1500 (the trace test has low power on 800 observations only when the AR(1) noise is persistent; 0.5 is not).

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/blocks.py tests/test_blocks.py pyproject.toml
git commit -m "feat: block filters, VECM and VAR with standardized innovations as residual layer"
```

---

### Task 2: Blocks in `AssetDynamics`, `parse_spec` and the markets

**Files:**
- Modify: `cvinemarketgen/dynamics.py` (`parse_spec`, `model_from_dict`, `AssetDynamics`)
- Modify: `cvinemarketgen/markets.py` (docstrings of `dynamics` and `simulate_paths`), `cvinemarketgen/paths.py` (docstring of `cumulative`)
- Modify: `cvinemarketgen/__init__.py` (export `BlockVECM`, `BlockVAR`)
- Test: `tests/test_blocks.py` (append), `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `BlockVECM`, `BlockVAR`, `block_from_dict` (Task 1).
- Produces: `AssetDynamics({('a', 'b', 'c'): 'vecm', 'd': 'ar1'})`; `parse_spec('vecm')`, `parse_spec('vecm(r=1,q=2)')`, `parse_spec('var')`, `parse_spec('var(q=2)')`; JSON keys `'a|b|c'` for blocks; `AssetDynamics.blocks` (list of tuples).

- [ ] **Step 1: Append the failing tests**

```python
# tests/test_blocks.py (append)
from cvinemarketgen.dynamics import AssetDynamics, parse_spec


def test_parse_block_specs():
    assert parse_spec('vecm').spec['rank'] == 'auto'
    m = parse_spec('vecm(r=1,q=2)'); assert m.rank_spec == 1 and m.lags_spec == 2
    assert parse_spec('var').spec['lags'] == 'auto' and parse_spec('var(q=3)').lags_spec == 3


def test_container_with_a_block_and_a_univariate_model():
    X = _cointegrated()
    rng = np.random.default_rng(3)
    X['d'] = 0.0002 + 0.01 * rng.standard_normal(len(X))
    ad = AssetDynamics({('a', 'b', 'c'): 'vecm(r=1,q=1)', 'd': 'ar1'}).fit(X)
    assert ad.blocks == [('a', 'b', 'c')]
    Z = ad.filter(X)
    assert list(Z.columns) == ['a', 'b', 'c', 'd'] and Z.shape[0] == len(X) - 2
    assert list(ad.report.index) == ['a', 'b', 'c', 'd'] and ad.report.loc['a', 'model'] == ad.report.loc['c', 'model'] == 'VECM(r=1, q=1)'
    Y = ad.unfilter(np.zeros((3, 6, 4)))
    assert Y.shape == (3, 6, 4) and np.isfinite(Y).all()
    ad2 = AssetDynamics.from_dict(ad.to_dict())
    assert ad2.blocks == [('a', 'b', 'c')] and np.allclose(ad2.unfilter(np.zeros((1, 3, 4))), Y[:1, :3])
    # the history's column order is respected even when the block is not contiguous
    X2 = X[['d', 'a', 'b', 'c']]
    ad3 = AssetDynamics({('a', 'b', 'c'): 'vecm(r=1,q=1)', 'd': 'ar1'}).fit(X2)
    assert list(ad3.filter(X2).columns) == ['d', 'a', 'b', 'c']
    Y3 = ad3.unfilter(np.zeros((1, 3, 4)))
    assert np.allclose(Y3[0, :, 1:], Y[0, :3, :3]) and np.allclose(Y3[0, :, 0], Y[0, :3, 3])
```

```python
# tests/test_api.py (append)
def test_cvine_market_with_a_block(tmp_path):
    pytest.importorskip('statsmodels')
    from tests.test_blocks import _cointegrated
    X = _cointegrated(n=600)
    X['d'] = 0.0002 + 0.01 * np.random.default_rng(4).standard_normal(len(X))
    t = Targets.from_history(X)
    cv = CVineMarket(t, families='gaussian', n_opt=2000, dynamics={('a', 'b', 'c'): 'vecm(r=1,q=1)', 'd': 'ar1'}).fit()
    assert cv.fit_targets.layer == 'residuals' and list(cv.dynamics_report['model'])[:3] == ['VECM(r=1, q=1)'] * 3
    P = cv.simulate_paths(10, 12, seed=0)
    assert P.array.shape == (10, 12, 4) and np.isfinite(P.array).all()
    assert abs(P.array[:, 0, 0].mean() - X['a'].iloc[-1]) < 1.0         # levels continue from the last observation
    p = tmp_path / 'block.json'; cv.save(str(p))
    cv2 = CVineMarket.load(str(p))
    assert np.allclose(cv2.simulate_paths(2, 4, seed=1).array, cv.simulate_paths(2, 4, seed=1).array)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_blocks.py tests/test_api.py -q -k "block"`
Expected: FAIL, `ValueError: cannot parse dynamics spec 'vecm'` and `AttributeError: ... 'blocks'`.

- [ ] **Step 3: Implement in `dynamics.py`**

Add after `_HMM_RE`:

```python
_BLOCK_RE = re.compile(r'^(vecm|var)(?:\(([^)]*)\))?$')


def _block_args(s):
    """``'r=1,q=2'`` -> {'rank': 1, 'lags': 2}; ``'q=3'`` -> {'lags': 3}."""
    out = {}
    for part in (s or '').split(','):
        part = part.strip()
        if not part:
            continue
        k, v = part.split('=')
        out[{'r': 'rank', 'q': 'lags'}[k.strip()]] = int(v)
    return out
```

In `parse_spec`, before the `_SPEC_RE` match:

```python
    m = _BLOCK_RE.match(s)
    if m:
        from .blocks import BlockVECM, BlockVAR
        args = _block_args(m.group(2))
        return BlockVECM(**args) if m.group(1) == 'vecm' else BlockVAR(**args)
```

In `model_from_dict`:

```python
def model_from_dict(d):
    """Model from its ``to_dict`` output (univariate or block)."""
    if d['kind'] == 'garch':
        return GarchFamily.from_dict(d)
    if d['kind'] in ('vecm', 'var'):
        from .blocks import block_from_dict
        return block_from_dict(d)
    from .hmm import GaussianHMM
    return GaussianHMM.from_dict(d)
```

Replace the `AssetDynamics` class by:

```python
def _key(k):
    return tuple(k) if isinstance(k, (tuple, list)) else k


class AssetDynamics:
    """
    Per-asset dynamics: one model per asset or per block of assets
    (``{asset: spec or model, (asset, asset, ...): 'vecm' | 'var'}``), sharing the
    market interface. ``report`` lists the fitted model, its number of parameters,
    log-likelihood and BIC per variable (the variables of a block share one row's
    values); ``candidates`` holds the selection table when the univariate models
    were chosen by :func:`selection.select_dynamics`; ``blocks`` lists the block keys.
    """

    name = 'assets'

    def __init__(self, models=None):
        self.models = {_key(a): parse_spec(m) for a, m in (models or {}).items()}
        self.columns = None
        self.report = None
        self.candidates = None

    def __repr__(self):
        return f'AssetDynamics({ {a: m.name for a, m in self.models.items()} })'

    @property
    def blocks(self):
        return [k for k in self.models if isinstance(k, tuple)]

    @staticmethod
    def _vars(key):
        return list(key) if isinstance(key, tuple) else [key]

    def fit(self, history):
        h = pd.DataFrame(history).astype(float)
        covered = [c for k in self.models for c in self._vars(k)]
        missing = [c for c in h.columns if c not in covered]
        if missing:
            raise ValueError(f'no dynamics model given for {missing}')
        extra = [c for c in covered if c not in h.columns]
        if extra or len(covered) != len(set(covered)):
            raise ValueError(f'dynamics keys must partition the history columns; unknown or repeated: {extra or covered}')
        for k, m in self.models.items():
            m.fit(h[list(k)] if isinstance(k, tuple) else h[k])
        self.columns = list(h.columns)
        if self.report is None:
            rows = {}
            for k, m in self.models.items():
                for c in self._vars(k):
                    rows[c] = {'model': m.name, 'n_params': m.n_params, 'loglik': m.loglik, 'bic': m.bic}
            self.report = pd.DataFrame(rows).T.loc[self.columns]
        return self

    def filter(self, history):
        """Residual layer of the fitted sample, one column per variable in the history's column order."""
        cols = list(pd.DataFrame(history).columns)
        parts = []
        for k, m in self.models.items():
            z = m.filter()
            parts.append(z if isinstance(z, pd.DataFrame) else z.to_frame(k))
        return pd.concat(parts, axis=1, sort=False).dropna()[cols]

    def unfilter(self, Z):
        """Returns (levels for blocks) from residual paths ``(n_paths, horizon, N)``, columns as in the history."""
        Z = np.asarray(Z, float)
        out = np.empty_like(Z)
        for k, m in self.models.items():
            idx = [self.columns.index(c) for c in self._vars(k)]
            if isinstance(k, tuple):
                out[:, :, idx] = m.unfilter(Z[:, :, idx])
            else:
                out[:, :, idx[0]] = m.unfilter(Z[:, :, idx[0]])
        return out

    def to_dict(self):
        return {'name': self.name, 'columns': self.columns,
                'models': {('|'.join(k) if isinstance(k, tuple) else k): m.to_dict() for k, m in self.models.items()},
                'report': None if self.report is None else self.report.to_dict(orient='index'),
                'candidates': None if self.candidates is None else self.candidates.to_dict(orient='list')}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.models = {(tuple(a.split('|')) if md['kind'] in ('vecm', 'var') else a): model_from_dict(md) for a, md in d['models'].items()}
        m.columns = d.get('columns') or [c for k in m.models for c in cls._vars(k)]
        if d.get('report'):
            m.report = pd.DataFrame(d['report']).T.loc[m.columns]
        if d.get('candidates'):
            m.candidates = pd.DataFrame(d['candidates'])
        return m
```

`select_dynamics` in `selection.py` builds `AssetDynamics()` then sets `ad.models = chosen` (univariate keys) and `ad.report`; add `ad.columns = list(h.columns)` right after `ad.models = chosen` so `unfilter` works.

`markets.py`: in the docstrings of `dynamics` (both markets) add the sentence "A block of variables filtered jointly is given as a tuple key, ``dynamics={('DGS2', 'DGS10'): 'vecm', 'SPY': 'ar1-garch'}``; its columns are simulated in the units of the history (levels for a VECM)." In `simulate_paths`'s docstring add "Columns handled by a block model are in the units of the history (levels for a VECM); ``Paths.cumulative`` and ``terminal`` apply to return columns." `paths.py`: in `cumulative` and `terminal` docstrings add "(meaningful for return columns only)". `__init__.py`: `from .blocks import BlockVECM, BlockVAR` and add both names to `__all__`.

- [ ] **Step 4: Run the full suite on both stacks**

Run: `python -m pytest tests -q` and the venv311 `pytest -q`.
Expected: all pass (the earlier 48 plus 7 new).

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/dynamics.py cvinemarketgen/selection.py cvinemarketgen/markets.py cvinemarketgen/paths.py cvinemarketgen/__init__.py tests/test_blocks.py tests/test_api.py
git commit -m "feat: blocks in AssetDynamics, 'vecm'/'var' specs, JSON round trip"
```

---

### Task 3: FRED monthly loader

**Files:**
- Modify: `cvinemarketgen/data.py` (add `load_fred_monthly`), `cvinemarketgen/__init__.py`, `.gitignore`
- Test: `tests/test_data_offline.py` (append)

**Interfaces:**
- Consumes: `fred_series(series_id, start)` (existing, daily Series).
- Produces: `load_fred_monthly(series_ids, start='2003-01', end=None, cache='data/fred_cache.csv', refresh=False, verbose=True) -> DataFrame` with `PeriodIndex('M')`, one column per id, monthly means of the daily values, rows where every id has a value.

- [ ] **Step 1: Append the failing test**

```python
# tests/test_data_offline.py (append)
def test_load_fred_monthly_offline(monkeypatch, tmp_path):
    import cvinemarketgen.data as data
    idx = pd.date_range('2020-01-01', '2020-03-31', freq='B')
    fake = {'DGS2': pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx, name='DGS2'),
            'DGS10': pd.Series(np.linspace(2.0, 3.0, len(idx)), index=idx, name='DGS10')}
    monkeypatch.setattr(data, 'fred_series', lambda sid, start='1986-01-01', timeout=60: fake[sid].loc[start:])
    cache = tmp_path / 'fred.csv'
    df = data.load_fred_monthly(['DGS2', 'DGS10'], start='2020-01', cache=str(cache), verbose=False)
    assert list(df.columns) == ['DGS2', 'DGS10'] and df.index.freqstr == 'M' and len(df) == 3
    assert abs(df.loc['2020-02', 'DGS2'] - fake['DGS2'].loc['2020-02'].mean()) < 1e-12
    monkeypatch.setattr(data, 'fred_series', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('network')))
    df2 = data.load_fred_monthly(['DGS2'], start='2020-01', cache=str(cache), verbose=False)   # served from the cache
    assert df2.shape == (3, 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_data_offline.py -q -k fred_monthly`
Expected: FAIL, `AttributeError: module 'cvinemarketgen.data' has no attribute 'load_fred_monthly'`.

- [ ] **Step 3: Implement**

```python
def load_fred_monthly(series_ids, start='2003-01', end=None, cache='data/fred_cache.csv', refresh=False, verbose=True):
    """
    Monthly means of daily FRED series (yields, breakevens, spreads), one column
    per series id, ``PeriodIndex('M')``, rows where every series has a value;
    cached locally as a CSV (git-ignored). Units are FRED's (percent for yields).
    """
    ids = list(series_ids)
    if cache and os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.PeriodIndex(df.index, freq='M')
        if set(ids) <= set(df.columns):
            df = (df.loc[start:end, ids] if end else df.loc[start:, ids]).dropna()
            if verbose:
                print(f'FRED series read from cache {cache}: {df.shape[0]} months, {df.index.min()} to {df.index.max()}')
            return df
    parts = []
    for sid in ids:
        s = fred_series(sid, start=str(pd.Period(start, 'M').start_time.date()))
        parts.append(s.groupby(s.index.to_period('M')).mean().rename(sid))
    df = pd.concat(parts, axis=1).dropna()
    df = df.loc[start:end] if end else df.loc[start:]
    df.index.name = 'month'
    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        df.to_csv(cache)
    if verbose:
        print(f'FRED series downloaded: {df.shape[0]} months, {df.index.min()} to {df.index.max()}'
              + (f', cached to {cache}' if cache else ''))
    return df
```

Export it in `__init__.py` (import and `__all__`); add `data/fred_cache.csv` to `.gitignore`; add `load_fred_monthly` to the "Market data" members of `docs/api.rst`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_data_offline.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/data.py cvinemarketgen/__init__.py .gitignore docs/api.rst tests/test_data_offline.py
git commit -m "feat: load_fred_monthly"
```

---

### Task 4: Notebook 10, block filters on yields and breakevens

**Files:**
- Create: `examples/10_block_filters_vecm.ipynb` (built with `nbformat` by a scratchpad script, executed with `jupyter nbconvert --to notebook --execute --inplace`)
- Modify: `README.md` (notebook table, row 10; "Nine tutorials" -> "Ten"), `docs/examples.rst` (toctree entry, "Nine" -> "Ten")

**Interfaces:**
- Consumes: `load_fred_monthly`, `load_etf_monthly`, `Targets`, `CVineMarket`, `BlockVECM`, `iid_tests`.

- [ ] **Step 1: Build the cells** (same helper as `build_tutorials.py`: `md(text)`, `code(src)`, Colab badge in the first cell, the standard import cell of notebook 08 with `from cvinemarketgen import load_fred_monthly, load_etf_monthly, Targets, CVineMarket, BlockVECM, iid_tests` and a `try: import statsmodels except ImportError: pip install statsmodels` block like notebook 07's `arch` block)

1. md: title "10 — Block filters: a VECM on yields and breakevens"; three sentences: variables in levels share long-run relations, a VECM filters them jointly, its standardized innovations join the residual layer and the paths come back in levels; runtime about 5 minutes.
2. md "Load the block and one asset" / code: `yields = load_fred_monthly(['DGS2', 'DGS10', 'T10YIE'], start='2003-01', cache=os.path.join(ROOT, 'data', 'fred_cache.csv'))`; `spy = load_etf_monthly(['SPY'], cache=os.path.join(ROOT, 'data', 'etf_cache.csv'))`; `data = yields.join(spy, how='inner'); data.tail()`; a plot of the three levels.
3. md "Rank and lags" / code: `m = BlockVECM().fit(data[['DGS2', 'DGS10', 'T10YIE']])`; `print(m.name, 'rank', m.rank, 'lags', m.lags)`; `pd.DataFrame(m.beta, index=m.columns, columns=[f'relation {k+1}' for k in range(m.rank)]).round(3)`; `pd.DataFrame(m.alpha, index=m.columns, columns=[f'loading {k+1}' for k in range(m.rank)]).round(3)`.
4. md "The residual layer of the block" / code: `z = m.filter(); z.describe().round(3)`; `pd.DataFrame({c: iid_tests(z[c].values) for c in z.columns}).round(3)`; `z.corr().round(2)` with a sentence that this correlation is what the vine will carry.
5. md "The block inside a market" / code: `t = Targets.from_history(data)`; `cv = CVineMarket(t, central='SPY', families='auto', dynamics={('DGS2', 'DGS10', 'T10YIE'): 'vecm', 'SPY': 'ar1-garch'}).fit()`; `cv.dynamics_report`; `cv.edges`.
6. md "Paths in levels from the last curve" / code: `P = cv.simulate_paths(1000, 24, seed=1)`; `last = data.iloc[-1]`; fan chart of DGS10 (quantiles 5, 25, 50, 75, 95 over the 24 months) with the last observed value; `P.to_frame().groupby('t')[['DGS2', 'DGS10', 'T10YIE']].quantile([0.05, 0.5, 0.95]).unstack().round(2).iloc[[0, 11, 23]]`.
7. md "The spread stays anchored" / code: the 2s10s spread `DGS10 - DGS2` along the paths against its history: `sp = P.array[:, :, 1] - P.array[:, :, 0]`; print the mean and standard deviation of the spread at month 24 against the historical spread's; a sentence on error correction pulling the spread back.
8. md "Without the block: three univariate AR(1) filters" / code: `cv1 = CVineMarket(t, central='SPY', families='auto', dynamics={'DGS2': 'ar1', 'DGS10': 'ar1', 'T10YIE': 'ar1', 'SPY': 'ar1-garch'}).fit()`; `P1 = cv1.simulate_paths(1000, 24, seed=1)`; the same spread statistics; `cv1.fit_targets.corr.round(2)` against `cv.fit_targets.corr.round(2)`: the residual correlations of the levels under AR(1) filters versus the innovations of the VECM.
9. md "Save" / code: `cv.save('block_market.json'); CVineMarket.load('block_market.json').simulate_paths(3, 6, seed=2)`.
10. md closing: the same tuple key takes any block; the exchange-rate notebook builds on it.

- [ ] **Step 2: Execute and inspect**

Run: `cd examples && jupyter nbconvert --to notebook --execute --inplace 10_block_filters_vecm.ipynb --ExecutePreprocessor.timeout=1800`
Expected: no error cell. Read the outputs: rank 1 or 2 on the three series, the cointegrating vector with opposite signs on DGS2 and DGS10, the residual tests passing for the block (p-values above 0.05) or a sentence added to cell 4 saying which fails, the spread's dispersion at month 24 smaller under the VECM than under the AR(1) filters.

- [ ] **Step 3: README and docs**

README notebook table: `| 10 | [`10_block_filters_vecm`](examples/10_block_filters_vecm.ipynb) | a VECM on 2-year and 10-year yields and breakevens as one block of the residual layer, paths in levels |`; "Nine tutorials" -> "Ten tutorials". `docs/examples.rst`: "Nine executed notebooks" -> "Ten", toctree entry `examples/10_block_filters_vecm`.

- [ ] **Step 4: Commit**

```bash
git add examples/10_block_filters_vecm.ipynb README.md docs/examples.rst
git commit -m "docs: notebook 10, a VECM block on yields and breakevens"
```

---

### Task 5: Docs, changelog, export script and the paper

**Files:**
- Modify: `docs/userguide.rst` (section "Blocks of variables" after "Choosing the dynamics"), `docs/api.rst` (section "Blocks", `cvinemarketgen.blocks`), `docs/method.rst` (one bullet), `CHANGELOG.md` (`## 0.4.0 (unreleased)`), `README.md` (one paragraph after the dynamics paragraph)
- Create: `OldVersion/overleaf-path-simulation/paper_blocks_export.py`, tables and figures in `OldVersion/overleaf-path-simulation/figs/blocks/`
- Modify: `OldVersion/overleaf-path-simulation/paper-blocks.tex` (the skeleton becomes the paper)

- [ ] **Step 1: User guide**

After the "Choosing the dynamics" section:

```rst
Blocks of variables
-------------------

.. code-block:: python

   yields = load_fred_monthly(['DGS2', 'DGS10', 'T10YIE'])
   t = Targets.from_history(yields.join(spy_returns, how='inner'))
   cv = CVineMarket(t, central='SPY', families='auto',
                    dynamics={('DGS2', 'DGS10', 'T10YIE'): 'vecm', 'SPY': 'ar1-garch'}).fit()
   P = cv.simulate_paths(1000, 24, seed=1)      # yields in levels, SPY in returns

Variables observed in levels that share long-run relations (yields, breakevens,
spreads) are filtered jointly: a tuple key gives the block, ``'vecm'`` fits a
vector error-correction model (cointegration rank by the Johansen trace test,
lags by BIC; ``'vecm(r=1,q=2)'`` fixes them) and ``'var'`` a vector
autoregression. The block's residual layer is its standardized innovations, one
per variable; their correlation is left to the vine, like everything else. Paths
of the block's variables come back in levels, from the last observed rows,
through the model's own recursion; ``Paths.cumulative`` and ``terminal`` apply to
the return columns. Needs ``pip install statsmodels``.
```

- [ ] **Step 2: API, method, changelog, README**

`docs/api.rst`: after the "Selection of the dynamics" section, add

```rst
Blocks
------

.. automodule:: cvinemarketgen.blocks
   :members: BlockVECM, BlockVAR
```

`docs/method.rst`, in "Beyond the chapter", a third bullet: "Block filters: :class:`~cvinemarketgen.blocks.BlockVECM` and :class:`~cvinemarketgen.blocks.BlockVAR` filter a block of variables jointly; their standardized innovations are the block's residual layer and paths come back in levels." `CHANGELOG.md`: new entry `## 0.4.0 (unreleased)` with two bullets, block filters (`BlockVECM`, `BlockVAR`, tuple keys in `dynamics`, `statsmodels` optional) and `load_fred_monthly`, plus "Notebook 10". README: after the `dynamics='auto'` paragraph, one paragraph: "Variables in levels that share long-run relations go in one block: `dynamics={('DGS2', 'DGS10', 'T10YIE'): 'vecm', 'SPY': 'ar1-garch'}` fits a vector error-correction model on the block (needs `statsmodels`), feeds its standardized innovations to the vine, and simulates the block in levels from the last observed curve; notebook 10 shows it on Treasury yields and breakevens."

- [ ] **Step 3: Export script**

`OldVersion/overleaf-path-simulation/paper_blocks_export.py`, run from `CVineMarketGen/`, `sys.path.insert(0, os.getcwd())`, `pd.set_option('display.max_colwidth', None)`, `OUT = <project>/figs/blocks`, the `tex`/`log` helpers of `article4_export.py`. It reproduces notebook 10 and writes: `tab_levels_moments` (mean, std, min, max of the three levels and the SPY return moments), `tab_johansen` (trace statistic and critical value per rank, from `statsmodels.tsa.vector_ar.vecm.select_coint_rank(...).test_stats` and `.crit_vals`), `tab_vecm_params` (beta and alpha per relation), `tab_block_iid` (the three p-values per variable of the block's residual layer and of the AR(1) residuals), `tab_residual_corr` (correlation of the residual layer under the VECM and under the AR(1) filters), `tab_daily_families` -> `tab_block_families` (the vine edges), `tab_paths_quantiles` (5, 50, 95 percent of DGS2, DGS10, T10YIE at months 1, 12, 24 under both dynamics), `tab_spread` (mean and standard deviation of the 2s10s spread at month 24 under both dynamics and in the history), `fig_levels.png`, `fig_fan_dgs10.png` (VECM and AR(1) side by side), `block_market.json`.

- [ ] **Step 4: Write `paper-blocks.tex`** with the skeleton's sections, in the style and length of `paper-selection.tex`, every number read from the tables of Step 3:
   - Introduction: levels versus returns; what a block filter is; the principle paper's block-filter paragraph; the plan of the paper.
   - The generator in brief: the same text as in the other short papers.
   - Vector autoregressions and error correction as filters: equation of the VECM in levels with `Pi = alpha beta'`, the Johansen rank test, the lag choice, the standardized innovations as the residual layer, `n_params`, the recursion and its inverse with the state of `q + 1` rows.
   - The residual layer of a block: why the innovation cross-correlation is left to the vine (one estimation of all dependence, tail dependence between block innovations and asset residuals), the Proposition of `\artfour{}` restated for a block (conditional law of the block given the past is the VECM's with the vine's marginals).
   - Paths in levels: from the last curve; what error correction does to spreads along a path; unconditional paths by burn-in.
   - Example: the tables of Step 3 in order, with the readings: the rank and the cointegrating vector (a 2s10s-type relation and the breakeven), the loadings (which variable adjusts), the residual tests, the correlation of the innovations against the correlation of AR(1) residuals of the levels, the families the vine picks on the innovations, the fan chart, the spread's dispersion at 24 months under the two dynamics.
   - Conclusion: what the block adds, what it does not (no GARCH on innovations, no exogenous regressors), the structural layer as the next paper.

- [ ] **Step 5: Compile the paper and run the suite**

Run: in the Overleaf project folder, `pdflatex paper-blocks; biber paper-blocks; pdflatex paper-blocks; pdflatex paper-blocks`, zero errors, no `??`; `python -m pytest tests -q` and the venv311 suite.

- [ ] **Step 6: Commit and stage**

```bash
git add docs/userguide.rst docs/api.rst docs/method.rst CHANGELOG.md README.md
git commit -m "docs: block filters in the user guide, API, changelog and README"
```

Re-zip `OldVersion/overleaf-path-simulation.zip`; stage the changed package files in `~/Desktop/CVineMarketGen-upload-0.3.0/` for the user's upload.

---

## Self-review

- **Spec coverage:** models (Task 1), container, parser, markets, JSON keys (Task 2), FRED loader (Task 3), notebook 10 with rank test, residual tests, vine on four residuals, paths in levels, AR(1) comparison (Task 4), export script and paper (Task 5). The spec's `filter_new`, `simulate`, `refit`, `name`, `spec`, `kind`, `state` are all defined in Task 1.
- **Placeholders:** none; the paper's prose is written in Task 5 from the tables, with its section-by-section content stated.
- **Type consistency:** `AssetDynamics.unfilter` slices `Z[:, :, idx]` for blocks and passes `(n_paths, horizon, p)` to `BlockVECM.unfilter`, which returns the same shape; `filter()` of a block returns a DataFrame and of a univariate model a Series, and `AssetDynamics.filter` handles both; `select_dynamics` sets `columns` so `unfilter` works for univariate-only containers.
