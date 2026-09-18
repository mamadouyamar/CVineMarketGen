# Structural Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A child variable driven by parent variables of the same market plus its own innovation (single-equation error-correction model in levels, or a linear model in returns), whose innovation is its residual layer and whose paths are rebuilt from the parents' simulated paths; notebook 11 on USDCAD and the short paper `paper-structural.tex`.

**Architecture:** New module `structural.py` with `Structural(parents, form, lags)` estimated by least squares (numpy only); `AssetDynamics` gains a dependency order (parents before children) used by `fit` and `unfilter`, and `parse_spec` understands `'ecm(p1, p2)'` and `'linear(p1, p2; lags=1)'`; markets unchanged beyond docstrings.

**Tech Stack:** numpy, pandas, pytest, nbformat/nbconvert, the export-script pattern of `paper_blocks_export.py`; FRED data through `load_fred_monthly`.

**Spec:** `docs/superpowers/specs/2026-09-18-structural-layer-design.md`

## Global Constraints

- Python >= 3.8; tests with the anaconda interpreter (`python -m pytest tests -q`) and the durable venv311 stack (`/Users/mamadouthioub/Desktop/CopulaGenerator/venv311/bin/python -m pytest tests -q`). CI runs bare `pytest` with the `test,garch,blocks` extras.
- No new dependency; statsmodels stays optional (the notebook's parent block uses it).
- No data file committed; the FRED cache is git-ignored.
- The article-3 engine is untouched. Commit locally only.
- Parents are taken as weakly exogenous (an assumption, stated in docstrings and paper); cycles between children raise `ValueError`.

---

### Task 1: `Structural` in `structural.py`

**Files:**
- Create: `cvinemarketgen/structural.py`
- Create: `tests/test_structural.py`

**Interfaces:**
- Produces: `Structural(parents, form='ecm', lags=1)`; after `fit(s: Series, Z: DataFrame)`: `child`, `parents`, `form`, `lags`, `const`, `kappa` (ecm), `b` (Series over parents, level coefficients, ecm), `theta` (Series, `-b / kappa`, ecm), `gamma` (Series over parents), `phi` (list of floats), `sigma`, `half_life`, `r2`, `tstat` (Series over the design columns), `n_obs`, `loglik`, `n_params`, `bic`, `state` (dict), `name`, `spec`, `kind = 'structural'`, `filter() -> Series`, `filter_new(s_new, Z_new) -> ndarray`, `unfilter(z: (n_paths, horizon), Zpath: (n_paths, horizon, k)) -> ndarray (n_paths, horizon)`, `simulate(n, seed, Zpath)`, `clone()`, `refit(s, Z)`, `to_dict()`, `from_dict(d)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_structural.py
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.structural import Structural


def _system(n=1500, seed=0, kappa=-0.1, theta=(0.5, -0.3), gamma=(0.2, 0.1), sigma=0.02):
    """Two random-walk parents and a child that error-corrects to theta' z with contemporaneous response gamma."""
    rng = np.random.default_rng(seed)
    rd = np.cumsum(0.1 * rng.standard_normal(n)); oil = np.cumsum(0.05 * rng.standard_normal(n))
    s = np.zeros(n); s[0] = theta[0] * rd[0] + theta[1] * oil[0]
    for t in range(1, n):
        ds = kappa * (s[t - 1] - theta[0] * rd[t - 1] - theta[1] * oil[t - 1]) + gamma[0] * (rd[t] - rd[t - 1]) + gamma[1] * (oil[t] - oil[t - 1]) + sigma * rng.standard_normal()
        s[t] = s[t - 1] + ds
    idx = pd.period_range('1990-01', periods=n, freq='M')
    return pd.Series(s, index=idx, name='fx'), pd.DataFrame({'rd': rd, 'oil': oil}, index=idx)


def test_ecm_recovers_parameters_and_names():
    s, Z = _system()
    m = Structural(['rd', 'oil'], form='ecm', lags=1).fit(s, Z)
    assert m.child == 'fx' and m.parents == ['rd', 'oil'] and m.name == 'ECM(fx | rd, oil)'
    assert abs(m.kappa + 0.1) < 0.03
    assert abs(m.theta['rd'] - 0.5) < 0.1 and abs(m.theta['oil'] + 0.3) < 0.1
    assert abs(m.gamma['rd'] - 0.2) < 0.05 and abs(m.gamma['oil'] - 0.1) < 0.05
    assert abs(m.sigma - 0.02) < 0.003 and 3 < m.half_life < 12 and 0 < m.r2 < 1
    assert m.tstat['s_l1'] < -3 and np.isfinite(m.bic) and m.n_params == 8
    z = m.filter()
    assert isinstance(z, pd.Series) and z.name == 'fx' and len(z) == len(s) - 2 and abs(z.std() - 1) < 1e-6


def test_ecm_unfilter_is_inverse_of_filter_new_and_round_trips():
    s, Z = _system(n=800)
    m = Structural(['rd', 'oil'], form='ecm', lags=2).fit(s, Z)
    rng = np.random.default_rng(1)
    Zpath = np.cumsum(rng.standard_normal((3, 24, 2)) * [0.1, 0.05], axis=1) + Z.iloc[-1].values
    z = rng.standard_normal((3, 24))
    Y = m.unfilter(z, Zpath)
    assert Y.shape == (3, 24) and np.isfinite(Y).all()
    assert np.allclose(m.filter_new(Y[1], Zpath[1]), z[1], atol=1e-9)
    m2 = Structural.from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(z, Zpath), Y) and m2.name == m.name
    assert m.simulate(10, seed=0, Zpath=Zpath[:1, :10]).shape == (10,)


def test_linear_without_lags_is_the_factor_map():
    rng = np.random.default_rng(2); n = 1000
    Z = pd.DataFrame(rng.standard_normal((n, 2)) * 0.03, columns=['f1', 'f2'])
    s = pd.Series(0.001 + 1.2 * Z['f1'] - 0.4 * Z['f2'] + 0.01 * rng.standard_normal(n), name='r')
    m = Structural(['f1', 'f2'], form='linear', lags=0).fit(s, Z)
    X = np.column_stack([np.ones(n), Z.values]); beta = np.linalg.lstsq(X, s.values, rcond=None)[0]
    assert np.allclose([m.const, m.gamma['f1'], m.gamma['f2']], beta) and m.phi == [] and m.name == 'Linear(r | f1, f2)'
    Zpath = np.zeros((2, 5, 2)); Y = m.unfilter(np.zeros((2, 5)), Zpath)
    assert np.allclose(Y, m.const)


def test_bad_form_and_positive_kappa_warning():
    with pytest.raises(ValueError):
        Structural(['a'], form='quadratic')
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_structural.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'cvinemarketgen.structural'`.

- [ ] **Step 3: Implement `cvinemarketgen/structural.py`**

```python
# -*- coding: utf-8 -*-
"""
The structural layer: a child variable driven by parent variables of the same
market plus its own innovation, which is the child's residual layer. On a path
the parents are simulated first and the child is rebuilt from them.

    ecm    (levels):  Delta s_t = c + kappa s_{t-1} + b' z_{t-1} + gamma' Delta z_t + sum_i phi_i Delta s_{t-i} + sigma eps_t
                      long-run relation theta = -b / kappa, adjustment kappa < 0
    linear (returns): s_t = c + gamma' z_t + sum_i phi_i s_{t-i} + sigma eps_t   (lags=0: the factor map)

Least squares, no dependency. Parents are taken as weakly exogenous: the child
does not feed back into them on the same date.
"""
import warnings

import numpy as np
import pandas as pd


class Structural:
    """
    Child on parents. ``parents`` are column names of the market's history;
    ``form`` is ``'ecm'`` (levels) or ``'linear'`` (returns); ``lags`` is the
    number of lagged changes (ecm) or lagged levels (linear) of the child.

    Attributes after ``fit``: ``child``, ``const``, ``kappa``, ``b``, ``theta``,
    ``gamma``, ``phi``, ``sigma``, ``half_life``, ``r2``, ``tstat``, ``n_obs``,
    ``loglik``, ``n_params``, ``bic``, ``state``.
    """

    kind = 'structural'

    def __init__(self, parents, form='ecm', lags=1):
        if form not in ('ecm', 'linear'):
            raise ValueError("form must be 'ecm' or 'linear'")
        self.parents = list(parents)
        self.form = form
        self.lags = int(lags)
        self.child = None
        self.const = self.kappa = self.sigma = None
        self.b = self.theta = self.gamma = None
        self.phi = []
        self.half_life = self.r2 = self.tstat = None
        self.state = None
        self._z = None
        self.n_obs = self.loglik = self.n_params = self.bic = None

    # ---- naming --------------------------------------------------------------
    @property
    def spec(self):
        return {'parents': self.parents, 'form': self.form, 'lags': self.lags}

    @property
    def name(self):
        head = 'ECM' if self.form == 'ecm' else 'Linear'
        return f"{head}({self.child or '?'} | {', '.join(self.parents)})"

    def __repr__(self):
        return f'Structural({self.name}{"" if self.sigma is None else ", fitted"})'

    def clone(self):
        return Structural(**self.spec)

    # ---- design --------------------------------------------------------------
    def _design(self, s, Z):
        cols = {'const': pd.Series(1.0, index=s.index)}
        if self.form == 'ecm':
            y = s.diff()
            cols['s_l1'] = s.shift(1)
            for p in self.parents:
                cols[f'{p}_l1'] = Z[p].shift(1)
            for p in self.parents:
                cols[f'd_{p}'] = Z[p].diff()
            for i in range(1, self.lags + 1):
                cols[f'd_s_l{i}'] = s.diff().shift(i)
        else:
            y = s
            for p in self.parents:
                cols[p] = Z[p]
            for i in range(1, self.lags + 1):
                cols[f's_l{i}'] = s.shift(i)
        X = pd.DataFrame(cols)
        ok = X.notna().all(axis=1) & y.notna()
        return y[ok], X[ok]

    def fit(self, s, Z):
        s = pd.Series(s).astype(float)
        Z = pd.DataFrame(Z)[self.parents].astype(float)
        self.child = s.name
        y, X = self._design(s, Z)
        coef, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        coef = pd.Series(coef, index=X.columns)
        e = y.values - X.values @ coef.values
        n, k = X.shape
        s2 = float(e @ e) / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.values.T @ X.values)))
        self.tstat = coef / se
        self.const = float(coef['const'])
        if self.form == 'ecm':
            self.kappa = float(coef['s_l1'])
            self.b = pd.Series([coef[f'{p}_l1'] for p in self.parents], index=self.parents)
            if self.kappa < 0:
                self.theta = -self.b / self.kappa
                self.half_life = float(np.log(0.5) / np.log(1.0 + self.kappa))
            else:
                warnings.warn(f'{self.name}: adjustment coefficient {self.kappa:.3f} is not negative, no error correction')
                self.theta = self.b * np.nan
                self.half_life = np.inf
            self.gamma = pd.Series([coef[f'd_{p}'] for p in self.parents], index=self.parents)
            self.phi = [float(coef[f'd_s_l{i}']) for i in range(1, self.lags + 1)]
        else:
            self.gamma = pd.Series([coef[p] for p in self.parents], index=self.parents)
            self.phi = [float(coef[f's_l{i}']) for i in range(1, self.lags + 1)]
        self.sigma = float(np.sqrt(s2))
        self._z = pd.Series(e / self.sigma, index=y.index, name=self.child)
        self.r2 = float(1.0 - (e @ e) / ((y.values - y.values.mean()) @ (y.values - y.values.mean())))
        self.n_obs = int(n)
        self.loglik = float(-0.5 * n * (np.log(2 * np.pi * (e @ e) / n) + 1.0))
        self.n_params = int(k + 1)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(n))
        L = self.lags + 1
        self.state = {'s': s.values[-L:].tolist(), 'z': Z.values[-1].tolist()}
        return self

    def refit(self, s, Z):
        return self.clone().fit(s, Z)

    # ---- recursion -----------------------------------------------------------
    def _step(self, s_hist, z_prev, z_now, eps):
        """Next child value from ``s_hist`` (n, lags + 1) most recent last, parents ``z_prev``/``z_now`` (n, k), innovations ``eps`` (n,)."""
        if self.form == 'ecm':
            s_prev = s_hist[:, -1]
            ds = self.const + self.kappa * s_prev + z_prev @ self.b.values + (z_now - z_prev) @ self.gamma.values
            d = np.diff(s_hist, axis=1)                                 # d[:, -i] is Delta s_{t-i}
            for i, ph in enumerate(self.phi, start=1):
                ds = ds + ph * d[:, -i]
            return s_prev + ds + self.sigma * eps
        val = self.const + z_now @ self.gamma.values
        for i, ph in enumerate(self.phi, start=1):
            val = val + ph * s_hist[:, -i]
        return val + self.sigma * eps

    def _hist0(self, n):
        L = max(self.lags + 1, 1)
        return np.tile(np.array(self.state['s'], float)[-L:], (n, 1))

    def unfilter(self, z, Zpath):
        """Child paths ``(n_paths, horizon)`` from innovation paths ``z`` and the parents' simulated paths ``Zpath (n_paths, horizon, k)``."""
        z = np.asarray(z, float); Zpath = np.asarray(Zpath, float)
        n, H = z.shape
        s_hist = self._hist0(n)
        z_prev = np.tile(np.array(self.state['z'], float), (n, 1))
        out = np.empty_like(z)
        for t in range(H):
            x = self._step(s_hist, z_prev, Zpath[:, t], z[:, t])
            out[:, t] = x
            s_hist = np.column_stack([s_hist[:, 1:], x]); z_prev = Zpath[:, t]
        return out

    def filter_new(self, s_new, Z_new):
        """Standardized innovations of new child values given the parents' new values, continuing from ``state``."""
        v = np.asarray(s_new, float).ravel(); Zn = np.asarray(Z_new, float).reshape(len(v), -1)
        s_hist = self._hist0(1); z_prev = np.array(self.state['z'], float)[None, :]
        eps = np.empty(len(v))
        for t in range(len(v)):
            pred = self._step(s_hist, z_prev, Zn[t:t + 1], np.zeros(1))[0]
            eps[t] = (v[t] - pred) / self.sigma
            s_hist = np.column_stack([s_hist[:, 1:], [v[t]]]); z_prev = Zn[t:t + 1]
        return eps

    def simulate(self, n, seed=None, Zpath=None):
        """One child series of length ``n`` with Gaussian innovations along the given parents' path ``Zpath (1, n, k)``."""
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z, Zpath)[0]

    def filter(self):
        return self._z

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'child': self.child, 'const': self.const, 'kappa': self.kappa,
                'b': None if self.b is None else self.b.tolist(), 'gamma': self.gamma.tolist(), 'phi': list(self.phi),
                'sigma': self.sigma, 'half_life': self.half_life, 'r2': self.r2, 'state': self.state,
                'n_obs': self.n_obs, 'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m.child, m.const, m.kappa, m.sigma = d['child'], d['const'], d['kappa'], d['sigma']
        m.b = None if d['b'] is None else pd.Series(d['b'], index=m.parents)
        m.theta = None if m.b is None or not m.kappa or m.kappa >= 0 else -m.b / m.kappa
        m.gamma = pd.Series(d['gamma'], index=m.parents)
        m.phi = [float(x) for x in d['phi']]
        m.half_life, m.r2, m.state = d['half_life'], d['r2'], d['state']
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m
```

Note: `half_life` is `inf` when `kappa >= 0` and JSON writes it as `Infinity`; Python's `json` reads it back.

- [ ] **Step 4: Run the tests on both stacks**

Run: `python -m pytest tests/test_structural.py -q` and the venv311 one.
Expected: 4 passed on both.

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/structural.py tests/test_structural.py
git commit -m "feat: structural layer, a child on parents with its own innovation (ECM and linear forms)"
```

---

### Task 2: Children in `AssetDynamics` and `parse_spec`

**Files:**
- Modify: `cvinemarketgen/dynamics.py` (`parse_spec`, `model_from_dict`, `AssetDynamics`)
- Modify: `cvinemarketgen/markets.py` (docstrings), `cvinemarketgen/__init__.py` (export `Structural`)
- Test: `tests/test_structural.py` (append), `tests/test_api.py` (append)

**Interfaces:**
- Consumes: `Structural` (Task 1); `AssetDynamics` with tuple keys (0.4.0 blocks).
- Produces: `parse_spec('ecm(rd, oil)')`, `parse_spec('linear(f1, f2; lags=1)')`; `AssetDynamics.order` (list of keys, parents before children); `AssetDynamics.children` (list of child column names).

- [ ] **Step 1: Append the failing tests**

```python
# tests/test_structural.py (append)
from cvinemarketgen.dynamics import AssetDynamics, parse_spec


def test_parse_structural_specs():
    m = parse_spec('ecm(rd, oil)'); assert m.form == 'ecm' and m.parents == ['rd', 'oil'] and m.lags == 1
    m = parse_spec('linear(f1, f2; lags=2)'); assert m.form == 'linear' and m.parents == ['f1', 'f2'] and m.lags == 2
    m = parse_spec('ecm(rd; lags=3)'); assert m.parents == ['rd'] and m.lags == 3


def test_container_orders_parents_before_children():
    pytest.importorskip('statsmodels')
    s, Z = _system(n=800)
    h = Z.copy(); h['fx'] = s; h['spy'] = 0.0003 + 0.01 * np.random.default_rng(3).standard_normal(len(h))
    h = h[['fx', 'spy', 'rd', 'oil']]                                   # child listed first on purpose
    ad = AssetDynamics({('rd', 'oil'): 'vecm(r=0,q=1)', 'fx': 'ecm(rd, oil)', 'spy': 'ar1'}).fit(h)
    assert ad.order[0] == ('rd', 'oil') and ad.order.index('fx') > ad.order.index(('rd', 'oil')) and ad.children == ['fx']
    Zr = ad.filter(h)
    assert list(Zr.columns) == ['fx', 'spy', 'rd', 'oil']
    Y = ad.unfilter(np.zeros((4, 12, 4)))
    assert Y.shape == (4, 12, 4) and np.isfinite(Y).all()
    # the child moves with its parents: shock the parents' innovations only
    Zs = np.zeros((4, 12, 4)); Zs[:, :, 2] = 3.0
    Ys = ad.unfilter(Zs)
    assert not np.allclose(Ys[:, :, 0], Y[:, :, 0]) and np.allclose(Ys[:, :, 1], Y[:, :, 1])
    ad2 = AssetDynamics.from_dict(ad.to_dict())
    assert ad2.order == ad.order and np.allclose(ad2.unfilter(np.zeros((1, 3, 4))), Y[:1, :3])
    with pytest.raises(ValueError):
        AssetDynamics({'a': 'ecm(b)', 'b': 'ecm(a)'}).fit(pd.DataFrame({'a': s.values, 'b': Z['rd'].values}))
```

```python
# tests/test_api.py (append)
def test_cvine_market_with_a_child_on_a_block(tmp_path):
    pytest.importorskip('statsmodels')
    from tests.test_structural import _system
    s, Z = _system(n=600)
    h = Z.copy(); h['fx'] = s; h['spy'] = 0.0003 + 0.01 * np.random.default_rng(5).standard_normal(len(h))
    t = Targets.from_history(h)
    cv = CVineMarket(t, central='spy', families='gaussian', n_opt=2000,
                     dynamics={('rd', 'oil'): 'vecm(r=0,q=1)', 'fx': 'ecm(rd, oil)', 'spy': 'ar1'}).fit()
    assert cv.dynamics_report.loc['fx', 'model'] == 'ECM(fx | rd, oil)'
    P = cv.simulate_paths(8, 12, seed=0)
    assert P.array.shape == (8, 12, 4) and np.isfinite(P.array).all()
    p = tmp_path / 'child.json'; cv.save(str(p))
    cv2 = CVineMarket.load(str(p))
    assert np.allclose(cv2.simulate_paths(2, 4, seed=1).array, cv.simulate_paths(2, 4, seed=1).array)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_structural.py tests/test_api.py -q -k "structural or child"`
Expected: FAIL, `ValueError: cannot parse dynamics spec 'ecm(rd, oil)'`.

- [ ] **Step 3: Implement in `dynamics.py`**

After `_BLOCK_RE`:

```python
_STRUCT_RE = re.compile(r'^(ecm|linear)\(([^;)]*)(?:;\s*lags\s*=\s*(\d+))?\)$')
```

In `parse_spec`, before the `_BLOCK_RE` match:

```python
    m = _STRUCT_RE.match(s)
    if m:
        from .structural import Structural
        parents = [p.strip() for p in m.group(2).split(',') if p.strip()]
        return Structural(parents, form=m.group(1), lags=int(m.group(3)) if m.group(3) else (1 if m.group(1) == 'ecm' else 0))
```

Note: `parse_spec` lower-cases the spec (`s = spec.strip().lower()`); parents' names must survive, so change the start of `parse_spec` to keep the original for the structural branch: `s = spec.strip()` and match `_STRUCT_RE` on `s` first, then continue with `s = s.lower()` for the other patterns.

In `model_from_dict`, add before the HMM fallback:

```python
    if d['kind'] == 'structural':
        from .structural import Structural
        return Structural.from_dict(d)
```

In `AssetDynamics`, add the dependency order and use it:

```python
    @property
    def children(self):
        return [k for k, m in self.models.items() if getattr(m, 'kind', None) == 'structural']

    def _dependency_order(self):
        """Keys with parents before children (Kahn's algorithm); a cycle raises."""
        owner = {c: k for k in self.models for c in self._vars(k)}
        deps = {}
        for k, m in self.models.items():
            ps = getattr(m, 'parents', None) or []
            missing = [p for p in ps if p not in owner]
            if missing:
                raise ValueError(f'{k!r}: parents {missing} are not columns of the history')
            deps[k] = {owner[p] for p in ps}
        order, done = [], set()
        while len(order) < len(deps):
            ready = [k for k in deps if k not in done and deps[k] <= done]
            if not ready:
                raise ValueError('the structural equations form a cycle')
            order += ready; done |= set(ready)
        return order
```

In `fit`, replace the loop `for k, m in self.models.items(): m.fit(...)` by:

```python
        self.order = self._dependency_order()
        for k in self.order:
            m = self.models[k]
            if getattr(m, 'kind', None) == 'structural':
                m.fit(h[k], h[m.parents])
            elif isinstance(k, tuple):
                m.fit(h[list(k)])
            else:
                m.fit(h[k])
```

(and set `self.order = None` in `__init__`). In `unfilter`, iterate `for k in self.order:` and add the child branch:

```python
            if getattr(m, 'kind', None) == 'structural':
                pidx = [self.columns.index(p) for p in m.parents]
                out[:, :, idx[0]] = m.unfilter(Z[:, :, idx[0]], out[:, :, pidx])
            elif isinstance(k, tuple):
                out[:, :, idx] = m.unfilter(Z[:, :, idx])
            else:
                out[:, :, idx[0]] = m.unfilter(Z[:, :, idx[0]])
```

`to_dict` adds `'order': [list(k) if isinstance(k, tuple) else k for k in self.order]`; `from_dict` recomputes `m.order = m._dependency_order()` after the models are restored. `__init__.py`: import and export `Structural`. `markets.py` docstrings: one sentence after the block sentence: "A variable driven by others is given as a child, ``'ecm(rate_diff, log_oil)'`` (levels) or ``'linear(f1, f2)'`` (returns); its parents are simulated first."

- [ ] **Step 4: Run the full suite on both stacks**

Run: `python -m pytest tests -q` and the venv311 suite.
Expected: all pass (56 plus 7 new).

- [ ] **Step 5: Commit**

```bash
git add cvinemarketgen/dynamics.py cvinemarketgen/markets.py cvinemarketgen/__init__.py tests/test_structural.py tests/test_api.py
git commit -m "feat: children in AssetDynamics, dependency order, 'ecm'/'linear' specs"
```

---

### Task 3: Notebook 11, USDCAD on the rate differential and oil

**Files:**
- Create: `examples/11_structural_layer_fx.ipynb` (built with `nbformat`, executed with nbconvert)
- Modify: `README.md` (row 11, "Ten" -> "Eleven"), `docs/examples.rst`

**Interfaces:**
- Consumes: `load_fred_monthly`, `load_etf_monthly`, `Targets`, `CVineMarket`, `Structural`, `iid_tests`.

- [ ] **Step 1: Build the cells** (import cell of notebook 10 with `from cvinemarketgen import load_fred_monthly, load_etf_monthly, Targets, CVineMarket, Structural, iid_tests`)

1. md: title "11 — The structural layer: USDCAD on the rate differential and oil"; the structural equation in words; parents simulated first; runtime about 5 minutes.
2. md "Load" / code: `raw = load_fred_monthly(['DEXCAUS', 'DGS10', 'IRLTLT01CAM156N', 'DCOILWTICO'], start='2003-01', cache=...)`; `spy = load_etf_monthly(['SPY'], cache=...)`; `data = pd.DataFrame({'log_fx': np.log(raw['DEXCAUS']), 'rate_diff': raw['DGS10'] - raw['IRLTLT01CAM156N'], 'log_oil': np.log(raw['DCOILWTICO'])}).join(spy, how='inner')`; `data.tail()`; a two-panel plot: USDCAD level; the rate differential and log oil.
3. md "The error-correction equation" / code: `m = Structural(['rate_diff', 'log_oil'], form='ecm', lags=1).fit(data['log_fx'], data[['rate_diff', 'log_oil']])`; `print(m.name, '| kappa', round(m.kappa, 4), '| half-life', round(m.half_life, 1), 'months | R2', round(m.r2, 3))`; a table of coefficients and t-statistics (`pd.DataFrame({'coefficient': coef_series, 't': m.tstat})` built from `m.const, m.kappa, m.b, m.gamma, m.phi` in the design order); `m.theta.round(3)` with a sentence: long-run elasticities.
4. md "The child's residual layer" / code: `z = m.filter(); pd.Series(iid_tests(z.values)).round(3)`; `z.describe().round(3)`.
5. md "The market: parents as a block, the child on them, SPY alongside" / code: `t = Targets.from_history(data)`; `cv = CVineMarket(t, central='SPY', families='auto', dynamics={('rate_diff', 'log_oil'): 'vecm(r=0,q=1)', 'log_fx': 'ecm(rate_diff, log_oil)', 'SPY': 'ar1-garch'}).fit()`; `cv.dynamics.order`; `cv.dynamics_report`; `cv.edges`.
6. md "Paths of the exchange rate from the last observed values" / code: `P = cv.simulate_paths(1000, 24, seed=1)`; fan chart of `np.exp(P.array[:, :, j])` for USDCAD with the last observed level; quantile table at months 1, 12, 24 for USDCAD (level), rate_diff, log_oil.
7. md "A scenario on a parent" / code: build a modified residual draw: `cv.simulate(1000 * 24, seed=1)` -> reshape to `(1000, 24, 4)` in the market's column order (use `cv.fit_targets.assets` for the order), then replace the oil column of the residual layer by a deterministic drift that brings log oil down 30 percent over 12 months (`-0.3 / 12 / sigma_oil` per month for `t < 12`, where `sigma_oil = cv.dynamics.models[('rate_diff', 'log_oil')].sigma[1]`), push through `cv.dynamics.unfilter`, and compare the USDCAD median path with the baseline; sentence on the short-run response via `gamma` and the long-run via `theta`.
8. md "Without the structure: an AR(1) on the exchange rate alone" / code: `cv1 = CVineMarket(t, central='SPY', families='auto', dynamics={('rate_diff', 'log_oil'): 'vecm(r=0,q=1)', 'log_fx': 'ar1', 'SPY': 'ar1-garch'}).fit()`; `P1 = cv1.simulate_paths(1000, 24, seed=1)`; the USDCAD median and 90 percent band at month 24 under both; the correlation of the simulated USDCAD change with the simulated oil change at month 12 under both (a scatter or a number).
9. md "Save" / code: `cv.save('fx_market.json'); CVineMarket.load('fx_market.json').simulate_paths(3, 6, seed=2)`.
10. md closing: the same object with `form='linear'` on returns is the factor map; the yield-curve notebook uses deterministic children.

- [ ] **Step 2: Execute and inspect**

Run: `cd examples && jupyter nbconvert --to notebook --execute --inplace 11_structural_layer_fx.ipynb --ExecutePreprocessor.timeout=1800`
Expected: no error cell; `kappa` negative with `|t| > 2`, `gamma['log_oil'] < 0`, the residual tests passing or the failing one named in the text, the scenario's USDCAD median above the baseline (a weaker Canadian dollar when oil falls), the AR(1) alternative showing no oil sensitivity.

- [ ] **Step 3: README and docs** (row 11 "USDCAD on the U.S. minus Canada 10-year differential and oil: the structural layer, a parent scenario, paths in levels"; counts to "Eleven"; toctree entry `examples/11_structural_layer_fx`).

- [ ] **Step 4: Commit**

```bash
git add examples/11_structural_layer_fx.ipynb README.md docs/examples.rst
git commit -m "docs: notebook 11, the structural layer on USDCAD"
```

---

### Task 4: Docs, changelog, export script and the paper

**Files:**
- Modify: `docs/userguide.rst` (section "The structural layer" after "Blocks of variables"), `docs/api.rst` (section "Structural layer"), `docs/method.rst` (bullet), `CHANGELOG.md` (0.4.0 bullets), `README.md` (paragraph)
- Create: `OldVersion/overleaf-path-simulation/paper_structural_export.py`, tables and figures in `figs/structural/`
- Modify: `OldVersion/overleaf-path-simulation/paper-structural.tex`

- [ ] **Step 1: User guide**

```rst
The structural layer
--------------------

.. code-block:: python

   cv = CVineMarket(t, central='SPY', families='auto',
                    dynamics={('rate_diff', 'log_oil'): 'vecm(r=0,q=1)',
                              'log_fx': 'ecm(rate_diff, log_oil)',
                              'SPY': 'ar1-garch'}).fit()
   cv.dynamics.order                 # parents before children
   cv.dynamics.models['log_fx'].theta   # long-run elasticities

A variable driven by others is a child: ``'ecm(p1, p2)'`` fits the
single-equation error-correction model in levels (adjustment ``kappa``, long-run
relation ``theta``, contemporaneous response ``gamma``), ``'linear(p1, p2)'`` the
linear model in returns, ``'linear(p1, p2; lags=0)'`` the plain factor map. The
child's innovation is its residual layer; on a path the parents are simulated
first, by their own models, and the child is rebuilt from them and from its own
draw. Parents are taken as weakly exogenous; cycles are refused.
```

- [ ] **Step 2: API, method, changelog, README**

`docs/api.rst`: section "Structural layer" with `.. automodule:: cvinemarketgen.structural` `:members: Structural`, after "Blocks". `docs/method.rst`: bullet "Structural filters: :class:`~cvinemarketgen.structural.Structural`, a child on parents plus its own innovation; parents simulated first." `CHANGELOG.md` 0.4.0: bullets for the structural layer and notebook 11. README: one paragraph after the block paragraph, the same example.

- [ ] **Step 3: Export script** `paper_structural_export.py` (pattern of `paper_blocks_export.py`, `OUT = figs/structural`): `tab_data_moments`, `tab_ecm` (coefficients, t-statistics, `theta`, half-life, R2), `tab_child_iid` (tests of the child's residual under the ECM and under the AR(1) alternative), `tab_residual_corr`, `tab_families`, `tab_fx_quantiles` (USDCAD level at months 1, 12, 24 under the structure and under the AR(1) alternative), `tab_scenario` (USDCAD median at months 3, 12, 24, baseline and oil down 30 percent), `fig_series.png`, `fig_fan_fx.png` (two panels), `fig_scenario.png`, `fx_market.json`.

- [ ] **Step 4: Write `paper-structural.tex`** in the style of `paper-blocks.tex`: introduction (a variable that is a function of others; parents and children; the principle paper's structural paragraph); the generator in brief; parents, children and exogeneity (the acyclic structure, weak exogeneity, the residual layer containing roots' innovations and children's `u_t`); the error-correction equation and its innovation (the equation, `theta`, `kappa`, `gamma`, estimation, the residual tests, the linear form as the factor map); paths of a dependent variable (parents first, the child's inverse, the proposition conditional on the parents, scenarios on a parent as a change of its residual draw); example (every table); conclusion (what is assumed, the VECM-with-exogenous extension, the yield curve as deterministic children).

- [ ] **Step 5: Compile, run the suites, commit, re-zip, stage**

```bash
git add docs/userguide.rst docs/api.rst docs/method.rst CHANGELOG.md README.md
git commit -m "docs: the structural layer in the user guide, API, changelog and README"
```

Re-zip `OldVersion/overleaf-path-simulation.zip`; restage `~/Desktop/CVineMarketGen-upload-0.3.0/` from `git diff --name-status origin/main HEAD`.

---

## Self-review

- **Spec coverage:** model with both forms, filter/inverse/JSON (Task 1); parse, order, cycles, container, market (Task 2); notebook with ECM table, tests, vine, paths, scenario, AR(1) comparison (Task 3); docs, export, paper (Task 4). `filter_new`, `simulate(Zpath)`, `refit`, `tstat`, `half_life`, `state` all defined in Task 1.
- **Placeholders:** none; the paper's prose is written from the tables of Task 4 Step 3, with its sections stated.
- **Type consistency:** `AssetDynamics.unfilter` passes `Z[:, :, idx[0]]` of shape `(n_paths, horizon)` and `out[:, :, pidx]` of shape `(n_paths, horizon, k)` to `Structural.unfilter(z, Zpath)`, which returns `(n_paths, horizon)`; `fit(h[k], h[m.parents])` matches `Structural.fit(s, Z)`; `parse_spec` keeps the case of parents' names.
