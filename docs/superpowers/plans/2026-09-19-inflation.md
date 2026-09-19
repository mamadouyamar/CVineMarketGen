# Inflation as a Child Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CPI inflation as a structural child of oil, unemployment and expectations in the market; price levels, year-on-year rates and real returns along the paths; notebook 13 and `paper-inflation.tex`.

**Architecture:** No new model: `Structural` in `linear` form with lags on the monthly annualized inflation rate; parents as a VECM block. Three helper functions in `paths.py` turn inflation paths into price levels, year-on-year rates and real returns.

**Tech Stack:** numpy, pandas, pytest, statsmodels (block), FRED and Yahoo data through the loaders.

**Spec:** `docs/superpowers/specs/2026-09-19-inflation-design.md`

## Global Constraints

- Python >= 3.8; both stacks pass (`python -m pytest tests -q`, venv311 likewise); commit locally only.
- Inflation in percent, month over month annualized (`scale=1200` per log unit).

---

### Task 1: `price_level`, `deflate`, `yoy` in `paths.py`

**Files:** Create `tests/test_paths.py`; Modify `cvinemarketgen/paths.py`, `cvinemarketgen/__init__.py` (export the three).

**Interfaces:** `price_level(P, column, base=100.0, scale=1200.0) -> ndarray (n_paths, horizon)`; `deflate(P, asset, inflation, scale=1200.0) -> Paths`; `yoy(P, column, history, scale=1200.0) -> ndarray (n_paths, horizon)` where `history` is a Series/array of the observed monthly annualized rates (last 11 used).

- [ ] **Step 1: Tests**

```python
import numpy as np, pandas as pd
from cvinemarketgen.paths import Paths, price_level, deflate, yoy

def test_price_level_and_yoy():
    pi = np.full((3, 24, 1), 2.4)                      # 2.4 percent a year, every month
    P = Paths(pi, ['pi'], layer='mixed')
    L = price_level(P, 'pi')
    assert L.shape == (3, 24) and np.allclose(L[:, 11], 100 * np.exp(0.024)) and np.allclose(L[:, 23], 100 * np.exp(0.048))
    hist = pd.Series(np.full(30, 2.4))
    Y = yoy(P, 'pi', hist)
    assert Y.shape == (3, 24) and np.allclose(Y, 100 * (np.exp(0.024) - 1))
    hist2 = pd.Series(np.full(30, 0.0))
    Y2 = yoy(P, 'pi', hist2)
    assert np.isclose(Y2[0, 0], 100 * (np.exp(0.002) - 1)) and np.isclose(Y2[0, 11], 100 * (np.exp(0.024) - 1))

def test_deflate():
    A = np.zeros((2, 6, 2)); A[:, :, 0] = 0.01; A[:, :, 1] = 12.0   # 1 percent a month nominal, 1 percent a month inflation
    P = Paths(A, ['SPY', 'pi'], layer='mixed')
    R = deflate(P, 'SPY', 'pi')
    assert isinstance(R, Paths) and R.assets == ['SPY (real)'] and R.array.shape == (2, 6, 1)
    assert np.allclose(R.array, 1.01 / np.exp(0.01) - 1)
```

- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement** in `paths.py` (module-level functions after the class):

```python
def price_level(P, column, base=100.0, scale=1200.0):
    """Price index along each path from a column of annualized monthly log inflation in percent: ``(n_paths, horizon)``."""
    x = P.array[:, :, P.assets.index(column)] / scale
    return base * np.exp(np.cumsum(x, axis=1))

def yoy(P, column, history, scale=1200.0):
    """Year-on-year inflation (percent) along the paths; the last 11 observed monthly rates in ``history`` complete the first windows."""
    x = P.array[:, :, P.assets.index(column)] / scale
    h = np.asarray(history, float)[-11:] / scale
    full = np.concatenate([np.tile(h, (x.shape[0], 1)), x], axis=1)
    c = np.cumsum(full, axis=1); c = np.concatenate([np.zeros((x.shape[0], 1)), c], axis=1)
    s = c[:, 12:] - c[:, :-12]
    return 100.0 * (np.exp(s[:, -x.shape[1]:]) - 1.0)

def deflate(P, asset, inflation, scale=1200.0):
    """Real per-period returns of ``asset`` deflated by the ``inflation`` column: a ``Paths`` with one column."""
    r = P.array[:, :, P.assets.index(asset)]; x = P.array[:, :, P.assets.index(inflation)] / scale
    return Paths(((1.0 + r) / np.exp(x) - 1.0)[:, :, None], [f'{asset} (real)'], layer='returns')
```

- [ ] **Step 4: Both stacks pass. Step 5: Commit** `feat: price level, year-on-year rate and real returns from inflation paths`.

---

### Task 2: Notebook 13

**Files:** Create `examples/13_inflation.ipynb`; Modify `README.md` (row 13, Thirteen), `docs/examples.rst`.

Cells (import cell of notebook 11 plus `Paths, price_level, deflate, yoy`): title; load (`raw = load_fred_monthly(['CPIAUCSL','CPILFESL','UNRATE','DCOILWTICO','MICH'], start='1993-09')`, build `pi`, `core`, `d_log_oil`, `unrate`, `mich`, join SPY since 1993-10, `dropna`); plots; headline equation `Structural(['d_log_oil','unrate','mich'], form='linear', lags=3).fit(data['pi'], data[parents])` with coefficient table (`m.const`, `m.gamma`, `m.phi`, `m.tstat`, `m.r2`, `m.sigma`); core equation compact; residual tests and correlations of both children; Johansen table (`coint_johansen`) on the block; market fit with the dynamics dict of the spec, `cv.dynamics.order`, `dynamics_report`, `cv.edges`; paths 1000 x 24, fans of `pi`, `yoy` and `price_level`, quantile table; oil scenario (`E` from `cv.simulate(..., accept=False)` in `cv.dynamics.columns` order, `+0.50*100/6/sigma_oil` on the oil innovation for six months, `unfilter`, medians baseline vs scenario for pi, core, price level at months 1, 3, 6, 12, 24); real SPY (`deflate`, cumulative at 12 and 24 months vs nominal); AR(1) alternative (`'pi': 'ar1'`: tests, residual correlations, yoy sd at 24 months); save `inflation_market.json`; where to go next.

Execute with nbconvert, check no error and the numbers; README/docs rows; commit `docs: notebook 13, inflation as a child of oil, unemployment and expectations`.

---

### Task 3: Docs, export script, paper

**Files:** Modify `docs/userguide.rst` (section "Inflation and real returns" after the yield curve), `docs/api.rst` (add the three functions to the Paths section), `docs/method.rst` (bullet), `CHANGELOG.md`, `README.md` (paragraph); Create `OldVersion/overleaf-path-simulation/paper_inflation_export.py` -> `figs/inflation/` (tables: data moments, headline equation, core equation, children iid (structural vs AR(1)), residual moments, residual corr (both markets), Johansen, VECM, families, inflation quantiles (pi, yoy, price level), scenario, real returns, yoy dispersion comparison; figures: series, fans, scenario, real vs nominal); write `paper-inflation.tex`; compile (pdflatex, biber, pdflatex x2); re-zip; restage upload folder; memory.

Commit `docs: inflation in the user guide, API, method, changelog and README`.
