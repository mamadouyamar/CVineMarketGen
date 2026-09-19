# GDP Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quarterly GDP growth from monthly paths through a bridge equation; notebook 14 and `paper-gdp.tex`.

**Architecture:** New module `bridge.py` (`Bridge`, `growth_to_level`, `recession_probability`) and a quarterly FRED loader; the monthly market is unchanged (block + SPY with `exclude`).

**Tech Stack:** numpy, pandas, pytest, statsmodels for the block, FRED data.

**Spec:** `docs/superpowers/specs/2026-09-19-gdp-bridge-design.md`

## Global Constraints

- Python >= 3.8; both stacks pass; commit locally only.
- Growth in percent annualized (`400 * dlog` quarterly, `1200 * dlog` monthly).

---

### Task 1: `bridge.py`, `load_fred_quarterly`, tests

**Files:** Create `cvinemarketgen/bridge.py`, `tests/test_bridge.py`; Modify `cvinemarketgen/data.py`, `cvinemarketgen/__init__.py`, `tests/test_data_offline.py`.

**Interfaces:** `Bridge(parents, lags=1).fit(y_q, X_m)`; `.simulate(P, y_hist, seed=None) -> Paths('gdp')`; `growth_to_level(Pq, base=100.0) -> ndarray (n_paths, quarters)`; `recession_probability(Pq, k=2) -> (float, ndarray by quarter)`; `load_fred_quarterly(ids, start='1990Q1', end=None, cache='data/fred_cache_q.csv', refresh=False, verbose=True) -> DataFrame PeriodIndex('Q')`.

- [ ] **Step 1: Tests** (synthetic: monthly parents AR(1), quarterly target from the equation with known coefficients; assert recovery within 0.1, `simulate` shape `(n, h // 3, 1)`, level at quarter 1 equals `100 * exp(g / 400)`, recession probability 0 for all-positive paths and 1 for all-negative, JSON round trip equal coefficients). Loader test with a fake `fred_series`.
- [ ] **Step 2: Implement.** `fit`: `Xq = X_m.groupby(X_m.index.asfreq('Q')).agg(['mean', 'count'])`, keep quarters with count 3; join with `y_q` and its lags; OLS by `np.linalg.lstsq` with t-statistics from the classical covariance; residual moments -> `fit_johnson_su` after `_floor` (`max(kurt, 3.1 + 2 skew^2)`). `simulate`: `A = P.array[:, :h3, idx]` reshaped `(n, h // 3, 3, p)` -> mean over the month axis; loop over quarters with lag state from `y_hist[-lags:]`; innovation `johnson_su_sample(self.residual, n * Q, seed)` reshaped.
- [ ] **Step 3: Both stacks pass; commit** `feat: bridge equation for quarterly GDP from monthly paths, quarterly FRED loader`.

### Task 2: Notebook 14 (cells as in the spec), README row 14 ("Fourteen"), `docs/examples.rst`; commit `docs: notebook 14, GDP from monthly paths through a bridge equation`.

### Task 3: Docs (userguide section "Quarterly output from monthly paths", api section "Bridge", method bullet, changelog, README paragraph), `paper_gdp_export.py` -> `figs/gdp/` (tables: data moments, bridge equation (all quarters and without 2020Q2-Q3), residual tests, Johansen, dynamics report, residual moments/corr, families, growth and level quantiles, recession probabilities, scenario, with/without pandemic comparison; figures: series, fitted vs actual, fans, scenario), `paper-gdp.tex`, compile, re-zip, restage, memory; commit `docs: GDP bridge in the user guide, API, method, changelog and README`.
