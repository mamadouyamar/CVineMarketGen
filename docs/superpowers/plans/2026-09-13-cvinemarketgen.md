# CVineMarketGen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the paper's code (`OldVersion/momentmatchingscript.py` plus `OldVersion/fleishman_comparison.py`) as `CVineMarketGen`, a pip-installable Python package in the style of GenHMM1d, with a README and a runnable `examples.ipynb`.

**Architecture:** Three classes relocated from the script into three modules (`MomentMatch`, `CopulaTools`, `CVineGenerator`) plus one new class `FleishmanGenerator` built from the comparison script. Relocation is done by an AST-based splitter so that method bodies are copied verbatim. `CVineGenerator` inherits from the other two, exactly as `FinancialSimulationSystem` inherited from `MomentMatch` and `DistributionManager`.

**Tech Stack:** Python ≥ 3.8, numpy, scipy, pandas, matplotlib, pyvinecopulib 0.6.x, statsmodels (for `mvnormcdf`). Anaconda `python3` on this machine has all of them.

**Spec:** `docs/superpowers/specs/2026-09-13-cvinemarketgen-design.md`

## Global Constraints

- Source of truth: `/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion/momentmatchingscript.py` and `/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion/fleishman_comparison.py`. Method bodies are copied, not rewritten. Method names are kept.
- Package name `CVineMarketGen`, import name `cvinemarketgen`, MIT licence, author "Mamadou Yamar Thioub", email `mamadou-yamar.thioub@hec.ca` (as in GenHMM1d).
- No GFD/Finaeon data in the repository. Only `data/jpm_ltcma_2024.csv`.
- No tests folder (user decision). Each task ends with a verification command run from the package root with `python3`.
- Example: 4 assets (U.S. Large Cap, U.S. Long Treasuries, Commodities, Gold), seeded, under 5 minutes.
- All work in `/Users/mamadouthioub/Desktop/CopulaGenerator/CVineMarketGen`. Commit after each task. No push.

---

### Task 1: Package skeleton and LTCMA data file

**Files:**
- Create: `setup.py`, `LICENSE.txt`, `cvinemarketgen/__init__.py`, `data/jpm_ltcma_2024.csv`, `.gitignore`

**Interfaces:**
- Produces: `data/jpm_ltcma_2024.csv` with columns `Assets, Geometric Mean, Arithmetic Mean, Volatility, <11 asset names>` (same layout as `jpm-ltcmas-2024.xlsx`, correlation matrix symmetrised). Later tasks read it with `pd.read_csv(path, index_col='Assets')`.

- [ ] **Step 1: Write `setup.py`** (GenHMM1d style)

```python
import setuptools

with open('README.md') as f:
    long_description = ''.join(f.readlines())

setuptools.setup(
    name="CVineMarketGen",
    version="0.1.0",
    author="Mamadou Yamar Thioub",
    author_email="mamadou-yamar.thioub@hec.ca",
    description="C-vine copula financial market generator with moment and tail dependence targeting, "
                "and the Fleishman / Vale-Maurelli benchmark. Code of the third article of the thesis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mamadouyamar/CVineMarketGen",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['cvinemarketgen'],
    package_data={'cvinemarketgen': ['../data/jpm_ltcma_2024.csv']},
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'pyvinecopulib', 'statsmodels'],
    python_requires='>=3.8',
)
```

- [ ] **Step 2: Write `LICENSE.txt`** — MIT text with "Copyright (c) 2026 Mamadou Yamar Thioub".

- [ ] **Step 3: Write `cvinemarketgen/__init__.py`**

```python
"""CVineMarketGen: C-vine copula financial market generator (thesis, article 3)."""
from .moment_match import MomentMatch
from .copulas import CopulaTools
from .cvine import CVineGenerator
from .fleishman import FleishmanGenerator

__all__ = ['MomentMatch', 'CopulaTools', 'CVineGenerator', 'FleishmanGenerator']
__version__ = '0.1.0'
```
(The imports will fail until Tasks 2 to 5 exist; that is expected.)

- [ ] **Step 4: Write `.gitignore`** with `__pycache__/`, `*.pyc`, `*.egg-info/`, `.ipynb_checkpoints/`, `build/`, `dist/`.

- [ ] **Step 5: Build the CSV from the Excel file** (run once, from the package root):

```python
import pandas as pd, numpy as np
x = pd.read_excel('/Users/mamadouthioub/Desktop/CopulaGenerator/data/jpm-ltcmas-2024.xlsx').set_index('Assets')
corr = x.iloc[:, 3:].values.astype(float)
corr[np.triu_indices(corr.shape[0])] = np.tril(corr).T[np.triu_indices(corr.shape[0])]
x.iloc[:, 3:] = corr
x.to_csv('data/jpm_ltcma_2024.csv')
print(x.shape, list(x.columns[:4]))
```
Expected: `(11, 14) ['Geometric Mean', 'Arithmetic Mean', 'Volatility', 'U.S. Large Cap']` (or whatever the first asset column is; the 11 asset columns follow the three moment columns).

- [ ] **Step 6: Commit**

```bash
git add setup.py LICENSE.txt .gitignore cvinemarketgen/__init__.py data/jpm_ltcma_2024.csv docs/
git commit -m "chore: package skeleton, licence, LTCMA target data"
```

---

### Task 2: AST splitter and `moment_match.py`

**Files:**
- Create: `/private/tmp/.../scratchpad/split_script.py` (tool, not committed), `cvinemarketgen/moment_match.py`

**Interfaces:**
- Produces: `class MomentMatch` with every method of the script's `MomentMatch` (lines 353 to 1184) plus the seven JSU methods of `DistributionManager`: `moments_JSU`, `univariate_moments_matching_func_JSU`, `find_params_for_moments_matching_JSU`, `sim_JSU_with_U`, `moments_JSU_with_U`, `univariate_moments_matching_func_JSU_with_U`, `find_params_for_moments_matching_JSU_with_U`. Signatures unchanged.

- [ ] **Step 1: Write the splitter** in the scratchpad. It parses the script with `ast`, indexes every method of every class by `(class, name) -> source lines`, and writes a module from a header string plus a list of `(class, method)` pairs.

```python
import ast, sys
SRC = '/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion/momentmatchingscript.py'
lines = open(SRC).read().split('\n')
tree = ast.parse('\n'.join(lines))
methods = {}
for node in tree.body:
    if isinstance(node, ast.ClassDef):
        for m in node.body:
            if isinstance(m, ast.FunctionDef):
                start = m.lineno - 1
                if m.decorator_list:
                    start = m.decorator_list[0].lineno - 1
                methods[(node.name, m.name)] = lines[start:m.end_lineno]

def write_module(path, header, class_line, docstring, wanted):
    out = [header, '', class_line, f'    """{docstring}"""', '']
    for key in wanted:
        out.extend(methods[key]); out.append('')
    open(path, 'w').write('\n'.join(out) + '\n')

def names(cls):
    return [k[1] for k in methods if k[0] == cls]
```
Each module task below calls `write_module` with its own list.

- [ ] **Step 2: Generate `moment_match.py`**

```python
HEADER = '''# -*- coding: utf-8 -*-
"""
Moment matching tools: Fleishman cubic transform, Vale-Maurelli helpers,
Johnson SU moment fitting. Relocated verbatim from momentmatchingscript.py
(article 3 of the thesis).
"""
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize, fsolve, brentq
from scipy.integrate import dblquad
from statsmodels.sandbox.distributions.extras import mvnormcdf
import pyvinecopulib as pv
'''
JSU = ['moments_JSU', 'univariate_moments_matching_func_JSU', 'find_params_for_moments_matching_JSU',
       'sim_JSU_with_U', 'moments_JSU_with_U', 'univariate_moments_matching_func_JSU_with_U',
       'find_params_for_moments_matching_JSU_with_U']
wanted = [('MomentMatch', n) for n in names('MomentMatch')] + [('DistributionManager', n) for n in JSU]
write_module('cvinemarketgen/moment_match.py', HEADER, 'class MomentMatch:',
             'Fleishman, Vale-Maurelli and Johnson SU moment matching (article 3, Sections 3 and 4.3).', wanted)
```

- [ ] **Step 3: Verify** — fit Fleishman and JSU on one target:

```bash
python3 -c "
from cvinemarketgen.moment_match import MomentMatch
import numpy as np
mm = MomentMatch()
p, res = mm.find_params_for_moments_matching([0,1,-0.56,0.79], x0=[0,1,0,0], method='Nelder-Mead', distr='gauss')
print('fleishman', np.round(p,4), res.fun < 1e-6)
q, r2 = mm.find_params_for_moments_matching_JSU([0,1,-0.56,0.79], x0=[0,1,1.5,1], method='Nelder-Mead')
print('jsu', np.round(q,3), r2.fun < 1e-5)
"
```
Expected: two lines ending with `True`. If `pandas`/`pyvinecopulib` missing at import, use the Anaconda `python3`.

- [ ] **Step 4: Commit** — `git add cvinemarketgen/moment_match.py && git commit -m "feat: MomentMatch module (Fleishman, Vale-Maurelli, JSU)"`

---

### Task 3: `copulas.py` (CopulaTools)

**Files:**
- Create: `cvinemarketgen/copulas.py`

**Interfaces:**
- Consumes: splitter from Task 2.
- Produces: `class CopulaTools` with every `DistributionManager` method except the seven JSU methods (Task 2) and the following, which move to Task 4 or are dropped: `estimate_CVine_preselected_V2`, `simulate_CVine`, `U_last`, `thetas_for_Corr_all_in_CVine` (to Task 4); `U_last_pyvine`, `thetas_for_Corr_all_in_CVine_pyvine`, `estimate_CVine`, `estimate_CVine_preselected` (dropped, superseded versions). Plus one new method `exceedance_correlation(self, x, y, z_grid, min_obs=10)` (the function from `OldVersion/plot_cond_corr.py`).

- [ ] **Step 1: Generate `copulas.py`**

```python
HEADER = '''# -*- coding: utf-8 -*-
"""
Bivariate copula tools: exceedance correlation, non-central squared (NCS)
copulas, mixture copulas, h-functions and inverses, tail-dependence catalogs
and BIC selection. Relocated verbatim from momentmatchingscript.py.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, skew, kurtosis, bernoulli
from scipy.optimize import minimize, fsolve, brentq
from itertools import product
import pyvinecopulib as pv
'''
DROP = set(JSU) | {'U_last_pyvine', 'thetas_for_Corr_all_in_CVine_pyvine', 'estimate_CVine',
                   'estimate_CVine_preselected', 'estimate_CVine_preselected_V2',
                   'simulate_CVine', 'U_last', 'thetas_for_Corr_all_in_CVine'}
wanted = [('DistributionManager', n) for n in names('DistributionManager') if n not in DROP]
write_module('cvinemarketgen/copulas.py', HEADER, 'class CopulaTools:',
             'Bivariate building blocks of the C-vine (article 3, Section 4 and Appendix C).', wanted)
```
Then append to the class (by editing the file):

```python
    def exceedance_correlation(self, x, y, z_grid, min_obs=10):
        """Empirical exceedance correlation of (x, y) conditional on x only (kind=1
        of ConditionalCorrelation): both series standardized, Pearson correlation on
        the subsample x < z for z < 0 and x >= z for z >= 0. NaN where the subsample
        has fewer than min_obs points."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
        out = np.full(len(z_grid), np.nan)
        for i, z in enumerate(z_grid):
            cond = (x >= z) if z >= 0 else (x < z)
            if cond.sum() >= min_obs:
                out[i] = np.corrcoef(x[cond], y[cond])[0, 1]
        return out
```

- [ ] **Step 2: Verify** — mixture h-function / inverse round trip and one mixture MLE:

```bash
python3 -c "
import numpy as np, pyvinecopulib as pv
from cvinemarketgen.copulas import CopulaTools
ct = CopulaTools()
comps = [pv.Bicop(pv.BicopFamily.clayton, 270), pv.Bicop(pv.BicopFamily.gumbel, 0)]
params = np.array([0.7, 0.66, 1.84])
np.random.seed(1)
u = ct.simulate_mixture(params[0], [pv.Bicop(pv.BicopFamily.clayton, 270, [[0.66]]), pv.Bicop(pv.BicopFamily.gumbel, 0, [[1.84]])], n=3000, seed=1)
h = ct.hfunc1_mixture(u[:,0], u[:,1], params, comps)
back = ct.hinv1_mixture(u[:,0], h, params, comps)
print('roundtrip max err', np.abs(back - u[:,1]).max())
fit = ct.est_mixture_MLE(pseudo_obs=u, copulas_families=comps)
print('mle', np.round(fit.x, 3), 'BIC', round(fit['BIC'], 1))
"
```
Expected: round-trip error below 1e-6; MLE close to `[0.7, 0.66, 1.84]`. If `simulate_mixture`'s signature differs from the guess above, read it in `copulas.py` and adapt the call; do not change the method.

- [ ] **Step 3: Commit** — `git add cvinemarketgen/copulas.py && git commit -m "feat: CopulaTools module (NCS, mixtures, h-functions, selection catalogs)"`

---

### Task 4: `cvine.py` (CVineGenerator) with DataFrame inputs and synthetic-vine helper

**Files:**
- Create: `cvinemarketgen/cvine.py`

**Interfaces:**
- Consumes: `MomentMatch`, `CopulaTools`.
- Produces: `class CVineGenerator(MomentMatch, CopulaTools)` with, verbatim from the script: `estimate_CVine_preselected_V2`, `simulate_CVine`, `U_last`, `thetas_for_Corr_all_in_CVine`, `__init__`, `_fit_jsu_parameters`, `fit_and_structure_CVine`, `_make_bound_callback`, `run_vine_optimization`, `run_multi_year_simulation`, `run_complete_simulation`. Modified: `load_data_and_setup(self, ltcma, historical_data, asset_order=None)` taking DataFrames. New: `make_vine_spec(self, asset_order, edges)` and `simulate_known_vine(self, spec, jsu_params, mu, sigma, n, seed=None)`.
  - `edges` is a dict `{(i, k): ('gaussian', 0, 0.3)}` or `{(i, k): ('mixture', [('clayton', 270, 0.66), ('gumbel', 0, 1.84)], 0.7)}` with 1-based variable indices `k < i`, matching the paper's $\theta_{ik|1:k-1}$.
  - `make_vine_spec` returns the nine matrices `simulate_CVine` needs, in the layout `estimate_CVine_preselected_V2` uses (row = tree k-1, column = variable i-2): `thetas_1p, thetas_2p, a1, a2, fams, rots, ncsstatus, mixturestatus, familystatus`.
  - `simulate_known_vine` calls `simulate_CVine` then applies `mu + sigma * sim_JSU_with_U(u, jsu_params[asset])` per asset and returns a DataFrame with `asset_order` columns.

- [ ] **Step 1: Generate the verbatim part**

```python
HEADER = '''# -*- coding: utf-8 -*-
"""
C-vine copula financial market generator: family selection (Algorithm 3),
correlation-targeting calibration (Algorithm 4), scenario generation
(Algorithm 5). Relocated from momentmatchingscript.py; load_data_and_setup
takes DataFrames instead of Excel paths.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize
import pyvinecopulib as pv

from .moment_match import MomentMatch
from .copulas import CopulaTools


class _EarlyStopSLSQP(Exception):
    pass
'''
CV = ['estimate_CVine_preselected_V2', 'simulate_CVine', 'U_last', 'thetas_for_Corr_all_in_CVine']
FS = [n for n in names('FinancialSimulationSystem') if n not in ('load_data_and_setup', 'fit_and_structure_CVine_pyvine')]
wanted = [('DistributionManager', n) for n in CV] + [('FinancialSimulationSystem', n) for n in FS]
write_module('cvinemarketgen/cvine.py', HEADER, 'class CVineGenerator(MomentMatch, CopulaTools):',
             'C-vine generator (article 3, Sections 4.5 and 5).', wanted)
```
Then edit the file: inside the class, right after the docstring, add `    _EarlyStopSLSQP = _EarlyStopSLSQP` (the script's `run_vine_optimization` catches `self._EarlyStopSLSQP`, which only works with this attribute).

- [ ] **Step 2: Add `load_data_and_setup` taking DataFrames** (replaces the Excel-reading version; the body after data loading is the script's, unchanged):

```python
    def load_data_and_setup(self, ltcma, historical_data, asset_order=None):
        """
        ltcma           : DataFrame indexed by asset with columns 'Arithmetic Mean',
                          'Volatility' (optionally 'Geometric Mean') followed by the
                          correlation matrix columns named after the assets.
        historical_data : DataFrame of returns, one column per asset (used for
                          skewness, kurtosis and copula family selection).
        """
        print("Loading data and setting up simulation parameters...")
        historical_data = historical_data.ffill().dropna()
        ltcma = ltcma.copy()
        if 'Geometric Mean' not in ltcma.columns:
            ltcma.insert(0, 'Geometric Mean', ltcma['Arithmetic Mean'] - 0.5 * ltcma['Volatility'] ** 2)
        ltcma_moments = ltcma.loc[:, ['Geometric Mean', 'Arithmetic Mean', 'Volatility']].copy()
        ltcma_moments['Variance'] = ltcma_moments['Volatility'] ** 2
        ltcma_corr = ltcma.loc[:, [c for c in ltcma.columns if c in ltcma.index]].copy()
        ltcma_corr_mat = ltcma_corr.values.astype(float)
        ltcma_corr_mat[np.triu_indices(ltcma_corr_mat.shape[0])] = np.tril(ltcma_corr_mat).T[
            np.triu_indices(ltcma_corr_mat.shape[0])]
        ltcma_corr[:] = ltcma_corr_mat
        if asset_order is None:
            asset_order = list(historical_data.columns)
        # ---- from here on: the script's body, unchanged ----
        ltcma_corr = ltcma_corr.loc[asset_order, asset_order].copy()
        ltcma_moments = ltcma_moments.loc[asset_order, :]
        targeted_mom1 = ltcma_moments.loc[asset_order, 'Arithmetic Mean']
        targeted_mom2 = ltcma_moments.loc[asset_order, 'Variance']
        targeted_GM = ltcma_moments.loc[asset_order, 'Geometric Mean']
        targeted_SD = ltcma_moments.loc[asset_order, 'Volatility']
        targeted_mom3 = historical_data[asset_order].skew()
        targeted_mom4 = historical_data[asset_order].kurtosis() + 3
        targeted_mom3.name = 'Skewness'
        targeted_mom4.name = 'Kurtosis'
        optimal_params = self._fit_jsu_parameters(targeted_mom3, targeted_mom4)
        df_target_stats = targeted_GM.to_frame().join(targeted_SD)
        df_target_stats = df_target_stats.merge(targeted_mom1.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom2.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom3.to_frame(), left_index=True, right_index=True)
        df_target_stats = df_target_stats.merge(targeted_mom4.to_frame(), left_index=True, right_index=True)
        ltcma_corr_PD = self.make_positive_definite(ltcma_corr)
        historical_data = historical_data.loc[:, asset_order].copy()
        df_target_stats = df_target_stats.loc[asset_order, :].copy()
        return {'ltcma_corr': ltcma_corr_PD, 'historical_data': historical_data, 'asset_order': asset_order,
                'targeted_mom1': targeted_mom1, 'targeted_mom2': targeted_mom2,
                'targeted_mom3': targeted_mom3, 'targeted_mom4': targeted_mom4,
                'optimal_params': optimal_params, 'df_target_stats': df_target_stats}
```
And in `run_complete_simulation`, rename the two path parameters to `ltcma, historical_data` and pass them through.

- [ ] **Step 3: Add the synthetic-vine helpers**

```python
    _FAMILY = {'gaussian': pv.BicopFamily.gaussian, 'clayton': pv.BicopFamily.clayton,
               'gumbel': pv.BicopFamily.gumbel, 'joe': pv.BicopFamily.joe, 'frank': pv.BicopFamily.frank}

    def make_vine_spec(self, asset_order, edges):
        """
        Build the nine matrices consumed by simulate_CVine from a dict of edges
        {(i, k): spec}, 1-based, k < i, i.e. the copula C_{ik|1:k-1} of the paper.
        spec = (family, rotation, theta)                                for a single family
        spec = ('mixture', [(fam1, rot1, th1), (fam2, rot2, th2)], w)  for a two-component mixture
        """
        d = len(asset_order)
        shape = (d - 1, d - 1)
        thetas_1p = np.full(shape, np.nan, dtype=object)
        thetas_2p = np.full(shape, np.nan)
        a1 = np.full(shape, np.nan); a2 = np.full(shape, np.nan)
        fams = np.full(shape, np.nan, dtype=object)
        rots = np.zeros(shape, dtype=int)
        ncs = np.full(shape, False, dtype=object)
        mix = np.full(shape, False, dtype=object)
        status = np.full(shape, np.nan, dtype=object)
        for (i, k), spec in edges.items():
            r, c = k - 1, i - 2
            if spec[0] == 'mixture':
                comps = [pv.Bicop(self._FAMILY[f], rot) for f, rot, _ in spec[1]]
                thetas_1p[r, c] = np.array([spec[2]] + [th for _, _, th in spec[1]], float)
                fams[r, c] = comps; mix[r, c] = True; status[r, c] = 'mixture'
            else:
                fam, rot, th = spec
                thetas_1p[r, c] = float(th); fams[r, c] = self._FAMILY[fam]
                rots[r, c] = int(rot); status[r, c] = 'ordinary'
        return dict(thetas_1p=thetas_1p, thetas_2p=thetas_2p, a1_s=a1, a2_s=a2, fams=fams,
                    rotations=rots, ncsstatus=ncs, mixturestatus=mix, familystatus=status)

    def simulate_known_vine(self, spec, jsu_params, mu, sigma, asset_order, n, seed=None):
        """Simulate n returns from a known C-vine (spec from make_vine_spec) with
        Johnson SU marginals jsu_params[asset] = (gamma, xi, delta, lambda), then
        rescale to mean mu[asset] and volatility sigma[asset]."""
        if seed is not None:
            np.random.seed(seed)
        U = self.simulate_CVine(n=n, **spec)
        out = pd.DataFrame(index=range(n), columns=asset_order, dtype=float)
        for j, a in enumerate(asset_order):
            out[a] = mu[a] + sigma[a] * self.sim_JSU_with_U(U[:, j], np.asarray(jsu_params[a], float))
        return out
```

- [ ] **Step 4: Verify** — simulate a 4-asset known vine, then run the full pipeline on it (this is the notebook's core, timed):

```bash
python3 - <<'EOF'
import time, numpy as np, pandas as pd
from cvinemarketgen.cvine import CVineGenerator
assets = ['U.S. Large Cap', 'U.S. Long Treasuries', 'Commodities', 'Gold']
ltcma = pd.read_csv('data/jpm_ltcma_2024.csv', index_col='Assets').loc[assets, ['Arithmetic Mean', 'Volatility'] + assets]
gen = CVineGenerator(tol_opt=1e-10, n_samples=5000, use_ncs_on_deepertrees=False, use_ncs_on_firsttree=False,
                     use_mixture_on_firsttree=True, use_mixture_on_deepertrees=False, force_try_ncscopula=False)
edges = {(2,1): ('mixture', [('clayton', 270, 0.66), ('gumbel', 0, 1.84)], 0.70),
         (3,1): ('gumbel', 180, 1.38), (4,1): ('gaussian', 0, 0.05),
         (3,2): ('gaussian', 0, -0.15), (4,2): ('gaussian', 0, 0.30), (4,3): ('gaussian', 0, 0.20)}
jsu = {'U.S. Large Cap': (2.570, 2.179, 3.540, 2.646), 'U.S. Long Treasuries': (-0.096, -0.095, 2.279, 2.062),
       'Commodities': (0.611, 0.589, 2.092, 1.774), 'Gold': (-0.523, -0.516, 2.893, 2.675)}
spec = gen.make_vine_spec(assets, edges)
hist = gen.simulate_known_vine(spec, jsu, ltcma['Arithmetic Mean'], ltcma['Volatility'], assets, n=3000, seed=1)
t0 = time.time()
np.random.seed(1)
res = gen.run_complete_simulation(ltcma=ltcma, historical_data=hist, asset_order=assets, n_year=1, n_per_year=25000, corr_tol=2e-2)
print('elapsed', round(time.time()-t0), 's')
print(res['vine_results']['fams_status'])
EOF
```
Expected: runs to "SIMULATION COMPLETED SUCCESSFULLY", elapsed under 300 s, first-tree statuses `['mixture', 'ordinary', 'ordinary']`. If slower than 300 s, lower `n_samples` to 3000 and `n_per_year` to 10000 and record the timing for the notebook constants.

- [ ] **Step 5: Commit** — `git add cvinemarketgen/cvine.py && git commit -m "feat: CVineGenerator module with DataFrame inputs and synthetic-vine helper"`

---

### Task 5: `fleishman.py` (FleishmanGenerator)

**Files:**
- Create: `cvinemarketgen/fleishman.py`

**Interfaces:**
- Consumes: `MomentMatch` (`find_params_for_moments_matching`, `moments_cubic_transform`, `cubic_transform`, `make_positive_definite`).
- Produces: `class FleishmanGenerator(MomentMatch)` with `fit(self, mu, sigma, skew, exkurt, corr_target)` returning `dict(coef=DataFrame[a,b,c,d,residual,feasible], corr_Z=DataFrame, infeasible_pairs=list)` and storing them on `self`; `simulate(self, n, corr_tol=2e-2, seed=None, max_tries=500)` returning `(DataFrame returns, dict errors, coef_refit DataFrame)`. Logic copied from `OldVersion/fleishman_comparison.py` sections 1 to 3.

- [ ] **Step 1: Write the module** — the three sections of `fleishman_comparison.py` become the two methods; module-level constants become arguments; `mm.` becomes `self.`; the `vale_maurelli_root` and `empirical_moment_residuals` helpers become private methods `_vale_maurelli_root` and `_empirical_moment_residuals`. No numerical change.

- [ ] **Step 2: Verify** on the paper's 11 assets, comparing with `OldVersion/fleishman_results.xlsx`:

```bash
python3 - <<'EOF'
import numpy as np, pandas as pd, pickle
from cvinemarketgen.fleishman import FleishmanGenerator
d = pickle.load(open('/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion/results_2026-01-12_12-28-03','rb'))
sd = d['setup_data']; ao = d['asset_order']; ts = sd['df_target_stats']
fg = FleishmanGenerator()
out = fg.fit(ts['Arithmetic Mean'], ts['Volatility'], ts['Skewness'], ts['Kurtosis'] - 3, sd['ltcma_corr'])
ref = pd.read_excel('/Users/mamadouthioub/Desktop/CopulaGenerator/OldVersion/fleishman_results.xlsx', sheet_name='fleishman coefficients', index_col=0)
print('coef max diff', np.abs(out['coef'][['a','b','c','d']].values.astype(float) - ref[['a','b','c','d']].values.astype(float)).max())
print('infeasible', len(out['infeasible_pairs']))
sim, err, _ = fg.simulate(n=25000, corr_tol=2e-2, seed=1)
print('errors', {k: f'{v:.1e}' for k, v in err.items()})
EOF
```
Expected: coefficient max diff below 1e-8, 0 infeasible pairs, correlation error below 0.02, moment errors below 1e-5.

- [ ] **Step 3: Commit** — `git add cvinemarketgen/fleishman.py && git commit -m "feat: FleishmanGenerator (Fleishman + Vale-Maurelli with accept-reject)"`

---

### Task 6: README and `examples.ipynb`, executed end to end

**Files:**
- Create: `README.md`, `examples.ipynb`

**Interfaces:**
- Consumes: everything above. The notebook uses only public methods named in Tasks 2 to 5.

- [ ] **Step 1: Write the notebook** with `nbformat`, cells in this order (markdown then code for each):
  1. Title, Colab badge `https://colab.research.google.com/github/mamadouyamar/CVineMarketGen/blob/master/examples.ipynb`, one paragraph: every step simulates with known parameters, then selects, calibrates and simulates back; seeded; runtime.
  2. Setup cell: imports, Colab install fallback (`pip install -q git+https://github.com/mamadouyamar/CVineMarketGen.git`), `N_HIST = 3000`, `N_SIM = 25000`, `N_OPT = 5000`, `SEED = 1`, the 4 assets, LTCMA targets read from `data/jpm_ltcma_2024.csv` (with a raw-GitHub URL fallback for Colab), `gen = CVineGenerator(...)` as in Task 4 Step 4, a `show_moments(target, sim)` and `show_corr(target, sim)` printing helper.
  3. Synthetic market: the `edges` and `jsu` dicts of Task 4, `hist = gen.simulate_known_vine(...)`, print the true edge table.
  4. Family selection: `setup = gen.load_data_and_setup(ltcma, hist, assets)`; `fit = gen.fit_and_structure_CVine(setup)`; print true vs selected family and rotation per edge from `fit['fams_cops_name']`, `fit['fams_rots']`, `fit['fams_status']`.
  5. Calibration and simulation: `vine = gen.run_vine_optimization(setup, fit)`; `sims, calib, targets = gen.run_multi_year_simulation(vine, n_year=1, n_per_year=N_SIM, corr_tol=2e-2)`; print moments and correlation errors.
  6. Fleishman benchmark: `fg = FleishmanGenerator(); fg.fit(...)` on the same targets; `fsim, ferr, _ = fg.simulate(N_SIM, seed=SEED)`; same printout.
  7. Conditional correlation figure: `gen.exceedance_correlation` on the three matrices for the three pairs with U.S. Large Cap, thresholds −1 to +1, one legend.
  8. Closing markdown: what the figure shows (vine follows the synthetic truth, Fleishman flat), pointer to the chapter.

- [ ] **Step 2: Execute it** — `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 examples.ipynb` with the Anaconda `python3` kernel. Fix any failure in the package, never by editing outputs. Record total runtime in the title cell.

- [ ] **Step 3: Write `README.md`** in GenHMM1d's order: title and one-paragraph description naming the chapter; Colab badge line; "Start with the worked example" blockquote; table of model classes (Fleishman generator: `FleishmanGenerator.fit` / `.simulate`; C-vine generator: `CVineGenerator.simulate_known_vine`, `.fit_and_structure_CVine`, `.run_vine_optimization`, `.run_multi_year_simulation`; bivariate tools: `CopulaTools`); Installation (`pip install git+https://github.com/mamadouyamar/CVineMarketGen.git`, requirements); Quick start with the notebook's Fleishman snippet and its real printed output; Quick start C-vine snippet with real printed output; Data note (LTCMA public targets shipped, no licensed returns); References (the chapter, Fleishman 1978, Vale and Maurelli 1983, Johnson 1949, Czado 2019, Nasri 2020, Pan et al. 2024); Contributing and Contact copied from GenHMM1d.

- [ ] **Step 4: Verify** — `python3 -c "import cvinemarketgen; print(cvinemarketgen.__version__)"` and `pip install -e .` in a fresh shell succeed; `git status` shows only intended files.

- [ ] **Step 5: Commit** — `git add README.md examples.ipynb && git commit -m "docs: README and executed examples notebook"`

---

## Self-review

- Spec coverage: layout (T1), mapping and three deliberate changes (T2 to T4), Fleishman module (T5), notebook six steps and README (T6), out-of-scope respected (no tests folder, no push).
- Placeholders: none; each step has code or an exact command.
- Consistency: `make_vine_spec` keys match `simulate_CVine` keyword names (`thetas_1p, thetas_2p, a1_s, a2_s, fams, rotations, ncsstatus, mixturestatus, familystatus`); `exceedance_correlation` is defined in T3 and used in T6; `FleishmanGenerator.fit/simulate` signatures in T5 match the calls in T6.

---

### Task 7: Second example, macro factors

**Files:**
- Create: `data/macro_factors_targets.csv`
- Modify: `examples.ipynb` (append a section), `README.md` (model table row and a short paragraph)

**Interfaces:**
- Consumes: `CVineGenerator`, `FleishmanGenerator`, same calls as Task 6.
- Produces: `data/macro_factors_targets.csv` with the same layout as the LTCMA file (`Assets, Arithmetic Mean, Volatility, <6 factor names>`).

- [ ] **Step 1: Write the targets file** with these illustrative values (annual, decimal):

| Factor | Mean | Vol | Equity DM | Equity EM | Real Premia | Inflation | Credit | Commodity |
|---|---|---|---|---|---|---|---|---|
| Equity DM | 0.070 | 0.160 | 1 | 0.80 | 0.30 | −0.10 | 0.55 | 0.35 |
| Equity EM | 0.085 | 0.220 | 0.80 | 1 | 0.25 | 0.05 | 0.50 | 0.45 |
| Real Premia | 0.030 | 0.070 | 0.30 | 0.25 | 1 | 0.20 | 0.35 | 0.15 |
| Inflation | 0.025 | 0.020 | −0.10 | 0.05 | 0.20 | 1 | −0.05 | 0.40 |
| Credit | 0.045 | 0.080 | 0.55 | 0.50 | 0.35 | −0.05 | 1 | 0.25 |
| Commodity | 0.040 | 0.180 | 0.35 | 0.45 | 0.15 | 0.40 | 0.25 | 1 |

Check the matrix is positive definite before saving (`np.linalg.eigvalsh(...).min() > 0`).

- [ ] **Step 2: Append the notebook section** "Example 2: macro factors", same cells as Example 1: constants cell (factors, targets from CSV, JSU marginals with skewness −0.5/−0.7 for the two equity factors, +0.3 for Inflation and Commodity, 0 otherwise, kurtosis 4 to 5), synthetic market with a Clayton 270° + Gumbel mixture on (Equity DM, Inflation) and Gumbel 180° on the equity and credit pairs, selection, calibration and simulation, Fleishman benchmark, conditional correlation figure for the five pairs with Equity DM (a 2 by 3 grid with one empty panel hidden).

- [ ] **Step 3: Execute the notebook** again end to end (same command as Task 6 Step 2). Both examples must finish; record both runtimes in the title cell.

- [ ] **Step 4: README** — add a row "Macro-factor example (6 factors, illustrative targets)" to the model table and one sentence under Data.

- [ ] **Step 5: Commit** — `git add data/macro_factors_targets.csv examples.ipynb README.md && git commit -m "docs: second example, macro factors"`
