# -*- coding: utf-8 -*-
"""
Choosing the dynamics of each asset: i.i.d. tests on the residual layer,
BIC among the candidates that pass, and a parametric-bootstrap Cramér-von
Mises goodness-of-fit test of the selected model, as GenHMM1d's ``GofHMMGen``
(the HMM candidates themselves are estimated by GenHMM1d).

Notation: ``epsilon_t`` is the standardized residual of a univariate model
and ``v_t = Phi(epsilon_t)`` its Rosenblatt uniform.
"""
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
    the statistic on the Rosenblatt uniforms ``v_t = Phi(epsilon_t)`` of the
    sample, against ``B`` refits on series simulated from the model, as
    GenHMM1d's ``GofHMMGen``. Returns ``stat``, ``pvalue`` and the bootstrap ``stats``.
    """
    u = stats.norm.cdf(np.asarray(model.filter(), float))
    stat = cvm_statistic(u)
    n = len(pd.Series(y))
    out = np.empty(B)
    for b in range(B):
        ys = model.simulate(n, seed=None if seed is None else seed + b)
        try:
            mb = model.refit(pd.Series(ys))
            out[b] = cvm_statistic(stats.norm.cdf(np.asarray(mb.filter(), float)))
        except Exception:
            out[b] = np.nan
        if verbose and (b + 1) % 10 == 0:
            print(f'  bootstrap {b + 1}/{B}')
    ok = np.isfinite(out)
    return {'stat': stat, 'pvalue': float(np.mean(out[ok] > stat)) if ok.any() else np.nan, 'stats': out}


# ---- candidates and selection -------------------------------------------------
def candidate_models(candidates=('const', 'garch', 'gjr', 'egarch', 'hmm'), means=('const', 'ar1'),
                     pq=(2, 2), states=(2, 3)):
    """
    Unfitted models of every candidate specification: for each variance family,
    every mean in ``means`` and orders ``p = 1..pq[0]``, ``q = 1..pq[1]``; for
    ``'hmm'``, one model per number of regimes in ``states``. The defaults give 28.
    """
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


REPORT_COLUMNS = ['model', 'mean', 'vol', 'p', 'q', 'states', 'n_params', 'loglik', 'bic',
                  'lb_z', 'lb_z2', 'arch_lm', 'passed', 'cvm', 'gof_pvalue', 'n_passed', 'n_candidates']


def select_dynamics(returns, candidates=('const', 'garch', 'gjr', 'egarch', 'hmm'), means=('const', 'ar1'),
                    pq=(2, 2), states=(2, 3), alpha=0.05, lags=20, gof=True, B=100, seed=0, verbose=True):
    """
    Per asset: fit every candidate, test its residual layer for i.i.d.-ness,
    keep the lowest BIC among the candidates that pass (all three p-values
    above ``alpha``); if none passes, the lowest BIC overall with a warning.
    With ``gof``, the bootstrap Cramér-von Mises test is run on the selected
    model. ``pq`` gives the largest ``p`` and ``q`` of the variance models. Returns an :class:`~cvinemarketgen.dynamics.AssetDynamics` whose
    ``report`` has one row per asset and ``candidates`` every fit.
    """
    h = pd.DataFrame(returns).astype(float)
    rows, cands, chosen = [], [], {}
    for a in h.columns:
        fits = []
        for m in candidate_models(candidates, means, pq, states):
            try:
                m.fit(h[a])
            except Exception as e:                      # non-convergence: skip the candidate
                if verbose:
                    print(f'{a}: {m.name} failed ({e.__class__.__name__}), skipped')
                continue
            t = iid_tests(np.asarray(m.filter(), float), lags)
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
        cands.append(tab.drop(columns='_model'))
        chosen[a] = best['_model']
        if verbose:
            print(f'{a}: {row["model"]}  BIC {row["bic"]:.1f}  p-values LB {row["lb_z"]:.2f} / LB2 {row["lb_z2"]:.2f} / ARCH {row["arch_lm"]:.2f}'
                  + (f'  GoF {row["gof_pvalue"]:.2f}' if gof else ''))
    ad = AssetDynamics()
    ad.models = chosen
    ad.columns = list(h.columns)
    ad.candidates = pd.concat(cands, ignore_index=True)
    ad.report = pd.DataFrame(rows).set_index('asset')[REPORT_COLUMNS]
    return ad
