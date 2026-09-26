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


def _jsu_on(eps):
    """Johnson SU fitted to the skewness and kurtosis (floored at the family's range) of a residual series, with its mean and scale."""
    from .functions import fit_johnson_su
    e = pd.Series(np.asarray(eps, float))
    skew, kurt = float(e.skew()), float(e.kurt() + 3.0)
    return fit_johnson_su(skew, max(kurt, 3.1 + 2.0 * skew ** 2), mean=float(e.mean()), vol=float(e.std(ddof=1)))


def gof_bootstrap(model, y, B=100, seed=0, verbose=False, innovation='auto'):
    """
    Parametric-bootstrap Cramér-von Mises goodness-of-fit test of a fitted
    univariate model, the test of the master's thesis and of GenHMM1d's
    ``GofHMMGen``: the statistic on the uniforms of the fitted sample against
    ``B`` refits on series simulated from the fitted model.

    ``innovation`` is what the model's generalized error is taken to follow.
    ``'gaussian'``: uniforms ``Phi(epsilon_t)``, simulation with normal errors;
    the natural choice for the Gaussian HMM, whose Rosenblatt uniforms are
    uniform under the model. ``'jsu'``: a Johnson SU fitted to the residual's
    skewness and kurtosis gives the uniforms ``G(epsilon_t)``; the bootstrap
    simulates the model with Johnson SU errors and refits both the model and
    the marginal on each simulated series. This tests what the generator
    simulates (a GARCH filter with a Johnson SU residual layer), where the
    Gaussian version rejects every GARCH whose only defect is a skewed error.
    ``'auto'`` (default): ``'jsu'`` for the GARCH family, ``'gaussian'`` for the HMM.

    Returns ``stat``, ``pvalue`` (share of bootstrap statistics above the
    observed one), the bootstrap ``stats`` and the ``innovation`` used.
    """
    from .functions import johnson_su_cdf, johnson_su_sample
    if innovation == 'auto':
        innovation = 'gaussian' if getattr(model, 'kind', 'garch') == 'hmm' else 'jsu'
    n = len(pd.Series(y))
    eps = np.asarray(model.filter(), float)
    if innovation == 'gaussian':
        stat = cvm_statistic(stats.norm.cdf(eps))
    else:
        p0 = _jsu_on(eps)
        stat = cvm_statistic(johnson_su_cdf(p0, eps))
    out = np.empty(B)
    for b in range(B):
        sb = None if seed is None else seed + b
        try:
            if innovation == 'gaussian':
                ys = model.simulate(n, seed=sb)
                mb = model.refit(pd.Series(ys))
                out[b] = cvm_statistic(stats.norm.cdf(np.asarray(mb.filter(), float)))
            else:
                e = johnson_su_sample(p0, n, seed=sb)
                ys = model.unfilter(e[None, :])[0]
                mb = model.refit(pd.Series(ys))
                eb = np.asarray(mb.filter(), float)
                out[b] = cvm_statistic(johnson_su_cdf(_jsu_on(eb), eb))
        except Exception:
            out[b] = np.nan
        if verbose and (b + 1) % 10 == 0:
            print(f'  bootstrap {b + 1}/{B}')
    ok = np.isfinite(out)
    return {'stat': stat, 'pvalue': float(np.mean(out[ok] > stat)) if ok.any() else np.nan, 'stats': out, 'innovation': innovation}


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
                    pq=(2, 2), states=(2, 3), criterion='gof', alpha=0.05, lags=20, B=100, seed=0, verbose=True, gof=None):
    """
    Per asset: fit every candidate, test it, keep the lowest BIC among the
    candidates that pass; if none passes, the lowest BIC overall with a warning.

    ``criterion='gof'`` (default, the rule of the master's thesis): the test is
    the parametric-bootstrap Cramér-von Mises goodness-of-fit test of
    :func:`gof_bootstrap` with ``B`` resamples, run on every candidate, and a
    candidate passes when its p-value is at least ``alpha``; the Ljung-Box and
    ARCH-LM tests of :func:`iid_tests` are reported as diagnostics.
    ``criterion='iid'``: the three i.i.d. tests on the generalized error decide
    admissibility (the rule of 0.3.0) and no bootstrap is run. ``pq`` gives the
    largest ``p`` and ``q`` of the variance models. Returns an
    :class:`~cvinemarketgen.dynamics.AssetDynamics` whose ``report`` has one row
    per asset and ``candidates`` every fit with its tests.
    """
    if gof is not None:                                   # 0.3.0 keyword
        criterion = 'gof' if gof else 'iid'
    if criterion not in ('gof', 'iid'):
        raise ValueError("criterion must be 'gof' or 'iid'")
    h = pd.DataFrame(returns).astype(float)
    rows, cands, chosen = [], [], {}
    for a in h.columns:
        fits = []
        models = candidate_models(candidates, means, pq, states)
        for j, m in enumerate(models):
            try:
                m.fit(h[a])
            except Exception as e:                      # non-convergence: skip the candidate
                if verbose:
                    print(f'{a}: {m.name} failed ({e.__class__.__name__}), skipped')
                continue
            t = iid_tests(np.asarray(m.filter(), float), lags)
            sp = m.spec
            row = {'asset': a, 'model': m.name, 'mean': sp.get('mean', ''), 'vol': sp.get('vol', 'hmm'),
                   'p': sp.get('p', 0), 'q': sp.get('q', 0), 'states': sp.get('n_states', 0),
                   'n_params': m.n_params, 'loglik': m.loglik, 'bic': m.bic, **t}
            if criterion == 'gof':
                if verbose:
                    print(f'{a}: {m.name} ({j + 1}/{len(models)}) bootstrap goodness-of-fit, B={B}', flush=True)
                g = gof_bootstrap(m, h[a], B=B, seed=seed, verbose=False)
                row['cvm'], row['gof_pvalue'] = g['stat'], g['pvalue']
                row['passed'] = bool(np.isfinite(g['pvalue']) and g['pvalue'] >= alpha)
            else:
                row['cvm'], row['gof_pvalue'] = np.nan, np.nan
                row['passed'] = all(v > alpha for v in t.values())
            row['_model'] = m
            fits.append(row)
        if not fits:
            raise RuntimeError(f'no candidate could be fitted for {a!r}')
        tab = pd.DataFrame(fits)
        ok = tab[tab['passed']]
        if len(ok):
            best = ok.sort_values('bic').iloc[0]
        else:
            best = tab.sort_values('bic').iloc[0]
            which = 'the goodness-of-fit test' if criterion == 'gof' else 'the i.i.d. tests'
            warnings.warn(f'{a}: no candidate passes {which} at {alpha}; keeping {best["model"]} (lowest BIC)')
        row = best.drop('_model').to_dict()
        row['n_passed'], row['n_candidates'] = int(tab['passed'].sum()), len(tab)
        rows.append(row)
        cands.append(tab.drop(columns='_model'))
        chosen[a] = best['_model']
        if verbose:
            print(f'{a}: {row["model"]}  BIC {row["bic"]:.1f}  p-values LB {row["lb_z"]:.2f} / LB2 {row["lb_z2"]:.2f} / ARCH {row["arch_lm"]:.2f}'
                  + (f'  GoF {row["gof_pvalue"]:.2f}' if criterion == 'gof' else ''))
    ad = AssetDynamics()
    ad.models = chosen
    ad.columns = list(h.columns)
    ad.candidates = pd.concat(cands, ignore_index=True)
    ad.report = pd.DataFrame(rows).set_index('asset')[REPORT_COLUMNS]
    return ad
