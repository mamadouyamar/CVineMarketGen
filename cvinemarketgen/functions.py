# -*- coding: utf-8 -*-
"""
Standalone functions for the pieces people want on their own: a marginal fit,
an exceedance curve, the classification of one pair, the family selection of
one pair. Thin wrappers over :class:`~cvinemarketgen.moment_match.MomentMatch`
and :class:`~cvinemarketgen.copulas.CopulaTools`.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .moment_match import MomentMatch
from .copulas import CopulaTools


class _Tools(MomentMatch, CopulaTools):
    pass


_mm = _Tools()
_ct = _mm


def fit_johnson_su(skew, kurt, mean=0.0, vol=1.0):
    """
    Johnson SU parameters matching a skewness and a raw kurtosis (normal = 3).

    Returns a dict with ``gamma, xi, delta, lambda`` for the *standardized*
    variable, the ``mean`` and ``vol`` to apply afterwards, and ``residual``,
    the norm of the moment mismatch. Use :func:`johnson_su_sample` to draw.
    """
    p, res = _mm.find_params_for_moments_matching_JSU([0.0, 1.0, float(skew), float(kurt) - 3.0],
                                                      x0=[0, 1, 1.5, 1], method='Nelder-Mead')
    return {'gamma': float(p[0]), 'xi': float(p[1]), 'delta': float(p[2]), 'lambda': float(p[3]),
            'mean': float(mean), 'vol': float(vol), 'residual': float(res.fun)}


def johnson_su_moments(params):
    """``[mean, variance, skewness, excess kurtosis]`` of the standardized Johnson SU with
    ``params`` from :func:`fit_johnson_su` (dict) or a ``(gamma, xi, delta, lambda)`` sequence."""
    p = [params[k] for k in ('gamma', 'xi', 'delta', 'lambda')] if isinstance(params, dict) else list(params)
    return _mm.moments_JSU(p)


def johnson_su_sample(params, n, seed=None):
    """Draw ``n`` values from the Johnson SU of :func:`fit_johnson_su`, rescaled by its mean and vol."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    p = [params[k] for k in ('gamma', 'xi', 'delta', 'lambda')]
    return params.get('mean', 0.0) + params.get('vol', 1.0) * _mm.sim_JSU_with_U(u, np.asarray(p, float))


def fit_fleishman(skew, kurt):
    """
    Fleishman coefficients ``(a, b, c, d)`` matching a skewness and a raw kurtosis.

    Returns a dict with the four coefficients, ``residual`` (max absolute moment
    mismatch) and ``feasible``. Draw with ``a + b Z + c Z**2 + d Z**3``, ``Z`` standard normal.
    """
    target = np.array([0.0, 1.0, float(skew), float(kurt) - 3.0])
    if target[3] < target[2] ** 2 - 2:
        return {'a': np.nan, 'b': np.nan, 'c': np.nan, 'd': np.nan, 'residual': np.nan, 'feasible': False}
    p0, _ = _mm.find_params_for_moments_matching(target, x0=[0.0, 1.0, 0.0, 0.0], method='Nelder-Mead', distr='gauss')
    p, _, ier, _ = fsolve(lambda q: _mm.moments_cubic_transform(q, distr='gauss') - target, p0, full_output=True)
    res = float(np.max(np.abs(_mm.moments_cubic_transform(p, distr='gauss') - target)))
    ok = bool(ier == 1 and res < 1e-8 and p[1] > 0)
    return {'a': float(p[0]), 'b': float(p[1]), 'c': float(p[2]), 'd': float(p[3]), 'residual': res, 'feasible': ok}


def exceedance_curve(x, y, z=None):
    """
    Exceedance correlation of ``(x, y)`` conditional on ``x`` (equation C.1 of the paper).

    ``x`` and ``y`` are standardized; for each threshold ``z`` the Pearson
    correlation is computed on ``x < z`` (``z < 0``) or ``x >= z`` (``z >= 0``).

    Returns a Series indexed by ``z`` (default grid −1 to 1 by 0.05 standard deviations).
    """
    if z is None:
        z = np.round(np.arange(-1.0, 1.0001, 0.05), 3)
    z = np.asarray(z, float)
    return pd.Series(_ct.exceedance_correlation(np.asarray(x, float), np.asarray(y, float), z), index=z, name='exceedance corr')


def classify_pair(x, y, target_corr=None, z_lb=-0.5, z_ub=0.5):
    """
    The four characteristics ``(l, u, s, m)`` of Algorithm 3 for one pair, from its exceedance curve.

    Parameters
    ----------
    x, y : array_like
        Returns of the two assets; ``x`` is the conditioning (central) one.
    target_corr : float, optional
        Sign of the target correlation (``s``); the sample correlation if omitted.
    z_lb, z_ub : float
        Threshold range used by the paper's classification (−0.5 to 0.5).

    Returns
    -------
    dict
        ``l`` (lower-tail flag), ``u`` (upper-tail flag), ``s`` (sign, −11 when
        the tiebreakers are inconclusive), ``m`` (+1 monotone, −1 sign change),
        ``metrics`` (the summary statistics) and ``curve`` (the Series).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    curve = _ct.draw_cond_corr(x=x, y=y, thetas_lb=z_lb, thetas_ub=z_ub, return_values=True,
                               inputs_are_obs=True, show_plot=False)
    met = _ct.get_condcorrelation_metrics(curve)
    l = bool(met['mean_leftside'] > met['halfway_from_jump'] and met['max_rightside'] < met['halfway_from_jump'])
    u = bool(met['mean_rightside'] > met['halfway_from_jump'] and met['max_leftside'] < met['halfway_from_jump'])
    rho = float(np.corrcoef(x, y)[0, 1]) if target_corr is None else float(target_corr)
    s = int(np.sign(rho))
    m = int(np.sign(curve.min() * curve.max()))
    if not l and not u:
        if met['max_leftside'] < met['min_rightside']:
            u = True
        elif met['min_leftside'] > met['max_rightside']:
            l = True
        else:
            left0 = curve[curve.index < 0].iloc[-1]; right0 = curve[curve.index > 0].iloc[0]
            if met['mean_leftside'] < left0 and met['mean_rightside'] > right0:
                u = True
            elif met['mean_leftside'] > left0 and met['mean_rightside'] < right0:
                l = True
            else:
                s = -11
    return {'l': int(l), 'u': int(u), 's': s, 'm': m, 'metrics': met, 'curve': curve}


def select_family(x, y, target_corr=None, mixtures=True):
    """
    Family selection of Algorithm 3 for one pair: classify, restrict the catalog, fit, keep the best BIC.

    Returns a dict with ``status`` (``'ordinary'`` or ``'mixture'``), ``family``
    (name, or the two component names), ``rotation`` (or the two rotations),
    ``parameters``, ``bic`` (per observation) and ``classification``.
    """
    import pyvinecopulib as pv
    cls = classify_pair(x, y, target_corr)
    data = pv.to_pseudo_obs(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    spec = _ct.get_copulas_specifications()
    mask = spec.eq([bool(cls['l']), bool(cls['u']), cls['s'], cls['m']]).all(axis=1)
    candidates = list(spec.index[mask])
    if len(candidates) >= 1:
        est = _ct.select_best_preselected_bivariate_copula(data, families_and_rotations=candidates)
        cop = est['copula']
        return {'status': 'ordinary', 'family': cop.family.name, 'rotation': int(cop.rotation),
                'parameters': cop.parameters.ravel().tolist(), 'bic': float(est['best_bic']), 'classification': cls}
    if cls['m'] == -1 and mixtures:
        mspec = _ct.get_copulas_mixture_specifications()
        mmask = mspec.eq([bool(cls['l']), bool(cls['u'])]).all(axis=1)
        cands = list(mspec.index[mmask]) or list(mspec.index)
        est = _ct.select_best_preselected_mixture_copula(data=data, list_of_copulas_families=cands)
        comps = est['best_copula']
        return {'status': 'mixture', 'family': [b.family.name for b in comps], 'rotation': [int(b.rotation) for b in comps],
                'parameters': [float(v) for v in est['best_params']], 'bic': float(est['best_bic']), 'classification': cls}
    est = _ct.select_best_bivariate_copula(data)
    cop = est['copula']
    return {'status': 'ordinary', 'family': cop.family.name, 'rotation': int(cop.rotation),
            'parameters': cop.parameters.ravel().tolist(), 'bic': float(est['best_bic']), 'classification': cls}
