# -*- coding: utf-8 -*-
"""
Standalone functions for the pieces people want on their own: a marginal fit,
an exceedance curve, the classification of one pair, the family selection of
one pair. Thin wrappers over :class:`~cvinemarketgen.moment_match.MomentMatch`
and :class:`~cvinemarketgen.copulas.CopulaTools`.
"""
import numpy as np
import pandas as pd
import warnings

from scipy.optimize import least_squares, fsolve, minimize

from .moment_match import MomentMatch
from .copulas import CopulaTools


class _Tools(MomentMatch, CopulaTools):
    pass


_mm = _Tools()
_ct = _mm


_JSU_STARTS = ([0, 1, 1.5, 1], [-0.5, -0.5, 1.2, 0.8], [0.5, 0.5, 1.2, 0.8], [0, 1, 3, 3], [-2, -2, 2, 1.5], [0, 0, 0.8, 0.5])


def _jsu_shape(gamma, delta):
    """Skewness and excess kurtosis of the Johnson SU with shape (gamma, delta); location and scale do not enter."""
    w = np.exp(delta ** -2.0); O = gamma / delta
    var = 0.5 * (w - 1.0) * (w * np.cosh(2 * O) + 1.0)
    skew = -(np.sqrt(w) * (w - 1.0) ** 2 * (w * (w + 2.0) * np.sinh(3 * O) + 3.0 * np.sinh(O))) / (4.0 * var ** 1.5)
    K1 = w ** 2 * (w ** 4 + 2.0 * w ** 3 + 3.0 * w ** 2 - 3.0) * np.cosh(4 * O)
    K2 = 4.0 * w ** 2 * (w + 2.0) * np.cosh(2 * O)
    K3 = 3.0 * (2.0 * w + 1.0)
    exk = (w - 1.0) ** 2 * (K1 + K2 + K3) / (8.0 * var ** 2) - 3.0
    return skew, exk


_JSU_SHAPE_STARTS = ((0.0, 1.5), (-0.5, 1.2), (0.5, 1.2), (0.0, 3.0), (-2.0, 2.0), (2.0, 2.0), (0.0, 0.8), (-1.0, 0.6), (1.0, 0.6), (-4.0, 1.5), (4.0, 1.5))


def fit_johnson_su(skew, kurt, mean=0.0, vol=1.0):
    """
    Johnson SU parameters matching a skewness and a raw kurtosis (normal = 3).

    Returns a dict with ``gamma, xi, delta, lambda`` for the *standardized*
    variable, the ``mean`` and ``vol`` to apply afterwards, and ``residual``,
    the norm of the moment mismatch. Use :func:`johnson_su_sample` to draw.

    Skewness and kurtosis depend on the shape ``(gamma, delta)`` only, so they are
    solved as two equations in two unknowns (Levenberg-Marquardt from a few
    starts); ``lambda`` and ``xi`` then standardize the variable in closed form.
    The four-parameter Nelder-Mead search of the earlier versions is the fallback.
    """
    target = [0.0, 1.0, float(skew), float(kurt) - 3.0]
    s_t, k_t = float(skew), float(kurt) - 3.0

    def resid(x):
        g, d = x[0], x[1]
        if d <= 1e-6:
            return np.array([1e3, 1e3])
        with np.errstate(all='ignore'):
            sk, ek = _jsu_shape(g, d)
        if not (np.isfinite(sk) and np.isfinite(ek)):
            return np.array([1e3, 1e3])
        return np.array([sk - s_t, ek - k_t])

    best = None
    for g0, d0 in _JSU_SHAPE_STARTS:
        try:
            r = least_squares(resid, [g0, d0], method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
        except Exception:
            continue
        g, d = float(r.x[0]), float(r.x[1])
        if d <= 1e-6:
            continue
        mism = float(np.linalg.norm(resid([g, d])))
        if best is None or mism < best[0]:
            best = (mism, g, d)
        if mism < 1e-10:
            break
    if best is not None and best[0] < 1e-6:
        _, g, d = best
        w = np.exp(d ** -2.0); O = g / d
        var1 = 0.5 * (w - 1.0) * (w * np.cosh(2 * O) + 1.0)           # variance for lambda = 1
        lam = 1.0 / np.sqrt(var1)
        xi = lam * np.sqrt(w) * np.sinh(O)                             # mean zero: xi - lambda e^{1/(2 delta^2)} sinh(gamma/delta) = 0
        p = np.array([g, xi, d, lam])
        with np.errstate(all='ignore'):
            mism = float(np.linalg.norm(np.asarray(_mm.moments_JSU(p), float) - target))
        if np.isfinite(mism) and mism < 1e-6:
            return {'gamma': float(p[0]), 'xi': float(p[1]), 'delta': float(p[2]), 'lambda': float(p[3]),
                    'mean': float(mean), 'vol': float(vol), 'residual': float(mism)}
    best = None
    for x0 in _JSU_STARTS:
        res = minimize(_mm.univariate_moments_matching_func_JSU, x0, args=([target],), method='Nelder-Mead',
                       tol=1e-10, options={'maxfev': 20000, 'xatol': 1e-10, 'fatol': 1e-14})
        with np.errstate(all='ignore'):
            mism = float(np.linalg.norm(np.asarray(_mm.moments_JSU(res.x), float) - target))
        if np.isfinite(mism) and res.x[2] > 0 and res.x[3] > 1e-6 and (best is None or mism < best[0]):
            best = (mism, res.x)
        if best is not None and best[0] < 1e-12:
            break
    if best is None:
        raise RuntimeError(f'Johnson SU moment fit failed for skew {skew:.3f}, kurt {kurt:.3f}')
    if best[0] > 1e-6:
        warnings.warn(f'Johnson SU moment fit: mismatch {best[0]:.2e} for skew {skew:.3f}, kurt {kurt:.3f}')
    mism, p = best
    return {'gamma': float(p[0]), 'xi': float(p[1]), 'delta': float(p[2]), 'lambda': float(p[3]),
            'mean': float(mean), 'vol': float(vol), 'residual': float(mism)}


def johnson_su_moments(params):
    """``[mean, variance, skewness, excess kurtosis]`` of the standardized Johnson SU with
    ``params`` from :func:`fit_johnson_su` (dict) or a ``(gamma, xi, delta, lambda)`` sequence."""
    p = [params[k] for k in ('gamma', 'xi', 'delta', 'lambda')] if isinstance(params, dict) else list(params)
    return _mm.moments_JSU(p)


def johnson_su_cdf(params, x):
    """Distribution function of the Johnson SU of :func:`fit_johnson_su` (dict with ``mean`` and ``vol``) at ``x``."""
    from scipy.stats import norm
    q = (np.asarray(x, float) - params.get('mean', 0.0)) / params.get('vol', 1.0)
    z = params['gamma'] + params['delta'] * np.arcsinh((q - params['xi']) / params['lambda'])
    return norm.cdf(z)


def johnson_su_from_normal(params, z):
    """The Johnson SU of :func:`fit_johnson_su` evaluated at the standard-normal draw ``z``:
    ``mean + vol * (xi + lambda * sinh((z - gamma) / delta))``, the inverse of :func:`johnson_su_cdf`.
    Feeding correlated normals gives a Gaussian copula with Johnson SU marginals."""
    z = np.asarray(z, float)
    x = params['xi'] + params['lambda'] * np.sinh((z - params['gamma']) / params['delta'])
    return params.get('mean', 0.0) + params.get('vol', 1.0) * x


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


def _normal_scores(v):
    """Normal scores ``Phi^{-1}(rank / (n + 1))``: the copula scale, marginals removed."""
    from scipy.stats import norm, rankdata
    v = np.asarray(v, float)
    return norm.ppf(rankdata(v) / (len(v) + 1.0))


def classify_pair(x, y, target_corr=None, z_lb=-0.5, z_ub=0.5, copula_scale=True):
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
    copula_scale : bool, default True
        Classify on the normal scores of the two series (the copula scale, as the
        vine's deeper trees do) rather than on the standardized returns, whose
        skewness would otherwise be read as copula asymmetry. ``False`` gives the
        paper's original first-tree rule.

    Returns
    -------
    dict
        ``l`` (lower-tail flag), ``u`` (upper-tail flag), ``s`` (sign, −11 when
        the tiebreakers are inconclusive), ``m`` (+1 monotone, −1 sign change),
        ``metrics`` (the summary statistics) and ``curve`` (the Series).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho_sign_src = (x, y)
    if copula_scale:
        x, y = _normal_scores(x), _normal_scores(y)
    curve = _ct.draw_cond_corr(x=x, y=y, thetas_lb=z_lb, thetas_ub=z_ub, return_values=True,
                               inputs_are_obs=True, show_plot=False)
    met = _ct.get_condcorrelation_metrics(curve)
    l = bool(met['mean_leftside'] > met['halfway_from_jump'] and met['max_rightside'] < met['halfway_from_jump'])
    u = bool(met['mean_rightside'] > met['halfway_from_jump'] and met['max_leftside'] < met['halfway_from_jump'])
    rho = float(np.corrcoef(*rho_sign_src)[0, 1]) if target_corr is None else float(target_corr)
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


def tail_asymmetry_test(x, y, n_boot=500, level=0.90, seed=0, copula_scale=True):
    """
    Is the exceedance-correlation profile of ``(x, y)`` asymmetric beyond sampling noise?

    The statistic is the mean of the profile for ``z < 0`` minus its mean for
    ``z > 0`` (thresholds from -1 to 1 standard deviations, the range the paper
    plots), computed on the normal scores of the two series (the copula
    scale; ``copula_scale=False`` for the standardized returns); its distribution
    is bootstrapped by resampling the observations. Returns a dict with
    ``difference``, ``lower``, ``upper`` (the central ``level`` interval),
    ``symmetric`` (True when the interval contains zero), ``level`` and ``n_boot``.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if copula_scale:
        x, y = _normal_scores(x), _normal_scores(y)
    return _ct.tail_asymmetry_test(x, y, n_boot=n_boot, level=level, seed=seed)


def select_family(x, y, target_corr=None, mixtures=True, symmetry_test=True, copula_scale=True):
    """
    Family selection of Algorithm 3 for one pair, with the symmetry test first.

    With ``symmetry_test`` (default), :func:`tail_asymmetry_test` decides whether
    the profile is asymmetric beyond sampling noise; a symmetric pair gets the
    Gaussian or the Student t copula by BIC. An asymmetric pair (or every pair
    when ``symmetry_test=False``) is classified, the catalog restricted to its
    class, the candidates fitted and the best BIC kept, as in the paper.

    Returns a dict with ``status`` (``'ordinary'`` or ``'mixture'``), ``family``
    (name, or the two component names), ``rotation`` (or the two rotations),
    ``parameters``, ``bic`` (per observation), ``symmetric``, ``asymmetry`` (the
    test) and ``classification``.
    """
    import pyvinecopulib as pv
    cls = classify_pair(x, y, target_corr, copula_scale=copula_scale)
    data = pv.to_pseudo_obs(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    asym = tail_asymmetry_test(x, y, copula_scale=copula_scale) if symmetry_test else None
    if asym is not None and asym['symmetric']:
        est = _ct.select_best_preselected_bivariate_copula(data, families_and_rotations=_ct.get_copulas_specifications_symmetric())
        cop = est['copula']
        return {'status': 'ordinary', 'family': cop.family.name, 'rotation': int(cop.rotation),
                'parameters': cop.parameters.ravel().tolist(), 'bic': float(est['best_bic']),
                'symmetric': True, 'asymmetry': asym, 'classification': cls}
    spec = _ct.get_copulas_specifications()
    mask = spec.eq([bool(cls['l']), bool(cls['u']), cls['s'], cls['m']]).all(axis=1)
    candidates = list(spec.index[mask])
    if len(candidates) >= 1:
        est = _ct.select_best_preselected_bivariate_copula(data, families_and_rotations=candidates)
        cop = est['copula']
        return {'status': 'ordinary', 'family': cop.family.name, 'rotation': int(cop.rotation),
                'parameters': cop.parameters.ravel().tolist(), 'bic': float(est['best_bic']),
                'symmetric': False, 'asymmetry': asym, 'classification': cls}
    if cls['m'] == -1 and mixtures:
        mspec = _ct.get_copulas_mixture_specifications()
        mmask = mspec.eq([bool(cls['l']), bool(cls['u'])]).all(axis=1)
        cands = list(mspec.index[mmask]) or list(mspec.index)
        est = _ct.select_best_preselected_mixture_copula(data=data, list_of_copulas_families=cands)
        comps = est['best_copula']
        return {'status': 'mixture', 'family': [b.family.name for b in comps], 'rotation': [int(b.rotation) for b in comps],
                'parameters': [float(v) for v in est['best_params']], 'bic': float(est['best_bic']),
                'symmetric': False, 'asymmetry': asym, 'classification': cls}
    est = _ct.select_best_bivariate_copula(data)
    cop = est['copula']
    return {'status': 'ordinary', 'family': cop.family.name, 'rotation': int(cop.rotation),
            'parameters': cop.parameters.ravel().tolist(), 'bic': float(est['best_bic']),
                'symmetric': False, 'asymmetry': asym, 'classification': cls}
