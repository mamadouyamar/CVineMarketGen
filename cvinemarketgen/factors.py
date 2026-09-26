# -*- coding: utf-8 -*-
"""
Assets on factors: a linear factor model whose betas map simulated factor
scenarios or paths (from a market on the factors) to asset returns, with a
Johnson SU residual per asset.

    r_t = alpha + beta' f_t + eps_t

Every asset is described by what is known about it and the rest is filled by a
stated rule: betas are regressed unless given (``exposures``), an asset without
a history is added from its mean, volatility and exposures (``add_asset``),
means and volatilities are imposed through alpha and the residual scale, and a
correlation between two assets through their residual correlation
(``with_targets``). Standard errors are Newey-West. Units and frequency are the
data's; targets are annual by default and converted.
"""
import copy as _copy
import json
import warnings

import numpy as np
import pandas as pd

from .functions import fit_johnson_su, johnson_su_sample, johnson_su_from_normal
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


def _gaussian_corr_for(params_i, params_j, rho_target, n=100000, seed=0):
    """Correlation of the normals such that the Johnson SU transforms have linear correlation ``rho_target``."""
    if abs(rho_target) < 1e-12:
        return 0.0
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    rho = float(rho_target)
    for _ in range(3):
        rho = float(np.clip(rho, -0.999, 0.999))
        x = johnson_su_from_normal(params_i, z1)
        y = johnson_su_from_normal(params_j, rho * z1 + np.sqrt(1 - rho ** 2) * z2)
        m = float(np.corrcoef(x, y)[0, 1])
        if abs(m) < 1e-6:
            break
        rho = rho * rho_target / m
    return float(np.clip(rho, -0.999, 0.999))


class FactorModel:
    """
    Linear factor model of assets on factors, ``r = alpha + beta' f + eps``.

    Parameters
    ----------
    asset_returns : DataFrame
        One column per asset with a history.
    factor_returns : DataFrame
        One column per factor with a history, same frequency; aligned on the common index.
    nw_lags : int, optional
        Newey-West lags for the standard errors; default ``floor(4 (n/100)^(2/9))``.
    exposures : dict, optional
        What the analyst knows about an asset's betas, ``{asset: spec}``, three forms:

        * a list of factors: the allowed set; regression on those, zero elsewhere;
        * a dict ``{factor: value}`` of given betas: fixed, and the remainder
          ``r - sum(given beta f)`` is regressed on the factors not named;
        * a dict with ``'fit'`` values, ``{'Equity DM': 1.0, 'Credit': 'fit'}``: the
          numbers are fixed, the ``'fit'`` factors regressed, every other factor zero.

        Assets absent from the dict load on every factor. A given beta has NaN
        standard error and t-statistic. ``beta_source`` records ``given``, ``fitted``
        or ``zero`` per entry.
    factor_targets : Targets, optional
        Moments of the factors used by the asset layer (``mu_f``, ``Sigma_f``): the
        sample moments of ``factor_returns`` by default. Needed when a factor has no
        history (:meth:`~cvinemarketgen.targets.Targets.add_factor`): the factor names
        are then those of the targets, regressions run on the columns of
        ``factor_returns``, and a beta on the synthetic factor can only be given.

    Notes
    -----
    After ``fit``: ``alpha`` (Series), ``beta`` (DataFrame assets x factors),
    ``se`` and ``tstat`` (DataFrames with an ``alpha`` column and one per factor),
    ``r2`` (Series), ``resid`` (DataFrame), ``resid_vol`` (Series), ``resid_params``
    (Johnson SU per asset, mean zero), ``resid_corr`` (identity unless pair targets),
    ``beta_source``, ``asset_source`` (``history`` or ``spec``) and ``report``.
    """

    def __init__(self, asset_returns, factor_returns, nw_lags=None, exposures=None, factor_targets=None):
        R = pd.DataFrame(asset_returns).astype(float)
        F = pd.DataFrame(factor_returns).astype(float)
        idx = R.index.intersection(F.index)
        if len(idx) < 3 * (F.shape[1] + 1):
            raise ValueError('not enough common observations to fit the factor model')
        self.R = R.loc[idx]
        self.F = F.loc[idx]
        self.assets = list(self.R.columns)
        self.factor_targets = factor_targets
        if factor_targets is not None:
            self.factors = list(factor_targets.assets)
            missing = [f for f in self.F.columns if f not in self.factors]
            if missing:
                raise ValueError(f'factor_returns has columns absent from factor_targets: {missing}')
            self.hist_factors = [f for f in self.factors if f in self.F.columns]
            nohist = [f for f in self.factors if f not in self.F.columns and f not in factor_targets.synthetic]
            if nohist:
                raise ValueError(f'factors {nohist} have neither a history nor a synthetic specification')
        else:
            self.factors = list(self.F.columns)
            self.hist_factors = list(self.factors)
        self.synthetic_factors = [f for f in self.factors if f not in self.hist_factors]
        self.n_obs = len(idx)
        self.nw_lags = int(np.floor(4 * (self.n_obs / 100.0) ** (2.0 / 9.0))) if nw_lags is None else int(nw_lags)
        self.exposures = None
        self._plan = {}
        if exposures is not None:
            self.exposures = {}
            for a, spec in dict(exposures).items():
                if a not in self.assets:
                    raise ValueError(f'exposures: unknown asset {a!r}')
                self.exposures[a] = self._parse_exposure(a, spec)
        self.fitted = False

    # ---- specification -------------------------------------------------------
    def _parse_exposure(self, a, spec):
        """Normalize one asset's exposure spec into (given dict, fit list, zero list) and return the stored form."""
        if isinstance(spec, str):
            spec = [spec]
        if isinstance(spec, dict):
            given = {}; fit = []
            for f, v in spec.items():
                if f not in self.factors:
                    raise ValueError(f'exposures of {a!r}: unknown factor {f!r}')
                if isinstance(v, str):
                    if v != 'fit':
                        raise ValueError(f"exposures of {a!r}: value of {f!r} must be a number or 'fit'")
                    if f not in self.hist_factors:
                        raise ValueError(f'exposures of {a!r}: {f!r} has no history, its beta must be given')
                    fit.append(f)
                else:
                    given[f] = float(v)
            if fit:                                   # pinned allowed set
                zero = [f for f in self.factors if f not in given and f not in fit]
            else:                                     # override the regression: fit every unnamed factor with a history
                fit = [f for f in self.hist_factors if f not in given]
                zero = [f for f in self.factors if f not in given and f not in fit]
            self._plan[a] = (given, fit, zero)
            return {f: (v if f in given else 'fit') for f, v in {**given, **{f: 'fit' for f in fit}}.items()}
        fs = list(spec)
        bad = [f for f in fs if f not in self.factors]
        if bad:
            raise ValueError(f'exposures of {a!r}: unknown factors {bad}')
        nohist = [f for f in fs if f not in self.hist_factors]
        if nohist:
            raise ValueError(f'exposures of {a!r}: {nohist} have no history; give their betas as numbers')
        self._plan[a] = ({}, fs, [f for f in self.factors if f not in fs])
        return fs

    def _factor_moments(self):
        """``mu_f`` and ``Sigma_f`` over ``self.factors``: from the factor targets if given, else the sample."""
        if self.factor_targets is not None:
            t = self.factor_targets
            mu = t.mean.loc[self.factors].values
            v = t.vol.loc[self.factors].values
            return mu, np.outer(v, v) * t.corr.loc[self.factors, self.factors].values
        return self.F[self.factors].mean().values, self.F[self.factors].cov().values

    # ---- fit -----------------------------------------------------------------------
    def fit(self):
        K = len(self.factors); cols = ['alpha'] + self.factors
        N = len(self.assets)
        coef = np.zeros((K + 1, N)); se = np.full((K + 1, N), np.nan)
        E = np.empty((self.n_obs, N)); dof = np.empty(N)
        source = pd.DataFrame('fitted', index=self.assets, columns=self.factors)
        mu_f, Sigma_f = self._factor_moments()
        reduced = {}
        for j, a in enumerate(self.assets):
            given, fit, zero = self._plan.get(a, ({}, list(self.hist_factors), list(self.synthetic_factors)))
            y = self.R[a].values.copy()
            for f, v in given.items():
                if f in self.hist_factors:
                    y = y - v * self.F[f].values
                source.loc[a, f] = 'given'; coef[1 + self.factors.index(f), j] = v
            for f in zero:
                source.loc[a, f] = 'zero'
            X = np.column_stack([np.ones(self.n_obs)] + [self.F[f].values for f in fit])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            e = y - X @ c
            pos = [0] + [1 + self.factors.index(f) for f in fit]
            coef[pos, j] = c
            se[pos, j] = _newey_west_se(X, e, self.nw_lags)
            E[:, j] = e; dof[j] = X.shape[1]
            syn = [f for f in given if f not in self.hist_factors]
            if syn:                                   # A4: keep the sample volatility, the synthetic loadings take from the residual
                b = coef[1:, j]
                bh = np.array([b[k] if self.factors[k] in self.hist_factors else 0.0 for k in range(K)])
                added = float(b @ Sigma_f @ b - bh @ Sigma_f @ bh)
                reduced[a] = added
        self.alpha = pd.Series(coef[0], index=self.assets)
        self.beta = pd.DataFrame(coef[1:].T, index=self.assets, columns=self.factors)
        self.se = pd.DataFrame(se.T, index=self.assets, columns=cols)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.tstat = pd.DataFrame(coef.T / se.T, index=self.assets, columns=cols)
        tss = ((self.R.values - self.R.values.mean(0)) ** 2).sum(0)
        self.r2 = pd.Series(1.0 - (E ** 2).sum(0) / tss, index=self.assets)
        self.resid = pd.DataFrame(E, index=self.R.index, columns=self.assets)
        rv = np.sqrt((E ** 2).sum(0) / (self.n_obs - dof))
        self.resid_vol = pd.Series(rv, index=self.assets)
        self.sample_resid_vol = self.resid_vol.copy()
        for a, added in reduced.items():
            v2 = self.resid_vol[a] ** 2 - added
            if v2 <= 0:
                raise ValueError(f'{a}: the given betas on {[f for f in self.synthetic_factors if self.beta.loc[a, f] != 0]} add '
                                 f'more systematic variance ({np.sqrt(added):.4f} in volatility) than the residual holds '
                                 f'({self.resid_vol[a]:.4f}); lower them or give the asset a volatility target')
            self.resid_vol[a] = float(np.sqrt(v2))
        rows = {}
        for a in self.assets:
            e = self.resid[a]
            rows[a] = fit_johnson_su(e.skew(), e.kurtosis() + 3.0, mean=0.0, vol=float(self.resid_vol[a]))
        self.resid_params = pd.DataFrame(rows).T[['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']].astype(float)
        self.beta_source = source
        self.asset_source = pd.Series('history', index=self.assets)
        self.resid_corr = pd.DataFrame(np.eye(N), index=self.assets, columns=self.assets)
        self._gauss_corr = None
        self.target_report = None
        self.pair_report = None
        self._targets_applied = pd.DataFrame(False, index=self.assets, columns=['mean', 'vol', 'pair'])
        self._build_report()
        self.fitted = True
        return self

    def _build_report(self):
        cols = ['alpha'] + self.factors
        rep = pd.concat([self.alpha.rename('alpha'), self.beta], axis=1)
        for c in cols:
            rep[f't({c})'] = self.tstat[c]
        rep['R2'] = self.r2
        rep['resid vol'] = self.resid_vol
        self.report = rep

    # ---- assets without a history -------------------------------------------------
    def add_asset(self, name, mean, vol, exposures=None, skew=0.0, kurt=3.0, annualized=True, periods=12):
        """
        A copy with one more asset that has no history, known by its mean, volatility
        and (optionally) its exposures.

        ``exposures`` is ``{factor: beta}`` (numbers; unnamed factors are zero). The
        rule: ``alpha = m - beta' mu_f``; ``sigma_e^2 = v^2 - beta' Sigma_f beta``,
        which must be positive (``ValueError`` names the floor); the residual is a
        Johnson SU with the given ``skew`` and ``kurt`` (normal by default). Without
        exposures the asset is pure residual, independent of everything, and
        ``asset_source`` says ``spec (pure residual)``. ``mean`` and ``vol`` are annual
        by default. Statistics that need a history (t-statistics, R2) are NaN.
        """
        self._check()
        name = str(name)
        if name in self.assets:
            raise ValueError(f'add_asset: {name!r} already in the model')
        exposures = {} if exposures is None else {str(k): float(v) for k, v in dict(exposures).items()}
        bad = [f for f in exposures if f not in self.factors]
        if bad:
            raise ValueError(f'add_asset: unknown factors {bad}')
        m = float(mean) / periods if annualized else float(mean)
        v = float(vol) / np.sqrt(periods) if annualized else float(vol)
        mu_f, Sigma_f = self._factor_moments()
        b = np.array([exposures.get(f, 0.0) for f in self.factors])
        sys_var = float(b @ Sigma_f @ b)
        if v ** 2 <= sys_var:
            raise ValueError(f'add_asset: {name}: volatility {v:.4f} is not above the systematic volatility {np.sqrt(sys_var):.4f} '
                             f'implied by the exposures; raise it or lower the exposures')
        new = _copy.deepcopy(self)
        new.assets = self.assets + [name]
        new.alpha = pd.concat([self.alpha, pd.Series({name: m - float(b @ mu_f)})])
        new.beta = pd.concat([self.beta, pd.DataFrame([b], index=[name], columns=self.factors)])
        cols = ['alpha'] + self.factors
        new.se = pd.concat([self.se, pd.DataFrame([[np.nan] * len(cols)], index=[name], columns=cols)])
        new.tstat = pd.concat([self.tstat, pd.DataFrame([[np.nan] * len(cols)], index=[name], columns=cols)])
        new.r2 = pd.concat([self.r2, pd.Series({name: np.nan})])
        rv = float(np.sqrt(v ** 2 - sys_var))
        new.resid_vol = pd.concat([self.resid_vol, pd.Series({name: rv})])
        new.sample_resid_vol = pd.concat([self.sample_resid_vol, pd.Series({name: np.nan})])
        p = fit_johnson_su(float(skew), float(kurt), mean=0.0, vol=rv)
        new.resid_params = pd.concat([self.resid_params, pd.DataFrame([p], index=[name])[self.resid_params.columns]])
        if self.resid is not None:
            new.resid = self.resid.copy(); new.resid[name] = np.nan
        new.beta_source = pd.concat([self.beta_source, pd.DataFrame([['given' if f in exposures else 'zero' for f in self.factors]],
                                                                     index=[name], columns=self.factors)])
        new.asset_source = pd.concat([self.asset_source, pd.Series({name: 'spec' if exposures else 'spec (pure residual)'})])
        rc = pd.DataFrame(np.eye(len(new.assets)), index=new.assets, columns=new.assets)
        rc.loc[self.assets, self.assets] = self.resid_corr.values
        new.resid_corr = rc; new._gauss_corr = None
        new._targets_applied = pd.concat([self._targets_applied, pd.DataFrame([[False] * 3], index=[name], columns=self._targets_applied.columns)])
        new._build_report()
        return new

    def add_assets(self, table, annualized=True, periods=12):
        """
        Several assets without a history from a DataFrame indexed by asset with columns
        ``mean``, ``vol``, optional ``skew`` and ``kurt``, and one column per factor for
        the exposures (NaN or absent = zero). Applies :meth:`add_asset` row by row.
        """
        T = pd.DataFrame(table)
        new = self
        for a, row in T.iterrows():
            exp = {f: float(row[f]) for f in self.factors if f in T.columns and pd.notna(row[f]) and row[f] != 0}
            new = new.add_asset(a, row['mean'], row['vol'], exposures=exp,
                                skew=float(row['skew']) if 'skew' in T.columns and pd.notna(row.get('skew')) else 0.0,
                                kurt=float(row['kurt']) if 'kurt' in T.columns and pd.notna(row.get('kurt')) else 3.0,
                                annualized=annualized, periods=periods)
        return new

    # ---- targets ----------------------------------------------------------------------
    def with_targets(self, targets=None, factor_targets=None, annualized=True, periods=12, pair_correlations=None):
        """
        A copy of the fitted model whose listed assets hit a target mean and/or
        volatility, and whose listed pairs hit a target correlation.

        ``targets`` is ``{asset: {'mean': m, 'vol': v}}`` (either key may be
        absent) or a DataFrame with one row per asset and columns ``mean`` and
        ``vol`` (NaN to keep the current value), for example read from Excel or CSV.
        In annual terms by default (mean times ``periods``, volatility times
        ``sqrt(periods)``), converted to the data's frequency.

        The betas are kept. The mean is hit through the alpha,
        ``alpha* = m - beta' mu_f``; the volatility through the residual scale,
        ``sigma_e*^2 = v^2 - beta' Sigma_f beta``, feasible only when ``v`` is at
        least the systematic volatility ``sqrt(beta' Sigma_f beta)``; an infeasible
        target raises ``ValueError`` and names the floor. ``mu_f`` and ``Sigma_f``
        are those of ``factor_targets`` (a :class:`~cvinemarketgen.targets.Targets`),
        else of the model's factor targets, else the sample moments of the factors.
        The residual keeps its skewness and kurtosis. The copy carries a
        ``target_report`` DataFrame with, per listed asset, the target and current
        mean, the new alpha, the target and current volatility, the systematic
        volatility and the new residual volatility.

        ``pair_correlations`` is ``{(asset_i, asset_j): rho}``. With the betas and
        volatilities fixed (the volatility targets above are applied first), the
        correlation of a pair is ``(beta_i' Sigma_f beta_j + d_ij) / (s_i s_j)`` and the
        only free quantity is the residual covariance ``d_ij``, bounded by
        ``sigma_e_i sigma_e_j``. The reachable range is therefore
        ``rho_sys +/- sqrt((1 - R2_i)(1 - R2_j))`` around the systematic correlation;
        a target outside it raises ``ValueError`` with the range. The residual
        correlation ``rho_e = d_ij / (sigma_e_i sigma_e_j)`` is stored in
        ``resid_corr``, which must stay positive semidefinite (``ValueError`` names
        the smallest eigenvalue otherwise); residuals of the named pairs are then
        drawn from a Gaussian copula with the assets' Johnson SU marginals, the
        others stay independent. ``pair_report`` lists, per pair: target,
        systematic correlation, eligible range, residual correlation used.
        """
        self._check()
        if targets is None:
            T = pd.DataFrame(columns=['mean', 'vol'], dtype=float)
        elif isinstance(targets, pd.DataFrame):
            T = targets.copy()
        else:
            T = pd.DataFrame({a: dict(v) for a, v in dict(targets).items()}).T
        for c in ('mean', 'vol'):
            if c not in T.columns:
                T[c] = np.nan
        T = T[['mean', 'vol']].astype(float)
        bad = [a for a in T.index if a not in self.assets]
        if bad:
            raise ValueError(f'with_targets: unknown assets {bad}')
        if annualized:
            T['mean'] = T['mean'] / periods
            T['vol'] = T['vol'] / np.sqrt(periods)
        if factor_targets is None:
            mu_f, Sigma_f = self._factor_moments()
        else:
            mu_f = factor_targets.mean.loc[self.factors].values
            vol_f = factor_targets.vol.loc[self.factors].values
            Sigma_f = np.outer(vol_f, vol_f) * factor_targets.corr.loc[self.factors, self.factors].values
        new = _copy.deepcopy(self)
        rows = {}
        for a in T.index:
            b = self.beta.loc[a].values
            sys_var = float(b @ Sigma_f @ b)
            m_old = float(self.alpha[a] + b @ mu_f)
            v_old = float(np.sqrt(sys_var + self.resid_vol[a] ** 2))
            m_new, v_new = T.loc[a, 'mean'], T.loc[a, 'vol']
            alpha_new = self.alpha[a] if np.isnan(m_new) else float(m_new - b @ mu_f)
            if np.isnan(v_new):
                resid_new = float(self.resid_vol[a])
            else:
                if v_new ** 2 < sys_var:
                    raise ValueError(f'with_targets: {a}: target volatility {v_new:.4f} is below the systematic volatility '
                                     f'{np.sqrt(sys_var):.4f} implied by its betas; raise the target or change the exposures')
                resid_new = float(np.sqrt(v_new ** 2 - sys_var))
            new.alpha[a] = alpha_new
            new.resid_vol[a] = resid_new
            p = self.resid_params.loc[a].to_dict(); p['vol'] = resid_new
            new.resid_params.loc[a] = pd.Series(p)
            new._targets_applied.loc[a, 'mean'] = new._targets_applied.loc[a, 'mean'] or not np.isnan(m_new)
            new._targets_applied.loc[a, 'vol'] = new._targets_applied.loc[a, 'vol'] or not np.isnan(v_new)
            rows[a] = {'target mean': m_new if not np.isnan(m_new) else m_old, 'sample mean': m_old, 'alpha': alpha_new,
                       'target vol': v_new if not np.isnan(v_new) else v_old, 'sample vol': v_old,
                       'systematic vol': float(np.sqrt(sys_var)), 'sample resid vol': float(self.resid_vol[a]), 'new resid vol': resid_new}
        new.target_report = pd.DataFrame(rows).T if rows else self.target_report
        if pair_correlations:
            new._set_pairs(pair_correlations, Sigma_f)
        new._build_report()
        return new

    def _set_pairs(self, pair_correlations, Sigma_f):
        """Residual correlations for the targeted pairs (in place, on a copy made by with_targets)."""
        B = self.beta.values
        S_sys = B @ Sigma_f @ B.T
        s2 = np.diag(S_sys) + self.resid_vol.values ** 2
        s = np.sqrt(s2)
        R2 = np.diag(S_sys) / s2
        rows = {}
        rc = self.resid_corr.copy()
        for (i, j), rho in dict(pair_correlations).items():
            for a in (i, j):
                if a not in self.assets:
                    raise ValueError(f'pair_correlations: unknown asset {a!r}')
            if i == j:
                raise ValueError('pair_correlations: a pair needs two different assets')
            ii, jj = self.assets.index(i), self.assets.index(j)
            rho_sys = float(S_sys[ii, jj] / (s[ii] * s[jj]))
            w = float(np.sqrt(max(1 - R2[ii], 0.0) * max(1 - R2[jj], 0.0)))
            lo, hi = rho_sys - w, rho_sys + w
            if not (lo - 1e-12 <= rho <= hi + 1e-12):
                raise ValueError(f'pair_correlations: ({i}, {j}): target {rho:.3f} is outside the eligible range '
                                 f'[{lo:.3f}, {hi:.3f}] = systematic correlation {rho_sys:.3f} +/- sqrt((1 - R2_i)(1 - R2_j)) '
                                 f'with R2 {R2[ii]:.2f} and {R2[jj]:.2f}; only the residual covariance is free')
            d = rho * s[ii] * s[jj] - S_sys[ii, jj]
            rho_e = float(np.clip(d / (self.resid_vol[i] * self.resid_vol[j]), -1.0, 1.0))
            rc.loc[i, j] = rc.loc[j, i] = rho_e
            rows[(i, j)] = {'target': float(rho), 'systematic': rho_sys, 'range low': lo, 'range high': hi,
                            'R2_i': float(R2[ii]), 'R2_j': float(R2[jj]), 'residual correlation': rho_e}
            self._targets_applied.loc[i, 'pair'] = True; self._targets_applied.loc[j, 'pair'] = True
        w_min = float(np.linalg.eigvalsh(rc.values).min())
        if w_min < -1e-10:
            involved = sorted({a for pair in pair_correlations for a in pair})
            raise ValueError(f'pair_correlations: the residual correlation matrix is not positive semidefinite '
                             f'(smallest eigenvalue {w_min:.4f}) for the pairs among {involved}; the targets are jointly infeasible')
        self.resid_corr = rc
        self._gauss_corr = None
        rep = pd.DataFrame(rows).T
        rep.index = pd.MultiIndex.from_tuples(rep.index, names=['asset i', 'asset j'])
        self.pair_report = rep if self.pair_report is None else pd.concat([self.pair_report, rep])

    def _gaussian_corr(self):
        """Correlation of the normals behind the residual Gaussian copula, corrected for the marginals (cached)."""
        if self._gauss_corr is None:
            G = np.eye(len(self.assets))
            rc = self.resid_corr.values
            for ii in range(len(self.assets)):
                for jj in range(ii + 1, len(self.assets)):
                    if abs(rc[ii, jj]) > 1e-12:
                        pi = self.resid_params.loc[self.assets[ii]].to_dict(); pj = self.resid_params.loc[self.assets[jj]].to_dict()
                        pi['mean'] = pj['mean'] = 0.0; pi['vol'] = pj['vol'] = 1.0
                        G[ii, jj] = G[jj, ii] = _gaussian_corr_for(pi, pj, rc[ii, jj], seed=ii * 1000 + jj)
            w, V = np.linalg.eigh(G)
            if w.min() < 1e-10:
                warnings.warn('residual Gaussian correlation projected onto the nearest positive-definite matrix')
                G = V @ np.diag(np.maximum(w, 1e-8)) @ V.T; d = np.sqrt(np.diag(G)); G = G / np.outer(d, d)
            self._gauss_corr = G
        return self._gauss_corr

    # ---- reports ------------------------------------------------------------------------
    def spec_report(self):
        """
        What was given and what was filled, per asset: source (``history`` / ``spec``),
        the number of given, fitted and zero betas, whether a mean, volatility or pair
        target was applied, the sample and current residual volatility.
        """
        self._check()
        bs = self.beta_source
        out = pd.DataFrame({'source': self.asset_source,
                            'betas given': (bs == 'given').sum(axis=1), 'betas fitted': (bs == 'fitted').sum(axis=1),
                            'betas zero': (bs == 'zero').sum(axis=1),
                            'mean target': self._targets_applied['mean'], 'vol target': self._targets_applied['vol'],
                            'pair target': self._targets_applied['pair'],
                            'sample resid vol': self.sample_resid_vol, 'resid vol': self.resid_vol})
        return out

    def implied_mean(self, factor_mean):
        """``alpha + beta @ factor_mean``: expected asset returns for a view on the factors."""
        self._check()
        m = pd.Series(factor_mean).astype(float).loc[self.factors]
        return self.alpha + self.beta.values @ m.values

    def implied_covariance(self, factor_targets=None):
        """``B Sigma_f B' + D`` with ``D`` the residual covariance (``resid_corr`` scaled by the residual volatilities)."""
        self._check()
        if factor_targets is None:
            _, Sigma_f = self._factor_moments()
        else:
            v = factor_targets.vol.loc[self.factors].values
            Sigma_f = np.outer(v, v) * factor_targets.corr.loc[self.factors, self.factors].values
        B = self.beta.values; rv = self.resid_vol.values
        S = B @ Sigma_f @ B.T + self.resid_corr.values * np.outer(rv, rv)
        return pd.DataFrame(S, index=self.assets, columns=self.assets)

    # ---- simulation -----------------------------------------------------------------------
    def simulate(self, F, residuals=True, seed=None):
        """
        Asset returns from factor scenarios: DataFrame (n x K) -> DataFrame (n x N);
        :class:`~cvinemarketgen.paths.Paths` -> ``Paths``. With ``residuals``, a
        Johnson SU residual is added per asset, independent across assets except for
        the pairs with a targeted correlation (Gaussian copula).
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
            n = len(Fd)
            if np.allclose(self.resid_corr.values, np.eye(len(self.assets))):
                for j, a in enumerate(self.assets):
                    X[:, j] += johnson_su_sample(self.resid_params.loc[a].to_dict(), n, seed=int(rng.integers(2 ** 31)))
            else:
                G = self._gaussian_corr()
                L = np.linalg.cholesky(G)
                Z = rng.standard_normal((n, len(self.assets))) @ L.T
                for j, a in enumerate(self.assets):
                    X[:, j] += johnson_su_from_normal(self.resid_params.loc[a].to_dict(), Z[:, j])
        return pd.DataFrame(X, index=Fd.index, columns=self.assets)

    # ---- persistence ------------------------------------------------------------------------
    def save(self, path):
        """Save the fitted model to JSON (coefficients, statistics, residual parameters, sources, report)."""
        self._check()
        d = {'kind': 'FactorModel', 'version': 2, 'assets': self.assets, 'factors': self.factors, 'hist_factors': self.hist_factors,
             'n_obs': self.n_obs, 'nw_lags': self.nw_lags, 'exposures': self.exposures,
             'alpha': self.alpha.to_dict(), 'beta': self.beta.to_dict(orient='index'),
             'se': self.se.to_dict(orient='index'), 'tstat': self.tstat.to_dict(orient='index'),
             'r2': self.r2.to_dict(), 'resid_vol': self.resid_vol.to_dict(), 'sample_resid_vol': self.sample_resid_vol.to_dict(),
             'resid_params': self.resid_params.to_dict(orient='index'), 'report': self.report.to_dict(orient='index'),
             'beta_source': self.beta_source.to_dict(orient='index'), 'asset_source': self.asset_source.to_dict(),
             'resid_corr': self.resid_corr.values.tolist(), 'targets_applied': self._targets_applied.to_dict(orient='index'),
             'factor_targets': None if self.factor_targets is None else self.factor_targets.to_dict(),
             'pair_report': None if self.pair_report is None else {'index': [list(i) for i in self.pair_report.index],
                                                                    'columns': list(self.pair_report.columns),
                                                                    'values': self.pair_report.values.tolist()}}
        with open(path, 'w') as f:
            json.dump(d, f, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))

    @classmethod
    def load(cls, path):
        """Rebuild a fitted model from :meth:`save`; ready to simulate."""
        from .targets import Targets
        with open(path) as f:
            d = json.load(f)
        m = cls.__new__(cls)
        m.assets, m.factors, m.n_obs, m.nw_lags = d['assets'], d['factors'], d['n_obs'], d['nw_lags']
        m.hist_factors = d.get('hist_factors', m.factors)
        m.synthetic_factors = [f for f in m.factors if f not in m.hist_factors]
        m.exposures = d.get('exposures')
        m._plan = {}
        m.factor_targets = None if d.get('factor_targets') is None else Targets.from_dict(d['factor_targets'])
        m.R = m.F = m.resid = None
        m.alpha = pd.Series(d['alpha']).loc[m.assets]
        m.beta = pd.DataFrame(d['beta']).T.loc[m.assets, m.factors]
        cols = ['alpha'] + m.factors
        m.se = pd.DataFrame(d['se']).T.loc[m.assets, cols]
        m.tstat = pd.DataFrame(d['tstat']).T.loc[m.assets, cols]
        m.r2 = pd.Series(d['r2']).loc[m.assets]
        m.resid_vol = pd.Series(d['resid_vol']).loc[m.assets]
        m.sample_resid_vol = pd.Series(d.get('sample_resid_vol', d['resid_vol'])).loc[m.assets]
        m.resid_params = pd.DataFrame(d['resid_params']).T.loc[m.assets, ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']].astype(float)
        if 'beta_source' in d:
            m.beta_source = pd.DataFrame(d['beta_source']).T.loc[m.assets, m.factors]
        else:
            m.beta_source = pd.DataFrame('fitted', index=m.assets, columns=m.factors)
        m.asset_source = pd.Series(d.get('asset_source', {a: 'history' for a in m.assets})).loc[m.assets]
        m.resid_corr = pd.DataFrame(np.array(d.get('resid_corr', np.eye(len(m.assets)).tolist())), index=m.assets, columns=m.assets)
        m._gauss_corr = None
        ta = d.get('targets_applied')
        m._targets_applied = (pd.DataFrame(ta).T.loc[m.assets, ['mean', 'vol', 'pair']].astype(bool) if ta
                              else pd.DataFrame(False, index=m.assets, columns=['mean', 'vol', 'pair']))
        m.target_report = None
        pr = d.get('pair_report')
        m.pair_report = None if pr is None else pd.DataFrame(pr['values'], columns=pr['columns'],
                                                             index=pd.MultiIndex.from_tuples([tuple(i) for i in pr['index']], names=['asset i', 'asset j']))
        m.fitted = True
        m._build_report()
        return m

    def _check(self):
        if not self.fitted:
            raise RuntimeError('call fit() first')
