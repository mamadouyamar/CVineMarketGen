# -*- coding: utf-8 -*-
"""
Assets on factors: a linear factor model fitted by least squares, whose betas
map simulated factor scenarios or paths (from a market on the factors) to
asset returns, with an optional independent Johnson SU residual per asset.

    r_t = alpha + beta' f_t + eps_t

Standard errors are Newey-West. Units and frequency are the data's.
"""
import json
import numpy as np
import pandas as pd

from .functions import fit_johnson_su, johnson_su_sample
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


class FactorModel:
    """
    Linear factor model of assets on factors, ``r = alpha + beta' f + eps``.

    Parameters
    ----------
    asset_returns : DataFrame
        One column per asset.
    factor_returns : DataFrame
        One column per factor, same frequency; aligned on the common index.
    nw_lags : int, optional
        Newey-West lags for the standard errors; default ``floor(4 (n/100)^(2/9))``.

    Notes
    -----
    After ``fit``: ``alpha`` (Series), ``beta`` (DataFrame assets x factors),
    ``se`` and ``tstat`` (DataFrames with an ``alpha`` column and one per factor),
    ``r2`` (Series), ``resid`` (DataFrame), ``resid_vol`` (Series), ``resid_params``
    (Johnson SU per asset, mean zero) and ``report`` (one table).
    """

    def __init__(self, asset_returns, factor_returns, nw_lags=None):
        R = pd.DataFrame(asset_returns).astype(float)
        F = pd.DataFrame(factor_returns).astype(float)
        idx = R.index.intersection(F.index)
        if len(idx) < 3 * (F.shape[1] + 1):
            raise ValueError('not enough common observations to fit the factor model')
        self.R = R.loc[idx]
        self.F = F.loc[idx]
        self.assets = list(self.R.columns)
        self.factors = list(self.F.columns)
        self.n_obs = len(idx)
        self.nw_lags = int(np.floor(4 * (self.n_obs / 100.0) ** (2.0 / 9.0))) if nw_lags is None else int(nw_lags)
        self.fitted = False

    def fit(self):
        X = np.column_stack([np.ones(self.n_obs), self.F.values])
        coef, *_ = np.linalg.lstsq(X, self.R.values, rcond=None)      # (K+1, N)
        E = self.R.values - X @ coef
        cols = ['alpha'] + self.factors
        self.alpha = pd.Series(coef[0], index=self.assets)
        self.beta = pd.DataFrame(coef[1:].T, index=self.assets, columns=self.factors)
        se = np.vstack([_newey_west_se(X, E[:, j], self.nw_lags) for j in range(len(self.assets))])
        self.se = pd.DataFrame(se, index=self.assets, columns=cols)
        self.tstat = pd.DataFrame(coef.T / se, index=self.assets, columns=cols)
        tss = ((self.R.values - self.R.values.mean(0)) ** 2).sum(0)
        self.r2 = pd.Series(1.0 - (E ** 2).sum(0) / tss, index=self.assets)
        self.resid = pd.DataFrame(E, index=self.R.index, columns=self.assets)
        self.resid_vol = self.resid.std(ddof=X.shape[1])
        rows = {}
        for a in self.assets:
            e = self.resid[a]
            rows[a] = fit_johnson_su(e.skew(), e.kurtosis() + 3.0, mean=0.0, vol=float(self.resid_vol[a]))
        self.resid_params = pd.DataFrame(rows).T[['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']].astype(float)
        rep = pd.concat([self.alpha.rename('alpha'), self.beta], axis=1)
        for c in cols:
            rep[f't({c})'] = self.tstat[c]
        rep['R2'] = self.r2
        rep['resid vol'] = self.resid_vol
        self.report = rep
        self.fitted = True
        return self

    def implied_mean(self, factor_mean):
        """``alpha + beta @ factor_mean``: expected asset returns for a view on the factors."""
        self._check()
        m = pd.Series(factor_mean).astype(float).loc[self.factors]
        return self.alpha + self.beta.values @ m.values

    def simulate(self, F, residuals=True, seed=None):
        """
        Asset returns from factor scenarios: DataFrame (n x K) -> DataFrame (n x N);
        :class:`~cvinemarketgen.paths.Paths` -> ``Paths``. With ``residuals``,
        an independent Johnson SU residual is added per asset.
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
            for j, a in enumerate(self.assets):
                X[:, j] += johnson_su_sample(self.resid_params.loc[a].to_dict(), len(Fd), seed=int(rng.integers(2 ** 31)))
        return pd.DataFrame(X, index=Fd.index, columns=self.assets)

    def save(self, path):
        """Save the fitted model to JSON (coefficients, statistics, residual parameters, report)."""
        self._check()
        d = {'kind': 'FactorModel', 'assets': self.assets, 'factors': self.factors, 'n_obs': self.n_obs,
             'nw_lags': self.nw_lags, 'alpha': self.alpha.to_dict(), 'beta': self.beta.to_dict(orient='index'),
             'se': self.se.to_dict(orient='index'), 'tstat': self.tstat.to_dict(orient='index'),
             'r2': self.r2.to_dict(), 'resid_vol': self.resid_vol.to_dict(),
             'resid_params': self.resid_params.to_dict(orient='index'), 'report': self.report.to_dict(orient='index')}
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        """Rebuild a fitted model from :meth:`save`; ready to simulate."""
        with open(path) as f:
            d = json.load(f)
        m = cls.__new__(cls)
        m.assets, m.factors, m.n_obs, m.nw_lags = d['assets'], d['factors'], d['n_obs'], d['nw_lags']
        m.R = m.F = m.resid = None
        m.alpha = pd.Series(d['alpha']).loc[m.assets]
        m.beta = pd.DataFrame(d['beta']).T.loc[m.assets, m.factors]
        cols = ['alpha'] + m.factors
        m.se = pd.DataFrame(d['se']).T.loc[m.assets, cols]
        m.tstat = pd.DataFrame(d['tstat']).T.loc[m.assets, cols]
        m.r2 = pd.Series(d['r2']).loc[m.assets]
        m.resid_vol = pd.Series(d['resid_vol']).loc[m.assets]
        m.resid_params = pd.DataFrame(d['resid_params']).T.loc[m.assets, ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']].astype(float)
        rep = pd.DataFrame(d['report']).T.loc[m.assets]
        m.report = rep[['alpha'] + m.factors + [f't({c})' for c in cols] + ['R2', 'resid vol']].astype(float)
        m.fitted = True
        return m

    def _check(self):
        if not self.fitted:
            raise RuntimeError('call fit() first')
