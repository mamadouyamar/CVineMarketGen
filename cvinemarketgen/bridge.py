# -*- coding: utf-8 -*-
"""
A quarterly variable in a monthly market: the bridge equation.

    g_q = c + gamma' zbar_q + sum_i phi_i g_{q-i} + sigma * eta_q,

with ``zbar_q`` the mean over the quarter's three months of monthly parents
``z_t`` and ``g_q`` the quarterly growth (annualized, percent). Estimated by least
squares on the history; on simulated monthly paths each quarter's parent means go
through the equation and the innovation is drawn from a Johnson SU fitted to the
residual's skewness and kurtosis. The innovation is quarterly, so it is not in the
monthly vine: it is independent of the monthly residual layer.
"""
import numpy as np
import pandas as pd

from .functions import fit_johnson_su, johnson_su_sample
from .paths import Paths


class Bridge:
    """Bridge equation of a quarterly series on monthly ``parents`` (column names), with ``lags`` of itself."""

    def __init__(self, parents, lags=1):
        self.parents = list(parents)
        self.lags = int(lags)
        self.const = self.sigma = self.r2 = self.n_obs = None
        self.gamma = self.tstat = self.resid = self.residual = None
        self.phi = []
        self.name = f'Bridge({", ".join(self.parents)}; lags={self.lags})'

    # ---- estimation --------------------------------------------------------------------------
    @staticmethod
    def quarterly_means(X_m):
        """Quarterly means of a monthly DataFrame, quarters with fewer than three months dropped."""
        X = pd.DataFrame(X_m)
        q = X.index.asfreq('Q') if isinstance(X.index, pd.PeriodIndex) else pd.PeriodIndex(X.index, freq='M').asfreq('Q')
        g = X.groupby(q)
        means, counts = g.mean(), g.size()
        return means[counts == 3]

    def _design(self, y, Xq):
        cols = {'const': pd.Series(1.0, index=y.index)}
        for p in self.parents:
            cols[p] = Xq[p]
        for i in range(1, self.lags + 1):
            cols[f'g_l{i}'] = y.shift(i)
        D = pd.DataFrame(cols).loc[y.index]
        return D

    def fit(self, y_q, X_m):
        """``y_q``: Series with a quarterly PeriodIndex (growth, percent annualized); ``X_m``: monthly DataFrame with the parents."""
        y = pd.Series(y_q).astype(float)
        if not isinstance(y.index, pd.PeriodIndex):
            y.index = pd.PeriodIndex(y.index, freq='Q')
        Xq = self.quarterly_means(pd.DataFrame(X_m)[self.parents])
        common = y.index.intersection(Xq.index)
        y = y.loc[common]
        D = self._design(y, Xq.loc[common]).dropna()
        yy = y.loc[D.index].values
        Xd = D.values
        coef, _, _, _ = np.linalg.lstsq(Xd, yy, rcond=None)
        e = yy - Xd @ coef
        n, k = Xd.shape
        s2 = float(e @ e / (n - k))
        cov = s2 * np.linalg.inv(Xd.T @ Xd)
        coef = pd.Series(coef, index=D.columns)
        self.tstat = coef / np.sqrt(np.diag(cov))
        self.const = float(coef['const'])
        self.gamma = coef[self.parents]
        self.phi = [float(coef[f'g_l{i}']) for i in range(1, self.lags + 1)]
        self.sigma = float(np.sqrt(s2))
        self.r2 = float(1.0 - (e @ e) / ((yy - yy.mean()) @ (yy - yy.mean())))
        self.n_obs = int(n)
        self.resid = pd.Series(e / self.sigma, index=D.index, name='eta')
        z = self.resid
        skew, kurt = float(z.skew()), float(z.kurt() + 3.0)
        kurt = max(kurt, 3.1 + 2.0 * skew ** 2)
        self.residual = fit_johnson_su(skew, kurt)
        self.coef_ = coef
        return self

    @property
    def report(self):
        """Coefficients and t-statistics."""
        return pd.DataFrame({'coefficient': self.coef_, 't': self.tstat})

    # ---- simulation --------------------------------------------------------------------------
    def simulate(self, P, y_hist, seed=None):
        """
        Quarterly growth along monthly paths ``P`` (a ``Paths`` whose columns include the
        parents and whose first month follows a quarter end): months are grouped by
        three (a remainder is dropped), the quarter's parent means go through the
        equation with the lags taken from the end of ``y_hist`` and then from the path,
        and the innovation is drawn from the fitted Johnson SU. Returns a ``Paths``
        ``(n_paths, horizon // 3, 1)`` with the column ``'gdp'``.
        """
        n, h = P.n_paths, P.horizon
        Q = h // 3
        if Q == 0:
            raise ValueError('the horizon must cover at least one quarter (three months)')
        idx = [P.assets.index(p) for p in self.parents]
        Z = P.array[:, :3 * Q, :][:, :, idx].reshape(n, Q, 3, len(idx)).mean(axis=2)     # (n, Q, p)
        hist = list(np.asarray(y_hist, float)[-self.lags:]) if self.lags else []
        eta = johnson_su_sample(self.residual, n * Q, seed=seed).reshape(n, Q)
        out = np.empty((n, Q))
        lagged = np.tile(np.array(hist, float), (n, 1)) if self.lags else np.zeros((n, 0))
        for q in range(Q):
            g = self.const + Z[:, q, :] @ self.gamma.values + self.sigma * eta[:, q]
            for i in range(1, self.lags + 1):
                g = g + self.phi[i - 1] * lagged[:, -i]
            out[:, q] = g
            if self.lags:
                lagged = np.concatenate([lagged[:, 1:], g[:, None]], axis=1)
        return Paths(out[:, :, None], ['gdp'], layer='growth')

    # ---- persistence -------------------------------------------------------------------------
    def to_dict(self):
        return {'parents': self.parents, 'lags': self.lags, 'const': self.const, 'gamma': self.gamma.tolist(),
                'phi': self.phi, 'sigma': self.sigma, 'r2': self.r2, 'n_obs': self.n_obs, 'residual': self.residual,
                'coef': self.coef_.tolist(), 'tstat': self.tstat.tolist(), 'columns': list(self.coef_.index)}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['parents'], d['lags'])
        m.const, m.phi, m.sigma, m.r2, m.n_obs, m.residual = d['const'], list(d['phi']), d['sigma'], d['r2'], d['n_obs'], d['residual']
        m.gamma = pd.Series(d['gamma'], index=m.parents)
        m.coef_ = pd.Series(d['coef'], index=d['columns']); m.tstat = pd.Series(d['tstat'], index=d['columns'])
        return m

    def __repr__(self):
        return self.name + ('' if self.sigma is None else ', fitted')


def growth_to_level(Pq, base=100.0):
    """Level paths ``(n_paths, quarters)`` from a one-column ``Paths`` of annualized quarterly growth in percent."""
    g = Pq.array[:, :, 0] / 400.0
    return base * np.exp(np.cumsum(g, axis=1))


def recession_probability(Pq, k=2):
    """
    Share of paths with ``k`` consecutive quarters of negative growth within the
    horizon, and, by quarter, the share of paths whose first such run ends there.
    """
    neg = Pq.array[:, :, 0] < 0
    n, Q = neg.shape
    run = np.zeros(n, int); first = np.full(n, -1)
    for q in range(Q):
        run = np.where(neg[:, q], run + 1, 0)
        hit = (run >= k) & (first < 0)
        first[hit] = q
    by_q = np.array([(first == q).mean() for q in range(Q)])
    return float((first >= 0).mean()), by_q
