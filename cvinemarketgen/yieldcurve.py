# -*- coding: utf-8 -*-
"""
The yield curve as three factors, and fixed income priced from it.

Nelson-Siegel (Nelson and Siegel, 1987; Diebold and Li, 2006):

    y_t(tau) = L_t + S_t (1 - e^{-lam tau}) / (lam tau) + C_t ((1 - e^{-lam tau}) / (lam tau) - e^{-lam tau}),

the factors L_t, S_t, C_t estimated by least squares date by date for a fixed
decay ``lam`` (per year; 0.7308 is Diebold and Li's 0.0609 per month), with
principal components as the benchmark. The factors are three observed series
that the market filters as a block; curves at any maturity, discount factors,
bond prices and the returns of zero-coupon or constant-maturity par bonds follow
from simulated factor paths. Yields in percent, maturities in years, continuous
compounding.
"""
import re

import numpy as np
import pandas as pd

from .paths import Paths

FACTOR_NAMES = ['level', 'slope', 'curvature']
_MAT_RE = re.compile(r'^\s*(?:DGS|GS|TB|Y)?\s*(\d+(?:\.\d+)?)\s*(MO|M|Y|YR)?\s*$', re.I)


def _maturity_of(name):
    """Years from a column name: '10', '10y', 'DGS10', 'DGS3MO', 'GS1', '0.5'."""
    m = _MAT_RE.match(str(name))
    if m is None:
        raise ValueError(f'cannot read a maturity from column {name!r}; pass maturities={{name: years}}')
    v = float(m.group(1))
    unit = (m.group(2) or 'Y').upper()
    return v / 12.0 if unit in ('MO', 'M') else v


class NelsonSiegel:
    """
    Nelson-Siegel curve with decay ``lam`` per year (``'auto'``: grid search on the
    total root mean squared fitting error over ``[0.2, 2.0]``).

    Attributes after ``fit``: ``factors`` (DataFrame level, slope, curvature),
    ``tau`` (maturities in years), ``columns`` (the yields' names), ``rmse``
    (Series by maturity, in the yields' unit).
    """

    def __init__(self, lam=0.7308):
        self.lam = lam
        self.tau = None
        self.columns = None
        self.factors = None
        self.rmse = None

    @staticmethod
    def _loadings(tau, lam):
        tau = np.atleast_1d(np.asarray(tau, float))
        x = lam * tau
        f1 = (1.0 - np.exp(-x)) / x
        return np.column_stack([np.ones_like(tau), f1, f1 - np.exp(-x)])

    def loadings(self, tau):
        """Loadings ``(len(tau), 3)`` of level, slope and curvature at maturities ``tau`` (years)."""
        return self._loadings(tau, self.lam)

    def fit(self, yields, maturities=None):
        """
        Factors by least squares per date. ``maturities`` maps column names to years
        when the names cannot be read (``'DGS10'``, ``'10y'``, ``'3MO'`` and plain
        numbers are read automatically).
        """
        Y = pd.DataFrame(yields).astype(float)
        self.columns = list(Y.columns)
        if maturities is not None:
            self.tau = [float(maturities[c]) for c in self.columns]
        else:
            self.tau = [_maturity_of(c) for c in self.columns]
        if isinstance(self.lam, str):
            grid = np.linspace(0.2, 2.0, 37)
            tot = []
            for g in grid:
                L = self._loadings(self.tau, g)
                F = np.linalg.lstsq(L, Y.values.T, rcond=None)[0].T
                tot.append(np.sqrt(((Y.values - F @ L.T) ** 2).mean()))
            self.lam = float(grid[int(np.argmin(tot))])
        L = self.loadings(self.tau)
        F = np.linalg.lstsq(L, Y.values.T, rcond=None)[0].T
        self.factors = pd.DataFrame(F, index=Y.index, columns=FACTOR_NAMES)
        fit = F @ L.T
        self.rmse = pd.Series(np.sqrt(((Y.values - fit) ** 2).mean(axis=0)), index=self.columns)
        return self

    def curve(self, factors, tau=None):
        """
        Yields at maturities ``tau`` (default: the fitted maturities, named as the
        fitted columns) from factors: a DataFrame gives a DataFrame, a ``Paths`` of
        the three factors gives a ``Paths`` of yields.
        """
        if tau is None:
            t, names = self.tau, self.columns
        else:
            t = [float(x) for x in np.atleast_1d(tau)]
            names = [f'{x:g}y' for x in t]
        L = self.loadings(t)
        if isinstance(factors, Paths):
            return Paths(factors.array @ L.T, names, layer='yields')
        F = pd.DataFrame(factors)
        return pd.DataFrame(F.values @ L.T, index=F.index, columns=names)

    def to_dict(self):
        return {'lam': self.lam, 'tau': self.tau, 'columns': self.columns,
                'rmse': None if self.rmse is None else self.rmse.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['lam'])
        m.tau, m.columns = d['tau'], d['columns']
        if d.get('rmse') is not None:
            m.rmse = pd.Series(d['rmse'], index=m.columns)
        return m

    def __repr__(self):
        n = 0 if self.factors is None else len(self.factors)
        return f'NelsonSiegel(lam={self.lam}, maturities={self.tau}, n={n})'


class PCACurve:
    """
    Principal components of the yields' levels, the benchmark to Nelson-Siegel with
    the same interface: ``fit``, ``factors`` (scores), ``explained`` (share of
    variance by component), ``curve`` (yields from scores, DataFrame or ``Paths``).
    """

    def __init__(self, n_components=3):
        self.n = int(n_components)
        self.columns = self.mean = self.vectors = self.factors = self.explained = None

    def fit(self, yields):
        Y = pd.DataFrame(yields).astype(float)
        self.columns = list(Y.columns)
        self.mean = Y.values.mean(axis=0)
        w, v = np.linalg.eigh(np.cov(Y.values.T))
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        v = v * np.where(v[-1, :] < 0, -1.0, 1.0)      # the longest maturity loads positively
        self.explained = pd.Series(w / w.sum(), index=[f'pc{i + 1}' for i in range(len(w))])
        self.vectors = v[:, :self.n]
        names = [f'pc{i + 1}' for i in range(self.n)]
        self.factors = pd.DataFrame((Y.values - self.mean) @ self.vectors, index=Y.index, columns=names)
        return self

    def loadings(self):
        """DataFrame of the component vectors by maturity."""
        return pd.DataFrame(self.vectors, index=self.columns, columns=list(self.factors.columns))

    def curve(self, factors):
        """Yields at the fitted maturities from component scores (DataFrame or ``Paths``)."""
        if isinstance(factors, Paths):
            return Paths(factors.array @ self.vectors.T + self.mean, self.columns, layer='yields')
        F = pd.DataFrame(factors)
        return pd.DataFrame(F.values @ self.vectors.T + self.mean, index=F.index, columns=self.columns)

    def __repr__(self):
        return f'PCACurve(n_components={self.n}, maturities={self.columns})'


# ---- pricing -------------------------------------------------------------------------------
def yield_at(F, tau, ns):
    """Yields (percent) at maturities ``tau`` from factor rows ``F`` of shape ``(..., 3)``: shape ``(..., len(tau))``."""
    F = np.asarray(F, float)
    return F @ ns.loadings(np.atleast_1d(tau)).T


def discount(y, tau):
    """Discount factor ``exp(-tau y / 100)`` for a yield in percent and a maturity in years."""
    return np.exp(-np.asarray(tau, float) * np.asarray(y, float) / 100.0)


def _cash_times(maturity, freq, dt=0.0):
    k = np.arange(1, int(round(maturity * freq)) + 1)
    return k / freq - dt


def bond_price(F, coupon, maturity, ns, freq=2):
    """Price per 100 of face of a bond paying ``coupon`` percent a year, ``freq`` times a year, on the curve of ``F``."""
    t = _cash_times(maturity, freq)
    P = discount(yield_at(F, t, ns), t)
    return (np.asarray(coupon, float)[..., None] / freq * P).sum(axis=-1) + 100.0 * P[..., -1]


def par_yield(F, maturity, ns, freq=2):
    """Coupon (percent a year) that prices a bond of ``maturity`` at par on the curve of ``F``."""
    t = _cash_times(maturity, freq)
    P = discount(yield_at(F, t, ns), t)
    return freq * 100.0 * (1.0 - P[..., -1]) / P.sum(axis=-1)


def zero_return(F0, F1, tau, ns, dt=1.0 / 12):
    """Return of a zero of maturity ``tau`` bought on curve ``F0`` and sold ``dt`` years later on curve ``F1``."""
    p0 = discount(yield_at(F0, [tau], ns)[..., 0], tau)
    p1 = discount(yield_at(F1, [tau - dt], ns)[..., 0], tau - dt)
    return p1 / p0 - 1.0


def constant_maturity_return(F0, F1, tau, ns, dt=1.0 / 12, freq=2):
    """
    Return of a par bond of maturity ``tau`` issued at par on curve ``F0`` (its coupon
    is the par yield) and valued ``dt`` years later on curve ``F1`` with its remaining
    cash flows (no coupon falls within ``dt`` when ``dt < 1/freq``): carry, roll-down
    and price change together.
    """
    c = par_yield(F0, tau, ns, freq)
    t = _cash_times(tau, freq, dt)
    P = discount(yield_at(F1, t, ns), t)
    price1 = (np.asarray(c, float)[..., None] / freq * P).sum(axis=-1) + 100.0 * P[..., -1]
    return price1 / 100.0 - 1.0


def curve_returns(P, tau, ns, F0, kind='par', dt=1.0 / 12, freq=2):
    """
    Per-period returns of bonds of maturities ``tau`` along simulated factor paths ``P``
    (a ``Paths`` of level, slope, curvature in that order), from the last observed
    factors ``F0``. ``kind`` is ``'par'`` (a constant-maturity par bond re-issued each
    period) or ``'zero'`` (a zero of maturity ``tau`` held for ``dt``). Returns a ``Paths``.
    """
    if kind not in ('par', 'zero'):
        raise ValueError("kind must be 'par' or 'zero'")
    A = P.array
    F0 = np.asarray(F0, float).reshape(1, 1, -1)
    prev = np.concatenate([np.repeat(F0, A.shape[0], axis=0), A[:, :-1]], axis=1)
    out = np.empty((A.shape[0], A.shape[1], len(tau)))
    for j, t in enumerate(tau):
        if kind == 'par':
            out[:, :, j] = constant_maturity_return(prev, A, float(t), ns, dt, freq)
        else:
            out[:, :, j] = zero_return(prev, A, float(t), ns, dt)
    return Paths(out, [f'{float(t):g}y' for t in tau], layer='returns')
