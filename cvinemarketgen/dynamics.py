# -*- coding: utf-8 -*-
"""
Serial dependence for path simulation: the generator is fitted on the
residuals of a per-asset time-series model, and simulated residuals are
filtered back through the model to produce paths.

Two models:

* :class:`AR1`: ``y_t = a + b y_{t-1} + e_t`` by least squares. The targets of
  the residual layer follow from the return targets by the moment transfer of
  Appendix A of the paper, so it works with LTCMA targets as well as with a
  history.
* :class:`AR1GARCH`: AR(1) mean with GARCH(1,1) variance, fitted with the
  ``arch`` package (optional dependency, ``pip install cvinemarketgen[garch]``).
  The generator is fitted on the standardized residuals; supported with
  history-based targets.
"""
import numpy as np
import pandas as pd

from .targets import Targets


class AR1:
    """Per-asset AR(1) filter, ``y_t = a + b y_{t-1} + e_t``."""

    name = 'ar1'

    def __init__(self):
        self.params = None      # DataFrame: a, b, sigma_e per asset
        self.last = None        # last observed return per asset

    def fit(self, returns):
        r = pd.DataFrame(returns).astype(float)
        rows = {}
        for a in r.columns:
            y = r[a].values
            X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
            coef, *_ = np.linalg.lstsq(X, y[1:], rcond=None)
            e = y[1:] - X @ coef
            rows[a] = {'a': coef[0], 'b': coef[1], 'sigma_e': e.std(ddof=2)}
        self.params = pd.DataFrame(rows).T
        self.last = r.iloc[-1]
        return self

    def filter(self, returns):
        """Residuals ``e_t`` of the fitted model (first observation dropped)."""
        r = pd.DataFrame(returns).astype(float)
        out = {}
        for a in r.columns:
            y = r[a].values
            out[a] = y[1:] - (self.params.loc[a, 'a'] + self.params.loc[a, 'b'] * y[:-1])
        return pd.DataFrame(out, index=r.index[1:])

    def unfilter(self, eps, y0=None):
        """
        Returns from residual paths. ``eps`` has shape (n_paths, horizon, N);
        ``y0`` (Series per asset) is the starting return, the last observed one by default.
        """
        eps = np.asarray(eps, float)
        n_paths, horizon, N = eps.shape
        a = self.params['a'].values; b = self.params['b'].values
        y_prev = np.tile((self.last if y0 is None else y0).values.astype(float), (n_paths, 1))
        out = np.empty_like(eps)
        for t in range(horizon):
            y_prev = a + b * y_prev + eps[:, t, :]
            out[:, t, :] = y_prev
        return out

    def transfer_targets(self, targets):
        """
        Targets of the residual layer from targets of the return layer, by the
        AR(1) moment transfer of Appendix A of the paper:
        ``mu_e = mu (1 - b)``, ``sigma_e^2 = sigma^2 (1 - b^2)``,
        ``skew_e = skew (1 - b^3) / (1 - b^2)^{3/2}``,
        ``kurt_e = (kurt (1 + b^2) - 6 b^2) / (1 - b^2)``,
        ``rho_e = rho (1 - b_x b_y) / sqrt((1 - b_x^2)(1 - b_y^2))``.
        """
        b = self.params.loc[targets.assets, 'b'].values
        mean = targets.mean.values * (1 - b)
        vol = targets.vol.values * np.sqrt(1 - b ** 2)
        skew = targets.skew.values * (1 - b ** 3) / (1 - b ** 2) ** 1.5
        kurt = (targets.kurt.values * (1 + b ** 2) - 6 * b ** 2) / (1 - b ** 2)
        rho = targets.corr.values
        bb = np.outer(b, b)
        corr = rho * (1 - bb) / np.sqrt(np.outer(1 - b ** 2, 1 - b ** 2))
        np.fill_diagonal(corr, 1.0)
        t = Targets(mean=mean, vol=vol, corr=corr, skew=skew, kurt=kurt, assets=targets.assets,
                    history=None if targets.history is None else self.filter(targets.history))
        t.layer = 'residuals'
        t.freq = targets.freq
        return t

    def to_dict(self):
        return {'name': self.name, 'params': self.params.to_dict(orient='index'), 'last': self.last.to_dict()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.params = pd.DataFrame(d['params']).T[['a', 'b', 'sigma_e']] if d['params'] else None
        m.last = pd.Series(d['last'])
        return m


class AR1GARCH:
    """
    Per-asset AR(1)-GARCH(1,1): ``y_t = a + b y_{t-1} + e_t``, ``e_t = sqrt(h_t) z_t``,
    ``h_t = omega + alpha e_{t-1}^2 + beta h_{t-1}``. Fitted with ``arch``; the
    generator works on the standardized residuals ``z_t``.
    """

    name = 'ar1-garch'

    def __init__(self):
        self.params = None      # DataFrame: a, b, omega, alpha, beta
        self.last = None        # DataFrame: y, e, h at the last observation
        self.scale = 1.0        # returns are rescaled by this factor before fitting (arch prefers percent-sized data)

    def _arch(self):
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("dynamics='ar1-garch' needs the arch package: pip install arch  "
                              "(or pip install cvinemarketgen[garch])") from e
        return arch_model

    def fit(self, returns):
        arch_model = self._arch()
        r = pd.DataFrame(returns).astype(float)
        self.scale = 100.0 if r.abs().mean().mean() < 0.5 else 1.0
        rows, last = {}, {}
        for a in r.columns:
            y = r[a].values * self.scale
            res = arch_model(y, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal', rescale=False).fit(disp='off')
            p = res.params
            rows[a] = {'a': p['Const'], 'b': p[[k for k in p.index if k.startswith('y[1]') or k.endswith('[1]')][0]],
                       'omega': p['omega'], 'alpha': p['alpha[1]'], 'beta': p['beta[1]']}
            e = res.resid; h = res.conditional_volatility ** 2
            last[a] = {'y': y[-1], 'e': float(np.asarray(e)[-1]), 'h': float(np.asarray(h)[-1])}
        self.params = pd.DataFrame(rows).T
        self.last = pd.DataFrame(last).T
        return self

    def _resid_and_var(self, y):
        a, b, om, al, be = (self.params.loc[y.name, k] for k in ('a', 'b', 'omega', 'alpha', 'beta'))
        v = y.values
        e = np.full(len(v), np.nan); h = np.full(len(v), np.nan)
        e[1:] = v[1:] - (a + b * v[:-1])
        h[1] = om / max(1 - al - be, 1e-6)
        for t in range(2, len(v)):
            h[t] = om + al * e[t - 1] ** 2 + be * h[t - 1]
        return e, h

    def filter(self, returns):
        """Standardized residuals ``z_t = e_t / sqrt(h_t)`` (first observation dropped)."""
        r = pd.DataFrame(returns).astype(float) * self.scale
        out = {}
        for a in r.columns:
            e, h = self._resid_and_var(r[a])
            out[a] = e / np.sqrt(h)
        return pd.DataFrame(out, index=r.index).iloc[1:]

    def unfilter(self, z, y0=None):
        """Returns from standardized-residual paths, shape (n_paths, horizon, N), starting from the last observed state."""
        z = np.asarray(z, float)
        n_paths, horizon, N = z.shape
        P = self.params
        a, b = P['a'].values, P['b'].values
        om, al, be = P['omega'].values, P['alpha'].values, P['beta'].values
        y_prev = np.tile(self.last['y'].values, (n_paths, 1)) if y0 is None else np.tile(np.asarray(y0, float) * self.scale, (n_paths, 1))
        e_prev = np.tile(self.last['e'].values, (n_paths, 1))
        h_prev = np.tile(self.last['h'].values, (n_paths, 1))
        out = np.empty_like(z)
        for t in range(horizon):
            h = om + al * e_prev ** 2 + be * h_prev
            e = np.sqrt(h) * z[:, t, :]
            y = a + b * y_prev + e
            out[:, t, :] = y
            y_prev, e_prev, h_prev = y, e, h
        return out / self.scale

    def to_dict(self):
        return {'name': self.name, 'scale': self.scale, 'params': self.params.to_dict(orient='index'),
                'last': self.last.to_dict(orient='index')}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.scale = d['scale']
        m.params = pd.DataFrame(d['params']).T[['a', 'b', 'omega', 'alpha', 'beta']]
        m.last = pd.DataFrame(d['last']).T[['y', 'e', 'h']]
        return m


DYNAMICS = {'ar1': AR1, 'ar1-garch': AR1GARCH}


def make_dynamics(name):
    """Instantiate a dynamics model by name (``'ar1'`` or ``'ar1-garch'``), or return None."""
    if name is None:
        return None
    if isinstance(name, (AR1, AR1GARCH)):
        return name
    try:
        return DYNAMICS[name]()
    except KeyError:
        raise ValueError(f"dynamics must be None, 'ar1' or 'ar1-garch', got {name!r}")
