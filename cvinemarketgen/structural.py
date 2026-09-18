# -*- coding: utf-8 -*-
"""
The structural layer: a child variable driven by parent variables of the same
market plus its own innovation, which is the child's residual layer. On a path
the parents are simulated first and the child is rebuilt from them.

    ecm    (levels):  Delta s_t = c + kappa s_{t-1} + b' z_{t-1} + gamma' Delta z_t + sum_i phi_i Delta s_{t-i} + sigma eps_t
                      long-run relation theta = -b / kappa, adjustment kappa < 0
    linear (returns): s_t = c + gamma' z_t + sum_i phi_i s_{t-i} + sigma eps_t   (lags=0: the factor map)

Least squares, no dependency. Parents are taken as weakly exogenous: the child
does not feed back into them on the same date.
"""
import warnings

import numpy as np
import pandas as pd


class Structural:
    """
    Child on parents. ``parents`` are column names of the market's history;
    ``form`` is ``'ecm'`` (levels) or ``'linear'`` (returns); ``lags`` is the
    number of lagged changes (ecm) or lagged levels (linear) of the child.

    Attributes after ``fit``: ``child``, ``const``, ``kappa``, ``b``, ``theta``,
    ``gamma``, ``phi``, ``sigma``, ``half_life``, ``r2``, ``tstat``, ``n_obs``,
    ``loglik``, ``n_params``, ``bic``, ``state``.
    """

    kind = 'structural'

    def __init__(self, parents, form='ecm', lags=1):
        if form not in ('ecm', 'linear'):
            raise ValueError("form must be 'ecm' or 'linear'")
        self.parents = list(parents)
        self.form = form
        self.lags = int(lags)
        self.child = None
        self.const = self.kappa = self.sigma = None
        self.b = self.theta = self.gamma = None
        self.phi = []
        self.half_life = self.r2 = self.tstat = None
        self.state = None
        self._z = None
        self.n_obs = self.loglik = self.n_params = self.bic = None

    # ---- naming --------------------------------------------------------------
    @property
    def spec(self):
        return {'parents': self.parents, 'form': self.form, 'lags': self.lags}

    @property
    def name(self):
        head = 'ECM' if self.form == 'ecm' else 'Linear'
        return f"{head}({self.child or '?'} | {', '.join(self.parents)})"

    def __repr__(self):
        return f'Structural({self.name}{"" if self.sigma is None else ", fitted"})'

    def clone(self):
        return Structural(**self.spec)

    # ---- design --------------------------------------------------------------
    def _design(self, s, Z):
        cols = {'const': pd.Series(1.0, index=s.index)}
        if self.form == 'ecm':
            y = s.diff()
            cols['s_l1'] = s.shift(1)
            for p in self.parents:
                cols[f'{p}_l1'] = Z[p].shift(1)
            for p in self.parents:
                cols[f'd_{p}'] = Z[p].diff()
            for i in range(1, self.lags + 1):
                cols[f'd_s_l{i}'] = s.diff().shift(i)
        else:
            y = s
            for p in self.parents:
                cols[p] = Z[p]
            for i in range(1, self.lags + 1):
                cols[f's_l{i}'] = s.shift(i)
        X = pd.DataFrame(cols)
        ok = X.notna().all(axis=1) & y.notna()
        return y[ok], X[ok]

    def fit(self, s, Z):
        s = pd.Series(s).astype(float)
        Z = pd.DataFrame(Z)[self.parents].astype(float)
        self.child = s.name
        y, X = self._design(s, Z)
        coef, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        coef = pd.Series(coef, index=X.columns)
        e = y.values - X.values @ coef.values
        n, k = X.shape
        s2 = float(e @ e) / (n - k)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.values.T @ X.values)))
        self.tstat = coef / se
        self.const = float(coef['const'])
        if self.form == 'ecm':
            self.kappa = float(coef['s_l1'])
            self.b = pd.Series([coef[f'{p}_l1'] for p in self.parents], index=self.parents)
            if self.kappa < 0:
                self.theta = -self.b / self.kappa
                self.half_life = float(np.log(0.5) / np.log(1.0 + self.kappa))
            else:
                warnings.warn(f'{self.name}: adjustment coefficient {self.kappa:.3f} is not negative, no error correction')
                self.theta = self.b * np.nan
                self.half_life = np.inf
            self.gamma = pd.Series([coef[f'd_{p}'] for p in self.parents], index=self.parents)
            self.phi = [float(coef[f'd_s_l{i}']) for i in range(1, self.lags + 1)]
        else:
            self.gamma = pd.Series([coef[p] for p in self.parents], index=self.parents)
            self.phi = [float(coef[f's_l{i}']) for i in range(1, self.lags + 1)]
        self.sigma = float(np.sqrt(s2))
        self._z = pd.Series(e / self.sigma, index=y.index, name=self.child)
        self.r2 = float(1.0 - (e @ e) / ((y.values - y.values.mean()) @ (y.values - y.values.mean())))
        self.n_obs = int(n)
        self.loglik = float(-0.5 * n * (np.log(2 * np.pi * (e @ e) / n) + 1.0))
        self.n_params = int(k + 1)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(n))
        L = self.lags + 1
        self.state = {'s': s.values[-L:].tolist(), 'z': Z.values[-1].tolist()}
        return self

    def refit(self, s, Z):
        return self.clone().fit(s, Z)

    # ---- recursion -----------------------------------------------------------
    def _step(self, s_hist, z_prev, z_now, eps):
        """Next child value from ``s_hist`` (n, lags + 1) most recent last, parents ``z_prev``/``z_now`` (n, k), innovations ``eps`` (n,)."""
        if self.form == 'ecm':
            s_prev = s_hist[:, -1]
            ds = self.const + self.kappa * s_prev + z_prev @ self.b.values + (z_now - z_prev) @ self.gamma.values
            d = np.diff(s_hist, axis=1)                                 # d[:, -i] is Delta s_{t-i}
            for i, ph in enumerate(self.phi, start=1):
                ds = ds + ph * d[:, -i]
            return s_prev + ds + self.sigma * eps
        val = self.const + z_now @ self.gamma.values
        for i, ph in enumerate(self.phi, start=1):
            val = val + ph * s_hist[:, -i]
        return val + self.sigma * eps

    def _hist0(self, n):
        L = max(self.lags + 1, 1)
        return np.tile(np.array(self.state['s'], float)[-L:], (n, 1))

    def unfilter(self, z, Zpath):
        """Child paths ``(n_paths, horizon)`` from innovation paths ``z`` and the parents' simulated paths ``Zpath (n_paths, horizon, k)``."""
        z = np.asarray(z, float); Zpath = np.asarray(Zpath, float)
        n, H = z.shape
        s_hist = self._hist0(n)
        z_prev = np.tile(np.array(self.state['z'], float), (n, 1))
        out = np.empty_like(z)
        for t in range(H):
            x = self._step(s_hist, z_prev, Zpath[:, t], z[:, t])
            out[:, t] = x
            s_hist = np.column_stack([s_hist[:, 1:], x]); z_prev = Zpath[:, t]
        return out

    def filter_new(self, s_new, Z_new):
        """Standardized innovations of new child values given the parents' new values, continuing from ``state``."""
        v = np.asarray(s_new, float).ravel(); Zn = np.asarray(Z_new, float).reshape(len(v), -1)
        s_hist = self._hist0(1); z_prev = np.array(self.state['z'], float)[None, :]
        eps = np.empty(len(v))
        for t in range(len(v)):
            pred = self._step(s_hist, z_prev, Zn[t:t + 1], np.zeros(1))[0]
            eps[t] = (v[t] - pred) / self.sigma
            s_hist = np.column_stack([s_hist[:, 1:], [v[t]]]); z_prev = Zn[t:t + 1]
        return eps

    def simulate(self, n, seed=None, Zpath=None):
        """One child series of length ``n`` with Gaussian innovations along the given parents' path ``Zpath (1, n, k)``."""
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z, Zpath)[0]

    def filter(self):
        return self._z

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'child': self.child, 'const': self.const, 'kappa': self.kappa,
                'b': None if self.b is None else self.b.tolist(), 'gamma': self.gamma.tolist(), 'phi': list(self.phi),
                'sigma': self.sigma, 'half_life': self.half_life, 'r2': self.r2, 'state': self.state,
                'n_obs': self.n_obs, 'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m.child, m.const, m.kappa, m.sigma = d['child'], d['const'], d['kappa'], d['sigma']
        m.b = None if d['b'] is None else pd.Series(d['b'], index=m.parents)
        m.theta = None if m.b is None or not m.kappa or m.kappa >= 0 else -m.b / m.kappa
        m.gamma = pd.Series(d['gamma'], index=m.parents)
        m.phi = [float(x) for x in d['phi']]
        m.half_life, m.r2, m.state = d['half_life'], d['r2'], d['state']
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m
