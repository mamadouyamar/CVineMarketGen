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

Per-asset models (version 0.3):

* :class:`GarchFamily`: one asset, ``'const'`` or ``'ar1'`` mean with a
  ``'const'``, ``'garch'``, ``'gjr'`` or ``'egarch'`` variance of orders
  ``(p, q)``; ``filter`` gives the standardized residuals ``epsilon_t`` and
  ``unfilter`` continues the recursion from the last observed state.
* :class:`AssetDynamics`: a dict ``{asset: model}`` sharing the market
  interface (``fit``, ``filter``, ``unfilter``, ``to_dict``). Specs are
  strings, ``'ar1-garch(1,1)'``, ``'const-gjr'``, ``'hmm(2)'``, parsed by
  :func:`parse_spec`.
"""
import re
import warnings

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


_SPEC_RE = re.compile(r'^(const|ar1)(?:-(const|garch|gjr|egarch)(?:\((\d+),(\d+)\))?)?$')
_HMM_RE = re.compile(r'^hmm(?:\((\d+)\))?$')


def _arch():
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("GARCH-family dynamics need the arch package: pip install arch  "
                          "(or pip install cvinemarketgen[garch])") from e
    return arch_model


class GarchFamily:
    """
    One asset's ``mean`` (``'const'`` or ``'ar1'``) with a ``vol`` model
    (``'const'``, ``'garch'``, ``'gjr'``, ``'egarch'``) of orders ``(p, q)``,
    fitted with ``arch`` on returns scaled by 100. ``filter`` gives the
    standardized residuals ``epsilon_t`` of the fitted sample; ``unfilter``
    continues the recursion from the last observed state, vectorised over paths.

    Attributes after ``fit``: ``params`` (dict ``a, b, omega, alpha, beta, gamma``),
    ``state`` (last return, residuals and variances), ``n_obs``, ``loglik``,
    ``n_params``, ``bic``.
    """

    kind = 'garch'

    def __init__(self, mean='const', vol='garch', p=1, q=1):
        if mean not in ('const', 'ar1') or vol not in ('const', 'garch', 'gjr', 'egarch'):
            raise ValueError(f'unknown model mean={mean!r}, vol={vol!r}')
        self.mean, self.vol = mean, vol
        self.p, self.q = (int(p), int(q)) if vol != 'const' else (0, 0)
        self.params = None
        self.state = None
        self.scale = 1.0
        self._z = None
        self.n_obs = self.loglik = self.n_params = self.bic = None

    # ---- naming --------------------------------------------------------------
    @property
    def spec(self):
        return {'mean': self.mean, 'vol': self.vol, 'p': self.p, 'q': self.q}

    @property
    def name(self):
        m = 'AR(1)' if self.mean == 'ar1' else 'Const'
        v = {'const': 'Const', 'garch': f'GARCH({self.p},{self.q})', 'gjr': f'GJR({self.p},{self.q})',
             'egarch': f'EGARCH({self.p},{self.q})'}[self.vol]
        return f'{m}-{v}'

    def __repr__(self):
        return f'GarchFamily({self.name}{"" if self.params is None else ", fitted"})'

    def clone(self):
        """Unfitted copy with the same specification."""
        return GarchFamily(**self.spec)

    def refit(self, y):
        """Fit a fresh copy on ``y`` (bootstrap refit)."""
        return self.clone().fit(y)

    # ---- fit -----------------------------------------------------------------
    def fit(self, y):
        arch_model = _arch()
        s = pd.Series(y).astype(float)
        self.scale = 100.0 if s.abs().mean() < 0.5 else 1.0
        v = s.values * self.scale
        mkw = {'mean': 'Constant'} if self.mean == 'const' else {'mean': 'AR', 'lags': 1}
        vkw = {'const': {'vol': 'Constant'}, 'garch': {'vol': 'GARCH', 'p': self.p, 'o': 0, 'q': self.q},
               'gjr': {'vol': 'GARCH', 'p': self.p, 'o': 1, 'q': self.q},
               'egarch': {'vol': 'EGARCH', 'p': self.p, 'o': 0, 'q': self.q}}[self.vol]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = arch_model(v, dist='normal', rescale=False, **mkw, **vkw).fit(disp='off')
        prm = res.params
        a = float(prm['mu']) if self.mean == 'const' else float(prm['Const'])
        b = 0.0 if self.mean == 'const' else float(prm[[k for k in prm.index if k.endswith('[1]')
                                                        and not k.startswith(('alpha', 'beta', 'gamma'))][0]])
        if self.vol == 'const':
            omega, alpha, beta, gamma = float(prm['sigma2']), [], [], 0.0
        else:
            omega = float(prm['omega'])
            alpha = [float(prm[f'alpha[{i}]']) for i in range(1, self.p + 1)]
            beta = [float(prm[f'beta[{j}]']) for j in range(1, self.q + 1)]
            gamma = float(prm['gamma[1]']) if self.vol == 'gjr' else 0.0
        self.params = {'a': a, 'b': b, 'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
        e = np.asarray(res.resid, float)
        h = np.asarray(res.conditional_volatility, float) ** 2
        ok = np.isfinite(e) & np.isfinite(h)
        self._z = pd.Series((e / np.sqrt(h))[ok], index=s.index[ok], name=s.name)
        L = max(self.p, self.q, 1)
        self.state = {'y': float(v[-1]), 'e': e[ok][-L:].tolist(), 'h': h[ok][-L:].tolist()}
        self.n_obs = int(ok.sum())
        self.n_params = int(len(prm))
        # arch's likelihood is that of ``y * scale``; put it back on the data's scale (Jacobian ``n log scale``)
        # so that log-likelihoods and BICs are comparable across candidates and with the HMM
        self.loglik = float(res.loglikelihood) + self.n_obs * np.log(self.scale)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    # ---- recursion -----------------------------------------------------------
    def _next_var(self, e_hist, h_hist):
        """Next variance from histories ``(n_paths, L)``, most recent last."""
        P = self.params
        if self.vol == 'const':
            return np.full(e_hist.shape[0], P['omega'])
        if self.vol == 'egarch':
            zh = e_hist / np.sqrt(h_hist)
            lh = P['omega'] + sum(P['alpha'][i - 1] * (np.abs(zh[:, -i]) - np.sqrt(2.0 / np.pi)) for i in range(1, self.p + 1)) \
                + sum(P['beta'][j - 1] * np.log(h_hist[:, -j]) for j in range(1, self.q + 1))
            return np.exp(lh)
        h = P['omega'] + sum(P['alpha'][i - 1] * e_hist[:, -i] ** 2 for i in range(1, self.p + 1)) \
            + sum(P['beta'][j - 1] * h_hist[:, -j] for j in range(1, self.q + 1))
        if self.vol == 'gjr':
            h = h + P['gamma'] * e_hist[:, -1] ** 2 * (e_hist[:, -1] < 0)
        return h

    def _start(self, n_paths):
        st = self.state
        return (np.full(n_paths, st['y']), np.tile(np.array(st['e'], float), (n_paths, 1)),
                np.tile(np.array(st['h'], float), (n_paths, 1)))

    def filter(self):
        """Standardized residuals ``epsilon_t`` of the fitted sample (Series)."""
        return self._z

    def filter_new(self, y_new):
        """Standardized residuals of new observations (scale of ``y``), continuing from the last state."""
        v = np.asarray(y_new, float) * self.scale
        y_prev, e_hist, h_hist = self._start(1)
        z = np.empty(len(v))
        for t in range(len(v)):
            h = self._next_var(e_hist, h_hist)
            e = v[t] - (self.params['a'] + self.params['b'] * y_prev)
            z[t] = e[0] / np.sqrt(h[0])
            y_prev = np.array([v[t]])
            e_hist = np.column_stack([e_hist[:, 1:], e]); h_hist = np.column_stack([h_hist[:, 1:], h])
        return z

    def unfilter(self, z):
        """Returns from standardized-residual paths ``(n_paths, horizon)``, from the last observed state."""
        z = np.asarray(z, float)
        n_paths, horizon = z.shape
        y_prev, e_hist, h_hist = self._start(n_paths)
        out = np.empty_like(z)
        for t in range(horizon):
            h = self._next_var(e_hist, h_hist)
            e = np.sqrt(h) * z[:, t]
            y = self.params['a'] + self.params['b'] * y_prev + e
            out[:, t] = y
            y_prev = y
            e_hist = np.column_stack([e_hist[:, 1:], e]); h_hist = np.column_stack([h_hist[:, 1:], h])
        return out / self.scale

    def simulate(self, n, seed=None):
        """One simulated series of length ``n`` with Gaussian innovations (used by the bootstrap test)."""
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z)[0]

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'scale': self.scale, 'params': self.params, 'state': self.state,
                'n_obs': self.n_obs, 'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m.scale, m.params, m.state = d['scale'], d['params'], d['state']
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m


def parse_spec(spec):
    """``'const-garch(1,1)'``, ``'ar1-gjr(1,2)'``, ``'ar1'`` (constant variance), ``'hmm(3)'`` -> model."""
    if not isinstance(spec, str):
        return spec
    s = spec.strip().lower()
    m = _HMM_RE.match(s)
    if m:
        from .hmm import GaussianHMM
        return GaussianHMM(int(m.group(1) or 2))
    m = _SPEC_RE.match(s)
    if not m:
        raise ValueError(f'cannot parse dynamics spec {spec!r}')
    mean, vol, p, q = m.group(1), m.group(2) or 'const', m.group(3), m.group(4)
    return GarchFamily(mean, vol, int(p or 1), int(q or 1))


def model_from_dict(d):
    """Univariate model from its ``to_dict`` output."""
    if d['kind'] == 'garch':
        return GarchFamily.from_dict(d)
    from .hmm import GaussianHMM
    return GaussianHMM.from_dict(d)


class AssetDynamics:
    """
    Per-asset dynamics: one univariate model per asset (``{asset: spec or model}``),
    sharing the market interface. ``report`` lists the fitted model, its number
    of parameters, log-likelihood and BIC per asset; ``candidates`` holds the
    selection table when the models were chosen by :func:`selection.select_dynamics`.
    """

    name = 'assets'

    def __init__(self, models=None):
        self.models = {a: parse_spec(m) for a, m in (models or {}).items()}
        self.report = None
        self.candidates = None

    def __repr__(self):
        return f'AssetDynamics({ {a: m.name for a, m in self.models.items()} })'

    def fit(self, history):
        h = pd.DataFrame(history).astype(float)
        for a in h.columns:
            if a not in self.models:
                raise ValueError(f'no dynamics model given for {a!r}')
            self.models[a].fit(h[a])
        self.models = {a: self.models[a] for a in h.columns}      # history order
        if self.report is None:
            self.report = pd.DataFrame({a: {'model': m.name, 'n_params': m.n_params, 'loglik': m.loglik, 'bic': m.bic}
                                        for a, m in self.models.items()}).T
        return self

    def filter(self, history):
        """Residual layer of the fitted sample, one column per asset, rows where every asset has a value."""
        cols = list(pd.DataFrame(history).columns)
        return pd.concat({a: self.models[a].filter() for a in cols}, axis=1, sort=False).dropna()[cols]

    def unfilter(self, Z):
        """Returns from residual paths ``(n_paths, horizon, N)``, assets in the order of ``models``."""
        Z = np.asarray(Z, float)
        out = np.empty_like(Z)
        for j, a in enumerate(self.models):
            out[:, :, j] = self.models[a].unfilter(Z[:, :, j])
        return out

    def to_dict(self):
        return {'name': self.name, 'models': {a: m.to_dict() for a, m in self.models.items()},
                'report': None if self.report is None else self.report.to_dict(orient='index'),
                'candidates': None if self.candidates is None else self.candidates.to_dict(orient='list')}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.models = {a: model_from_dict(md) for a, md in d['models'].items()}
        if d.get('report'):
            m.report = pd.DataFrame(d['report']).T.loc[list(m.models)]
        if d.get('candidates'):
            m.candidates = pd.DataFrame(d['candidates'])
        return m


DYNAMICS = {'ar1': AR1, 'ar1-garch': AR1GARCH}


def make_dynamics(name):
    """
    None, ``'ar1'``, ``'ar1-garch'``, ``'auto'`` (kept as the string, resolved at fit),
    a per-asset dict of specs, or a model container.
    """
    if name is None or name == 'auto':
        return name
    if isinstance(name, (AR1, AR1GARCH, AssetDynamics)):
        return name
    if isinstance(name, dict):
        return AssetDynamics(name)
    try:
        return DYNAMICS[name]()
    except KeyError:
        raise ValueError(f"dynamics must be None, 'ar1', 'ar1-garch', 'auto', a dict of specs or an AssetDynamics, got {name!r}")


def dynamics_from_dict(d):
    """Dynamics container from its ``to_dict`` output."""
    return {'ar1': AR1, 'ar1-garch': AR1GARCH, 'assets': AssetDynamics}[d['name']].from_dict(d)
