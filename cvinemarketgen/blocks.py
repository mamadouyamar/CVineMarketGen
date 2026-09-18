# -*- coding: utf-8 -*-
"""
Block filters: a vector error-correction model or a vector autoregression on a
block of variables, whose standardized innovations form the block's residual
layer (one column per variable; their cross-correlation is left to the vine).
Paths are rebuilt by the recursion from the last observed rows, in the units of
the fitted series (levels for a VECM). Estimated with statsmodels, an optional
dependency (``pip install statsmodels`` or ``pip install cvinemarketgen[blocks]``).

    VECM:  Delta x_t = c + alpha beta' x_{t-1} + sum_{i=1}^{q} Gamma_i Delta x_{t-i} + diag(sigma) eps_t
    VAR:   x_t = c + sum_{i=1}^{q} Gamma_i x_{t-i} + diag(sigma) eps_t
"""
import numpy as np
import pandas as pd


def _sm():
    try:
        from statsmodels.tsa.vector_ar import vecm
        from statsmodels.tsa.api import VAR
    except ImportError as e:
        raise ImportError('block dynamics need statsmodels: pip install statsmodels '
                          '(or pip install cvinemarketgen[blocks])') from e
    return vecm, VAR


class _Block:
    """Shared pieces of the two block models."""

    kind = 'block'

    def __init__(self):
        self.columns = None
        self.Gamma = []
        self.const = None
        self.sigma = None
        self.state = None
        self._z = None
        self.n_obs = self.loglik = self.n_params = self.bic = None

    @property
    def p(self):
        return len(self.columns)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name}{"" if self.sigma is None else ", fitted"})'

    def filter(self):
        """Standardized innovations of the fitted sample, one column per variable (DataFrame)."""
        return self._z

    def simulate(self, n, seed=None):
        """One simulated block of length ``n`` with Gaussian innovations, shape ``(n, p)``."""
        z = np.random.default_rng(seed).standard_normal((1, int(n), self.p))
        return self.unfilter(z)[0]

    def refit(self, X):
        """Fit a fresh copy on ``X`` (bootstrap refit)."""
        return self.clone().fit(X)

    def _base_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'columns': self.columns, 'lags': self.lags,
                'Gamma': [G.tolist() for G in self.Gamma], 'const': self.const.tolist(), 'sigma': self.sigma.tolist(),
                'state': self.state.tolist(), 'n_obs': self.n_obs, 'loglik': self.loglik,
                'n_params': self.n_params, 'bic': self.bic}

    def _load_base(self, d):
        self.columns, self.lags = list(d['columns']), int(d['lags'])
        self.Gamma = [np.array(G, float) for G in d['Gamma']]
        self.const, self.sigma, self.state = (np.array(d[k], float) for k in ('const', 'sigma', 'state'))
        self.n_obs, self.loglik, self.n_params, self.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']


class BlockVECM(_Block):
    """
    Vector error-correction model on a block observed in levels.

    Parameters
    ----------
    rank : int or 'auto'
        Cointegration rank; ``'auto'`` uses the Johansen trace test at ``signif``.
    lags : int or 'auto'
        Number ``q`` of lagged differences; ``'auto'`` picks the BIC choice in ``1..max_lags``.
    deterministic : 'co' (constant outside the cointegration relation) or 'n' (none).
    """

    kind = 'vecm'

    def __init__(self, rank='auto', lags='auto', deterministic='co', max_lags=4, signif=0.05):
        super().__init__()
        if deterministic not in ('co', 'n'):
            raise ValueError("deterministic must be 'co' or 'n'")
        self.rank_spec, self.lags_spec = rank, lags
        self.deterministic, self.max_lags, self.signif = deterministic, int(max_lags), float(signif)
        self.rank = self.lags = None
        self.alpha = self.beta = None

    @property
    def spec(self):
        return {'rank': self.rank_spec, 'lags': self.lags_spec, 'deterministic': self.deterministic,
                'max_lags': self.max_lags, 'signif': self.signif}

    @property
    def name(self):
        return 'VECM' if self.rank is None else f'VECM(r={self.rank}, q={self.lags})'

    def clone(self):
        return BlockVECM(**self.spec)

    def fit(self, X):
        vecm, _ = _sm()
        X = pd.DataFrame(X).astype(float)
        self.columns = list(X.columns)
        V = X.values
        q = self.lags_spec
        if q == 'auto':
            q = max(1, int(vecm.select_order(V, maxlags=self.max_lags, deterministic=self.deterministic).bic))
        q = int(q)
        r = self.rank_spec
        if r == 'auto':
            r = int(vecm.select_coint_rank(V, det_order=0, k_ar_diff=q, method='trace', signif=self.signif).rank)
        r = int(r)
        res = vecm.VECM(V, k_ar_diff=q, coint_rank=r, deterministic=self.deterministic).fit()
        self.rank, self.lags = r, q
        self.alpha = np.asarray(res.alpha, float).reshape(self.p, r)
        self.beta = np.asarray(res.beta, float).reshape(self.p, r)
        self.Gamma = [np.asarray(res.gamma[:, i * self.p:(i + 1) * self.p], float) for i in range(q)]
        self.const = np.asarray(res.det_coef, float).ravel() if self.deterministic == 'co' else np.zeros(self.p)
        E = np.asarray(res.resid, float)
        self.sigma = E.std(axis=0, ddof=1)
        self._z = pd.DataFrame(E / self.sigma, index=X.index[q + 1:], columns=self.columns)
        self.state = V[-(q + 1):].copy()                                    # last q + 1 levels, most recent last
        self.n_obs = int(res.nobs)
        self.loglik = float(res.llf)
        # loadings and cointegrating vectors (minus the r^2 normalisations), short-run matrices, constants, variances
        self.n_params = int(2 * self.p * r - r * r + self.p * self.p * q + (self.p if self.deterministic == 'co' else 0) + self.p)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    # ---- recursion ---------------------------------------------------------
    def _step(self, levels, z):
        """Next levels from ``levels`` (n_paths, q + 1, p), most recent last, and innovations ``z`` (n_paths, p)."""
        x_prev = levels[:, -1]
        d = np.diff(levels, axis=1)                                          # d[:, -1] is Delta x_{t-1}
        Pi = self.alpha @ self.beta.T
        dx = self.const + x_prev @ Pi.T + sum(d[:, -(i + 1)] @ self.Gamma[i].T for i in range(self.lags)) + z * self.sigma
        return x_prev + dx

    def unfilter(self, Z):
        """Levels from innovation paths ``(n_paths, horizon, p)``, from the last observed rows."""
        Z = np.asarray(Z, float)
        n, H, p = Z.shape
        levels = np.tile(self.state, (n, 1, 1))
        out = np.empty_like(Z)
        for t in range(H):
            x = self._step(levels, Z[:, t])
            out[:, t] = x
            levels = np.concatenate([levels[:, 1:], x[:, None, :]], axis=1)
        return out

    def filter_new(self, X_new):
        """Standardized innovations of new rows of levels ``(n, p)``, continuing from the last state."""
        V = np.asarray(X_new, float).reshape(-1, self.p)
        levels = self.state[None].copy()
        z = np.empty_like(V)
        for t in range(len(V)):
            pred = self._step(levels, np.zeros((1, self.p)))[0]
            z[t] = (V[t] - pred) / self.sigma
            levels = np.concatenate([levels[:, 1:], V[t][None, None, :]], axis=1)
        return z

    # ---- persistence -------------------------------------------------------
    def to_dict(self):
        d = self._base_dict()
        d.update({'rank': self.rank, 'alpha': self.alpha.tolist(), 'beta': self.beta.tolist()})
        return d

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m._load_base(d)
        m.rank = int(d['rank'])
        m.alpha, m.beta = np.array(d['alpha'], float).reshape(m.p, m.rank), np.array(d['beta'], float).reshape(m.p, m.rank)
        return m


class BlockVAR(_Block):
    """Vector autoregression on a block, in the units of the series given (levels or returns)."""

    kind = 'var'

    def __init__(self, lags='auto', max_lags=4):
        super().__init__()
        self.lags_spec, self.max_lags = lags, int(max_lags)
        self.lags = None

    @property
    def spec(self):
        return {'lags': self.lags_spec, 'max_lags': self.max_lags}

    @property
    def name(self):
        return 'VAR' if self.lags is None else f'VAR({self.lags})'

    def clone(self):
        return BlockVAR(**self.spec)

    def fit(self, X):
        _, VAR = _sm()
        X = pd.DataFrame(X).astype(float)
        self.columns = list(X.columns)
        V = X.values
        q = self.lags_spec
        if q == 'auto':
            q = max(1, int(VAR(V).select_order(self.max_lags).bic))
        q = int(q)
        res = VAR(V).fit(q, trend='c')
        self.lags = q
        self.Gamma = [np.asarray(res.coefs[i], float) for i in range(q)]
        self.const = np.asarray(res.intercept, float).ravel()
        E = np.asarray(res.resid, float)
        self.sigma = E.std(axis=0, ddof=1)
        self._z = pd.DataFrame(E / self.sigma, index=X.index[q:], columns=self.columns)
        self.state = V[-q:].copy()
        self.n_obs = int(res.nobs)
        self.loglik = float(res.llf)
        self.n_params = int(self.p * self.p * q + self.p + self.p)
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    def _step(self, hist, z):
        """Next values from ``hist`` (n_paths, q, p), most recent last."""
        return self.const + sum(hist[:, -(i + 1)] @ self.Gamma[i].T for i in range(self.lags)) + z * self.sigma

    def unfilter(self, Z):
        Z = np.asarray(Z, float)
        n, H, p = Z.shape
        hist = np.tile(self.state, (n, 1, 1))
        out = np.empty_like(Z)
        for t in range(H):
            x = self._step(hist, Z[:, t])
            out[:, t] = x
            hist = np.concatenate([hist[:, 1:], x[:, None, :]], axis=1)
        return out

    def filter_new(self, X_new):
        V = np.asarray(X_new, float).reshape(-1, self.p)
        hist = self.state[None].copy()
        z = np.empty_like(V)
        for t in range(len(V)):
            z[t] = (V[t] - self._step(hist, np.zeros((1, self.p)))[0]) / self.sigma
            hist = np.concatenate([hist[:, 1:], V[t][None, None, :]], axis=1)
        return z

    def to_dict(self):
        return self._base_dict()

    @classmethod
    def from_dict(cls, d):
        m = cls(**d['spec'])
        m._load_base(d)
        return m


def block_from_dict(d):
    """Block model from its ``to_dict`` output."""
    return {'vecm': BlockVECM, 'var': BlockVAR}[d['kind']].from_dict(d)
