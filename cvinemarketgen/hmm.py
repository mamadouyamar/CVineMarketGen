# -*- coding: utf-8 -*-
"""
Gaussian hidden Markov model of one asset as a dynamics model, estimated by
GenHMM1d (Nasri, Rémillard and Thioub; ``pip install
git+https://github.com/mamadouyamar/GenHMM1d.git``). The residual layer is the
normal score of the Rosenblatt uniform ``v_t = F_t(y_t)``, ``F_t`` the
one-step-ahead predictive mixture, as returned by GenHMM1d; path simulation
inverts that map exactly from the last filtered regime probabilities.
"""
import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats


def _genhmm1d():
    try:
        from genhmm1d.hmm import HMM
    except ImportError as e:
        raise ImportError('HMM dynamics need the GenHMM1d package: '
                          'pip install git+https://github.com/mamadouyamar/GenHMM1d.git') from e
    return HMM()


class GaussianHMM:
    """
    ``y_t | s_t = k ~ N(mu_k, sigma_k^2)``, ``s_t`` a Markov chain with transition
    matrix ``Q`` and a uniform initial distribution, estimated by GenHMM1d's
    ``EstHMMGen`` (EM, regimes sorted by mean).

    Parameters
    ----------
    n_states : int
    max_iter, eps, ninit : EM settings passed to GenHMM1d (``ninit`` EM steps are
        always run, then up to ``max_iter`` until the relative parameter change is below ``eps``).

    Attributes after ``fit``: ``mu``, ``sigma``, ``Q``, ``eta_T`` (filtered
    probabilities at the last observation), ``regimes`` (filtered probabilities of
    the sample), ``cvm`` (Cramér-von Mises statistic of the uniforms), ``loglik``,
    ``n_params = K(K-1) + 2K``, ``bic = -2 loglik + n_params log n`` (on the data's
    scale, comparable with the GARCH family).
    """

    kind = 'hmm'

    def __init__(self, n_states=2, max_iter=10000, eps=1e-6, ninit=50, seed=0):
        self.K = int(n_states)
        self.max_iter, self.eps, self.ninit, self.seed = int(max_iter), float(eps), int(ninit), seed
        self.mu = self.sigma = self.Q = None
        self.eta_T = None
        self._u = self._eta = None
        self.cvm = self.n_obs = self.loglik = self.n_params = self.bic = None

    @property
    def spec(self):
        return {'n_states': self.K}

    @property
    def name(self):
        return f'HMM({self.K})'

    def __repr__(self):
        return f'GaussianHMM({self.name}{"" if self.mu is None else ", fitted"})'

    def clone(self):
        """Unfitted copy with the same specification."""
        return GaussianHMM(self.K, self.max_iter, self.eps, self.ninit, self.seed)

    # ---- fit -----------------------------------------------------------------
    def fit(self, y, init=None):
        """
        Estimate on ``y`` with GenHMM1d. ``init``, optional ``(theta, Q)`` with
        ``theta`` of shape ``(K, 2)`` = ``(mu, sigma)`` per regime, replaces the
        quantile-split starting values (used by :meth:`refit`).
        """
        s = pd.Series(y).astype(float)
        v = s.values.reshape(-1, 1)
        kw = {}
        if init is None:
            kw['percentiles'] = list(np.round(np.linspace(0, 100, self.K + 1)[1:-1], 6))
        else:
            kw['initial_theta'], kw['initial_Q'] = np.asarray(init[0], float), np.asarray(init[1], float)
        with contextlib.redirect_stdout(io.StringIO()):
            out = _genhmm1d().EstHMMGen(y=v, reg=self.K, family='norm', max_iter=self.max_iter,
                                        ninit=self.ninit, eps=self.eps, **kw)
        theta = np.asarray(out['theta'], float)
        self.mu, self.sigma, self.Q = theta[:, 0].copy(), theta[:, 1].copy(), np.asarray(out['Q'], float)
        eta = np.asarray(out['eta_EM'], float)
        self._eta = pd.DataFrame(eta, index=s.index, columns=[f'state {k + 1}' for k in range(self.K)])
        self.eta_T = eta[-1].copy()
        self._u = pd.Series(np.clip(np.ravel(out['U']), 1e-12, 1 - 1e-12), index=s.index, name=s.name)
        self.cvm = float(np.ravel(out['cvm'])[0])
        self.loglik = float(np.ravel(out['LL'])[0])
        self.n_obs = len(v)
        self.n_params = self.K * (self.K - 1) + 2 * self.K
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    def refit(self, y):
        """Fit a fresh copy on ``y`` starting from this model's parameters (bootstrap refit)."""
        return self.clone().fit(y, init=(np.column_stack([self.mu, self.sigma]), self.Q))

    # ---- residual layer ------------------------------------------------------
    def uniforms(self):
        """Rosenblatt uniforms ``v_t`` of the fitted sample (GenHMM1d's ``U``)."""
        return self._u

    def filter(self):
        """Normal scores of the Rosenblatt uniforms (i.i.d. N(0,1) under the model)."""
        return pd.Series(stats.norm.ppf(self._u.values), index=self._u.index, name=self._u.name)

    @property
    def regimes(self):
        """Filtered regime probabilities of the fitted sample (GenHMM1d's ``eta_EM``)."""
        return self._eta

    def _mixture_cdf(self, y, W):
        return (W * stats.norm.cdf(y[:, None], self.mu[None, :], self.sigma[None, :])).sum(1)

    def _mixture_inv(self, u, W):
        lo = np.full(len(u), self.mu.min() - 12 * self.sigma.max())
        hi = np.full(len(u), self.mu.max() + 12 * self.sigma.max())
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            below = self._mixture_cdf(mid, W) < u
            lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    def _step_eta(self, eta, y):
        f = stats.norm.pdf(y[:, None], self.mu[None, :], self.sigma[None, :])
        v = (eta @ self.Q) * f
        return v / v.sum(1, keepdims=True)

    def unfilter(self, z):
        """Returns from normal-score paths ``(n_paths, horizon)``: the exact inverse of the filter from ``eta_T``."""
        z = np.asarray(z, float)
        n_paths, horizon = z.shape
        eta = np.tile(self.eta_T, (n_paths, 1))
        out = np.empty_like(z)
        for t in range(horizon):
            W = eta @ self.Q
            y = self._mixture_inv(stats.norm.cdf(z[:, t]), W)
            out[:, t] = y
            eta = self._step_eta(eta, y)
        return out

    def filter_new(self, y_new):
        """Normal scores of new observations, continuing the filter from ``eta_T``."""
        v = np.asarray(y_new, float)
        eta = self.eta_T[None, :].copy()
        z = np.empty(len(v))
        for t in range(len(v)):
            W = eta @ self.Q
            u = np.clip(self._mixture_cdf(v[t:t + 1], W), 1e-12, 1 - 1e-12)
            z[t] = stats.norm.ppf(u[0])
            eta = self._step_eta(eta, v[t:t + 1])
        return z

    def simulate(self, n, seed=None):
        """One series of length ``n`` drawn from the fitted chain (GenHMM1d's ``SimHMMGen``), for the parametric bootstrap."""
        if seed is not None:
            np.random.seed(seed)
        sim, _, _ = _genhmm1d().SimHMMGen(self.Q, 'norm', np.column_stack([self.mu, self.sigma]), int(n))
        return np.ravel(np.asarray(sim, float))

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                'Q': self.Q.tolist(), 'eta_T': self.eta_T.tolist(), 'n_obs': self.n_obs, 'cvm': self.cvm,
                'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['spec']['n_states'])
        m.mu, m.sigma, m.Q, m.eta_T = (np.array(d[k], float) for k in ('mu', 'sigma', 'Q', 'eta_T'))
        m.n_obs, m.loglik, m.n_params, m.bic, m.cvm = d['n_obs'], d['loglik'], d['n_params'], d['bic'], d.get('cvm')
        return m
