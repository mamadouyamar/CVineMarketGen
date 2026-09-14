# -*- coding: utf-8 -*-
"""
Gaussian hidden Markov model of one asset, as a dynamics model: the residual
layer is the normal score of the Rosenblatt uniform ``v_t = F_t(y_t)``, where
``F_t`` is the one-step-ahead predictive mixture. Compact reimplementation of
the Gaussian case of GenHMM1d (Nasri, Rémillard and Thioub); the initial
regime distribution is uniform, as there.
"""
import numpy as np
import pandas as pd
from scipy import stats


class GaussianHMM:
    """
    ``y_t | s_t = k ~ N(mu_k, sigma_k^2)``, ``s_t`` a Markov chain with transition matrix ``Q``.

    Parameters
    ----------
    n_states : int
    n_init : int
        EM starts: one from quantile splits, the others random perturbations of it.
    max_iter, tol : EM stopping rule on the log-likelihood increase.

    Attributes after ``fit``: ``mu``, ``sigma`` (sorted by ``mu``), ``Q``, ``eta_T``
    (filtered probabilities at the last observation), ``regimes`` (filtered
    probabilities of the sample), ``loglik``, ``n_params = K(K-1) + 2K``, ``bic``.
    """

    kind = 'hmm'
    _short_iter = 50           # EM iterations spent on each start before the full run from the best

    def __init__(self, n_states=2, n_init=10, max_iter=2000, tol=1e-10, seed=0):
        self.K = int(n_states)
        self.n_init, self.max_iter, self.tol, self.seed = int(n_init), int(max_iter), float(tol), seed
        self.mu = self.sigma = self.Q = None
        self.eta_T = None
        self._u = self._eta = None
        self._init = None                                                   # warm start (mu, sigma, Q), see refit
        self.n_obs = self.loglik = self.n_params = self.bic = None

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
        return GaussianHMM(self.K, self.n_init, self.max_iter, self.tol, self.seed)

    def refit(self, y):
        """
        Fit a fresh copy on ``y`` starting from this model's parameters (plus two
        perturbed starts): what the parametric bootstrap needs, at a fraction of
        the cost of the full multi-start.
        """
        m = self.clone()
        m.n_init = 3
        m._init = (self.mu.copy(), self.sigma.copy(), self.Q.copy())
        return m.fit(y)

    # ---- likelihood pieces -----------------------------------------------------
    def _dens(self, v):
        return stats.norm.pdf(v[:, None], self.mu[None, :], self.sigma[None, :])

    def _forward(self, f):
        """Filtered ``eta`` (n, K), predictive ``W`` (n, K), scaling ``c`` (n,)."""
        n, K = f.shape
        eta = np.empty((n, K)); W = np.empty((n, K)); c = np.empty(n)
        prev = np.full(K, 1.0 / K)
        Q = self.Q
        for t in range(n):
            w = prev @ Q
            v = w * f[t]
            c[t] = v.sum()
            eta[t] = v / c[t]
            W[t] = w
            prev = eta[t]
        return eta, W, c

    def _backward(self, f, c):
        n, K = f.shape
        beta = np.empty((n, K)); beta[-1] = 1.0
        Q = self.Q
        for t in range(n - 2, -1, -1):
            beta[t] = (Q @ (f[t + 1] * beta[t + 1])) / c[t + 1]
        return beta

    def _em(self, v, mu, sigma, Q, max_iter=None):
        self.mu, self.sigma, self.Q = mu.copy(), sigma.copy(), Q.copy()
        ll_old = -np.inf
        for _ in range(self.max_iter if max_iter is None else max_iter):
            f = self._dens(v)
            eta, W, c = self._forward(f)
            ll = float(np.log(c).sum())
            beta = self._backward(f, c)
            g = eta * beta; g /= g.sum(1, keepdims=True)                    # smoothed probabilities
            xi = self.Q * (eta[:-1].T @ (f[1:] * beta[1:] / c[1:, None]))   # expected transition counts
            self.Q = xi / xi.sum(1, keepdims=True)
            wsum = g.sum(0)
            self.mu = (g * v[:, None]).sum(0) / wsum
            self.sigma = np.sqrt((g * (v[:, None] - self.mu) ** 2).sum(0) / wsum)
            if ll - ll_old < self.tol:
                break
            ll_old = ll
        f = self._dens(v)
        _, _, c = self._forward(f)
        return float(np.log(c).sum())

    def _init_params(self, v, rng, perturb):
        if self._init is not None:
            mu, sigma, Q = (np.array(x, float) for x in self._init)
        else:
            qs = np.quantile(v, np.linspace(0, 1, self.K + 1))
            mu = np.array([v[(v >= qs[k]) & (v <= qs[k + 1])].mean() for k in range(self.K)])
            sigma = np.array([v[(v >= qs[k]) & (v <= qs[k + 1])].std() for k in range(self.K)])
            Q = np.full((self.K, self.K), 0.1 / max(self.K - 1, 1)); np.fill_diagonal(Q, 0.9)
        if perturb:
            mu = mu + rng.standard_normal(self.K) * v.std() * 0.5
            sigma = sigma * np.exp(rng.standard_normal(self.K) * 0.3)
        return mu, np.maximum(sigma, 1e-8), Q

    # ---- fit -----------------------------------------------------------------
    def fit(self, y):
        s = pd.Series(y).astype(float)
        v = s.values
        rng = np.random.default_rng(self.seed)
        best = (-np.inf, None)
        for i in range(self.n_init):                                        # short EM from every start ...
            mu, sigma, Q = self._init_params(v, rng, perturb=i > 0)
            with np.errstate(all='raise'):
                try:
                    ll = self._em(v, mu, sigma, Q, max_iter=min(self.max_iter, self._short_iter))
                except FloatingPointError:
                    continue
            if ll > best[0]:
                best = (ll, (self.mu.copy(), self.sigma.copy(), self.Q.copy()))
        if best[1] is None:
            raise RuntimeError('HMM estimation failed from every start')
        self.loglik = self._em(v, *best[1])                                 # ... full EM from the best one
        order = np.argsort(self.mu)                                          # regimes sorted by mean, as GenHMM1d
        self.mu, self.sigma, self.Q = self.mu[order], self.sigma[order], self.Q[np.ix_(order, order)]
        f = self._dens(v)
        eta, W, c = self._forward(f)
        cdf = stats.norm.cdf(v[:, None], self.mu[None, :], self.sigma[None, :])
        u = np.clip((W * cdf).sum(1), 1e-12, 1 - 1e-12)
        self._u = pd.Series(u, index=s.index, name=s.name)
        self._eta = pd.DataFrame(eta, index=s.index, columns=[f'state {k + 1}' for k in range(self.K)])
        self.eta_T = eta[-1].copy()
        self.n_obs = len(v)
        self.n_params = self.K * (self.K - 1) + 2 * self.K
        self.bic = float(-2 * self.loglik + self.n_params * np.log(self.n_obs))
        return self

    # ---- residual layer ------------------------------------------------------
    def uniforms(self):
        """Rosenblatt uniforms ``v_t`` of the fitted sample."""
        return self._u

    def filter(self):
        """Normal scores of the Rosenblatt uniforms (i.i.d. N(0,1) under the model)."""
        return pd.Series(stats.norm.ppf(self._u.values), index=self._u.index, name=self._u.name)

    @property
    def regimes(self):
        """Filtered regime probabilities of the fitted sample."""
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
        """One simulated series of length ``n`` from the last state (used by the bootstrap test)."""
        z = np.random.default_rng(seed).standard_normal((1, int(n)))
        return self.unfilter(z)[0]

    # ---- persistence ---------------------------------------------------------
    def to_dict(self):
        return {'kind': self.kind, 'spec': self.spec, 'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                'Q': self.Q.tolist(), 'eta_T': self.eta_T.tolist(), 'n_obs': self.n_obs,
                'loglik': self.loglik, 'n_params': self.n_params, 'bic': self.bic}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['spec']['n_states'])
        m.mu, m.sigma, m.Q, m.eta_T = (np.array(d[k], float) for k in ('mu', 'sigma', 'Q', 'eta_T'))
        m.n_obs, m.loglik, m.n_params, m.bic = d['n_obs'], d['loglik'], d['n_params'], d['bic']
        return m
