# -*- coding: utf-8 -*-
"""
Fleishman + Vale-Maurelli generator (article 3, Section 3), the benchmark
without tail dependence. Same logic as OldVersion/fleishman_comparison.py:

  fit      : Fleishman coefficients (a, b, c, d) per asset from the standardized
             targets (0, 1, skew, excess kurtosis); Vale-Maurelli intermediate
             correlations rho_Z for every pair; nearest positive-definite
             projection of the intermediate matrix.
  simulate : Z ~ N(0, Sigma_Z), Y_i = mu_i + sigma_i (a_i + b_i Z_i + c_i Z_i^2 + d_i Z_i^3),
             with the accept-reject loop of Algorithm 5: coefficients re-fitted on
             the specific draw, draw kept only if it meets the tolerances.
"""
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import skew, kurtosis

from .moment_match import MomentMatch


class FleishmanGenerator(MomentMatch):
    """Fleishman cubic transform with Vale-Maurelli correlation matching."""

    TOL_MEAN_VOL = 2e-4
    TOL_SKEW_KURT = 5e-2

    # ------------------------------------------------------------------
    def _theoretical_residuals(self, p, target):
        """Implied [mean, var, skew, exkurt] of a + bZ + cZ^2 + dZ^3 minus target."""
        return self.moments_cubic_transform(p, distr='gauss') - target

    def _empirical_residuals(self, p, z, target):
        """Empirical moments of a + bz + cz^2 + dz^3 on the draw z, minus target."""
        y = self.cubic_transform(z, p)
        return np.array([np.mean(y), np.var(y), skew(y), kurtosis(y)]) - target

    @staticmethod
    def _vale_maurelli_root(c1, c2, rho_target):
        """
        Solve rho_Y = rho_Z (b1 b2 + 3 b1 d2 + 3 d1 b2 + 9 d1 d2) + rho_Z^2 (2 c1 c2)
                      + rho_Z^3 (6 d1 d2)
        for rho_Z. Returns (root, ok); ok=False when no real root lies in [-1, 1].
        """
        _, b1, cc1, d1 = c1
        _, b2, cc2, d2 = c2
        poly = [6 * d1 * d2,
                2 * cc1 * cc2,
                b1 * b2 + 3 * b1 * d2 + 3 * d1 * b2 + 9 * d1 * d2,
                -rho_target]
        roots = np.roots(poly)
        real = [r.real for r in roots if abs(r.imag) < 1e-9 and abs(r.real) <= 1.0]
        if len(real) == 0:
            return np.nan, False
        return min(real, key=lambda r: abs(r - rho_target)), True

    # ------------------------------------------------------------------
    def fit(self, mu, sigma, skewness, exkurt, corr_target):
        """
        Steps 1 and 2 of Section 3.2: Fleishman coefficients per asset, then the
        Vale-Maurelli intermediate correlation for every pair.

        mu, sigma, skewness, exkurt : Series indexed by asset (exkurt = kurtosis - 3)
        corr_target                 : DataFrame, target correlation matrix (same index)

        Returns and stores a dict with
          'coef'            : DataFrame [a, b, c, d, residual, feasible] per asset
          'corr_Z'          : DataFrame of intermediate correlations (NaN where no real root)
          'corr_Z_pd'       : nearest positive-definite version, used for simulation
          'infeasible_pairs': list of (asset_i, asset_j, target, reason)
        """
        assets = list(mu.index)
        self.assets = assets
        self.mu = mu.loc[assets].astype(float)
        self.sigma = sigma.loc[assets].astype(float)
        self.skewness = skewness.loc[assets].astype(float)
        self.exkurt = exkurt.loc[assets].astype(float)
        self.corr_target = corr_target.loc[assets, assets].astype(float)

        coef = pd.DataFrame(index=assets, columns=['a', 'b', 'c', 'd', 'residual', 'feasible'], dtype=object)
        for a in assets:
            target = np.array([0.0, 1.0, self.skewness[a], self.exkurt[a]])
            try:
                p0, _ = self.find_params_for_moments_matching(target, x0=[0.0, 1.0, 0.0, 0.0],
                                                              method='Nelder-Mead', distr='gauss')
                p, _, ier, _ = fsolve(self._theoretical_residuals, p0, args=(target,), full_output=True)
                res = np.max(np.abs(self._theoretical_residuals(p, target)))
                ok = (ier == 1) and (res < 1e-8) and (p[1] > 0)
                coef.loc[a, ['a', 'b', 'c', 'd']] = p
                coef.loc[a, 'residual'] = res
                coef.loc[a, 'feasible'] = bool(ok)
            except Exception as e:
                coef.loc[a, ['a', 'b', 'c', 'd']] = np.nan
                coef.loc[a, 'residual'] = np.nan
                coef.loc[a, 'feasible'] = False
                print(f'Fleishman infeasible for {a}: {e}')

        N = len(assets)
        corr_Z = pd.DataFrame(np.eye(N), index=assets, columns=assets)
        infeasible = []
        for i in range(N):
            for j in range(i + 1, N):
                ai, aj = assets[i], assets[j]
                if not (coef.loc[ai, 'feasible'] and coef.loc[aj, 'feasible']):
                    corr_Z.iloc[i, j] = corr_Z.iloc[j, i] = np.nan
                    infeasible.append((ai, aj, self.corr_target.iloc[i, j], 'univariate step failed'))
                    continue
                r, ok = self._vale_maurelli_root(coef.loc[ai, ['a', 'b', 'c', 'd']].astype(float).values,
                                                 coef.loc[aj, ['a', 'b', 'c', 'd']].astype(float).values,
                                                 self.corr_target.iloc[i, j])
                corr_Z.iloc[i, j] = corr_Z.iloc[j, i] = r
                if not ok:
                    infeasible.append((ai, aj, self.corr_target.iloc[i, j], 'no real root in [-1, 1]'))

        self.coef = coef
        self.corr_Z = corr_Z
        self.corr_Z_pd = self.make_positive_definite(corr_Z.fillna(0.0))
        self.infeasible_pairs = infeasible
        self.fit_result = {'coef': coef, 'corr_Z': corr_Z, 'corr_Z_pd': self.corr_Z_pd,
                           'infeasible_pairs': infeasible}
        return self.fit_result

    # ------------------------------------------------------------------
    def simulate(self, n, corr_tol=2e-2, seed=None, max_tries=500, verbose=True):
        """
        Step 3 of Section 3.2 with the accept-reject loop of Algorithm 5.

        Returns (returns DataFrame n x N, errors dict, coef_refit DataFrame), where
        errors holds the max absolute errors on mean, vol, skew, kurt and correlation
        of the accepted draw.
        """
        if not hasattr(self, 'coef'):
            raise RuntimeError('call fit(...) first')
        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(seed)
        assets = self.assets
        N = len(assets)
        L = np.linalg.cholesky(self.corr_Z_pd.values)
        coef_post = self.coef[['a', 'b', 'c', 'd']].astype(float).copy()

        accepted, n_tries = False, 0
        while not accepted and n_tries < max_tries:
            n_tries += 1
            Z = rng.standard_normal((n, N)) @ L.T
            sim = pd.DataFrame(np.zeros((n, N)), columns=assets)
            for i, a in enumerate(assets):
                target = np.array([0.0, 1.0, self.skewness[a], self.exkurt[a]])
                p0 = self.coef.loc[a, ['a', 'b', 'c', 'd']].astype(float).values
                p, _, ier, _ = fsolve(self._empirical_residuals, p0, args=(Z[:, i], target), full_output=True)
                if ier != 1:
                    p = p0
                coef_post.loc[a] = p
                sim[a] = self.mu[a] + self.sigma[a] * self.cubic_transform(Z[:, i], p)

            errors = {
                'mean': np.max(np.abs(sim.mean(0).values - self.mu.values)),
                'vol': np.max(np.abs(sim.std(0).values - self.sigma.values)),
                'skew': np.max(np.abs(skew(sim) - self.skewness.values)),
                'kurt': np.max(np.abs(kurtosis(sim) - self.exkurt.values)),
                'corr': np.max(np.abs(np.corrcoef(sim.T.values) - self.corr_target.values)),
            }
            accepted = (max(errors['mean'], errors['vol']) <= self.TOL_MEAN_VOL) and \
                       (max(errors['skew'], errors['kurt']) <= self.TOL_SKEW_KURT) and \
                       (errors['corr'] <= corr_tol)
            if verbose:
                print(f"try {n_tries}: err mean/vol={max(errors['mean'], errors['vol']):.2e}  "
                      f"skew/kurt={max(errors['skew'], errors['kurt']):.3f}  corr={errors['corr']:.4f}  "
                      f"accepted={accepted}")
        if not accepted:
            raise RuntimeError(f'no draw accepted after {max_tries} tries')
        self.coef_refit = coef_post
        return sim, errors, coef_post
