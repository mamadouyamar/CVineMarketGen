# -*- coding: utf-8 -*-
"""
Bivariate copula tools: exceedance correlation, non-central squared (NCS)
copulas, mixture copulas, h-functions and inverses, tail-dependence catalogs
and BIC selection. Relocated verbatim from momentmatchingscript.py.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, skew, kurtosis, bernoulli
from scipy.optimize import minimize, fsolve, brentq
from itertools import product
import pyvinecopulib as pv
from .moment_match import bicop_params, bicop_bounds


class CopulaTools:
    """
    Bivariate building blocks of the C-vine (Section 4.4 and Appendix C of the paper).

    Exceedance-correlation classification of a pair, the catalogs of candidate
    families and mixtures with their tail signatures, BIC selection among the
    admissible families, two-component mixture copulas (density, ``h``-function,
    inverse ``h``-function, maximum likelihood), and the non-central squared (NCS)
    copulas of Nasri (2020).

    Single-family copulas are :class:`pyvinecopulib.Bicop` objects; a mixture is a
    list of two ``Bicop`` components with a parameter vector
    ``[w, theta_1, theta_2]`` where ``w`` is the weight on the first component.
    """

    def exceedance_correlation(self, x, y, z_grid, min_obs=10):
        """
        Empirical exceedance correlation of (x, y) conditional on x only (kind=1
        of ConditionalCorrelation): both series are standardized and the Pearson
        correlation is computed on the subsample x < z for z < 0 and x >= z for
        z >= 0. NaN where the subsample has fewer than min_obs points.
        """
        x = np.asarray(x, float); y = np.asarray(y, float)
        x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
        out = np.full(len(z_grid), np.nan)
        for i, z in enumerate(z_grid):
            cond = (x >= z) if z >= 0 else (x < z)
            if cond.sum() >= min_obs:
                out[i] = np.corrcoef(x[cond], y[cond])[0, 1]
        return out


    def draw_cond_corr(self, x, y, thetas_lb=-2, thetas_ub=2, figsize=(5, 5), names=['Asset1', 'Asset2'],
                       LTCMA_corr=np.nan, return_values=False, inputs_are_obs=True, show_plot=True):

        """
        Exceedance-correlation curve of a pair on a grid of thresholds (equation C.1), optionally plotted.

        Parameters
        ----------
        x, y : array_like
            Raw returns when ``inputs_are_obs`` is True; conditional pseudo-observations
            (``h``-function values) otherwise, in which case ``Phi^{-1}`` is applied first.
        thetas_lb, thetas_ub : float, default -2, 2
            Threshold range in standard deviations; the grid step is 0.02.
        figsize, names, LTCMA_corr, show_plot
            Plot options.
        return_values : bool, default False
            If True, return the curve instead of only plotting it.
        inputs_are_obs : bool, default True

        Returns
        -------
        pandas.Series
            Exceedance correlation indexed by threshold (only when ``return_values`` is True).
        """
        if inputs_are_obs == False:
            x = norm.ppf(x)
            y = norm.ppf(y)

        Sigma = self.corr2cov_biv(np.corrcoef(np.array([x, y]))[0, 1], np.array([np.std(x), np.std(y)]))
        thetas = np.arange(thetas_lb, thetas_ub, 0.02)
        corr_emp = np.zeros((len(thetas), 1))
        corr_emp_y_x = np.zeros((len(thetas), 1))
        df_condcorr = pd.DataFrame([], columns=['Emp'])
        for i, theta in enumerate(thetas):
            corr_emp[i, 0], _ = self.ConditionalCorrelation(x, y, Sigma, theta=theta, kind=1)
            corr_emp_y_x[i, 0], _ = self.ConditionalCorrelation(y, x, Sigma, theta=theta, kind=1)

        if show_plot:
            pearson_corr = np.round(np.corrcoef([x, y])[0, 1], 3)
            ltcma_corr = np.round(LTCMA_corr, 2)
            title = (
                f"Cond corr ({names[0]} and {names[1]} | {names[0]})\n"
                f"Pearson correlation: {pearson_corr}\n"
                f"LTCMA corr: {ltcma_corr}"
            )
            plt.figure(figsize=figsize)
            plt.plot(thetas, corr_emp[:, 0], label='Empirical')
            plt.plot(thetas, corr_emp_y_x[:, 0], label='Empirical y_x')
            plt.legend()
            plt.xlabel('Number of standard deviation from the mean of the first serie')
            plt.title(title, wrap=True)
            # plt.title('Cond corr (' + names[0] + ' and ' + names[1] + ' | ' + names[0] + \
            #           ') with Pearson correlation of ' + \
            #           str(np.round(np.corrcoef(np.array([x, y]))[0, 1], 3)) + \
            #           ' LTCMA corr : ' + str(np.round(LTCMA_corr, 2))
            #           )
            plt.show()

        if return_values:
            return pd.Series(corr_emp[:, 0], index=thetas)

    def draw_cond_corr_V2(self, x1, y1, x2, y2, figsize=(5, 5), thetas_lb=-2, thetas_ub=2, LTCMAS_corr=np.nan,
                          names=['Asset_1_1', 'Asset_1_2'], code_famille=1, ncs_status=False, show_plot_yx=False):
        if isinstance(code_famille, list):
            class code_famille:
                pass
            code_famille.name = 'mixture'
        Sigma1 = self.corr2cov_biv(np.corrcoef(np.array([x1, y1]))[0, 1], np.array([np.std(x1), np.std(y1)]))
        Sigma2 = self.corr2cov_biv(np.corrcoef(np.array([x2, y2]))[0, 1], np.array([np.std(x2), np.std(y2)]))
        thetas = np.arange(thetas_lb, thetas_ub, 0.02)
        corr_emp_1 = np.zeros((len(thetas), 1))
        corr_emp_2 = np.zeros((len(thetas), 1))
        corr_emp_1_y_x = np.zeros((len(thetas), 1))
        corr_emp_2_y_x = np.zeros((len(thetas), 1))
        df_condcorr = pd.DataFrame([], columns=['Emp'])
        for i, theta in enumerate(thetas):
            corr_emp_1[i, 0], _ = self.ConditionalCorrelation(x1, y1, Sigma1,
                                                              theta=theta,
                                                              kind=1)
            corr_emp_2[i, 0], _ = self.ConditionalCorrelation(x2, y2, Sigma2,
                                                              theta=theta,
                                                              kind=1)

            corr_emp_1_y_x[i, 0], _ = self.ConditionalCorrelation(y1, x1, Sigma1,
                                                              theta=theta,
                                                              kind=1)
            corr_emp_2_y_x[i, 0], _ = self.ConditionalCorrelation(y2, x2, Sigma2,
                                                              theta=theta,
                                                              kind=1)

        plt.figure(figsize=figsize)
        plt.plot(thetas, corr_emp_1[:, 0], label='Simulated series')
        plt.plot(thetas, corr_emp_2[:, 0], label='Historical series')
        if show_plot_yx:
            plt.plot(thetas, corr_emp_1_y_x[:, 0], label='Simulated series y_x')
            plt.plot(thetas, corr_emp_2_y_x[:, 0], label='Historical series y_x')
        plt.legend()
        plt.xlabel('Number of standard deviation from the mean of the first serie')
        plt.title(
            f"Cond corr ({names[0]} and {names[1]} | {names[0]})\n"
            f"Corr simulated data : ({np.round(np.corrcoef([x1, y1])[0, 1], 2)}) ; "
            f"historical data : ({np.round(np.corrcoef([x2, y2])[0, 1], 2)})\n"
            f"LTCMAs : ({np.round(LTCMAS_corr, 2)}) ; copula : {code_famille.name} ; ncs : {ncs_status}"
        )
        plt.show()

    def draw_cond_corr_V3(self, x1, y1, x2, y2,
                          figsize=(5, 5),
                          thetas_lb=-2,
                          thetas_ub=2,
                          LTCMAS_corr=np.nan,
                          names=['Asset_1_1', 'Asset_1_2'],
                          code_famille=1):

        if isinstance(code_famille, list):
            class code_famille:
                pass
            code_famille.name = 'mixture'
        Sigma1 = self.corr2cov_biv(np.corrcoef(np.array([x1, y1]))[0, 1], np.array([np.std(x1), np.std(y1)]))
        Sigma2 = self.corr2cov_biv(np.corrcoef(np.array([x2, y2]))[0, 1], np.array([np.std(x2), np.std(y2)]))
        thetas = np.arange(thetas_lb, thetas_ub, 0.02)
        corr_emp_1 = np.zeros((len(thetas), 1))
        corr_emp_2 = np.zeros((len(thetas), 1))
        corr_emp_1_y_x = np.zeros((len(thetas), 1))
        corr_emp_2_y_x = np.zeros((len(thetas), 1))
        df_condcorr = pd.DataFrame([], columns=['Emp'])
        for i, theta in enumerate(thetas):
            corr_emp_1[i, 0], _ = self.ConditionalCorrelation(x1, y1, Sigma1,
                                                              theta=theta,
                                                              kind=1)
            corr_emp_2[i, 0], _ = self.ConditionalCorrelation(x2, y2, Sigma2,
                                                              theta=theta,
                                                              kind=1)

            corr_emp_1_y_x[i, 0], _ = self.ConditionalCorrelation(y1, x1, Sigma1,
                                                              theta=theta,
                                                              kind=1)
            corr_emp_2_y_x[i, 0], _ = self.ConditionalCorrelation(y2, x2, Sigma2,
                                                              theta=theta,
                                                              kind=1)


        mse_x_y = np.mean((corr_emp_1[:, 0] - corr_emp_2[:, 0]) ** 2)
        mse_y_x = np.mean((corr_emp_1_y_x[:, 0] - corr_emp_2_y_x[:, 0]) ** 2)

        if mse_x_y <= mse_y_x:
            plt.figure(figsize=figsize)
            plt.plot(thetas, corr_emp_1[:, 0], label='Simulated series')
            plt.plot(thetas, corr_emp_2[:, 0], label='Historical series')
            plt.legend()
            plt.xlabel('Number of standard deviation from the mean of '+names[0])
            plt.title(
                f"Cond corr ({names[0]} and {names[1]} | {names[0]})\n"
                f"Corr simulated data : ({np.round(np.corrcoef([x1, y1])[0, 1], 2)}) ; "
                f"historical data : ({np.round(np.corrcoef([x2, y2])[0, 1], 2)})\n"
                f"LTCMAs : ({np.round(LTCMAS_corr, 2)})"
            )
            plt.show()

        else:
            plt.figure(figsize=figsize)
            plt.plot(thetas, corr_emp_1_y_x[:, 0], label='Simulated series')
            plt.plot(thetas, corr_emp_2_y_x[:, 0], label='Historical series')
            plt.legend()
            plt.xlabel('Number of standard deviation from the mean of ' + names[1])
            plt.title(
                f"Cond corr ({names[0]} and {names[1]} | {names[1]})\n"
                f"Corr simulated data : ({np.round(np.corrcoef([x1, y1])[0, 1], 2)}) ; "
                f"historical data : ({np.round(np.corrcoef([x2, y2])[0, 1], 2)})\n"
                f"LTCMAs : ({np.round(LTCMAS_corr, 2)})"
            )
            plt.show()

    def draw_cond_corr_V4(self, x1, y1, x2, y2,
                          figsize=(5, 5),
                          thetas_lb=-2,
                          thetas_ub=2,
                          LTCMAS_corr=np.nan,
                          names=['Asset_1_1', 'Asset_1_2'],
                          code_famille=1,
                          ax=None,
                          show=True):

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        if isinstance(code_famille, list):
            class code_famille:
                pass

            code_famille.name = 'mixture'

        Sigma1 = self.corr2cov_biv(np.corrcoef(np.array([x1, y1]))[0, 1],
                                   np.array([np.std(x1), np.std(y1)]))
        Sigma2 = self.corr2cov_biv(np.corrcoef(np.array([x2, y2]))[0, 1],
                                   np.array([np.std(x2), np.std(y2)]))

        thetas = np.arange(thetas_lb, thetas_ub, 0.02)
        corr_emp_1 = np.zeros((len(thetas), 1))
        corr_emp_2 = np.zeros((len(thetas), 1))
        corr_emp_1_y_x = np.zeros((len(thetas), 1))
        corr_emp_2_y_x = np.zeros((len(thetas), 1))

        for i, theta in enumerate(thetas):
            corr_emp_1[i, 0], _ = self.ConditionalCorrelation(x1, y1, Sigma1, theta=theta, kind=1)
            corr_emp_2[i, 0], _ = self.ConditionalCorrelation(x2, y2, Sigma2, theta=theta, kind=1)

            corr_emp_1_y_x[i, 0], _ = self.ConditionalCorrelation(y1, x1, Sigma1, theta=theta, kind=1)
            corr_emp_2_y_x[i, 0], _ = self.ConditionalCorrelation(y2, x2, Sigma2, theta=theta, kind=1)

        mse_x_y = np.mean((corr_emp_1[:, 0] - corr_emp_2[:, 0]) ** 2)
        mse_y_x = np.mean((corr_emp_1_y_x[:, 0] - corr_emp_2_y_x[:, 0]) ** 2)

        # Create fig/ax only if not provided (backwards compatible)
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        corr_sim = float(np.corrcoef([x1, y1])[0, 1])
        corr_hist = float(np.corrcoef([x2, y2])[0, 1])

        if mse_x_y <= mse_y_x:
            ax.plot(thetas, corr_emp_1[:, 0], label='Simulated series')
            ax.plot(thetas, corr_emp_2[:, 0], label='Historical series')
            ax.set_xlabel('Number of standard deviation from the mean of ' + names[0])
            ax.set_title(
                f"Cond corr ({names[0]} and {names[1]} | {names[0]})\n"
                f"Corr sim: ({np.round(corr_sim, 2)}) ; hist: ({np.round(corr_hist, 2)})\n"
                f"LTCMAs: ({np.round(LTCMAS_corr, 2)})"
            )
        else:
            ax.plot(thetas, corr_emp_1_y_x[:, 0], label='Simulated series')
            ax.plot(thetas, corr_emp_2_y_x[:, 0], label='Historical series')
            ax.set_xlabel('Number of standard deviation from the mean of ' + names[1])
            ax.set_title(
                f"Cond corr ({names[0]} and {names[1]} | {names[1]})\n"
                f"Corr sim: ({np.round(corr_sim, 2)}) ; hist: ({np.round(corr_hist, 2)})\n"
                f"LTCMAs: ({np.round(LTCMAS_corr, 2)})"
            )

        # Optional: only show legend on each subplot (or you can do one global legend)
        ax.legend(fontsize=8)

        # Only show if we created the figure here, or user requested show=True
        if created_fig and show:
            plt.tight_layout()
            plt.show()

        return ax

    def _theta_from_tau(self, family, tau) -> float:
        """
        Map Kendall's tau -> base copula parameter theta (or rho for Gaussian/t).

        tau    : Kendall's tau in (0,1)
        dof    : degrees of freedom (for 't' only, not used in the mapping itself)

        """
        if not (0.0 < tau < 1.0):
            raise ValueError("tau must be in (0,1).")

        if family == pv.BicopFamily.gaussian:
            # For Gaussian and t (with any df), Kendall's tau depends only on correlation
            rho = np.sin(np.pi * tau / 2.0)
            return float(rho)

        if family == pv.BicopFamily.clayton:
            return float(2.0 * tau / (1.0 - tau))

        if family == pv.BicopFamily.gumbel:
            return float(1.0 / (1.0 - tau))

        if family == pv.BicopFamily.joe:
            def tau_of_theta(theta):
                bic = pv.Bicop(
                    family=pv.BicopFamily.joe,
                    parameters=bicop_params(theta)
                )
                return bic.tau

            # Joe copula domain: theta >= 1
            theta_min = 1.0 + 1e-6
            theta_max = 50.0  # usually plenty; Joe dependence saturates fast

            return float(
                brentq(
                    lambda th: tau_of_theta(th) - tau,
                    theta_min,
                    theta_max
                )
            )

        raise ValueError(f"Unknown family '{family}'")

    def _tau_from_theta(self, family, theta) -> float:
        """
        Map base copula parameter theta (or rho for Gaussian) -> Kendall's tau.

        theta : copula parameter
        """
        if family == pv.BicopFamily.gaussian:
            # tau = 2/pi * arcsin(rho)
            if not (-1.0 < theta < 1.0):
                raise ValueError("Gaussian rho must be in (-1,1).")
            return float(2.0 / np.pi * np.arcsin(theta))

        if family == pv.BicopFamily.clayton:
            # tau = theta / (theta + 2)
            if theta <= 0:
                raise ValueError("Clayton theta must be > 0.")
            return float(theta / (theta + 2.0))

        if family == pv.BicopFamily.gumbel:
            # tau = 1 - 1/theta
            if theta < 1:
                raise ValueError("Gumbel theta must be >= 1.")
            return float(1.0 - 1.0 / theta)

        if family == pv.BicopFamily.joe:
            # no closed form → use pv.Bicop
            if theta < 1:
                raise ValueError("Joe theta must be >= 1.")
            bic = pv.Bicop(
                family=pv.BicopFamily.joe,
                parameters=bicop_params(theta)
            )
            return float(bic.tau)

        raise ValueError(f"Unknown family '{family}'")

    def sim_ncs_copula(self, family, n, a1=0, a2=0, tau=None, theta=None, rotation=0, random_state=None):
        """
        Replicates the behavior of R's SimNCSCop(family, n, param, DoF).

        Parameters
        ----------
        n : int
            Number of observations.
        param : (a1, a2, tau)
            a1, a2 >= 0 are shift parameters for the non-central χ² margins.
            tau in (0,1) is Kendall's tau of the BASE copula (not the NCS copula).
        random_state : int, np.random.Generator, or None
            Seed or Generator for reproducibility.

        Returns
        -------
        U : ndarray, shape (n, 2)
            Simulated pseudo-observations from the NCS copula.
        """

        if tau is None and theta is None:
            raise ValueError("tau and theta cannot be both None")

        elif tau is not None and theta is not None:
            tau = None
            theta_or_rho = theta
            print('tau set to None')

        elif tau is None and theta is not None:
            theta_or_rho = theta
            print('tau set to None')

        if a1 < 0 or a2 < 0:
            raise ValueError("a1 and a2 must be non-negative.")
        if tau is not None and not (0.0 < tau < 1.0):
            raise ValueError("tau must be in (0,1).")

        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(random_state)

        # 1) base copula parameter from tau
        if tau is not None:
            theta_or_rho = self._theta_from_tau(family, tau)

        # 2) build base bicop in vinecopulib and simulate from it
        base = pv.Bicop(family=family, parameters=bicop_params(theta_or_rho),
                        rotation=rotation)

        # 2.1) Simulate base copula uniforms (n x 2) from vinecopulib
        U0 = base.simulate(n)  # U0[:,0], U0[:,1] ~ base copula

        # 3) map to standard normal margins
        Z1 = norm.ppf(U0[:, 0])
        Z2 = norm.ppf(U0[:, 1])

        # 4) non-central squared transform
        X1 = np.abs(Z1 + a1)
        X2 = np.abs(Z2 + a2)

        # 5) back to uniforms using non-central chi-square CDF with df=1, nc = a^2
        U1 = stats.norm.cdf(X1 - a1) - stats.norm.cdf(-X1 - a1)
        U2 = stats.norm.cdf(X2 - a2) - stats.norm.cdf(-X2 - a2)

        return np.column_stack([U1, U2])

    def _gaussian_copula_pdf_from_z(self, z1, z2, rho):
        """
        Gaussian copula density c(u,v) written in terms of z1=Phi^{-1}(u), z2=Phi^{-1}(v).
        Here we pass z's directly, avoiding any ppf calls.

        rho in (-1,1)
        """
        rho = float(rho)
        if not (-1.0 < rho < 1.0):
            return np.full_like(z1, np.nan, dtype=float)

        one_m_r2 = 1.0 - rho * rho
        inv = 1.0 / one_m_r2
        quad = (z1 * z1 - 2.0 * rho * z1 * z2 + z2 * z2) * inv
        logc = -0.5 * np.log(one_m_r2) - 0.5 * quad + 0.5 * (z1 * z1 + z2 * z2)
        return np.exp(logc)

    def _pdf_ncs_general(self, params, U, family, rotation, eps=1e-10):
        """
        Fast density for a bivariate Non-Central Squared copula.

        params = [theta, a1, a2]
        U      = array shape (n,2) with entries in [0,1]
        family, rotation used for pv.Bicop
        eps    = clip for stability

        Returns:
          ncsq_copula_pdf: shape (n,)
          Coefs_of_copulapdf: shape (n,4) for eps combos [--, -+, +-, ++]
          biv_copulapdf: shape (n,4) for eps combos [--, -+, +-, ++]
        """
        theta = float(params[0])
        a1 = float(params[1])
        a2 = float(params[2])

        U = np.asarray(U, dtype=float)
        if U.ndim != 2 or U.shape[1] != 2:
            return -np.inf

        # clip U into (0,1) for stability
        u1 = np.clip(U[:, 0], eps, 1.0 - eps)
        u2 = np.clip(U[:, 1], eps, 1.0 - eps)
        n = U.shape[0]

        # ---- helper: precompute everything needed per margin, ONCE ----
        def _precompute_margin(u, a, eps_u=1e-12):
            u = np.clip(u, eps_u, 1.0 - eps_u)

            # x = G_a^{-1}(u) = sqrt(ncx2.ppf(u; df=1, nc=a^2))
            x = np.sqrt(stats.ncx2.ppf(u, df=1, nc=a * a))

            # h_plus / h_minus
            h_plus = x - a
            h_minus = -x - a

            pdf_plus = stats.norm.pdf(h_plus)
            pdf_minus = stats.norm.pdf(h_minus)
            den = np.maximum(pdf_plus + pdf_minus, 1e-300)

            # tilde h = Phi(h)
            t_plus = stats.norm.cdf(h_plus)
            t_minus = stats.norm.cdf(h_minus)

            return (pdf_plus, pdf_minus, den,
                    t_plus, t_minus,
                    h_plus, h_minus)

        pdf1_p, pdf1_m, den1, t1_p, t1_m, h1_p, h1_m = _precompute_margin(u1, a1)
        pdf2_p, pdf2_m, den2, t2_p, t2_m, h2_p, h2_m = _precompute_margin(u2, a2)

        # ---- coefficients for the 4 epsilon combinations: [--, -+, +-, ++] ----
        Coefs_of_copulapdf = np.empty((n, 4), dtype=float)

        # eps1=-1, eps2=-1
        Coefs_of_copulapdf[:, 0] = (pdf1_m / den1) * (pdf2_m / den2)
        # eps1=-1, eps2=+1
        Coefs_of_copulapdf[:, 1] = (pdf1_m / den1) * (pdf2_p / den2)
        # eps1=+1, eps2=-1
        Coefs_of_copulapdf[:, 2] = (pdf1_p / den1) * (pdf2_m / den2)
        # eps1=+1, eps2=+1
        Coefs_of_copulapdf[:, 3] = (pdf1_p / den1) * (pdf2_p / den2)

        # ---- base copula pdf evaluated at 4 transformed pairs, in ONE call ----
        # order must match coefficients: [--, -+, +-, ++]
        X = np.vstack([
            np.column_stack([t1_m, t2_m]),  # --
            np.column_stack([t1_m, t2_p]),  # -+
            np.column_stack([t1_p, t2_m]),  # +-
            np.column_stack([t1_p, t2_p])  # ++
        ])  # shape (4n, 2)

        # --- base copula pdf for the 4 transformed pairs ---
        biv_copulapdf = np.empty((n, 4), dtype=float)

        # Fast Gaussian path (rotation=0 only)
        if family == pv.BicopFamily.gaussian and int(rotation) == 0:
            # For Gaussian copula: z = Phi^{-1}(tilde_h) = h
            biv_copulapdf[:, 0] = self._gaussian_copula_pdf_from_z(h1_m, h2_m, theta)  # --
            biv_copulapdf[:, 1] = self._gaussian_copula_pdf_from_z(h1_m, h2_p, theta)  # -+
            biv_copulapdf[:, 2] = self._gaussian_copula_pdf_from_z(h1_p, h2_m, theta)  # +-
            biv_copulapdf[:, 3] = self._gaussian_copula_pdf_from_z(h1_p, h2_p, theta)  # ++
        else:
            # Fallback to pyvinecopulib for other families or rotated Gaussian
            X = np.vstack([
                np.column_stack([t1_m, t2_m]),  # --
                np.column_stack([t1_m, t2_p]),  # -+
                np.column_stack([t1_p, t2_m]),  # +-
                np.column_stack([t1_p, t2_p])  # ++
            ])  # (4n,2)

            cop = pv.Bicop(family=family, rotation=rotation, parameters=bicop_params(theta))
            pdf_all = cop.pdf(X)  # (4n,)
            biv_copulapdf[:] = pdf_all.reshape(4, n).T

        # ---- assemble final density ----
        interim = Coefs_of_copulapdf * biv_copulapdf
        ncsq_copula_pdf = interim.sum(axis=1)
        ncsq_copula_pdf = np.clip(ncsq_copula_pdf, 1e-300, np.inf)

        return ncsq_copula_pdf, Coefs_of_copulapdf, biv_copulapdf

    def _nLL_ncs_general(self, params, U, family, rotation):
        """
        General negative log-likelihood for a bivariate Non-Central Squared copula.
        """
        return -np.sum(np.log(self._pdf_ncs_general(params, U, family, rotation)[0]))

    def est_ncs_general(self, U, family, rotation, n_grid=(1, 3, 3)):

        # objective
        def obj(p):
            val = self._nLL_ncs_general(p, U, family, rotation)
            return val

        if family == pv.BicopFamily.gaussian:
            params0 = np.array([0, 3, 3], dtype=float)
            bounds = [(-0.99999, 0.9999)] + [(1e-5, 3)] * 2

        elif family in [pv.BicopFamily.gumbel, pv.BicopFamily.joe]:
            params0 = np.array([2, 3, 3], dtype=float)
            bounds = [(1.00001, 15)] + [(1e-5, 3)] * 2

        elif family == pv.BicopFamily.clayton:
            params0 = np.array([1, 0.1, 3], dtype=float)
            bounds = [(0.00001, 15)] + [(1e-5, 3)] * 2

        def grid_starts_interior(bounds, n_grid, margin=1e-3):
            grids = []
            for (lo, hi), n in zip(bounds, n_grid):
                lo2 = lo + margin * (hi - lo)
                hi2 = hi - margin * (hi - lo)
                grids.append(np.linspace(lo2, hi2, n))
            return np.array(list(product(*grids)))

        def grid_starts_interior_fixed_theta(bounds, n_grid, params0, margin=1e-3):
            """
            Build grid starts where the FIRST parameter (theta) is fixed at params0[0],
            and the remaining parameters are gridded inside their bounds.

            bounds  : list of (lo, hi) for [theta, a1, a2]
            n_grid  : tuple like (n_theta, n_a1, n_a2)
                      n_theta is ignored (kept for compatibility)
            params0 : initial parameter vector, theta = params0[0]
            """
            theta0 = float(params0[0])

            grids = []
            for (lo, hi), n in zip(bounds[1:], n_grid[1:]):
                lo2 = lo + margin * (hi - lo)
                hi2 = hi - margin * (hi - lo)
                grids.append(np.linspace(lo2, hi2, n))

            # Cartesian product over a1, a2
            starts_rest = np.array(list(product(*grids)))

            # prepend fixed theta
            theta_col = np.full((starts_rest.shape[0], 1), theta0)
            starts = np.hstack([theta_col, starts_rest])

            return starts

        starts = grid_starts_interior_fixed_theta(bounds, n_grid=n_grid, params0=params0)
        ## starts = grid_starts_interior(bounds, n_grid=n_grid)

        best = None

        for x0 in starts:
            res = minimize(obj,
                           x0,
                           # method="SLSQP",
                           method='L-BFGS-B',
                           bounds=bounds,
                           # tol=1e-15
                           )
            if best is None or res.fun < best.fun:
                best = res

        return {"params": best.x,  ## [theta, a1, a2]
                "nll": float(best.fun),
                'AIC': 2 * 3 + 2 * float(best.fun),
                'BIC': 2 * np.log(len(U)) + 2 * float(best.fun),
                "converged": bool(best.success),
                "message": best.message}

    def haU(self, u, a):
        """NCS helper ``h_a(u) = sign(u) * G_a^{-1}(abs(u)) - a`` for a 1-D array ``u``."""
        u = np.asarray(u)
        x = stats.ncx2.ppf(np.abs(u), 1, a ** 2) ** 0.5
        return np.sign(u) * x - a

    def coef_1d_pre_copulahfunc(self, epsilon, u, a):
        if epsilon not in [-1, 1]:
            raise ValueError('Epsilon must be in {-1, 1}')

        pdf1 = stats.norm.pdf(self.haU(u=epsilon * u, a=a))
        pdf2 = stats.norm.pdf(self.haU(u=u, a=a))
        pdf3 = stats.norm.pdf(self.haU(u=-u, a=a))
        den = pdf2 + pdf3
        den = np.maximum(den, 1e-300)
        return epsilon * pdf1 / den

    def hfunc1_ncs_general(self, u1, u2, params, family, rotation, eps=1e-10, cop=None):
        """
        Same logic/output as your hfunc1_ncs_general_vec, but faster:
        - precomputes tilde_haU(±u, a) for both margins once via ncx2.ppf
        - precomputes coef_1d_pre_copulahfunc for eps=±1 once
        - avoids repeated stacking calls
        - optionally reuses a pre-built copula object
        """
        theta = float(params[0])
        a1, a2 = float(params[1]), float(params[2])

        u1 = np.asarray(u1, dtype=float)
        u2 = np.asarray(u2, dtype=float)

        # broadcast scalars to same length
        u1 = np.atleast_1d(u1)
        u2 = np.atleast_1d(u2)
        if u1.size == 1 and u2.size > 1:
            u1 = np.full_like(u2, float(u1[0]))
        elif u2.size == 1 and u1.size > 1:
            u2 = np.full_like(u1, float(u2[0]))
        elif u1.size != u2.size:
            raise ValueError("u1 and u2 must have same length, or one must be scalar.")

        n = u1.size
        u1 = np.clip(u1, eps, 1.0 - eps)
        u2 = np.clip(u2, eps, 1.0 - eps)

        # ---- precompute tilde_haU(±u, a) ONCE per margin ----
        # This matches your helper: x = sqrt(ncx2.ppf(u; df=1, nc=a^2))
        # then h_plus = x - a ; h_minus = -x - a ; tilde = Phi(h)
        def _tilde_plus_minus(u, a, eps_u=1e-12):
            u = np.clip(u, eps_u, 1.0 - eps_u)
            x = np.sqrt(stats.ncx2.ppf(u, df=1, nc=a * a))
            h_plus = x - a
            h_minus = -x - a
            t_plus = stats.norm.cdf(h_plus)
            t_minus = stats.norm.cdf(h_minus)
            return t_plus, t_minus

        t1_p, t1_m = _tilde_plus_minus(u1, a1)  # tilde_haU(+u1), tilde_haU(-u1)
        t2_p, t2_m = _tilde_plus_minus(u2, a2)  # tilde_haU(+u2), tilde_haU(-u2)

        # ---- precompute coefficients (you call the same thing 4x) ----
        c_minus = self.coef_1d_pre_copulahfunc(epsilon=-1, u=u1, a=a1)
        c_plus = self.coef_1d_pre_copulahfunc(epsilon=+1, u=u1, a=a1)

        # your current code duplicates columns; keep identical behavior:
        # Coefs[:,0]=c_minus; Coefs[:,1]=c_minus; Coefs[:,2]=c_plus; Coefs[:,3]=c_plus

        # ---- build / reuse copula ----
        if cop is None:
            try:
                cop = pv.Bicop(family=family, rotation=rotation, parameters=bicop_params(theta))
            except TypeError:
                cop = pv.Bicop(family, rotation, np.array([[theta]], dtype=float))

        # ---- precompute the 4 base hfunc calls with precomputed t's ----
        h_mm = cop.hfunc1(np.column_stack([t1_m, t2_m]))  # (-u1, -u2)
        h_mp = cop.hfunc1(np.column_stack([t1_m, t2_p]))  # (-u1, +u2)
        h_pm = cop.hfunc1(np.column_stack([t1_p, t2_m]))  # (+u1, -u2)
        h_pp = cop.hfunc1(np.column_stack([t1_p, t2_p]))  # (+u1, +u2)

        # ---- reproduce your algebra EXACTLY ----
        # ncsq = c_minus*h_mm - c_minus*h_mp - c_plus*h_pm + c_plus*h_pp
        out = c_minus * (h_mm - h_mp) + c_plus * (-h_pm + h_pp)

        return out

    def hinv1_ncs_general(self, u1, p, params,  ##[theta, a1, a2]
            family, rotation, tol=1e-12):
        """
        Solve func(a, b, c) = target for a.
        """

        p = float(np.asarray(p).ravel()[0])
        u1 = float(np.asarray(u1).ravel()[0])

        eps = 0
        u2_min = eps
        u2_max = 1.0 - eps

        def gfunc(u2):
            val = self.hfunc1_ncs_general(
                u1=np.array([u1], dtype=float),
                u2=np.array([u2], dtype=float),
                params=params,
                family=family,
                rotation=rotation,
                eps=eps
            )[0]
            return float(val - p)

        # sanity check (important)
        gfunc_min = gfunc(u2_min)
        gfunc_max = gfunc(u2_max)

        if not np.isfinite(gfunc_min) or not np.isfinite(gfunc_max):
            raise ValueError("Function not finite at the bracket endpoints.")

        if not gfunc_min * gfunc_max < 0:
            print(f'gfunc_min = {gfunc_min}, gfunc_max={gfunc_max}')
            print(gfunc_min + p, gfunc_max + p)
            raise ValueError("Function is always positive")

        return brentq(gfunc, u2_min, u2_max, xtol=tol, rtol=tol)

    def hinv1_ncs_general_vec(self, u1, p, params,  # [theta, a1, a2]
            family, rotation, tol=1e-5, eps=1e-5, n_iter=60, cop=None):
        """
        Solve for u2 in:  hfunc1_ncs_general(u1, u2, ...) = p
        Vectorized over p (and u1).

        Assumes monotonicity in u2 for each fixed u1 (required for bisection).
        """

        u1 = np.asarray(u1, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        # broadcast scalars
        if u1.size == 1 and p.size > 1:
            u1 = np.full_like(p, float(u1[0]))
        elif p.size == 1 and u1.size > 1:
            p = np.full_like(u1, float(p[0]))
        elif u1.size != p.size:
            raise ValueError("u1 and p must have same length, or one must be scalar.")

        u1 = np.clip(u1, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)

        lo = np.full_like(p, eps)
        hi = np.full_like(p, 1.0 - eps)

        # Evaluate endpoints once to detect unattainable p
        h_lo = self.hfunc1_ncs_general(u1, lo, params, family, rotation, eps=eps, cop=cop)
        h_hi = self.hfunc1_ncs_general(u1, hi, params, family, rotation, eps=eps, cop=cop)

        out = np.empty_like(p)

        too_low = p <= h_lo
        too_high = p >= h_hi
        good = ~(too_low | too_high)

        # clip out-of-range targets
        out[too_low] = lo[too_low]
        out[too_high] = hi[too_high]

        # bisection on the rest
        lo_g = lo[good].copy()
        hi_g = hi[good].copy()
        u1_g = u1[good].copy()
        p_g = p[good].copy()

        # optional early-stop threshold in terms of bracket width
        width_stop = max(tol, 1e-16)

        for _ in range(n_iter):
            mid = 0.5 * (lo_g + hi_g)
            h_mid = self.hfunc1_ncs_general(u1_g, mid, params, family, rotation, eps=eps, cop=cop)

            left = h_mid < p_g
            lo_g[left] = mid[left]
            hi_g[~left] = mid[~left]

            if np.max(hi_g - lo_g) < width_stop:
                break

        out[good] = 0.5 * (lo_g + hi_g)
        return out

    def hfunc2_ncs_general(self, u1, u2, params, family, rotation, eps=1e-10, cop=None):
        """
        Same logic/output as your hfunc1_ncs_general_vec, but faster:
        - precomputes tilde_haU(±u, a) for both margins once via ncx2.ppf
        - precomputes coef_1d_pre_copulahfunc for eps=±1 once
        - avoids repeated stacking calls
        - optionally reuses a pre-built copula object
        """
        theta = float(params[0])
        a1, a2 = float(params[1]), float(params[2])

        u1 = np.asarray(u1, dtype=float)
        u2 = np.asarray(u2, dtype=float)

        # broadcast scalars to same length
        u1 = np.atleast_1d(u1)
        u2 = np.atleast_1d(u2)
        if u1.size == 1 and u2.size > 1:
            u1 = np.full_like(u2, float(u1[0]))
        elif u2.size == 1 and u1.size > 1:
            u2 = np.full_like(u1, float(u2[0]))
        elif u1.size != u2.size:
            raise ValueError("u1 and u2 must have same length, or one must be scalar.")

        n = u1.size
        u1 = np.clip(u1, eps, 1.0 - eps)
        u2 = np.clip(u2, eps, 1.0 - eps)

        # ---- precompute tilde_haU(±u, a) ONCE per margin ----
        # This matches your helper: x = sqrt(ncx2.ppf(u; df=1, nc=a^2))
        # then h_plus = x - a ; h_minus = -x - a ; tilde = Phi(h)
        def _tilde_plus_minus(u, a, eps_u=1e-12):
            u = np.clip(u, eps_u, 1.0 - eps_u)
            x = np.sqrt(stats.ncx2.ppf(u, df=1, nc=a * a))
            h_plus = x - a
            h_minus = -x - a
            t_plus = stats.norm.cdf(h_plus)
            t_minus = stats.norm.cdf(h_minus)
            return t_plus, t_minus

        t1_p, t1_m = _tilde_plus_minus(u1, a1)  # tilde_haU(+u1), tilde_haU(-u1)
        t2_p, t2_m = _tilde_plus_minus(u2, a2)  # tilde_haU(+u2), tilde_haU(-u2)

        # ---- precompute coefficients (you call the same thing 4x) ----
        c_minus = self.coef_1d_pre_copulahfunc(epsilon=-1, u=u2, a=a2)
        c_plus = self.coef_1d_pre_copulahfunc(epsilon=+1, u=u2, a=a2)

        # your current code duplicates columns; keep identical behavior:
        # Coefs[:,0]=c_minus; Coefs[:,1]=c_minus; Coefs[:,2]=c_plus; Coefs[:,3]=c_plus

        # ---- build / reuse copula ----
        if cop is None:
            try:
                cop = pv.Bicop(family=family, rotation=rotation, parameters=bicop_params(theta))
            except TypeError:
                cop = pv.Bicop(family, rotation, np.array([[theta]], dtype=float))

        # ---- precompute the 4 base hfunc calls with precomputed t's ----
        h_mm = cop.hfunc1(np.column_stack([t1_m, t2_m]))  # (-u1, -u2)
        h_mp = cop.hfunc1(np.column_stack([t1_m, t2_p]))  # (-u1, +u2)
        h_pm = cop.hfunc1(np.column_stack([t1_p, t2_m]))  # (+u1, -u2)
        h_pp = cop.hfunc1(np.column_stack([t1_p, t2_p]))  # (+u1, +u2)

        # ---- reproduce your algebra EXACTLY ----
        # ncsq = c_minus*h_mm - c_minus*h_mp - c_plus*h_pm + c_plus*h_pp
        out = c_minus * (h_mm - h_mp) + c_plus * (-h_pm + h_pp)

        return out

    def hinv2_ncs_general(self, p, u2, params,  ##[theta, a1, a2]
            family, rotation, tol=1e-12):
        """
        Solve func(a, b, c) = target for a.
        """

        p = float(np.asarray(p).ravel()[0])
        u2 = float(np.asarray(u2).ravel()[0])

        eps = 0
        u1_min = eps
        u1_max = 1.0 - eps

        def gfunc(u1):
            val = self.hfunc2_ncs_general(
                u1=np.array([u1], dtype=float),
                u2=np.array([u2], dtype=float),
                params=params,
                family=family,
                rotation=rotation,
                eps=eps
            )[0]
            return float(val - p)

        # sanity check (important)
        gfunc_min = gfunc(u1_min)
        gfunc_max = gfunc(u1_max)

        if not np.isfinite(gfunc_min) or not np.isfinite(gfunc_max):
            raise ValueError("Function not finite at the bracket endpoints.")

        if not gfunc_min * gfunc_max < 0:
            print(f'gfunc_min = {gfunc_min}, gfunc_max={gfunc_max}')
            raise ValueError("Function is always positive")

        return brentq(gfunc, u1_min, u1_max, xtol=tol, rtol=tol)

    def hinv2_ncs_general_vec(self, p, u2, params,  # [theta, a1, a2]
            family, rotation, tol=1e-12, eps=1e-10, n_iter=60, cop=None):
        """
        Solve for u1 in:  hfunc2_ncs_general(u1, u2, ...) = p
        Vectorized over p (and u2).

        Assumes monotonicity in u1 for each fixed u2 (required for bisection).
        """

        u2 = np.asarray(u2, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        # broadcast scalars
        if u2.size == 1 and p.size > 1:
            u2 = np.full_like(p, float(u2[0]))
        elif p.size == 1 and u2.size > 1:
            p = np.full_like(u2, float(p[0]))
        elif u2.size != p.size:
            raise ValueError("u2 and p must have same length, or one must be scalar.")

        u2 = np.clip(u2, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)

        lo = np.full_like(p, eps)
        hi = np.full_like(p, 1.0 - eps)

        # Evaluate endpoints once to detect unattainable p
        h_lo = self.hfunc2_ncs_general(lo, u2, params, family, rotation, eps=eps, cop=cop)
        h_hi = self.hfunc2_ncs_general(hi, u2, params, family, rotation, eps=eps, cop=cop)

        out = np.empty_like(p)

        too_low = p <= h_lo
        too_high = p >= h_hi
        good = ~(too_low | too_high)

        # clip out-of-range targets
        out[too_low] = lo[too_low]
        out[too_high] = hi[too_high]

        # bisection on the rest
        lo_g = lo[good].copy()
        hi_g = hi[good].copy()
        u2_g = u2[good].copy()
        p_g = p[good].copy()

        # optional early-stop threshold in terms of bracket width
        width_stop = max(tol, 1e-16)

        for _ in range(n_iter):
            mid = 0.5 * (lo_g + hi_g)
            h_mid = self.hfunc2_ncs_general(mid, u2_g, params, family, rotation, eps=eps, cop=cop)

            left = h_mid < p_g
            lo_g[left] = mid[left]
            hi_g[~left] = mid[~left]

            if np.max(hi_g - lo_g) < width_stop:
                break

        out[good] = 0.5 * (lo_g + hi_g)
        return out

    def get_condcorrelation_metrics(self, condcorr):
        r"""
        Summary statistics of an exceedance-correlation curve (equations C.2 to C.5).

        Parameters
        ----------
        condcorr : pandas.Series
            Curve from :meth:`draw_cond_corr`, indexed by threshold.

        Returns
        -------
        pandas.Series
            ``mean_leftside``, ``mean_rightside``, ``max_leftside``, ``min_leftside``,
            ``max_rightside``, ``min_rightside``, ``jump_at0_size`` and
            ``halfway_from_jump`` (the reference level :math:`\hat\rho_{mid}`).
        """

        ret = pd.Series([condcorr.loc[condcorr.index < 0].mean(),  ## average corr lower tail
                         condcorr.loc[condcorr.index > 0].mean(),  ## average corr upper tail
                         condcorr.loc[condcorr.index < 0].max(),  ## max corr lower tail
                         condcorr.loc[condcorr.index < 0].min(),  ## min corr lower tail
                         condcorr.loc[condcorr.index > 0].max(),  ## max corr upper tail
                         condcorr.loc[condcorr.index > 0].min(),  ## max corr upper tail
                         np.abs(condcorr.loc[condcorr.index < 0].iloc[-1] - \
                                condcorr.loc[condcorr.index > 0].iloc[0]),  ## just from 0
                         np.max([condcorr.loc[condcorr.index < 0].iloc[-1],
                                 condcorr.loc[condcorr.index > 0].iloc[0]]) - \
                         np.abs(condcorr.loc[condcorr.index < 0].iloc[-1] - \
                                condcorr.loc[condcorr.index > 0].iloc[0]) / 2,  ## Half line point
                         ],
                        index=['mean_leftside',
                               'mean_rightside',
                               'max_leftside',
                               'min_leftside',
                               'max_rightside',
                               'min_rightside',
                               'jump_at0_size',
                               'halfway_from_jump'])

        return ret

    def get_copulas_specifications(self):

        r"""
        Catalog :math:`\mathcal{C}` of single-family candidates with their tail signature.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(copula, rotation, ncs_status)``: Clayton and Gumbel at 0, 90,
            180 and 270 degrees, and Gaussian; columns ``lower_tail_dependence``,
            ``upper_tail_dependence``, ``target_correlation_sign`` (+1, -1, or -11 for
            either sign) and ``monotone_dependence``. This is the table of Appendix C.
        """
        copulas_specifications = pd.DataFrame(index=range(5),
                                              columns=['copula', 'rotation', 'ncs_status',
                                                       'lower_tail_dependence', 'upper_tail_dependence',
                                                       'target_correlation_sign', 'monotone_dependence'])

        copulas_specifications.loc[0, copulas_specifications.columns] = \
            [pv.BicopFamily.clayton, 0, False, True, False, 1, 1]
        copulas_specifications.loc[1, copulas_specifications.columns] = \
            [pv.BicopFamily.clayton, 90, False, True, False, -1, 1]
        copulas_specifications.loc[2, copulas_specifications.columns] = \
            [pv.BicopFamily.clayton, 180, False, False, True, 1, 1]
        copulas_specifications.loc[3, copulas_specifications.columns] = \
            [pv.BicopFamily.clayton, 270, False, False, True, -1, 1]

        copulas_specifications.loc[4, copulas_specifications.columns] = \
            [pv.BicopFamily.gumbel, 0, False, False, True, 1, 1]
        copulas_specifications.loc[5, copulas_specifications.columns] = \
            [pv.BicopFamily.gumbel, 90, False, False, True, -1, 1]
        copulas_specifications.loc[6, copulas_specifications.columns] = \
            [pv.BicopFamily.gumbel, 180, False, True, False, 1, 1]
        copulas_specifications.loc[7, copulas_specifications.columns] = \
            [pv.BicopFamily.gumbel, 270, False, True, False, -1, 1]

        copulas_specifications.loc[8, copulas_specifications.columns] = \
            [pv.BicopFamily.gaussian, 0, False, False, False, -11, 1]

        # copulas_specifications.loc[9, copulas_specifications.columns] = \
        #     [pv.BicopFamily.student, 0, False, False, False, -11, 1]

        copulas_specifications.set_index(['copula', 'rotation', 'ncs_status'], inplace=True)

        return copulas_specifications

    def get_copulas_specifications_Joe_student(self):

        copulas_specifications = pd.DataFrame(index=range(5),
                                              columns=['copula', 'rotation', 'ncs_status',
                                                       'lower_tail_dependence', 'upper_tail_dependence',
                                                       'target_correlation_sign', 'monotone_dependence'])

        copulas_specifications.loc[0, copulas_specifications.columns] = \
            [pv.BicopFamily.joe, 0, False, False, True, 1, 1]
        copulas_specifications.loc[1, copulas_specifications.columns] = \
            [pv.BicopFamily.joe, 90, False, False, True, -1, 1]
        copulas_specifications.loc[2, copulas_specifications.columns] = \
            [pv.BicopFamily.joe, 180, False, True, False, 1, 1]
        copulas_specifications.loc[3, copulas_specifications.columns] = \
            [pv.BicopFamily.joe, 270, False, True, False, -1, 1]

        copulas_specifications.loc[4, copulas_specifications.columns] = \
            [pv.BicopFamily.student, 0, False, False, False, -11, 1]

        copulas_specifications.set_index(['copula', 'rotation', 'ncs_status'], inplace=True)

        return copulas_specifications

    def get_copulas_mixture_specifications(self):

        r"""
        Catalog :math:`\mathcal{M}` of two-component mixtures with their tail signature.

        Returns
        -------
        pandas.DataFrame
            Indexed by the list of two ``Bicop`` components, with columns
            ``lower_tail_dependence`` and ``upper_tail_dependence``.
        """
        copulas_specs_mixture = pd.DataFrame(index=range(5),
                                             columns=['list_copula', 'lower_tail_dependence',
                                                      'upper_tail_dependence'],
                                             dtype=object)

        copulas_specs_mixture.loc[0, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=0),
             pv.Bicop(family=pv.BicopFamily.clayton, rotation=90)]
        copulas_specs_mixture.loc[0, ['lower_tail_dependence', 'upper_tail_dependence']] = [True, False]

        copulas_specs_mixture.loc[1, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=0),
             pv.Bicop(family=pv.BicopFamily.gumbel, rotation=270)]
        copulas_specs_mixture.loc[1, ['lower_tail_dependence', 'upper_tail_dependence']] = [True, False]

        copulas_specs_mixture.loc[2, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=90),
             pv.Bicop(family=pv.BicopFamily.gumbel, rotation=180)]
        copulas_specs_mixture.loc[2, ['lower_tail_dependence', 'upper_tail_dependence']] = [True, False]

        copulas_specs_mixture.loc[3, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=180),
             pv.Bicop(family=pv.BicopFamily.clayton, rotation=270)]
        copulas_specs_mixture.loc[3, ['lower_tail_dependence', 'upper_tail_dependence']] = [False, True]

        copulas_specs_mixture.loc[4, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=180),
             pv.Bicop(family=pv.BicopFamily.gumbel, rotation=90)]
        copulas_specs_mixture.loc[4, ['lower_tail_dependence', 'upper_tail_dependence']] = [False, True]

        copulas_specs_mixture.loc[5, 'list_copula'] = \
            [pv.Bicop(family=pv.BicopFamily.clayton, rotation=270),
             pv.Bicop(family=pv.BicopFamily.gumbel, rotation=0)]
        copulas_specs_mixture.loc[5, ['lower_tail_dependence', 'upper_tail_dependence']] = [False, True]

        copulas_specs_mixture.set_index('list_copula', inplace=True)

        return copulas_specs_mixture

    def select_best_bivariate_copula(self, data, families=None, rotations=None):
        """
        Unrestricted BIC selection among Gaussian, Gumbel, Clayton and Joe copulas with all rotations.

        Parameters
        ----------
        data : array_like of shape (n, 2)
            Pseudo-observations.
        families, rotations : lists, optional
            Restrict the search.

        Returns
        -------
        dict
            ``best_family`` (name), ``best_bic`` (BIC per observation), ``copula``
            (fitted :class:`pyvinecopulib.Bicop`).
        """
        if families == None:
            families = [
                pv.BicopFamily.gaussian,
                pv.BicopFamily.gumbel,
                pv.BicopFamily.clayton,
                pv.BicopFamily.joe,
                # pv.BicopFamily.student,
            ]

        if rotations == None:
            rotations = [0, 90, 180, 270]

        best_bic = float('inf')
        best_copula = None
        best_family = None

        for family in families:
            if family not in [pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.frank]:
                for rotation in rotations:
                    copula = pv.Bicop(family, rotation)

                    copula.fit(np.asarray(data, dtype=float))
                    bic = copula.bic(np.asarray(data, dtype=float)) / len(data)
                    if bic < best_bic:
                        best_bic = bic
                        best_copula = copula
                        best_family = family.name

            else:
                rotation = 0
                copula = pv.Bicop(family, rotation)
                copula.fit(np.asarray(data, dtype=float))
                bic = copula.bic(np.asarray(data, dtype=float)) / len(data)
                if bic < best_bic:
                    best_bic = bic
                    best_copula = copula
                    best_family = family.name

        return {'best_family': best_family,
                'best_bic': best_bic,
                'copula': best_copula
                }

    def select_best_bivariate_ncscopula(self, data, families=None, rotations=None):
        """
        Selects the best bivariate noncentral squared copula based on BIC using partially the pyvinecopulib

        Parameters:
        - data: numpy.ndarray of shape (n_samples, 2), with values in [0, 1]

        Returns:
        - dict with best copula family name, BIC value, and fitted copula object
        """
        if families == None:
            families = [
                pv.BicopFamily.gaussian,
                pv.BicopFamily.gumbel,
                pv.BicopFamily.clayton,
            ]

        if rotations == None:
            rotations = [0, 90, 180, 270]

        best_bic = float('inf')
        best_copula = None
        best_family = None
        best_rotation = None

        for family in families:
            if family not in [pv.BicopFamily.gaussian]:
                for rotation in rotations:
                    est_cop = self.est_ncs_general(U=data,
                                              family=family,
                                              rotation=rotation,
                                              n_grid=(1, 3, 3))
                    bic = est_cop['BIC'] / len(data)
                    if bic < best_bic:
                        best_bic = bic
                        best_copula = est_cop
                        best_family = family
                        best_rotation = rotation
            else:
                rotation = 0
                est_cop = self.est_ncs_general(U=data,
                                          family=family,
                                          rotation=rotation,
                                          n_grid=(1, 3, 3))

                bic = est_cop['BIC'] / len(data)
                if bic < best_bic:
                    best_bic = bic
                    best_copula = est_cop
                    best_family = family
                    best_rotation = rotation

        return {'best_family': best_family,
                'best_rotation': best_rotation,
                'copula': best_copula,
                'best_bic': best_bic,
                }

    def select_best_preselected_bivariate_copula(self, data, families_and_rotations):
        r"""
        BIC selection restricted to the admissible families of a pair (Step 3 of Algorithm 3).

        Parameters
        ----------
        data : array_like of shape (n, 2)
            Pseudo-observations.
        families_and_rotations : list of (BicopFamily, rotation) tuples
            The admissible set :math:`\mathcal{C}_{ik|1:k-1}`.

        Returns
        -------
        dict
            ``best_family``, ``best_bic``, ``copula``.
        """

        best_bic = float('inf')
        best_copula = None
        best_family = None

        for fam_rot in families_and_rotations:
            family = fam_rot[0]
            rotation = fam_rot[1]

            copula = pv.Bicop(family, rotation)

            copula.fit(np.asarray(data, dtype=float))
            bic = copula.bic(np.asarray(data, dtype=float)) / len(data)
            if bic < best_bic:
                best_bic = bic
                best_copula = copula
                best_family = family.name

        return {'best_family': best_family,
                'best_bic': best_bic,
                'copula': best_copula}

    def select_best_preselected_mixture_copula(self, data, list_of_copulas_families):
        """
        Maximum-likelihood fit of each candidate mixture and BIC selection (non-monotone pairs of Algorithm 3).

        Parameters
        ----------
        data : array_like of shape (n, 2)
            Pseudo-observations.
        list_of_copulas_families : list
            Candidate mixtures, each a list of two ``Bicop`` components.

        Returns
        -------
        dict
            ``best_copula`` (the two components), ``best_bic``, ``best_params``
            (``[w, theta_1, theta_2]``).
        """
        best_bic = float('inf')
        best_copula = None
        best_family = None

        for t_copulas_families in list_of_copulas_families:
            t_fit = self.est_mixture_MLE(pseudo_obs=data,
                                         copulas_families=t_copulas_families)

            bic = t_fit['BIC'] / len(data)
            if bic < best_bic:
                best_bic = bic
                best_copula = t_copulas_families
                best_params = t_fit.x

        return {'best_copula': best_copula,
                'best_bic': best_bic,
                'best_params': best_params}

    def hfunc1_ncs_general_fast(self, u1, u2, params, family, rotation, eps=1e-10, cop=None):
        """
        Optimized version with better memory usage and fewer function calls.
        """
        theta = float(params[0])
        a1, a2 = float(params[1]), float(params[2])

        u1 = np.atleast_1d(np.asarray(u1, dtype=float))
        u2 = np.atleast_1d(np.asarray(u2, dtype=float))

        # broadcast
        if u1.size == 1 and u2.size > 1:
            u1 = np.full_like(u2, float(u1[0]))
        elif u2.size == 1 and u1.size > 1:
            u2 = np.full_like(u1, float(u2[0]))

        u1 = np.clip(u1, eps, 1.0 - eps)
        u2 = np.clip(u2, eps, 1.0 - eps)

        # Optimized tilde computation (vectorized)
        def compute_tilde_batch(u_vals, a):
            u_clipped = np.clip(u_vals, 1e-12, 1.0 - 1e-12)
            x = np.sqrt(stats.ncx2.ppf(u_clipped, df=1, nc=a * a))
            h_plus = x - a
            h_minus = -x - a
            return stats.norm.cdf(h_plus), stats.norm.cdf(h_minus)

        t1_p, t1_m = compute_tilde_batch(u1, a1)
        t2_p, t2_m = compute_tilde_batch(u2, a2)

        # Compute coefficients once
        c_minus = self.coef_1d_pre_copulahfunc(epsilon=-1, u=u1, a=a1)
        c_plus = self.coef_1d_pre_copulahfunc(epsilon=+1, u=u1, a=a1)

        # Build copula once
        if cop is None:
            try:
                cop = pv.Bicop(family=family, rotation=rotation, parameters=bicop_params(theta))
            except TypeError:
                cop = pv.Bicop(family, rotation, np.array([[theta]], dtype=float))

        # Batch all copula evaluations
        all_pairs = np.column_stack([
            np.concatenate([t1_m, t1_m, t1_p, t1_p]),
            np.concatenate([t2_m, t2_p, t2_m, t2_p])
        ])

        all_h = cop.hfunc1(all_pairs)
        n = len(u1)
        h_mm, h_mp, h_pm, h_pp = all_h[:n], all_h[n:2 * n], all_h[2 * n:3 * n], all_h[3 * n:]

        return c_minus * (h_mm - h_mp) + c_plus * (-h_pm + h_pp)

    def hinv1_ncs_general_vec_fast(self, u1, p, params,  # [theta, a1, a2]
            family, rotation, tol=1e-5, eps=1e-10, n_iter=60, cop=None):
        """
        Much faster version - pre-computes u1-dependent terms once.

        Key optimizations:
        1. Pre-compute u1-dependent tilde values and coefficients
        2. Create specialized u2-only evaluation function
        3. Avoid redundant computations in bisection loop
        """

        theta = float(params[0])
        a1, a2 = float(params[1]), float(params[2])

        u1 = np.asarray(u1, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        # broadcast scalars
        if u1.size == 1 and p.size > 1:
            u1 = np.full_like(p, float(u1[0]))
        elif p.size == 1 and u1.size > 1:
            p = np.full_like(u1, float(p[0]))
        elif u1.size != p.size:
            raise ValueError("u1 and p must have same length, or one must be scalar.")

        u1 = np.clip(u1, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)

        # ---- PRE-COMPUTE u1-dependent terms (expensive stuff) ----
        def _tilde_plus_minus_fast(u, a):
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            x = np.sqrt(stats.ncx2.ppf(u, df=1, nc=a * a))
            h_plus = x - a
            h_minus = -x - a
            return stats.norm.cdf(h_plus), stats.norm.cdf(h_minus)

        # Pre-compute u1-dependent terms
        t1_p, t1_m = _tilde_plus_minus_fast(u1, a1)
        c_minus = self.coef_1d_pre_copulahfunc(epsilon=-1, u=u1, a=a1)
        c_plus = self.coef_1d_pre_copulahfunc(epsilon=+1, u=u1, a=a1)

        # Pre-build copula
        if cop is None:
            try:
                cop = pv.Bicop(family=family, rotation=rotation, parameters=bicop_params(theta))
            except TypeError:
                cop = pv.Bicop(family, rotation, np.array([[theta]], dtype=float))

        # ---- FAST u2-only evaluation function ----
        def eval_hfunc_u2_only(u2_vals):
            """
            Evaluate hfunc1 for fixed u1 (pre-computed) and varying u2.
            u2_vals: shape (n_points, n_equations)
            Returns: shape (n_points, n_equations)
            """
            n_points, n_eq = u2_vals.shape
            u2_flat = u2_vals.ravel()
            u2_clipped = np.clip(u2_flat, eps, 1.0 - eps)

            # Compute u2-dependent tilde values
            t2_p, t2_m = _tilde_plus_minus_fast(u2_clipped, a2)

            # Expand u1-dependent terms to match u2 grid
            t1_p_exp = np.tile(t1_p, n_points)
            t1_m_exp = np.tile(t1_m, n_points)
            c_minus_exp = np.tile(c_minus, n_points)
            c_plus_exp = np.tile(c_plus, n_points)

            # Compute all 4 copula evaluations in batch
            h_mm = cop.hfunc1(np.column_stack([t1_m_exp, t2_m]))
            h_mp = cop.hfunc1(np.column_stack([t1_m_exp, t2_p]))
            h_pm = cop.hfunc1(np.column_stack([t1_p_exp, t2_m]))
            h_pp = cop.hfunc1(np.column_stack([t1_p_exp, t2_p]))

            # Combine results
            result = c_minus_exp * (h_mm - h_mp) + c_plus_exp * (-h_pm + h_pp)

            return result.reshape(n_points, n_eq)

        # ---- VECTORIZED BISECTION ----
        n_eq = len(u1)
        lo = np.full(n_eq, eps)
        hi = np.full(n_eq, 1.0 - eps)

        # Check endpoints once (vectorized)
        endpoints = np.array([lo, hi])  # shape (2, n_eq)
        h_endpoints = eval_hfunc_u2_only(endpoints)  # shape (2, n_eq)
        h_lo, h_hi = h_endpoints[0], h_endpoints[1]

        out = np.empty_like(p)

        # Handle out-of-range cases
        too_low = p <= h_lo
        too_high = p >= h_hi
        good = ~(too_low | too_high)

        out[too_low] = lo[too_low]
        out[too_high] = hi[too_high]

        if not np.any(good):
            return out

        # Vectorized bisection for valid cases
        lo_g = lo[good].copy()
        hi_g = hi[good].copy()
        p_g = p[good].copy()
        n_good = len(lo_g)

        width_stop = max(tol, 1e-16)

        for _ in range(n_iter):
            mid = 0.5 * (lo_g + hi_g)

            # Evaluate at midpoints (vectorized)
            mid_grid = mid.reshape(1, -1)  # shape (1, n_good)
            h_mid = eval_hfunc_u2_only(mid_grid)[0]  # shape (n_good,)

            # Update brackets
            left = h_mid < p_g
            lo_g[left] = mid[left]
            hi_g[~left] = mid[~left]

            # Early stopping
            if np.max(hi_g - lo_g) < width_stop:
                break

        out[good] = 0.5 * (lo_g + hi_g)
        return out

    def hfunc1_mixture(self, u1, u2, params, families_w_rotations, eps=1e-10):
        """
        ``h``-function of a mixture copula, conditional on the first argument.

        ``h(u2 | u1) = w h_1(u2 | u1) + (1 - w) h_2(u2 | u1)``.

        Parameters
        ----------
        u1, u2 : array_like
        params : sequence
            ``[w, theta_1, theta_2]``.
        families_w_rotations : list of two :class:`pyvinecopulib.Bicop`
        eps : float, default 1e-10
            Clipping of the arguments away from 0 and 1.

        Returns
        -------
        numpy.ndarray
        """

        (w, par_fam1, par_fam2) = params

        u1 = np.atleast_1d(np.asarray(u1, dtype=float))
        u2 = np.atleast_1d(np.asarray(u2, dtype=float))

        # broadcast
        if u1.size == 1 and u2.size > 1:
            u1 = np.full_like(u2, float(u1[0]))
        elif u2.size == 1 and u1.size > 1:
            u2 = np.full_like(u1, float(u2[0]))

        u1 = np.clip(u1, eps, 1.0 - eps)
        u2 = np.clip(u2, eps, 1.0 - eps)

        families_w_rotations_1 = families_w_rotations[0]
        families_w_rotations_2 = families_w_rotations[1]

        families_w_rotations_1.parameters = bicop_params(par_fam1)
        families_w_rotations_2.parameters = bicop_params(par_fam2)

        h1 = families_w_rotations_1.hfunc1(np.array([u1, u2]).T)
        h2 = families_w_rotations_2.hfunc1(np.array([u1, u2]).T)

        h = w * h1 + (1 - w) * h2

        return h

    def hinv1_mixture(self, u1, p, params, families_w_rotations, eps=1e-10, tol=1e-9, n_iter=80):
        """
        Inverse ``h``-function of a mixture copula by vectorized bisection.

        Solves ``hfunc1_mixture(u1, u2) = p`` for ``u2``, which has no closed form for
        a mixture; used by the C-vine sampler and the calibration.

        Parameters
        ----------
        u1, p : array_like
        params : sequence
            ``[w, theta_1, theta_2]``.
        families_w_rotations : list of two :class:`pyvinecopulib.Bicop`
        eps, tol, n_iter
            Clipping, bisection tolerance and maximum number of bisection steps.

        Returns
        -------
        numpy.ndarray
        """
        u1 = np.asarray(u1, dtype=float).ravel()
        p = np.asarray(p, dtype=float).ravel()

        # broadcast scalars
        if u1.size == 1 and p.size > 1:
            u1 = np.full_like(p, float(u1[0]))
        elif p.size == 1 and u1.size > 1:
            p = np.full_like(u1, float(p[0]))
        elif u1.size != p.size:
            raise ValueError("u1 and p must have same length, or one must be scalar.")

        u1 = np.clip(u1, eps, 1.0 - eps)
        p = np.clip(p, eps, 1.0 - eps)

        # Bracket in u2
        lo = np.full_like(p, eps)
        hi = np.full_like(p, 1.0 - eps)

        # Evaluate endpoints once
        h_lo = self.hfunc1_mixture(u1, lo, params, families_w_rotations, eps=eps)
        h_hi = self.hfunc1_mixture(u1, hi, params, families_w_rotations, eps=eps)

        out = np.empty_like(p)

        # Handle out-of-range (or non-bracketing) cases safely
        too_low = p <= h_lo
        too_high = p >= h_hi
        good = ~(too_low | too_high)

        out[too_low] = lo[too_low]
        out[too_high] = hi[too_high]

        if not np.any(good):
            return out

        # Work only on the "good" ones
        idx = np.where(good)[0]
        u1g = u1[idx]
        pg = p[idx]
        log = lo[idx].copy()
        hig = hi[idx].copy()

        for _ in range(n_iter):
            mid = 0.5 * (log + hig)

            h_mid = self.hfunc1_mixture(u1g, mid, params, families_w_rotations, eps=eps)

            left = h_mid < pg
            log[left] = mid[left]
            hig[~left] = mid[~left]

            if np.max(hig - log) < tol:
                break

        out[idx] = 0.5 * (log + hig)
        return out

    def simulate_mixture(self, w, mix_components, n=1000, seed=None):

        """
        Sample from a two-component mixture copula.

        Parameters
        ----------
        w : float
            Weight on the first component.
        mix_components : list of two :class:`pyvinecopulib.Bicop`
            Components with their parameters set.
        n : int, default 1000
        seed : int, optional

        Returns
        -------
        numpy.ndarray of shape (n, 2)
        """
        bern = bernoulli.rvs(1 - w, size=n)

        comp1_sim = mix_components[0].simulate(n=n).copy()
        comp2_sim = mix_components[1].simulate(n=n).copy()

        res = comp1_sim.copy() * 0

        mask1 = bern == 0
        mask2 = bern == 1

        res[mask1] = comp1_sim[mask1].copy()
        res[mask2] = comp2_sim[mask2].copy()

        return res

    def get_biv_mix_nLL(self, params, families_w_rotations, pseudo_obs):

        #     Ex: get_biv_mix_nLL(params=[1, 2.66741, 1.1],
        #                         families_w_rotations=[pv.Bicop(pv.BicopFamily.gumbel, 180),
        #                                               pv.Bicop(pv.BicopFamily.gumbel, 0)],
        #                         pseudo_obs=pseudo_obs)

        w = params[0]
        par_fam1 = params[1]
        par_fam2 = params[2]

        families_w_rotations_1 = families_w_rotations[0]
        families_w_rotations_2 = families_w_rotations[1]

        families_w_rotations_1.parameters = bicop_params(par_fam1)
        families_w_rotations_2.parameters = bicop_params(par_fam2)

        nLL = -np.sum(np.log(w * families_w_rotations_1.pdf(np.asarray(pseudo_obs, dtype=float)) + \
                             (1 - w) * families_w_rotations_2.pdf(np.asarray(pseudo_obs, dtype=float)))
                      )

        return 2 * nLL + np.log(len(pseudo_obs)) * (len(params) + 1)

    def est_mixture_MLE(self, pseudo_obs, copulas_families, constraint_corr=False):
        #     Example : est_mixture_MLE(pseudo_obs=pseudo_obs,
        #                               copulas_families=[pv.Bicop(pv.BicopFamily.gumbel, 180),
        #                                                 pv.Bicop(pv.BicopFamily.gumbel, 0)]
        #                              )

        """
        Maximum-likelihood estimation of a two-component mixture copula.

        Parameters
        ----------
        pseudo_obs : array_like of shape (n, 2)
        copulas_families : list of two :class:`pyvinecopulib.Bicop`
            Components (family and rotation); their parameters are the unknowns.
        constraint_corr : bool, default False
            Unused in the pipeline.

        Returns
        -------
        scipy.optimize.OptimizeResult
            ``x`` is ``[w, theta_1, theta_2]``; the result also carries ``BIC``.
        """
        mixture_bounds = [(0, 1)] + [(bicop_bounds(val, 'lower')[0, 0] + 1e-5,
                                      bicop_bounds(val, 'upper')[0, 0] - 1e-5) for val in copulas_families]

        init_val = np.asarray([0.5] + [val.parameters[0, 0] for val in copulas_families])

        def spearman_const(x):
            t_copulas_families = []
            for i, v in enumerate(copulas_families):
                v.parameters = bicop_params(x[i + 1])
                t_copulas_families.append(v)

            _rho = x[0] * t_copulas_families[0].rho() + (1 - x[0]) * t_copulas_families[1].rho()

            _val = (_rho - (stats.spearmanr(pseudo_obs)[0])) ** 2

            return _val

        if constraint_corr:
            constr = [{"type": "eq", "fun": spearman_const}]
        else:
            constr = []

        results = minimize(self.get_biv_mix_nLL,
                           x0=init_val,
                           args=(copulas_families, pseudo_obs,),
                           method='SLSQP',
                           bounds=mixture_bounds,
                           constraints=constr,
                           )

        if results.success == False:
            class Results:
                pass

            results = Results()
            results.fun = None
            results.x = None
            results.copulas_families = None
            results.BIC = None

        else:
            new_copulas_families = []
            for i, val in enumerate(copulas_families):
                val.parameters = bicop_params(results.x[i + 1])
                new_copulas_families.append(val)
            results.copulas_families = new_copulas_families

            results.BIC = self.get_biv_mix_nLL(params=results.x,
                                          families_w_rotations=copulas_families,
                                          pseudo_obs=pseudo_obs)

        return results

