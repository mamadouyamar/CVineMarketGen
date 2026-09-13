# -*- coding: utf-8 -*-
"""
Moment matching tools: Fleishman cubic transform, Vale-Maurelli helpers,
Johnson SU moment fitting. Relocated verbatim from momentmatchingscript.py
(article 3 of the thesis).
"""
import math
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, skew, kurtosis
from scipy.optimize import minimize, fsolve, brentq
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal


def mvnormcdf(upper, mu, cov):
    """P(X <= upper) for X ~ N(mu, cov). Replaces statsmodels' mvnormcdf, whose
    SciPy backend (mvndst) was removed in SciPy 1.16; same value, SciPy only."""
    return float(multivariate_normal(mean=np.asarray(mu, float), cov=np.asarray(cov, float)).cdf(np.asarray(upper, float)))
import pyvinecopulib as pv


def bicop_bounds(bicop, which):
    """Lower ('lower') or upper ('upper') parameter bounds of a pyvinecopulib Bicop as a 2-D
    array; the accessor is a method in pyvinecopulib 0.6 and a property in 0.7."""
    attr = getattr(bicop, f'parameters_{which}_bounds')
    return np.asarray(attr() if callable(attr) else attr, dtype=float)


def bicop_params(*values):
    """Column vector of copula parameters as pyvinecopulib expects (2-D float array).
    pyvinecopulib 0.6 accepted a Python list; 0.7 requires an ndarray of shape (k, 1)."""
    return np.asarray(values, dtype=float).reshape(-1, 1)


class MomentMatch:
    """Fleishman, Vale-Maurelli and Johnson SU moment matching (article 3, Sections 3 and 4.3)."""


    def __init__(self):
        """Initialize MomentMatch with default parameters."""
        pass

    def is_positive_definite(self, corr_df: pd.DataFrame, tol: float = 1e-8) -> bool:
        """Returns True if all eigenvalues are above tol."""
        eigvals = np.linalg.eigvalsh(corr_df.values)
        return bool(np.all(eigvals > tol))

    def make_positive_definite(self, corr_df: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Projects a correlation matrix onto the nearest positive definite matrix.

        Uses eigenvalue clamping (Higham 2002 approximation):
        - Clamp negative eigenvalues to epsilon
        - Reconstruct and rescale to enforce unit diagonal (valid correlation matrix)

        Parameters
        ----------
        corr_df : pd.DataFrame
            Symmetric correlation matrix (values in [-1, 1], diagonal = 1).
        epsilon : float
            Minimum eigenvalue floor. Default 1e-8.

        Returns
        -------
        pd.DataFrame
            Nearest positive definite correlation matrix with same index/columns.
        """
        labels = corr_df.index
        C = corr_df.values.astype(float)

        # Symmetrize to eliminate floating point asymmetry
        C = (C + C.T) / 2

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)

        # Clamp negative (or near-zero) eigenvalues
        eigvals_clamped = np.maximum(eigvals, epsilon)

        # Reconstruct
        C_pd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

        # Rescale to restore unit diagonal (correlation matrix constraint)
        d = np.sqrt(np.diag(C_pd))
        C_pd = C_pd / np.outer(d, d)

        # Final symmetrization pass
        C_pd = (C_pd + C_pd.T) / 2
        np.fill_diagonal(C_pd, 1.0)

        return pd.DataFrame(C_pd, index=labels, columns=labels)

    def spearman_rho(self, family, theta):
        if family == 'Clayton':
            f = dblquad(lambda u, v: (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta), 0, 1, 0, 1)
            res = 12 * f[0] - 3
        elif family == 'Gaussian':
            res = (6 / math.pi) * math.asin(theta / 2)
        return res

    def correlation_rho(self, family, theta, rotation=0):
        if family == 'Clayton':  ## Theta>=0
            if (rotation == 0) or (rotation == 180):
                f = dblquad(
                    lambda x, y: (norm.cdf(x) ** (-theta) + norm.cdf(y) ** (-theta) - 1) ** (-1 / theta) - norm.cdf(
                        x) * norm.cdf(y),
                    -3, 3, -3, 3)
                res = f[0]
            elif (rotation == 90) or (rotation == 270):
                f = dblquad(lambda x, y: (norm.cdf(x) ** (-theta) + (1 - norm.cdf(y)) ** (-theta) - 1) ** (-1 / theta) - \
                                         norm.cdf(x) * (norm.cdf(y)),
                            -3, 3, 3, -3)
                res = f[0]
        elif family == 'Gumbel':  ## Theta>1
            f = dblquad(lambda x, y: np.exp(
                -((-np.log(norm.cdf(x))) ** (theta) + (-np.log(norm.cdf(y))) ** (theta)) ** (1 / theta)) - \
                                     norm.cdf(x) * norm.cdf(y),
                        -3, 3, -3, 3)
            res = f[0]
        elif family == 'Frank':  ## Theta!=0
            f = dblquad(lambda x, y: -(1 / theta) * np.log(1 + ((np.exp(-theta * norm.cdf(x)) - 1) * \
                                                                (np.exp(-theta * norm.cdf(y)) - 1)) / \
                                                           (np.exp(-theta) - 1)) - \
                                     norm.cdf(x) * norm.cdf(y),
                        -3, 3, -3, 3)
            res = f[0]
        elif family == 'Joe':  ## Theta>1
            f = dblquad(lambda x, y: 1 - ((1 - norm.cdf(x)) ** (-theta) + (1 - norm.cdf(y)) ** (-theta) - \
                                          ((1 - norm.cdf(x)) ** (-theta)) * ((1 - norm.cdf(y)) ** (-theta))) * (
                                                 1 / theta) - \
                                     norm.cdf(x) * norm.cdf(y),
                        -3, 3, -3, 3)
            res = f[0]
        return res

    def theta_for_corr(self, theta, corr_target=0, family='Clayton', rotation=0):
        return self.correlation_rho(family, theta, rotation) - corr_target

    def corr2cov_biv(self, corr, std):
        cov = np.array([[1, corr], [corr, 1]]) * np.outer(std, std)
        return cov

    def Psi(self, h, k, rho):
        return -norm.pdf(h) * norm.cdf((k - rho * h) / (np.sqrt(1 - rho ** 2)))

    def Lambda(self, h, k, rho):
        return -(np.sqrt(1 - rho ** 2) / (np.sqrt(2 * np.pi))) * norm.pdf(
            np.sqrt(h ** 2 - 2 * rho * h * k + k ** 2) / np.sqrt(1 - rho ** 2))

    def chi(self, h, k, rho):
        return -k * self.Psi(k, h, rho) + (rho / (1 + rho ** 2)) * self.Lambda(h, k, rho)

    def ConditionalCorrelation(self, x, y, Sigma, theta=0, kind=1):
        ## Conditional Correlations
        ## Longin and Solnik (2001) introduced the notion of exceedance correlation
        ## The Myth of Diversification Reconsidered, William Kinlaw, Mark Kritzman, Sébastien Page, and David Turkington, JPM 2021
        ## kind : param
        ### 1: conditional to x only
        ### 2: conditional to x and y
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        if kind == 1:  ## conditional to x only
            if theta >= 0:
                h = -theta
                k = 10000
                cond = x >= theta
                cc_empirical = np.corrcoef(x[cond], y[cond])[0, 1]
            else:
                h = theta
                k = 10000
                cond = x < theta
                cc_empirical = np.corrcoef(x[cond], y[cond])[0, 1]

        elif kind == 2:
            if theta >= 0:
                h = -theta
                k = -theta
                cond = (x > theta) & (y > theta)
                cc_empirical = np.corrcoef(x[cond], y[cond])[0, 1]
            else:
                h = theta
                k = theta
                cond = (x < theta) & (y < theta)
                cc_empirical = np.corrcoef(x[cond], y[cond])[0, 1]
        rho = Sigma[0, 1] / np.sqrt(np.prod(np.diag(Sigma)))
        Sigma = np.array([[1, rho],
                          [rho, 1]])
        m10 = (self.Psi(h, k, rho) + rho * self.Psi(k, h, rho)) / mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma)
        m01 = (self.Psi(k, h, rho) + rho * self.Psi(h, k, rho)) / mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma)
        m20 = (mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma) - self.chi(k, h, rho) - rho ** 2 * self.chi(h, k, rho)) / \
              mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma)
        m02 = (mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma) - self.chi(h, k, rho) - rho ** 2 * self.chi(k, h, rho)) / \
              mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma)
        m11 = (rho * mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma) + rho * h * self.Psi(h, k, rho) + rho * k * self.Psi(
            k, h, rho) - \
               self.Lambda(h, k, rho)) / mvnormcdf(upper=[h, k], mu=[0, 0], cov=Sigma)
        var_x = m20 - m10 ** 2;
        var_y = m02 - m01 ** 2;
        cov_xy = m11 - m10 * m01;
        cc_theoretical = cov_xy / np.sqrt(var_x * var_y);
        return cc_empirical, cc_theoretical

    def copula_analysis(self, corr_x_y=0, std_x=1, std_y=1):
        Sigma_x_y = self.corr2cov_biv(corr_x_y,
                                      np.array([std_x, std_y]))
        theta_clayton = fsolve(self.theta_for_corr,
                               x0=[0.2],
                               args=(corr_x_y, 'Clayton'))
        clay_cop = pv.Bicop(family=pv.BicopFamily.clayton, parameters=bicop_params(theta_clayton))
        u_clay_cop = clay_cop.simulate(n=1000000, seeds=[1])
        X_clay_cop = norm.ppf(np.array([u_clay_cop[:, 0],
                                        u_clay_cop[:, 1]]).T)
        x_clay, y_clay = X_clay_cop[:, 0], X_clay_cop[:, 1]
        clay180_cop = pv.Bicop(family=pv.BicopFamily.clayton, parameters=bicop_params(theta_clayton))
        u_clay180_cop = clay180_cop.simulate(n=1000000, seeds=[1])
        X_clay180_cop = norm.ppf(np.array([1 - u_clay180_cop[:, 0],
                                           1 - u_clay180_cop[:, 1]]).T)
        x_clay180, y_clay180 = X_clay180_cop[:, 0], X_clay180_cop[:, 1]
        thetas = np.arange(-2, 2, 0.02)
        corr_theo_clay = np.zeros((len(thetas), 1))
        corr_theo_clay180 = np.zeros((len(thetas), 1))
        corr_theo_gauss = np.zeros((len(thetas), 1))
        for i, theta in enumerate(thetas):
            corr_theo_clay[i, 0], corr_theo_gauss[i, 0] = self.ConditionalCorrelation(x_clay,
                                                                                      y_clay,
                                                                                      Sigma_x_y,
                                                                                      theta=theta,
                                                                                      kind=1)
            corr_theo_clay180[i, 0], _ = self.ConditionalCorrelation(x_clay180,
                                                                     y_clay180,
                                                                     Sigma_x_y,
                                                                     theta=theta,
                                                                     kind=1)
        plt.plot(thetas, corr_theo_gauss[:, 0], label='Gaussian')
        plt.plot(thetas, corr_theo_clay[:, 0], label='Clayton')
        plt.plot(thetas, corr_theo_clay180[:, 0], label='Clayton 180')
        # plt.axhline(0.8)
        plt.legend()
        plt.title('Conditional correlation for different threshold with pearson correlation of ' + \
                  str(np.round(corr_x_y, 2)))
        plt.show()
        return corr_theo_clay, corr_theo_clay180, corr_theo_gauss

    def std_norm_moments(self, order=1):
        """
        Calculate theoretical moments of standard normal distribution.

        Parameters:
        -----------
        order : int
            Moment order to calculate

        Returns:
        --------
        float
            Theoretical moment value
        """
        if order % 2 == 0:
            moment = (2 ** (-order / 2)) * (math.factorial(order) / math.factorial(int(order / 2)))
        else:
            moment = 0
        return moment

    def cubic_transform(self, x, params):
        """
        Apply Fleishman cubic transformation.

        Parameters:
        -----------
        x : array-like
            Input variables
        params : array-like
            Transformation parameters [a, b, c, d]

        Returns:
        --------
        numpy.ndarray
            Transformed variables
        """
        a, b, c, d = params
        return a + b * x + c * x ** 2 + d * x ** 3

    def moments_cubic_transform(self, params, distr='gauss', x=0):
        """
        Calculate theoretical moments for cubic transformation.

        Parameters:
        -----------
        params : array-like
            Transformation parameters [a, b, c, d]
        distr : str, default='gauss'
            Distribution type ('gauss', 'gaussian', 'empirical')
        x : array-like, default=0
            Empirical data (for empirical distribution)

        Returns:
        --------
        numpy.ndarray
            Array of moments [mean, variance, skewness, excess_kurtosis]
        """
        a, b, c, d = params[0], params[1], params[2], params[3]

        if distr == 'gauss':
            E_y = a + c
            E_y_2 = b ** 2 + 6 * b * d + 2 * c ** 2 + 15 * d ** 2
            E_y_3 = 2 * c * (b ** 2 + 24 * b * d + 105 * d ** 2 + 2)
            E_y_4 = 24 * (b * d + (c ** 2) * (1 + b ** 2 + 28 * b * d) + (d ** 2) * (
                        12 + 48 * b * d + 141 * (c ** 2) + 225 * d ** 2))
            moments_y = np.array([E_y, E_y_2, E_y_3, E_y_4])

        elif distr == 'gaussian':
            E_x = self.std_norm_moments(1)
            E_x_2 = self.std_norm_moments(2)
            E_x_3 = self.std_norm_moments(3)
            E_x_4 = self.std_norm_moments(4)

            E_y = a + c
            E_y_2 = a ** 2 + 2 * a * c + b ** 2 + 6 * b * d + 3 * c ** 4 + 15 * d ** 2
            E_y_3 = a ** 3 + 3 * a ** 2 * c + 3 * a * b ** 2 + 18 * a * b * d + 9 * a * c ** 2 + 45 * a * d ** 2 + 9 * b ** 2 * c + 90 * b * c * d + 15 * c ** 3 + 315 * c * d ** 2
            E_y_4 = a ** 4 + 4 * a ** 3 * c + 6 * a ** 2 * b ** 2 + 36 * a ** 2 * b * d + 18 * a ** 2 * c ** 2 + 90 * a ** 2 * d ** 2 + 36 * a * b ** 2 * c + 360 * a * b * c * d + 60 * a * c ** 3 + 1260 * a * c * d ** 2 + 3 * b ** 4 + 60 * b ** 3 * d + 90 * b ** 2 * c ** 2 + 630 * b ** 2 * d ** 2 + 1260 * b * c ** 2 * d + 3780 * b * d ** 3 + 105 * c ** 4 + 5670 * c ** 2 * d ** 2 + 10395 * d ** 4

            moments_y = np.array([E_y,
                                  E_y_2 - E_y ** 2,
                                  (E_y_3 - 3 * E_y * (E_y_2 - E_y ** 2) - E_y ** 3) / ((E_y_2 - E_y ** 2) ** 1.5),
                                  (E_y_4 - 4 * E_y * E_y_3 + 6 * (E_y ** 2) * E_y_2 - 3 * E_y ** 4) / (
                                              (E_y_2 - E_y ** 2) ** 2)])

        return moments_y

    def univariate_moments_matching_func(self, params, args):
        """
        Objective function for moment matching optimization.

        Parameters:
        -----------
        params : array-like
            Parameters to optimize
        args : list
            [target_moments, distribution_type, empirical_data]

        Returns:
        --------
        float
            Distance between target and implied moments
        """
        targets_moments = args[0]
        distr = args[1]
        x = args[2]
        implied_moments = self.moments_cubic_transform(params, distr, x=x)
        distance = np.linalg.norm(implied_moments - targets_moments)
        return distance

    def find_params_for_moments_matching(self, targeted_moments, x0, method='CG', distr='gauss', x=0):
        """
        Find parameters for Fleishman moment matching.

        Parameters:
        -----------
        targeted_moments : array-like
            Target statistical moments
        x0 : array-like
            Initial parameter guess
        method : str, default='CG'
            Optimization method
        distr : str, default='gauss'
            Distribution type
        x : array-like, default=0
            Empirical data

        Returns:
        --------
        tuple
            (optimal_parameters, optimization_result)
        """
        if targeted_moments[3] < (targeted_moments[2] ** 2 - 2):
            raise Exception("kurt must be greater or equal than (skew**2 - 2)")

        res = minimize(fun=self.univariate_moments_matching_func,
                       x0=x0,
                       method=method,
                       args=[targeted_moments, distr, x])
        return res.x, res

    def mat_CVine(self, d):
        """
        Create C-vine matrix structure.

        Parameters:
        -----------
        d : int
            Dimension of the vine

        Returns:
        --------
        numpy.ndarray
            C-vine structure matrix
        """
        mat = np.array([d * [0] for i in range(d)])
        for i in range(d):
            mat[i, :d - i] = np.int32(i + 1)
        return mat

    def pcs_RVine(self, fams_code, thetas_1p, thetas_2p):
        """
        Create R-vine pair copula structure.

        Parameters:
        -----------
        fams_code : numpy.ndarray
            Copula family codes
        thetas_1p : numpy.ndarray
            Primary copula parameters
        thetas_2p : numpy.ndarray
            Secondary copula parameters

        Returns:
        --------
        list
            Nested list of Bicop objects representing the vine structure
        """
        d = fams_code.shape[1] + 1
        pcs = [[None for _ in range(d - 1 - i)] for i in range(d - 1)]

        for i in range(d - 1):
            for j in range(d - 1 - i):
                # Map family codes to pyvinecopulib families and rotations
                fam_code = fams_code[i, d - 2 - j]

                if fam_code == 0:
                    family = pv.BicopFamily.indep
                    rotation = 0
                elif fam_code == 1:
                    family = pv.BicopFamily.gaussian
                    rotation = 0
                elif fam_code == 2:
                    family = pv.BicopFamily.clayton
                    rotation = 0
                elif fam_code == 3:
                    family = pv.BicopFamily.clayton
                    rotation = 90
                elif fam_code == 4:
                    family = pv.BicopFamily.clayton
                    rotation = 180
                elif fam_code == 5:
                    family = pv.BicopFamily.clayton
                    rotation = 270
                elif fam_code == 6:
                    family = pv.BicopFamily.gumbel
                    rotation = 0
                elif fam_code == 7:
                    family = pv.BicopFamily.gumbel
                    rotation = 90
                elif fam_code == 8:
                    family = pv.BicopFamily.gumbel
                    rotation = 180
                elif fam_code == 9:
                    family = pv.BicopFamily.gumbel
                    rotation = 270
                elif fam_code == 10:
                    family = pv.BicopFamily.frank
                    rotation = 0
                elif fam_code == 11:
                    family = pv.BicopFamily.joe
                    rotation = 0
                elif fam_code == 12:
                    family = pv.BicopFamily.joe
                    rotation = 90
                elif fam_code == 13:
                    family = pv.BicopFamily.joe
                    rotation = 180
                elif fam_code == 14:
                    family = pv.BicopFamily.joe
                    rotation = 270
                elif fam_code == 15:
                    family = pv.BicopFamily.student
                    rotation = 0
                else:
                    family = pv.BicopFamily.gaussian
                    rotation = 0

                # Set parameters
                if np.isnan(thetas_2p[i, d - 2 - j]) or thetas_2p[i, d - 2 - j] == 0:
                    params = [thetas_1p[i, d - 2 - j]]
                else:
                    params = [thetas_1p[i, d - 2 - j], thetas_2p[i, d - 2 - j]]

                pcs[i][j] = pv.Bicop(family=family, rotation=rotation, parameters=params)

        return pcs

    def copula_equivalence(self, code):
        """
        Map problematic copula to equivalent family (Student-t).

        Parameters:
        -----------
        code : int or float
            Original copula family code

        Returns:
        --------
        int
            Equivalent family code (15 for Student-t)
        """
        return 15

    def init_theta_1p(self, code):
        """
        Initialize primary parameter for copula family.

        Parameters:
        -----------
        code : int or float
            Copula family code

        Returns:
        --------
        float
            Initial parameter value
        """
        if code == 15:  # Student-t
            return 0
        elif code in [2, 3, 4, 6, 5, 7, 8, 9, 11, 12, 13, 14]:
            return 1.5
        else:
            return 1.0

    def init_theta_2p(self, code):
        """
        Initialize secondary parameter for copula family.

        Parameters:
        -----------
        code : int or float
            Copula family code

        Returns:
        --------
        float
            Initial parameter value
        """
        if code == 15:  # Student-t degrees of freedom
            return 10
        else:
            return 0

    def t_famcode_to_vine_fam(self, t_famcode_):
        """
        Map family code to pyvinecopulib family.

        Parameters:
        -----------
        t_famcode_ : int or float
            Family code

        Returns:
        --------
        pv.BicopFamily
            Corresponding pyvinecopulib family
        """
        if t_famcode_ in [6, 7, 8, 9]:
            return pv.BicopFamily.gumbel
        elif t_famcode_ in [2, 3, 4, 5]:
            return pv.BicopFamily.clayton
        elif t_famcode_ == 1:
            return pv.BicopFamily.gaussian
        elif t_famcode_ == 10:
            return pv.BicopFamily.frank
        elif t_famcode_ == 15:
            return pv.BicopFamily.student
        elif t_famcode_ in [11, 12, 13, 14]:
            return pv.BicopFamily.joe
        else:
            return pv.BicopFamily.gaussian

    def vine_fam_to_t_fam(self, vine_fam):
        if vine_fam == pv.BicopFamily.gumbel:
            t_fam = 'Gumbel'
        elif vine_fam == pv.BicopFamily.clayton:
            t_fam = 'Clayton'
        elif vine_fam == pv.BicopFamily.gaussian:
            t_fam = 'Gaussian'
        elif vine_fam == pv.BicopFamily.frank:
            t_fam = 'Frank'
        elif vine_fam == pv.BicopFamily.joe:
            t_fam = 'Joe'
        elif vine_fam == pv.BicopFamily.student:
            t_fam = 'Student'
        return t_fam

    def pcs_CVine_V2(self, fams_code, thetas):
        d = fams_code.shape[1] + 1
        pcs = [(d - i - 1) * [0] for i in range(d - 1)]
        for i in range(len(pcs)):
            for j in range(len(pcs[i])):
                if fams_code[i, d - 2 - j] == 0:
                    tt = pv.Bicop(pv.BicopFamily.indep)
                elif fams_code[i, d - 2 - j] == 1:
                    tt = pv.Bicop(pv.BicopFamily.gaussian, 0, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 2:
                    tt = pv.Bicop(pv.BicopFamily.clayton, 0, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 3:
                    tt = pv.Bicop(pv.BicopFamily.clayton, 90, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 4:
                    tt = pv.Bicop(pv.BicopFamily.clayton, 180, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 5:
                    tt = pv.Bicop(pv.BicopFamily.clayton, 270, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 6:
                    tt = pv.Bicop(pv.BicopFamily.gumbel, 0, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 7:
                    tt = pv.Bicop(pv.BicopFamily.gumbel, 90, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 8:
                    tt = pv.Bicop(pv.BicopFamily.gumbel, 180, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 9:
                    tt = pv.Bicop(pv.BicopFamily.gumbel, 270, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 10:
                    tt = pv.Bicop(pv.BicopFamily.frank, 0, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 11:
                    tt = pv.Bicop(pv.BicopFamily.joe, 0, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 12:
                    tt = pv.Bicop(pv.BicopFamily.joe, 90, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 13:
                    tt = pv.Bicop(pv.BicopFamily.joe, 180, [thetas[i, d - 2 - j]])
                elif fams_code[i, d - 2 - j] == 14:
                    tt = pv.Bicop(pv.BicopFamily.joe, 270, [thetas[i, d - 2 - j]])
                pcs[i][j] = tt
        return pcs

    def fam_rot_to_code(self, t_fam, t_rot):
        if t_fam == 'Independence':
            t_fam_code = 0
        elif t_fam == 'Gaussian':
            t_fam_code = 1
        elif t_fam == 'Clayton' and t_rot == 0:
            t_fam_code = 2
        elif t_fam == 'Clayton' and t_rot == 90:
            t_fam_code = 3
        elif t_fam == 'Clayton' and t_rot == 180:
            t_fam_code = 4
        elif t_fam == 'Clayton' and t_rot == 270:
            t_fam_code = 5
        elif t_fam == 'Gumbel' and t_rot == 0:
            t_fam_code = 6
        elif t_fam == 'Gumbel' and t_rot == 90:
            t_fam_code = 7
        elif t_fam == 'Gumbel' and t_rot == 180:
            t_fam_code = 8
        elif t_fam == 'Gumbel' and t_rot == 270:
            t_fam_code = 9
        elif t_fam == 'Frank' and t_rot == 0:
            t_fam_code = 10
        elif t_fam == 'Joe' and t_rot == 0:
            t_fam_code = 11
        elif t_fam == 'Joe' and t_rot == 90:
            t_fam_code = 12
        elif t_fam == 'Joe' and t_rot == 180:
            t_fam_code = 13
        elif t_fam == 'Joe' and t_rot == 270:
            t_fam_code = 14
        elif t_fam == 'Student' and t_rot == 0:
            t_fam_code = 15
        elif t_fam == 'BB1' and t_rot == 0:
            t_fam_code = 16
        elif t_fam == 'BB1' and t_rot == 90:
            t_fam_code = 17
        elif t_fam == 'BB1' and t_rot == 180:
            t_fam_code = 18
        elif t_fam == 'BB1' and t_rot == 270:
            t_fam_code = 19
        elif t_fam == 'BB6' and t_rot == 0:
            t_fam_code = 20
        elif t_fam == 'BB6' and t_rot == 90:
            t_fam_code = 21
        elif t_fam == 'BB6' and t_rot == 180:
            t_fam_code = 22
        elif t_fam == 'BB6' and t_rot == 270:
            t_fam_code = 23
        elif t_fam == 'BB7' and t_rot == 0:
            t_fam_code = 24
        elif t_fam == 'BB7' and t_rot == 90:
            t_fam_code = 25
        elif t_fam == 'BB7' and t_rot == 180:
            t_fam_code = 26
        elif t_fam == 'BB7' and t_rot == 270:
            t_fam_code = 27
        elif t_fam == 'BB8' and t_rot == 0:
            t_fam_code = 28
        elif t_fam == 'BB8' and t_rot == 90:
            t_fam_code = 29
        elif t_fam == 'BB8' and t_rot == 180:
            t_fam_code = 30
        elif t_fam == 'BB8' and t_rot == 270:
            t_fam_code = 31
        return t_fam_code

    def code_to_fam_rot(self, t_code):

        if t_code == 0:
            t_fam = 'Independence'
            t_rot = 0
        elif t_code == 1:
            t_fam = 'Gaussian'
            t_rot = 0
        elif t_code == 2:
            t_fam = 'Clayton'
            t_rot = 0
        elif t_code == 3:
            t_fam = 'Clayton'
            t_rot = 90
        elif t_code == 4:
            t_fam = 'Clayton'
            t_rot = 180
        elif t_code == 5:
            t_fam = 'Clayton'
            t_rot = 270
        elif t_code == 6:
            t_fam = 'Gumbel'
            t_rot = 0
        elif t_code == 7:
            t_fam = 'Gumbel'
            t_rot = 90
        elif t_code == 8:
            t_fam = 'Gumbel'
            t_rot = 180
        elif t_code == 9:
            t_fam = 'Gumbel'
            t_rot = 270
        elif t_code == 10:
            t_fam = 'Frank'
            t_rot = 0
        elif t_code == 11:
            t_fam = 'Joe'
            t_rot = 0
        elif t_code == 12:
            t_fam = 'Joe'
            t_rot = 90
        elif t_code == 13:
            t_fam = 'Joe'
            t_rot = 180
        elif t_code == 14:
            t_fam = 'Joe'
            t_rot = 270
        elif t_code == 15:
            t_fam = 'Student'
            t_rot = 0
        return t_fam, t_rot

    def h_j_cond_i(self, u_i, u_j, theta, fam):
        if fam == 'Clayton':
            res = 1 / (((u_i / u_j) ** theta + 1 - u_i ** theta) ** (1 / (theta + 1)))

        elif (fam == 'Normal') or (fam == 'Gaussian'):
            res = norm.cdf((norm.ppf(u_j) - theta * norm.ppf(u_i)) / np.sqrt(1 - theta ** 2))

        elif (fam == 'Gumbel'):
            C = np.exp(-((-np.log(u_i)) ** (theta) + (-np.log(u_j)) ** (theta)) ** (1 / theta))
            res = (C / u_i) * ((np.log(u_j) / np.log(u_i)) ** theta + 1) ** ((1 / theta) - 1)

        elif fam == 'Frank':
            L = (np.exp(-theta) - 1) / ((np.exp(-theta * u_j) - 1) * (np.exp(-theta * u_i)))
            R = (np.exp(-theta * u_i) - 1) / (np.exp(-theta * u_i))
            res = 1 / (L + R)

        elif fam == 'Joe':
            res = ((1 - u_j ** (-theta)) * (u_i ** (-theta - 1))) / \
                  (u_j ** (-theta) + u_i ** (-theta) - (u_j ** (-theta)) * (u_i ** (-theta))) ** (1 - 1 / theta)

        return res

    def hinv_j_cond_i(self, u_i, u_j, theta, fam, rotation=0):
        ## https://www.uio.no/studier/emner/matnat/math/STK4520/h10/undervisningsmateriale/copula.pdf
        if fam == 'Clayton':
            if (rotation == 0) or (rotation == 270):
                res = ((u_j ** (-theta / (1 + theta)) - 1) * (u_i ** (-theta)) + 1) ** (-1 / theta)
            elif (rotation == 90) or (rotation == 180):
                res = (((u_j) ** (-theta / (1 + theta)) - 1) * ((1 - u_i) ** (-theta)) + 1) ** (-1 / theta)

        elif (fam == 'Normal') or (fam == 'Gaussian'):
            res = norm.cdf(norm.ppf(u_j) * np.sqrt(1 - theta ** 2) + theta * norm.ppf(u_i))
        # elif (fam=='Gumbel'):
        #    z =
        #    B = (theta-1)*self.W_gumbel(z)
        #    res = np.exp( -(B**(theta) -(-np.log(u_j))**theta )**(1/theta) )
        elif fam == 'Frank':
            res = -(1 / theta) * np.log(
                (u_j * np.exp(-theta * (1 - u_i)) + (1 - u_j)) / ((1 - u_j) * u_j * np.exp(theta * u_i)))

        return res

    def hinv_j_cond_i_v2(self, u_i, u_j, theta, fam_code):
        ## https://www.uio.no/studier/emner/matnat/math/STK4520/h10/undervisningsmateriale/copula.pdf
        if fam_code == 1:
            res = norm.cdf(norm.ppf(u_j) * np.sqrt(1 - theta ** 2) + theta * norm.ppf(u_i))
        #         elif fam_code==2: ##0
        #             res = ( (u_j**(-theta/(1+theta))-1) * (u_i**(-theta)) + 1)**(-1/theta)
        #         elif fam_code==3:  ##90
        #             res = ( ((u_j)**(-theta/(1+theta))-1) * ((1-u_i)**(-theta)) + 1)**(-1/theta)
        #         elif fam_code==4:  ##180
        #             res = 1 - ( ((u_j)**(-theta/(1+theta))-1) * ((1-u_i)**(-theta)) + 1)**(-1/theta)
        #         elif fam_code==5:  ##270
        #             res = 1 - ( (u_j**(-theta/(1+theta))-1) * (u_i**(-theta)) + 1)**(-1/theta)
        elif fam_code in [2, 5]:  ##[0,270]
            res = ((u_j ** (-theta / (1 + theta)) - 1) * (u_i ** (-theta)) + 1) ** (-1 / theta)
        elif fam_code in [3, 4]:  ##[90,180]
            res = (((u_j) ** (-theta / (1 + theta)) - 1) * ((1 - u_i) ** (-theta)) + 1) ** (-1 / theta)

        return res

    def t_fam_to_vine_fam(self, t_fam):
        if t_fam == 'Gumbel':
            vine_fam = pv.BicopFamily.gumbel
        elif t_fam == 'Clayton':
            vine_fam = pv.BicopFamily.clayton
        elif t_fam == 'Gaussian':
            vine_fam = pv.BicopFamily.gaussian
        elif t_fam == 'Frank':
            vine_fam = pv.BicopFamily.frank
        elif t_fam == 'Student':
            vine_fam = pv.BicopFamily.student
        elif t_fam == 'Joe':
            vine_fam = pv.BicopFamily.joe
        elif t_fam == 'BB1':
            vine_fam = pv.BicopFamily.bb1
        elif t_fam == 'BB6':
            vine_fam = pv.BicopFamily.bb6
        elif t_fam == 'BB7':
            vine_fam = pv.BicopFamily.bb7
        elif t_fam == 'BB8':
            vine_fam = pv.BicopFamily.bb8
        return vine_fam

    def vine_fam_to_t_fam(self, vine_fam):
        if vine_fam == pv.BicopFamily.gumbel:
            t_fam = 'Gumbel'
        elif vine_fam == pv.BicopFamily.clayton:
            t_fam = 'Clayton'
        elif vine_fam == pv.BicopFamily.gaussian:
            t_fam = 'Gaussian'
        elif vine_fam == pv.BicopFamily.frank:
            t_fam = 'Frank'
        elif vine_fam == pv.BicopFamily.joe:
            t_fam = 'Joe'
        elif vine_fam == pv.BicopFamily.student:
            t_fam = 'Student'
        return t_fam

    def moments_JSU(self, params):
        """
        Calculate theoretical moments for Johnson SU distribution.

        Parameters:
        -----------
        params : array-like
            Distribution parameters [gamma, chi, delta, lambda]

        Returns:
        --------
        numpy.ndarray
            Array containing [mean, variance, skewness, excess_kurtosis]
        """
        gamma, chi, delta, lambda_ = params[0], params[1], params[2], params[3]

        # Calculate mean
        mu_JU = chi - lambda_ * np.exp(delta ** -2 / 2) * np.sinh(gamma / delta)

        # Calculate variance
        var_JU = (lambda_ ** 2 / 2) * (np.exp(delta ** -2) - 1) * (np.exp(delta ** -2) * np.cosh(2 * gamma / delta) + 1)

        # Calculate skewness
        skew_JU = - ((lambda_ ** 3) * np.sqrt(np.exp(delta ** -2)) * (np.exp(delta ** -2) - 1) ** 2 * \
                     (np.exp(delta ** -2) * (np.exp(delta ** -2) + 2) * np.sinh(3 * gamma / delta) + 3 * np.sinh(
                         gamma / delta))) / \
                  (4 * var_JU ** 1.5)

        # Calculate excess kurtosis components
        K_1 = np.exp(delta ** -2) ** 2 * (
                    np.exp(delta ** -2) ** 4 + 2 * np.exp(delta ** -2) ** 3 + 3 * np.exp(delta ** -2) ** 2 - 3) * \
              np.cosh(4 * gamma / delta)
        K_2 = 4 * np.exp(delta ** -2) ** 2 * (np.exp(delta ** -2) + 2) * np.cosh(2 * gamma / delta)
        K_3 = 3 * (2 * np.exp(delta ** -2) + 1)

        # Calculate excess kurtosis
        ex_kurt_JU = (lambda_ ** 4) * (np.exp(delta ** -2) - 1) ** 2 * (K_1 + K_2 + K_3) / \
                     (8 * var_JU ** 2) - 3

        return np.array([mu_JU, var_JU, skew_JU, ex_kurt_JU])

    def univariate_moments_matching_func_JSU(self, params, args):
        """
        Objective function for JSU distribution parameter estimation.

        Parameters:
        -----------
        params : array-like
            JSU distribution parameters to optimize
        args : list
            Contains target moments [mean, variance, skewness, excess_kurtosis]

        Returns:
        --------
        float
            L2 norm distance between target and implied moments
        """
        targets_moments = args[0]
        implied_moments = self.moments_JSU(params)
        distance = np.linalg.norm(implied_moments - targets_moments)
        return distance

    def find_params_for_moments_matching_JSU(self, targeted_moments, x0, method='CG'):
        """
        Find JSU distribution parameters that match target statistical moments.

        Parameters:
        -----------
        targeted_moments : array-like
            Target moments [mean, variance, skewness, excess_kurtosis]
        x0 : array-like
            Initial parameter guess [gamma, chi, delta, lambda]
        method : str, default='CG'
            Optimization method for scipy.optimize.minimize

        Returns:
        --------
        tuple
            (optimal_parameters, optimization_result)
        """
        res = minimize(fun=self.univariate_moments_matching_func_JSU,
                       x0=x0,
                       method=method,
                       args=[targeted_moments],
                       tol=1e-10)
        return res.x, res

    def sim_JSU_with_U(self, U, params):
        """
        Simulate JSU distribution using uniform random variables.

        Parameters:
        -----------
        U : array-like
            Uniform random variables in [0,1]
        params : array-like
            JSU parameters [gamma, chi, delta, lambda]

        Returns:
        --------
        numpy.ndarray
            JSU-distributed random variables
        """
        gamma, chi, delta, lambda_ = params[0], params[1], params[2], params[3]
        return lambda_ * np.sinh((norm.ppf(U) - gamma) / delta) + chi

    def moments_JSU_with_U(self, params, U):
        """
        Calculate empirical moments from JSU simulation with given uniform inputs.

        Parameters:
        -----------
        params : array-like
            JSU distribution parameters
        U : array-like
            Uniform random variables

        Returns:
        --------
        numpy.ndarray
            Empirical moments [mean, variance, skewness, kurtosis]
        """
        sim_ = self.sim_JSU_with_U(U, params)
        return np.array([np.mean(sim_), np.var(sim_), skew(sim_), kurtosis(sim_)])

    def univariate_moments_matching_func_JSU_with_U(self, params, args):
        """
        Objective function for JSU parameter fitting using specific uniform inputs.

        Parameters:
        -----------
        params : array-like
            JSU parameters to optimize
        args : list
            [target_moments, uniform_variables]

        Returns:
        --------
        float
            Distance between target and empirical moments
        """
        targets_moments = args[0]
        U = args[1]
        implied_moments = self.moments_JSU_with_U(params, U)
        distance = np.linalg.norm(implied_moments - targets_moments)
        return distance

    def find_params_for_moments_matching_JSU_with_U(self, targeted_moments, x0, U, method='CG', maxiter=100):
        """
        Find JSU parameters that match target moments for specific uniform inputs.

        Parameters:
        -----------
        targeted_moments : array-like
            Target statistical moments
        x0 : array-like
            Initial parameter guess
        U : array-like
            Specific uniform random variables to use
        method : str, default='CG'
            Optimization method
        maxiter : int, default=100
            Maximum optimization iterations

        Returns:
        --------
        tuple
            (optimal_parameters, optimization_result)
        """
        res = minimize(fun=self.univariate_moments_matching_func_JSU_with_U,
                       x0=x0,
                       method=method,
                       args=[targeted_moments, U],
                       tol=1e-10,
                       options={'maxiter': maxiter})
        return res.x, res

