# -*- coding: utf-8 -*-
"""
User-facing generators: :class:`FleishmanMarket` and :class:`CVineMarket`.

Both take a :class:`~cvinemarketgen.targets.Targets`, expose ``fit``,
``simulate``, ``simulate_paths``, ``diagnostics``, ``plot_exceedance``,
``save`` and ``load``, and sit on top of the paper's engine
(:class:`~cvinemarketgen.cvine.CVineGenerator`,
:class:`~cvinemarketgen.fleishman.FleishmanGenerator`), which is left unchanged.
"""
import contextlib
import io
import json

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from .targets import Targets
from .cvine import CVineGenerator
from .fleishman import FleishmanGenerator
from .dynamics import make_dynamics, AR1, AR1GARCH
from .paths import Paths
from .functions import exceedance_curve

_FAMILY_DEFAULT_THETA = {'gaussian': 0.0, 'clayton': 1.0, 'gumbel': 1.5, 'joe': 1.5, 'frank': 2.0}


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _reorder(targets, order):
    """A Targets with the assets in the given order."""
    t = Targets(mean=targets.mean.loc[order], vol=targets.vol.loc[order], corr=targets.corr.loc[order, order],
                skew=targets.skew.loc[order], kurt=targets.kurt.loc[order], assets=list(order),
                history=None if targets.history is None else targets.history.loc[:, order])
    t.layer, t.freq, t.higher_moments_source = targets.layer, targets.freq, targets.higher_moments_source
    return t


def partial_correlations(corr, order):
    """
    Partial correlations of a C-vine with root order ``order``: a dict
    ``{(i, k): rho_{ik|1:k-1}}`` (1-based, k < i), the Gaussian-copula
    parameters that reproduce ``corr`` exactly.
    """
    C = np.asarray(corr.loc[order, order], float)
    N = len(order)
    out = {}
    for k in range(1, N):
        S = list(range(k - 1))            # conditioning set 1..k-1 (0-based indices)
        R = list(range(k - 1, N))         # remaining variables, first is variable k
        if S:
            Css = C[np.ix_(S, S)]; Crs = C[np.ix_(R, S)]
            P = C[np.ix_(R, R)] - Crs @ np.linalg.solve(Css, Crs.T)
        else:
            P = C[np.ix_(R, R)].copy()
        d = np.sqrt(np.diag(P))
        P = P / np.outer(d, d)
        for j, i in enumerate(R[1:], start=1):
            out[(i + 1, k)] = float(P[0, j])
    return out


class Diagnostics:
    """Target versus simulated moments and correlations. Print it or read ``.moments`` and ``.corr_error``."""

    def __init__(self, targets, X):
        X = pd.DataFrame(X).loc[:, targets.assets]
        sim = pd.concat([X.mean().rename('mean'), X.std(ddof=1).rename('vol'),
                         X.skew().rename('skew'), (X.kurtosis() + 3).rename('kurt')], axis=1)
        tgt = targets.moments
        cols = []
        for m in ['mean', 'vol', 'skew', 'kurt']:
            cols += [tgt[m].rename(f'{m} target'), sim[m].rename(f'{m} simulated'), (sim[m] - tgt[m]).rename(f'{m} diff')]
        self.moments = pd.concat(cols, axis=1)
        self.corr_error = X.corr() - targets.corr
        lo = self.corr_error.values[np.tril_indices(len(targets.assets), -1)]
        self.max_abs_corr_error = float(np.abs(lo).max()) if len(lo) else 0.0
        self.mean_abs_corr_error = float(np.abs(lo).mean()) if len(lo) else 0.0
        self.n = len(X)
        self.layer = targets.layer

    def summary(self, digits=4):
        print(f'{self.n} simulated observations ({self.layer} layer)')
        print(self.moments.round(digits).to_string())
        print(f'\nmax |diff| per moment: ' + ', '.join(f"{m} {self.moments[f'{m} diff'].abs().max():.1e}" for m in ['mean', 'vol', 'skew', 'kurt']))
        print(f'correlation error: max |.| {self.max_abs_corr_error:.4f}, mean |.| {self.mean_abs_corr_error:.4f}')
        return self.moments

    def __repr__(self):
        return (f'Diagnostics(n={self.n}, max |corr error|={self.max_abs_corr_error:.4f}, '
                + ', '.join(f"{m} {self.moments[f'{m} diff'].abs().max():.1e}" for m in ['mean', 'vol', 'skew', 'kurt']) + ')')


class _Market:
    """Shared behaviour of the two generators."""

    def __init__(self, targets, dynamics=None):
        if not isinstance(targets, Targets):
            raise TypeError('targets must be a Targets object')
        self.targets = targets
        self.dynamics = make_dynamics(dynamics)
        self.fit_targets = None     # targets of the layer the generator is fitted on
        self.fitted = False

    # ---- dynamics -----------------------------------------------------------
    def _prepare_layer(self):
        t = self.targets
        if self.dynamics is None:
            self.fit_targets = t
            return
        if t.history is None:
            raise ValueError('dynamics need a history in the targets')
        self.dynamics.fit(t.history)
        if isinstance(self.dynamics, AR1) and t.higher_moments_source != 'history':
            self.fit_targets = self.dynamics.transfer_targets(t)      # LTCMA targets, Appendix A transfer
        elif isinstance(self.dynamics, AR1GARCH) and t.higher_moments_source != 'history':
            raise ValueError("dynamics='ar1-garch' is supported with history-based targets (Targets.from_history)")
        else:
            ft = Targets.from_history(self.dynamics.filter(t.history), assets=t.assets)
            ft.layer = 'residuals'; ft.freq = t.freq
            self.fit_targets = ft

    # ---- public -------------------------------------------------------------
    @property
    def layer(self):
        """``'returns'``, or ``'residuals'`` when dynamics are used: the layer ``simulate`` produces."""
        return 'returns' if self.dynamics is None else 'residuals'

    def simulate(self, n, seed=None, corr_tol=None, accept=True):
        """
        One cross-section of ``n`` simulated observations (DataFrame, one column per asset).

        With ``accept=True`` (default) the draw is kept only if it meets the
        tolerances of Algorithm 5 on moments and correlations, redrawing
        otherwise; the marginal parameters are re-fitted on the draw either way.
        Use ``accept=False`` for small ``n``, where a correlation tolerance is
        not attainable. With dynamics, the observations are residuals of the
        time-series model; use :meth:`simulate_paths` for returns.
        """
        self._check_fitted()
        return self._draw(int(n), seed, corr_tol, accept)

    def simulate_paths(self, n_paths, horizon, seed=None, corr_tol=None, accept=False):
        """
        Return paths: :class:`~cvinemarketgen.paths.Paths` of shape ``(n_paths, horizon, N)``.

        The ``n_paths * horizon`` per-period observations are drawn as one
        cross-section (``accept=False`` by default: no accept-reject, since a
        small pool cannot meet the correlation tolerance) and reshaped. Without
        dynamics the periods are i.i.d.; with dynamics the simulated residuals
        are filtered through the fitted model from the last observed state.
        """
        self._check_fitted()
        X = self._draw(int(n_paths) * int(horizon), seed, corr_tol, accept).values.reshape(int(n_paths), int(horizon), -1)
        if self.dynamics is not None:
            X = self.dynamics.unfilter(X)
        return Paths(X, self.targets.assets)

    def diagnostics(self, X):
        """Compare a simulated cross-section with the targets of its layer."""
        return Diagnostics(self.fit_targets if self.fit_targets is not None else self.targets, X)

    def plot_exceedance(self, X, base=None, pairs=None, zlim=1.0, history=True, ncols=3, ax=None):
        """
        Exceedance-correlation curves of the simulated observations ``X`` (and of
        the history when available), for ``base`` against each asset in ``pairs``.
        """
        import matplotlib.pyplot as plt
        t = self.fit_targets if self.fit_targets is not None else self.targets
        base = base or t.assets[0]
        pairs = [a for a in t.assets if a != base] if pairs is None else list(pairs)
        z = np.round(np.arange(-zlim, zlim + 0.0001, 0.05), 3)
        X = pd.DataFrame(X)
        n = len(pairs); nrows = int(np.ceil(n / ncols))
        if ax is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
            axes = list(axes.flat)
        else:
            axes = [ax]; fig = ax.figure
        for a_, asset in zip(axes, pairs):
            if history and t.history is not None:
                a_.plot(z, exceedance_curve(t.history[base], t.history[asset], z).values, color='black', lw=2, label='historical')
            a_.plot(z, exceedance_curve(X[base], X[asset], z).values, color='tab:blue', lw=1.6, label='simulated')
            a_.axvline(0, color='grey', lw=0.6, ls=':'); a_.axhline(t.corr.loc[base, asset], color='grey', lw=0.6, ls=':')
            a_.set_title(f'{base} vs {asset}  (target {t.corr.loc[base, asset]:+.2f})', fontsize=9)
            a_.set_xlabel(f'threshold on {base} (std)', fontsize=8); a_.tick_params(labelsize=8); a_.grid(alpha=0.25)
        for a_ in axes[n:]:
            a_.axis('off')
        h, l = axes[0].get_legend_handles_labels()
        if ax is None:
            fig.legend(h, l, loc='lower center', ncol=2, frameon=False); fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError('call fit() first')


# =============================================================================
class FleishmanMarket(_Market):
    """
    Fleishman + Vale-Maurelli generator (Section 3 of the paper): four moments
    and the correlation matrix, Gaussian dependence.

    Parameters
    ----------
    targets : Targets
    dynamics : None, 'ar1' or 'ar1-garch'
        Optional serial dependence for :meth:`simulate_paths`.

    Attributes
    ----------
    coefficients : DataFrame
        Fleishman coefficients a, b, c, d per asset (after ``fit``).
    intermediate_corr : DataFrame
        Vale-Maurelli correlations of the underlying normals.
    infeasible_pairs : list
        Pairs whose Vale-Maurelli equation has no real root in [-1, 1].
    """

    def __init__(self, targets, dynamics=None):
        super().__init__(targets, dynamics)
        self.engine = FleishmanGenerator()

    def fit(self):
        self._prepare_layer()
        t = self.fit_targets
        with _quiet():
            out = self.engine.fit(t.mean, t.vol, t.skew, t.kurt - 3.0, t.corr)
        self.coefficients = out['coef'][['a', 'b', 'c', 'd']].astype(float)
        self.intermediate_corr = out['corr_Z']
        self.infeasible_pairs = out['infeasible_pairs']
        self.fitted = True
        return self

    def _draw(self, n, seed, corr_tol, accept=True):
        sim, _, _ = self.engine.simulate(n, corr_tol=0.05 if corr_tol is None else corr_tol, seed=seed,
                                         verbose=False, accept=accept)
        return sim.loc[:, self.fit_targets.assets]

    def save(self, path):
        d = {'kind': 'FleishmanMarket', 'targets': self.targets.to_dict(),
             'dynamics': None if self.dynamics is None else self.dynamics.to_dict()}
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls(Targets.from_dict(d['targets']))
        if d['dynamics'] is not None:
            m.dynamics = (AR1 if d['dynamics']['name'] == 'ar1' else AR1GARCH).from_dict(d['dynamics'])
        m.fit_targets = m.targets if m.dynamics is None else None
        if m.dynamics is not None:
            raise NotImplementedError('loading a FleishmanMarket with dynamics: refit from the history instead')
        return m.fit()


# =============================================================================
class CVineMarket(_Market):
    """
    C-vine copula generator with Johnson SU marginals (Sections 4 and 5 of the paper).

    Parameters
    ----------
    targets : Targets
    central : str, optional
        Asset at the root of the C-vine; default the first asset of ``targets``.
    families : 'auto', 'gaussian' or dict
        ``'auto'``: selected from the history by Algorithm 3 (needs a history).
        ``'gaussian'``: Gaussian pair copulas, initialised at the partial
        correlations of the target matrix, then calibrated.
        dict ``{(asset_i, asset_k): spec}``: chosen per pair, unspecified pairs
        Gaussian. ``spec`` is ``(family, rotation)``, ``(family, rotation, theta)``,
        or ``('mixture', [(fam1, rot1[, th1]), (fam2, rot2[, th2])], w)``.
        Families: ``'gaussian'``, ``'clayton'``, ``'gumbel'``, ``'joe'``, ``'frank'``.
    mixtures : bool, default True
        With ``'auto'``, allow mixture copulas on non-monotone pairs of the first tree.
    mixtures_deeper_trees : bool, default False
        Same for the deeper trees (slower).
    n_opt : int, default 10000
        Draws used inside the calibration objective (paper: 20000).
    corr_tol : float, default 0.05
        Acceptance tolerance on pairwise correlations of a simulated draw
        (paper: 0.02 with 20000 draws).
    tol_func : float, default 1e-6
        Objective value below which the calibration of a variable is accepted
        without trying fallback families (paper: 5e-8).
    dynamics : None, 'ar1' or 'ar1-garch'
        Serial dependence for :meth:`simulate_paths`, fitted on the history.

    Notes
    -----
    After ``fit``, read ``marginals`` (Johnson SU parameters per asset), ``edges``
    (family, rotation and calibrated parameters per edge), and ``engine`` (the
    paper's :class:`~cvinemarketgen.cvine.CVineGenerator`, with ``fit_results``
    and ``vine_results`` as its dictionaries).
    """

    def __init__(self, targets, central=None, families='auto', mixtures=True, mixtures_deeper_trees=False,
                 n_opt=10000, corr_tol=0.05, tol_func=1e-6, dynamics=None):
        super().__init__(targets, dynamics)
        self.central = central or targets.assets[0]
        if self.central not in targets.assets:
            raise ValueError(f'central asset {self.central!r} not in targets')
        self.families = families
        self.mixtures = mixtures
        self.mixtures_deeper_trees = mixtures_deeper_trees
        self.n_opt = int(n_opt)
        self.corr_tol = corr_tol
        self.tol_func = tol_func
        self.order = [self.central] + [a for a in targets.assets if a != self.central]
        self.engine = CVineGenerator(tol_opt=1e-10, n_samples=self.n_opt,
                                     use_ncs_on_deepertrees=False, use_ncs_on_firsttree=False,
                                     use_mixture_on_firsttree=bool(mixtures),
                                     use_mixture_on_deepertrees=bool(mixtures_deeper_trees),
                                     force_try_ncscopula=False, tol_for_optimization_func=tol_func)

    # ---- specification of user-chosen families -------------------------------
    def _spec_from_families(self, t):
        order = self.order
        idx = {a: j + 1 for j, a in enumerate(order)}
        pc = partial_correlations(t.corr, order)
        edges = {}
        given = {} if self.families == 'gaussian' else dict(self.families)
        for (a, b), spec in given.items():
            if a not in idx or b not in idx:
                raise ValueError(f'unknown asset in pair {(a, b)}')
            i, k = max(idx[a], idx[b]), min(idx[a], idx[b])
            if spec[0] == 'mixture':
                comps = []
                for c in spec[1]:
                    fam, rot = c[0], int(c[1])
                    th = c[2] if len(c) > 2 else _FAMILY_DEFAULT_THETA[fam]
                    comps.append((fam, rot, float(th)))
                edges[(i, k)] = ('mixture', comps, float(spec[2]))
            else:
                fam, rot = spec[0], int(spec[1])
                th = spec[2] if len(spec) > 2 else (pc[(i, k)] if fam == 'gaussian' else _FAMILY_DEFAULT_THETA[fam])
                edges[(i, k)] = (fam, rot, float(th))
        N = len(order)
        for k in range(1, N):
            for i in range(k + 1, N + 1):
                edges.setdefault((i, k), ('gaussian', 0, pc[(i, k)]))
        return self.engine.make_vine_spec(order, edges)

    # ---- fit ------------------------------------------------------------------
    def fit(self, verbose=False):
        """Select (or set) the families and calibrate the vine to the targets. Returns self."""
        self._prepare_layer()
        t = _reorder(self.fit_targets, self.order)
        self.fit_targets = t
        if self.families == 'auto' and t.history is None:
            raise ValueError("families='auto' needs a history in the targets; use 'gaussian' or a dict of families")
        ctx = contextlib.nullcontext() if verbose else _quiet()
        with ctx:
            self.setup = self.engine.load_data_and_setup(t.to_setup_frame(), t.history, self.order,
                                                         higher_moments=t.higher_moments_frame())
            if self.families == 'auto':
                self.fit_results = self.engine.fit_and_structure_CVine(self.setup)
            else:
                spec = self._spec_from_families(t)
                self.fit_results = self.engine.fit_results_from_spec(spec, self.order, t.corr)
            self.vine_results = self.engine.run_vine_optimization(self.setup, self.fit_results)
        self.fitted = True
        return self

    # ---- views ------------------------------------------------------------------
    @property
    def marginals(self):
        """Johnson SU parameters per asset (standardized), with the mean and volatility applied afterwards."""
        self._check_fitted()
        p = self.setup['optimal_params'][['a', 'b', 'c', 'd', 'fun']].astype(float)
        p.columns = ['gamma', 'xi', 'delta', 'lambda', 'fit residual']
        p.insert(0, 'vol', self.fit_targets.vol.loc[p.index].values)
        p.insert(0, 'mean', self.fit_targets.mean.loc[p.index].values)
        return p

    @property
    def edges(self):
        """Selected family, rotation and calibrated parameters per edge (tree, variable)."""
        self._check_fitted()
        return self.engine.selected_edge_table(self.order, self.fit_results, self.vine_results)

    @property
    def selected_edges(self):
        """Families and parameters as selected on the history, before calibration (``'auto'`` only)."""
        self._check_fitted()
        return self.engine.selected_edge_table(self.order, self.fit_results)

    def _draw(self, n, seed, corr_tol, accept=True):
        if seed is not None:
            np.random.seed(seed)
        if accept:
            with _quiet():
                sims, _, _ = self.engine.run_multi_year_simulation(
                    self.vine_results, n_year=1, n_per_year=n, corr_tol=self.corr_tol if corr_tol is None else corr_tol)
            return sims[0].loc[:, self.targets.assets]
        # direct draw: one sample from the calibrated vine, marginals re-fitted on it (Steps 1 to 3 of Algorithm 5)
        vr = self.vine_results
        U = self.engine.simulate_CVine(thetas_1p=vr['thetas_final_1p'], thetas_2p=vr['thetas_final_2p'],
                                       a1_s=vr['a1_final'], a2_s=vr['a2_final'], fams=vr['fams_cops'],
                                       rotations=vr['fams_rots'], ncsstatus=vr['fams_ncscops_status'],
                                       mixturestatus=vr['fams_mixture_status'], familystatus=vr['fams_status'], n=n)
        opt = vr['optimal_params']
        out = pd.DataFrame(np.zeros((n, len(self.order))), columns=self.order)
        for j, a in enumerate(self.order):
            x0 = opt.loc[a, ['a', 'b', 'c', 'd']].values.astype(float)
            with _quiet():
                p, _ = self.engine.find_params_for_moments_matching_JSU_with_U(
                    [0, 1, vr['targeted_mom3'][a], vr['targeted_mom4'][a] - 3], x0=x0, U=U[:, j], method='SLSQP')
            out[a] = vr['targeted_mom1'][a] + self.engine.sim_JSU_with_U(U[:, j], p) * vr['targeted_mom2'][a] ** 0.5
        return out.loc[:, self.targets.assets]

    # ---- save / load ---------------------------------------------------------------
    def _edge_records(self):
        vr = self.vine_results
        d = len(self.order)
        recs = []
        for k in range(1, d):
            for i in range(k + 1, d + 1):
                r, c = k - 1, i - 2
                st = vr['fams_status'][r, c]
                if st == 'ordinary':
                    recs.append({'i': i, 'k': k, 'status': st, 'family': vr['fams_cops'][r, c].name,
                                 'rotation': int(vr['fams_rots'][r, c]), 'params': [float(vr['thetas_final_1p'][r, c])]})
                elif st == 'mixture':
                    comps = vr['fams_cops'][r, c]
                    recs.append({'i': i, 'k': k, 'status': st, 'family': [b.family.name for b in comps],
                                 'rotation': [int(b.rotation) for b in comps],
                                 'params': [float(v) for v in np.asarray(vr['thetas_final_1p'][r, c], float)]})
                else:
                    raise NotImplementedError('saving NCS edges is not supported')
        return recs

    def save(self, path):
        """Save the fitted model to JSON (targets, settings, marginals, edges, dynamics)."""
        self._check_fitted()
        d = {'kind': 'CVineMarket', 'version': 1,
             'targets': self.targets.to_dict(), 'fit_targets': self.fit_targets.to_dict(),
             'settings': {'central': self.central, 'families': 'auto' if self.families == 'auto' else 'given',
                          'mixtures': self.mixtures, 'mixtures_deeper_trees': self.mixtures_deeper_trees,
                          'n_opt': self.n_opt, 'corr_tol': self.corr_tol, 'tol_func': self.tol_func},
             'order': self.order,
             'marginals': self.setup['optimal_params'][['a', 'b', 'c', 'd', 'fun']].astype(float).to_dict(orient='index'),
             'edges': self._edge_records(),
             'dynamics': None if self.dynamics is None else self.dynamics.to_dict()}
        with open(path, 'w') as f:
            json.dump(d, f, indent=1)

    @classmethod
    def load(cls, path):
        """Rebuild a fitted model from :meth:`save`; ready to simulate, no refit."""
        with open(path) as f:
            d = json.load(f)
        t = Targets.from_dict(d['targets'])
        s = d['settings']
        m = cls(t, central=s['central'], families='gaussian' if s['families'] == 'given' else 'auto',
                mixtures=s['mixtures'], mixtures_deeper_trees=s['mixtures_deeper_trees'],
                n_opt=s['n_opt'], corr_tol=s['corr_tol'], tol_func=s['tol_func'])
        m.families = s['families']
        m.fit_targets = _reorder(Targets.from_dict(d['fit_targets']), d['order'])
        m.order = d['order']
        if d['dynamics'] is not None:
            m.dynamics = (AR1 if d['dynamics']['name'] == 'ar1' else AR1GARCH).from_dict(d['dynamics'])
        edges = {}
        for e in d['edges']:
            if e['status'] == 'ordinary':
                edges[(e['i'], e['k'])] = (e['family'], e['rotation'], e['params'][0])
            else:
                edges[(e['i'], e['k'])] = ('mixture', [(f, r, th) for f, r, th in zip(e['family'], e['rotation'], e['params'][1:])], e['params'][0])
        spec = m.engine.make_vine_spec(m.order, edges)
        ft = m.fit_targets
        opt = pd.DataFrame(d['marginals']).T[['a', 'b', 'c', 'd', 'fun']].astype(float).loc[m.order]
        opt['Distr'] = 'JSU'
        m.setup = {'ltcma_corr': ft.corr, 'historical_data': None, 'asset_order': m.order,
                   'targeted_mom1': ft.mean, 'targeted_mom2': ft.vol ** 2, 'targeted_mom3': ft.skew,
                   'targeted_mom4': ft.kurt, 'optimal_params': opt, 'df_target_stats': ft.moments}
        m.fit_results = m.engine.fit_results_from_spec(spec, m.order, ft.corr)
        m.vine_results = {'thetas_final_1p': spec['thetas_1p'], 'thetas_final_2p': spec['thetas_2p'],
                          'a1_final': spec['a1_s'], 'a2_final': spec['a2_s'], 'fams_cops': spec['fams'],
                          'fams_rots': spec['rotations'], 'fams_ncscops_status': spec['ncsstatus'],
                          'fams_mixture_status': spec['mixturestatus'], 'fams_status': spec['familystatus'],
                          'ltcma_corr': ft.corr, 'asset_order': m.order, 'optimal_params': opt,
                          'targeted_mom1': ft.mean, 'targeted_mom2': ft.vol ** 2,
                          'targeted_mom3': ft.skew, 'targeted_mom4': ft.kurt}
        m.fitted = True
        return m
