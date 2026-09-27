# -*- coding: utf-8 -*-
"""
The simulator for a universe: what you know about each factor and asset goes in
one workbook (:class:`MarketSpec`), one call fits everything (:class:`FactorMarket`),
and every gap is filled by a stated rule (design note
``docs/superpowers/specs/2026-09-26-specification-layer-design.md``, revision 2).

Workbook sheets (all optional except that an asset without a history needs a
mean and a volatility; moments are annual):

* ``assets``: ticker, tag, mean, vol, skew, kurt
* ``tags``: tag, one column per factor; empty = zero, ``fit`` = regressed, a number = the beta
* ``exposures``: ticker, factor, beta, corr  (overrides the tag for that asset and factor)
* ``pairs``: ticker_1, ticker_2, corr
* ``factors``: factor, mean, vol, skew, kurt
* ``factor_corr``: factor_1, factor_2, corr

An empty cell is filled from the history when the object has one, otherwise by a
fallback: skewness 0 and kurtosis 3; mean and volatility have none.

The rule for an asset, in a fixed order, each step taking the previous as given:
betas (values, correlations solved for the beta, regression on the ``fit``
factors, zero elsewhere); volatility (residual variance = target variance minus
systematic variance); shape (cumulants of independent terms add, the residual
carries what the systematic part does not); pair correlations (residual
correlation within its range); mean (alpha).
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .factors import FactorModel, _newey_west_se
from .functions import fit_johnson_su, johnson_su_kurtosis_floor
from .markets import CVineMarket, FleishmanMarket, Diagnostics
from .paths import Paths
from .targets import Targets, nearest_positive_definite

_SHEETS = ('assets', 'tags', 'exposures', 'pairs', 'factors', 'factor_corr')
_MOMENTS = ('mean', 'vol', 'skew', 'kurt')
_SKEW_MAX, _KURT_MAX = 3.5, 40.0          # the residual shape the rule may ask for, at most (the floor at skew 3.5 is 33, below the kurtosis cap)


def _frame(x, columns):
    """A DataFrame with at least ``columns`` (missing ones NaN); None -> empty."""
    df = pd.DataFrame(columns=list(columns)) if x is None else pd.DataFrame(x).copy()
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _is_empty(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == '')


class MarketSpec:
    """
    What is known about each asset and factor: the six sheets of the workbook as
    DataFrames. Build it from frames, or read it with :meth:`read`; write the
    workbook with :meth:`write`. Moments are annual (``periods`` per year).
    """

    def __init__(self, assets=None, tags=None, exposures=None, pairs=None, factors=None, factor_corr=None, periods=12):
        self.periods = int(periods)
        self.assets = _frame(assets, ['ticker', 'tag'] + list(_MOMENTS))
        self.assets['ticker'] = self.assets['ticker'].astype(str)
        if self.assets['ticker'].duplicated().any():
            raise ValueError('assets: duplicated tickers')
        self.tags = pd.DataFrame() if tags is None else pd.DataFrame(tags).copy()
        if 'tag' in self.tags.columns:
            self.tags = self.tags.set_index('tag')
        self.exposures = _frame(exposures, ['ticker', 'factor', 'beta', 'corr'])
        self.pairs = _frame(pairs, ['ticker_1', 'ticker_2', 'corr'])
        self.factors = _frame(factors, ['factor'] + list(_MOMENTS))
        self.factor_corr = _frame(factor_corr, ['factor_1', 'factor_2', 'corr'])
        for df in (self.assets, self.factors):
            for c in _MOMENTS:
                df[c] = pd.to_numeric(df[c], errors='coerce')

    # ---- io -----------------------------------------------------------------------
    @classmethod
    def read(cls, path, periods=12):
        """Read the workbook (``.xlsx``, needs ``openpyxl``); absent sheets are empty."""
        sheets = pd.read_excel(path, sheet_name=None)
        kw = {k: sheets.get(k) for k in _SHEETS}
        return cls(periods=periods, **kw)

    def write(self, path):
        """Write the six sheets to an ``.xlsx`` workbook."""
        with pd.ExcelWriter(path) as xw:
            self.assets.to_excel(xw, sheet_name='assets', index=False)
            self.tags.reset_index().rename(columns={'index': 'tag'}).to_excel(xw, sheet_name='tags', index=False)
            self.exposures.to_excel(xw, sheet_name='exposures', index=False)
            self.pairs.to_excel(xw, sheet_name='pairs', index=False)
            self.factors.to_excel(xw, sheet_name='factors', index=False)
            self.factor_corr.to_excel(xw, sheet_name='factor_corr', index=False)

    def to_dict(self):
        d = {'periods': self.periods}
        for k in _SHEETS:
            df = getattr(self, k)
            if k == 'tags':
                df = df.reset_index().rename(columns={'index': 'tag'})
            d[k] = json.loads(df.to_json(orient='split', date_format='iso'))
        return d

    @classmethod
    def from_dict(cls, d):
        kw = {}
        for k in _SHEETS:
            j = d.get(k)
            kw[k] = None if j is None else pd.DataFrame(j['data'], columns=j['columns'])
        return cls(periods=d.get('periods', 12), **kw)

    # ---- queries ---------------------------------------------------------------------
    def asset_moments(self, ticker):
        """Given annual moments of an asset as a dict (only the filled ones)."""
        row = self.assets[self.assets['ticker'] == ticker]
        if row.empty:
            return {}
        row = row.iloc[0]
        return {m: float(row[m]) for m in _MOMENTS if not _is_empty(row[m])}

    def factor_moments(self, factor):
        row = self.factors[self.factors['factor'].astype(str) == factor]
        if row.empty:
            return {}
        row = row.iloc[0]
        return {m: float(row[m]) for m in _MOMENTS if not _is_empty(row[m])}

    def asset_states(self, ticker, factors, has_history):
        """
        Exposure state of one asset on every factor: ``('value', x)``, ``('corr', rho)``,
        ``('fit',)`` or ``('zero',)``. From the tag row (empty = zero, ``fit``, number),
        overridden by the ``exposures`` rows of the asset; an asset with no tag and no
        rows is ``fit`` everywhere when it has a history, zero everywhere otherwise.
        """
        row = self.assets[self.assets['ticker'] == ticker]
        tag = None if row.empty or _is_empty(row.iloc[0]['tag']) else str(row.iloc[0]['tag'])
        st = {}
        if tag is not None:
            if tag not in self.tags.index:
                raise ValueError(f'asset {ticker!r}: tag {tag!r} not in the tags sheet')
            trow = self.tags.loc[tag]
            for f in factors:
                v = trow[f] if f in trow.index else np.nan
                if _is_empty(v):
                    st[f] = ('zero',)
                elif isinstance(v, str):
                    if v.strip().lower() != 'fit':
                        raise ValueError(f"tags sheet, tag {tag!r}, factor {f!r}: expected a number, 'fit' or empty, got {v!r}")
                    st[f] = ('fit',)
                else:
                    st[f] = ('value', float(v))
        else:
            st = {f: ('fit',) if has_history else ('zero',) for f in factors}
        ex = self.exposures[self.exposures['ticker'].astype(str) == ticker]
        for _, r in ex.iterrows():
            f = str(r['factor'])
            if f not in factors:
                raise ValueError(f'exposures sheet, asset {ticker!r}: unknown factor {f!r}')
            if not _is_empty(r['beta']):
                st[f] = ('value', float(r['beta']))
            elif not _is_empty(r['corr']):
                st[f] = ('corr', float(r['corr']))
            else:
                raise ValueError(f'exposures sheet, asset {ticker!r}, factor {f!r}: give a beta or a corr')
        return st, tag

    def copy(self):
        return MarketSpec(assets=self.assets, tags=self.tags.reset_index().rename(columns={'index': 'tag'}), exposures=self.exposures,
                          pairs=self.pairs, factors=self.factors, factor_corr=self.factor_corr, periods=self.periods)

    def add(self, sheet, rows):
        """
        A copy with ``rows`` (a DataFrame or a list of dicts) appended to ``sheet``.
        For ``tags`` the rows are indexed by ``tag`` and may bring new factor columns;
        an existing tag is replaced. For ``assets`` and ``factors`` an existing ticker
        or factor is replaced. Other sheets append.
        """
        if sheet not in _SHEETS:
            raise ValueError(f'unknown sheet {sheet!r}')
        new = self.copy()
        add = pd.DataFrame(rows)
        if sheet == 'tags':
            if 'tag' in add.columns:
                add = add.set_index('tag')
            base = new.tags.drop(index=[t for t in add.index if t in new.tags.index], errors='ignore')
            new.tags = pd.concat([base, add], axis=0, sort=False)
        else:
            key = {'assets': 'ticker', 'factors': 'factor'}.get(sheet)
            base = getattr(new, sheet)
            if key is not None:
                base = base[~base[key].astype(str).isin(add[key].astype(str))]
            setattr(new, sheet, _frame(pd.concat([base, add], axis=0, ignore_index=True, sort=False), base.columns))
        return MarketSpec(assets=new.assets, tags=new.tags.reset_index().rename(columns={'index': 'tag'}), exposures=new.exposures,
                          pairs=new.pairs, factors=new.factors, factor_corr=new.factor_corr, periods=new.periods)

    def __repr__(self):
        return (f'MarketSpec({len(self.assets)} assets, {len(self.tags)} tags, {len(self.exposures)} exposure rows, '
                f'{len(self.pairs)} pairs, {len(self.factors)} factor rows, {len(self.factor_corr)} factor correlations)')


class FactorMarket:
    """
    One call from what you have to scenarios for a universe.

    Parameters
    ----------
    asset_history, factor_history : DataFrame or None
        Monthly returns, one column per asset / factor. A ticker or factor has a
        history iff it is a column here.
    spec : MarketSpec or None
        The workbook. None = histories only, everything estimated.
    generator : 'cvine' or 'fleishman'
        The generator on the factors.
    central : str, optional
        Central factor of the C-vine (default the first factor with a history).
    n_pilot : int
        Size of the pilot draw used for the systematic cumulants.
    market_kwargs : dict
        Passed to the generator (families, symmetry_level, ...).

    After ``fit``: ``targets`` (factor Targets), ``market`` (the generator), ``model``
    (a :class:`~cvinemarketgen.factors.FactorModel` ready to simulate), ``report``
    (per asset), ``factor_report`` (per factor).
    """

    def __init__(self, asset_history=None, factor_history=None, spec=None, generator='cvine', central=None,
                 n_pilot=50000, seed=0, market_kwargs=None):
        self.R = None if asset_history is None else pd.DataFrame(asset_history).astype(float)
        self.F = None if factor_history is None else pd.DataFrame(factor_history).astype(float)
        if self.R is not None and self.F is not None:
            idx = self.R.index.intersection(self.F.index)
            self.R, self.F = self.R.loc[idx], self.F.loc[idx]
        self.spec = spec if spec is not None else MarketSpec()
        if generator not in ('cvine', 'fleishman'):
            raise ValueError("generator must be 'cvine' or 'fleishman'")
        self.generator = generator
        self.central = central
        self.n_pilot = int(n_pilot)
        self.seed = seed
        self.market_kwargs = dict(market_kwargs or {})
        self.fitted = False

    # ---- factors ---------------------------------------------------------------------
    def _fit_factors(self):
        sp = self.spec; per = sp.periods
        hist = [] if self.F is None else list(self.F.columns)
        listed = [str(f) for f in sp.factors['factor'].tolist()] if len(sp.factors) else []
        synthetic = [f for f in listed if f not in hist]
        self.factor_names = hist + synthetic
        rows = {}
        if hist:
            t = Targets.from_history(self.F[hist])
            mean, vol, skew, kurt, corr = t.mean.copy(), t.vol.copy(), t.skew.copy(), t.kurt.copy(), t.corr.copy()
            for f in hist:
                g = sp.factor_moments(f)
                src = {}
                for m in _MOMENTS:
                    if m in g:
                        val = g[m] / per if m == 'mean' else g[m] / np.sqrt(per) if m == 'vol' else g[m]
                        {'mean': mean, 'vol': vol, 'skew': skew, 'kurt': kurt}[m][f] = val
                        src[m] = 'workbook'
                    else:
                        src[m] = 'history'
                rows[f] = {'history': True, **{f'{m} source': src[m] for m in _MOMENTS}, 'corr given': 0, 'corr completed': 0}
            n_edit = 0
            for _, r in sp.factor_corr.iterrows():
                a, b = str(r['factor_1']), str(r['factor_2'])
                if a in hist and b in hist:
                    corr.loc[a, b] = corr.loc[b, a] = float(r['corr']); n_edit += 1
                    rows[a]['corr given'] += 1; rows[b]['corr given'] += 1
            corr2, projected = nearest_positive_definite(corr)
            if projected:
                warnings.warn('factor_corr: the edited correlation matrix was projected onto the nearest positive-definite one')
            t = Targets(mean=mean, vol=vol, corr=corr2, skew=skew, kurt=kurt, history=self.F[hist], assets=hist)
            t.higher_moments_source = 'history'
        else:
            t = None
        for f in synthetic:
            g = sp.factor_moments(f)
            if 'mean' not in g or 'vol' not in g:
                raise ValueError(f'factor {f!r} has no history: give its mean and vol in the factors sheet')
            cr = {}
            for _, r in sp.factor_corr.iterrows():
                a, b = str(r['factor_1']), str(r['factor_2'])
                if f in (a, b):
                    other = b if a == f else a
                    if other == f:
                        continue
                    if t is None or other not in t.assets:
                        raise ValueError(f'factor_corr: {f!r} with {other!r}: {other!r} must be a factor with a history or one listed before {f!r}')
                    cr[other] = float(r['corr'])
            skew_, kurt_ = g.get('skew', 0.0), g.get('kurt', 3.0)
            if t is None:
                t = Targets(mean={f: g['mean'] / per}, vol={f: g['vol'] / np.sqrt(per)}, corr=[[1.0]], skew={f: skew_}, kurt={f: kurt_}, assets=[f])
                t.synthetic = [f]
            else:
                t = t.add_factor(f, g['mean'], g['vol'], cr, skew=skew_, kurt=kurt_, annualized=True, periods=per)
            rows[f] = {'history': False, 'mean source': 'workbook', 'vol source': 'workbook',
                       'skew source': 'workbook' if 'skew' in g else 'normal', 'kurt source': 'workbook' if 'kurt' in g else 'normal',
                       'corr given': len(cr), 'corr completed': len(t.assets) - 1 - len(cr)}
        if t is None:
            raise ValueError('no factors: give a factor history or a factors sheet')
        self.targets = t
        self.factor_report = pd.DataFrame(rows).T.loc[self.factor_names]

    # ---- assets -----------------------------------------------------------------------
    def _fit_assets(self):
        sp = self.spec; per = sp.periods; t = self.targets
        factors = list(t.assets); K = len(factors)
        mu_f = t.mean.loc[factors].values
        Sigma_f = np.outer(t.vol.loc[factors].values, t.vol.loc[factors].values) * t.corr.loc[factors, factors].values
        sigma_f = np.sqrt(np.diag(Sigma_f))
        hist_assets = [] if self.R is None else list(self.R.columns)
        listed = [str(a) for a in sp.assets['ticker'].tolist()]
        assets = list(listed) + [a for a in hist_assets if a not in listed] if len(listed) else list(hist_assets)
        if not assets:
            raise ValueError('no assets: give an asset history or an assets sheet')
        unknown = [c for c in sp.tags.columns if str(c) not in factors]
        if unknown:
            raise ValueError(f'tags sheet: columns {unknown} are not factors (declare them in the factors sheet or give their history)')
        P = self.pilot.loc[:, factors].values
        n_obs = 0 if self.R is None else len(self.R)
        nw = int(np.floor(4 * (max(n_obs, 1) / 100.0) ** (2.0 / 9.0)))
        rows, alpha, beta, se, tstat, r2, rvol, rparams, src, resid, samp_rvol = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        for a in assets:
            has_hist = a in hist_assets
            states, tag = sp.asset_states(a, factors, has_hist)
            given = sp.asset_moments(a)
            flags = []
            # ---- target moments of the asset (monthly) and their sources
            mom, msrc = {}, {}
            for m in _MOMENTS:
                if m in given:
                    mom[m] = given[m] / per if m == 'mean' else given[m] / np.sqrt(per) if m == 'vol' else given[m]; msrc[m] = 'workbook'
                elif has_hist:
                    x = self.R[a]
                    mom[m] = {'mean': x.mean(), 'vol': x.std(ddof=1), 'skew': x.skew(), 'kurt': x.kurtosis() + 3.0}[m]; msrc[m] = 'history'
                elif m in ('skew', 'kurt'):
                    mom[m] = 0.0 if m == 'skew' else 3.0; msrc[m] = 'normal'
                else:
                    raise ValueError(f'asset {a!r} has no history: give its mean and vol in the assets sheet')
            # ---- step 1: betas
            value = {f: x[1] for f, x in states.items() if x[0] == 'value'}
            corr_g = {f: x[1] for f, x in states.items() if x[0] == 'corr'}
            fit = [f for f, x in states.items() if x[0] == 'fit']
            zero = [f for f, x in states.items() if x[0] == 'zero']
            if fit and not has_hist:
                raise ValueError(f'asset {a!r} has no history: give values for {fit} instead of fit')
            b = np.zeros(K); s_ = np.full(K + 1, np.nan); c0 = 0.0; e = None; R2 = np.nan
            for f, x in value.items():
                b[factors.index(f)] = x
            fixed_idx = [factors.index(f) for f in list(value) + list(corr_g)]
            fit_idx = [factors.index(f) for f in fit]
            for _ in range(3 if corr_g else 1):
                if fit_idx:
                    y = self.R[a].values - self.F[[factors[i] for i in fixed_idx if factors[i] in self.F.columns]].values @ b[[i for i in fixed_idx if factors[i] in self.F.columns]]
                    X = np.column_stack([np.ones(n_obs)] + [self.F[factors[i]].values for i in fit_idx])
                    c, *_ = np.linalg.lstsq(X, y, rcond=None)
                    e = y - X @ c; c0 = c[0]; b[fit_idx] = c[1:]
                    s_[[0] + [1 + i for i in fit_idx]] = _newey_west_se(X, e, nw)
                if corr_g:
                    Kc = [factors.index(f) for f in corr_g]; others = [i for i in range(K) if i not in Kc]
                    rhs = np.array([corr_g[factors[i]] * mom['vol'] * sigma_f[i] for i in Kc]) - Sigma_f[np.ix_(Kc, others)] @ b[others]
                    b[Kc] = np.linalg.solve(Sigma_f[np.ix_(Kc, Kc)], rhs)
            if has_hist:
                if e is None:
                    e = self.R[a].values - self.F[[f for f in factors if f in self.F.columns]].values @ b[[i for i, f in enumerate(factors) if f in self.F.columns]]
                    e = e - e.mean()
                tss = ((self.R[a].values - self.R[a].values.mean()) ** 2).sum(); R2 = 1.0 - (e ** 2).sum() / tss
                resid[a] = e; samp_rvol[a] = float(np.std(e, ddof=1))
            # ---- step 2: volatility
            sys_var = float(b @ Sigma_f @ b); v = mom['vol']
            if v ** 2 <= sys_var * (1 + 1e-9):
                contrib = pd.Series(b ** 2 * np.diag(Sigma_f), index=factors).sort_values(ascending=False)
                top = ', '.join(f'{f} (beta {b[factors.index(f)]:.2f})' for f in contrib.index[:2] if contrib[f] > 0)
                if msrc['vol'] == 'workbook':
                    raise ValueError(f'asset {a!r}: the betas alone give a volatility of {np.sqrt(sys_var) * np.sqrt(per) * 100:.1f}% a year, '
                                     f'above the target {v * np.sqrt(per) * 100:.1f}%; largest contributors {top}. Lower them or raise the volatility')
                # the sample volatility is a fallback, not a target: the factor targets (a view on a correlation, on a volatility)
                # can push the systematic variance above it; the asset then takes the systematic volatility, with a small residual, and a flag
                v_new = float(np.sqrt(sys_var) * 1.02)
                flags.append(f'volatility raised from the sample {v * np.sqrt(per) * 100:.1f}% to the systematic {v_new * np.sqrt(per) * 100:.1f}% a year '
                             f'implied by the betas and the factor targets ({top})')
                mom['vol'] = v = v_new; msrc['vol'] = 'systematic'
            sig_e = float(np.sqrt(v ** 2 - sys_var))
            # ---- step 3: shape by the cumulant rule
            sysr = P @ b; sd_s = sysr.std()
            k3_sys = float(stats.skew(sysr) * sd_s ** 3) if sd_s > 0 else 0.0
            k4_sys = float(stats.kurtosis(sysr) * sd_s ** 4) if sd_s > 0 else 0.0
            k3_e = mom['skew'] * v ** 3 - k3_sys; k4_e = (mom['kurt'] - 3.0) * v ** 4 - k4_sys
            skew_e = k3_e / sig_e ** 3; kurt_e = 3.0 + k4_e / sig_e ** 4
            share = sig_e ** 2 / v ** 2                                   # unexplained share of the variance
            exact_normal = abs(skew_e) < 1e-9 and abs(kurt_e - 3.0) < 1e-9
            feasible = lambda sk, ku: abs(sk) <= _SKEW_MAX and ku <= _KURT_MAX and ku >= johnson_su_kurtosis_floor(sk)
            if not exact_normal and not feasible(skew_e, kurt_e):
                if msrc['skew'] == 'workbook' or msrc['kurt'] == 'workbook':
                    raise ValueError(f'asset {a!r}: the shape you gave (skew {mom["skew"]:.2f}, kurt {mom["kurt"]:.2f}) would need a residual '
                                     f'with skew {skew_e:.1f} and kurt {kurt_e:.1f}, which no Johnson SU has; the residual is only '
                                     f'{100 * share:.0f}% of the variance and the factors already give the asset skew {k3_sys / v ** 3:.2f} and '
                                     f'kurtosis {3 + k4_sys / v ** 4:.2f}. Give a shape closer to the factors\', or lower the exposures')
                # nearest shape a Johnson SU has: the skewness is kept (capped), the kurtosis raised to the floor for that skewness
                sk0, ku0 = skew_e, kurt_e
                skew_e = float(np.clip(skew_e, -_SKEW_MAX, _SKEW_MAX))
                fl = johnson_su_kurtosis_floor(skew_e)
                kurt_e = float(min(max(kurt_e, fl), max(_KURT_MAX, fl)))
                what = []
                if abs(sk0) > _SKEW_MAX:
                    what.append(f'skew {sk0:.1f} capped at {skew_e:.1f}')
                if ku0 < kurt_e - 1e-9:
                    what.append(f'kurt {ku0:.1f} raised to the floor {kurt_e:.1f}')
                if ku0 > _KURT_MAX:
                    what.append(f'kurt {ku0:.0f} capped at {kurt_e:.0f}')
                flags.append(f'residual shape not reproducible ({100 * share:.0f}% of the variance): ' + ', '.join(what))
            p = fit_johnson_su(0.0, 3.1, mean=0.0, vol=sig_e) if exact_normal else fit_johnson_su(skew_e, kurt_e, mean=0.0, vol=sig_e)
            # ---- step 5: mean (pairs are set after every asset is known)
            alpha[a] = float(mom['mean'] - b @ mu_f)
            beta[a] = b; se[a] = s_; r2[a] = R2; rvol[a] = sig_e; rparams[a] = p
            src[a] = {f: states[f][0] for f in factors}
            rows[a] = {'history': has_hist, 'tag': tag, 'value': len(value), 'corr': len(corr_g), 'fit': len(fit), 'zero': len(zero),
                       **{f'{m} source': msrc[m] for m in _MOMENTS}, 'pairs': 0,
                       'resid vol': sig_e, 'resid skew': skew_e, 'resid kurt': kurt_e, 'flags': '; '.join(flags)}
        # ---- the FactorModel that simulates
        fm = FactorModel.__new__(FactorModel)
        fm.assets, fm.factors = assets, factors
        fm.hist_factors = [f for f in factors if self.F is not None and f in self.F.columns]
        fm.synthetic_factors = [f for f in factors if f not in fm.hist_factors]
        fm.exposures = None; fm._plan = {}; fm.factor_targets = t
        fm.R, fm.F = self.R, self.F
        fm.n_obs, fm.nw_lags = n_obs, nw
        fm.alpha = pd.Series(alpha).loc[assets]
        fm.beta = pd.DataFrame(beta, index=factors).T.loc[assets]
        cols = ['alpha'] + factors
        fm.se = pd.DataFrame(se, index=cols).T.loc[assets]
        coef = pd.concat([fm.alpha.rename('alpha'), fm.beta], axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            fm.tstat = coef / fm.se
        fm.r2 = pd.Series(r2).loc[assets]
        fm.resid = None if not resid else pd.DataFrame(resid, index=self.R.index)
        fm.resid_vol = pd.Series(rvol).loc[assets]
        fm.sample_resid_vol = pd.Series({a: samp_rvol.get(a, np.nan) for a in assets})
        fm.resid_params = pd.DataFrame(rparams).T.loc[assets, ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']].astype(float)
        fm.beta_source = pd.DataFrame(src).T.loc[assets, factors]
        fm.asset_source = pd.Series({a: 'history' if rows[a]['history'] else 'spec' for a in assets})
        fm.resid_corr = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
        fm._gauss_corr = None; fm.target_report = None; fm.pair_report = None
        fm._targets_applied = pd.DataFrame(False, index=assets, columns=['mean', 'vol', 'pair'])
        fm.fitted = True; fm._build_report()
        # ---- step 4: pairs
        pairs = {}
        for _, r in sp.pairs.iterrows():
            i, j = str(r['ticker_1']), str(r['ticker_2'])
            for x in (i, j):
                if x not in assets:
                    raise ValueError(f'pairs sheet: unknown asset {x!r}')
            pairs[(i, j)] = float(r['corr']); rows[i]['pairs'] += 1; rows[j]['pairs'] += 1
        if pairs:
            fm._set_pairs(pairs, Sigma_f)
        self.model = fm
        rep = pd.DataFrame(rows).T.loc[assets]
        for c in ('value', 'corr', 'fit', 'zero', 'pairs'):
            rep[c] = rep[c].astype(int)
        self.report = rep
        self.pair_report = fm.pair_report

    # ---- public --------------------------------------------------------------------------
    def fit(self, verbose=False):
        """Factor targets, generator, pilot draw, the rule for every asset. Returns self."""
        self._fit_factors()
        if self.generator == 'cvine':
            central = self.central or self.targets.historical[0]
            kw = {'families': 'auto'}; kw.update(self.market_kwargs)
            self.market = CVineMarket(self.targets, central=central, **kw).fit(verbose=verbose)
        else:
            self.market = FleishmanMarket(self.targets, **self.market_kwargs).fit()
        self.pilot = self.market.simulate(self.n_pilot, seed=self.seed)
        self._fit_assets()
        self.fitted = True
        return self

    def with_spec(self, spec):
        """
        A new fitted simulator on ``spec`` with the same histories and settings. The
        factor generator is reused when the ``factors`` and ``factor_corr`` sheets are
        unchanged (only the asset rule reruns, a fraction of a second); otherwise the
        factors are refitted.
        """
        new = FactorMarket(self.R, self.F, spec, generator=self.generator, central=self.central, n_pilot=self.n_pilot,
                           seed=self.seed, market_kwargs=self.market_kwargs)
        same = self.fitted and spec.factors.equals(self.spec.factors) and spec.factor_corr.equals(self.spec.factor_corr) \
            and spec.periods == self.spec.periods
        if same:
            new.targets, new.factor_report, new.market, new.pilot, new.factor_names = self.targets, self.factor_report, self.market, self.pilot, self.factor_names
            new._fit_assets(); new.fitted = True
            return new
        return new.fit()

    def simulate(self, n, seed=None):
        """``n`` scenario months, one column per asset; the factor months are in ``last_factors``."""
        self._check()
        Fs = self.market.simulate(int(n), seed=seed)
        self.last_factors = Fs
        return self.model.simulate(Fs, seed=seed)

    def simulate_paths(self, n_paths, horizon, seed=None):
        """Independent months arranged in paths: a :class:`~cvinemarketgen.paths.Paths` of assets."""
        self._check()
        Fp = self.market.simulate_paths(int(n_paths), int(horizon), seed=seed)
        self.last_factor_paths = Fp
        return self.model.simulate(Fp, seed=seed)

    def check(self, X=None, n=25000, seed=0):
        """
        Given against simulated. Returns a dict of DataFrames: ``factors`` (moments,
        with the correlation error in ``.attrs``), ``assets`` (mean, vol, skew, kurt:
        target, source, simulated, Monte Carlo error), ``pairs`` (target, simulated).
        """
        self._check()
        if X is None:
            X = self.simulate(n, seed=seed)
        Fs = self.last_factors
        d = Diagnostics(self.targets, Fs)
        fac = d.moments.copy(); fac.attrs['max_abs_corr_error'] = d.max_abs_corr_error; fac.attrs['mean_abs_corr_error'] = d.mean_abs_corr_error
        nX = len(X); fm = self.model
        mu = pd.Series(fm.alpha.values + fm.beta.values @ self.targets.mean.loc[fm.factors].values, index=fm.assets)
        S = fm.implied_covariance(); s = pd.Series(np.sqrt(np.diag(S.values)), index=fm.assets)
        rows = {}
        for a in fm.assets:
            x = X[a]; r = self.report.loc[a]
            tgt = {'mean': mu[a], 'vol': s[a]}
            # the asset's target shape is the sum of the systematic and residual cumulants it was built from
            rows[a] = {'mean target': tgt['mean'], 'mean source': r['mean source'], 'mean simulated': x.mean(), 'mean MC error': x.std() / np.sqrt(nX),
                       'vol target': tgt['vol'], 'vol source': r['vol source'], 'vol simulated': x.std(), 'vol MC error': x.std() / np.sqrt(2 * nX),
                       'skew source': r['skew source'], 'skew simulated': x.skew(), 'skew MC error': np.sqrt(6.0 / nX),
                       'kurt source': r['kurt source'], 'kurt simulated': x.kurtosis() + 3.0, 'kurt MC error': np.sqrt(24.0 / nX)}
        assets = pd.DataFrame(rows).T
        for a in fm.assets:                                    # the target shape: what the analyst gave or the sample had
            given = self.spec.asset_moments(a)
            for m in ('skew', 'kurt'):
                if m in given:
                    assets.loc[a, f'{m} target'] = given[m]
                elif self.R is not None and a in self.R.columns:
                    assets.loc[a, f'{m} target'] = self.R[a].skew() if m == 'skew' else self.R[a].kurtosis() + 3.0
                else:
                    assets.loc[a, f'{m} target'] = 0.0 if m == 'skew' else 3.0
        order = []
        for m in _MOMENTS:
            order += [f'{m} target', f'{m} source', f'{m} simulated', f'{m} MC error']
        assets = assets[order]
        pairs = None
        if fm.pair_report is not None:
            pr = fm.pair_report
            pairs = pd.DataFrame({'target': pr['target'], 'simulated': [X[i].corr(X[j]) for i, j in pr.index],
                                  'MC error': [(1 - pr.loc[(i, j), 'target'] ** 2) / np.sqrt(nX) for i, j in pr.index]}, index=pr.index)
        return {'factors': fac, 'assets': assets, 'pairs': pairs}

    # ---- persistence -----------------------------------------------------------------------
    def save(self, stem):
        """Three files: ``<stem>_market.json``, ``<stem>_model.json``, ``<stem>_spec.json``."""
        self._check()
        self.market.save(f'{stem}_market.json')
        self.model.save(f'{stem}_model.json')
        with open(f'{stem}_spec.json', 'w') as f:
            json.dump({'spec': self.spec.to_dict(), 'generator': self.generator, 'central': self.central, 'n_pilot': self.n_pilot,
                       'seed': self.seed, 'factor_names': self.factor_names,
                       'report': json.loads(self.report.reset_index().rename(columns={'index': 'asset'}).to_json(orient='split')),
                       'factor_report': json.loads(self.factor_report.reset_index().rename(columns={'index': 'factor'}).to_json(orient='split'))}, f, indent=1)

    @classmethod
    def load(cls, stem):
        """Rebuild from :meth:`save`; ready to simulate, no refit (histories are not stored)."""
        with open(f'{stem}_spec.json') as f:
            d = json.load(f)
        m = cls(spec=MarketSpec.from_dict(d['spec']), generator=d['generator'], central=d['central'], n_pilot=d['n_pilot'], seed=d['seed'])
        m.market = (CVineMarket if d['generator'] == 'cvine' else FleishmanMarket).load(f'{stem}_market.json')
        m.targets = m.market.targets
        m.model = FactorModel.load(f'{stem}_model.json')
        m.factor_names = d['factor_names']
        m.report = pd.DataFrame(d['report']['data'], columns=d['report']['columns']).set_index('asset')
        m.factor_report = pd.DataFrame(d['factor_report']['data'], columns=d['factor_report']['columns']).set_index('factor')
        m.pair_report = m.model.pair_report
        m.fitted = True
        return m

    def _check(self):
        if not self.fitted:
            raise RuntimeError('call fit() first')

    def __repr__(self):
        if not self.fitted:
            return f'FactorMarket(unfitted, generator={self.generator!r})'
        return (f'FactorMarket({len(self.model.assets)} assets on {len(self.targets.assets)} factors, generator={self.generator!r}, '
                f'{int((~self.report["history"].astype(bool)).sum())} assets without history, {len(self.targets.synthetic)} factors without history)')
