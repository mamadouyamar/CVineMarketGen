# -*- coding: utf-8 -*-
"""
Targets: what the user wants the simulated returns to match.

A :class:`Targets` object holds, per asset, the mean, volatility, skewness and
kurtosis, plus the correlation matrix, and optionally the history they came
from. It accepts whatever the user has (see the constructors) and is the input
of :class:`~cvinemarketgen.markets.FleishmanMarket` and
:class:`~cvinemarketgen.markets.CVineMarket`.

Units are the user's: annual targets give annual returns, daily give daily.
Kurtosis is raw (a normal has 3).
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_series(x, assets, name):
    """Series indexed by assets from a Series, dict, list or array."""
    if x is None:
        return None
    if isinstance(x, pd.Series):
        s = x.astype(float)
        if assets is not None:
            missing = [a for a in assets if a not in s.index]
            if missing:
                raise ValueError(f'{name}: missing assets {missing}')
            s = s.loc[assets]
        return s.rename(name)
    if isinstance(x, dict):
        s = pd.Series(x, dtype=float)
        return _as_series(s, assets, name)
    arr = np.asarray(x, dtype=float).ravel()
    if assets is None:
        raise ValueError(f'{name}: give asset names when passing an array')
    if len(arr) != len(assets):
        raise ValueError(f'{name}: {len(arr)} values for {len(assets)} assets')
    return pd.Series(arr, index=list(assets), name=name)


def _as_corr(corr, assets):
    """Correlation DataFrame indexed and columned by assets, symmetrized, unit diagonal."""
    if isinstance(corr, pd.DataFrame):
        if assets is None:
            assets = list(corr.columns)
        c = corr.loc[assets, assets].astype(float)
    else:
        arr = np.asarray(corr, dtype=float)
        if assets is None:
            raise ValueError('corr: give asset names when passing an array')
        if arr.shape != (len(assets), len(assets)):
            raise ValueError(f'corr: shape {arr.shape} for {len(assets)} assets')
        c = pd.DataFrame(arr, index=list(assets), columns=list(assets))
    m = c.values
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=c.index, columns=c.columns)


def nearest_positive_definite(corr, epsilon=1e-8):
    """Eigenvalue clamping and rescaling to unit diagonal (Algorithm 1 of the paper)."""
    C = (corr.values + corr.values.T) / 2.0
    w, V = np.linalg.eigh(C)
    if w.min() > epsilon:
        return corr.copy(), False
    C2 = V @ np.diag(np.maximum(w, epsilon)) @ V.T
    d = np.sqrt(np.diag(C2))
    C2 = C2 / np.outer(d, d)
    C2 = (C2 + C2.T) / 2.0
    np.fill_diagonal(C2, 1.0)
    return pd.DataFrame(C2, index=corr.index, columns=corr.columns), True


def complete_correlation(corr, given):
    """
    Maximum-determinant completion of a partially specified correlation matrix.

    ``corr`` is a square DataFrame whose entries flagged ``True`` in the boolean
    DataFrame ``given`` (same shape; the diagonal counts as given) are kept, and
    whose other entries are chosen to maximize ``log det``. The result is the
    unique valid completion in which every unspecified pair has zero partial
    correlation given the specified entries: nothing is assumed beyond what was
    given. For one new variable with specified set S and unspecified set U it is
    the closed form ``r_U = R_US R_SS^-1 r_S``; the general case is solved by
    maximizing the concave ``log det`` over the free entries from that start.

    Returns the completed DataFrame. Raises ``ValueError`` when the given
    entries admit no valid completion.
    """
    C = np.array(corr.values, float); G = np.array(given.values, bool)
    n = C.shape[0]; np.fill_diagonal(G, True); G = G | G.T; C = (C + C.T) / 2
    free = [(i, j) for i in range(n) for j in range(i + 1, n) if not G[i, j]]
    if not free:
        return corr.copy()
    # start: closed form, one incomplete row at a time, earlier completions treated as given
    X = C.copy(); done = G.copy()
    for i in range(n):
        U = [j for j in range(n) if not done[i, j]]
        if not U:
            continue
        S = [j for j in range(n) if done[i, j] and j != i]
        if S:
            RSS = X[np.ix_(S, S)]; rS = X[i, S]
            q = float(rS @ np.linalg.solve(RSS, rS))
            if q >= 1.0:
                raise ValueError(f'complete_correlation: the given correlations of {corr.index[i]!r} admit no valid '
                                 f'completion (r\' R^-1 r = {q:.3f} >= 1)')
            X[i, U] = X[np.ix_(U, S)] @ np.linalg.solve(RSS, rS)
        else:
            X[i, U] = 0.0
        X[U, i] = X[i, U]; done[i, U] = done[U, i] = True
    x0 = np.array([X[i, j] for i, j in free])

    def build(x):
        M = C.copy()
        for k, (i, j) in enumerate(free):
            M[i, j] = M[j, i] = x[k]
        return M

    def f(x):
        M = build(x); w = np.linalg.eigvalsh(M)
        if w.min() <= 1e-12:
            return 1e6
        return -float(np.sum(np.log(w)))

    def grad(x):
        M = build(x); w = np.linalg.eigvalsh(M)
        if w.min() <= 1e-12:
            return np.zeros_like(x)
        Mi = np.linalg.inv(M)
        return np.array([-2.0 * Mi[i, j] for i, j in free])

    res = minimize(f, x0, jac=grad, method='L-BFGS-B', bounds=[(-0.999, 0.999)] * len(free), options={'maxiter': 500, 'gtol': 1e-10})
    M = build(res.x if res.fun < f(x0) + 1e-12 else x0)
    if np.linalg.eigvalsh(M).min() <= 0:
        raise ValueError('complete_correlation: the given correlations admit no valid completion')
    return pd.DataFrame(M, index=corr.index, columns=corr.columns)


class Targets:
    """
    Per-asset moment targets and a correlation matrix, the input of both generators.

    Parameters
    ----------
    mean, vol : Series, dict, list or array
        Mean and volatility per asset, in the user's units (annual, monthly,
        daily). ``vol`` must be positive.
    corr : DataFrame or square array
        Target correlation matrix. Symmetrized; projected onto the nearest
        positive-definite matrix with a warning if needed.
    skew, kurt : Series, dict, list or array, optional
        Skewness and raw kurtosis (normal = 3). If omitted and ``history`` is
        given, estimated from the history; if omitted without history, set to
        0 and 3 (normal marginals) and flagged in :meth:`summary`.
    history : DataFrame, optional
        Returns, one column per asset, any frequency. Used for the higher
        moments when they are not given, and by :class:`~cvinemarketgen.markets.CVineMarket`
        to select copula families and to fit dynamics.
    assets : list of str, optional
        Asset names and order. Inferred from ``mean`` (Series/dict) or ``corr``
        (DataFrame) when omitted. The first asset is the default central node
        of the C-vine.

    Examples
    --------
    Own long-term assumptions, nothing else::

        t = Targets(mean={'Equity': 0.07, 'Bonds': 0.03}, vol={'Equity': 0.16, 'Bonds': 0.05},
                    corr=[[1, -0.1], [-0.1, 1]], assets=['Equity', 'Bonds'])

    A history only (targets are the sample)::

        t = Targets.from_history(returns)

    A published LTCMA table plus a history for the higher moments::

        t = Targets.from_ltcma('data/jpm_ltcma_2024.csv', assets=[...], history=returns)
    """

    def __init__(self, mean, vol, corr, skew=None, kurt=None, history=None, assets=None):
        if assets is None:
            if isinstance(mean, pd.Series):
                assets = list(mean.index)
            elif isinstance(mean, dict):
                assets = list(mean.keys())
            elif isinstance(corr, pd.DataFrame):
                assets = list(corr.columns)
            else:
                raise ValueError('give asset names (assets=...) when the inputs carry none')
        self.assets = list(assets)
        self.mean = _as_series(mean, self.assets, 'mean')
        self.vol = _as_series(vol, self.assets, 'vol')
        if (self.vol <= 0).any():
            raise ValueError('vol must be positive')
        self.corr, projected = nearest_positive_definite(_as_corr(corr, self.assets))
        if projected:
            warnings.warn('corr was not positive definite; projected onto the nearest positive-definite matrix')
        self.history = None
        if history is not None:
            h = pd.DataFrame(history).astype(float)
            missing = [a for a in self.assets if a not in h.columns]
            if missing:
                raise ValueError(f'history: missing columns {missing}')
            self.history = h.loc[:, self.assets].ffill().dropna()
        self.higher_moments_source = 'given'
        if skew is None or kurt is None:
            if self.history is not None:
                s_h = self.history.skew()
                k_h = self.history.kurtosis() + 3.0
                skew = s_h if skew is None else skew
                kurt = k_h if kurt is None else kurt
                self.higher_moments_source = 'history'
            else:
                skew = pd.Series(0.0, index=self.assets) if skew is None else skew
                kurt = pd.Series(3.0, index=self.assets) if kurt is None else kurt
                self.higher_moments_source = 'default (normal)'
        self.skew = _as_series(skew, self.assets, 'skew')
        self.kurt = _as_series(kurt, self.assets, 'kurt')
        bad = self.kurt < self.skew ** 2 + 1
        if bad.any():
            raise ValueError(f'kurt must be >= skew**2 + 1; violated for {list(self.kurt.index[bad])}')
        self.layer = 'returns'
        self.synthetic = []                 # assets without a history, added by add_factor
        self.completion_report = None       # given / completed correlations of the synthetic assets
        self.freq = None
        if self.history is not None and isinstance(self.history.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            try:
                self.freq = pd.infer_freq(self.history.index) if isinstance(self.history.index, pd.DatetimeIndex) else self.history.index.freqstr
            except (TypeError, ValueError):
                self.freq = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_history(cls, returns, assets=None):
        """All targets from a sample of returns (any frequency): mean, volatility,
        skewness, kurtosis and correlation matrix of the columns."""
        h = pd.DataFrame(returns).astype(float).ffill().dropna()
        if assets is not None:
            h = h.loc[:, list(assets)]
        t = cls(mean=h.mean(), vol=h.std(ddof=1), corr=h.corr(), skew=h.skew(), kurt=h.kurtosis() + 3.0,
                history=h, assets=list(h.columns))
        t.higher_moments_source = 'history'
        return t

    @classmethod
    def from_ltcma(cls, table, assets=None, history=None, mean_col='Arithmetic Mean', vol_col='Volatility',
                   skew=None, kurt=None):
        """
        Targets from a capital-market-assumptions table: a DataFrame or a CSV /
        Excel path, indexed by asset (or with an 'Assets' column), with a mean
        column, a volatility column, and the correlation-matrix columns named
        after the assets. Skewness and kurtosis come from ``history`` if given,
        from ``skew``/``kurt`` if given, else default to normal.
        """
        if isinstance(table, str):
            df = pd.read_excel(table) if table.lower().endswith(('.xlsx', '.xls')) else pd.read_csv(table)
        else:
            df = pd.DataFrame(table)
        if 'Assets' in df.columns:
            df = df.set_index('Assets')
        if assets is None:
            assets = [a for a in df.index if a in df.columns]
        df = df.loc[assets]
        corr = df.loc[assets, assets]
        return cls(mean=df[mean_col], vol=df[vol_col], corr=corr, skew=skew, kurt=kurt, history=history, assets=list(assets))

    def add_factor(self, name, mean, vol, corr, skew=0.0, kurt=3.0, annualized=True, periods=12):
        """
        A copy with one more asset that has no history: a factor (or asset) known only
        by its mean, volatility, and correlation with *some* of the existing assets.

        ``corr`` is ``{existing asset: correlation}``; the correlations not given are
        filled by :func:`complete_correlation` (maximum determinant: the new asset
        relates to the unnamed ones only through the named ones). ``mean`` and ``vol``
        are annual by default and converted to the frequency of the other targets;
        ``skew`` and ``kurt`` default to the normal. The copy has ``synthetic``
        extended with ``name``, the same history (the new column has none), and a
        ``completion_report`` with one row per existing asset: correlation and
        ``source`` (``given`` or ``completed``). :class:`~cvinemarketgen.markets.CVineMarket`
        attaches such an asset to the fitted vine by a Gaussian conditional draw;
        :class:`~cvinemarketgen.markets.FleishmanMarket` needs nothing more.
        """
        name = str(name)
        if name in self.assets:
            raise ValueError(f'add_factor: {name!r} already in the targets')
        corr = {str(k): float(v) for k, v in dict(corr).items()}
        bad = [k for k in corr if k not in self.assets]
        if bad:
            raise ValueError(f'add_factor: unknown assets in corr {bad}')
        if any(abs(v) >= 1 for v in corr.values()):
            raise ValueError('add_factor: correlations must lie strictly inside (-1, 1)')
        assets = self.assets + [name]
        C = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
        C.loc[self.assets, self.assets] = self.corr.values
        G = pd.DataFrame(True, index=assets, columns=assets)
        G.loc[name, :] = False; G.loc[:, name] = False; G.loc[name, name] = True
        for k, v in corr.items():
            C.loc[name, k] = C.loc[k, name] = v; G.loc[name, k] = G.loc[k, name] = True
        C = complete_correlation(C, G)
        m = float(mean) / periods if annualized else float(mean)
        v = float(vol) / np.sqrt(periods) if annualized else float(vol)
        t = Targets(mean=pd.concat([self.mean, pd.Series({name: m})]), vol=pd.concat([self.vol, pd.Series({name: v})]),
                    corr=C, skew=pd.concat([self.skew, pd.Series({name: float(skew)})]),
                    kurt=pd.concat([self.kurt, pd.Series({name: float(kurt)})]), assets=assets)
        t.history = None if self.history is None else self.history.copy()
        t.layer, t.freq, t.higher_moments_source = self.layer, self.freq, self.higher_moments_source
        t.synthetic = list(self.synthetic) + [name]
        rep = pd.DataFrame({'correlation': C.loc[name, self.assets],
                            'source': ['given' if a in corr else 'completed' for a in self.assets]})
        rep.index = pd.MultiIndex.from_product([[name], self.assets], names=['new', 'with'])
        t.completion_report = rep if self.completion_report is None else pd.concat([self.completion_report, rep])
        return t

    @property
    def historical(self):
        """The assets that have a history (all but ``synthetic``)."""
        return [a for a in self.assets if a not in self.synthetic]

    # ------------------------------------------------------------------ views
    @property
    def moments(self):
        """DataFrame with columns mean, vol, skew, kurt, one row per asset."""
        return pd.concat([self.mean, self.vol, self.skew, self.kurt], axis=1)

    def summary(self, digits=4):
        """Print the moments table and the correlation matrix; return the moments table."""
        n_hist = 0 if self.history is None else len(self.history)
        note = {'given': 'skewness and kurtosis: given',
                'history': f'skewness and kurtosis: estimated from the history ({n_hist} observations)',
                'default (normal)': 'skewness and kurtosis: not given, set to 0 and 3 (normal marginals)'}[self.higher_moments_source]
        print(f'Targets for {len(self.assets)} assets ({self.layer} layer'
              + (f', history frequency {self.freq}' if self.freq else '') + ')')
        print(self.moments.round(digits).to_string())
        print('\ncorrelation matrix:')
        print(self.corr.round(2).to_string())
        print('\n' + note)
        return self.moments

    def to_setup_frame(self):
        """The LTCMA-layout DataFrame consumed by the engine (mean, vol, correlation columns)."""
        out = pd.DataFrame(index=pd.Index(self.assets, name='Assets'))
        out['Arithmetic Mean'] = self.mean.values
        out['Volatility'] = self.vol.values
        for a in self.assets:
            out[a] = self.corr[a].values
        return out

    def higher_moments_frame(self):
        """DataFrame with 'Skewness' and 'Kurtosis' columns, as the engine expects."""
        return pd.DataFrame({'Skewness': self.skew.values, 'Kurtosis': self.kurt.values}, index=self.assets)

    def to_dict(self):
        """JSON-serializable representation (history excluded)."""
        return {'assets': self.assets, 'mean': self.mean.tolist(), 'vol': self.vol.tolist(),
                'skew': self.skew.tolist(), 'kurt': self.kurt.tolist(), 'corr': self.corr.values.tolist(),
                'layer': self.layer, 'freq': self.freq, 'higher_moments_source': self.higher_moments_source,
                'synthetic': list(self.synthetic),
                'completion_report': None if self.completion_report is None else
                {'index': [list(i) for i in self.completion_report.index], 'correlation': self.completion_report['correlation'].tolist(),
                 'source': self.completion_report['source'].tolist()}}

    @classmethod
    def from_dict(cls, d):
        t = cls(mean=d['mean'], vol=d['vol'], corr=np.array(d['corr']), skew=d['skew'], kurt=d['kurt'], assets=d['assets'])
        t.layer = d.get('layer', 'returns')
        t.freq = d.get('freq')
        t.higher_moments_source = d.get('higher_moments_source', 'given')
        t.synthetic = list(d.get('synthetic', []))
        cr = d.get('completion_report')
        if cr:
            t.completion_report = pd.DataFrame({'correlation': cr['correlation'], 'source': cr['source']},
                                               index=pd.MultiIndex.from_tuples([tuple(i) for i in cr['index']], names=['new', 'with']))
        return t

    def __repr__(self):
        return f"Targets({len(self.assets)} assets: {', '.join(self.assets[:4])}{', ...' if len(self.assets) > 4 else ''}; layer={self.layer})"
