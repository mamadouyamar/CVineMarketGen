"""Tests of the user-facing layer: Targets, markets, dynamics, paths, functions, save/load."""
import json
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen import (Targets, FleishmanMarket, CVineMarket, Paths, AR1,
                            fit_johnson_su, johnson_su_moments, johnson_su_sample, fit_fleishman,
                            exceedance_curve, classify_pair, select_family, partial_correlations)

ASSETS = ['Equity', 'Bonds', 'Gold']
MEAN = {'Equity': 0.07, 'Bonds': 0.03, 'Gold': 0.04}
VOL = {'Equity': 0.16, 'Bonds': 0.05, 'Gold': 0.15}
CORR = [[1, -0.1, 0.05], [-0.1, 1, 0.3], [0.05, 0.3, 1]]


def _synthetic_history(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3)) @ np.linalg.cholesky([[1, .5, .2], [.5, 1, .1], [.2, .1, 1]]).T
    r = np.zeros_like(z); r[0] = z[0] * 0.01
    for i in range(1, n):
        r[i] = 0.0002 + 0.15 * r[i - 1] + 0.01 * z[i]
    return pd.DataFrame(r, columns=['A', 'B', 'C'], index=pd.bdate_range('2018-01-01', periods=n))


# ---------------------------------------------------------------- Targets
def test_targets_case_a_defaults_to_normal():
    t = Targets(mean=MEAN, vol=VOL, corr=CORR, assets=ASSETS)
    assert t.assets == ASSETS and (t.skew == 0).all() and (t.kurt == 3).all()
    assert t.higher_moments_source == 'default (normal)'
    assert np.allclose(np.diag(t.corr.values), 1.0) and t.moments.shape == (3, 4)


def test_targets_validation():
    with pytest.raises(ValueError):
        Targets(mean=MEAN, vol={'Equity': 0.16, 'Bonds': -0.05, 'Gold': 0.15}, corr=CORR, assets=ASSETS)
    with pytest.raises(ValueError):
        Targets(mean=MEAN, vol=VOL, corr=CORR, skew={'Equity': 2.0, 'Bonds': 0, 'Gold': 0},
                kurt={'Equity': 3.0, 'Bonds': 3, 'Gold': 3}, assets=ASSETS)
    with pytest.warns(UserWarning):
        Targets(mean=MEAN, vol=VOL, corr=[[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]], assets=ASSETS)


def test_targets_from_history_and_ltcma(tmp_path):
    h = _synthetic_history()
    t = Targets.from_history(h)
    assert t.assets == ['A', 'B', 'C'] and t.higher_moments_source == 'history' and t.freq == 'B'
    assert np.allclose(t.mean.values, h.mean().values) and np.allclose(t.corr.values, h.corr().values)
    table = pd.DataFrame({'Arithmetic Mean': [0.07, 0.03], 'Volatility': [0.16, 0.05],
                          'Equity': [1.0, -0.1], 'Bonds': [-0.1, 1.0]}, index=pd.Index(['Equity', 'Bonds'], name='Assets'))
    p = tmp_path / 'ltcma.csv'; table.to_csv(p)
    t2 = Targets.from_ltcma(str(p))
    assert t2.assets == ['Equity', 'Bonds'] and t2.corr.loc['Equity', 'Bonds'] == -0.1
    d = Targets.from_dict(t2.to_dict())
    assert d.assets == t2.assets and np.allclose(d.corr.values, t2.corr.values)


# ---------------------------------------------------------------- Fleishman market
def test_fleishman_market_case_a():
    t = Targets(mean=MEAN, vol=VOL, corr=CORR, assets=ASSETS)
    fm = FleishmanMarket(t).fit()
    assert fm.infeasible_pairs == [] and fm.coefficients.shape == (3, 4)
    X = fm.simulate(20000, seed=1)
    d = fm.diagnostics(X)
    assert list(X.columns) == ASSETS and d.max_abs_corr_error < 0.05
    assert d.moments['mean diff'].abs().max() < 1e-6 and d.moments['vol diff'].abs().max() < 1e-4


# ---------------------------------------------------------------- C-vine market
def test_cvine_market_gaussian_and_roundtrip(tmp_path):
    t = Targets(mean=MEAN, vol=VOL, corr=CORR, assets=ASSETS)
    cv = CVineMarket(t, families='gaussian', n_opt=2000).fit()
    e = cv.edges
    assert len(e) == 3 and (e['selected family'] == 'gaussian 0°').all()
    assert cv.marginals.loc['Equity', 'delta'] == 1e4          # exact normal marginal
    X = cv.simulate(10000, seed=1)
    assert cv.diagnostics(X).max_abs_corr_error < 0.05
    p = tmp_path / 'm.json'; cv.save(str(p))
    assert json.load(open(p))['kind'] == 'CVineMarket'
    cv2 = CVineMarket.load(str(p))
    assert cv2.edges.equals(cv.edges)
    X2 = cv2.simulate(5000, seed=1, accept=False)
    assert X2.shape == (5000, 3) and list(X2.columns) == ASSETS


def test_cvine_market_user_families_with_mixture():
    t = Targets(mean=MEAN, vol=VOL, corr=CORR, skew={'Equity': -0.6, 'Bonds': 0.1, 'Gold': 0.2},
                kurt={'Equity': 4.5, 'Bonds': 4.0, 'Gold': 3.6}, assets=ASSETS)
    fam = {('Bonds', 'Equity'): ('mixture', [('clayton', 270), ('gumbel', 0)], 0.6), ('Gold', 'Equity'): ('gumbel', 180)}
    cv = CVineMarket(t, families=fam, n_opt=2000).fit()
    e = cv.edges.set_index('edge')
    assert e.loc['Bonds , Equity', 'selected family'] == 'clayton 270° + gumbel 0°'
    assert e.loc['Gold , Equity', 'selected family'] == 'gumbel 180°'
    d = cv.diagnostics(cv.simulate(10000, seed=1))
    assert d.max_abs_corr_error < 0.06 and d.moments['skew diff'].abs().max() < 0.05


def test_cvine_market_auto_needs_history():
    t = Targets(mean=MEAN, vol=VOL, corr=CORR, assets=ASSETS)
    with pytest.raises(ValueError):
        CVineMarket(t, families='auto').fit()


def test_cvine_market_history_ar1_paths():
    h = _synthetic_history()
    t = Targets.from_history(h)
    cv = CVineMarket(t, families='auto', n_opt=2000, dynamics='ar1').fit()
    assert isinstance(cv.dynamics, AR1) and abs(cv.dynamics.params.loc['A', 'b'] - 0.15) < 0.08
    assert cv.fit_targets.layer == 'residuals'
    P = cv.simulate_paths(100, 40, seed=3)
    assert isinstance(P, Paths) and P.array.shape == (100, 40, 3)
    assert P.to_frame().shape == (4000, 3) and P.wide('A').shape == (40, 100) and P.terminal().shape == (100, 3)
    s = P.summary()
    assert np.allclose(s['vol'].values, h.std().values, rtol=0.25)


def test_partial_correlations_reproduce_gaussian():
    corr = pd.DataFrame(CORR, index=ASSETS, columns=ASSETS)
    pc = partial_correlations(corr, ASSETS)
    assert set(pc) == {(2, 1), (3, 1), (3, 2)}
    assert abs(pc[(2, 1)] + 0.1) < 1e-12 and abs(pc[(3, 1)] - 0.05) < 1e-12
    r21, r31 = -0.1, 0.05
    expected = (0.3 - r21 * r31) / np.sqrt((1 - r21 ** 2) * (1 - r31 ** 2))
    assert abs(pc[(3, 2)] - expected) < 1e-12


# ---------------------------------------------------------------- functions
def test_functions():
    p = fit_johnson_su(-0.56, 3.79, mean=0.08, vol=0.16)
    assert p['residual'] < 1e-5 and np.allclose(johnson_su_moments(p), [0, 1, -0.56, 0.79], atol=1e-4)
    assert johnson_su_sample(p, 100, seed=1).shape == (100,)
    f = fit_fleishman(-0.56, 3.79)
    assert f['feasible'] and abs(f['b'] - 0.9503) < 5e-4
    assert not fit_fleishman(0.0, 1.0)['feasible']
    h = _synthetic_history()
    c = exceedance_curve(h['A'], h['B'])
    assert isinstance(c, pd.Series) and len(c) == 41 and np.isfinite(c.values).all()
    cl = classify_pair(h['A'], h['B'])
    assert set(cl) >= {'l', 'u', 's', 'm', 'metrics', 'curve'} and cl['m'] in (-1, 1)
    sel = select_family(h['A'], h['B'])
    assert sel['status'] in ('ordinary', 'mixture') and 'bic' in sel


# ---------------------------------------------------------------- per-asset dynamics (0.3.0)
def test_cvine_market_auto_dynamics_and_roundtrip(tmp_path):
    pytest.importorskip('arch')
    h = _synthetic_history(n=1500)
    t = Targets.from_history(h)
    cv = CVineMarket(t, families='gaussian', n_opt=2000, dynamics='auto',
                     dynamics_kwargs=dict(candidates=('const', 'garch'), pq=(1, 1), gof=False, verbose=False)).fit()
    rep = cv.dynamics_report
    assert list(rep.index) == ['A', 'B', 'C'] and set(rep.columns) >= {'model', 'bic', 'passed'}
    P = cv.simulate_paths(20, 30, seed=1)
    assert P.array.shape == (20, 30, 3) and np.isfinite(P.array).all()
    p = tmp_path / 'auto.json'; cv.save(str(p))
    cv2 = CVineMarket.load(str(p))
    assert list(cv2.dynamics_report['model']) == list(rep['model'])
    assert list(cv2.dynamics.candidates.columns) == list(cv.dynamics.candidates.columns) and len(cv2.dynamics.candidates) == len(cv.dynamics.candidates)
    assert np.allclose(cv2.simulate_paths(3, 5, seed=2).array, cv.simulate_paths(3, 5, seed=2).array)


def test_cvine_market_dict_dynamics_with_hmm():
    pytest.importorskip('arch')
    h = _synthetic_history(n=1500)
    t = Targets.from_history(h)
    cv = CVineMarket(t, families='gaussian', n_opt=2000,
                     dynamics={'A': 'ar1-garch(1,1)', 'B': 'hmm(2)', 'C': 'const'}).fit()
    assert list(cv.dynamics_report['model']) == ['AR(1)-GARCH(1,1)', 'HMM(2)', 'Const-Const']
    assert cv.fit_targets.layer == 'residuals'
    P = cv.simulate_paths(5, 10, seed=0)
    assert P.array.shape == (5, 10, 3) and np.isfinite(P.array).all()


def test_residual_layer_kurtosis_floor_keeps_marginals_feasible():
    pytest.importorskip('arch')
    h = _synthetic_history(n=1500)          # Gaussian innovations: residual kurtosis below 3, outside the Johnson SU range
    t = Targets.from_history(h)
    with pytest.warns(UserWarning, match='kurtosis'):
        cv = CVineMarket(t, families='gaussian', n_opt=2000, dynamics={'A': 'ar1', 'B': 'ar1', 'C': 'ar1'}).fit()
    assert (cv.fit_targets.kurt >= 3.1 - 1e-9).all() and (cv.marginals['fit residual'] < 1e-6).all()
    X = cv.simulate(3000, seed=0)            # accept=True must terminate
    assert np.isfinite(X.values).all()


def test_cvine_market_with_a_block(tmp_path):
    pytest.importorskip('statsmodels')
    from tests.test_blocks import _cointegrated
    X = _cointegrated(n=600)
    X['d'] = 0.0002 + 0.01 * np.random.default_rng(4).standard_normal(len(X))
    t = Targets.from_history(X)
    cv = CVineMarket(t, families='gaussian', n_opt=2000, dynamics={('a', 'b', 'c'): 'vecm(r=1,q=1)', 'd': 'ar1'}).fit()
    assert cv.fit_targets.layer == 'residuals' and list(cv.dynamics_report['model'])[:3] == ['VECM(r=1, q=1)'] * 3
    P = cv.simulate_paths(10, 12, seed=0)
    assert P.array.shape == (10, 12, 4) and np.isfinite(P.array).all()
    assert abs(P.array[:, 0, 0].mean() - X['a'].iloc[-1]) < 1.0         # levels continue from the last observation
    p = tmp_path / 'block.json'; cv.save(str(p))
    cv2 = CVineMarket.load(str(p))
    assert np.allclose(cv2.simulate_paths(2, 4, seed=1).array, cv.simulate_paths(2, 4, seed=1).array)
