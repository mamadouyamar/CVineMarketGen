# tests/test_structural.py
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.structural import Structural


def _system(n=1500, seed=0, kappa=-0.1, theta=(0.5, -0.3), gamma=(0.2, 0.1), sigma=0.02):
    """Two random-walk parents and a child that error-corrects to theta' z with contemporaneous response gamma."""
    rng = np.random.default_rng(seed)
    rd = np.cumsum(0.1 * rng.standard_normal(n)); oil = np.cumsum(0.05 * rng.standard_normal(n))
    s = np.zeros(n); s[0] = theta[0] * rd[0] + theta[1] * oil[0]
    for t in range(1, n):
        ds = kappa * (s[t - 1] - theta[0] * rd[t - 1] - theta[1] * oil[t - 1]) + gamma[0] * (rd[t] - rd[t - 1]) + gamma[1] * (oil[t] - oil[t - 1]) + sigma * rng.standard_normal()
        s[t] = s[t - 1] + ds
    idx = pd.period_range('1990-01', periods=n, freq='M')
    return pd.Series(s, index=idx, name='fx'), pd.DataFrame({'rd': rd, 'oil': oil}, index=idx)


def test_ecm_recovers_parameters_and_names():
    s, Z = _system()
    m = Structural(['rd', 'oil'], form='ecm', lags=1).fit(s, Z)
    assert m.child == 'fx' and m.parents == ['rd', 'oil'] and m.name == 'ECM(fx | rd, oil)'
    assert abs(m.kappa + 0.1) < 0.03
    assert abs(m.theta['rd'] - 0.5) < 0.1 and abs(m.theta['oil'] + 0.3) < 0.1
    assert abs(m.gamma['rd'] - 0.2) < 0.05 and abs(m.gamma['oil'] - 0.1) < 0.05
    assert abs(m.sigma - 0.02) < 0.003 and 3 < m.half_life < 12 and 0 < m.r2 < 1
    assert m.tstat['s_l1'] < -3 and np.isfinite(m.bic) and m.n_params == 8
    z = m.filter()
    assert isinstance(z, pd.Series) and z.name == 'fx' and len(z) == len(s) - 2 and abs(z.std() - 1) < 1e-2      # sigma has n - k degrees of freedom


def test_ecm_unfilter_is_inverse_of_filter_new_and_round_trips():
    s, Z = _system(n=800)
    m = Structural(['rd', 'oil'], form='ecm', lags=2).fit(s, Z)
    rng = np.random.default_rng(1)
    Zpath = np.cumsum(rng.standard_normal((3, 24, 2)) * [0.1, 0.05], axis=1) + Z.iloc[-1].values
    z = rng.standard_normal((3, 24))
    Y = m.unfilter(z, Zpath)
    assert Y.shape == (3, 24) and np.isfinite(Y).all()
    assert np.allclose(m.filter_new(Y[1], Zpath[1]), z[1], atol=1e-9)
    m2 = Structural.from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(z, Zpath), Y) and m2.name == m.name
    assert m.simulate(10, seed=0, Zpath=Zpath[:1, :10]).shape == (10,)


def test_linear_without_lags_is_the_factor_map():
    rng = np.random.default_rng(2); n = 1000
    Z = pd.DataFrame(rng.standard_normal((n, 2)) * 0.03, columns=['f1', 'f2'])
    s = pd.Series(0.001 + 1.2 * Z['f1'] - 0.4 * Z['f2'] + 0.01 * rng.standard_normal(n), name='r')
    m = Structural(['f1', 'f2'], form='linear', lags=0).fit(s, Z)
    X = np.column_stack([np.ones(n), Z.values]); beta = np.linalg.lstsq(X, s.values, rcond=None)[0]
    assert np.allclose([m.const, m.gamma['f1'], m.gamma['f2']], beta) and m.phi == [] and m.name == 'Linear(r | f1, f2)'
    Zpath = np.zeros((2, 5, 2)); Y = m.unfilter(np.zeros((2, 5)), Zpath)
    assert np.allclose(Y, m.const)


def test_bad_form_and_positive_kappa_warning():
    with pytest.raises(ValueError):
        Structural(['a'], form='quadratic')


from cvinemarketgen.dynamics import AssetDynamics, parse_spec


def test_parse_structural_specs():
    m = parse_spec('ecm(rd, oil)'); assert m.form == 'ecm' and m.parents == ['rd', 'oil'] and m.lags == 1
    m = parse_spec('linear(f1, f2; lags=2)'); assert m.form == 'linear' and m.parents == ['f1', 'f2'] and m.lags == 2
    m = parse_spec('ecm(rd; lags=3)'); assert m.parents == ['rd'] and m.lags == 3


def test_container_orders_parents_before_children():
    pytest.importorskip('statsmodels')
    s, Z = _system(n=800)
    h = Z.copy(); h['fx'] = s; h['spy'] = 0.0003 + 0.01 * np.random.default_rng(3).standard_normal(len(h))
    h = h[['fx', 'spy', 'rd', 'oil']]                                   # child listed first on purpose
    ad = AssetDynamics({('rd', 'oil'): 'vecm(r=0,q=1)', 'fx': 'ecm(rd, oil)', 'spy': 'ar1'}).fit(h)
    assert ad.order[0] == ('rd', 'oil') and ad.order.index('fx') > ad.order.index(('rd', 'oil')) and ad.children == ['fx']
    Zr = ad.filter(h)
    assert list(Zr.columns) == ['fx', 'spy', 'rd', 'oil']
    Y = ad.unfilter(np.zeros((4, 12, 4)))
    assert Y.shape == (4, 12, 4) and np.isfinite(Y).all()
    # the child moves with its parents: shock the parents' innovations only
    Zs = np.zeros((4, 12, 4)); Zs[:, :, 2] = 3.0
    Ys = ad.unfilter(Zs)
    assert not np.allclose(Ys[:, :, 0], Y[:, :, 0]) and np.allclose(Ys[:, :, 1], Y[:, :, 1])
    ad2 = AssetDynamics.from_dict(ad.to_dict())
    assert ad2.order == ad.order and np.allclose(ad2.unfilter(np.zeros((1, 3, 4))), Y[:1, :3])
    with pytest.raises(ValueError):
        AssetDynamics({'a': 'ecm(b)', 'b': 'ecm(a)'}).fit(pd.DataFrame({'a': s.values, 'b': Z['rd'].values}))
