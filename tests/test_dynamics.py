import numpy as np
import pandas as pd
import pytest

arch = pytest.importorskip('arch')
from cvinemarketgen.dynamics import GarchFamily, AssetDynamics, parse_spec
from cvinemarketgen.selection import cvm_statistic


def _garch_series(n=2500, seed=0, a=0.0002, b=0.05, om=2e-6, al=0.08, be=0.90):
    rng = np.random.default_rng(seed)
    y = np.zeros(n); e = np.zeros(n); h = np.full(n, om / (1 - al - be))
    for t in range(1, n):
        h[t] = om + al * e[t - 1] ** 2 + be * h[t - 1]
        e[t] = np.sqrt(h[t]) * rng.standard_normal()
        y[t] = a + b * y[t - 1] + e[t]
    return pd.Series(y, index=pd.bdate_range('2015-01-01', periods=n), name='X')


@pytest.mark.parametrize('spec', ['const-const', 'ar1', 'const-garch(1,1)', 'ar1-garch(2,1)', 'ar1-gjr(1,1)', 'const-egarch(1,2)'])
def test_filter_unfilter_are_inverse(spec):
    y = _garch_series()
    m = parse_spec(spec).fit(y)
    z = m.filter()
    assert isinstance(z, pd.Series) and np.isfinite(z.values).all() and len(z) >= len(y) - 1
    rng = np.random.default_rng(1)
    znew = rng.standard_normal((3, 40))
    ynew = m.unfilter(znew)
    assert ynew.shape == (3, 40)
    back = m.filter_new(ynew[1])
    assert np.allclose(back, znew[1], atol=1e-9)
    expected = (1 if m.mean == 'const' else 2) + (1 if m.vol == 'const' else 1 + m.p + m.q + (1 if m.vol == 'gjr' else 0))
    assert m.n_params == expected and np.isfinite(m.bic)


def test_garch_recovers_parameters_and_names():
    y = _garch_series()
    m = GarchFamily('ar1', 'garch', 1, 1).fit(y)
    assert m.name == 'AR(1)-GARCH(1,1)'
    assert abs(m.params['alpha'][0] - 0.08) < 0.04 and abs(m.params['beta'][0] - 0.90) < 0.05
    s = m.simulate(500, seed=0)
    assert s.shape == (500,) and abs(s.std() / y.std() - 1) < 0.5
    d = m.to_dict(); m2 = GarchFamily.from_dict(d)
    assert np.allclose(m2.unfilter(np.ones((1, 5))), m.unfilter(np.ones((1, 5))))


def test_asset_dynamics_container():
    h = pd.DataFrame({'A': _garch_series(seed=1).values, 'B': _garch_series(seed=2, b=0.0).values},
                     index=pd.bdate_range('2015-01-01', periods=2500))
    ad = AssetDynamics({'A': parse_spec('ar1-garch(1,1)'), 'B': parse_spec('const-gjr(1,1)')}).fit(h)
    Z = ad.filter(h)
    assert list(Z.columns) == ['A', 'B'] and Z.shape[0] == 2499
    Y = ad.unfilter(np.zeros((4, 10, 2)))
    assert Y.shape == (4, 10, 2)
    assert list(ad.report['model']) == ['AR(1)-GARCH(1,1)', 'Const-GJR(1,1)']
    ad2 = AssetDynamics.from_dict(ad.to_dict())
    assert np.allclose(ad2.unfilter(np.zeros((1, 3, 2))), Y[:1, :3])


# ---- Gaussian HMM (Task 6) -------------------------------------------------------
from cvinemarketgen.hmm import GaussianHMM


def _hmm_ref():
    import json, os
    return json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'genhmm1d_reference.json')))


def test_hmm_matches_genhmm1d_reference():
    ref = _hmm_ref()
    y = pd.Series(ref['y'])
    m = GaussianHMM(2).fit(y)
    theta = np.array(ref['theta'])
    assert np.allclose(m.mu, theta[:, 0], atol=1e-4) and np.allclose(m.sigma, theta[:, 1], atol=1e-4)
    # the reference was produced by a direct GenHMM1d call (percentiles=[0.5], eps=1e-12); the wrapper
    # starts from the median split with eps=1e-6, so the EM end points differ by a few 1e-4
    assert np.allclose(m.Q, np.array(ref['Q']), atol=5e-4)
    assert np.allclose(m.uniforms().values, np.array(ref['U']), atol=5e-4)
    assert abs(cvm_statistic(m.uniforms().values) - ref['cvm']) < 1e-3 and abs(m.cvm - ref['cvm']) < 1e-3
    assert abs(m.loglik - ref['LL']) < 1e-3


def test_hmm_filter_unfilter_inverse_and_persistence():
    ref = _hmm_ref()
    m = GaussianHMM(2).fit(pd.Series(ref['y']))
    assert m.name == 'HMM(2)' and m.n_params == 6 and m.regimes.shape == (2000, 2)
    znew = np.random.default_rng(3).standard_normal((2, 30))
    ynew = m.unfilter(znew)
    assert np.allclose(m.filter_new(ynew[0]), znew[0], atol=1e-8)
    m2 = GaussianHMM.from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(znew), ynew)
    assert m.simulate(100, seed=0).shape == (100,)


def test_selector_picks_hmm_on_hmm_data():
    from cvinemarketgen.selection import select_dynamics
    ref = _hmm_ref()
    y = pd.DataFrame({'H': ref['y']})
    ad = select_dynamics(y, candidates=('const', 'hmm'), states=(2, 3), gof=False, verbose=False)
    assert ad.report.loc['H', 'model'] in ('HMM(2)', 'HMM(3)')


def test_garch_family_loglik_is_on_the_data_scale():
    # arch fits returns scaled by 100; the reported log-likelihood and BIC must be on the data's own scale,
    # so that they are comparable with the HMM's
    y = _garch_series(n=1500)
    m = parse_spec('const-const').fit(y)
    v = y.values
    expected = -0.5 * len(v) * (np.log(2 * np.pi * np.mean((v - v.mean()) ** 2)) + 1)   # Gaussian MLE log-likelihood
    assert abs(m.loglik - expected) < 0.5
    assert abs(m.bic - (-2 * m.loglik + 2 * np.log(len(v)))) < 1e-6
    m2 = parse_spec('const-garch(1,1)').fit(y)
    assert m2.loglik > m.loglik and m2.bic < m.bic


def test_hmm_fits_heavy_tailed_data_with_three_states():
    # extreme days underflow the density of a narrow state: not an error
    rng = np.random.default_rng(5)
    y = pd.Series(rng.standard_t(2.5, 3000) * 0.01)
    m = GaussianHMM(3).fit(y)
    assert np.isfinite(m.loglik) and m.regimes.shape == (3000, 3) and np.isfinite(m.filter().values).all()
    assert np.allclose(m.Q.sum(1), 1.0) and (m.sigma > 0).all()


def test_hmm_refit_leaves_the_parent_unchanged():
    # GenHMM1d's EM writes into the starting matrix it is given: the warm start must be a copy
    ref = _hmm_ref()
    m = GaussianHMM(2).fit(pd.Series(ref['y']))
    Q0, mu0, s0, eta0 = m.Q.copy(), m.mu.copy(), m.sigma.copy(), m.eta_T.copy()
    m.refit(pd.Series(m.simulate(2000, seed=0)))
    assert np.array_equal(m.Q, Q0) and np.array_equal(m.mu, mu0) and np.array_equal(m.sigma, s0) and np.array_equal(m.eta_T, eta0)
