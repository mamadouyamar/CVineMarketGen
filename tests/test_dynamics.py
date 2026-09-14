import numpy as np
import pandas as pd
import pytest

arch = pytest.importorskip('arch')
from cvinemarketgen.dynamics import GarchFamily, AssetDynamics, parse_spec


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
