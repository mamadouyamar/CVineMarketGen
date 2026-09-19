import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.yieldcurve import (NelsonSiegel, PCACurve, yield_at, discount, bond_price, par_yield,
                                        zero_return, constant_maturity_return, curve_returns)
from cvinemarketgen.paths import Paths

TAU = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]


def _synthetic(n=300, seed=0, noise=0.005):
    rng = np.random.default_rng(seed)
    level = 4.0 + np.cumsum(0.1 * rng.standard_normal(n))
    slope = -1.5 + np.cumsum(0.1 * rng.standard_normal(n))
    curv = -2.0 + np.cumsum(0.15 * rng.standard_normal(n))
    F = pd.DataFrame({'level': level, 'slope': slope, 'curvature': curv},
                     index=pd.period_range('2000-01', periods=n, freq='M'))
    ns = NelsonSiegel(0.7308)
    Y = pd.DataFrame(F.values @ ns.loadings(TAU).T + noise * rng.standard_normal((n, len(TAU))),
                     index=F.index, columns=[str(t) for t in TAU])
    return F, Y


def test_loadings_shape_and_limits():
    ns = NelsonSiegel(0.7308)
    L = ns.loadings(TAU)
    assert L.shape == (8, 3) and np.allclose(L[:, 0], 1.0)
    small = ns.loadings([1e-6])[0]
    assert abs(small[1] - 1.0) < 1e-4 and abs(small[2]) < 1e-4
    big = ns.loadings([1000.0])[0]
    assert abs(big[1]) < 1e-2 and abs(big[2]) < 1e-2


def test_fit_recovers_factors_and_curve():
    F, Y = _synthetic()
    ns = NelsonSiegel(0.7308).fit(Y)
    assert list(ns.factors.columns) == ['level', 'slope', 'curvature']
    assert ns.tau == TAU and ns.columns == list(Y.columns)
    assert np.abs(ns.factors.values - F.values).max() < 0.15 and (ns.rmse < 0.01).all()   # curvature is the least identified
    back = ns.curve(ns.factors)
    assert list(back.columns) == list(Y.columns) and np.abs(back.values - Y.values).max() < 0.1
    assert ns.curve(ns.factors.iloc[:5], tau=[4.0, 15.0]).shape == (5, 2)
    auto = NelsonSiegel('auto').fit(Y)
    assert 0.2 <= auto.lam <= 2.0
    ns2 = NelsonSiegel.from_dict(ns.to_dict())
    assert ns2.lam == ns.lam and ns2.tau == ns.tau
    assert np.allclose(ns2.curve(ns.factors.iloc[:3]).values, back.values[:3])


def test_fit_with_fred_names():
    F, Y = _synthetic()
    Y.columns = ['DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7', 'DGS10', 'DGS20', 'DGS30']
    ns = NelsonSiegel(0.7308).fit(Y)
    assert ns.tau == TAU
    ns2 = NelsonSiegel(0.7308).fit(Y, maturities={c: t for c, t in zip(Y.columns, TAU)})
    assert ns2.tau == TAU


def test_pca_curve():
    F, Y = _synthetic()
    pc = PCACurve(3).fit(Y)
    assert list(pc.factors.columns) == ['pc1', 'pc2', 'pc3']
    assert pc.explained.iloc[:3].sum() > 0.99
    assert np.abs(pc.curve(pc.factors).values - Y.values).max() < 0.1
    P = Paths(np.tile(pc.factors.values[-1], (2, 3, 1)), list(pc.factors.columns))
    C = pc.curve(P)
    assert isinstance(C, Paths) and C.assets == list(Y.columns)
    assert np.allclose(C.array[0, 0], pc.curve(pc.factors.iloc[[-1]]).values[0])


def test_pricing_on_a_flat_curve():
    ns = NelsonSiegel(0.7308)
    F = np.array([[5.0, 0.0, 0.0]])                        # flat curve at 5 percent
    assert np.allclose(yield_at(F, [1.0, 10.0], ns), 5.0)
    assert np.allclose(discount(5.0, 2.0), np.exp(-0.10))
    c = par_yield(F, 10.0, ns, freq=2)
    assert np.allclose(bond_price(F, c, 10.0, ns, freq=2), 100.0, atol=1e-8)
    assert c[0] > 5.0                                      # semi-annual coupon above the continuous rate
    r0 = zero_return(F, F, 10.0, ns, dt=1 / 12)
    assert np.allclose(r0, np.exp(0.05 / 12) - 1, atol=1e-10)
    r_flat = constant_maturity_return(F, F, 10.0, ns, dt=1 / 12)
    assert abs(r_flat[0] - r0[0]) < 5e-4                    # flat curve: par bond earns about the rate
    F_up = np.array([[5.0, -1.0, 0.0]])                    # upward sloping, unchanged: roll-down adds to carry
    r_up = constant_maturity_return(F_up, F_up, 10.0, ns, dt=1 / 12)
    carry = par_yield(F_up, 10.0, ns)[0] / 100.0 / 12
    assert r_up[0] > carry
    F_dn = np.array([[5.0, 1.0, 0.0]])                     # inverted, unchanged: roll-down costs
    assert constant_maturity_return(F_dn, F_dn, 10.0, ns, dt=1 / 12)[0] < par_yield(F_dn, 10.0, ns)[0] / 100.0 / 12


def test_curve_returns_along_paths():
    ns = NelsonSiegel(0.7308)
    F0 = np.array([4.0, -1.0, -2.0])
    P = Paths(np.tile(F0, (5, 12, 1)), ['level', 'slope', 'curvature'])
    R = curve_returns(P, [2.0, 10.0], ns, F0, kind='par')
    assert isinstance(R, Paths) and R.assets == ['2y', '10y'] and R.array.shape == (5, 12, 2)
    assert np.allclose(R.array[0], R.array[4]) and np.allclose(R.array[:, 0, :], R.array[:, 5, :])
    Rz = curve_returns(P, [2.0], ns, F0, kind='zero')
    assert np.allclose(Rz.array[:, 0, 0], zero_return(F0[None], F0[None], 2.0, ns)[0])
    # a rise in the level lowers bond prices along the path
    P2 = Paths(np.tile(F0 + np.array([1.0, 0.0, 0.0]), (1, 1, 1)), ['level', 'slope', 'curvature'])
    R2 = curve_returns(P2, [10.0], ns, F0)
    assert R2.array[0, 0, 0] < -0.05
    with pytest.raises(ValueError):
        curve_returns(P, [2.0], ns, F0, kind='other')
