import json
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.bridge import Bridge, growth_to_level, recession_probability
from cvinemarketgen.paths import Paths


def _synthetic(n_months=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range('1950-01', periods=n_months, freq='M')
    a = np.zeros(n_months); b = np.zeros(n_months)
    for t in range(1, n_months):
        a[t] = 0.7 * a[t - 1] + rng.standard_normal(); b[t] = 0.5 * b[t - 1] + rng.standard_normal()
    X = pd.DataFrame({'a': a, 'b': b}, index=idx)
    Xq = Bridge.quarterly_means(X)
    g = pd.Series(0.0, index=Xq.index)
    for i in range(1, len(g)):
        g.iloc[i] = 2.0 + 1.5 * Xq['a'].iloc[i] - 0.8 * Xq['b'].iloc[i] + 0.3 * g.iloc[i - 1] + 1.0 * rng.standard_normal()
    return X, g


def test_fit_recovers_coefficients():
    X, g = _synthetic()
    m = Bridge(['a', 'b'], lags=1).fit(g, X)
    assert abs(m.const - 2.0) < 0.3 and abs(m.gamma['a'] - 1.5) < 0.1 and abs(m.gamma['b'] + 0.8) < 0.1
    assert abs(m.phi[0] - 0.3) < 0.1 and abs(m.sigma - 1.0) < 0.15 and m.r2 > 0.7
    assert m.n_obs == len(g) - 1 and abs(m.resid.mean()) < 1e-8 and abs(m.resid.std() - 1) < 0.05
    assert list(m.report.index) == ['const', 'a', 'b', 'g_l1'] and abs(m.tstat['a']) > 10
    m2 = Bridge.from_dict(json.loads(json.dumps(m.to_dict())))
    assert m2.const == m.const and list(m2.gamma) == list(m.gamma) and m2.phi == m.phi


def test_quarterly_means_drop_incomplete_quarters():
    X = pd.DataFrame({'a': np.arange(8.0)}, index=pd.period_range('2020-01', periods=8, freq='M'))
    Q = Bridge.quarterly_means(X)
    assert list(Q.index.astype(str)) == ['2020Q1', '2020Q2'] and Q['a'].tolist() == [1.0, 4.0]


def test_simulate_shapes_level_and_recession():
    X, g = _synthetic()
    m = Bridge(['a', 'b'], lags=1).fit(g, X)
    P = Paths(np.zeros((50, 26, 3)), ['a', 'b', 'other'])                  # parents at zero, 26 months -> 8 quarters
    G = m.simulate(P, g.values, seed=1)
    assert isinstance(G, Paths) and G.assets == ['gdp'] and G.array.shape == (50, 8, 1)
    L = growth_to_level(G)
    assert L.shape == (50, 8) and np.allclose(L[:, 0], 100 * np.exp(G.array[:, 0, 0] / 400))
    pos = Paths(np.full((10, 8, 1), 2.0), ['gdp']); neg = Paths(np.full((10, 8, 1), -1.0), ['gdp'])
    p_pos, by_pos = recession_probability(pos); p_neg, by_neg = recession_probability(neg)
    assert p_pos == 0.0 and p_neg == 1.0 and by_neg[1] == 1.0 and by_neg.sum() == 1.0
    mixed = Paths(np.array([[1, -1, -1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, -1, 1]], float)[:, :, None], ['gdp'])
    p, by = recession_probability(mixed)
    assert p == 0.5 and by[2] == 0.5
    with pytest.raises(ValueError):
        m.simulate(Paths(np.zeros((2, 2, 3)), ['a', 'b', 'other']), g.values)
