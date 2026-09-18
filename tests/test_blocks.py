# tests/test_blocks.py
import numpy as np
import pandas as pd
import pytest

sm = pytest.importorskip('statsmodels')
from cvinemarketgen.blocks import BlockVECM, BlockVAR, block_from_dict


def _cointegrated(n=800, seed=0):
    """x1 random walk, x2 = x1 + AR(1) noise, x3 random walk: rank 1, beta ~ (1, -1, 0)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 3)) * [0.2, 0.1, 0.3]
    x1 = np.cumsum(e[:, 0]); u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.5 * u[t - 1] + e[t, 1]
    x3 = np.cumsum(e[:, 2])
    return pd.DataFrame({'a': x1, 'b': x1 + u, 'c': x3}, index=pd.period_range('1960-01', periods=n, freq='M'))


def test_vecm_finds_rank_and_cointegrating_vector():
    X = _cointegrated()
    m = BlockVECM().fit(X)
    assert m.rank == 1 and m.lags >= 1 and m.columns == ['a', 'b', 'c']
    b = m.beta[:, 0] / m.beta[0, 0]
    assert abs(b[1] + 1) < 0.05 and abs(b[2]) < 0.05
    assert m.name == f'VECM(r=1, q={m.lags})' and np.isfinite(m.bic) and m.n_params > 0
    z = m.filter()
    assert list(z.columns) == ['a', 'b', 'c'] and len(z) == len(X) - m.lags - 1
    assert np.allclose(z.std(ddof=1).values, 1.0, atol=1e-6)


def test_vecm_unfilter_is_inverse_of_filter_new_and_round_trips():
    X = _cointegrated()
    m = BlockVECM(rank=1, lags=2).fit(X)
    Z = np.random.default_rng(1).standard_normal((4, 30, 3))
    Y = m.unfilter(Z)
    assert Y.shape == (4, 30, 3) and np.isfinite(Y).all()
    assert np.allclose(m.filter_new(Y[2]), Z[2], atol=1e-9)
    m2 = block_from_dict(m.to_dict())
    assert np.allclose(m2.unfilter(Z), Y) and m2.name == m.name and m2.columns == m.columns
    s = m.simulate(50, seed=0)
    assert s.shape == (50, 3)


def test_var_recovers_coefficients_on_returns():
    rng = np.random.default_rng(2); n = 2000
    A = np.array([[0.3, 0.1], [0.0, 0.2]]); X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = A @ X[t - 1] + rng.standard_normal(2) * 0.01
    m = BlockVAR(lags=1).fit(pd.DataFrame(X, columns=['u', 'v']))
    assert np.allclose(m.Gamma[0], A, atol=0.05) and m.name == 'VAR(1)' and m.state.shape == (1, 2)
    Z = np.zeros((1, 5, 2))
    Y = m.unfilter(Z)
    assert np.allclose(m.filter_new(Y[0]), 0.0, atol=1e-9)


def test_statsmodels_missing_message(monkeypatch):
    import cvinemarketgen.blocks as b
    def boom():
        raise ImportError('block dynamics need statsmodels: pip install statsmodels')
    monkeypatch.setattr(b, '_sm', boom)
    with pytest.raises(ImportError, match='statsmodels'):
        BlockVAR(lags=1).fit(pd.DataFrame({'u': [0.0, 1.0, 0.5, 0.2], 'v': [0.0, 0.5, 0.1, 0.3]}))
