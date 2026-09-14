import json
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen import FactorModel, Paths


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame(rng.standard_normal((n, 3)) * 0.03, columns=['f1', 'f2', 'f3'],
                     index=pd.period_range('2000-01', periods=n, freq='M'))
    B = pd.DataFrame([[1.0, 0.5, 0.0], [0.2, -0.3, 0.8]], index=['A', 'B'], columns=F.columns)
    alpha = pd.Series([0.002, -0.001], index=['A', 'B'])
    eps = rng.standard_normal((n, 2)) * 0.01
    R = pd.DataFrame(alpha.values + F.values @ B.values.T + eps, columns=['A', 'B'], index=F.index)
    return R, F, B, alpha


def test_betas_recovered():
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    assert fm.beta.shape == (2, 3) and fm.n_obs == 240
    assert (np.abs(fm.beta - B) < 2 * fm.se[F.columns]).all().all()
    assert (np.abs(fm.alpha - alpha) < 2 * fm.se['alpha']).all()
    assert (fm.r2 > 0.85).all() and set(fm.report.columns) >= {'alpha', 'f1', 'f2', 'f3', 'R2', 'resid vol'}
    assert list(fm.resid_params.columns) == ['gamma', 'xi', 'delta', 'lambda', 'mean', 'vol', 'residual']


def test_simulate_shapes_and_no_residual_case():
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    Fs = F.iloc[:50]
    X0 = fm.simulate(Fs, residuals=False)
    assert X0.shape == (50, 2) and list(X0.columns) == ['A', 'B']
    assert np.allclose(X0.values, fm.alpha.values + Fs.values @ fm.beta.values.T)
    X1 = fm.simulate(Fs[['f3', 'f1', 'f2']], residuals=True, seed=1)     # columns in another order
    assert X1.shape == (50, 2) and not np.allclose(X1.values, X0.values)
    P = Paths(np.tile(F.values[:12], (7, 1, 1)), list(F.columns))
    PX = fm.simulate(P, seed=2)
    assert isinstance(PX, Paths) and PX.array.shape == (7, 12, 2) and PX.assets == ['A', 'B']


def test_implied_mean_and_roundtrip(tmp_path):
    R, F, B, alpha = _data()
    fm = FactorModel(R, F).fit()
    m = fm.implied_mean(pd.Series({'f1': 0.01, 'f2': 0.0, 'f3': -0.01}))
    assert np.allclose(m.values, fm.alpha.values + fm.beta.values @ np.array([0.01, 0.0, -0.01]))
    p = tmp_path / 'fm.json'; fm.save(str(p))
    assert json.load(open(p))['kind'] == 'FactorModel'
    fm2 = FactorModel.load(str(p))
    assert np.allclose(fm2.beta.values, fm.beta.values)
    assert list(fm2.report.columns) == list(fm.report.columns) and np.allclose(fm2.report.values, fm.report.values)
    assert np.allclose(fm2.simulate(F.iloc[:5], seed=3).values, fm.simulate(F.iloc[:5], seed=3).values)


def test_fit_johnson_su_heavy_tailed_residual_is_not_degenerate():
    # HYG regression residual of notebook 08: a single Nelder-Mead start stalls at lambda = 0
    from cvinemarketgen import fit_johnson_su, johnson_su_moments
    p = fit_johnson_su(1.5634625201637253, 15.453695663202758)
    assert p['lambda'] > 0.1 and p['residual'] < 1e-6
    m = johnson_su_moments(p)
    assert abs(m[2] - 1.5634625201637253) < 1e-4 and abs(m[3] - 12.453695663202758) < 1e-3
