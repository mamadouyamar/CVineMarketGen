"""The specification layer: what the analyst gives, what the rules fill (spec 2026-09-26)."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from cvinemarketgen import Targets, FactorModel, FleishmanMarket, CVineMarket, complete_correlation


def _factors(n=500, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 3))
    F = pd.DataFrame(np.column_stack([Z[:, 0], 0.5 * Z[:, 0] + np.sqrt(0.75) * Z[:, 1], 0.2 * Z[:, 0] + np.sqrt(0.96) * Z[:, 2]]) * 0.04,
                     columns=['eq', 'credit', 'cmd'], index=pd.period_range('1985-01', periods=n, freq='M'))
    return F, rng


def test_completion_closed_form_and_zero_partial_correlation():
    F, _ = _factors()
    t = Targets.from_history(F)
    t2 = t.add_factor('illiq', mean=0.04, vol=0.08, corr={'eq': 0.4}, annualized=True)
    R = t.corr.values
    assert np.allclose(t2.corr.loc['illiq', ['credit', 'cmd']].values, R[[1, 2], 0] * 0.4, atol=1e-6)   # r_U = R_US R_SS^-1 r_S
    P = np.linalg.inv(t2.corr.values)
    partial = -P / np.sqrt(np.outer(np.diag(P), np.diag(P)))
    assert abs(partial[3, 1]) < 1e-6 and abs(partial[3, 2]) < 1e-6                                    # completed pairs: zero partial correlation
    assert t2.synthetic == ['illiq'] and t2.historical == ['eq', 'credit', 'cmd']
    rep = t2.completion_report
    assert rep.loc[('illiq', 'eq'), 'source'] == 'given' and rep.loc[('illiq', 'cmd'), 'source'] == 'completed'
    assert abs(t2.mean['illiq'] - 0.04 / 12) < 1e-12 and abs(t2.vol['illiq'] - 0.08 / np.sqrt(12)) < 1e-12
    with pytest.raises(ValueError):
        t.add_factor('bad', mean=0.0, vol=0.1, corr={'eq': 0.95, 'credit': -0.9})                     # no valid completion
    d = t2.to_dict(); back = Targets.from_dict(d)
    assert back.synthetic == ['illiq'] and back.completion_report.shape == rep.shape


def test_completion_general_optimizer_matches_closed_form():
    idx = ['a', 'b', 'c', 'd']
    C = pd.DataFrame([[1, .5, .2, .3], [.5, 1, .1, 0], [.2, .1, 1, 0], [.3, 0, 0, 1]], index=idx, columns=idx, dtype=float)
    G = pd.DataFrame(True, index=idx, columns=idx); G.loc['d', ['b', 'c']] = G.loc[['b', 'c'], 'd'] = False
    M = complete_correlation(C, G)
    R = C.loc[['a', 'b', 'c'], ['a', 'b', 'c']].values
    assert np.allclose(M.loc['d', ['b', 'c']].values, R[[1, 2], 0] * 0.3, atol=1e-5)
    assert np.linalg.eigvalsh(M.values).min() > 0 and np.allclose(M.loc['a', 'd'], 0.3)


def test_synthetic_factor_attached_by_both_generators():
    F, _ = _factors()
    t2 = Targets.from_history(F).add_factor('illiq', mean=0.04, vol=0.08, corr={'eq': 0.4, 'cmd': 0.1}, skew=-0.5, kurt=5.0)
    for mk in (FleishmanMarket(t2).fit(), CVineMarket(t2, central='eq', families='gaussian').fit()):
        X = mk.simulate(60000, seed=1)
        assert list(X.columns) == ['eq', 'credit', 'cmd', 'illiq']
        got = X.corr().loc['illiq', ['eq', 'credit', 'cmd']].values
        assert np.abs(got - t2.corr.loc['illiq', ['eq', 'credit', 'cmd']].values).max() < 0.02
        assert abs(X['illiq'].mean() * 12 - 0.04) < 0.003 and abs(X['illiq'].std() * np.sqrt(12) - 0.08) < 0.003
        assert X['illiq'].skew() < -0.3
    with pytest.raises(ValueError):
        CVineMarket(t2, central='illiq', families='gaussian')
    p = os.path.join(tempfile.mkdtemp(), 'cv.json'); mk.save(p); back = CVineMarket.load(p)
    Xb = back.simulate(5000, seed=2)
    assert 'illiq' in Xb.columns and back.targets.synthetic == ['illiq']


def test_exposures_given_partial_and_pinned():
    F, rng = _factors()
    n = len(F)
    R = pd.DataFrame({'A': 1.0 * F['eq'] + 0.5 * F['credit'] + 0.01 * rng.standard_normal(n),
                      'B': 0.3 * F['eq'] + 0.8 * F['cmd'] + 0.01 * rng.standard_normal(n)}, index=F.index)
    fm = FactorModel(R, F, exposures={'A': {'eq': 1.0}, 'B': {'cmd': 0.8, 'eq': 'fit'}}).fit()
    assert fm.beta.loc['A', 'eq'] == 1.0 and np.isnan(fm.tstat.loc['A', 'eq']) and fm.beta_source.loc['A', 'eq'] == 'given'
    assert abs(fm.beta.loc['A', 'credit'] - 0.5) < 0.05 and fm.beta_source.loc['A', 'credit'] == 'fitted'     # unnamed factors regressed
    assert fm.beta_source.loc['A', 'cmd'] == 'fitted'
    assert fm.beta.loc['B', 'cmd'] == 0.8 and abs(fm.beta.loc['B', 'eq'] - 0.3) < 0.05
    assert fm.beta.loc['B', 'credit'] == 0 and fm.beta_source.loc['B', 'credit'] == 'zero'                    # pinned allowed set
    rep = fm.spec_report()
    assert rep.loc['A', 'betas given'] == 1 and rep.loc['B', 'betas zero'] == 1 and (rep['source'] == 'history').all()
    with pytest.raises(ValueError):
        FactorModel(R, F, exposures={'A': {'eq': 'nope'}})
    p = os.path.join(tempfile.mkdtemp(), 'fm.json'); fm.save(p); back = FactorModel.load(p)
    assert back.beta_source.equals(fm.beta_source) and np.allclose(back.beta.values, fm.beta.values)


def test_add_asset_without_history():
    F, rng = _factors()
    R = pd.DataFrame({'A': 1.0 * F['eq'] + 0.01 * rng.standard_normal(len(F))}, index=F.index)
    fm = FactorModel(R, F).fit()
    fm2 = fm.add_asset('PC', mean=0.07, vol=0.10, exposures={'credit': 0.3}, skew=-0.8, kurt=6.0)   # residual-dominated, so its skew shows
    assert fm2.assets == ['A', 'PC'] and fm2.asset_source['PC'] == 'spec' and fm.assets == ['A']
    assert fm2.beta_source.loc['PC', 'cmd'] == 'zero' and np.isnan(fm2.r2['PC'])
    Fsim = pd.DataFrame(rng.multivariate_normal(F.mean(), F.cov(), size=200000), columns=F.columns)
    X = fm2.simulate(Fsim, seed=1)
    assert abs(X['PC'].mean() * 12 - 0.07) < 0.003 and abs(X['PC'].std() * np.sqrt(12) - 0.10) < 0.003
    assert X['PC'].skew() < -0.2
    fm3 = fm.add_asset('cash-like', mean=0.02, vol=0.01)
    assert fm3.asset_source['cash-like'] == 'spec (pure residual)' and abs(fm3.simulate(Fsim.iloc[:50000], seed=2)['cash-like'].corr(X['A'].iloc[:50000])) < 0.02
    with pytest.raises(ValueError):
        fm.add_asset('too-calm', mean=0.05, vol=0.02, exposures={'eq': 1.0})
    tab = pd.DataFrame({'mean': [0.05, 0.03], 'vol': [0.12, 0.06], 'eq': [0.5, np.nan], 'credit': [np.nan, 0.3]}, index=['X', 'Y'])
    fm4 = fm.add_assets(tab)
    assert fm4.assets == ['A', 'X', 'Y'] and fm4.beta.loc['Y', 'eq'] == 0 and fm4.beta.loc['X', 'eq'] == 0.5


def test_asset_with_history_loading_on_a_synthetic_factor():
    F, rng = _factors()
    R = pd.DataFrame({'A': 1.0 * F['eq'] + 0.03 * rng.standard_normal(len(F))}, index=F.index)
    t2 = Targets.from_history(F).add_factor('illiq', mean=0.04, vol=0.08, corr={'eq': 0.2})
    fm = FactorModel(R, F, exposures={'A': {'illiq': 0.5}}, factor_targets=t2).fit()
    assert fm.factors == ['eq', 'credit', 'cmd', 'illiq'] and fm.beta.loc['A', 'illiq'] == 0.5 and abs(fm.beta.loc['A', 'eq'] - 1.0) < 0.05
    assert fm.resid_vol['A'] < fm.sample_resid_vol['A']                                # the synthetic loading takes from the residual
    with pytest.raises(ValueError):
        FactorModel(R, F, exposures={'A': {'illiq': 'fit'}}, factor_targets=t2)
    with pytest.raises(ValueError):
        FactorModel(R, F, exposures={'A': {'illiq': 5.0}}, factor_targets=t2).fit()      # more than the residual holds
    mk = FleishmanMarket(t2).fit(); X = fm.simulate(mk.simulate(100000, seed=1), seed=1)
    assert abs(X['A'].std() - R['A'].std()) / R['A'].std() < 0.03                        # sample volatility kept


def test_pair_correlations_range_and_simulation():
    F, rng = _factors()
    n = len(F)
    R = pd.DataFrame({'A': 1.0 * F['eq'] + 0.02 * rng.standard_normal(n), 'B': 0.8 * F['eq'] + 0.02 * rng.standard_normal(n),
                      'C': 0.5 * F['cmd'] + 0.03 * rng.standard_normal(n)}, index=F.index)
    fm = FactorModel(R, F).fit()
    S = fm.implied_covariance(); s = np.sqrt(np.diag(S.values))
    rho_sys = S.loc['A', 'B'] / (s[0] * s[1])
    with pytest.raises(ValueError, match='eligible range'):
        fm.with_targets(pair_correlations={('A', 'B'): 0.2})                          # far below what the shared factor imposes
    rho_star = float(rho_sys) + 0.5 * float(np.sqrt((1 - fm.r2['A']) * (1 - fm.r2['B'])))
    fm2 = fm.with_targets(pair_correlations={('A', 'B'): rho_star, ('A', 'C'): 0.3})
    rep = fm2.pair_report
    assert abs(rep.loc[('A', 'B'), 'residual correlation'] - 0.5) < 0.02 and rep.loc[('A', 'C'), 'range low'] < 0.3 < rep.loc[('A', 'C'), 'range high']
    assert fm.resid_corr.values[0, 1] == 0 and fm2.resid_corr.loc['A', 'B'] != 0
    Fsim = pd.DataFrame(rng.multivariate_normal(F.mean(), F.cov(), size=200000), columns=F.columns)
    X = fm2.simulate(Fsim, seed=1); X0 = fm.simulate(Fsim, seed=1)
    assert abs(X['A'].corr(X['B']) - rho_star) < 0.01 and abs(X['A'].corr(X['C']) - 0.3) < 0.01
    assert abs(X['B'].corr(X['C']) - X0['B'].corr(X0['C'])) < 0.02                       # the untouched pair keeps its correlation
    assert np.allclose(X.std().values, X0.std().values, rtol=0.02)                       # volatilities unchanged
    def inside(i, j, k=-0.8):                                                              # each pair inside its own range ...
        S_ = fm.implied_covariance(); s_ = np.sqrt(np.diag(S_.values)); ii, jj = fm.assets.index(i), fm.assets.index(j)
        return S_.values[ii, jj] / (s_[ii] * s_[jj]) + k * np.sqrt((1 - fm.r2[i]) * (1 - fm.r2[j]))
    with pytest.raises(ValueError, match='positive semidefinite'):                          # ... but three residual correlations of -0.8 cannot coexist
        fm.with_targets(pair_correlations={('A', 'B'): inside('A', 'B'), ('A', 'C'): inside('A', 'C'), ('B', 'C'): inside('B', 'C')})
    p = os.path.join(tempfile.mkdtemp(), 'fm.json'); fm2.save(p); back = FactorModel.load(p)
    assert np.allclose(back.resid_corr.values, fm2.resid_corr.values) and back.pair_report.shape == rep.shape
    assert back.spec_report().loc['A', 'pair target'] and not back.spec_report().loc['A', 'mean target']
