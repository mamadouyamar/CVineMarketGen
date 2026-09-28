"""The simulator: workbook (MarketSpec) and the fixed-order rule (FactorMarket)."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from cvinemarketgen import MarketSpec, FactorMarket


def _universe(n=400, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 3))
    F = pd.DataFrame(np.column_stack([Z[:, 0], 0.5 * Z[:, 0] + np.sqrt(0.75) * Z[:, 1], 0.2 * Z[:, 0] + np.sqrt(0.96) * Z[:, 2]]) * 0.04,
                     columns=['eq', 'credit', 'cmd'], index=pd.period_range('1985-01', periods=n, freq='M'))
    R = pd.DataFrame({'A': 1.0 * F['eq'] + 0.5 * F['credit'] + 0.02 * rng.standard_normal(n),
                      'B': 0.3 * F['eq'] + 0.8 * F['cmd'] + 0.02 * rng.standard_normal(n),
                      'C': 0.9 * F['eq'] + 0.02 * rng.standard_normal(n)}, index=F.index)
    return R, F


def _spec():
    return MarketSpec(
        assets=pd.DataFrame({'ticker': ['A', 'B', 'C', 'PC', 'Cash'], 'tag': ['eqcredit', 'eqcmd', 'eq', 'private', None],
                             'mean': [np.nan, 0.05, np.nan, 0.07, 0.02], 'vol': [np.nan, np.nan, 0.20, 0.14, 0.01],
                             'skew': [np.nan, np.nan, -1.0, -0.8, np.nan], 'kurt': [np.nan, np.nan, 8.0, 9.0, np.nan]}),
        tags=pd.DataFrame({'tag': ['eqcredit', 'eqcmd', 'eq', 'private'], 'eq': ['fit', 'fit', 1.0, 0.2], 'credit': [0.5, np.nan, np.nan, 0.6],
                           'cmd': [np.nan, 'fit', np.nan, np.nan], 'illiq': [np.nan, np.nan, np.nan, 0.5]}),
        exposures=pd.DataFrame({'ticker': ['B'], 'factor': ['eq'], 'beta': [np.nan], 'corr': [0.4]}),
        pairs=pd.DataFrame({'ticker_1': ['A'], 'ticker_2': ['B'], 'corr': [0.5]}),
        factors=pd.DataFrame({'factor': ['eq', 'illiq'], 'mean': [0.06, 0.03], 'vol': [np.nan, 0.06]}),
        factor_corr=pd.DataFrame({'factor_1': ['eq', 'illiq'], 'factor_2': ['credit', 'eq'], 'corr': [0.3, 0.35]}))


def test_minimum_input_reproduces_the_sample_moments():
    R, F = _universe()
    m = FactorMarket(R, F, generator='fleishman').fit()
    assert (m.report['fit'] == 3).all() and (m.report['mean source'] == 'history (alpha)').all() and m.report['history'].all()
    X = m.simulate(60000, seed=1); chk = m.check(X)
    a = chk['assets']
    assert (np.abs(a['mean simulated'] - a['mean target']) < 3 * a['mean MC error']).all()
    assert (np.abs(a['vol simulated'] - a['vol target']) / a['vol target'] < 0.01).all()
    assert (np.abs(a['skew simulated'] - a['skew sample']) < 0.3).all()           # residual shape from the history (a sample skewness on 400 months has an error of about 0.12)
    assert chk['factors'].attrs['max_abs_corr_error'] < 0.02 and chk['pairs'] is None


def test_workbook_every_lever():
    R, F = _universe()
    m = FactorMarket(R, F, spec=_spec(), generator='fleishman').fit()
    rep, fr = m.report, m.factor_report
    assert list(m.targets.assets) == ['eq', 'credit', 'cmd', 'illiq'] and m.targets.synthetic == ['illiq']
    assert fr.loc['eq', 'mean source'] == 'workbook' and fr.loc['eq', 'vol source'] == 'history' and fr.loc['illiq', 'corr completed'] == 2
    assert abs(m.targets.mean['eq'] - 0.06 / 12) < 1e-12 and abs(m.targets.corr.loc['eq', 'credit'] - 0.3) < 1e-9
    assert rep.loc['A', ['value', 'fit', 'zero']].tolist() == [1, 1, 2] and m.model.beta.loc['A', 'credit'] == 0.5      # tag: value + fit
    assert rep.loc['B', 'corr'] == 1 and rep.loc['B', 'mean source'] == 'workbook' and rep.loc['B', 'vol source'] == 'history (residual)'
    assert rep.loc['C', 'vol source'] == 'workbook' and rep.loc['C', 'skew source'] == 'workbook'
    assert not rep.loc['PC', 'history'] and rep.loc['PC', 'value'] == 3 and m.model.beta.loc['PC', 'illiq'] == 0.5
    assert rep.loc['Cash', 'zero'] == 4 and rep.loc['Cash', 'skew source'] == 'normal'
    assert rep.loc['A', 'pairs'] == 1 and rep.loc['B', 'pairs'] == 1 and m.pair_report.loc[('A', 'B'), 'target'] == 0.5
    X = m.simulate(60000, seed=2); Fs = m.last_factors; chk = m.check(X)
    a = chk['assets']
    assert abs(X['B'].corr(Fs['eq']) - 0.4) < 0.015                                   # correlation given instead of a beta
    assert abs(chk['pairs'].loc[('A', 'B'), 'simulated'] - 0.5) < 0.015
    assert abs(a.loc['PC', 'mean simulated'] * 12 - 0.07) < 0.003 and abs(a.loc['PC', 'vol simulated'] * np.sqrt(12) - 0.14) < 0.003
    assert a.loc['PC', 'skew simulated'] < -0.6 and a.loc['C', 'skew simulated'] < -0.8
    assert abs(Fs['illiq'].corr(Fs['eq']) - 0.35) < 0.02 and abs(Fs['eq'].corr(Fs['credit']) - 0.3) < 0.02
    assert abs(X['Cash'].corr(X['A'])) < 0.02


def test_refusals_name_the_cause():
    R, F = _universe()
    sp = _spec()
    sp.assets.loc[sp.assets['ticker'] == 'PC', 'vol'] = 0.05                         # below the systematic floor
    with pytest.raises(ValueError, match='betas alone'):
        FactorMarket(R, F, spec=sp, generator='fleishman').fit()
    sp = _spec(); sp.assets.loc[sp.assets['ticker'] == 'C', 'kurt'] = 4.0              # a shape the residual cannot carry
    with pytest.raises(ValueError, match='no Johnson SU has'):
        FactorMarket(R, F, spec=sp, generator='fleishman').fit()
    sp = _spec(); sp.tags.loc['private', 'eq'] = 'fit'                                  # fit without a history
    with pytest.raises(ValueError, match='give values'):
        FactorMarket(R, F, spec=sp, generator='fleishman').fit()
    sp = _spec(); sp.factors = sp.factors[sp.factors['factor'] != 'illiq']             # a factor used by a tag but never declared
    with pytest.raises(ValueError):
        FactorMarket(R, F, spec=sp, generator='fleishman').fit()
    sp = MarketSpec(assets=pd.DataFrame({'ticker': ['New'], 'mean': [0.05]}))          # no history and no vol
    with pytest.raises(ValueError, match='mean and vol'):
        FactorMarket(R, F, spec=sp, generator='fleishman').fit()


def test_platykurtic_residual_is_floored_with_a_flag():
    R, F = _universe()
    R = R.copy(); R['C'] = 0.9 * F['eq'] + 0.02 * np.random.default_rng(3).uniform(-1.7, 1.7, len(F))   # platykurtic residual
    m = FactorMarket(R, F, generator='fleishman').fit()
    assert 'raised to the floor' in m.report.loc['C', 'flags']


def test_a_factor_view_reaches_the_assets_with_a_history():
    R, F = _universe()
    m0 = FactorMarket(R, F, generator='fleishman').fit()
    sp = MarketSpec(factors=pd.DataFrame({'factor': ['eq'], 'mean': [F['eq'].mean() * 12 - 0.03]}))     # equity premium 3 points lower
    m1 = m0.with_spec(sp)
    d = (m1.model.alpha + m1.model.beta @ m1.targets.mean.loc[m1.model.factors]) - (m0.model.alpha + m0.model.beta @ m0.targets.mean.loc[m0.model.factors])
    assert np.allclose(d.values * 12, -0.03 * m0.model.beta['eq'].values, atol=1e-9)                     # every asset moves by beta times the change
    assert m1.model.alpha.equals(m0.model.alpha)                                                          # the alphas did not absorb it


def test_workbook_roundtrip_and_persistence():
    R, F = _universe()
    d = tempfile.mkdtemp(); xp = os.path.join(d, 'inputs.xlsx')
    sp = _spec(); sp.write(xp); back = MarketSpec.read(xp)
    assert back.tags.loc['eqcredit', 'eq'] == 'fit' and float(back.tags.loc['eq', 'eq']) == 1.0 and back.assets['tag'].isna().sum() == 1
    m = FactorMarket(R, F, spec=back, generator='fleishman').fit()
    m2 = FactorMarket(R, F, spec=_spec(), generator='fleishman').fit()
    assert np.allclose(m.model.beta.values, m2.model.beta.values)
    stem = os.path.join(d, 'sim'); m.save(stem); m3 = FactorMarket.load(stem)
    assert m3.report.shape == m.report.shape and list(m3.model.assets) == list(m.model.assets)
    X3 = m3.simulate(2000, seed=1)
    assert X3.shape == (2000, 5) and list(m3.last_factors.columns) == ['eq', 'credit', 'cmd', 'illiq']
    P = m3.simulate_paths(50, 12, seed=2)
    assert P.array.shape == (50, 12, 5)
    assert m3.simulate(3, seed=2).shape == (3, 5)                                           # a tiny draw must not loop on the acceptance test


def test_cvine_generator_with_a_synthetic_factor():
    R, F = _universe()
    m = FactorMarket(R, F, spec=_spec(), generator='cvine', central='eq', market_kwargs={'families': 'gaussian'}).fit()
    X = m.simulate(20000, seed=1)
    assert list(m.last_factors.columns) == ['eq', 'credit', 'cmd', 'illiq'] and X.shape == (20000, 5)
    assert abs(m.last_factors['illiq'].corr(m.last_factors['eq']) - 0.35) < 0.03


def test_a_correlation_view_moves_the_volatility_of_a_history_asset():
    R, F = _universe()
    R = R.copy(); R['D'] = 1.0 * F['eq'] - 1.0 * F['credit'] + 0.005 * np.random.default_rng(5).standard_normal(len(F))   # long eq, short credit
    m0 = FactorMarket(R, F, generator='fleishman').fit()
    sp = MarketSpec(factor_corr=pd.DataFrame({'factor_1': ['eq'], 'factor_2': ['credit'], 'corr': [-0.6]}))
    m1 = m0.with_spec(sp)
    v0 = np.sqrt(m0.model.implied_covariance().loc['D', 'D']); v1 = np.sqrt(m1.model.implied_covariance().loc['D', 'D'])
    assert v1 > v0 * 1.1 and 'volatility' not in m1.report.loc['D', 'flags']                             # the residual is the history's, the asset's volatility follows the view
    sp2 = MarketSpec(assets=pd.DataFrame({'ticker': ['D'], 'vol': [R['D'].std() * np.sqrt(12)]}),
                     factor_corr=pd.DataFrame({'factor_1': ['eq'], 'factor_2': ['credit'], 'corr': [-0.6]}))
    with pytest.raises(ValueError, match='betas alone'):                                                 # the same volatility, given: refused
        FactorMarket(R, F, spec=sp2, generator='fleishman').fit()


def test_add_edits_a_row_in_place_and_keeps_its_cells():
    sp = MarketSpec(assets=pd.DataFrame({'ticker': ['A', 'B', 'C'], 'tag': ['x', 'y', 'z'], 'vol': [np.nan, 0.2, np.nan]}),
                    tags=pd.DataFrame({'tag': ['x', 'y', 'z'], 'eq': ['fit', 'fit', np.nan]}))
    sp2 = sp.add('assets', pd.DataFrame({'ticker': ['A'], 'mean': [0.07]}))
    assert list(sp2.assets['ticker']) == ['A', 'B', 'C']                       # the row keeps its place
    assert sp2.assets.loc[0, 'mean'] == 0.07 and sp2.assets.loc[0, 'tag'] == 'x'   # and its other cells
    sp3 = sp2.add('assets', pd.DataFrame({'ticker': ['D'], 'mean': [0.05], 'vol': [0.1]}))
    assert list(sp3.assets['ticker']) == ['A', 'B', 'C', 'D'] and sp3.assets.loc[1, 'vol'] == 0.2
    sp4 = sp3.add('tags', pd.DataFrame({'tag': ['y'], 'credit': [0.5]}))        # a new column on an existing tag
    assert list(sp4.tags.index) == ['x', 'y', 'z'] and sp4.tags.loc['y', 'credit'] == 0.5 and sp4.tags.loc['y', 'eq'] == 'fit'
    sp4b = sp4.add('assets', {'ticker': 'B', 'mean': 0.04})                     # one row, given as a dict
    assert list(sp4b.assets['ticker']) == ['A', 'B', 'C', 'D'] and sp4b.assets.loc[1, 'mean'] == 0.04 and sp4b.assets.loc[1, 'vol'] == 0.2
    sp5 = sp4.add('exposures', pd.DataFrame({'ticker': ['A'], 'factor': ['eq'], 'beta': [0.8]}))
    sp5 = sp5.add('exposures', pd.DataFrame({'ticker': ['A'], 'factor': ['eq'], 'beta': [1.0]}))
    assert len(sp5.exposures) == 1 and sp5.exposures.loc[0, 'beta'] == 1.0      # the same pair is one row, edited


def test_a_factor_correlation_is_solved_against_the_asset_own_volatility():
    R, F = _universe()
    sp = MarketSpec(exposures=pd.DataFrame({'ticker': ['A'], 'factor': ['credit'], 'beta': [np.nan], 'corr': [0.55]}))
    m = FactorMarket(R, F, spec=sp, generator='fleishman').fit()
    X = m.simulate(60000, seed=1)
    assert abs(X['A'].corr(m.last_factors['credit']) - 0.55) < 0.02          # delivered at the volatility the asset ends up with
    sp2 = MarketSpec(exposures=pd.DataFrame({'ticker': ['A'], 'factor': ['credit'], 'beta': [np.nan], 'corr': [0.99]}))
    with pytest.raises(ValueError, match='not attainable while its volatility cell is empty'):
        FactorMarket(R, F, spec=sp2, generator='fleishman').fit()
