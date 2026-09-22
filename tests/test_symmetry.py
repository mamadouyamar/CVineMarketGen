import json
import os

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from cvinemarketgen import Targets, CVineMarket, tail_asymmetry_test, select_family


def _pair(family, theta, n, seed, rotation=0, df=None):
    b = pv.Bicop(family, rotation, parameters=np.array([[theta]] if df is None else [[theta], [df]], float))
    U = b.simulate(n, seeds=[seed, seed + 1, seed + 2, seed + 3, seed + 4])
    from scipy.stats import norm
    return norm.ppf(U[:, 0]), norm.ppf(U[:, 1])


def test_asymmetry_test_on_known_pairs():
    x, y = _pair(pv.BicopFamily.gaussian, 0.3, 260, 1)
    r = tail_asymmetry_test(x, y)
    assert set(r) >= {'difference', 'lower', 'upper', 'symmetric'} and r['lower'] <= 0.0 <= r['upper'] and r['symmetric']
    x, y = _pair(pv.BicopFamily.clayton, 3.0, 1000, 2)                       # strong lower-tail dependence
    r = tail_asymmetry_test(x, y)
    assert r['difference'] > 0.2 and not r['symmetric']


def test_select_family_symmetric_and_asymmetric():
    # Gaussian pairs at the ETF sample size: the test keeps most of them out of the asymmetric classes
    hits = 0
    for seed in range(20):
        x, y = _pair(pv.BicopFamily.gaussian, 0.0, 260, 100 + 5 * seed)
        sel = select_family(x, y)
        hits += sel['family'] in ('gaussian', 'student')
    assert hits >= 15
    # a Student pair: symmetric, and the Student t wins by BIC
    x, y = _pair(pv.BicopFamily.student, 0.2, 1500, 7, df=3.0)
    sel = select_family(x, y)
    assert sel['symmetric'] and sel['family'] == 'student' and len(sel['parameters']) == 2
    # a Clayton pair: asymmetric, the paper's rule applies and finds it
    x, y = _pair(pv.BicopFamily.clayton, 2.0, 1500, 9)
    sel = select_family(x, y)
    assert not sel['symmetric'] and sel['family'] == 'clayton' and sel['rotation'] == 0
    # the paper's rule alone is available
    sel0 = select_family(x, y, symmetry_test=False)
    assert sel0['symmetric'] is False and sel0['asymmetry'] is None


def test_student_family_in_a_market_and_persistence(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.period_range('2000-01', periods=300, freq='M')
    H = pd.DataFrame(rng.standard_normal((300, 3)) * 0.04, index=idx, columns=['A', 'B', 'C'])
    t = Targets.from_history(H)
    fam = {('A', 'B'): ('student', 0, 0.3, 4.0), ('A', 'C'): ('gumbel', 0, 1.5)}
    cv = CVineMarket(t, central='A', families=fam, n_opt=2000).fit()
    e = cv.edges
    row = e[e['edge'].str.startswith('B , A')].iloc[0]
    assert row['selected family'] == 'student 0°' and 'df=4.0' in row['parameters']
    X = cv.simulate(500, seed=1, accept=False)
    assert np.all(np.isfinite(X.values)) and list(X.columns) == ['A', 'B', 'C']
    path = tmp_path / 'm.json'
    cv.save(str(path))
    back = CVineMarket.load(str(path))
    assert back.edges.equals(e)
    assert np.all(np.isfinite(back.simulate(200, seed=2, accept=False).values))


def test_auto_selection_reports_the_symmetry_test():
    rng = np.random.default_rng(3)
    idx = pd.period_range('1995-01', periods=300, freq='M')
    Z = rng.multivariate_normal([0, 0, 0], [[1, .5, .1], [.5, 1, .2], [.1, .2, 1]], size=300) * 0.04
    t = Targets.from_history(pd.DataFrame(Z, index=idx, columns=['A', 'B', 'C']))
    cv = CVineMarket(t, central='A', families='auto', n_opt=2000).fit()
    c = cv.classification
    assert list(c.index) == ['B', 'C'] and 'symmetric' in c.columns and c['asymmetry'].notna().all()
    fams = dict(zip(cv.edges['edge'].str.split(' , ').str[0], cv.edges['selected family']))
    for a in ['B', 'C']:
        if c.loc[a, 'symmetric']:
            assert fams[a].split()[0] in ('gaussian', 'student')
    cv0 = CVineMarket(t, central='A', families='auto', n_opt=2000, symmetry_test=False).fit()
    assert cv0.classification['symmetric'].isna().all()
