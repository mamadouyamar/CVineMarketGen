import json, os
import numpy as np
import pandas as pd
import pytest

from cvinemarketgen.selection import ljung_box, arch_lm, iid_tests, cvm_statistic, gof_bootstrap, select_dynamics
from tests.test_dynamics import _garch_series

HERE = os.path.dirname(__file__)


def test_tests_match_statsmodels_reference():
    ref = json.load(open(os.path.join(HERE, 'data', 'ljungbox_reference.json')))
    x = np.array(ref['x'])
    s, p = ljung_box(x, 20)
    assert abs(s - ref['lb_stat']) < 1e-6 and abs(p - ref['lb_p']) < 1e-8
    s, p = arch_lm(x, 20)
    assert abs(s - ref['lm_stat']) < 1e-4 and abs(p - ref['lm_p']) < 1e-6


def test_cvm_statistic_uniform_is_small():
    u = (np.arange(1, 1001) - 0.5) / 1000
    assert abs(cvm_statistic(u) - 1 / 12000) < 1e-12
    assert cvm_statistic(np.linspace(0, 0.5, 1000)) > 10


def test_selector_picks_garch_family_and_passes():
    pytest.importorskip('arch')
    y = _garch_series().to_frame()
    ad = select_dynamics(y, candidates=('const', 'garch', 'gjr'), pq=(1, 1), gof=False, verbose=False)
    r = ad.report.iloc[0]
    assert r['vol'] in ('garch', 'gjr') and r['passed'] and r['n_candidates'] == 6
    assert set(ad.candidates.columns) >= {'asset', 'model', 'bic', 'lb_z', 'lb_z2', 'arch_lm', 'passed'}
    z = ad.filter(y)
    assert min(iid_tests(z['X'].values).values()) > 0.05


def test_gof_bootstrap_runs_small():
    pytest.importorskip('arch')
    from cvinemarketgen.dynamics import parse_spec
    y = _garch_series(n=800)
    m = parse_spec('ar1-garch(1,1)').fit(y)
    g = gof_bootstrap(m, y, B=5, seed=0)
    assert 0 <= g['pvalue'] <= 1 and len(g['stats']) == 5 and g['stat'] > 0


def test_gof_bootstrap_hmm_uses_warm_refits():
    import json
    from cvinemarketgen.hmm import GaussianHMM
    ref = json.load(open(os.path.join(HERE, 'data', 'genhmm1d_reference.json')))
    y = pd.Series(ref['y'])
    m = GaussianHMM(2).fit(y)
    r = m.refit(y)                                   # warm start: same optimum, fewer starts
    assert r is not m and np.allclose(r.mu, m.mu, atol=1e-5) and np.allclose(r.Q, m.Q, atol=1e-4)
    g = gof_bootstrap(m, y, B=3, seed=0)
    assert len(g['stats']) == 3 and np.isfinite(g['stats']).all() and 0 <= g['pvalue'] <= 1


def test_default_candidate_set_has_28_models():
    from cvinemarketgen.selection import candidate_models
    ms = candidate_models()
    assert len(ms) == 28 and sum(m.name.startswith('HMM') for m in ms) == 2
    assert {(m.p, m.q) for m in ms if m.kind == 'garch' and m.vol != 'const'} == {(1, 1), (1, 2), (2, 1), (2, 2)}
