"""
Smoke tests for CVineMarketGen: the package imports, and the numerical building
blocks give the values verified against the paper's results. No network access.
Run with:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest
import pyvinecopulib as pv

import cvinemarketgen
from cvinemarketgen import (MomentMatch, CopulaTools, CVineGenerator, FleishmanGenerator,
                            factor_targets, FACTORS)


def test_version():
    assert cvinemarketgen.__version__ == "0.1.0"


def test_fleishman_coefficients_match_paper():
    # U.S. Large Cap targets of the paper: skewness -0.5625, excess kurtosis 0.7901
    # -> Table of Fleishman coefficients: a = 0.0868, b = 0.9503, c = -0.0868, d = 0.0138
    fg = FleishmanGenerator()
    assets = ["A"]
    out = fg.fit(pd.Series([0.0819], index=assets), pd.Series([0.1619], index=assets),
                 pd.Series([-0.562509], index=assets), pd.Series([0.790058], index=assets),
                 pd.DataFrame([[1.0]], index=assets, columns=assets))
    a, b, c, d = out["coef"].loc["A", ["a", "b", "c", "d"]].astype(float)
    assert abs(a - 0.0868) < 5e-4 and abs(b - 0.9503) < 5e-4
    assert abs(c + 0.0868) < 5e-4 and abs(d - 0.0138) < 5e-4
    assert out["coef"].loc["A", "feasible"] and out["infeasible_pairs"] == []


def test_vale_maurelli_root_and_simulation():
    assets = ["A", "B"]
    corr = pd.DataFrame([[1.0, -0.1], [-0.1, 1.0]], index=assets, columns=assets)
    fg = FleishmanGenerator()
    fg.fit(pd.Series([0.08, 0.06], index=assets), pd.Series([0.16, 0.12], index=assets),
           pd.Series([-0.56, 0.07], index=assets), pd.Series([0.79, 1.06], index=assets), corr)
    assert fg.infeasible_pairs == []
    sim, err, _ = fg.simulate(n=20000, corr_tol=5e-2, seed=1, verbose=False)
    assert sim.shape == (20000, 2)
    assert err["mean"] < 1e-8 and err["skew"] < 1e-6 and err["kurt"] < 1e-6 and err["corr"] < 5e-2


def test_jsu_fit_reproduces_moments():
    mm = MomentMatch()
    target = [0.0, 1.0, -0.56, 0.79]
    p, res = mm.find_params_for_moments_matching_JSU(target, x0=[0, 1, 1.5, 1], method="Nelder-Mead")
    assert res.fun < 1e-5
    assert np.allclose(mm.moments_JSU(p), target, atol=1e-4)


def test_mixture_hfunction_roundtrip():
    ct = CopulaTools()
    comps = [pv.Bicop(pv.BicopFamily.clayton, 270), pv.Bicop(pv.BicopFamily.gumbel, 0)]
    params = np.array([0.7, 0.66, 1.84])
    np.random.seed(1)
    u = ct.simulate_mixture(params[0], [pv.Bicop(pv.BicopFamily.clayton, 270, [[0.66]]),
                                        pv.Bicop(pv.BicopFamily.gumbel, 0, [[1.84]])], n=2000, seed=1)
    h = ct.hfunc1_mixture(u[:, 0], u[:, 1], params, comps)
    back = ct.hinv1_mixture(u[:, 0], h, params, comps)
    assert np.abs(back - u[:, 1]).max() < 1e-6


def test_known_vine_simulation():
    gen = CVineGenerator(tol_opt=1e-10, n_samples=1000, use_ncs_on_deepertrees=False,
                         use_ncs_on_firsttree=False, use_mixture_on_firsttree=True,
                         use_mixture_on_deepertrees=False, force_try_ncscopula=False)
    assets = ["X1", "X2", "X3"]
    edges = {(2, 1): ("gumbel", 180, 2.0), (3, 1): ("gaussian", 0, -0.4), (3, 2): ("gaussian", 0, 0.1)}
    jsu = {a: (0.0, 0.0, 2.0, 1.5) for a in assets}
    spec = gen.make_vine_spec(assets, edges)
    sim = gen.simulate_known_vine(spec, jsu, pd.Series(0.0, index=assets), pd.Series(1.0, index=assets),
                                  assets, n=5000, seed=1)
    assert sim.shape == (5000, 3)
    c = sim.corr()
    assert c.loc["X1", "X2"] > 0.5 and c.loc["X1", "X3"] < -0.2
    with pytest.raises(ValueError):
        gen.make_vine_spec(assets, {(2, 1): ("gaussian", 0, 0.3)})   # missing edges


def test_exceedance_correlation_shape():
    ct = CopulaTools()
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000); y = 0.5 * x + rng.standard_normal(3000)
    z = np.arange(-1, 1.01, 0.1)
    r = ct.exceedance_correlation(x, y, z)
    assert r.shape == z.shape and np.isfinite(r).all() and (np.abs(r) < 1).all()


def test_factor_targets_layout():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.standard_normal((100, len(FACTORS))) / 100, columns=FACTORS)
    t = factor_targets(ret)
    assert list(t.columns) == ["Arithmetic Mean", "Volatility"] + FACTORS
    assert np.allclose(np.diag(t[FACTORS].values), 1.0)
