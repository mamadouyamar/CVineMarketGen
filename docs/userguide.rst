User guide
==========

This page is for practitioners: what to give the package, what it gives back,
and which options matter. The paper's algorithms run underneath; see
:doc:`method` for that map and :doc:`api` for every signature.

Three objects
-------------

* :class:`~cvinemarketgen.targets.Targets` holds what the simulated returns
  must match: mean, volatility and skewness and kurtosis per asset, the
  correlation matrix, and optionally the history they came from.
* :class:`~cvinemarketgen.markets.FleishmanMarket` matches the four moments
  and the correlation matrix with Gaussian dependence. Fast, exact on
  correlations, no tail dependence.
* :class:`~cvinemarketgen.markets.CVineMarket` matches the same targets with a
  C-vine copula whose pair families carry the tail dependence and the
  sign-changing conditional correlations seen in the data.

Both markets have the same methods: ``fit``, ``simulate``, ``simulate_paths``,
``diagnostics``, ``plot_exceedance``, ``save``, ``load``.

Units are yours: annual targets give annual returns, daily give daily. Nothing
is annualized. Kurtosis is raw (a normal has 3).

What you have decides how you build the targets
-----------------------------------------------

**Only mean, volatility and correlations** (your own capital-market assumptions):

.. code-block:: python

   t = Targets(mean={'Equity': 0.07, 'Bonds': 0.03}, vol={'Equity': 0.16, 'Bonds': 0.05},
               corr=[[1, -0.1], [-0.1, 1]], assets=['Equity', 'Bonds'])

Skewness and kurtosis default to 0 and 3 (normal marginals); ``t.summary()``
says so. Without a history there is nothing to select copula families from, so
use ``CVineMarket(t, families='gaussian')`` or choose the families yourself
(below).

**A published table** (CSV, Excel or DataFrame with one row per asset, a mean
column, a volatility column and the correlation columns named after the assets):

.. code-block:: python

   t = Targets.from_ltcma('data/jpm_ltcma_2024.csv', assets=[...])
   t = Targets.from_ltcma(table, assets=[...], history=returns)     # higher moments and families from the history
   t = Targets.from_ltcma(table, assets=[...], skew=..., kurt=...)  # higher moments given

**Skewness and kurtosis given directly**:

.. code-block:: python

   t = Targets(mean=..., vol=..., corr=..., skew={'Equity': -0.6, ...}, kurt={'Equity': 4.5, ...})

**A history only**, at any frequency (the targets are the sample):

.. code-block:: python

   t = Targets.from_history(returns)        # DataFrame, one column per asset, daily / weekly / monthly

Copula families
---------------

``CVineMarket(t, families=...)`` accepts three things:

``'auto'``
   Selected from the history by the paper's Algorithm 3: each pair is
   classified from its exceedance-correlation curve, the admissible families
   are fitted, the best BIC is kept; non-monotone pairs get a mixture. Needs a
   history in the targets.

``'gaussian'``
   Gaussian pair copulas everywhere, initialised at the partial correlations of
   the target matrix, then calibrated. Exact on correlations, no tail dependence.
   The honest default when there is no history.

a dict, one entry per pair you want to set
   .. code-block:: python

      families = {('Bonds', 'Equity'): ('mixture', [('clayton', 270), ('gumbel', 0)], 0.6),
                  ('Gold', 'Equity'): ('gumbel', 180),
                  ('Gold', 'Bonds'): ('gaussian', 0, 0.3)}

   ``(family, rotation)``, optionally with a starting parameter; a mixture is
   ``('mixture', [(fam1, rot1), (fam2, rot2)], weight_on_first)``. Pairs not
   listed are Gaussian. Families: ``gaussian``, ``clayton``, ``gumbel``,
   ``joe``, ``frank``; rotations 0, 90, 180, 270. Clayton has lower-tail
   dependence, Gumbel upper-tail; a 180° rotation swaps the tail; 90° and 270°
   give negative dependence. Choosing a mixture without a history to check it
   against is expert use.

The first asset of the targets is the root of the C-vine (``central=`` changes
it). Choose the asset that drives the others, typically the broad equity index.

Inspecting a fitted market
--------------------------

.. code-block:: python

   cv = CVineMarket(t, families='auto').fit()
   cv.edges          # family, rotation, calibrated parameters per edge (tree, variable)
   cv.selected_edges # the same before calibration, as selected on the history
   cv.marginals      # Johnson SU parameters per asset

   X = cv.simulate(25000, seed=1)     # DataFrame, one row per scenario
   d = cv.diagnostics(X)              # target vs simulated moments and correlation errors
   d.summary(); d.moments; d.corr_error
   cv.plot_exceedance(X)              # exceedance-correlation curves, history overlaid when there is one

``simulate`` keeps a draw only if it meets the tolerances of the paper on
moments (2e-4 on mean and volatility, 5e-2 on skewness and kurtosis) and on
correlations (``corr_tol``, 0.05 by default); the marginal parameters are
re-fitted on the accepted draw. Pass ``accept=False`` for small ``n``.

Paths
-----

.. code-block:: python

   P = cv.simulate_paths(n_paths=1000, horizon=252, seed=1)
   P.array            # (n_paths, horizon, N)
   P.to_frame()       # long table: path, t, one column per asset
   P.wide('SPY')      # horizon x n_paths
   P.cumulative()     # cumulative returns along the horizon
   P.terminal()       # cumulative return at the horizon, n_paths x N
   P.summary()        # per-period moments pooled over paths

Without dynamics the periods are independent draws, right for annual
scenarios. With daily or weekly data, add dynamics.

Dynamics for daily data
-----------------------

.. code-block:: python

   t = Targets.from_history(daily_returns)
   cv = CVineMarket(t, families='auto', dynamics='auto').fit()        # needs: pip install arch
   P = cv.simulate_paths(1000, 252, seed=1)

``dynamics='auto'``
   One model per asset, chosen among the GARCH family and a Gaussian hidden
   Markov model by i.i.d. tests, BIC and a bootstrap goodness-of-fit test; see
   the next section. A dict gives the model per asset instead.

``dynamics='ar1'``
   Per-asset AR(1) fitted on the history; the generator is fitted on the
   residuals. Works with a history or with capital-market targets, through the
   AR(1) moment transfer of Appendix A of the paper.

``dynamics='ar1-garch'``
   AR(1) mean with GARCH(1,1) variance per asset (``arch`` package, optional
   dependency). The generator is fitted on the standardized residuals; the
   simulated residuals are filtered back from the last observed state, so the
   paths carry volatility clustering and the fat tails of the history. Supported
   with history-based targets.

With dynamics, ``simulate`` returns residuals (the layer the generator is
fitted on; ``cv.fit_targets`` shows its targets) and ``simulate_paths``
returns returns. The Johnson SU marginal cannot be lighter-tailed than the
normal: when a residual layer has kurtosis at or below 3 (near-normal
residuals, as an HMM produces), its target kurtosis is floored just above the
family's boundary, with a warning.

Choosing the dynamics
---------------------

.. code-block:: python

   cv = CVineMarket(t, families='auto', dynamics='auto').fit()
   cv.dynamics_report          # one row per asset: model, BIC, the three p-values, goodness-of-fit p-value
   cv.dynamics.candidates      # every candidate fitted, per asset

For each asset the candidates are a constant or AR(1) mean with a constant,
GARCH(p,q), GJR(p,q) or EGARCH(p,q) variance (``p, q`` up to 2 by default),
plus a Gaussian hidden Markov model with 2 or 3 regimes. Each candidate's
residual layer (standardized residuals, or the normal scores of the
Rosenblatt uniforms for the HMM) is tested for the absence of autocorrelation
(Ljung-Box) and of remaining ARCH (Ljung-Box on the squares, ARCH-LM); among
the candidates that pass at 5 percent the lowest BIC is kept, otherwise the
lowest BIC overall with a warning. The selected model is then checked by a
parametric-bootstrap Cramér-von Mises test on its Rosenblatt uniforms
(``gof_pvalue``), as in GenHMM1d, which also estimates the HMM candidates.

Options through ``dynamics_kwargs``: ``candidates``, ``means``, ``pq``,
``states``, ``alpha``, ``lags``, ``gof``, ``B``, ``seed``, ``verbose``. To
force a model per asset give a dict:
``dynamics={'SPY': 'ar1-gjr(1,1)', 'TLT': 'hmm(2)', 'GLD': 'const-garch(1,1)'}``.
Cost: about 28 fits per asset (seconds each) and one bootstrap of ``B``
refits per asset. The pieces are available on their own:
:func:`~cvinemarketgen.selection.select_dynamics`,
:func:`~cvinemarketgen.selection.iid_tests`,
:func:`~cvinemarketgen.selection.gof_bootstrap`,
:class:`~cvinemarketgen.hmm.GaussianHMM` (with ``regimes``, ``uniforms``,
``filter``, ``unfilter``).

Blocks of variables
-------------------

.. code-block:: python

   yields = load_fred_monthly(['DGS10', 'DFII10'])
   t = Targets.from_history(yields.join(spy_returns, how='inner'))
   cv = CVineMarket(t, central='SPY', families='auto',
                    dynamics={('DGS10', 'DFII10'): 'vecm', 'SPY': 'ar1-garch'}).fit()
   P = cv.simulate_paths(1000, 24, seed=1)      # yields in levels, SPY in returns

Variables observed in levels that share long-run relations (yields, breakevens,
spreads) are filtered jointly: a tuple key gives the block, ``'vecm'`` fits a
vector error-correction model (cointegration rank by the Johansen trace test,
lags by BIC; ``'vecm(r=1,q=2)'`` fixes them) and ``'var'`` a vector
autoregression. The block's residual layer is its standardized innovations, one
per variable; their correlation is left to the vine, like everything else. Paths
of the block's variables come back in levels, from the last observed rows,
through the model's own recursion; ``Paths.cumulative`` and ``terminal`` apply to
the return columns. Needs ``pip install statsmodels``.

The structural layer
--------------------

.. code-block:: python

   cv = CVineMarket(t, central='SPY', families='auto',
                    dynamics={('rate_diff', 'log_oil'): 'vecm(r=0,q=1)',
                              'log_fx': 'ecm(rate_diff, log_oil)',
                              'SPY': 'ar1-garch'}).fit()
   cv.dynamics.order                    # parents before children
   cv.dynamics.models['log_fx'].theta   # long-run elasticities

A variable driven by others is a child: ``'ecm(p1, p2)'`` fits the
single-equation error-correction model in levels (adjustment ``kappa``, long-run
relation ``theta``, contemporaneous response ``gamma``), ``'linear(p1, p2)'`` the
linear model in returns, ``'linear(p1, p2; lags=0)'`` the plain factor map. The
child's innovation is its residual layer; on a path the parents are simulated
first, by their own models, and the child is rebuilt from them and from its own
draw. Parents are taken as weakly exogenous; cycles are refused.

The yield curve and fixed income
--------------------------------

.. code-block:: python

   Y = load_fred_monthly(['DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30'], start='1993-10')
   ns = NelsonSiegel(0.7308).fit(Y)             # level, slope, curvature by least squares per date
   ns.rmse                                      # fitting error by maturity
   data = ns.factors.join(spy_returns, how='inner')
   cv = CVineMarket(Targets.from_history(data), central='SPY', families='auto',
                    dynamics={('level', 'slope', 'curvature'): 'vecm', 'SPY': 'ar1-garch'}).fit()
   P = cv.simulate_paths(1000, 24, seed=1)
   PF = Paths(P.array[:, :, :3], ['level', 'slope', 'curvature'])
   curves = ns.curve(PF)                                       # yields at the fitted maturities
   R = curve_returns(PF, [2, 10, 30], ns, ns.factors.iloc[-1].values)   # constant-maturity bond returns

The curve is three observed factors, level, slope and curvature, estimated by
least squares per date on the Nelson-Siegel loadings with the decay ``lam`` in
years (``'auto'`` searches it; ``PCACurve`` is the benchmark with the same
interface). They enter the market as a block like any other variables in
levels; simulated factors give curves at any maturity through ``curve``, and
``bond_price``, ``par_yield`` and ``curve_returns`` price zero-coupon and
constant-maturity par bonds along the paths, with continuous compounding on
yields in percent.

Inflation and real returns
--------------------------

.. code-block:: python

   spec = 'linear(d_log_oil, unrate, mich; lags=3)'
   cv = CVineMarket(Targets.from_history(data), central='SPY', families='auto',
                    dynamics={('d_log_oil', 'unrate', 'mich'): 'vecm', 'pi': spec, 'core': spec,
                              'SPY': 'ar1-garch'}).fit()
   P = cv.simulate_paths(1000, 24, seed=1)
   yoy(P, 'pi', data['pi'])            # year-on-year inflation along the paths (percent)
   price_level(P, 'pi')                # CPI index from 100 at the last observation
   deflate(P, 'SPY', 'pi')             # real returns of SPY, a one-column Paths

Inflation is a child in the linear form on its monthly annualized rate, with
lags: a Phillips-curve equation on the change of oil, the unemployment rate
and survey expectations. ``yoy``, ``price_level`` and ``deflate`` turn a column
of monthly log inflation in percent a year into year-on-year rates, price
levels and real returns of another column. ``exclude=['2020-03', '2020-04']``
leaves known one-off months out of the residual layer on which the marginals
and the vine are fitted (the April 2020 unemployment jump is a 14-sigma
innovation); the filters still use the whole history.

Quarterly output from monthly paths
-----------------------------------

.. code-block:: python

   gdp = load_fred_quarterly(['GDPC1'], start='1993Q1')
   g = (400 * np.log(gdp['GDPC1']).diff()).dropna()             # growth, percent annualized
   br = Bridge(['d_payems', 'd_indpro', 'd_unrate'], lags=1).fit(g, data[parents])
   br.report                                                     # coefficients and t-statistics
   P = cv.simulate_paths(1000, 24, seed=1)                       # monthly paths of the parents' block and SPY
   G = br.simulate(P, g.values, seed=1)                          # Paths (1000, 8, 1) of quarterly growth
   growth_to_level(G)                                            # level of output, last quarter = 100
   recession_probability(G, k=2)                                 # share of paths with two negative quarters

A quarterly series is a child at its own frequency: ``Bridge`` regresses it on
the quarterly means of monthly parents plus its lags (the bridge equation), and
on simulated monthly paths it averages each quarter's parents, applies the
equation and draws a quarterly Johnson SU innovation, independent of the
monthly residual layer. The monthly sample must end at a quarter end, so that
the simulated months start a new quarter; ``load_fred_quarterly`` fetches
quarterly FRED series.

Assets on factors
-----------------

.. code-block:: python

   fm = FactorModel(asset_returns, factor_returns).fit()
   fm.report                                   # alpha, betas, t-statistics, R2, residual vol
   F = CVineMarket(Targets.from_history(factor_returns), families='auto').fit().simulate(25000, seed=1)
   X = fm.simulate(F)                          # asset scenarios: alpha + beta' f + Johnson SU residual
   X0 = fm.simulate(F, residuals=False)        # factor exposure only
   fm.implied_mean(view_on_factor_means)       # expected asset returns under a view on the factors

A ``Paths`` object of factor paths gives a ``Paths`` of asset paths. Residuals
are independent across assets and of the factors; drop them to study the
factor exposure alone. ``load_etf_monthly`` fetches monthly ETF returns to
regress on the seven factors of ``load_factor_data``.

Save and load
-------------

.. code-block:: python

   cv.save('model.json')
   cv = CVineMarket.load('model.json')      # ready to simulate, no refit

The file is JSON: targets, settings, marginal parameters, every edge's family
and parameters, and the dynamics parameters. The history is not stored.

Pieces on their own
-------------------

.. code-block:: python

   from cvinemarketgen import fit_johnson_su, fit_fleishman, exceedance_curve, classify_pair, select_family

   fit_johnson_su(skew=-0.6, kurt=4.5)       # Johnson SU parameters for a marginal
   fit_fleishman(skew=-0.6, kurt=4.5)        # Fleishman coefficients (and whether feasible)
   exceedance_curve(x, y)                    # the curve of one pair, Series indexed by threshold
   classify_pair(x, y)                       # (l, u, s, m) of one pair
   select_family(x, y)                       # the family the paper's selection picks for one pair

Runtime and size
----------------

The cost is in ``fit`` of ``CVineMarket``: the calibration simulates
``n_opt`` draws (default 10000) per objective evaluation, variable by
variable, and a mixture edge needs a bisection each time. Rules of thumb on a
laptop: 4 assets with a mixture, about 3 minutes; 6 assets, about 3 minutes;
Gaussian families, seconds. For wide cross-sections start with
``families='gaussian'`` or lower ``n_opt``; the paper used 20000 draws and a
correlation tolerance of 0.02.
