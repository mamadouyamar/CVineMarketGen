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
   cv = CVineMarket(t, families='auto', dynamics='ar1-garch').fit()   # needs: pip install arch
   P = cv.simulate_paths(1000, 252, seed=1)

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
returns returns.

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
