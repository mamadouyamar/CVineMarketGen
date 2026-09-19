API reference
=============

The user-facing layer first (:doc:`userguide`), then the engine of the paper
(:doc:`method`). :class:`~cvinemarketgen.cvine.CVineGenerator` inherits from
:class:`~cvinemarketgen.moment_match.MomentMatch` and
:class:`~cvinemarketgen.copulas.CopulaTools`; method names follow the paper's
algorithms.

Targets
-------

.. automodule:: cvinemarketgen.targets
   :members: Targets, nearest_positive_definite

Markets
-------

.. automodule:: cvinemarketgen.markets
   :members: CVineMarket, FleishmanMarket, Diagnostics, partial_correlations

Paths
-----

.. automodule:: cvinemarketgen.paths
   :members: Paths, price_level, yoy, deflate

Dynamics
--------

.. automodule:: cvinemarketgen.dynamics
   :members: AR1, AR1GARCH, GarchFamily, AssetDynamics, parse_spec, make_dynamics

Hidden Markov model
-------------------

.. automodule:: cvinemarketgen.hmm
   :members: GaussianHMM

Selection of the dynamics
-------------------------

.. automodule:: cvinemarketgen.selection
   :members: ljung_box, arch_lm, iid_tests, cvm_statistic, gof_bootstrap, candidate_models, select_dynamics

Blocks
------

.. automodule:: cvinemarketgen.blocks
   :members: BlockVECM, BlockVAR

Structural layer
----------------

.. automodule:: cvinemarketgen.structural
   :members: Structural

Yield curve
-----------

.. automodule:: cvinemarketgen.yieldcurve
   :members: NelsonSiegel, PCACurve, yield_at, discount, bond_price, par_yield, zero_return, constant_maturity_return, curve_returns

Factors
-------

.. automodule:: cvinemarketgen.factors
   :members: FactorModel

Functions
---------

.. automodule:: cvinemarketgen.functions
   :members: fit_johnson_su, johnson_su_moments, johnson_su_sample, fit_fleishman, exceedance_curve, classify_pair, select_family

Market data
-----------

.. automodule:: cvinemarketgen.data
   :members: load_factor_data, factor_targets, load_daily_returns, load_etf_monthly, load_fred_monthly, fred_series, fama_french_monthly, yahoo_monthly_adjclose, yahoo_daily_adjclose, FACTORS, DURATIONS

Engine: C-vine generator
------------------------

.. automodule:: cvinemarketgen.cvine
   :members: CVineGenerator
   :exclude-members: U_last, thetas_for_Corr_all_in_CVine

Engine: Fleishman generator
---------------------------

.. automodule:: cvinemarketgen.fleishman
   :members: FleishmanGenerator

Engine: moment matching
-----------------------

.. automodule:: cvinemarketgen.moment_match
   :members: MomentMatch, bicop_params, bicop_bounds, mvnormcdf

Engine: bivariate copula tools
------------------------------

.. automodule:: cvinemarketgen.copulas
   :members: CopulaTools
