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
   :members: Paths

Dynamics
--------

.. automodule:: cvinemarketgen.dynamics
   :members: AR1, AR1GARCH, make_dynamics

Functions
---------

.. automodule:: cvinemarketgen.functions
   :members: fit_johnson_su, johnson_su_moments, johnson_su_sample, fit_fleishman, exceedance_curve, classify_pair, select_family

Market data
-----------

.. automodule:: cvinemarketgen.data
   :members: load_factor_data, factor_targets, load_daily_returns, fred_series, fama_french_monthly, yahoo_monthly_adjclose, yahoo_daily_adjclose, FACTORS, DURATIONS

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
