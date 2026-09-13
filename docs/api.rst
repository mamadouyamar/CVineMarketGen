API reference
=============

One object gives access to everything: :class:`~cvinemarketgen.cvine.CVineGenerator`
inherits from :class:`~cvinemarketgen.moment_match.MomentMatch` and
:class:`~cvinemarketgen.copulas.CopulaTools`. Method names follow the paper's
algorithms; see :doc:`method` for the map.

C-vine generator
----------------

.. automodule:: cvinemarketgen.cvine
   :members: CVineGenerator
   :exclude-members: U_last, thetas_for_Corr_all_in_CVine

Fleishman generator
-------------------

.. automodule:: cvinemarketgen.fleishman
   :members: FleishmanGenerator

Moment matching
---------------

.. automodule:: cvinemarketgen.moment_match
   :members: MomentMatch, bicop_params, bicop_bounds, mvnormcdf

Bivariate copula tools
----------------------

.. automodule:: cvinemarketgen.copulas
   :members: CopulaTools

Market factor data
------------------

.. automodule:: cvinemarketgen.data
   :members: load_factor_data, factor_targets, fred_series, fama_french_monthly, yahoo_monthly_adjclose, FACTORS, DURATIONS
