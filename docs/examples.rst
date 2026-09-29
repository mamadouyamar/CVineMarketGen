Examples
========

Two kinds of page. The **course** runs in order, each notebook motivated by
what the previous one could not do; start with 01. The **recipes** are
standalone, one lever each, and can be read in any order once notebook 03 has
been seen. All of them carry a Colab badge and run on the data the package
downloads itself.

They are written to one standard, ``docs/notebook-principles.md``: explain then
show, one exhibit at a time; draw an input as the artifact the reader types,
and show a modification in its context; every number in the prose comes from
the run beside it. The derivations live in the methods note
(:download:`methods-note.pdf <_static/methods-note.pdf>`), so that a notebook
states a rule and shows it working instead of arguing for it.

The course
----------

.. toctree::
   :maxdepth: 1

   examples/01_matching_moments_and_correlations
   examples/02_tail_dependence
   examples/03_many_assets_factors
   examples/04_time_series_of_a_factor

Recipes
-------

One workbook cell, from what you type to the checked scenario, in about twelve
seconds. Each one starts from the same base file, ``examples/inputs/
recipe_base.xlsx``: eight funds, six factors, and nothing imposed.

.. toctree::
   :maxdepth: 1

   examples/recipes/r1_exposures
   examples/recipes/r2_asset_without_history
   examples/recipes/r3_asset_targets
   examples/recipes/r4_pair_correlation
   examples/recipes/r5_factor_views
   examples/recipes/r6_factor_without_history

Earlier notebooks
-----------------

Written against the previous structure of the package. They cover the layers
the new course has not reached yet: block filters, the structural layer, the
yield curve, inflation and the GDP bridge.

.. toctree::
   :maxdepth: 1

   examples/05_ltcma_targeting
   examples/10_block_filters_vecm
   examples/11_structural_layer_fx
   examples/12_yield_curve_fixed_income
   examples/13_inflation
   examples/14_gdp_bridge
