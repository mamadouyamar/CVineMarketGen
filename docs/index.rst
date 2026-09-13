CVineMarketGen
==============

C-vine copula financial market generator with moment and tail dependence
targeting, and the Fleishman / Vale-Maurelli benchmark. This is the code of
the third article of the thesis, *Vine-Copula Based Financial Market Simulation
with Moment and Tail Dependence Targeting*.

The package simulates multivariate asset returns whose first four moments,
pairwise correlations and tail dependence match prescribed targets, such as
long-term capital market assumptions (LTCMAs), through two generators:

* the **Fleishman + Vale-Maurelli** generator, matching the first four moments
  and the correlation matrix with Gaussian dependence
  (:class:`~cvinemarketgen.fleishman.FleishmanGenerator`);
* the **C-vine copula** generator with Johnson SU marginals, which in addition
  reproduces the tail dependence and the sign-changing conditional correlations
  observed in historical data, with copula families selected from the
  exceedance-correlation curve of each pair, mixtures included
  (:class:`~cvinemarketgen.cvine.CVineGenerator`).

Installation
------------

.. code-block:: sh

   pip install git+https://github.com/mamadouyamar/CVineMarketGen.git

Requires Python 3.8 or later with numpy, scipy, pandas, matplotlib, requests
and `pyvinecopulib <https://github.com/vinecopulib/pyvinecopulib>`_.

First simulation
----------------

Thirty lines that build a synthetic 4-asset market from known copula families,
select the families back, calibrate the vine to J.P. Morgan's 2024 targets, and
simulate 25,000 scenarios:

.. code-block:: python

   import numpy as np, pandas as pd
   from cvinemarketgen import CVineGenerator

   assets = ['U.S. Large Cap', 'U.S. Long Treasuries', 'Commodities', 'Gold']
   ltcma = pd.read_csv('data/jpm_ltcma_2024.csv', index_col='Assets').loc[assets]

   gen = CVineGenerator(tol_opt=1e-10, n_samples=10000,
                        use_ncs_on_firsttree=False, use_ncs_on_deepertrees=False,
                        use_mixture_on_firsttree=True, use_mixture_on_deepertrees=False,
                        force_try_ncscopula=False, tol_for_optimization_func=1e-6)

   edges = {(2, 1): ('mixture', [('clayton', 270, 0.66), ('gumbel', 0, 1.84)], 0.70),
            (3, 1): ('gumbel', 180, 1.38), (4, 1): ('gaussian', 0, 0.05),
            (3, 2): ('gaussian', 0, -0.15), (4, 2): ('gaussian', 0, 0.30), (4, 3): ('gaussian', 0, 0.20)}
   jsu = {'U.S. Large Cap': (2.570, 2.179, 3.540, 2.646), 'U.S. Long Treasuries': (-0.096, -0.095, 2.279, 2.062),
          'Commodities': (0.611, 0.589, 2.092, 1.774), 'Gold': (-0.523, -0.516, 2.893, 2.675)}
   spec = gen.make_vine_spec(assets, edges)
   hist = gen.simulate_known_vine(spec, jsu, ltcma['Arithmetic Mean'], ltcma['Volatility'], assets, n=3000, seed=1)

   np.random.seed(1)
   res = gen.run_complete_simulation(ltcma=ltcma[['Arithmetic Mean', 'Volatility'] + assets],
                                     historical_data=hist, asset_order=assets,
                                     n_year=1, n_per_year=25000, corr_tol=5e-2)
   print(gen.selected_edge_table(assets, res['CVinefitresults'], res['vine_results']).to_string(index=False))
   sim = res['simulated_years'][0]          # 25000 x 4 DataFrame

The two example notebooks show the full workflow with printed outputs and the
exceedance-correlation figures.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   method
   examples
   api
   changelog

Citation
--------

.. code-block:: text

   Thioub, M. Y. (2026). CVineMarketGen: C-vine copula financial market generator with
   moment and tail dependence targeting (Version 0.1.0) [Computer software].
   https://github.com/mamadouyamar/CVineMarketGen

The repository's ``CITATION.cff`` holds the citation metadata.
