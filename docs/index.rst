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

Your own assumptions, nothing else, ten lines:

.. code-block:: python

   from cvinemarketgen import Targets, CVineMarket

   t = Targets(mean={'Equity': 0.07, 'Bonds': 0.03, 'Gold': 0.04},
               vol={'Equity': 0.16, 'Bonds': 0.05, 'Gold': 0.15},
               corr=[[1, -0.1, 0.05], [-0.1, 1, 0.3], [0.05, 0.3, 1]],
               assets=['Equity', 'Bonds', 'Gold'])
   cv = CVineMarket(t, families='gaussian').fit()
   X = cv.simulate(25000, seed=1)          # DataFrame, one row per scenario
   cv.diagnostics(X).summary()             # target vs simulated moments and correlations
   P = cv.simulate_paths(1000, 10, seed=1) # 1000 paths of 10 periods

Add skewness and kurtosis, a history to select tail-dependent copula families
from, or daily data with GARCH dynamics: the :doc:`userguide` covers each case,
and the notebooks show them with outputs.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   userguide
   method
   examples
   api
   changelog

Citation
--------

.. code-block:: text

   Thioub, M. Y. (2026). CVineMarketGen: C-vine copula financial market generator with
   moment and tail dependence targeting (Version 0.2.0) [Computer software].
   https://github.com/mamadouyamar/CVineMarketGen

The repository's ``CITATION.cff`` holds the citation metadata.
