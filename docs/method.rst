Method
======

This page maps the algorithms of the thesis chapter to the functions of the
package. It assumes the reader has the chapter; it does not re-derive it.

Notation
--------

:math:`N` assets, ordered so that asset 1 is the central node of the C-vine.
The copula of tree :math:`k` linking variable :math:`i` to variable :math:`k`,
given variables :math:`1, \dots, k-1`, is :math:`C_{ik|1:k-1}` with parameter
:math:`\theta_{ik|1:k-1}`, :math:`1 \le k < i \le N`; there are
:math:`\binom{N}{2}` of them. The parameters building variable :math:`i` form
the vector :math:`\boldsymbol{\theta}_i = (\theta_{i1}, \theta_{i2|1}, \dots,
\theta_{i,i-1|1:i-2})`.

In the code, every parameter quantity is an :math:`(N-1) \times (N-1)` matrix
whose entry ``[k-1, i-2]`` is :math:`\theta_{ik|1:k-1}`; column ``i-2`` is
:math:`\boldsymbol{\theta}_i`. A single-family entry is a float; a mixture
entry is ``[w, theta_1, theta_2]``.

Two generators
--------------

**Fleishman + Vale-Maurelli** (paper, Section 3). For each asset, the cubic
:math:`Y = a + bZ + cZ^2 + dZ^3` of a standard normal :math:`Z` is fitted to
the target skewness and excess kurtosis; the mean and volatility are applied
by an affine rescaling. For each pair, the Vale-Maurelli equation gives the
correlation :math:`\rho_Z` of the underlying normals that produces the target
correlation of the returns. The generator samples :math:`\mathbf{Z} \sim
\mathcal{N}(0, \Sigma_Z)` and transforms asset by asset. Its dependence is
inherited from the Gaussian vector: no tail dependence, no sign change.

.. code-block:: python

   fg = FleishmanGenerator()
   fg.fit(mu, sigma, skewness, exkurt, corr_target)     # coefficients, rho_Z, infeasible pairs
   sim, errors, coef_refit = fg.simulate(n, corr_tol)   # accept-reject as in Algorithm 5

**C-vine with Johnson SU marginals** (paper, Sections 4 and 5). The Johnson SU
quantile function is strictly increasing, so the copula of the uniforms is the
copula of the returns and the copula parameters can be calibrated to the
target Pearson correlations directly. The C-vine assigns one bivariate copula
per edge, selected from the data.

The pipeline
------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Paper
     - What it does
     - Function
   * - Algorithm 1, data and marginals
     - Targets :math:`\mu_i, \sigma_i, \Sigma^*` from the LTCMA table,
       :math:`\gamma_{1,i}, \gamma_{2,i}` from the history, Johnson SU parameters
       per asset, positive-definite projection of :math:`\Sigma^*`.
     - :meth:`~cvinemarketgen.cvine.CVineGenerator.load_data_and_setup`
   * - Algorithm 3, family selection and vine fit
     - For every edge :math:`(i,k)`: exceedance-correlation curve of the pair,
       characteristics :math:`(\ell, u, s, m)`, admissible families from the
       catalog :math:`\mathcal{C}` (mixtures :math:`\mathcal{M}` when
       non-monotone), maximum likelihood, best BIC, :math:`h`-function
       propagation, fallback lists :math:`\mathcal{A}, \mathcal{B}`.
     - :meth:`~cvinemarketgen.cvine.CVineGenerator.fit_and_structure_CVine`
   * - Algorithm 4, calibration
     - For :math:`i = 2, \dots, N`, minimize
       :math:`\sum_{k<i} (\mathrm{Corr}(Y_i(\boldsymbol{\theta}_i), Y_k) - \Sigma^*_{ik})^2`
       over :math:`\boldsymbol{\theta}_i` with the families of Algorithm 3
       fixed; try the fallback families if the residual stays above the
       tolerance.
     - :meth:`~cvinemarketgen.cvine.CVineGenerator.run_vine_optimization`
   * - Algorithm 5, scenario generation
     - Sample the calibrated vine, re-fit the marginals on the draw, accept the
       draw if it meets the tolerances on moments and correlations.
     - :meth:`~cvinemarketgen.cvine.CVineGenerator.run_multi_year_simulation`
   * - Algorithm 6, orchestrator
     - The four steps in sequence.
     - :meth:`~cvinemarketgen.cvine.CVineGenerator.run_complete_simulation`

Beyond the chapter (version 0.3)
--------------------------------

Two steps added for daily data and for factor-based use; they sit on top of
the pipeline and leave it unchanged.

* Daily dynamics: :func:`~cvinemarketgen.selection.select_dynamics` chooses a
  model per asset among the GARCH family and a Gaussian hidden Markov model,
  by i.i.d. tests on the residual layer, BIC among the candidates that pass,
  and a parametric-bootstrap Cramér-von Mises test of the selected model;
  :class:`~cvinemarketgen.hmm.GaussianHMM` is estimated by GenHMM1d and uses the
  Rosenblatt uniforms of the predictive mixture as its residual layer, and their
  exact inverse for paths.
* Assets on factors: :class:`~cvinemarketgen.factors.FactorModel` regresses
  assets on simulated factors, with Newey-West t-statistics and a Johnson SU
  residual per asset.

Family selection in short
-------------------------

For a pair, the exceedance correlation :math:`\hat\rho(z)` is the Pearson
correlation on the subsample where the central variable is below :math:`z`
(for :math:`z < 0`) or above :math:`z` (for :math:`z \ge 0`), :math:`z` in
standard deviations. Its summary statistics give four characteristics:

* :math:`\ell = 1` if the correlation is stronger on the left side,
* :math:`u = 1` if it is stronger on the right side,
* :math:`s = \mathrm{sign}(\Sigma^*_{ik})`,
* :math:`m = -1` if the curve changes sign (non-monotone dependence).

Each catalog family has a theoretical signature from its four-corner tail
dependence coefficients (Clayton: lower tail; Gumbel: upper tail; rotations
move the mass to the other corners; Gaussian: none). The admissible set is the
families whose signature matches :math:`(\ell, u, s)` when :math:`m = +1`, and
the mixtures matching :math:`(\ell, u)` when :math:`m = -1`. The retained
family is the admissible one with the lowest BIC.

:meth:`CVineGenerator.selected_edge_table <cvinemarketgen.cvine.CVineGenerator.selected_edge_table>`
prints the result edge by edge.

Synthetic markets
-----------------

To test the selection against a known truth, build a vine from chosen families
and simulate a history from it:

.. code-block:: python

   edges = {(2, 1): ('mixture', [('clayton', 270, 0.66), ('gumbel', 0, 1.84)], 0.70),
            (3, 1): ('gumbel', 180, 1.38), (4, 1): ('gaussian', 0, 0.05),
            (3, 2): ('gaussian', 0, -0.15), (4, 2): ('gaussian', 0, 0.30), (4, 3): ('gaussian', 0, 0.20)}
   spec = gen.make_vine_spec(assets, edges)
   hist = gen.simulate_known_vine(spec, jsu_params, mu, sigma, assets, n=3000, seed=1)

The keys ``(i, k)`` are the paper's indices, ``k < i``.

Runtime
-------

The calibration dominates: every evaluation of the objective simulates
``n_samples`` uniform vectors through the nested inverse :math:`h`-functions,
and a mixture edge needs a bisection per evaluation. The paper used 20,000
draws and a correlation tolerance of 0.02; the examples use 10,000 and 0.05,
which keeps a 4-asset example at a few minutes. With 10,000 draws the sampling
error of a correlation on a heavy-tailed pair is about 0.02 to 0.03, hence the
looser tolerance.
