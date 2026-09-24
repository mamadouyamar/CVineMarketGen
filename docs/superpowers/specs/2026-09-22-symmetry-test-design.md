# A Symmetry Test Before the Family Classification: Design

**Date:** 2026-09-22. **Status:** approved in chat, implemented the same day.

## The defect

Algorithm 3 of article 3 classifies each pair from the shape of its exceedance
correlation profile (stronger left, stronger right, sign change) and restricts
the copula candidates to the families with that tail signature. The shape is
estimated: at $z = -1$ the conditional correlation uses about 16 percent of the
sample, 40 observations out of 260, with a standard error near 0.15. On pairs
simulated from a Gaussian copula with 260 observations (no tail dependence, no
asymmetry) the rule selected a mixture 73 percent of the time at correlation
-0.09 and a Gumbel or a Clayton 80 percent of the time at correlation 0.6.

## The change

1. `tail_asymmetry_test(x, y)`: statistic = mean of the profile for $z < 0$
   minus mean for $z > 0$ on the classification grid; 500 bootstrap resamples;
   symmetric when the central 90 percent interval contains zero.
2. In `estimate_CVine_preselected_V2` (first and deeper trees), a symmetric pair
   skips the classification: candidates Gaussian and Student t, BIC. Asymmetric
   pairs unchanged. Flag `symmetry_test` (default True) on the engine,
   `CVineMarket` and `select_family`; the test is stored in
   `trees_prespecification` and exposed as `CVineMarket.classification`.
3. The Student t exposed as a family: `('student', 0, rho, df)` in the families
   dict, `make_vine_spec` 4-tuple, `_edge_records`/`load` carry two parameters,
   the edge table shows `rho, df`.

The paper's rule is the special case of a test that always rejects; every
result of article 3 stands. The paragraph for the paper is in `docs/method.rst`.

## Addendum 2026-09-23: the first tree on the copula scale

The paper computes the first tree's exceedance profile on the standardized
returns (article-3.tex, "obtained from the raw returns when k = 1") and the
deeper trees' on Phi^{-1} of the h-function values. On returns the profile
slopes with the skewness of the conditioning variable whatever the copula, so
marginal skewness was read as copula asymmetry (SPY/EFA: 0.19 on returns, 0.10
on normal scores; on the copula scale the pair is symmetric and Gaussian by BIC).
Change: tree 1 classifies and tests on the pseudo-observations through
Phi^{-1}, like the deeper trees; `classify_pair`, `select_family`,
`tail_asymmetry_test` take `copula_scale=True` by default. Reported profiles in
notebooks and papers stay on returns (the paper's definition). Sentence to amend
in article 3, Algorithm 3, step "Empirical tail signature": "obtained from
Phi^{-1} of the pseudo-observations when k = 1 and from Phi^{-1}(H^{(k-1)}) when
k >= 2".
