# Specification layer: from the minimum input to a full view

Date: 2026-09-26. Status: approved in conversation, implementation follows.

## Purpose

Notebook 03 is rebuilt for the analyst who runs simulations and takes the
scenarios into their own module. The organizing principle: every factor and
every asset is described by what the analyst knows about it, and every gap is
filled by a stated rule, from history when there is one, from a default
otherwise. This document fixes the rules and the API that carries them.

Four kinds of information an object can carry: a return history, a mean and
volatility target, exposures (assets only), correlation targets. The minimum
input is a factor history and an asset history. Everything below extends it.

## Factors

### F1. History only (existing)
`Targets.from_history(F)`: mean, volatility, skewness, kurtosis and correlation
matrix from the sample. Families and marginals fitted from it by `CVineMarket`.

### F2. Views on a factor with a history (existing)
Edit `t.mean`, `t.vol`, `t.corr` entries; the market is refitted on the edited
targets. Correlation edits are projected on the nearest positive-definite
matrix with a warning when needed.

### F3. A factor without a history (new): `Targets.add_factor`
```
t2 = t.add_factor('Illiquidity', mean=0.04, vol=0.08, corr={'Equity DM': 0.30, 'Credit': 0.50},
                  skew=0.0, kurt=3.0, annualized=True, periods=12)
```
Given: mean and volatility (annual by default, converted to the data's
frequency), correlations with a subset of the existing factors, optional
skewness and kurtosis (default normal). Unknown: correlations with the other
factors, the history.

Rule for the missing correlations: the maximum-determinant completion. Among
all valid correlation matrices that agree with the given entries, take the one
with the largest determinant. It is the unique completion in which every
unspecified pair has zero partial correlation given the specified entries: the
new factor is related to the unspecified factors only through the specified
ones, nothing is invented. For one new factor with specified set S and
unspecified set U it has the closed form

    r_U = R_US R_SS^-1 r_S

with R the correlation matrix of the existing factors. For several new factors
the general problem is solved numerically (maximize log det over the free
entries, convex; started from the closed form applied factor by factor). If
the given entries themselves admit no valid completion (r_S' R_SS^-1 r_S >= 1),
`ValueError` names the bound.

The returned `Targets` has the new factor appended, `history` unchanged (the
new column has none), `synthetic = ['Illiquidity']`, and a
`completion_report` DataFrame with one row per (new factor, existing factor):
target correlation, `source` = `given` or `completed`.

Simulation. `FleishmanMarket` needs nothing: it works from moments and the
completed matrix. `CVineMarket` fits the vine on the factors with a history
and attaches each synthetic factor after the draw by a Gaussian conditional
draw on the normal scale:

    z_h  = Phi^-1(rank(x_h) / (n + 1))            normal scores of the simulated historical factors
    z_s  = c' z_h + sqrt(1 - r' R_hh^-1 r) * eps,  c = R_hh^-1 r,  eps ~ N(0, 1)
    x_s  = Q_JSU(Phi(z_s); mean, vol, skew, kurt)

`r` holds the (completed) correlations of the synthetic factor with the
historical ones. The marginal transform pulls the linear correlation slightly
below the Gaussian one; `r` is corrected once by the ratio target/measured on a
pilot draw so the simulated correlation lands on target. `Diagnostics` then
compares all factors, synthetic included. Save/load carry `synthetic` and the
completion report through `Targets.to_dict`.

## Assets

The factor model is `r_j = alpha_j + sum_k beta_jk f_k + e_j`. The residuals
are independent across assets unless a pair correlation is targeted (A6).

### A1. History only, no view (existing)
`FactorModel(R, F)`: every asset regressed on every factor, Newey-West
t-statistics, Johnson SU residual per asset.

### A2. Exposures, three forms (generalized)
`exposures={asset: spec}` with three accepted specs:

- **list of factors** (existing): the allowed set. Regression on those, zero
  elsewhere.
- **dict `{factor: value}`** (new): given betas. They are fixed, and the
  remainder `r_j - sum_{given} beta f` is regressed on the factors not named.
  Start from the regression, override what the analyst knows.
- **dict with `'fit'` values**: `{'Equity DM': 1.0, 'Credit': 'fit'}` fixes
  Equity DM, fits Credit, and sets every other factor to zero. A dict that
  contains at least one `'fit'` is an allowed-set specification with some betas
  pinned; a dict with only numbers regresses on all the unnamed factors.

Assets not in the dict load on every factor. A given beta has NaN standard
error and t-statistic. `fm.beta_source` (DataFrame assets x factors) records
`given`, `fitted` or `zero` for every entry; it is printed by the report.

A given beta on a synthetic factor (F3) is allowed only as a number, never
`'fit'`: nothing can be regressed on a factor without history.

### A3. An asset without a history (new): `FactorModel.add_asset`
```
fm2 = fm.add_asset('PrivateCredit', mean=0.07, vol=0.10, exposures={'Credit': 0.6, 'Equity DM': 0.2},
                   skew=0.0, kurt=3.0, annualized=True, periods=12)
```
Given: mean, volatility, exposures (numbers), optional residual shape. Rule:

    alpha     = m - beta' mu_f
    sigma_e^2 = v^2 - beta' Sigma_f beta      (ValueError with the floor sqrt(beta' Sigma_f beta) if negative)
    e         ~ Johnson SU(0, sigma_e, skew, kurt), Gaussian by default

with `mu_f`, `Sigma_f` the factor moments of the model (see "factor moments"
below). Without exposures the asset is pure residual and `asset_source` says
so. The asset is appended to `alpha`, `beta`, `resid_vol`, `resid_params`,
`report` (R2 and t-statistics NaN); `asset_source` (Series) is `history` or
`spec` per asset. Several assets: `add_assets(table)` with a DataFrame of
mean, vol, skew, kurt and one column per factor.

### A4. An asset with a history loading on a synthetic factor
Regression on the historical factors as in A2, with the given beta on the
synthetic factor added to the systematic part. To keep the asset's sample
volatility the residual variance is reduced,

    sigma_e^2 = s_j^2 - beta' Sigma_f beta

with `Sigma_f` including the synthetic factor, subject to the same floor. The
report shows the sample residual volatility and the reduced one.

### A5. Mean and volatility targets (existing): `with_targets`
alpha for the mean, residual scale for the volatility, floor at the systematic
volatility. Works on `spec` assets too.

### A6. Pair correlation targets (new): `with_targets(..., pair_correlations=...)`
```
fm3 = fm.with_targets(assumptions, pair_correlations={('HYG', 'LQD'): 0.85, ('GLD', 'SLV'): 0.70})
```
For the pair (i, j), with the volatilities and betas fixed by the earlier steps:

    rho_sys = beta_i' Sigma_f beta_j / (s_i s_j)
    range   = [rho_sys - w, rho_sys + w],  w = sqrt((1 - R2_i)(1 - R2_j)),  R2_i = beta_i' Sigma_f beta_i / s_i^2
    d_ij    = rho* s_i s_j - beta_i' Sigma_f beta_j
    rho_e   = d_ij / (sigma_e_i sigma_e_j)                 must lie in [-1, 1]

A target outside the range raises `ValueError` with the range. The residual
correlation matrix `resid_corr` (identity plus the specified entries) must be
positive semidefinite; otherwise `ValueError` names the smallest eigenvalue
and the assets involved. `target_report` gains a `pairs` table: target,
systematic correlation, eligible range, residual correlation used.

Simulation of correlated residuals: a Gaussian copula with correlation
`resid_corr` and the assets' Johnson SU marginals. The Gaussian correlation is
corrected once so that the linear correlation of the residuals hits `rho_e`
(pilot draw, ratio correction, as for F3). Residuals of assets outside the
specified pairs stay independent.

The beta route (changing exposures to move a correlation) is not implemented
as a solver: it is the exposure table of A2, and the notebook names it as the
alternative with its spillover.

## Factor moments used by the asset layer

`mu_f`, `Sigma_f` come from the factor `Targets` passed to `FactorModel`
(`factor_targets=`) when given, else from the sample factor history. With a
synthetic factor they must come from `factor_targets`, since the sample has no
column for it. `FactorModel(R, F, factor_targets=t2)` takes the factor names
from `t2.assets`; regressions run on the columns of `F`.

## The specification report

`fm.spec_report()` prints, per asset: source (history/spec), the number of
given, fitted and zero betas, whether a mean, a volatility or a pair target
was applied, the sample and current residual volatility. Per factor (from the
`Targets`): source (history/synthetic), number of given and completed
correlations. It is what each section of the notebook prints after a change,
so the analyst sees what the rules filled for them.

## Save / load

`FactorModel.save` adds `beta_source`, `asset_source`, `resid_corr`,
`pair_report`. `Targets.to_dict` adds `synthetic` and `completion_report`.
Old files without these keys load with the defaults (all fitted, all history,
identity).

## Tests

- completion: closed form recovered by the optimizer; zero partial
  correlations; infeasible given entries raise.
- attachment: simulated synthetic factor hits its mean, vol and target
  correlations within Monte Carlo error on 100,000 draws, for a skewed marginal.
- exposures: dict form fixes the given beta exactly and recovers the others;
  `'fit'` form zeroes the unnamed; `beta_source` correct; NaN t-statistics on
  given betas; error on `'fit'` for a synthetic factor.
- add_asset: mean and vol recovered by simulation; floor error; pure residual
  flagged.
- pair correlations: simulated correlation within 0.01 of target for a skewed
  residual; range error message; PSD error on a bad triangle; other pairs
  unchanged; volatilities unchanged.
- save/load round trip with every new field.

## Notebook 03 sections (the deliverable)

0 setup and input contract; 1 minimum input full run; 2 views on factors;
3 exposures from none to complete; 4 assets without history; 5 factors
without history; 6 targets on assets (mean, vol, pairs); 7 check, given
against simulated; 8 use (simulate, retrieve, aggregate, save, export);
appendix method notes. Each section opens with the mathematics of its rule
and prints the specification report after the change.
