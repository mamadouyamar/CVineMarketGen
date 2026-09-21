# The Notebook Series: Design (definitive)

**Date:** 2026-09-21 (replaces the 2026-09-20 draft). **Status:** approved in chat.

## Logic

The notebooks are a course. Each one is motivated by a limitation the previous
one runs into, introduces the one object that removes it, runs the whole process
again with it, and checks the result. No object is named before the reader needs
it. Readers: asset managers and portfolio construction teams, risk managers,
traders and systematic structurers, econometricians. Every notebook carries the
mathematics of each step before the cell that runs it (the object, its formula
with the symbols defined, the estimator or algorithm, how to read the output).

The process, always the same: **targets** (what the synthetic data must
reproduce), **fit** (learn the generator), **check** (diagnostics against the
targets and the history), **use** (scenarios, paths, a change, a stress). Both
generators, Fleishman and C-vine, are kept from notebook 2 on; Fleishman is the
benchmark.

## The chain

| | Title | Motivation (the limitation found before) | Introduces | Ends by finding |
|---|---|---|---|---|
| 01 | Matching moments and correlations | A history of a few assets is one sample; more data like it is needed | targets from a history (moments, correlations); the Fleishman generator; diagnostics; scenarios; a portfolio's distribution against the normal; i.i.d. paths; save | the empirical exceedance-correlation profile against the simulated one: the tails' dependence is missing |
| 02 | Tail dependence | that plot | what tail dependence is for a pair (exceedance correlation, families, tail coefficients, mixtures); the C-vine generator with families selected from the history; the process again; the plot again; Fleishman kept as benchmark | many assets: pairwise calibration does not scale |
| 03 | Many assets: factors | 100+ assets | the factor model (assets = betas on a few factors + residual); factor targets from history; both generators on the factors; assets mapped; a view as a shift of the factor targets | the factor histories are time series; one-period scenarios ignore that |
| 04 | The time series of a factor | autocorrelation, ARCH, regimes in one factor, daily and monthly | tests (Ljung-Box, ARCH-LM); AR-GARCH family and the HMM as candidates, selection by tests and BIC, bootstrap GoF; the generalized error (standardized residual for GARCH, Rosenblatt uniform for the HMM), i.i.d.; paths by inversion from the last state | each factor now has its own dynamics; how do they go together? |
| 05 | Integration | | factors with their dynamics, generalized errors joined by the C-vine or Fleishman, assets on the factors, paths at the horizon; checks against the history's dynamics and dependence | the principle behind it, and dynamics shared by a subset of variables |
| 06 | The generalized-error principle | | statement of the principle (Thioub's master's thesis, SSRN 3347348; article 4): any dynamics, univariate or multivariate, produces i.i.d. generalized errors, the copula generator simulates those, the dynamics rebuild the paths; the VECM/VAR block as the multivariate case on a subset (nominal and real yields), rank and lags by tests, paths in levels | fixed income needs the curve, not one yield |
| 07 | The yield curve and fixed income | yields are the first risk factor and bond returns are functions of the curve; coverage of every maturity; scenarios stated in yields; horizon; the curve as a state variable | Nelson-Siegel factors as a block (instance of 06); curves along paths; fixed income as a child computed by pricing (discount factors, par yields, constant-maturity and zero returns); the duration factor of 03 replaced by the priced return; historical priced returns against realized indices; a 100 bp level shock propagated by projection | variables that have causes |
| 08 | Variables with causes | the exchange rate follows rates and oil, inflation follows oil, slack and expectations, output is quarterly | the structural layer (parents, children with their own generalized error, order); inflation, price level, real returns; the bridge to a quarterly series; one oil shock reaching all three | assembling everything |
| 09 | One market | | the three layers (economy, factors filtered or priced or derived, assets on factors or priced) in one simulation; 24-month paths; the assets' moments against history; two propagated scenarios; a portfolio's nominal and real distribution | what the tail of that portfolio holds |
| 10 | Tail risk and stress tests | the risk manager's question | VaR and expected shortfall on the market of 09 against the normal; P&L by position and by factor under the stresses; drawdowns along paths; reverse stress by conditional expectation | the daily reader's question |
| 11 | Daily paths and backtesting | one history for a strategy | the daily market of 04 and 05 on several assets, a thousand synthetic years; a strategy as a weight function run on every year; Sharpe, drawdown and turnover distributions against the one history | path-dependent payoffs |
| 12 | Structured payoffs | | payoff functions on paths (protected note, autocall, volatility target); their distribution and hedging cost under realistic dynamics against the lognormal | |

Readers' paths: asset managers 01-09; risk managers 01, 02, 06-10; traders and
structurers 01, 02, 04, 11, 12; econometricians 01-08, 11.

## Mapping from the current notebooks

Current 01, 03 -> 01. Current 02, 04, 05 -> 02. Current 06, 08 -> 03 (factors),
with the factor construction of 06 kept. Current 07, 09 -> 04 (and 11 for the
backtest). Current 10 -> 06. Current 12 -> 07. Current 11, 13, 14 -> 08. New:
05, 09, 10, 11, 12. The companion papers stay one per model and are the
references; the notebooks are the course. Old notebooks are deleted as their
content is absorbed; README, docs, badges and papers updated at each step.

## Package additions (when the notebook needs them)

- 03: `FactorModel.fit` on each asset's own sample.
- 09: `assembly.py` (`Market` of the three layers, `simulate`, `scenario` with projection of a block's other innovations).
- 10: `risk.py` (`var_es`, `drawdowns`, `reverse_stress`, `pnl_by_factor`).
- 11: `backtest.py` (`run_strategy`, `sharpe`, `max_drawdown`, `turnover`, `summary`).
- 12: `payoffs.py` (`protected_note`, `autocall`, `vol_target`, `lognormal_benchmark`).

## Execution

One notebook at a time, in order: write, execute, check the numbers, delete the
superseded notebooks, update README/docs, commit. Papers and article 4 updated
at the end (article 4 gains the generalized-error section with the thesis
citation and the three-layer market).
