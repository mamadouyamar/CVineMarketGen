# The Notebook Series as a Course: Design

**Date:** 2026-09-20. **Status:** for approval.

## Why

The fourteen notebooks were built one feature at a time. Each is a demo of one
object, with SPY alongside, and the series has no spine: objects are named
before they are introduced ("Targets is the object both generators take" in
01, before any generator is shown), the econometric models of 10 to 14 never
reach an asset, and the factor model of 08 is never connected to them. The
notebooks are going to asset managers. They must read as a course with two
tracks, each notebook answering a question its reader asks, using only what
earlier notebooks defined, and ending in a deliverable the reader can use.

## The readers

- **Calibration and portfolio construction teams.** Monthly or annual data,
  horizons of months to years. They want scenario sets consistent with their
  capital-market assumptions, with realistic tails; views and stresses on the
  economy propagated to every asset class; fixed income repriced from the
  curve; real returns.
- **Traders and systematic strategy builders.** Daily data, horizons of days.
  They want many synthetic years of daily paths with the right clustering,
  tails and regimes, to backtest and stress a strategy beyond the one history.

Both need the core: targets, marginals, copula, generator, and the idea of a
filter to an i.i.d. residual layer with an exact inverse from the last state.

## The series (new numbering)

Every notebook has the same arc. Opening cell: the question, the answer in one
paragraph, "what you need from earlier notebooks", "what this notebook adds".
Each section: the reader's question in plain words, the idea in one sentence,
the mathematics (already written), the code, how to read the output. Closing
cell: what was learned, which notebook comes next and why. No object is named
before the section that introduces it. Titles carry no em-dash: "01 Getting
started".

### Part I. The generator (everyone)

| New | Old | Question | Introduces | Deliverable |
|---|---|---|---|---|
| 01 Getting started | 01 | I have a table of means, volatilities and correlations. How do I get scenarios consistent with it, with fat tails? | the problem; `Targets` as the table; the C-vine generator as a black box; `simulate`, `diagnostics`; one-period scenarios; `Paths` for i.i.d. periods; save/load | a 25,000-scenario set from four LTCMA rows, its diagnostics |
| 02 One asset, one pair | 02 (+ the "skewness and kurtosis targets" section of old 01) | Why not a multivariate normal? What do fat tails and tail dependence mean for one asset and for a pair? | Johnson SU marginal; the copula of a pair; exceedance correlation; families and their tails; mixtures; adding skewness and kurtosis to the targets | the reader can read an exceedance curve and a family table |
| 03 The Fleishman generator | 03 | Is there a faster way, and what does it give up? | `FleishmanMarket`; Gaussian dependence; the comparison of exceedance curves | the benchmark and its limits |
| 04 Inside the C-vine generator | 04 | What happens between the targets and the scenarios? | the four algorithms on a known truth | trust in the black box of 01 |
| 05 From scenarios to paths | new (concepts from old 07) | Scenarios are one period. How do I get paths, when returns are not independent over time? | the filter $\Psi$: a model with an invertible one-step conditional distribution; the residual layer; the generator on the residual layer; the inverse from the last observed state; the same construction at monthly (AR(1) on a yield) and daily (GARCH on an equity) frequency; the i.i.d. tests | the reader knows what "dynamics" means in every later notebook |

### Part II. Portfolio construction (monthly)

| New | Old | Question | Introduces | Deliverable |
|---|---|---|---|---|
| 06 Capital-market assumptions as targets | 05 | My assumptions are a published table. How do I use it, and where do the higher moments come from? | `Targets.from_ltcma` with a history for skewness and kurtosis; both generators on the same table; the choice between them | the LTCMA workflow |
| 07 Targets from a history: macro factors | 06 | I have no table, I have data. | `Targets.from_history`; the seven factors built from public data; monthly to annual | a factor scenario set from history |
| 08 Variables in levels: a block of yields | 10 | Yields are levels, not returns, and they move together. How are they filtered? | blocks (VECM/VAR), cointegration, paths in levels from the last observed values; the breakeven kept anchored | rates paths in levels |
| 09 The yield curve and fixed income | 12 | I need the whole curve, and bond prices, not one yield. | Nelson-Siegel factors as a block; curves and bond returns along the paths | Treasury returns at any maturity, priced |
| 10 A variable explained by others: the exchange rate | 11 | The Canadian dollar follows the rate differential and oil. | the structural layer (parents, children, order); scenarios on a parent | FX paths and a propagated oil scenario |
| 11 Inflation and real returns | 13 | Real returns need inflation, and inflation has causes. | inflation as a child; price levels, year-on-year, `deflate`; `exclude` for one-off months | inflation paths, real returns |
| 12 Output from a monthly market | 14 | I want GDP, which is quarterly. | the bridge equation; recession probabilities | GDP paths |
| 13 Assets on factors | 08 | My assets are explained by a few factors. How do factor scenarios become asset scenarios, and how do I express a view? | `FactorModel` on the factors of the assembled market (Section below), including priced factors; views as location shifts | asset scenarios and paths from factor paths |
| 14 One market: from the economy to the portfolio | new | Put it together: the economy, the market factors and my assets in one simulation, with scenarios that propagate. | the three-layer market (Section below); the assembly helper | 24-month asset paths, real returns, two propagated scenarios, a portfolio's distribution |

### Part III. Systematic strategies (daily)

| New | Old | Question | Introduces | Deliverable |
|---|---|---|---|---|
| 15 Daily dynamics chosen per asset | 07 | Daily returns cluster their variance and have regimes. Which model for each asset, chosen by the data? | GARCH family and HMM candidates; tests, BIC, bootstrap GoF; `dynamics='auto'`; 1,000 daily years | a fitted daily market, saved |
| 16 Regimes | 09 | One asset with regimes, in depth. | `GaussianHMM`: probabilities, the Rosenblatt residual layer, the GoF test | regime paths |
| 17 Backtesting on synthetic years | new | My strategy has one history. How does it do on a thousand? | the backtest helper: a strategy as a weight function of past returns, run on every path; the distribution of Sharpe ratio, drawdown and turnover against the one history | the trader's deliverable |

Renumbering map (old -> new): 01->01, 02->02, 03->03, 04->04, 05->06, 06->07,
07->15, 08->13, 09->16, 10->08, 11->10, 12->09, 13->11, 14->12; new 05, 14, 17.
Files are renamed with `git mv`; Colab badges, README table, `docs/examples.rst`,
the papers (`\package` notebook references), the Overleaf README and the
memory notes are updated to the new numbers.

## Notebook 14: the assembled market

**Frequency and sample.** Monthly, October 1993 to the last complete quarter;
assets with shorter histories (TIP 2003, EEM 2003, VNQ 2004, DBC 2006, HYG
2007) are regressed on their own available sample.

**Layer 1, the economy** (all in one `AssetDynamics`, one vine):

| Variables | Model | Source |
|---|---|---|
| level, slope, curvature | block `'vecm(q=1)'` | Nelson-Siegel on DGS1..DGS30 (notebook 09) |
| log_oil, unrate, mich | block `'vecm'` (rank by the test) | WTI, UNRATE, MICH |
| d_payems, d_indpro | block `'var'` | PAYEMS, INDPRO growth |
| rate_diff (US minus Canada 10-year) | `'ar1'` | DGS10, IRLTLT01CAM156N |
| log_cpi | child `'ecm(log_oil, unrate, mich; lags=2)'` | CPIAUCSL; inflation = 1200 * d log_cpi, price level direct |
| log_fx (USDCAD) | child `'ecm(rate_diff, log_oil)'` | DEXCAUS |
| GDP growth (quarterly) | `Bridge(['d_payems', 'd_indpro', 'd_unrate'])`, `d_unrate` derived from the unrate path | GDPC1 |

`exclude=['2020-03', ..., '2020-06']` on the residual layer.

**Layer 2, the market factors**, in the same vine when filtered, priced from
layer 1 when a pricing exists:

| Factor | How |
|---|---|
| equity_dm (SPY), equity_em (EEM), commodities (DBC), gold (GLD) | filtered, `'ar1-garch'`, columns of the same market |
| duration | priced: 10-year constant-maturity return from the simulated curve (`curve_returns`) |
| credit | HYG return minus the 5-year constant-maturity return, filtered `'ar1-garch'` |
| fx | derived: change of log_fx |
| inflation | derived: 1200 * change of log_cpi |

**Layer 3, the assets**: IWM, EFA, TLT, IEF, TIP, LQD, VNQ (plus the factor
ETFs themselves as investable) through `FactorModel` on the eight factors;
Treasuries alternatively priced directly from the curve (TLT as the 20-year
constant-maturity par bond, IEF as the 7-year), the two shown side by side.

**Outputs**: 24-month paths of every variable; the assets' simulated mean,
volatility and correlation against their history; real cumulative returns at
12 and 24 months; recession probability; two scenarios propagated through
coefficients, the block innovations shifted by their projection on the shocked
one: oil up 50 percent over six months (inflation, real returns, USDCAD,
commodities, TIP against TLT) and the curve's level up 100 basis points over
six months (every Treasury and the duration exposure of every asset); a 60/40
portfolio's nominal and real distribution under baseline and scenarios.

## Package additions

- `cvinemarketgen/assembly.py`: `Market(core, derived=None, priced=None, factor_model=None, bridge=None)`:
  `core` a fitted `CVineMarket` (layer 1 plus filtered factors); `derived` a
  dict `name -> function(Paths) -> array (n, h)` (differences, inflation from
  log CPI); `priced` a dict `name -> function(Paths) -> array` (curve returns);
  `factor_model` a `FactorModel` whose factors are columns of the core, derived
  or priced; `bridge` a `(Bridge, parents-as-derived-names, history)` triple.
  `simulate(n, h, seed)` returns a `MarketPaths` object with `.economy`,
  `.factors`, `.assets` (`Paths`) and `.gdp`; `scenario(shifts, months, projection=True)`
  applies drifts to named innovations (in the variables' units), projecting the
  other innovations of the same block when `projection=True`, and returns the
  same object for baseline and scenario. Tests on a synthetic core.
- `FactorModel.fit`: each asset regressed on the factors over its own
  non-missing sample (currently a common sample); `report` gains `n_obs`.
- `Structural`: `'ecm'` with `lags` on the child already exists; no change.
- `cvinemarketgen/backtest.py`: `run_strategy(paths, weights_fn, rebalance='M')`
  (weights from past returns, returns per path), `sharpe`, `max_drawdown`,
  `turnover`, `summary` (distribution across paths against the history's
  values). Tests on synthetic paths.

## Papers

- Notebook references updated to the new numbers in every paper and in the
  Overleaf README.
- A new short paper for notebook 14, "One market: the economy, the factors and
  the portfolio", stating the three-layer construction and its assumptions
  (priced factors as deterministic children, factor-model residuals
  independent, the bridge innovation independent), with the propagated
  scenarios as the application.
- A new short paper for notebook 17, "Backtesting on synthetic years".
- `article4-principle.tex` gains a section "Three layers" that states the
  assembled market as the general case of the filters, blocks and children.

## Execution phases (one plan each, inline, one commit per task)

- A. Renumbering and references (mechanical, no content change).
- B. Part I: rewrite 01 to 04 to the arc (01 rebuilt around the C-vine; 02
  absorbs the higher-moment targets; 03 the benchmark), write 05, re-execute
  01 to 05.
- C. Package additions (`assembly.py`, `FactorModel` own-sample fit,
  `backtest.py`) with tests.
- D. Part II: rewrite the openings, transitions and closings of 06 to 12 to the
  arc (markdown-only), rewrite 13 on the assembled market's factors and write
  14 (both executed), the paper of 14.
- E. Part III: rewrite 15 and 16 to the arc, write 17 (executed), the paper of 17.
- F. Reading guide in README and docs (the three parts, the two readers, the
  order), article-4 section, papers' references, Overleaf zip, upload staging.

## Out of scope

Intraday data; a second currency; credit curves; MIDAS; re-estimating the
article-3 engine.
