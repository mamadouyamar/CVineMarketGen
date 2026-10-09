# Earlier notebooks

These five were written against the previous structure of the package, before
the standard in [`docs/notebook-principles.md`](../../docs/notebook-principles.md)
existed. They are kept because each one is the only worked example of a module
the course has not reached yet, and they are here rather than in `examples/`
because they do not obey that standard: no contents, no numbered sections, and
prose written as an essay rather than a manual.

| Notebook | The layer it shows | Module |
|---|---|---|
| `10_block_filters_vecm` | two yields filtered as one block, paths in levels, the breakeven anchored | `BlockVECM`, `BlockVAR` |
| `11_structural_layer_fx` | USDCAD on a rate differential and oil, parents simulated first | `Structural` |
| `12_yield_curve_fixed_income` | the Treasury curve as level, slope and curvature, bonds priced along the paths | `NelsonSiegel`, `PCACurve` |
| `13_inflation` | headline and core CPI as children of oil, unemployment and expectations | `Structural`, `yoy` |
| `14_gdp_bridge` | quarterly growth from monthly paths, recession probabilities | `Bridge` |

They run and their outputs are current. Two caveats for a reader.

**They cite a notebook that no longer exists.** Notebooks 10 and 11 refer to
"notebook 08", which mapped assets onto macro factors. The course notebook
that now does that job, on a universe of seventy assets, is
[`03_many_assets_factors`](../03_many_assets_factors.ipynb).

**Their numbering is not the course's.** The course runs 01 to 04 so far, and
these keep the numbers they were written with. When a course notebook absorbs
one of these layers, that notebook is deleted from here.
