# How the notebooks of this course are written

Two parts: how a notebook is built, and how its sentences are written. The
rules were settled while writing notebooks 03 and 04, and hold for every
notebook of the series. They are not style preferences.
Each one exists because breaking it produced a notebook that misled a reader,
and several of them caught defects in the package itself.

A notebook is read by an analyst who will act on it: an asset manager, a
trader, a structurer, a risk manager, an econometrician. They are competent
and impatient. They will not read a paragraph to find out what a table is.

**Size.** The published examples in this field run ten to twenty code blocks
and a runtime in seconds. A tutorial that has grown past about twenty-five is
doing more than one job: the worked case belongs in the notebook, the recipes
in a gallery of standalone pages, the derivations in the methods note. Nothing
inside a notebook chains state across sections; each section can be read on
its own.

**The three places.** A *course notebook* is one worked case, read in order,
motivated by what the previous notebook could not do. A *recipe* is one page
per lever: six to eight code blocks, a runtime of seconds, no dependence on
any other page, and it ends on the refusal that marks the lever's boundary. The
*methods note* carries every derivation, every bound and every proof, so that a
notebook can state a rule and show it working without arguing for it. A
sentence that justifies belongs there; a sentence that demonstrates belongs in
a notebook.

**The shape of a recipe.** The base (the same file for every recipe, nothing
imposed), the mathematics of the lever, what you give drawn as the sheet, what
the simulator does with it, the check against the simulation, and what is
refused. Six sections, one screen of prose each.

---

# How to build a notebook

---

## 1. Open on the problem, not on the tool

The first cell states the problem the reader already has, in their words, and
why it is hard. Two or three numbered difficulties, each one a fact about
their situation, not about the package. The route comes second, the code
third.

A notebook that opens with what the library does has skipped the only part
that earns the reader's attention.

## 2. The reader must always know where they are

A table of contents at the top, linking to every section and subsection.
Numbered sections, an anchor on each, and a link back to the contents under
every heading. One line under each heading saying what the section is for.

A heading names the job, not the mechanism: "Set an asset's volatility", not
"Volatility targets". Someone scanning the contents is looking for the thing
they want to do.

## 3. Explain, then show, one exhibit at a time

**One beat per cell.** A beat is one idea with one sentence introducing it and
at most two exhibits under it. A table never appears without the sentence
directly above it saying what it is and what to look at.

If the prose is written in three parts, the code is split into three cells.
Writing "What you give / What the simulator does / Check" and then emitting
five tables from one cell destroys the split the prose just made.

## 4. Exemplify, never describe

Show the artifact itself. A description of a file is not a substitute for the
file.

- An input the analyst types in Excel is drawn as an Excel sheet: column
  letters, the file's own row numbers, a header row, empty cells left empty.
- A workbook is shown filled with real rows, never empty. An empty template
  shows the shape and teaches nothing.
- Anything derived from an input is an ordinary table, because that is what it
  is.

## 5. Every exhibit must teach

`head(4)` on a sorted file gives four identical rows: the layout once and then
three repetitions. Show the rows that differ, or the rows the notebook follows
throughout, and say how many rows the file really has.

## 6. When an artifact is modified, show the context and the modification

Never show the modification alone. Draw the artifact as it now stands: the
rows that were already there for context, the rows just written shaded, and
the real row numbers so a filtered view cannot lie about where a row sits.

When a cell that already existed is filled in, draw the same window twice,
before and after. An edit must be visible as an edit.

## 7. The mathematics comes before the code that implements it

Every code cell is preceded by the objects it manipulates, the formulas with
their symbols, the estimator or algorithm, and how to read what comes out.
A reader must be able to reproduce the cell from the text above it.

## 8. Every number in the text comes from the run

Readings are written from the executed outputs, never from memory or from a
previous run. After any re-execution, every numeric claim is re-checked. If a
number moved, the sentence moves with it.

Corollary: never promise an output the cell does not produce. A sentence
announcing a refusal the code no longer raises is worse than no sentence.

## 9. Show the refusals

Error messages are part of the teaching. A lever that can be pushed too far
shows both: the value that works and the value that is refused, with the
message naming what is infeasible and what to do about it. An analyst learns
the boundary faster from the message than from a paragraph.

## 10. Label what a number is

Simulated, implied by the model, or observed in the sample. Never present a
simulated quantity as if it were a target, and never put a history column
beside a target column without saying which is which. Means and volatilities
are annualized in every table that a reader will quote.

## 11. State the rule for what is missing

Wherever the reader may leave something out, the notebook says what happens:
what is estimated, from what, and what the fallback is when there is nothing
to estimate from. Given is a target; empty is an estimate. The rule is stated
once, precisely, and the reports then show which rule fired for every value.

## 12. No duplication between prose and exhibit

If the drawing shows the columns, the prose does not list them. If the table
below states the numbers, the paragraph does not repeat them; it reads them.
Two tables saying the same thing at different levels of abstraction is one
table too many, and the abstract one is the one to delete.

## 13. A defect found while writing is fixed in the package

Writing a notebook honestly is a test of the library. When an exhibit exposes
a defect, the defect is fixed in the code, with a test, and committed
separately. The text is never written around a bug.

Notebook 03 found three this way: an edit that moved the row it edited and
cleared its other cells, a correlation target solved against the wrong
volatility, and a draw of three months that hung an acceptance loop forever.

## 14. Delete what the method outgrew

When a rule changes, the whole notebook is re-read, not just the cell that
changed. Sentences written under the previous semantics survive in places
nobody thinks to look: an introduction, a cross-reference, a caption. Stale
cross-references to cell numbers are replaced by names that do not move.

## 15. Each notebook is motivated by what the previous one could not do

The opening says what the reader already has from the earlier notebooks, and
the closing says what this one still misses and which notebook takes it up.
The course is a chain; a notebook that stands alone is out of place in it.

## 16. No digression the section did not ask for

A derivation nobody requested, a bootstrap nobody asked for, an aside about a
method variant: all of it goes. Depth belongs in an appendix marked as
internal diagnostics the reader may skip.

## 17. End by handing over artifacts

The last section produces the things the reader takes away: the scenarios,
the paths, the saved objects, the workbook, with the code that writes them and
a check that reloading reproduces them exactly.

---

# How to write the sentences

Measured against the published notebooks in this field: skfolio's example
gallery, Riskfolio-Lib's numbered tutorials, the `arch` documentation. Their
prose is a manual's. Ours drifted into an essay's. The rules below close that
gap, and each one names the habit it replaces.

## S1. Announce a code block in fifteen words or fewer

skfolio: *"Let's generate 10,000 synthetic returns from the vine copula
model."* Ten words, one verb.

Not an inventory of what the cell contains. If the sentence lists the objects
the cell will print, the reader reads the list and then reads the same list
again in the output.

## S2. Comment a result in four sentences or fewer

skfolio, under a plot: *"We notice that the conditioning has been respected
and has impacted the other assets following the vine structure. This allows
the creation of Stress Tests that are both extreme and plausible."* Thirty
words.

State the fact, then its consequence. Stop. A reading of two hundred words
buries the one number that mattered. Where a result needs more, the extra
belongs in the methods note, not under the table.

## S3. Do not rank your own paragraphs

Banned: "the useful part", "this needs its own explanation", "the reader
should slow down here", "is a warning, not a result", "worth knowing".
Telling the reader how to feel about a number is not information. Write the
number and what follows from it, and let it carry its own weight.

## S4. Use a plain voice

"We fit", "the model gives", the imperative. Not "one caveat governs the next
step", not "the reader will notice". Present tense, subject then verb then
object.

## S5. Bold marks a term, not the start of a paragraph

A bold label on every cell stops being structure and becomes a tic. Keep it
where it marks a repeated slot a reader navigates by, and nowhere else.

The purpose line under a heading is a sentence, not a label: *"Eight funds, six
factors, nothing imposed."*, not *"**What this section is for.** Eight
funds..."*. A reading under a table opens with the fact, not with *"**What the
table says.**"*.

## S6. Justify nowhere but the user guide and the methods note

A tutorial says what the rule is and shows it working. It does not argue for
it. Every "this is why the order is not the usual one" is a sentence that
belongs somewhere else, and its absence makes the notebook shorter and the
argument easier to find.

## S7. Keep the judgment, cut the rhetoric

Published examples rarely say when a method fails. A course that teaches
judgment must: the infinite kurtosis, the shape that will not land closer
than two tenths, the copula that cannot be identified. These are the most
valuable sentences in the course. Three of them, not fifteen.

---

## The shape of a notebook

1. **The problem.** What the reader needs and why it is hard.
2. **The route.** The object and the equation, then why this answers the
   difficulties, then the steps.
3. **What you have.** The inputs, drawn as the artifacts they are, and the
   rule for every gap.
4. **Targets.** What the generator must reproduce.
5. **Fit.** The estimation, with the mathematics above it.
6. **The levers.** One per kind of knowledge, each: *what you give* (the
   artifact), *what the simulator does* (the rule and its report), *check*
   (did it do what it said), *reading* (what the numbers mean).
7. **Check.** Every input against the simulation, one object type per cell,
   with the Monte Carlo error to read the differences against.
8. **Use.** The artifacts out.
9. **Appendix.** Method notes and diagnostics, marked skippable.
10. **What this misses, and what comes next.**

## Before committing a notebook

The mechanical half of this list is a script, kept beside the design notes as
`docs/superpowers/specs/2026-09-27-notebook-checklist.py.txt`: it counts the
cells, the code blocks and the words, resolves every link against an anchor,
and fails on a code cell with no output or no sentence above it, an
announcement over twenty words, a sentence over forty, a cell emitting more
than two exhibits, and the banned phrases of S3. Run it before every commit.

- [ ] Executed end to end, no errors, outputs in the file.
- [ ] Contents at the top, every section numbered and anchored, a link back
      under every heading, a purpose line under every section.
- [ ] No code cell without a sentence above it; none emitting more than two
      exhibits.
- [ ] No sentence over about forty words; nothing in prose that an exhibit
      below it already shows.
- [ ] Every input drawn as its artifact; every modification shown in context.
- [ ] Every number in the prose checked against this run's outputs.
- [ ] Every promise in the prose produced by a cell.
- [ ] The whole notebook re-read after the last semantic change.
- [ ] Runtime stated in the opening if it exceeds a minute.
- [ ] Every code block announced in fifteen words or fewer; every result
      commented in four sentences or fewer.
- [ ] No sentence ranking another paragraph's importance; no justification
      that belongs in the user guide or the methods note.

---

## Sources for the style

- skfolio's example gallery, in particular *Vine Copula & Stress Test*:
  fifteen short sections, ten code blocks, ten seconds of runtime, standalone.
  <https://skfolio.org/auto_examples/index.html>
- Riskfolio-Lib, fifty-eight numbered standalone tutorials, one topic each.
  <https://github.com/dcajasn/Riskfolio-Lib/tree/master/examples>
- The `arch` documentation, *ARCH Modeling*: thirteen sections, twenty code
  blocks, one to five sentences of prose between them.
  <https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html>
