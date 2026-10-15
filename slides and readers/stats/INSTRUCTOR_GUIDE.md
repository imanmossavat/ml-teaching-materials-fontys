# Instructor guide — stats/04_dependence_structures_counts_to_graphs

A ~2 hour lab in which students discover a dependence structure from data
(counts, pairwise associations, conditional tables) and learn that a graph is a
compact recipe for a probability distribution.

**Core idea:** observe, predict, be surprised, then explain. The main lab uses
only *depends on / independent of* language. Causality, exact factorisation,
and interventions are reserved for the optional `OPTIONAL_extensions.md` and the
`reveal_simulator.py` notebook extension.

## Before class

| Step | What to do |
|---|---|
| Print | `student_worksheet.md` (one per group) |
| Install | Python 3.10+, `numpy pandas scipy matplotlib jupyter` |
| Give each group | `04_dependence_structures_counts_to_graphs.ipynb`, `lab_helpers.py`, and **only their own** data file from `data/` plus `collider_upgrade.csv` |
| Keep back | `slides and readers/stats/` (all instructor materials) and `OPTIONAL_extensions.md` (Module D / fast groups) |
| Assign groups | `group_assignments.csv` is the private map. **Groups D and G have a hidden direct ticket_class -> party edge.** Do not tell students. |

## Lab structure

| Module | Time | What students do | Key idea |
|---|---|---|---|
| A. From counts to a graph | 20 min | Marginals, conditional proportions, predict shares | A graph encodes which proportions factorise |
| B. Pairwise and conditional tests | 50 min | Crosstabs, bubble plots, `assoc_test` with conditioning | Conditioning can remove (or create) dependence |
| C. Noise, direction, colliders | 35 min | Sample-size sweep, permutation test, upgrade dataset | Data cannot orient edges; colliders behave differently |
| D. Optional: from graph to simulator | 30 min | `reveal_simulator.py`, exact joint probability, intervention | Optional extension; causal language allowed here |

Group reports (2 min each) fit between Modules B and C. Have each group draw its
graph on the board before the reveal.

## Facilitation notes

**Module B** is the core. Typical mistakes:

- Concluding independence from a single non-significant p-value. Ask about sample size.
- Conditioning on the wrong variable. Push students to test *all* reasonable conditioning sets and to look at the diagonal conditional plot.
- Hidden-edge groups (D/G) will find that ticket_class and party remain associated after every conditioning set and will draw a 4-node cycle. This is *correct for their data*.

**Module C collider.** In `collider_upgrade.csv`, `loyalty_member` and `peak_season` are marginally independent but dependent among upgraded (or non-upgraded) travellers. This sets up the idea that conditioning on a common effect can create dependence.

## Expected answers (standard groups)

For standard groups the skeleton is `C - F - P - T`. The separating sets are:

| Pair | Conditioning sets that make them independent |
|---|---|
| F - C | none of the other variables separates them |
| F - P | none of the other variables separates them |
| F - T | C and P together (P alone is enough in the standard model) |
| C - P | F |
| C - T | P |
| P - T | none of the other variables separates them |

For hidden-edge groups (D/G) add a direct `C - T` edge; C and T remain dependent
after every conditioning set.

## Discussion prompts

- Two variables are associated. List all the possible explanations.
- Why does conditioning on `purpose` remove the ticket_class/party association,
  while conditioning on `upgraded` creates one?
- Is "not significant" evidence of independence? How much data do we need?
- What changes if `financial_status` is unobserved? (Extension: drop the F column and re-run Module B.)

## Files in this folder

- `model.py` — generative model.
- `generate_data.py` — regenerate shipped CSVs.
- `qa.py` — quality-assurance functions.
- `validate_data.py` — Monte-Carlo validation (writes `validation_results.pkl`).
- `data_validation_report.md` — summary of validation results.
- `group_assignments.csv` — private map of group -> hidden edge.
- `answer_keys/` — expected test verdicts for each group.
- `extension/` — 100 000-row CSVs for fast groups.
- `reveal_simulator.py` — optional simulator used in Module D.
