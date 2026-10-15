# Data validation report — stats/04_dependence_structures_counts_to_graphs

This report documents the Monte-Carlo checks run on the generative model and the shipped datasets.
It is a companion to the instructor materials in this folder.

## Summary

- **Model**: `model.py` defines a discrete DAG over four variables:
  `financial_status (F) -> ticket_class (C)`, `F -> purpose (P)`, `P -> party (T)`.
  For groups **D and G** a direct `C -> T` edge is added.
- **Sample size per group**: n = 10 000.
- **QA gate**: `qa_dataset` in `qa.py` checks structural sanity, independence/dependence
  verdicts, stratum power, and fidelity to the specified CPTs.

## Dataset generation

```bash
python generate_data.py ../../stats/04_dependence_structures_counts_to_graphs/data
```

This regenerates `group_A.csv` … `group_H.csv` and `collider_upgrade.csv`.

## Monte-Carlo results (R = 150)

| world | QA pass rate | main concern |
|-------|--------------|--------------|
| standard (A, B, C, E, F, H) | ~100 % | C-T and F-T marginal associations are small but detectable at n=10k |
| hidden edge (D, G) | ~100 % | C-T association is strong; C-P | F,T collider signal is underpowered |

## Sample-size sensitivity

The asymptotic C-T test becomes clearly non-significant for standard groups by n ≈ 500
when conditioned on the appropriate variable, and remains clearly dependent for hidden-edge groups.

## Collider check

In `collider_upgrade.csv`, `loyalty_member` and `peak_season` are marginally independent,
but become dependent within each level of `upgraded` (conditional association induced by
conditioning on a common effect).

## Extension files

`extension/large_sample_A.csv` and `extension/large_sample_D.csv` contain 100 000 rows each
for the optional large-sample extension. D/G instructors should hand out `large_sample_D.csv`;
all other groups receive `large_sample_A.csv`.

## Files in this folder

- `model.py` — generative model.
- `generate_data.py` — regenerate shipped CSVs.
- `qa.py` — quality-assurance functions.
- `validate_data.py` — full Monte-Carlo validation (writes `validation_results.pkl`).
- `CSVs/` — expected test verdicts for each group.
- `extension/` — large-sample CSVs.
- `group_assignments.csv` — private map of group -> hidden edge.
- `reveal_simulator.py` — optional simulator for Module D (intervention / do-operator).
- `INSTRUCTOR_GUIDE.md` — teaching notes.
