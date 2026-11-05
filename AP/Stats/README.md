# Statistics and probability labs

A short sequence of hands-on notebooks for building intuition about distributions, estimation, and dependence structures.

## Labs

| # | Notebook | What it covers | Time |
|---|----------|----------------|------|
| 1 | `01_Categorical_Distributions.ipynb` | Discrete distributions, sampling, joint/marginal/conditional probabilities, independence | ~30 min |
| 2 | `02_bernoulli_to_gaussian.ipynb` | Bernoulli, Binomial, sample proportions, emergence of the Gaussian | ~25–30 min |
| 3 | `03_estimating_bernoulli.ipynb` | Estimating a probability, standard error, effect of sample size | ~30 min |
| 4 | `04_dependence_structures_counts_to_graphs/` | From counts to conditional tables to graphs; dependence, independence, colliders | ~2 h |

## Prerequisites

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`, `scipy`, `jupyter`
- For lab 4 you also need `lab_helpers.py` in the same folder as the notebook.

## How to run

Open a notebook in JupyterLab or VS Code and run the cells in order. Cells marked `# TODO` are for you to fill in; markdown cells with `**Your answer:**` are for short written reflections.

## Folder map

```
stats/
├── README.md
├── 01_Categorical_Distributions.ipynb
├── 02_bernoulli_to_gaussian.ipynb
├── 03_estimating_bernoulli.ipynb
└── 04_dependence_structures_counts_to_graphs/
    ├── 04_dependence_structures_counts_to_graphs.ipynb
    ├── lab_helpers.py
    ├── student_worksheet.md
    ├── OPTIONAL_extensions.md
    ├── README.md
    └── data/
        ├── group_A.csv … group_H.csv
        └── collider_upgrade.csv
```

`Joint Distributions, Independence, and Marginalization.pdf` is an additional reference PDF in this folder.

## Pedagogy

The labs follow a **"predict before you compute"** style. The aim is to make the distinction between a true distribution and a sample estimate concrete, and to show that graphs of variables can be read as recipes of probability tables. Students should already be comfortable with basic Python, simple plots, and the idea of a proportion.

## Reader

The accompanying theory is in `slides and readers/Stats_imos_base.pdf` (also available as `.pptx`).
