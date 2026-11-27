# Decision Theory (DETH)

Material for the Decision Theory course (Fontys ICT): Markov decision processes (MDPs), the Bellman equations,
dynamic programming (policy and value iteration) and reinforcement learning.

Start with [**Curriculum and assessment**](decision-theory-curriculum-and-assessment.pdf). It has the goals, the schedule, the dates and
the assessment. For the **midterm** you may use a calculator; no notes are allowed.

## What is in this folder

| Folder / file | What it is |
|---|---|
| [decision-theory-curriculum-and-assessment.pdf](decision-theory-curriculum-and-assessment.pdf) | Curriculum and assessment: goals, weekly plan, dates, assessment. |
| [slides/](slides/) | [Lecture slides](slides/deth-slides-2023-24.pdf) (2023–24 version). |
| [notes/](notes/) | Short lecture notes (PDF). |
| [notebooks/](notebooks/) | Jupyter notebooks. |
| [quiz/](quiz/) | Practice quiz and its final answers. |
| [mdp-design/](mdp-design/) | The "design your own MDP" workshop: the reader and an optional coding standard for your environment. |

## Lecture notes and notebooks

| Topic | Note | Notebook |
|---|---|---|
| Solving systems of linear equations: substitution, elimination, matrices, determinant, inverse, condition number | [PDF](notes/solving-linear-systems.pdf) | [notebook](notebooks/solving-linear-systems.ipynb) |
| Geometry of linear systems | [PDF](notes/geometry-of-linear-systems.pdf) | [notebook](notebooks/geometry-of-linear-systems.ipynb) |
| Monte Carlo reinforcement learning on FrozenLake | | [notebook](notebooks/frozenlake-alpha-mc.ipynb) |

## Practice quiz

[quiz/practice-quiz.pdf](quiz/practice-quiz.pdf): 45 minutes, 5 questions. One small MDP (a coffee cart) is used in all
questions: the model and the Markov property, the return, Bellman equations for a fixed policy, one step of policy
iteration, and two sweeps of value iteration. Allowed: a simple calculator. Marks are given for reasoning, not for the
final number alone.

[quiz/practice-quiz-answers.pdf](quiz/practice-quiz-answers.pdf): the final answers, to check your own work. Try the quiz first.

## Software

For the notebooks: Python 3.10+ with `numpy`, `matplotlib` and `jupyter`; the FrozenLake notebook also needs
`gymnasium` and `scipy`. Open a notebook in JupyterLab or VS Code and run the cells in order.
