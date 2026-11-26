# Decision Theory (DETH)

Material for the Decision Theory course (Fontys ICT): Markov decision processes (MDPs), the Bellman equations,
dynamic programming (policy and value iteration) and reinforcement learning.

Start with the **course guide**. It has the schedule, the assessment and what to do each week.

## What is in this folder

| Folder / file | What it is |
|---|---|
| [Decision Theory Course Guide.pdf](Decision%20Theory%20Course%20Guide.pdf) | The course guide: goals, weekly plan, dates, assessment. |
| [slides/](slides/) | Lecture slides. |
| [Notes/](Notes/) | Short lecture notes that go with the slides (PDF, with the LaTeX source). |
| [Quiz/](Quiz/) | Practice quiz, in the style of the in-class quizzes and the exam. |
| [MDP design/](MDP%20design/) | The "design your own MDP" workshop: the reader and an optional coding standard for your environment. |
| [notebooks/](notebooks/) | Jupyter notebooks used in class. |

## Lecture notes

| Note | What it covers |
|---|---|
| [Solving systems of linear equations](Notes/solving-linear-systems.pdf) | Substitution, elimination, and matrices ($A\mathbf x = \mathbf b$, the determinant step by step, the $2 \times 2$ inverse, `numpy`). Exercises with answers, including almost parallel lines and the condition number. |

## Practice quiz

[Quiz/practice-quiz.pdf](Quiz/practice-quiz.pdf): 45 minutes, 5 questions. One small MDP (a coffee cart) is used in all
questions: the model and the Markov property, the return, Bellman equations for a fixed policy, one step of policy
iteration, and two sweeps of value iteration. Allowed: a simple calculator. Marks are given for reasoning, not for the
final number alone.

[Quiz/practice-quiz-answers.pdf](Quiz/practice-quiz-answers.pdf): the final answers, to check your own work. Try the quiz first.

## Software

For the notebooks and the Python parts of the notes: Python 3.10+ with `numpy`, `matplotlib` and `jupyter`.
Open a notebook in JupyterLab or VS Code and run the cells in order.
