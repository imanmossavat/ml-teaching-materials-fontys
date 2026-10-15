# Optional extensions for Lab 4

These activities go beyond the main lab. They are intended for groups that finish early or for a follow-up session where causal language is explicitly allowed.

---

## Extension 1: From data to an exact joint distribution

The graph you built in the main lab is a picture of which variables are directly associated. If we are willing to interpret the arrows as a generative recipe, we can write the joint probability as a product of conditional probabilities:

$$
\begin{aligned}
P(F, C, P, T) = P(F) \cdot P(C|F) \cdot P(P|F) \cdot P(T|P)
\end{aligned}
$$

For the standard groups this factorisation predicts the counts very well. For groups D and G it does not, because their data contain an extra direct association between `ticket_class` and `party`.

Try it:

1. Estimate $P(F)$, $P(C|F)$, $P(P|F)$, and $P(T|P)$ from your data.
2. Multiply the probabilities for a few $(F, C, P, T)$ combinations.
3. Compare the predicted counts with the observed counts.

Files: `slides and readers/stats/model.py`, `slides and readers/stats/reveal_simulator.py`

---

## Extension 2: Build the simulator

`reveal_simulator.py` contains the actual probability tables used to generate the data.

1. Read the file and identify the parents of each variable.
2. Use `simulate(n, seed, world="standard")` and `simulate(n, seed, world="hidden_edge")` to generate synthetic data.
3. Run the same association tests on the synthetic data. Do you recover the same graph?

---

## Extension 3: Intervention vs observation

In the main lab you estimated $P(T \mid C, P)$ from the observed data. An **intervention** sets `ticket_class` to a value rather than observing it. The simulator's `do_C=` argument lets you do this.

Compare:

- $P(T \mid C=\text{First}, P=\text{Leisure})$ from the observed data.
- $P(T \mid \text{do}(C=\text{First}), P=\text{Leisure})$ from the simulator.

If they differ, why? (Hint: what other path is opened or closed by observing vs setting ticket class?)

---

## Extension 4: Large-sample hidden-edge check

Some groups received a `large_sample.csv` file with 100 000 rows.

1. For standard groups, check that the independence conclusions still hold with much more data.
2. For hidden-edge groups (D/G), look at the pair `ticket_class — purpose` conditioned on **both** `financial_status` and `party`. In the smaller sample this cell was often noisy; with 100 000 rows the collider signal at `party` should become visible.

---

## For instructors

- `slides and readers/stats/group_assignments.csv` maps each group letter to whether it receives the hidden-edge world.
- `slides and readers/stats/extension/` contains `large_sample_A.csv` and `large_sample_D.csv`.
- `slides and readers/stats/reveal_simulator.py` is the self-contained simulator used in Extensions 2 and 3.
