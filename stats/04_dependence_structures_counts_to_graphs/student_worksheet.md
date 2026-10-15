# Lab 4 worksheet: From Counts to Dependence Structures

**Group:** __________  **Names:** _________________________________

---

## 1. Predict before you compute

Before you look at any numbers, draw a quick prediction of which variables you think will be directly linked. (You will revise this later.)

```
financial_status     ticket_class     purpose     party
```

Draw arrows or lines between the variables you expect to be associated.

---

## 2. Marginal counts

Pick one variable and write its level counts and proportions.

| level | count | proportion |
|---|---|---|
| | | |
| | | |
| | | |

Which level is most common? ___________________________________________

---

## 3. Pairwise associations

Use `assoc_test(df, a, b)` to fill the table.

| pair | Cramer's V | p-value | associated? (yes / no / borderline) |
|---|---|---|---|
| financial_status — ticket_class | | | |
| financial_status — purpose | | | |
| financial_status — party | | | |
| ticket_class — purpose | | | |
| ticket_class — party | | | |
| purpose — party | | | |

Surprises? ____________________________________________________________

---

## 4. Conditioning table

For each pair, test independence while conditioning on a third variable. Write **I** (independent), **D** (dependent), or **~** (noisy / unsure). The pair's own column stays blank.

| pair | given F | given C | given P | given T |
|---|---|---|---|---|
| F — C | — | | | |
| F — P | | — | | |
| F — T | | | | |
| C — P | | — | | |
| C — T | | | | |
| P — T | | | | — |

Which pairs stay associated no matter what you condition on? __________________

Which pairs become independent once you condition on the right variable? ________

---

## 5. Your graph

Draw a graph with one node per variable. Connect two nodes if they stay associated after conditioning.

```
financial_status     ticket_class     purpose     party
```

What can this graph **not** tell you? (direction? hidden variables? causation?)

______________________________________________________________________

---

## 6. Sample size reflection

Look at your sample-size sweep plot.

With n = 200, would you trust a non-significant p-value as evidence of independence? Why or why not?

______________________________________________________________________

---

## 7. Collider reflection

In `collider_upgrade.csv`, `loyalty_member` and `peak_season` are marginally independent but associated among upgraded travellers.

Explain in your own words why conditioning on a common outcome can create an association:

______________________________________________________________________

---

## 8. Exit ticket

1. What is the difference between a marginal association and a conditional association?

______________________________________________________________________

2. When is a graph of variables a useful summary of a dataset?

______________________________________________________________________

3. Why do we need to be careful with p-values when the sample is small?

______________________________________________________________________
