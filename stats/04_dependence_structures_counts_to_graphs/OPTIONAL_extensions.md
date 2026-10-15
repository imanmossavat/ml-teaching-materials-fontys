# Optional extensions (read after the core lab)

**The core lab needs none of this.** In the core lab an arrow only means "this variable is drawn from a table that depends on that one". We use the story *"wealth influences which ticket class you can afford"* to make the arrows easy to remember, and we never change anything in the world: we only **look** at data.

These extensions are for groups that finish early or want more. Each stands alone.

| | Question it answers | Time | Where |
|---|---|---|---|
| A. False alarms and sample size | Can unrelated things look related by chance? How much data is enough? | 20 min | notebook, "Module A" |
| B. Which arrows can the data tell apart? | If the data are the same, are the arrow directions the same? | 15 min | notebook, "Module B" |
| C. Colliders | Can *conditioning* create a relationship? | 20 min | notebook, "Module C" |
| D. Seeing versus doing | Is "families are more common in third class" a reason to sell more third-class tickets? | 25 min | **this note, below** |

---

## D. Seeing versus doing

### Why it matters (plain words)

Two different questions sound alike:

* **Seeing** (conditioning): *"Among people who happen to have a First-class ticket, how do they travel?"* We look at a group that already exists. That group may differ in other ways (they are richer, they travel for different reasons).
* **Doing** (intervening): *"If we gave **everyone** a First-class ticket, how would they travel?"* We change one thing by hand and leave the rest of the world alone. We write this as **do(class = First)**.

A graph helps you decide whether data you already have can answer a "doing" question. A business, a school or a hospital usually cares about the doing question ("what happens **if we** change this?") but usually only has data for the seeing question. Randomised experiments are the way to do things *physically*; a simulator does it by overwriting one line of code.

One rule of thumb: when you set a variable by hand, only variables **downstream** of it (its children and their children) can change. Variables upstream or elsewhere stay as they were.

### Hands-on

This code is self-contained (numpy and pandas only). It rebuilds our travel world as a recipe of tables, then lets you look and set.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200_000

# ---- the world as a recipe of tables (rows = parent value, columns = child value) ----
P_wealth  = [0.10, 0.40, 0.50]                       # Wealthy, Middle, Limited
P_class   = np.array([[.70, .25, .05],               # given wealth: First, Second, Third
                      [.20, .50, .30],
                      [.05, .15, .80]])
P_purpose = np.array([[.25, .70, .05],               # given wealth: Business, Leisure, Migration
                      [.30, .40, .30],
                      [.15, .15, .70]])
P_party   = np.array([[.70, .20, .10],               # given purpose: Solo, Couple, Family
                      [.25, .40, .35],
                      [.35, .15, .50]])

def draw(table, parent):                             # one random draw per person
    u = rng.random(len(parent))
    return (u[:, None] > np.cumsum(table[parent], axis=1)).sum(axis=1)

def make_people(force_class=None, direct_effect=False):
    wealth  = rng.choice(3, size=n, p=P_wealth)
    cls     = draw(P_class, wealth)
    purpose = draw(P_purpose, wealth)
    if force_class is not None:                      # <-- the ONLY line that differs: "set it by hand"
        cls = np.full(n, force_class)
    if not direct_effect:
        party = draw(P_party, purpose)               # note: class is never used here
    else:                                            # EXERCISE 3: class now nudges party directly
        tilt = np.array([[1, 1.8, .6], [1, 1, 1], [1, .7, 1.3]])      # rows = class
        tab = P_party[purpose] * tilt[cls]
        tab = tab / tab.sum(axis=1, keepdims=True)
        party = (rng.random(n)[:, None] > np.cumsum(tab, axis=1)).sum(axis=1)
    return pd.DataFrame({"wealth": wealth, "cls": cls, "purpose": purpose, "party": party})

shares = lambda s: s.value_counts(normalize=True).sort_index().round(3).values   # Solo, Couple, Family

seen = make_people()
print("everyone                  P(party):", shares(seen.party))
print("seen: First class         P(party | class=First):", shares(seen[seen.cls == 0].party))
print("set:  First class for all P(party | do(class=First)):", shares(make_people(force_class=0).party))
```

**Exercise 1: predict, then run.** Before running, write down which two of the three lines you expect to be (nearly) equal, and why.

**Exercise 2: explain.** The second line differs from the first, but the third does not. In one or two sentences: where do the First-class people differ from everyone else? (Hint: look at the `wealth` and `purpose` columns of `seen[seen.cls == 0]`.)

**Exercise 3: change the world.** Run the three lines again with `direct_effect=True` in all three calls to `make_people`. Now class really does nudge how people travel. Which lines change, and which still match? What does that tell you about when "seeing" and "doing" agree?

**Exercise 4 (challenge).** Add a `force_wealth` option, set everyone to Wealthy, and predict the direction of the change in `party` before running. Which variables change on the way, and which arrow carries the effect to `party`?

### Check your answers

* Exercise 1: about `[0.396 0.237 0.367]` for everyone, `[0.39 0.282 0.328]` for First-class people seen, `[0.397 0.237 0.366]` when we set First class for all. Lines 1 and 3 match; line 2 does not.
* Exercise 2: First-class people are mostly wealthy and travel for leisure, and leisure travellers are more often in couples. Class was only a *marker* for those.
* Exercise 3: about `[0.384 0.231 0.386]` for everyone, `[0.359 0.454 0.187]` seen, `[0.384 0.397 0.219]` set. Now setting First class *does* move the numbers (more couples), but it still differs from the seen group, because the seen group also differs in wealth and purpose.
* Exercise 4: wealth changes class and purpose, and purpose carries the effect to party (more leisure, so more couples).

### Honest limits

* This is a simulated world, so we can overwrite a variable. In real life you need an experiment or strong assumptions.
* "Seeing equals doing" held in the first run only because the graph has no arrow from class to party. If you do not know the graph, you cannot know this.

### If a student asks

* **"Is the arrow the same as cause?"** In the core lab: no, an arrow means "this table depends on that one". Here we go one step further and read arrows as "setting this changes that". That reading is an *assumption* you add, not something the data prove.
* **"Why not just use the data?"** Because data show who ends up in First class, not what First class does to people.
* **"Does the data ever tell us the arrow direction?"** Sometimes (see Module B and C), often not.
