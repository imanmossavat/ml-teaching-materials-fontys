"""
Helper functions for the causal-structure lab.

You are NOT expected to read this file closely: it contains plotting and
testing utilities so you can spend your time interpreting results.
(Read the docstrings of `assoc_test` if you want to know how p-values are made.)
"""
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as _chi2_dist

# ---- column names (so we never type df.T by accident: that is the transpose!) ----
F = "financial_status"
C = "ticket_class"
P = "purpose"
T = "party"
SHORT = {F: "F", C: "C", P: "P", T: "T"}

ORDER = {
    F: ["Wealthy", "Middle income", "Limited means"],
    C: ["First", "Second", "Third"],
    P: ["Business", "Leisure", "Migration"],
    T: ["Solo", "Couple", "Family"],
}

_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]


def load_file(path):
    """Load any travellers CSV, putting the levels of each variable in a sensible order."""
    df = pd.read_csv(path)
    for col, lv in ORDER.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=lv, ordered=True)
    return df


def load_group(group, folder="data"):
    """Load <folder>/group_<X>.csv."""
    return load_file(f"{folder}/group_{group}.csv")


def subsample(df, n, seed=0):
    """Random subsample of n rows (without replacement)."""
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _levels(s):
    return list(s.cat.categories) if hasattr(s, "cat") else sorted(s.unique())


# ------------------------------------------------------------------ tables
def counts(df, a, b, given=None):
    """Contingency table of counts for (a, b); a dict of tables if `given` is set."""
    def one(d):
        return pd.crosstab(d[a], d[b]).reindex(index=_levels(df[a]), columns=_levels(df[b]), fill_value=0)
    if given is None:
        return one(df)
    given = [given] if isinstance(given, str) else list(given)
    return {k: one(g) for k, g in df.groupby(given, observed=True)}


def cond_props(df, a, b, given=None, warn_below=100):
    """
    Row-normalised table P(b | a)  (each ROW sums to 1), with a column `n`
    giving the number of people in each row.  If `given` is set, returns a dict
    with one table per level of the conditioning variable(s).
    Strata smaller than `warn_below` people trigger a warning: be careful reading them.
    """
    def one(d):
        ct = pd.crosstab(d[a], d[b]).reindex(index=_levels(df[a]), columns=_levels(df[b]), fill_value=0)
        n = ct.sum(axis=1)
        out = ct.div(n.replace(0, np.nan), axis=0).round(3)
        out["n"] = n
        return out
    if given is None:
        return one(df)
    given = [given] if isinstance(given, str) else list(given)
    out = {}
    for k, g in df.groupby(given, observed=True):
        if len(g) < warn_below:
            print(f"warning: stratum {k} has only {len(g)} people; proportions are noisy")
        out[k] = one(g)
    return out


# --------------------------------------------------------------- the test
def _chi2_stack(cnt):
    """Pearson chi-square for a stack of tables, shape (m, r, c)."""
    rows = cnt.sum(axis=2)
    cols = cnt.sum(axis=1)
    n = rows.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = rows[:, :, None] * cols[:, None, :] / n[:, None, None]
        term = np.where(E > 0, (cnt - E) ** 2 / E, 0.0)
    return term.sum(axis=(1, 2))


def assoc_test(df, a, b, given=None, n_perm=999, seed=0, method="perm"):
    """
    Test whether a and b are independent (optionally: within each level of `given`).

    statistic : sum over strata of the Pearson chi-square statistic.
    p-value   : method="perm"  -> permutation test: shuffle b relative to a *within each
                                  stratum* n_perm times and see how often the shuffled
                                  statistic is >= the observed one. Valid even when some
                                  strata are small.
                method="asymptotic" -> classical chi-square approximation (fast, but unreliable
                                  when strata have few people).
    V         : pooled Cramer's V, an effect size on a 0-1 scale (0.1 small, 0.3 medium).
    Small p  => evidence AGAINST independence. Large p => no evidence of dependence
    (which is NOT the same as proof of independence: think about sample size).
    """
    if isinstance(given, str):
        given = [given]
    ca = pd.Categorical(df[a]).codes.astype(np.int64)
    cb = pd.Categorical(df[b]).codes.astype(np.int64)
    ra, rb = int(ca.max()) + 1, int(cb.max()) + 1
    n = len(df)
    if given:
        key = np.zeros(n, dtype=np.int64)
        for g in given:
            gc = pd.Categorical(df[g]).codes.astype(np.int64)
            key = key * (int(gc.max()) + 1) + gc
        _, inv = np.unique(key, return_inverse=True)
        strata = [np.where(inv == s)[0] for s in range(inv.max() + 1)]
    else:
        strata = [np.arange(n)]

    obs_tabs = np.stack([np.bincount(ca[i] * rb + cb[i], minlength=ra * rb).reshape(ra, rb) for i in strata])
    obs = float(_chi2_stack(obs_tabs).sum())
    V = float(np.sqrt(obs / (n * max(1, min(ra, rb) - 1))))
    out = dict(stat=obs, V=V, n=n, n_strata=len(strata), min_stratum_n=min(len(i) for i in strata))

    if method == "asymptotic":
        dof = 0
        for t in obs_tabs:
            dof += max(int((t.sum(1) > 0).sum()) - 1, 0) * max(int((t.sum(0) > 0).sum()) - 1, 0)
        out["p"] = float(_chi2_dist.sf(obs, dof)) if dof > 0 else 1.0
        out["df"] = dof
        return out

    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(n_perm)
    for idx in strata:
        a_s, b_s = ca[idx], cb[idx]
        done = 0
        while done < n_perm:
            B = min(100, n_perm - done)
            bp = rng.permuted(np.tile(b_s, (B, 1)), axis=1)
            codes = a_s[None, :] * rb + bp + (ra * rb) * np.arange(B)[:, None]
            tabs = np.bincount(codes.ravel(), minlength=B * ra * rb).reshape(B, ra, rb)
            perm_stats[done:done + B] += _chi2_stack(tabs)
            done += B
    out["p"] = float((1 + np.sum(perm_stats >= obs - 1e-9)) / (n_perm + 1))
    return out


def ci_long(df, cols=(F, C, P, T), n_perm=499, seed=0):
    """
    Run `assoc_test` for every pair of variables, conditioning on nothing, on each other
    variable alone, and on both other variables together (24 tests for 4 variables).
    Returns a long table with numeric p-values: columns a, b, given (tuple of letters), p, V.
    """
    rows = []
    for pair in combinations(cols, 2):
        others = [c for c in cols if c not in pair]
        for s in [[]] + [[o] for o in others] + [others]:
            r = assoc_test(df, pair[0], pair[1], given=s or None, n_perm=n_perm, seed=seed)
            rows.append(dict(a=SHORT.get(pair[0], pair[0]), b=SHORT.get(pair[1], pair[1]),
                             given=tuple(SHORT.get(x, x) for x in s), p=r["p"], V=r["V"]))
    return pd.DataFrame(rows)


def ci_table(df, cols=(F, C, P, T), n_perm=499, alpha=0.01, seed=0):
    """
    Same tests as `ci_long`, displayed as a table of p-values.
    A "~" marks p > alpha (no evidence of dependence).
    Columns are conditioning sets; empty cells = set not applicable to that pair.
    """
    long = ci_long(df, cols, n_perm, seed)
    floor = 1 / (n_perm + 1)
    lab = lambda g: "none" if not g else "|" + ",".join(g)
    long["col"] = long.given.apply(lab)
    long["row"] = long.a + " - " + long.b
    long["txt"] = [(f"<{floor:.3f}" if p <= floor + 1e-12 else f"{p:.3f}") + (" ~" if p > alpha else "")
                   for p in long.p]
    wide = long.pivot(index="row", columns="col", values="txt").fillna("")
    letters = [SHORT.get(c, c) for c in cols]
    order = ["none"] + [lab([c]) for c in letters] + [lab(list(g)) for g in combinations(letters, 2)]
    order = [o for o in order if o in wide.columns]
    return wide.loc[[f"{SHORT.get(a, a)} - {SHORT.get(b, b)}" for a, b in combinations(cols, 2)], order]


def separations(edges, nodes=("F", "C", "P", "T")):
    """
    Given a directed graph as a list of (parent, child) pairs, return the set of all
    conditional independencies it implies, as triples (a, b, Z): a and b are d-separated by Z.
    Z is a tuple (in the order F, C, P, T); we only consider |Z| <= 2, as in `ci_long`.
    """
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    found = set()
    for a, b in combinations(nodes, 2):
        others = [v for v in nodes if v not in (a, b)]
        for k in range(len(others) + 1):
            for Z in combinations(others, k):
                if nx.is_d_separator(G, {a}, {b}, set(Z)):
                    found.add((a, b, tuple(Z)))
    return found


# ------------------------------------------------------------------- plots
def bubble(ax, df, a, b, title=None, given_label=None):
    """Bubble scatter: x = levels of a, y = levels of b, bubble size and label = P(b | a)."""
    la, lb = _levels(df[a]), _levels(df[b])
    ct = pd.crosstab(df[a], df[b]).reindex(index=la, columns=lb, fill_value=0)
    prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    for xi, x in enumerate(la):
        for yi, y in enumerate(lb):
            v = prop.loc[x, y]
            ax.scatter(xi, yi, s=max(v, 0.003) * 3500, color=_PALETTE[yi % 5], alpha=.75, edgecolor="k")
            ax.text(xi, yi, f"{v:.0%}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(la)))
    ax.set_xticklabels(la)
    ax.set_yticks(range(len(lb)))
    ax.set_yticklabels(lb)
    ax.set_xlim(-.6, len(la) - .4)
    ax.set_ylim(-.6, len(lb) - .4)
    ax.set_xlabel(a)
    ax.set_title(title or f"P({b} | {a})\n(n={len(df):,})", fontsize=10)
    ax.grid(alpha=.2)


def bubble_panels(df, a, b, given=None, figsize=None):
    """Bubble scatter overall and, if `given` is set, within each level of `given`."""
    groups = [("All", df)] + ([(f"{given} = {k}", g) for k, g in df.groupby(given, observed=True)] if given else [])
    fig, axes = plt.subplots(1, len(groups), figsize=figsize or (4.5 * len(groups), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, g) in zip(axes, groups):
        bubble(ax, g, a, b, title=f"{name}\n(n={len(g):,})")
    axes[0].set_ylabel(b)
    fig.tight_layout()
    return fig


def diagonal(df, a, b, lo, hi, given=None, ax=None):
    """
    Scatter of P(b = t | a = lo) against P(b = t | a = hi), one point per level t of b.
    Overall points are squares; points within each level of `given` use other markers.
    Points ON the dashed diagonal mean: knowing a gives no information about b.
    """
    ax = ax or plt.subplots(figsize=(5.2, 5))[1]
    lb = _levels(df[b])

    def pts(d):
        ct = pd.crosstab(d[a], d[b]).reindex(columns=lb, fill_value=0)
        pr = ct.div(ct.sum(axis=1), axis=0)
        return pr.loc[lo].values, pr.loc[hi].values

    x, y = pts(df)
    for i, t in enumerate(lb):
        ax.scatter(x[i], y[i], s=140, marker="s", color=_PALETTE[i % 5], edgecolor="k", label=f"{t} (all)", zorder=3)
    if given:
        marks = ["o", "^", "D", "v", "P", "X"]
        for m, (k, g) in zip(marks, df.groupby(given, observed=True)):
            if (g[a] == lo).sum() < 5 or (g[a] == hi).sum() < 5:
                continue
            x, y = pts(g)
            for i, t in enumerate(lb):
                ax.scatter(x[i], y[i], s=70, marker=m, color=_PALETTE[i % 5], edgecolor="k", alpha=.85, zorder=3)
    top = 1.0
    ax.plot([0, top], [0, top], "k--", lw=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"P({b} | {a} = {lo})")
    ax.set_ylabel(f"P({b} | {a} = {hi})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    return ax


def jitter(ax, df, a, b, n=2500, seed=0, title=None):
    """Raw jittered scatter of two categorical variables (one dot per person)."""
    rng = np.random.default_rng(seed)
    d = df.sample(min(n, len(df)), random_state=seed)
    la, lb = _levels(df[a]), _levels(df[b])
    x = pd.Categorical(d[a], categories=la).codes + rng.uniform(-.3, .3, len(d))
    y = pd.Categorical(d[b], categories=lb).codes
    for i, t in enumerate(lb):
        m = y == i
        ax.scatter(x[m], y[m] + rng.uniform(-.3, .3, m.sum()), s=5, alpha=.5, color=_PALETTE[i % 5], label=t)
    ax.set_xticks(range(len(la)))
    ax.set_xticklabels(la)
    ax.set_yticks(range(len(lb)))
    ax.set_yticklabels(lb)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(title or f"{len(d):,} random people")
