"""
Helper functions for Lab 4: From Counts to Dependence Structures.

You do not need to read this file in detail. It contains small utilities for
loading data, computing conditional proportions, running association tests, and
plotting so that you can focus on interpreting the results.
"""
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2 as _chi2_dist

# ---- column names -----------------------------------------------------------
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
    """Load a travellers CSV and put categorical levels in a sensible order."""
    df = pd.read_csv(path)
    for col, lv in ORDER.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=lv, ordered=True)
    return df


def load_group(group, folder="data"):
    """Load <folder>/group_<group>.csv."""
    return load_file(f"{folder}/group_{group}.csv")


def subsample(df, n, seed=0):
    """Random subsample of n rows (without replacement)."""
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def _levels(s):
    return list(s.cat.categories) if hasattr(s, "cat") else sorted(s.unique())


# ------------------------------------------------------------------ tables --
def counts(df, a, b, given=None):
    """Contingency table of counts for (a, b); returns a dict if `given` is set."""
    def one(d):
        ct = pd.crosstab(d[a], d[b]).reindex(
            index=_levels(df[a]), columns=_levels(df[b]), fill_value=0
        )
        return ct
    if given is None:
        return one(df)
    given = [given] if isinstance(given, str) else list(given)
    return {k: one(g) for k, g in df.groupby(given, observed=True)}


def cond_props(df, a, b, given=None, warn_below=100):
    """
    Row-normalised table P(b | a) (each row sums to 1), with a column `n`
    giving the number of people in each row.
    If `given` is set, returns a dict with one table per level of the
    conditioning variable(s).
    Strata smaller than `warn_below` people trigger a warning.
    """
    def one(d):
        ct = pd.crosstab(d[a], d[b]).reindex(
            index=_levels(df[a]), columns=_levels(df[b]), fill_value=0
        )
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


# ------------------------------------------------------------- association --
def cramers_v(ct):
    """Cramer's V for a contingency table."""
    ct = np.asarray(ct, dtype=float)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    rows = ct.sum(axis=1, keepdims=True)
    cols = ct.sum(axis=0, keepdims=True)
    n = ct.sum()
    if n == 0:
        return 0.0
    E = rows @ cols / n
    chi2 = ((ct - E) ** 2 / np.where(E == 0, 1, E)).sum()
    k = min(ct.shape)
    return np.sqrt(chi2 / (n * (k - 1)))


def _chi2_stack(cnt):
    """Pearson chi-square for a stack of tables, shape (m, r, c)."""
    rows = cnt.sum(axis=2)
    cols = cnt.sum(axis=1)
    n = rows.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = rows[:, :, None] * cols[:, None, :] / n[:, None, None]
        term = np.where(E == 0, 0, (cnt - E) ** 2 / E)
    return np.sum(term, axis=(1, 2)), np.sum((E > 0).astype(int), axis=(1, 2))


def _table_stack(df, a, b, given):
    """Return a stack of contingency tables for every combination of `given`."""
    given = list(given)
    levels = [(_levels(df[g]) if g in ORDER else sorted(df[g].unique())) for g in given]
    la, lb = _levels(df[a]), _levels(df[b])
    all_tabs = []
    for combo in np.array(np.meshgrid(*levels)).T.reshape(-1, len(given)):
        mask = np.ones(len(df), dtype=bool)
        for g, lvl in zip(given, combo):
            mask &= df[g].astype(str) == str(lvl)
        ct = pd.crosstab(df.loc[mask, a], df.loc[mask, b]).reindex(
            index=la, columns=lb, fill_value=0
        )
        all_tabs.append(ct.values)
    return np.array(all_tabs)


def assoc_test(df, a, b, given=None, method="asymptotic", n_perm=999, seed=0):
    """
    Test association between categorical variables a and b.

    Parameters
    ----------
    df : DataFrame
    a, b : column names
    given : str or list of str, optional conditioning variable(s)
    method : "asymptotic" or "perm"
    n_perm : number of permutations for the permutation test
    seed : random seed for permutation reproducibility

    Returns
    -------
    dict with keys:
        V      : Cramer's V (effect size, 0 = no association)
        stat   : chi-square statistic (asymptotic) or pseudo-statistic (perm)
        p      : p-value
        dof    : degrees of freedom (asymptotic only)
        n      : number of observations used
    """
    df = df.dropna(subset=[a, b] + ([] if given is None else ([given] if isinstance(given, str) else list(given))))
    if given is None:
        ct = pd.crosstab(df[a], df[b]).reindex(index=_levels(df[a]), columns=_levels(df[b]), fill_value=0)
        V = cramers_v(ct)
        n = int(ct.values.sum())
        if method == "asymptotic":
            stat, p, dof, _ = _chi2(ct.values)
            return {"V": V, "stat": float(stat), "p": float(p), "dof": int(dof), "n": n}
        # permutation
        rng = np.random.default_rng(seed)
        stat0, _ = _chi2_stack(ct.values[None, :, :])
        stat_perm = []
        for _ in range(n_perm):
            shuffled = df.copy()
            shuffled[b] = rng.permutation(shuffled[b].values)
            ct_perm = pd.crosstab(shuffled[a], shuffled[b]).reindex(
                index=_levels(df[a]), columns=_levels(df[b]), fill_value=0
            )
            s, _ = _chi2_stack(ct_perm.values[None, :, :])
            stat_perm.append(s[0])
        p = (np.sum(np.array(stat_perm) >= stat0[0]) + 1) / (n_perm + 1)
        return {"V": V, "stat": float(stat0[0]), "p": float(p), "dof": None, "n": n}

    given = [given] if isinstance(given, str) else list(given)
    stack = _table_stack(df, a, b, given)
    V = cramers_v(np.sum(stack, axis=0))
    n = int(stack.sum())
    if method == "asymptotic":
        stats, dof = _chi2_stack(stack)
        dof_total = int(dof.sum() - len(stack))
        stat_total = float(stats.sum())
        p = float(_chi2_dist.sf(stat_total, max(dof_total, 1)))
        return {"V": V, "stat": stat_total, "p": p, "dof": dof_total, "n": n}

    rng = np.random.default_rng(seed)
    stat0, _ = _chi2_stack(stack)
    stat_perm = []
    for _ in range(n_perm):
        shuffled = df.copy()
        shuffled[b] = rng.permutation(shuffled[b].values)
        s, _ = _chi2_stack(_table_stack(shuffled, a, b, given))
        stat_perm.append(s.sum())
    p = (np.sum(np.array(stat_perm) >= stat0.sum()) + 1) / (n_perm + 1)
    return {"V": V, "stat": float(stat0.sum()), "p": float(p), "dof": None, "n": n}


def _chi2(ct):
    """Chi-square test for one table. Returns (stat, p, dof, expected)."""
    from scipy.stats import chi2_contingency
    return chi2_contingency(ct)


def format_p(p):
    """Pretty p-value string."""
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


# ----------------------------------------------------------------- plotting --
def bubble(ax, df, a, b, title=None):
    """Bubble scatter of P(b | a); size of bubble = probability."""
    la, lb = _levels(df[a]), _levels(df[b])
    ct = pd.crosstab(df[a], df[b]).reindex(index=la, columns=lb, fill_value=0)
    prop = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    for yi, y in enumerate(lb):
        for xi, x in enumerate(la):
            v = prop.loc[x, y]
            ax.scatter(xi, yi, s=max(v, 0.003) * 3500,
                       color=_PALETTE[yi % 5], alpha=.75, edgecolor="k")
            ax.text(xi, yi, f"{v:.0%}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(la)))
    ax.set_xticklabels(la)
    ax.set_yticks(range(len(lb)))
    ax.set_yticklabels(lb)
    ax.set_xlim(-.6, len(la) - .4)
    ax.set_ylim(-.6, len(lb) - .4)
    ax.set_xlabel(a)
    ax.set_title(title or f"P({b} | {a})  (n={len(df):,})", fontsize=10)
    ax.grid(alpha=.2)


def bubble_panels(df, a, b, given=None, figsize=None):
    """Bubble plot overall and, if `given` is set, within each level of `given`."""
    groups = [("All", df)]
    if given:
        groups += [(f"{given} = {k}", g) for k, g in df.groupby(given, observed=True)]
    fig, axes = plt.subplots(1, len(groups), figsize=figsize or (4.5 * len(groups), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, g) in zip(axes, groups):
        bubble(ax, g, a, b, title=f"{name}\n(n={len(g):,})")
    axes[0].set_ylabel(b)
    fig.tight_layout()
    return fig


def diagonal(df, a, b, lo, hi, given=None, ax=None):
    """
    Scatter of P(b | a = lo) against P(b | a = hi), one point per level of b.
    Points on the dashed diagonal mean: knowing a gives no information about b.
    """
    ax = ax or plt.subplots(figsize=(5.2, 5))[1]
    lb = _levels(df[b])

    def pts(d):
        ct = pd.crosstab(d[a], d[b]).reindex(columns=lb, fill_value=0)
        pr = ct.div(ct.sum(axis=1), axis=0)
        return pr.loc[lo].values, pr.loc[hi].values

    x, y = pts(df)
    for i, t in enumerate(lb):
        ax.scatter(x[i], y[i], s=140, marker="s", color=_PALETTE[i % 5],
                   edgecolor="k", label=f"{t} (all)", zorder=3)
    if given:
        marks = ["o", "^", "D", "v", "P", "X"]
        for m, (k, g) in zip(marks, df.groupby(given, observed=True)):
            if (g[a] == lo).sum() < 5 or (g[a] == hi).sum() < 5:
                continue
            x, y = pts(g)
            for i, t in enumerate(lb):
                ax.scatter(x[i], y[i], s=70, marker=m, color=_PALETTE[i % 5],
                           edgecolor="k", alpha=.85, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"P({b} | {a} = {lo})")
    ax.set_ylabel(f"P({b} | {a} = {hi})")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    return ax


# ------------------------------------------------------------- sample sweep --
def sample_sweep(df, a, b, given=None, sizes=(200, 500, 1000, 5000), n_rep=50, seed=1):
    """
    Re-run assoc_test(a, b | given) on subsamples of different sizes.
    Returns a DataFrame with columns: n, rep, V, p.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in sizes:
        if n > len(df):
            continue
        for rep in range(n_rep):
            sub = df.sample(n=n, random_state=rng.integers(1_000_000))
            r = assoc_test(sub, a, b, given=given, method="asymptotic")
            rows.append({"n": n, "rep": rep, "V": r["V"], "p": r["p"]})
    return pd.DataFrame(rows)
