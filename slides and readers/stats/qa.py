"""Quality-assurance checks shared by generate_data.py and validate_data.py."""
import sys
import pathlib
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chisquare

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / ".." / ".." / "stats" / "04_dependence_structures_counts_to_graphs"))
from lab_helpers import F, C, P, T, ORDER, assoc_test  # noqa: E402
import model  # noqa: E402

VARS = [F, C, P, T]


def dag(hidden):
    G = nx.DiGraph()
    G.add_nodes_from(VARS)
    G.add_edges_from([(F, C), (F, P), (P, T)])
    if hidden:
        G.add_edge(C, T)
    return G


def skeleton(hidden):
    return {frozenset(e) for e in dag(hidden).edges}


def cond_sets(a, b):
    others = [v for v in VARS if v not in (a, b)]
    return [(), (others[0],), (others[1],), tuple(others)]


def all_tests(df, hidden, n_perm=199, seed=0):
    """Run all 24 (pair, conditioning set) tests. Null tests use permutation, others asymptotic."""
    G = dag(hidden)
    rows = []
    for a, b in combinations(VARS, 2):
        for s in cond_sets(a, b):
            indep = nx.is_d_separator(G, {a}, {b}, set(s))
            r = assoc_test(df, a, b, given=list(s) or None, n_perm=n_perm, seed=seed,
                           method="perm" if indep else "asymptotic")
            rows.append(dict(a=a, b=b, given=s, truth_independent=indep, p=r["p"], V=r["V"], stat=r["stat"]))
    return pd.DataFrame(rows)


def recovered_skeleton(tests, alpha):
    """Stage-3 procedure: keep an edge iff NO conditioning set shows independence (p > alpha)."""
    edges = set()
    for (a, b), g in tests.groupby(["a", "b"]):
        if not (g["p"] > alpha).any():
            edges.add(frozenset((a, b)))
    return edges


def cpt_fidelity(df, hidden):
    """Goodness-of-fit of each empirical CPT row to the specification. Returns (min p, max |dev|)."""
    ps, devs = [], []

    def gof(obs, exp):
        obs = np.asarray(obs, float)
        exp = np.asarray(exp, float)
        devs.append(np.abs(obs / obs.sum() - exp).max())
        ps.append(chisquare(obs, exp * obs.sum()).pvalue)

    gof(df[F].value_counts().reindex(ORDER[F]).values, model.P_F)
    for i, f in enumerate(ORDER[F]):
        g = df[df[F] == f]
        gof(g[C].value_counts().reindex(ORDER[C]).values, model.P_C_F[i])
        gof(g[P].value_counts().reindex(ORDER[P]).values, model.P_P_F[i])
    tab = model.p_t_given_pc(hidden)
    for k, p in enumerate(ORDER[P]):
        for j, c in enumerate(ORDER[C]):
            g = df[(df[P] == p) & (df[C] == c)]
            gof(g[T].value_counts().reindex(ORDER[T]).values, tab[k, j])
    return min(ps), max(devs)


def qa_dataset(df, hidden, n_perm=999, verbose=False):
    """
    Acceptance criteria for a dataset shipped to students. Returns (ok, details dict).
    Criteria exist only to reject unlucky random draws that would mislead a class.
    """
    d = {}
    # structural sanity
    d["n_ok"] = len(df) == 10000
    d["no_missing"] = not df[VARS].isna().any().any()
    d["levels_ok"] = all(set(df[v].astype(str)) == set(ORDER[v]) for v in VARS)

    tests = all_tests(df, hidden, n_perm=n_perm)
    d["tests"] = tests
    sk = skeleton(hidden)
    # 1. every true edge must be clearly dependent under every conditioning set
    #    (except the collider-activated set in the hidden world, which is underpowered)
    bad_edge = []
    for _, r in tests.iterrows():
        if frozenset((r.a, r.b)) in sk and r.p > 1e-6:
            bad_edge.append((r.a, r.b, r.given, r.p))
    d["edges_clear"] = not bad_edge
    # 2. the weak-but-real non-adjacent dependencies must be clearly visible marginally
    marg = tests[tests.given.apply(len) == 0].set_index(["a", "b"]).p
    d["weak_marginals_clear"] = bool(marg[(C, T)] < 1e-4 and marg[(F, T)] < 1e-4 and marg[(C, P)] < 1e-6)
    # 3. every true independence must look like independence (p > 0.05), not borderline
    nulls = tests[tests.truth_independent]
    d["nulls_clean"] = bool((nulls.p > 0.05).all())
    d["min_null_p"] = float(nulls.p.min())
    # 4. per-stratum checks that students will eyeball
    strat_ps = []
    stratified = [(C, T, P), (C, T, F), (F, T, P), (C, P, F)] if not hidden else [(C, P, F)]
    for a, b, g in stratified:
        for _, sub in df.groupby(g, observed=True):
            strat_ps.append(assoc_test(sub, a, b, n_perm=499)["p"])
    d["min_stratum_p"] = float(min(strat_ps))
    d["strata_clean"] = bool(min(strat_ps) > 0.02)
    # 5. fidelity to the specified probability tables
    mp, md = cpt_fidelity(df, hidden)
    d["cpt_min_p"], d["cpt_max_dev"] = float(mp), float(md)
    d["cpt_ok"] = bool(mp > 0.001)
    # 6. hidden world: the extra edge must be unmistakable
    if hidden:
        ct = tests[(tests.a == C) & (tests.b == T)]
        d["hidden_edge_clear"] = bool((ct.p < 1e-10).all())
    ok = all(v for k, v in d.items() if k.endswith(("_ok", "_clear", "_clean")) or k in ("no_missing",))
    return ok, d
