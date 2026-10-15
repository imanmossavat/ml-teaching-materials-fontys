"""
Generative model for the 'travel' causal-structure lab.

Standard world:      C <- F -> P -> T
Hidden-edge world:   same, plus a direct C -> T edge
All probabilities are invented for teaching.
"""
import numpy as np
import pandas as pd

F_LV = ["Wealthy", "Middle income", "Limited means"]
C_LV = ["First", "Second", "Third"]
P_LV = ["Business", "Leisure", "Migration"]
T_LV = ["Solo", "Couple", "Family"]

COLS = {"F": "financial_status", "C": "ticket_class", "P": "purpose", "T": "party"}

P_F = np.array([0.10, 0.40, 0.50])
P_C_F = np.array([[.70, .25, .05],
                  [.20, .50, .30],
                  [.05, .15, .80]])
P_P_F = np.array([[.25, .70, .05],
                  [.30, .40, .30],
                  [.15, .15, .70]])
P_T_P = np.array([[.70, .20, .10],
                  [.25, .40, .35],
                  [.35, .15, .50]])

# direct-edge tilt: multiplicative weights on (Solo, Couple, Family) by class
W_C = np.array([[1.0, 1.8, 0.6],     # First: more couples, fewer families
                [1.0, 1.0, 1.0],     # Second: neutral
                [1.0, 0.7, 1.3]])    # Third: fewer couples, more families


def p_t_given_pc(direct_edge):
    """Array [P, C, T] of P(T | P, C)."""
    out = np.zeros((3, 3, 3))
    for k in range(3):
        for j in range(3):
            w = W_C[j] if direct_edge else np.ones(3)
            v = P_T_P[k] * w
            out[k, j] = v / v.sum()
    return out


def _sample(parent_flat, table_flat, rng):
    u = rng.random(len(parent_flat))
    cum = np.cumsum(table_flat[parent_flat], axis=1)
    return (u[:, None] > cum).sum(axis=1)


def simulate(n, seed, direct_edge=False, do_C=None, prior=None):
    """Simulate n travellers. do_C = 0/1/2 forces ticket class (intervention)."""
    rng = np.random.default_rng(seed)
    pF = P_F if prior is None else np.asarray(prior)
    f = rng.choice(3, size=n, p=pF)
    c = _sample(f, P_C_F, rng)
    p = _sample(f, P_P_F, rng)
    if do_C is not None:
        c = np.full(n, do_C)
    tab = p_t_given_pc(direct_edge)
    t = _sample(p * 3 + c, tab.reshape(9, 3), rng)
    return pd.DataFrame({
        COLS["F"]: pd.Categorical.from_codes(f, F_LV),
        COLS["C"]: pd.Categorical.from_codes(c, C_LV),
        COLS["P"]: pd.Categorical.from_codes(p, P_LV),
        COLS["T"]: pd.Categorical.from_codes(t, T_LV),
    })


def exact_joint(direct_edge=False, prior=None):
    """Exact joint array [F, C, P, T]."""
    pF = P_F if prior is None else np.asarray(prior)
    tab = p_t_given_pc(direct_edge)
    J = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    J[i, j, k, l] = pF[i] * P_C_F[i, j] * P_P_F[i, k] * tab[k, j, l]
    return J


def collider_data(n, seed):
    """Stage-5 dataset: two independent causes (loyalty, peak season) of an upgrade."""
    rng = np.random.default_rng(seed)
    L = rng.random(n) < 0.25
    S = rng.random(n) < 0.40
    pU = np.where(L, np.where(S, 0.20, 0.70), np.where(S, 0.01, 0.15))
    U = rng.random(n) < pU
    return pd.DataFrame({
        "loyalty_member": np.where(L, "Yes", "No"),
        "peak_season": np.where(S, "Peak", "Off-peak"),
        "upgraded": np.where(U, "Yes", "No"),
    })
