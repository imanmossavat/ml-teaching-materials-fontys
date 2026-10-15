"""
THE REVEAL: the actual story behind the data.

Everything is invented for teaching.  Each variable is drawn from a conditional
probability table (CPT) given its parents.  Order of drawing:

    F  (no parents)
    C  given F
    P  given F
    T  given P          (standard world)
    T  given P and C    (hidden-edge world: ticket class also nudges who travels as a couple)
"""
import numpy as np
import pandas as pd

F_LV = ["Wealthy", "Middle income", "Limited means"]
C_LV = ["First", "Second", "Third"]
P_LV = ["Business", "Leisure", "Migration"]
T_LV = ["Solo", "Couple", "Family"]

# P(F)
P_F = np.array([0.10, 0.40, 0.50])
# P(C | F): rows = F (Wealthy, Middle, Limited); columns = C (First, Second, Third)
P_C_F = np.array([[.70, .25, .05],
                  [.20, .50, .30],
                  [.05, .15, .80]])
# P(P | F): rows = F; columns = P (Business, Leisure, Migration)
P_P_F = np.array([[.25, .70, .05],
                  [.30, .40, .30],
                  [.15, .15, .70]])
# P(T | P): rows = P; columns = T (Solo, Couple, Family)
P_T_P = np.array([[.70, .20, .10],
                  [.25, .40, .35],
                  [.35, .15, .50]])
# Hidden-edge world only: multiply P(T | P) by these weights (by class), then renormalise
W_C = np.array([[1.0, 1.8, 0.6],     # First
                [1.0, 1.0, 1.0],     # Second
                [1.0, 0.7, 1.3]])    # Third


def p_t_given_pc(world="standard"):
    """Array indexed [P, C, T] = P(T | P, C)."""
    out = np.zeros((3, 3, 3))
    for k in range(3):
        for j in range(3):
            w = W_C[j] if world == "hidden_edge" else np.ones(3)
            v = P_T_P[k] * w
            out[k, j] = v / v.sum()
    return out


def _draw(parent_codes, table, rng):
    u = rng.random(len(parent_codes))
    return (u[:, None] > np.cumsum(table[parent_codes], axis=1)).sum(axis=1)


# ---- one sampler per variable: codes are 0, 1, 2 (index into the *_LV lists) ----
def draw_F(n, rng):
    return rng.choice(3, size=n, p=P_F)


def draw_C(f, rng):
    return _draw(f, P_C_F, rng)


def draw_P(f, rng):
    return _draw(f, P_P_F, rng)


def draw_T(p, c, rng, world="standard"):
    tab = p_t_given_pc(world).reshape(9, 3)
    return _draw(p * 3 + c, tab, rng)


def to_frame(f, c, p, t):
    return pd.DataFrame({
        "financial_status": pd.Categorical.from_codes(f, F_LV),
        "ticket_class": pd.Categorical.from_codes(c, C_LV),
        "purpose": pd.Categorical.from_codes(p, P_LV),
        "party": pd.Categorical.from_codes(t, T_LV),
    })


def simulate(n, seed, world="standard"):
    """Observational simulation (no intervention)."""
    rng = np.random.default_rng(seed)
    f = draw_F(n, rng)
    c = draw_C(f, rng)
    p = draw_P(f, rng)
    t = draw_T(p, c, rng, world)
    return to_frame(f, c, p, t)


def exact_joint(world="standard"):
    """Exact joint distribution as an array indexed [F, C, P, T]."""
    tab = p_t_given_pc(world)
    return np.einsum("f,fc,fp,pct->fcpt", P_F, P_C_F, P_P_F, tab)
