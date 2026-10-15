"""
Monte-Carlo validation of the lab's datasets and teaching claims.
Writes validation_results.pkl (raw) and ../instructor/data_validation_report.md.
Run:  python validate_data.py [R]
"""
import sys, pickle, time, pathlib
import numpy as np
import pandas as pd
from scipy.stats import kstest
import model
from qa import *          # noqa
from qa import all_tests, qa_dataset, recovered_skeleton, skeleton, VARS
from lab_helpers import F, C, P, T, assoc_test

_HERE = pathlib.Path(__file__).resolve().parent
R = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED0 = 900_000
res = {"R": R}
t0 = time.time()

# ---------------------------------------------------------------- 1. full pipeline over R fresh datasets
for hidden in (False, True):
    key = "hidden" if hidden else "standard"
    tests_all, qa_flags = [], []
    for i in range(R):
        df = model.simulate(10_000, SEED0 + i, direct_edge=hidden)
        ok, d = qa_dataset(df, hidden, n_perm=199)
        t = d["tests"].copy()
        t["rep"] = i
        tests_all.append(t)
        qa_flags.append({k: v for k, v in d.items() if k != "tests"} | {"qa_ok": ok, "rep": i})
        if (i + 1) % 25 == 0:
            print(key, i + 1, f"{time.time()-t0:.0f}s", flush=True)
    res[key] = dict(tests=pd.concat(tests_all, ignore_index=True), qa=pd.DataFrame(qa_flags))

# ---------------------------------------------------------------- 2. power / size vs sample size
sizes = [200, 500, 1000, 2000, 5000, 10_000]
Rn = 300
rows = []
for n in sizes:
    for hidden in (False, True):
        for rep in range(Rn):
            df = model.simulate(n, 700_000 + rep + n, direct_edge=hidden)
            r1 = assoc_test(df, C, T, method="asymptotic")["p"]
            r2 = assoc_test(df, C, T, given=P, n_perm=199, seed=rep)["p"]
            r3 = assoc_test(df, F, T, method="asymptotic")["p"]
            rows.append(dict(n=n, hidden=hidden, rep=rep, p_CT=r1, p_CT_given_P=r2, p_FT=r3))
    print("size", n, f"{time.time()-t0:.0f}s", flush=True)
res["by_n"] = pd.DataFrame(rows)

# ---------------------------------------------------------------- 3. collider (upgrade) dataset
rows = []
for rep in range(300):
    d = model.collider_data(10_000, 600_000 + rep)
    pm = assoc_test(d, "loyalty_member", "peak_season", method="asymptotic")["p"]
    yes = d[d.upgraded == "Yes"]
    no = d[d.upgraded == "No"]
    py = assoc_test(yes, "loyalty_member", "peak_season", method="asymptotic")["p"]
    pn = assoc_test(no, "loyalty_member", "peak_season", method="asymptotic")["p"]
    rows.append(dict(rep=rep, p_marginal=pm, p_given_upgraded=py, p_given_not_upgraded=pn, n_upgraded=len(yes)))
res["collider"] = pd.DataFrame(rows)

# ---------------------------------------------------------------- 4. large-sample extension
big = {}
for name, hidden in [("large_sample_A", False), ("large_sample_D", True)]:
    d = pd.read_csv(_HERE / "extension" / f"{name}.csv")
    from lab_helpers import ORDER
    for col, lv in ORDER.items():
        d[col] = pd.Categorical(d[col], categories=lv)
    big[name] = {
        "C-P | F": assoc_test(d, C, P, given=F, n_perm=499),
        "C-P | F,T": assoc_test(d, C, P, given=[F, T], n_perm=499),
    }
res["big"] = big

with open(_HERE / "validation_results.pkl", "wb") as fh:
    pickle.dump(res, fh)
print("saved", f"{time.time()-t0:.0f}s")
