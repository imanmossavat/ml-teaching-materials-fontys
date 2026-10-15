"""
Generate the lab datasets and run a small QA gate.

Usage:
    python generate_data.py [output_dir]

Defaults to ../../stats/04_dependence_structures_counts_to_graphs/data
"""
import sys
import pathlib
import numpy as np
import pandas as pd

import model
from qa import qa_dataset

GROUPS = {
    "A": (False, 1000),
    "B": (False, 2000),
    "C": (False, 3000),
    "D": (True,  4000),
    "E": (False, 5000),
    "F": (False, 6000),
    "G": (True,  7000),
    "H": (False, 8000),
}
COLLIDER_SEED = 5153


def main(outdir):
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for grp, (hidden, seed) in GROUPS.items():
        df = model.simulate(10_000, seed, direct_edge=hidden)
        ok, d = qa_dataset(df, hidden, n_perm=199)
        print(f"group {grp}: QA ok={ok}, min_null_p={d['min_null_p']:.3f}, "
              f"min_stratum_p={d['min_stratum_p']:.3f}, cpt_max_dev={d['cpt_max_dev']:.3f}")
        if not ok:
            print("  WARNING: QA gate did not pass for this random draw.")
        df.insert(0, "traveller_id", range(1, len(df) + 1))
        df.to_csv(out / f"group_{grp}.csv", index=False)
    coll = model.collider_data(10_000, COLLIDER_SEED)
    coll.insert(0, "traveller_id", range(1, len(coll) + 1))
    coll.to_csv(out / "collider_upgrade.csv", index=False)
    print(f"wrote {len(GROUPS)} group files and collider_upgrade.csv to {out}")


if __name__ == "__main__":
    default = "../../stats/04_dependence_structures_counts_to_graphs/data"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
