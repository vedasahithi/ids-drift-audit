"""Final run: append the extra-seed traces (e.g. dad + periodic, seeds 10-19) to the main policy trace, so the H5
test uses 20 seeds for dad and periodic. Keeps the original as trace_main.csv.
Usage: python tools/merge_seeds.py <main policies dir> <extra policies dir>"""
import sys
from pathlib import Path

import pandas as pd

main, extra = Path(sys.argv[1]), Path(sys.argv[2])
if not (main / "trace_main.csv").exists():
    (main / "trace.csv").replace(main / "trace_main.csv")
t0 = pd.read_csv(main / "trace_main.csv")
t1 = pd.read_csv(extra / "trace.csv")
assert not set(t0.seed) & set(t1.seed), "seed overlap"
pd.concat([t0, t1], ignore_index=True).to_csv(main / "trace.csv", index=False)
print(f"{main.name}: {t0.seed.nunique()} + {t1.seed.nunique()} seeds; policies in extra: {sorted(t1.policy.unique())}")
