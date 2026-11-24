# ids-drift-audit

Monitoring network intrusion detectors under temporal and cross-network drift with a small number of labels.

A network intrusion detector is trained once and then runs for months. Over time the traffic changes (new attack
tools, new normal behaviour, a different network), and the detector can quietly get worse without any visible sign.
This code shows how to **notice** that, **measure** how bad it is with only a few hundred labelled flows, and
**decide** what to do about it.

## How it works

The monitor, called **detect-audit-decide**, looks at the detector's traffic one window at a time (a window is a
fixed number of consecutive flows).

1. **Detect.** Cheap signals that need no labels are checked on every window: how often the detector is unsure
   (conformal abstention), whether a small classifier can tell the new window from the training data (domain
   classifier AUC), whether two detectors disagree, and how confident the predictions are. Each signal has a
   threshold set from windows known to be unshifted. If a signal fires, the window is sent to an audit.
2. **Audit.** An analyst labels a small sample of flows from the window (50 to 1,000, usually about 200). The sample
   is not drawn uniformly: flows where the detector is least certain are more likely to be picked ("active
   testing"), and every flow is weighted so that the estimate of the detector's macro-F1 stays unbiased. The result
   comes with an interval, including an allowance for attack classes that were never seen in the sample.
3. **Decide.** The audit is also used to name the kind of drift: a new attack class, a change in class proportions
   (prior), a change in the input features (covariate) or a change in how features map to labels (concept). A simple
   rule then chooses between waiting, recalibrating and retraining, and it counts the labels it spends.

Several simpler monitoring policies are simulated on the same data for comparison (never retrain, retrain on a
schedule, retrain on an alarm, a drift-detection method, and a conformal method).

## Data

Three public NetFlow v3 benchmarks are used: NF-UNSW-NB15-v3, NF-ToN-IoT-v3 and NF-CSE-CIC-IDS2018-v3. They are not
included. Download them from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ (a browser session is needed) and
unpack the three bundles into `data/raw/` inside this folder, keeping the bundle folder names:

    data/raw/unsw_v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv
    data/raw/cic_v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv
    data/raw/ton_v3/02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv

`python tools/check_bag.py <bundle folder>` checks a bundle against its SHA-1 manifest. The data paths in `configs/`
are relative to the `configs/` folder; change them if you keep the data elsewhere.

## Installation

Python 3.13 was used.

    pip install -r requirements.txt

## What is in this folder

| Path | What it is |
|---|---|
| `driftids/` | The library: data loading, metrics, audit designs and estimators (`audit.py`), label-free estimators, controlled shifts, alarms, drift diagnosis (`attribution.py`), detectors, policies |
| `phase0.py` | Describes the data, makes the time split, runs leakage and shortcut-feature checks, and checks that drift exists |
| `phase1.py` | Controlled shifts: compares audit designs and label-free estimators |
| `phase2a_attribution.py` | Controlled shifts: tests the drift diagnosis |
| `phase2b_streams.py` | Real streams (over time, and from one network to another): audits, label-free estimates, alarms |
| `phase2a_policies.py`, `phase2b_policies.py` | Simulates the monitoring policies over a stream and counts labels used |
| `analysis_hypotheses.py` | Statistical checks of the main claims from the saved outputs |
| `configs/` | One configuration per dataset or scenario (source period, windows, splits) |
| `tools/` | Result tables and figures, seed merging, dataset checksum check |
| `scripts/` | `run_all.sh` and `run_all.ps1` run the complete experiment in order |
| `results/final/` | Saved outputs of the complete run (per window, audits, policy traces, analysis) |
| `figures/` | Result figures (PDF and PNG) |
| `tests/` | Unit tests |

## Running the experiment

    bash scripts/run_all.sh                 # Linux / macOS
    powershell -File scripts\run_all.ps1    # Windows

The script runs, in order: the time split with leakage and shortcut checks, controlled shifts (estimation and
diagnosis), the real streams and cross-network pairs, the policy simulations (10 seeds, then 10 more seeds for
detect-audit-decide and periodic retraining), and the statistical checks. Results go to `results/final/` (set
`DRIFTIDS_RESULTS` to write somewhere else). The full run took about 18 hours on a 16-core / 24-thread CPU with 32 GB
of RAM; the CSE-CIC-IDS2018 and ToN-IoT steps need about 10-15 GB of RAM. All random seeds are fixed.

Tables and figures from the saved results:

    python tools/make_tables.py       # result tables (markdown)
    python tools/make_figures.py      # result figures
    python tools/make_fig1.py         # diagram of the monitor

The included `results/final/` lets you run these without repeating the experiments.

## Settings

The defaults are: posterior intervals with the unseen-class allowance (`DRIFTIDS_INTERVAL=posterior3`), the diagnosis
rule `diagnose_drift`, and error-driven retraining with tolerance 0.01 (`DRIFTIDS_ERR_TOL`). Other interval methods
for comparison are `posterior2` (without the unseen-class allowance) and `bootstrap`. `DRIFTIDS_THREADS` sets the
threads per detector; `--jobs` sets the number of parallel workers.

## Tests

    python -m unittest discover -s tests

## License

MIT. See `LICENSE`.
