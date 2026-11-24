"""Unit tests for the Phase 0 data path and leakage checks. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftids import schema as S  # noqa: E402
from driftids.data import dedupe_source, make_time_split, read_netflow, row_hashes  # noqa: E402
from driftids.leakage import (LeakageError, assert_before, assert_fitted_on,  # noqa: E402
                              assert_no_excluded_features, assert_no_raw_column_copies)
from driftids.metrics import binary_metrics, macro_f1, macro_f1_block_bootstrap  # noqa: E402


def synthetic_csv(path: Path, n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(1_700_000_000_000, 1_700_000_000_000 + 3_600_000 * 10, n))
    start[100:110] = start[100]                              # ties
    dur = rng.integers(0, 2000, n)
    dur[50] = 3_600_000 * 4                                   # one very long flow crossing boundaries
    phase = np.arange(n) * 3 // n
    attack = np.where(rng.random(n) < 0.8, "Benign", np.array(["DoS", "Recon", "Exploits"])[phase])
    df = pd.DataFrame({
        "IPV4_SRC_ADDR": [f"10.0.0.{i % 250}" for i in range(n)], "L4_SRC_PORT": rng.integers(1024, 65535, n),
        "IPV4_DST_ADDR": "10.0.1.1", "L4_DST_PORT": rng.choice([80, 443, 53], n),
        "PROTOCOL": rng.choice([6, 17], n), "IN_BYTES": rng.integers(40, 5000, n) + (attack != "Benign") * 3000,
        "OUT_BYTES": rng.integers(40, 5000, n), "FLOW_DURATION_MILLISECONDS": dur,
        S.TIME_START: start, S.TIME_END: start + dur,
        "SRC_TO_DST_IAT_MIN": rng.random(n), S.LABEL_BIN: (attack != "Benign").astype(int), S.LABEL_CLASS: attack})
    perm = rng.permutation(n)                                 # file not in time order
    df.iloc[perm].to_csv(path, index=False)
    return df


class TestSchema(unittest.TestCase):
    def test_excluded(self):
        for c in ("IPV4_SRC_ADDR", "L4_DST_PORT", S.TIME_START, S.TIME_END, "Label", "Attack", "IPV6_SRC_ADDR",
                  "DNS_QUERY_ID"):
            self.assertTrue(S.is_excluded(c), c)
        for c in ("IN_BYTES", "FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_IAT_MIN"):
            self.assertFalse(S.is_excluded(c), c)


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.csv = cls.tmp / "syn.csv"
        cls.raw = synthetic_csv(cls.csv)
        cls.f = read_netflow(cls.csv, chunksize=1000)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_sorted_and_stable(self):
        f = self.f
        self.assertTrue(np.all(np.diff(f.start) >= 0))
        ties = f.start[1:] == f.start[:-1]
        self.assertTrue(np.all(np.diff(f.row_id)[ties] > 0), "ties must keep original row order")
        self.assertEqual(len(f), len(self.raw))

    def test_no_identifier_or_time_features(self):
        assert_no_excluded_features(self.f.features)
        for c in S.IDENTIFIER_COLS + S.TIME_COLS + S.LABEL_COLS:
            self.assertNotIn(c, self.f.features)

    def test_split_is_causal(self):
        f = self.f
        sp = make_time_split(f.start, f.end, n_source=2000, window_size=500, gap_flows=200, inner_gap=50)
        src = sp.source
        for w in sp.windows:
            self.assertLess(max(f.start[src].max(), f.end[src].max()), f.start[w].min())
        self.assertGreater(sp.dropped["crossing"], 0, "the long flow must be removed")
        self.assertNotIn(int(np.where(f.end - f.start == 3_600_000 * 4)[0][0]), set(src.tolist()))

    def test_interleaved_split_covers_staged_classes_and_stays_causal(self):
        f = self.f
        sp = make_time_split(f.start, f.end, n_source=3000, window_size=500, gap_flows=200, inner_gap=20,
                             mode="interleaved", chunk_size=100)
        cls = np.asarray(f.attack, dtype=object).astype(str)
        src_classes = set(cls[sp.source])
        for b in ("train", "val", "calib", "test"):            # staged attacks reach every block
            self.assertEqual(set(cls[sp.blocks[b]]) & src_classes, src_classes, b)
        n = {b: len(v) for b, v in sp.blocks.items()}
        self.assertGreater(n["train"], 2.5 * n["test"])         # roughly 6 : 2
        src = sp.source
        for w in sp.windows:
            self.assertLess(max(f.start[src].max(), f.end[src].max()), f.start[w].min())
        segs = [a for _, a in sp.segments if len(a)]
        for a, b in zip(segs, segs[1:]):                        # each chunk ends before the next starts
            self.assertLess(f.end[a].max(), f.start[b].min())

    def test_class_map(self):
        from driftids.data import apply_class_map
        f = read_netflow(self.csv, chunksize=1000)
        apply_class_map(f, {"DoS": "Flood", "Recon": "Probe", "Exploits": "Probe"})
        self.assertEqual(set(np.asarray(f.attack, dtype=object)), {"Benign", "Flood", "Probe"})
        g = read_netflow(self.csv, chunksize=1000)
        with self.assertRaises(ValueError):                     # unmapped attack label must stop the run
            apply_class_map(g, {"DoS": "Flood"})

    def test_lookahead_detected(self):
        f = self.f
        with self.assertRaises(LeakageError):
            assert_before(np.arange(0, 1000), np.arange(900, 1200), f.start, f.end)

    def test_dedupe_keeps_earliest(self):
        f = self.f
        sp = make_time_split(f.start, f.end, n_source=2000, window_size=500, gap_flows=200)
        h = row_hashes(f.X, f.attack)
        t_row, e_row = sp.blocks["train"][0], sp.blocks["test"][0]
        h[e_row] = h[t_row]                                     # plant a copy of a train row in the test block
        dedupe_source(sp, h)
        self.assertIn(t_row, sp.blocks["train"])
        self.assertNotIn(e_row, sp.blocks["test"])

    def test_row_hashes(self):
        X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 1.0], "b": [3.0, 3.0, 3.0, 3.0]})
        y = pd.Categorical(["Benign", "Benign", "Benign", "DoS"])
        h = row_hashes(X, y)
        self.assertEqual(h[0], h[1])                            # same features, same class
        self.assertNotEqual(h[0], h[2])                         # different features
        self.assertNotEqual(h[0], h[3])                         # same features, different class
        hx = row_hashes(X)
        self.assertEqual(hx[0], hx[3])

    def test_raw_copy_detected(self):
        X = np.column_stack([np.arange(5.0), np.array([7, 8, 9, 10, 11.0])])
        with self.assertRaises(LeakageError):
            assert_no_raw_column_copies(X, {"FLOW_START_MILLISECONDS": np.array([7, 8, 9, 10, 11])})

    def test_fitted_on(self):
        with self.assertRaises(LeakageError):
            assert_fitted_on(np.array([1, 2, 99]), np.arange(10))


class TestMetrics(unittest.TestCase):
    def test_macro_f1_min_count(self):
        y = ["Benign"] * 50 + ["DoS"] * 30 + ["Worms"] * 2
        p = ["Benign"] * 50 + ["DoS"] * 30 + ["Benign"] * 2
        f1, cls = macro_f1(y, p, m=20)
        self.assertEqual(cls, ["Benign", "DoS"])               # Worms has < 20 flows: not in the class set
        self.assertGreater(f1, 0.98)

    def test_bootstrap_matches_point_and_is_reproducible(self):
        rng = np.random.default_rng(0)
        y = rng.choice(["Benign", "DoS", "Exploits"], 5000, p=[.8, .1, .1])
        p = np.where(rng.random(5000) < 0.85, y, rng.choice(["Benign", "DoS", "Exploits"], 5000))
        f1, cls = macro_f1(y, p, m=20)
        b1 = macro_f1_block_bootstrap(y, p, cls, block=100, B=300, seed=3)
        b2 = macro_f1_block_bootstrap(y, p, cls, block=100, B=300, seed=3)
        np.testing.assert_array_equal(b1, b2)
        self.assertLess(abs(b1.mean() - f1), 3 * b1.std())
        self.assertGreater(b1.std(), 0)
        # with a block covering everything in order, a resample of 1 block of size n equals the point estimate
        b3 = macro_f1_block_bootstrap(y, p, cls, block=5000, B=3, seed=0)
        self.assertAlmostEqual(b3[0], f1, places=12)

    def test_macro_f1_equals_sklearn(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(5)
        names = ["Benign", "DoS", "Exploits", "Fuzzers", "Worms", "xss"]
        y = rng.choice(names, 4000, p=[.6, .15, .1, .1, .004, .046])
        p = np.where(rng.random(4000) < 0.7, y, rng.choice(names + ["Ghost"], 4000)).astype(object)
        for restrict in (None, ["benign", "dos", "XSS"]):
            f1, cls = macro_f1(y, p, m=20, restrict_to=restrict)
            ref = f1_score(y.astype(str), p.astype(str), labels=cls, average="macro", zero_division=0)
            self.assertAlmostEqual(f1, ref, places=12)
        n_worms = int((y == "Worms").sum())
        self.assertEqual("Worms" in macro_f1(y, p, m=20)[1], n_worms >= 20)    # class set follows the m rule
        self.assertNotIn("Worms", macro_f1(y, p, m=n_worms + 1)[1])

    def test_vectorised_bootstrap_equals_loop(self):
        from driftids.metrics import _macro_f1_block_bootstrap_loop
        rng = np.random.default_rng(1)
        y = rng.choice(["Benign", "DoS", "Exploits", "Fuzzers"], 7350, p=[.7, .1, .1, .1])   # partial last block
        p = np.where(rng.random(7350) < 0.8, y, rng.choice(["Benign", "DoS", "Exploits", "Fuzzers"], 7350))
        _, cls = macro_f1(y, p, m=20)
        a = macro_f1_block_bootstrap(y, p, cls, block=1000, B=150, seed=7)
        b = _macro_f1_block_bootstrap_loop(y, p, cls, block=1000, B=150, seed=7)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_binary(self):
        b = binary_metrics(["Benign", "DoS", "DoS", "Benign"], ["DoS", "DoS", "Benign", "Benign"])
        self.assertAlmostEqual(b["detection_rate"], 0.5)
        self.assertAlmostEqual(b["false_alarm_rate"], 0.5)


class TestPhase0EndToEnd(unittest.TestCase):
    def test_stages(self):
        import phase0
        tmp = Path(tempfile.mkdtemp())
        old = phase0.RESULTS
        try:
            phase0.RESULTS = tmp / "results"
            csv = tmp / "syn.csv"
            synthetic_csv(csv, n=8000)
            phase0.main(["describe", "--csv", str(csv), "--name", "syn"])
            self.assertTrue((tmp / "results" / "syn" / "summary.json").exists())
            cfg = {"name": "syn", "csv": str(csv), "window_size": 1000, "n_source": 3000, "gap_flows": 500, "m": 20}
            cp = tmp / "cfg.json"
            cp.write_text(json.dumps(cfg))
            phase0.main(["screen", "--config", str(cp)])
            with self.assertRaises(SystemExit):                 # not fixed yet: must refuse
                phase0.main(["drift", "--config", str(cp)])
            cfg.update({"shortcut_rule": {"exclude": []}, "fixed": True})
            cp.write_text(json.dumps(cfg))
            phase0.main(["drift", "--config", str(cp), "--detector", "hgb"])
            s = json.loads((tmp / "results" / "syn" / "drift_summary_hgb_preview.json").read_text())
            v = s["variants"]["main"]
            self.assertIn("source_test_macro_f1_known", v)
            self.assertGreater(v["source_test_se"], 0)
            self.assertLessEqual(v["windows_with_drop_ge_delta_beyond_noise"], v["windows_with_drop_ge_delta"])
            # config with interleaved split, class map, mechanical shortcut policy
            cfg2 = {"name": "syn2", "csv": str(csv), "window_size": 1000, "n_source": 3000, "gap_flows": 500,
                    "split_mode": "interleaved", "chunk_size": 200, "inner_gap": 20,
                    "class_map": {"DoS": "DoS", "Recon": "Probe", "Exploits": "Probe"},
                    "shortcut_policy": {"auc_threshold": 0.9, "fingerprint_features": ["IN_BYTES"],
                                        "all_features_variant": True}, "fixed": True}
            cp2 = tmp / "cfg2.json"
            cp2.write_text(json.dumps(cfg2))
            phase0.main(["screen", "--config", str(cp2)])
            phase0.main(["drift", "--config", str(cp2), "--detector", "hgb"])
            s2 = json.loads((tmp / "results" / "syn2" / "drift_summary_hgb_preview.json").read_text())
            self.assertEqual(s2["excluded_by_rule"], ["IN_BYTES"])  # IN_BYTES AUC ~0.92 >= 0.9 and listed
            self.assertIn("all_features", s2["variants"])
            self.assertEqual(s2["split"]["mode"], "interleaved")
            self.assertIn("n_windows_without_new_class", s2["variants"]["main"])
            # cross-network mode: train on syn, deploy on syn2's file
            phase0.main(["drift", "--config", str(cp), "--deploy-config", str(cp2), "--detector", "hgb"])
            self.assertTrue((tmp / "results" / "syn_to_syn2" / "drift_summary_hgb_preview.json").exists())
        finally:
            phase0.RESULTS = old
            phase0._CACHE.clear()
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
