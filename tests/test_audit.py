"""Statistical checks of the audit estimators on a synthetic window with a known truth."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftids.audit import (active_audit, estimate, random_audit, stratified_active_audit,  # noqa: E402
                            stratified_audit, true_metrics)
from driftids.labelfree import atc, atc_threshold, soft_confusion  # noqa: E402

CLASSES = np.array(["Benign", "DoS", "Exploits", "Fuzzers"])


def make_window(N=10_000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(CLASSES, N, p=[0.85, 0.05, 0.06, 0.04])
    # detector: right 80% of the time; misses half of DoS as Benign ("silent" errors with high confidence)
    pred = np.where(rng.random(N) < 0.8, y, rng.choice(CLASSES, N))
    silent = (y == "DoS") & (rng.random(N) < 0.5)
    pred[silent] = "Benign"
    proba = np.full((N, 4), 0.05)
    proba[np.arange(N), np.searchsorted(CLASSES, pred)] = 0.85
    return y, pred, proba


class TestAuditEstimators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.pred, cls.proba = make_window()
        cls.truth = true_metrics(cls.y, cls.pred, m=20)
        cls.acq = 1 - cls.proba.max(1) + 0.01

    def run_design(self, design, n=500, R=300):
        rng = np.random.default_rng(1)
        est, cover = [], []
        pred_codes = np.searchsorted(CLASSES, self.pred)
        for r in range(R):
            if design == "random":
                s = random_audit(len(self.y), n, rng)
            elif design == "stratified":
                s = stratified_audit(pred_codes, n, rng)
            elif design == "stratactive":
                s = stratified_active_audit(pred_codes, self.acq, n, rng, lam=0.5)
            else:
                s = active_audit(self.acq, n, rng, lam=0.5)
            e = estimate(s, self.y[s.idx], self.pred[s.idx], m=20, B=200, seed=r)
            est.append(e["macro_f1"])
            cover.append(e["macro_f1_lo"] <= self.truth["macro_f1"] <= e["macro_f1_hi"])
        est = np.array(est)
        return est.mean() - self.truth["macro_f1"], est.std(), np.mean(cover)

    def test_designs_approximately_unbiased_with_coverage(self):
        for design in ("random", "stratified", "active", "stratactive"):
            bias, sd, cov = self.run_design(design)
            with self.subTest(design=design):
                self.assertLess(abs(bias), 0.25 * sd + 0.01, f"bias {bias:.4f} sd {sd:.4f}")
                self.assertGreater(cov, 0.88, f"coverage {cov:.3f}")

    def test_posterior2_handles_zero_events(self):
        # a window with very few false alarms: an audit often sees none; the bootstrap interval then collapses to
        # [0, 0], the posterior2 interval must still reach above zero
        rng = np.random.default_rng(4)
        N = 10_000
        y = np.where(rng.random(N) < 0.1, "DoS", "Benign")
        pred = y.copy()
        fa = np.flatnonzero(y == "Benign")[:20]                 # 20 false alarms: rate ~0.0022
        pred[fa] = "DoS"
        truth = true_metrics(y, pred, m=20)
        s = random_audit(N, 100, np.random.default_rng(0))
        if (pred[s.idx][y[s.idx] == "Benign"] == "DoS").any():
            self.skipTest("sample happened to contain a false alarm")
        b = estimate(s, y[s.idx], pred[s.idx], m=20, interval="bootstrap")
        p = estimate(s, y[s.idx], pred[s.idx], m=20, interval="posterior2")
        self.assertEqual(b["false_alarm_rate_hi"], 0.0)
        self.assertGreater(p["false_alarm_rate_hi"], truth["false_alarm_rate"])
        self.assertLessEqual(p["false_alarm_rate_lo"], truth["false_alarm_rate"])

    def test_posterior2_boundary_when_truth_is_exactly_zero(self):
        # a perfect detector: true false-alarm rate exactly 0 and binary F1 exactly 1; intervals must include them
        rng = np.random.default_rng(6)
        y = np.where(rng.random(5000) < 0.1, "DoS", "Benign")
        s = random_audit(5000, 200, np.random.default_rng(1))
        e = estimate(s, y[s.idx], y[s.idx], m=20, interval="posterior2")
        self.assertEqual(e["false_alarm_rate_lo"], 0.0)
        self.assertEqual(e["bin_f1_hi"], 1.0)
        self.assertEqual(e["detection_rate_hi"], 1.0)
        self.assertEqual(e["macro_f1_hi"], 1.0)

    def test_posterior2_coverage_synthetic(self):
        rng = np.random.default_rng(2)
        cov = []
        for r in range(150):
            s = random_audit(len(self.y), 200, rng)
            e = estimate(s, self.y[s.idx], self.pred[s.idx], m=20, B=200, seed=r, interval="posterior2")
            cov.append(e["macro_f1_lo"] <= self.truth["macro_f1"] <= e["macro_f1_hi"])
        self.assertGreater(np.mean(cov), 0.88)

    def test_weights_sum_to_window_size(self):
        rng = np.random.default_rng(3)
        N = len(self.y)
        s = random_audit(N, 400, rng)
        self.assertAlmostEqual(s.w.sum(), N)
        s = stratified_audit(np.searchsorted(CLASSES, self.pred), 400, rng)
        self.assertAlmostEqual(s.w.sum(), N)
        self.assertEqual(len(s.idx), 400)
        s = stratified_active_audit(np.searchsorted(CLASSES, self.pred), self.acq, 4000, rng)
        self.assertLess(abs(s.w.sum() / N - 1), 0.05)
        s = active_audit(self.acq, 4000, rng)              # Hansen-Hurwitz: E[sum w] = N
        self.assertLess(abs(s.w.sum() / N - 1), 0.05)

    def test_labelfree_blind_to_silent_errors(self):
        sc = soft_confusion(self.proba, CLASSES, self.pred, m=20)
        tau = atc_threshold(self.proba, np.ones(len(self.y), bool) * 0.9 > 0.5)   # source accuracy 1.0 -> tau 0
        a = atc(self.proba, CLASSES, self.pred, tau, m=20)
        # both overestimate the true detection-related macro-F1, because errors are confident
        self.assertGreater(sc["macro_f1"], self.truth["macro_f1"])
        self.assertGreater(a["macro_f1"], self.truth["macro_f1"])


if __name__ == "__main__":
    unittest.main()
