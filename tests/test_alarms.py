"""Checks of the label-free alarm signals on synthetic data."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driftids.alarms import (conformal_rates, disagreement, domain_auc, fires, fit_conformal,  # noqa: E402
                             null_thresholds)

CLASSES = np.array(["Benign", "DoS", "Exploits"])


def probs(y, rng, sharp=0.9):
    """Probabilities concentrated on the true class with some noise."""
    k = np.searchsorted(CLASSES, y)
    p = rng.dirichlet(np.ones(3), size=len(y)) * (1 - sharp)
    p[np.arange(len(y)), k] += sharp
    return p


class TestConformal(unittest.TestCase):
    def test_coverage_on_exchangeable_data(self):
        rng = np.random.default_rng(0)
        y_cal = rng.choice(CLASSES, 4000, p=[.8, .1, .1])
        y_tst = rng.choice(CLASSES, 4000, p=[.8, .1, .1])
        # noisy: 15% of flows get probabilities centred on a wrong class
        def noisy(y):
            p = probs(y, rng)
            flip = rng.random(len(y)) < 0.15
            p[flip] = p[flip][:, rng.permutation(3)]
            return p
        cp = fit_conformal(noisy(y_cal), y_cal, CLASSES, alpha=0.1, mondrian=True)
        pt = noisy(y_tst)
        sets = (1 - pt) <= cp.q[None, :]
        covered = sets[np.arange(len(y_tst)), np.searchsorted(CLASSES, y_tst)]
        for c in CLASSES:                                        # class-conditional coverage close to 0.9
            self.assertGreater(covered[y_tst == c].mean(), 0.86, c)

    def test_abstention_rises_for_unfamiliar_flows(self):
        rng = np.random.default_rng(1)
        y = rng.choice(CLASSES, 3000, p=[.8, .1, .1])
        cp = fit_conformal(probs(y, rng), y, CLASSES, alpha=0.1)
        familiar = conformal_rates(probs(y, rng), cp)["abstention"]
        flat = conformal_rates(rng.dirichlet(np.ones(3), size=3000), cp)["abstention"]   # unsure everywhere
        self.assertGreater(flat, familiar + 0.3)


class TestOtherSignals(unittest.TestCase):
    def test_domain_auc(self):
        rng = np.random.default_rng(2)
        src = rng.normal(size=(2000, 4))
        same = domain_auc(src, rng.normal(size=(2000, 4)), seed=0)[0]
        moved = domain_auc(src, rng.normal(loc=1.0, size=(2000, 4)), seed=0)[0]
        self.assertLess(abs(same - 0.5), 0.05)
        self.assertGreater(moved, 0.8)

    def test_disagreement_and_thresholds(self):
        self.assertAlmostEqual(disagreement(["a", "b", "c", "d"], ["a", "b", "x", "d"]), 0.25)
        rules = null_thresholds({"abst": np.linspace(0, 1, 101), "conf": np.linspace(0, 1, 101)},
                                q=0.95, higher_is_alarm={"conf": False})
        self.assertTrue(fires(0.99, rules["abst"]))
        self.assertFalse(fires(0.5, rules["abst"]))
        self.assertTrue(fires(0.01, rules["conf"]))


if __name__ == "__main__":
    unittest.main()
