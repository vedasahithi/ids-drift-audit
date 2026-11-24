import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driftids.attribution import diagnose_drift  # noqa: E402

BASE = dict(audit_unseen_share=0.0, audit_lo=0.9, audit_hi=1.0, auc_threshold=0.6, pred_none=0.95, pred_prior=0.95,
            pred_cov=0.95, conf_err_lo=0.0, conf_ref=0.01, conf_ref_prior=0.01)


class TestDiagnoseDrift(unittest.TestCase):
    def test_types(self):
        self.assertEqual(diagnose_drift(**{**BASE, "audit_unseen_share": 0.05}, dom_auc=0.9, res_auc_audit=0.9), "new_class")
        self.assertEqual(diagnose_drift(**BASE, dom_auc=0.55, res_auc_audit=0.55), "none")
        self.assertEqual(diagnose_drift(**BASE, dom_auc=0.80, res_auc_audit=0.70), "prior")       # rel drop 0.33
        self.assertEqual(diagnose_drift(**BASE, dom_auc=0.80, res_auc_audit=0.79), "covariate")   # rel drop 0.03
        self.assertEqual(diagnose_drift(**{**BASE, "conf_err_lo": 0.05}, dom_auc=0.80, res_auc_audit=0.79), "concept")
        # prior base: confident-error reference doubled (0.02 + margin 0.01), so 0.025 stays prior
        self.assertEqual(diagnose_drift(**{**BASE, "conf_err_lo": 0.025}, dom_auc=0.80, res_auc_audit=0.70), "prior")


if __name__ == "__main__":
    unittest.main()
