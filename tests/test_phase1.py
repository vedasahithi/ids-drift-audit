"""Reproducibility of  seeds across processes (Python hash randomisation must not matter)."""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


class TestSeeds(unittest.TestCase):
    def test_seed_same_in_fresh_processes_with_different_hash_seeds(self):
        prog = "import phase1; print(phase1.seed_for('covariate', 0.5, 3, 200, 7))"
        outs = []
        for hs in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hs)
            r = subprocess.run([sys.executable, "-c", prog], cwd=CODE, env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            outs.append(r.stdout.strip().splitlines()[-1])
        self.assertEqual(outs[0], outs[1])


class TestS3Types(unittest.TestCase):
    def test_new_class_skipped_only_without_held_out_class(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import phase1
        from driftids import shifts
        self.assertEqual(phase1.s3_types({"held_out": "DoS"}), shifts.TYPES)
        self.assertEqual(phase1.s3_types({"held_out": None}), ("covariate", "prior", "concept"))


if __name__ == "__main__":
    unittest.main()
