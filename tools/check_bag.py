"""Verify a UQ RDM dataset bundle (BagIt) against its manifest-sha1.txt.

Usage: python tools/check_bag.py <folder that contains manifest-sha1.txt>
Prints OK or the mismatching files; exit code 1 on any mismatch.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    manifests = list(root.rglob("manifest-sha1.txt"))
    if not manifests:
        print(f"no manifest-sha1.txt under {root}")
        return 1
    bad = 0
    for man in manifests:
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, rel = line.split(maxsplit=1)
            f = man.parent / rel
            ok = f.exists() and sha1(f) == digest
            print(f"{'OK      ' if ok else 'MISMATCH'} {f}")
            bad += not ok
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
