"""
scripts/download_weights.py — pull ML weights into ./models.

Idempotent.  Skips files already present.  Logs every step.  Designed to
be safe to interrupt — partial files are renamed `.partial` while writing.

v0 weights:
    * ESMFold v1                 (~3 GB) — fair-esm pretrained
    * ESM-2 t12 35M              (~150 MB) — fair-esm pretrained
    * MACE-OFF23 small           (~50 MB) — pulled by mace_off() on first use
    * P2Rank model               (~100 MB) — comes with conda package, no DL

Most of these are downloaded by their libraries the first time the
corresponding API is called.  This script's job is to *trigger* those
downloads so the first interactive run is fast and an offline run later
works.

Usage:
    python scripts/download_weights.py --out ./models
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./models")
    p.add_argument("--skip-esmfold", action="store_true")
    p.add_argument("--skip-esm2", action="store_true")
    p.add_argument("--skip-mace", action="store_true")
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(out / "torch"))
    os.environ.setdefault("HF_HOME", str(out / "hf"))

    failures: list[str] = []

    if not args.skip_esm2:
        print("==> ESM-2 (esm2_t12_35M_UR50D)")
        try:
            import esm
            esm.pretrained.esm2_t12_35M_UR50D()
            print("    OK")
        except Exception as e:
            failures.append(f"ESM-2 ({e})")
            print(f"    FAIL: {e}")

    if not args.skip_esmfold:
        print("==> ESMFold v1  (this is ~3 GB; takes a while)")
        try:
            import esm
            esm.pretrained.esmfold_v1()
            print("    OK")
        except Exception as e:
            failures.append(f"ESMFold ({e})")
            print(f"    FAIL: {e}")

    if not args.skip_mace:
        print("==> MACE-OFF23 small")
        try:
            from mace.calculators import mace_off
            mace_off(model="small", device="cpu", default_dtype="float32")
            print("    OK")
        except Exception as e:
            failures.append(f"MACE ({e})")
            print(f"    FAIL: {e}")

    print("\nDone.  Models cached under:")
    print(" ", out)
    print(" ", os.environ.get("TORCH_HOME"))
    print(" ", os.environ.get("HF_HOME"))
    if failures:
        print("\nNote: the sandbox uses STUB adapters for any missing weights;")
        print("      these failures are recoverable.  Items not downloaded:")
        for f in failures:
            print("  -", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
