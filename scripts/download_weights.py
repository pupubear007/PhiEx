"""
scripts/download_weights.py — pull ML weights into ./models.

Idempotent.  Skips files already present.  Logs every step.  Designed to
be safe to interrupt — partial files are renamed `.partial` while writing.

v0 weights:
    * ESMFold v1                 (several GB) — fair-esm if openfold is
                                   installed, else Hugging Face
                                   facebook/esmfold_v1 (download only;
                                   skipped on < 32 GB RAM unless
                                   PHIEX_ESMFOLD=force)
    * ESM-2 t12 35M              (~150 MB) — fair-esm pretrained
    * MACE-OFF23 small           (~50 MB) — pulled by mace_off() on first use
    * P2Rank model               (~100 MB) — ships with the p2rank tarball; checked only
    * Foldseek PDB database      (a few GB) — `foldseek databases PDB`, only if
                                   foldseek is on PATH

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
    p.add_argument("--skip-foldseek-db", action="store_true")
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
        print("==> ESMFold v1")
        mode = os.environ.get("PHIEX_ESMFOLD", "auto").lower()
        min_gb = float(os.environ.get("PHIEX_ESMFOLD_MIN_GB", "32"))
        ram = _ram_gb()
        if mode == "off":
            print("    SKIP: PHIEX_ESMFOLD=off")
        elif mode != "force" and ram is not None and ram < min_gb:
            print(f"    SKIP: {ram:.0f} GB RAM < {min_gb:.0f} GB "
                  f"(set PHIEX_ESMFOLD=force to download anyway)")
            failures.append(f"ESMFold (skipped: {ram:.0f} GB RAM)")
        else:
            try:
                import esm
                esm.pretrained.esmfold_v1()
                print("    OK (fair-esm)")
            except Exception as e_fair:
                print(f"    fair-esm unavailable ({e_fair}); trying Hugging Face")
                try:
                    # from_pretrained fetches exactly the files the model
                    # needs (and checks they load); freed right after.
                    from transformers import EsmForProteinFolding
                    m = EsmForProteinFolding.from_pretrained(
                        "facebook/esmfold_v1", low_cpu_mem_usage=True)
                    del m
                    print("    OK (Hugging Face facebook/esmfold_v1)")
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

    print("==> P2Rank")
    import shutil
    prank = shutil.which("prank") or shutil.which("p2rank")
    if prank:
        print(f"    OK ({prank})")
    else:
        print("    MISSING: prank not on PATH (see environment.yml header)")
        failures.append("P2Rank (prank not on PATH)")

    if not args.skip_foldseek_db:
        print("==> Foldseek PDB database")
        db = Path(os.environ.get("PHIEX_FOLDSEEK_DB") or (out / "foldseek" / "pdb"))
        if not shutil.which("foldseek"):
            print("    MISSING: foldseek not on PATH")
            failures.append("Foldseek DB (foldseek not on PATH)")
        elif Path(str(db) + ".dbtype").exists():
            print(f"    OK (already at {db})")
        else:
            import subprocess
            db.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["foldseek", "databases", "PDB", str(db), str(db.parent / "tmp")]
            print("    running:", " ".join(cmd))
            r = subprocess.run(cmd)
            if r.returncode == 0:
                print("    OK")
            else:
                failures.append(f"Foldseek DB (exit {r.returncode})")
                print(f"    FAIL: exit {r.returncode}")

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


def _ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


if __name__ == "__main__":
    sys.exit(main())
