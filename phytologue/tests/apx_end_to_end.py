"""
phytologue.tests.apx_end_to_end — end-to-end APX sandbox sanity check.

Runs the full pipeline (in stub mode if heavy dependencies are missing)
and prints a summary that mirrors the README's worked example.

Invoked by `make test`:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python -m phytologue.tests.apx_end_to_end

Exit code 0 if every stage produces a non-empty result; non-zero with a
clear message otherwise.
"""

from __future__ import annotations
import sys

from ..pipeline.apx import run_apx_pipeline
from ..ticker import log


def main() -> int:
    log("i", "─── apx_end_to_end ───")
    result = run_apx_pipeline(md_total_ps=2.0, al_iterations=3)

    failures: list[str] = []
    if result.experimental_pdb_text is None and result.predicted_pdb_text is None:
        failures.append("stage 1: neither experimental nor predicted structure produced")
    if not result.go_aggregate and not result.annotations.get("go_terms"):
        failures.append("stage 2: no GO annotations or Foldseek hits")
    if not result.pockets:
        failures.append("stage 3: no pockets identified")
    if not result.poses:
        failures.append("stage 4: no docking poses generated")
    if result.n_md_frames == 0:
        failures.append("stage 5: zero MD frames")
    if not result.al_history:
        failures.append("stage 6: AL loop produced no iterations")

    print("\n══ APX pipeline summary ══")
    print(f"  experimental: {result.experimental_pdb_id}  text={'yes' if result.experimental_pdb_text else 'no'}")
    print(f"  predicted   : pLDDT {result.mean_plddt or 0:.1f} ± {result.plddt_sd or 0:.1f}")
    print(f"  RMSD pred↔exp: {result.rmsd_pred_vs_exp}")
    print(f"  GO top      : {[g for g, _ in result.go_aggregate[:3]]}")
    print(f"  pockets     : {len(result.pockets)} (active site = {result.active_site_id})")
    print(f"  poses       : {len(result.poses)} (best score = {result.poses[0]['score'] if result.poses else None})")
    print(f"  MD          : {result.n_md_frames} frames over {result.md_total_ps:.1f} ps")
    print(f"  MACE region : {result.mace_region_size} atoms; "
          f"E={result.mace_energy_mean} ± {result.mace_energy_sd}")
    print(f"  residence   : {result.residence_ps_near_heme_fe:.2f} ps near heme Fe")
    print(f"  AL history  : {len(result.al_history)} iterations")
    if result.al_history:
        last = result.al_history[-1]
        print(f"    last chosen: {last['chosen']}")
        print(f"    predicted  : {last['predicted']:.3f} ± {last['predicted_sd']:.3f}")
        print(f"    observed   : {last['observed']:.3f} ± {last['observed_sd']:.3f}")
    print(f"  AL converged residues: {result.al_converged_residues}")
    print(f"  ESM-2 ↔ trajectory agreement: {result.agreement_residues}")
    print(f"  ESM-2 ↔ trajectory discovery flags: {result.disagreement_residues}")
    print(f"  theories registered: {[t['name'] for t in result.theories]}")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nOK — every stage produced output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
