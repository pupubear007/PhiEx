"""
PhiEx.runners.docker — docking payload runner.

Stand-alone shape:

    payload = {
        "ligand": {"name": "ascorbate", "smiles": "..."},
        "pocket": {"id": "P1", "center": [x,y,z], "score": ...,
                   "is_active_site": true},
        "protein_pdb": "<PDB text or file path>",
        "method": "vina"      # or "diffdock"
    }
    result = run_docking_payload(payload)
"""

from __future__ import annotations
import json
import sys
from typing import Any

from ..core.state import (Atom, Residue, Protein, Pocket, Ligand,
                          ComplexState, Cofactor, DockingPose)
from ..adapters.vina import VinaAdapter
from ..adapters.diffdock import DiffDockAdapter
from ..adapters.pdb import parse_pdb_text
from ..ticker import log


def run_docking_payload(payload: dict) -> dict:
    method = payload.get("method", "vina").lower()
    ligand = Ligand(name=payload["ligand"]["name"],
                    smiles=payload["ligand"]["smiles"])
    pocket = Pocket(id=payload["pocket"]["id"],
                    center=tuple(payload["pocket"]["center"]),
                    score=float(payload["pocket"].get("score", 0.0)),
                    is_active_site=bool(payload["pocket"].get("is_active_site", False)))

    pdb_text = payload.get("protein_pdb", "")
    if pdb_text and pdb_text.startswith("/"):
        with open(pdb_text) as fh:
            pdb_text = fh.read()
    if pdb_text:
        protein, cofactors = parse_pdb_text(pdb_text, name=payload.get("protein_name", ""))
    else:
        protein, cofactors = Protein(name="?"), []
    state = ComplexState(protein=protein, cofactors=tuple(cofactors), ligands=(ligand,))

    if method == "diffdock":
        pred = DiffDockAdapter().predict(state, ligand, pocket)
        poses = pred.value
    else:
        poses = VinaAdapter().dock(state, ligand, pocket)

    return {"method": method,
            "n_poses": len(poses),
            "poses": [_pose_to_json(p) for p in poses]}


def _pose_to_json(p: DockingPose) -> dict:
    return {
        "pose_id": p.pose_id,
        "score": p.score,
        "score_uncertainty": p.score_uncertainty,
        "rmsd_to_best": p.rmsd_to_best,
        "ligand": {"name": p.ligand.name, "smiles": p.ligand.smiles},
        "atoms": [{"name": a.name, "element": a.element, "xyz": list(a.xyz)}
                  for a in p.ligand.atoms],
    }


# ────────────────────────────────────────────────────────────────────────
# stdin/stdout entrypoint for Slurm
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":   # pragma: no cover
    payload = json.load(sys.stdin)
    out = run_docking_payload(payload)
    json.dump(out, sys.stdout, indent=2)
