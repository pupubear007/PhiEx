"""
PhiEx.runners.simulator — short-MD payload runner.

Stand-alone shape:

    payload = {
        "protein_pdb": "<PDB text or path>",
        "ligands": [{"name": "ascorbate", "smiles": "...", "atoms": [...]}],
        "cofactors": [{"name": "HEME", "resname": "HEM", "atoms": [...],
                       "metal": [x, y, z]}],
        "total_ps": 10.0,
        "method": "openmm"          # or "openmm-mace"
    }
    result = run_simulator_payload(payload)
"""

from __future__ import annotations
import json
import sys
from typing import Any

from ..core.state import (Atom, Residue, Protein, Cofactor, Ligand,
                          ComplexState)
from ..adapters.openmm_calc import OpenMMSimulator
from ..adapters.pdb import parse_pdb_text
from ..analysis import (ligand_residue_contacts, contact_frequency,
                         rmsf_per_residue, residence_time)


def run_simulator_payload(payload: dict) -> dict:
    pdb_text = payload.get("protein_pdb", "")
    if pdb_text and pdb_text.startswith("/"):
        with open(pdb_text) as fh:
            pdb_text = fh.read()
    protein, cofactors = parse_pdb_text(pdb_text, name=payload.get("protein_name", ""))

    ligs: list[Ligand] = []
    for L in payload.get("ligands", []):
        atoms = tuple(Atom(index=i, name=a.get("name", f"L{i}"),
                           element=a.get("element", "C"),
                           xyz=tuple(a.get("xyz", (0,0,0))))
                      for i, a in enumerate(L.get("atoms", [])))
        ligs.append(Ligand(name=L["name"], smiles=L.get("smiles", ""), atoms=atoms))

    state = ComplexState(protein=protein, cofactors=tuple(cofactors),
                         ligands=tuple(ligs))

    sim = OpenMMSimulator(total_ps=float(payload.get("total_ps", 10.0)))
    traj = sim.run(state, label=payload.get("label", ""))

    contacts = ligand_residue_contacts(traj)
    rmsf = rmsf_per_residue(traj)

    metal_xyz = None
    for c in cofactors:
        if c.metal_atom is not None:
            metal_xyz = c.metal_atom.xyz; break
    res_time = residence_time(traj, metal_xyz, cutoff_a=6.0) if metal_xyz else 0.0

    return {
        "n_frames": len(traj.frames),
        "total_ps": traj.frames[-1].time_ps if traj.frames else 0.0,
        "energies_ev": [f.energy_ev for f in traj.frames],
        "contacts": contacts,
        "top_contacts": contact_frequency(contacts, top_k=10),
        "rmsf": rmsf,
        "residence_ps_near_heme_fe": res_time,
    }


if __name__ == "__main__":   # pragma: no cover
    payload = json.load(sys.stdin)
    out = run_simulator_payload(payload)
    json.dump(out, sys.stdout, indent=2)
