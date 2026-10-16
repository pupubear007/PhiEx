"""
PhiEx.analysis.contacts — contact frequency between ligand and residues.

A residue is "in contact" with the ligand at frame f iff any heavy atom
of the residue is within `cutoff_a` (default 4.0 Å) of any heavy atom of
the ligand.  Contact frequency = fraction of frames in which this is true.

Output:  dict { residue_index: frequency_in_[0,1] }.
"""

from __future__ import annotations
import math

from ..core.trajectory import Trajectory
from .topology import frame_matches


def ligand_residue_contacts(traj: Trajectory, cutoff_a: float = 4.0,
                            chain: "str | None" = None) -> dict[int, float]:
    """Return per-residue contact-frequency over the trajectory, for one
    chain (default: the chain the MD simulated — homo-oligomers repeat
    residue numbers)."""
    state = traj.parent_state
    chain = chain if chain is not None else traj.metadata.get("md_chain")
    if not traj.frames or not state.ligands:
        return {}

    # build per-residue heavy-atom index lists from frame 0 ordering.
    res_atoms: dict[int, list[int]] = {}
    idx = 0
    for r in state.protein.residues:
        for a in r.atoms:
            if a.element.upper() != "H" and (chain is None or r.chain == chain):
                res_atoms.setdefault(r.index, []).append(idx)
            idx += 1
    n_protein_atoms = idx
    # cofactor atoms come next (we won't index them here)
    for c in state.cofactors:
        idx += len(c.atoms)
    cof_offset = n_protein_atoms
    cof_end = idx
    # ligand atoms last
    lig_atoms_idx = list(range(cof_end, cof_end + sum(len(L.atoms) for L in state.ligands)))

    counts: dict[int, int] = {}
    n_frames = 0
    for frame in traj.frames:
        if not frame.positions or not frame_matches(traj, frame):
            continue
        n_frames += 1
        # ligand atoms positions (heavy atoms only)
        lig_positions = []
        flat_lig = []
        for L in state.ligands:
            flat_lig.extend(L.atoms)
        for k, a in zip(lig_atoms_idx, flat_lig):
            if k >= len(frame.positions): continue
            if a.element.upper() == "H": continue
            lig_positions.append(frame.positions[k])

        for resid, atom_indices in res_atoms.items():
            in_contact = False
            for ai in atom_indices:
                if ai >= len(frame.positions): continue
                ax, ay, az = frame.positions[ai]
                for lx, ly, lz in lig_positions:
                    dx, dy, dz = ax-lx, ay-ly, az-lz
                    if dx*dx + dy*dy + dz*dz <= cutoff_a * cutoff_a:
                        in_contact = True; break
                if in_contact: break
            if in_contact:
                counts[resid] = counts.get(resid, 0) + 1

    if n_frames == 0:
        return {}
    return {r: counts.get(r, 0) / n_frames for r in res_atoms}


def contact_frequency(map_: dict[int, float], top_k: int = 10) -> list[tuple[int, float]]:
    """Top-k residues by contact frequency."""
    return sorted(map_.items(), key=lambda kv: -kv[1])[:top_k]
