"""
PhiEx.analysis.residence — residence time of a ligand near a target atom.

Heuristic (called out in README): ligand is "near" the target iff its
center-of-heavy-atoms is within `cutoff_a` of the target atom.  Residence
time = fraction of trajectory frames satisfying this, multiplied by
total trajectory time, in picoseconds.
"""

from __future__ import annotations
import math
from typing import Optional

from ..core.trajectory import Trajectory


def residence_time(traj: Trajectory, target_xyz: tuple[float, float, float],
                   cutoff_a: float = 6.0) -> float:
    state = traj.parent_state
    if not traj.frames or not state.ligands:
        return 0.0
    flat_atoms = list(state.protein.all_atoms())
    n_protein = len(flat_atoms)
    cof_n = sum(len(c.atoms) for c in state.cofactors)
    lig_start = n_protein + cof_n
    lig_count = sum(len(L.atoms) for L in state.ligands)
    if lig_count == 0:
        return 0.0
    flat_lig_atoms = []
    for L in state.ligands:
        flat_lig_atoms.extend(L.atoms)
    near_frames = 0
    total = 0
    for frame in traj.frames:
        if not frame.positions: continue
        total += 1
        cx = cy = cz = 0.0
        n = 0
        for k, atom in enumerate(flat_lig_atoms):
            if atom.element.upper() == "H": continue
            i = lig_start + k
            if i >= len(frame.positions): continue
            x, y, z = frame.positions[i]
            cx += x; cy += y; cz += z; n += 1
        if n == 0: continue
        cx /= n; cy /= n; cz /= n
        d = math.sqrt((cx-target_xyz[0])**2 + (cy-target_xyz[1])**2 + (cz-target_xyz[2])**2)
        if d <= cutoff_a:
            near_frames += 1
    if total == 0:
        return 0.0
    last_t = traj.frames[-1].time_ps if traj.frames else 0.0
    return last_t * (near_frames / total)
