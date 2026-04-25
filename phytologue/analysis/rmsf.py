"""
phytologue.analysis.rmsf — per-residue Cα RMSF over a Trajectory.
"""

from __future__ import annotations
import math

from ..core.trajectory import Trajectory


def rmsf_per_residue(traj: Trajectory) -> dict[int, float]:
    state = traj.parent_state
    if not traj.frames or not state.protein.residues:
        return {}
    # find Cα atom flat indices, in iteration order of all_atoms()
    ca_indices: dict[int, int] = {}
    idx = 0
    for r in state.protein.residues:
        for a in r.atoms:
            if a.name == "CA":
                ca_indices[r.index] = idx
            idx += 1
    if not ca_indices:
        return {}
    # collect positions over frames
    means = {r: [0.0, 0.0, 0.0] for r in ca_indices}
    n_per_res = {r: 0 for r in ca_indices}
    for frame in traj.frames:
        for resid, ai in ca_indices.items():
            if ai >= len(frame.positions): continue
            x, y, z = frame.positions[ai]
            means[resid][0] += x; means[resid][1] += y; means[resid][2] += z
            n_per_res[resid] += 1
    for r in means:
        n = max(1, n_per_res[r])
        means[r] = [c/n for c in means[r]]
    sq = {r: 0.0 for r in ca_indices}
    for frame in traj.frames:
        for resid, ai in ca_indices.items():
            if ai >= len(frame.positions): continue
            x, y, z = frame.positions[ai]
            mx, my, mz = means[resid]
            sq[resid] += (x-mx)**2 + (y-my)**2 + (z-mz)**2
    return {r: math.sqrt(sq[r] / max(1, n_per_res[r])) for r in ca_indices}
