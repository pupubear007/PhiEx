"""
PhiEx.analysis.topology — does a trajectory's atom layout match its state?

Every analysis function maps flat frame positions back to atoms by index,
assuming the order protein → cofactors → ligands of `parent_state`.  A
frame with a different atom count (e.g. the real OpenMM path, whose system
is built from the raw PDB text: waters/hetero atoms included, docked ligand
absent) would be silently mis-indexed, so analysis refuses such frames.
"""

from __future__ import annotations

from ..core.trajectory import Frame, Trajectory


def frame_matches(traj: Trajectory, frame: Frame) -> bool:
    return len(frame.positions) == traj.parent_state.n_atoms()


def frames_match_topology(traj: Trajectory) -> bool:
    """True iff the trajectory has frames and every one of them has exactly
    one position per atom of `parent_state`."""
    return bool(traj.frames) and all(frame_matches(traj, f) for f in traj.frames)
