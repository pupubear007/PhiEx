"""
PhiEx.core.trajectory — s∃, a sampled run.

Same idea as engine.py's Trajectory, lifted to ComplexState.  A Trajectory
is a list of Frames each holding a snapshot of positions/velocities and
the energy at that step.  Adapters (OpenMM, MACE-OFF23) emit Frames; the
analysis layer (contacts, RMSF, residence time) consumes them.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .state import ComplexState


@dataclass
class Frame:
    step: int
    time_ps: float
    energy_ev: Optional[float] = None
    # Positions are kept in a flat list of (x,y,z) tuples in Å.  Order
    # matches the iteration order of ComplexState.all_atoms() at the time
    # the trajectory was captured.  Velocities optional.
    positions: tuple[tuple[float, float, float], ...] = ()
    velocities: Optional[tuple[tuple[float, float, float], ...]] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """A single ϕ → ∃ run, or s∃ in framework terms.

    The `parent_state` snapshot at frame 0 carries topology so analysis
    can map flat position arrays back to (residue, atom-name).
    """
    parent_state: ComplexState
    frames: list[Frame] = field(default_factory=list)
    label: str = ""              # e.g. "wild-type 300K"
    metadata: dict = field(default_factory=dict)

    def append(self, frame: Frame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def positions_iter(self) -> Iterable[tuple[tuple[float, float, float], ...]]:
        for f in self.frames:
            yield f.positions

    def energies(self) -> list[float]:
        return [f.energy_ev for f in self.frames if f.energy_ev is not None]
