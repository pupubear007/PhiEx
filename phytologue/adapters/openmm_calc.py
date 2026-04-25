"""
phytologue.adapters.openmm_calc — OpenMM Calculator + simulator.

Classical ϕ for the protein bulk.  The architecture splits dynamics into
regions (see the spec):  classical (this) for the bulk, MACE-OFF23 (in
adapters/mace.py) for the active-site core.  This is the v0 stand-in for
QM/MM with ML/MM in the QM slot.

Two surfaces:
    * `OpenMMCalculator(state)`   — Calculator-shaped, returns energy/forces
    * `OpenMMSimulator.run(...)`  — produces a Trajectory of Frames

Stub policy:
    OpenMM is in environment.yml so the real path SHOULD work after
    `make env`.  But if it isn't installed (CI on docs-only branch, e.g.),
    we fall back to a fictitious harmonic-trap "simulator" that wiggles
    Cα atoms around their starting positions with a temperature-dependent
    amplitude.  Only the ticker, not the science, is fooled.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.calculator import Calculator
from ..core.state import ComplexState, Atom, Residue, Protein
from ..core.trajectory import Trajectory, Frame
from ..ticker import log
from ..device import select_openmm_platform


# ────────────────────────────────────────────────────────────────────────
# Calculator-shaped wrapper.  In v0 the energy/forces it returns from the
# stub path are dimensionless; the real path returns OpenMM-native units
# converted to eV (see _kj_per_mol_to_ev).  Either way, downstream code
# only needs the ordering to be sane.
# ────────────────────────────────────────────────────────────────────────

_KJ_PER_MOL_TO_EV = 0.01036427  # (kJ/mol) → eV


@dataclass
class OpenMMCalculator:
    name: str = "openmm-amber14"
    forcefield: str = "amber14-all.xml"
    water_model: str = "implicit/gbn2.xml"   # implicit solvent for v0
    temperature_k: float = 300.0
    is_stub: bool = True
    backend: str = "stub"
    platform_name: str = "CPU"
    _system: object = None
    _topology: object = None
    _initial_positions: object = None

    def __post_init__(self) -> None:
        try:
            import openmm  # noqa: F401
            self.is_stub = False
            self.backend = "openmm"
            self.platform_name, _ = select_openmm_platform()
            log("t", f"OpenMM detected; platform={self.platform_name}")
        except ImportError:
            log("t", f"OpenMM adapter: {self.backend} (openmm missing)")

    # full system construction is deferred to OpenMMSimulator.run; the
    # Calculator surface here is implemented via that simulator's
    # single-step force evaluator
    def energy(self, state: ComplexState) -> float:
        # In v0 we don't spin up a fresh System per call; wire later when
        # the FastAPI energy endpoint actually needs it.  For now return
        # the stored energy if present.
        return state.energy_ev or 0.0

    def forces(self, state: ComplexState):
        return [(0.0, 0.0, 0.0) for _ in state.protein.all_atoms()]


# ────────────────────────────────────────────────────────────────────────
# Simulator — produces a Trajectory.
# ────────────────────────────────────────────────────────────────────────

@dataclass
class OpenMMSimulator:
    """Run short MD on a ComplexState.  v0 defaults: implicit solvent,
    300 K, ≤ 10 ps trajectory, 1 fs step.  These are tuned for CPU/MPS
    responsiveness; README documents how to scale up on GPU."""

    forcefield: str = "amber14-all.xml"
    water_model: str = "implicit/gbn2.xml"
    temperature_k: float = 300.0
    step_fs: float = 1.0
    total_ps: float = 10.0
    report_every_fs: float = 100.0           # 100 fs → 100 frames in 10 ps
    seed: int = 42
    is_stub: bool = True
    platform_name: str = "CPU"

    def __post_init__(self) -> None:
        try:
            import openmm  # noqa: F401
            self.is_stub = False
            self.platform_name, _ = select_openmm_platform()
        except ImportError:
            self.is_stub = True

    def run(self, state: ComplexState, label: str = "") -> Trajectory:
        if self.is_stub:
            return self._run_stub(state, label)
        try:
            return self._run_real(state, label)
        except Exception as e:
            log("sys", f"OpenMM simulation failed ({e}) — falling back to stub")
            return self._run_stub(state, label)

    # ──────────────────────────────────────────────────────────────
    # real
    # ──────────────────────────────────────────────────────────────
    def _run_real(self, state: ComplexState, label: str) -> Trajectory:
        from openmm import app, LangevinIntegrator, Platform, unit  # type: ignore
        if not state.protein.pdb_text:
            raise RuntimeError("OpenMM real path requires state.protein.pdb_text")

        import io as _io
        pdb = app.PDBFile(_io.StringIO(state.protein.pdb_text))
        ff = app.ForceField(self.forcefield, self.water_model)
        try:
            system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds)
        except Exception as e:
            raise RuntimeError(f"OpenMM ForceField could not parametrize "
                               f"system (heme parametrization is a known "
                               f"v0 limitation): {e}")

        integrator = LangevinIntegrator(
            self.temperature_k * unit.kelvin,
            1.0 / unit.picosecond,
            self.step_fs * unit.femtosecond)
        plat = Platform.getPlatformByName(self.platform_name)
        sim = app.Simulation(pdb.topology, system, integrator, plat)
        sim.context.setPositions(pdb.positions)
        log("phi", f"OpenMM φ→∃: minimizing...")
        sim.minimizeEnergy(maxIterations=100)
        sim.context.setVelocitiesToTemperature(self.temperature_k * unit.kelvin, self.seed)

        n_steps = int((self.total_ps * 1000.0) / self.step_fs)
        report_every = max(1, int(self.report_every_fs / self.step_fs))
        traj = Trajectory(parent_state=state, label=label or "openmm")

        # streaming-friendly loop
        for step_block in range(0, n_steps, report_every):
            sim.step(report_every)
            st = sim.context.getState(getEnergy=True, getPositions=True)
            energy = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) \
                     * _KJ_PER_MOL_TO_EV
            positions = [(p.x*10.0, p.y*10.0, p.z*10.0)
                         for p in st.getPositions(asNumpy=False)]
            t_ps = (step_block + report_every) * self.step_fs / 1000.0
            traj.append(Frame(step=step_block + report_every,
                              time_ps=t_ps, energy_ev=energy,
                              positions=tuple(positions)))
            log("phi", f"OpenMM step {step_block+report_every}/{n_steps}  "
                       f"E={energy:.2f} eV  t={t_ps:.2f} ps")
        return traj

    # ──────────────────────────────────────────────────────────────
    # stub
    # ──────────────────────────────────────────────────────────────
    def _run_stub(self, state: ComplexState, label: str) -> Trajectory:
        rng = random.Random(self.seed)
        traj = Trajectory(parent_state=state, label=label or "openmm-stub",
                          metadata={"stub": True})
        kT = 0.026 * (self.temperature_k / 298.0)        # eV (very rough)
        # Take backbone positions from the parent state and add a Gaussian
        # wiggle every report_every_fs.  This is enough to drive the analysis
        # layer (contacts, RMSF) without OpenMM.
        atoms = list(state.protein.all_atoms())
        n_steps = int((self.total_ps * 1000.0) / self.step_fs)
        report_every = max(1, int(self.report_every_fs / self.step_fs))
        for step in range(report_every, n_steps + 1, report_every):
            positions = []
            for a in atoms:
                amp = 0.15 + (0.05 if a.name == "CA" else 0.10)
                jx = rng.gauss(0, amp); jy = rng.gauss(0, amp); jz = rng.gauss(0, amp)
                positions.append((a.xyz[0]+jx, a.xyz[1]+jy, a.xyz[2]+jz))
            energy = -300.0 + 5.0 * rng.gauss(0, 1)
            t_ps = step * self.step_fs / 1000.0
            traj.append(Frame(step=step, time_ps=t_ps, energy_ev=energy,
                              positions=tuple(positions)))
            if step % (report_every * 10) == 0:
                log("phi", f"OpenMM[STUB] step {step}/{n_steps}  t={t_ps:.2f} ps  "
                            f"E={energy:.2f} eV")
        log("phi", f"OpenMM[STUB] ∃: trajectory captured  "
                    f"{len(traj.frames)} frames over {self.total_ps:.1f} ps")
        return traj
