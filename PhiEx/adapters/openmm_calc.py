"""
PhiEx.adapters.openmm_calc — OpenMM Calculator + simulator.

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
    water_model: str = "implicit/obc2.xml"   # ~25% faster than GBn2 on CPU
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
        """Protein-only implicit-solvent MD of the chain holding the site.

        * hydrogens are added with Modeller (AlphaFold and most crystal
          structures have none — the old path failed here every time and
          silently fell back to the stub);
        * cofactors and the docked ligand are NOT simulated (no force-field
          parameters for them in v0); they are held at their input
          coordinates and the trajectory says so
          (`metadata["ligand_simulated"] = False`);
        * frames are mapped back onto the ComplexState atom order by
          (chain, residue number, atom name), so contacts and RMSF work.
        """
        from openmm import app, LangevinMiddleIntegrator, Platform, unit  # type: ignore
        import io as _io
        if not state.protein.residues:
            raise RuntimeError("OpenMM real path needs a structure with residues")
        chain = state.metadata.get("md_chain") or state.protein.residues[0].chain
        pdb_text = _protein_chain_pdb(state, chain)
        topology, positions = None, None
        try:
            # crystal structures often miss side-chain atoms; rebuild them
            # (but do not invent missing loops) before adding hydrogens
            from pdbfixer import PDBFixer  # type: ignore
            fixer = PDBFixer(pdbfile=_io.StringIO(pdb_text))
            fixer.findMissingResidues()
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            n_fix = sum(len(v) for v in fixer.missingAtoms.values())
            fixer.addMissingAtoms()
            if n_fix:
                log("phi", f"pdbfixer: rebuilt {n_fix} missing heavy atoms")
            topology, positions = fixer.topology, fixer.positions
        except ImportError:
            pass
        if topology is None:
            pdb = app.PDBFile(_io.StringIO(pdb_text))
            topology, positions = pdb.topology, pdb.positions
        ff = app.ForceField(self.forcefield, self.water_model)
        modeller = app.Modeller(topology, positions)
        try:
            modeller.addHydrogens(ff, pH=7.0)
            system = ff.createSystem(modeller.topology,
                                     nonbondedMethod=app.CutoffNonPeriodic,
                                     nonbondedCutoff=1.2 * unit.nanometer,
                                     constraints=app.HBonds)
        except Exception as e:
            raise RuntimeError(f"OpenMM could not parametrize chain {chain} "
                               f"(missing heavy atoms or non-standard residues?): {e}")

        step_fs = max(self.step_fs, 2.0)         # HBonds constraints allow 2 fs
        integrator = LangevinMiddleIntegrator(
            self.temperature_k * unit.kelvin, 1.0 / unit.picosecond,
            step_fs * unit.femtosecond)
        integrator.setRandomNumberSeed(self.seed)
        try:
            plat = Platform.getPlatformByName(self.platform_name)
            sim = app.Simulation(modeller.topology, system, integrator, plat)
        except Exception as e:
            if self.platform_name == "CPU":
                raise
            log("sys", f"OpenMM platform {self.platform_name} failed ({e}); using CPU")
            self.platform_name = "CPU"
            integrator = LangevinMiddleIntegrator(
                self.temperature_k * unit.kelvin, 1.0 / unit.picosecond,
                step_fs * unit.femtosecond)
            integrator.setRandomNumberSeed(self.seed)
            sim = app.Simulation(modeller.topology, system, integrator,
                                 Platform.getPlatformByName("CPU"))
        sim.context.setPositions(modeller.positions)
        n_atoms_sys = modeller.topology.getNumAtoms()
        log("phi", f"OpenMM φ→∃: chain {chain}, {n_atoms_sys} atoms with H, "
                   f"implicit solvent (OBC2), platform {self.platform_name}; minimizing…")
        sim.minimizeEnergy(tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
                           maxIterations=100)
        sim.context.setVelocitiesToTemperature(self.temperature_k * unit.kelvin, self.seed)

        # system atom → ComplexState flat index, by (chain, resseq, name)
        flat = list(state.protein.all_atoms())
        key_to_state = {}
        k = 0
        for r in state.protein.residues:
            for a in r.atoms:
                key_to_state[(r.chain, r.index, a.name)] = k
                k += 1
        sys_to_state = {}
        for atom in modeller.topology.atoms():
            key = (atom.residue.chain.id, int(atom.residue.id), atom.name)
            if key in key_to_state:
                sys_to_state[atom.index] = key_to_state[key]
        base = [a.xyz for a in flat] + \
               [a.xyz for c in state.cofactors for a in c.atoms] + \
               [a.xyz for L in state.ligands for a in L.atoms]
        log("phi", f"OpenMM: {len(sys_to_state)}/{len(flat)} protein atoms mapped "
                   f"back to the complex; ligand/cofactors held fixed")

        n_steps = int((self.total_ps * 1000.0) / step_fs)
        report_every = max(1, int(self.report_every_fs / step_fs))
        traj = Trajectory(parent_state=state, label=label or "openmm",
                          metadata={"ligand_simulated": False, "md_chain": chain,
                                    "atoms_with_h": n_atoms_sys})
        for step_block in range(0, n_steps, report_every):
            sim.step(report_every)
            st = sim.context.getState(getEnergy=True, getPositions=True)
            energy = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) \
                     * _KJ_PER_MOL_TO_EV
            pos = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            frame_xyz = list(base)
            for si, ti in sys_to_state.items():
                p = pos[si]
                frame_xyz[ti] = (float(p[0]), float(p[1]), float(p[2]))
            t_ps = (step_block + report_every) * step_fs / 1000.0
            traj.append(Frame(step=step_block + report_every, time_ps=t_ps,
                              energy_ev=energy, positions=tuple(frame_xyz)))
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
        # Take positions from the parent state and add a Gaussian wiggle
        # every report_every_fs.  This is enough to drive the analysis layer
        # (contacts, RMSF, residence) without OpenMM.  Atom order must match
        # what the analysis layer assumes: protein, then cofactors, then
        # ligands — omitting the last two left contacts/residence at zero.
        atoms = list(state.protein.all_atoms())
        for c in state.cofactors:
            atoms.extend(c.atoms)
        for L in state.ligands:
            atoms.extend(L.atoms)
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


def _protein_chain_pdb(state: ComplexState, chain: str) -> str:
    """Standard-residue ATOM records of one chain (no HETATM, no water)."""
    lines, serial = [], 1
    for r in state.protein.residues:
        if r.chain != chain:
            continue
        for a in r.atoms:
            x, y, z = a.xyz
            el = a.element.strip().upper()
            el = el[0] + el[1:].lower() if len(el) > 1 else el
            lines.append(f"ATOM  {serial:>5d} {a.name:<4s} {r.name:<3s} {r.chain:>1s}"
                         f"{r.index:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                         f"          {el:>2s}")
            serial += 1
    lines += ["TER", "END"]
    return "\n".join(lines) + "\n"
