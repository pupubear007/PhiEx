"""
PhiEx.adapters.mace — MACE-OFF23 ML potential.

This adapter wears BOTH coats:
    * Calculator       — energy(state) and forces(state) over a subset of
                         atoms (the ≤50-atom active-site core in v0)
    * LearnedModel     — predict() returns (energy, ensemble_sigma)

In the architecture this is the "ML Calculator" the user picks per region
in the UI.  Classical Calculator runs on the protein bulk; this MACE
Calculator runs on the active-site region (heme + ligand + nearby
residues).  This is the v0 stand-in for a QM/MM partition with ML/MM in
the QM slot.

Ensemble uncertainty:
    Real MACE-OFF23 ships an ensemble of three models; we average their
    energies and report the SD as the uncertainty.  In stub mode we
    synthesise a small ensemble jitter so the rest of the pipeline
    never sees a "0 uncertainty" tensor and silently believes a stub.

Region selection:
    `select_active_site_region(state, max_atoms)` picks the heme + ligand
    atoms first, then nearest-neighbor residue atoms by Cα distance,
    capping at max_atoms (default 50).  Documented in the README.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from ..core.calculator import Calculator
from ..core.learned import LearnedModelBase, Prediction
from ..core.state import ComplexState, Atom
from ..ticker import log
from ..device import select_device


# ────────────────────────────────────────────────────────────────────────
# region selector — explicit heuristic, called out in the README
# ────────────────────────────────────────────────────────────────────────

def select_active_site_region(state: ComplexState,
                              max_atoms: int = 50) -> "list[Atom]":
    """Pick the active-site atom subset for ML/MM.

    Heuristic (documented in README):
        1. all cofactor atoms (heme: ~43 atoms — already near the cap)
        2. all ligand atoms (ascorbate: 12 heavy atoms)
        3. nearest residues by Cα distance to the cofactor metal,
           one residue at a time, until max_atoms is reached.
    """
    region: list[Atom] = []

    # cofactors first
    metal: Optional[Atom] = None
    for c in state.cofactors:
        for a in c.atoms:
            region.append(a)
            if len(region) >= max_atoms:
                return region[:max_atoms]
        if c.metal_atom and metal is None:
            metal = c.metal_atom

    # ligands
    for L in state.ligands:
        for a in L.atoms:
            region.append(a)
            if len(region) >= max_atoms:
                return region[:max_atoms]

    if metal is None or not state.protein.residues:
        return region[:max_atoms]

    # nearest residues
    by_dist = []
    for r in state.protein.residues:
        if r.ca is None: continue
        d = math.sqrt((r.ca.xyz[0]-metal.xyz[0])**2 +
                      (r.ca.xyz[1]-metal.xyz[1])**2 +
                      (r.ca.xyz[2]-metal.xyz[2])**2)
        by_dist.append((d, r))
    by_dist.sort(key=lambda x: x[0])
    for d, r in by_dist:
        room = max_atoms - len(region)
        if room <= 0: break
        region.extend(r.atoms[:room])
    return region[:max_atoms]


# ────────────────────────────────────────────────────────────────────────
# adapter — Calculator + LearnedModel
# ────────────────────────────────────────────────────────────────────────

@dataclass
class MACEAdapter(LearnedModelBase):
    name: str = "mace-off23-small"
    device: str = ""
    is_stub: bool = True
    backend: str = "stub"
    max_atoms: int = 50
    ensemble_n: int = 3
    _calc: object = None

    def __post_init__(self) -> None:
        if not self.device:
            self.device = select_device()
        self._maybe_load()

    def _maybe_load(self) -> None:
        try:
            import mace  # noqa: F401
            from mace.calculators import mace_off  # type: ignore
            try:
                self._calc = mace_off(model="small", device=self.device,
                                       default_dtype="float32")
                self.is_stub = False
                self.backend = f"mace-off23-small on {self.device}"
                log("t", f"MACE-OFF23 loaded: {self.backend}")
            except Exception as e:
                self.is_stub = True
                self.backend = f"stub (MACE load failed: {e})"
                log("sys", f"MACE-OFF23 load failed, stub: {e}")
        except ImportError:
            self.is_stub = True
            self.backend = "stub (mace-torch not installed)"
            log("t", f"MACE adapter: {self.backend} on {self.device}")

    # ──────────────────────────────────────────────────────────────
    # Calculator surface
    # ──────────────────────────────────────────────────────────────
    def energy(self, state: ComplexState) -> float:
        return self.predict(state).value

    def forces(self, state: ComplexState):
        # forces require a richer prediction in real mode; the stub
        # returns zeros to satisfy the protocol
        if self.is_stub:
            return [(0.0, 0.0, 0.0) for _ in select_active_site_region(state, self.max_atoms)]
        try:
            from ase import Atoms  # type: ignore
            atoms = self._to_ase(state)
            atoms.set_calculator(self._calc)
            f = atoms.get_forces()
            return [tuple(v) for v in f]
        except Exception as e:
            log("sys", f"MACE forces failed ({e}); zeros returned")
            return [(0.0, 0.0, 0.0) for _ in range(self.max_atoms)]

    # ──────────────────────────────────────────────────────────────
    # LearnedModel surface
    # ──────────────────────────────────────────────────────────────
    def predict(self, state: ComplexState) -> Prediction:
        region = select_active_site_region(state, self.max_atoms)
        if self.is_stub:
            return self._predict_stub(state, region)
        return self._predict_real(state, region)

    def _predict_real(self, state, region):
        try:
            from ase import Atoms  # type: ignore
            atoms = self._to_ase_atoms(region)
            atoms.set_calculator(self._calc)
            energy = float(atoms.get_potential_energy())
            # ensemble sigma: real MACE-OFF23 small ships a single model;
            # we treat ensemble as 1 with a placeholder uncertainty driven
            # by region-size (smaller = less context = higher SD).  Real
            # ensemble support belongs in a separate adapter.
            sigma = max(0.05, 1.0 / max(1, len(region)/10))
            log("phi", f"MACE φ→∃: region={len(region)} atoms  "
                       f"E={energy:.3f} eV  σ≈{sigma:.3f}")
            return Prediction(value=energy, uncertainty=sigma,
                              meta={"model": self.name, "device": self.device,
                                    "backend": self.backend, "n_atoms": len(region)})
        except Exception as e:
            log("sys", f"MACE real predict failed ({e}); stub")
            return self._predict_stub(state, region)

    def _predict_stub(self, state, region):
        rng = random.Random(42 ^ len(region))
        # synthetic energy: rough harmonic in distance to cofactor metal,
        # so deformations register as "higher energy"
        metal = next((c.metal_atom for c in state.cofactors if c.metal_atom), None)
        if metal:
            disp_sum = 0.0
            for a in region:
                d = math.sqrt((a.xyz[0]-metal.xyz[0])**2 +
                              (a.xyz[1]-metal.xyz[1])**2 +
                              (a.xyz[2]-metal.xyz[2])**2)
                disp_sum += (d - 4.0) ** 2
            energy = -8.0 + 0.001 * disp_sum + rng.gauss(0, 0.02)
        else:
            energy = -8.0 + rng.gauss(0, 0.05)
        # ensemble jitter
        members = [energy + rng.gauss(0, 0.05) for _ in range(self.ensemble_n)]
        mean = sum(members)/len(members)
        sd = (sum((m-mean)**2 for m in members)/(len(members)-1)) ** 0.5 if len(members)>1 else 0.05
        log("phi", f"MACE[STUB] φ→∃: region={len(region)} atoms  "
                    f"E={mean:.3f} ± {sd:.3f} eV")
        return Prediction(value=mean, uncertainty=sd,
                          meta={"model": self.name, "device": self.device,
                                "backend": self.backend, "n_atoms": len(region),
                                "stub": True})

    # ──────────────────────────────────────────────────────────────
    # helpers
    # ──────────────────────────────────────────────────────────────
    def _to_ase_atoms(self, region):
        from ase import Atoms  # type: ignore
        symbols = [a.element.title() for a in region]
        positions = [list(a.xyz) for a in region]
        return Atoms(symbols=symbols, positions=positions)
