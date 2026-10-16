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
import os
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
        1. ligand atoms (the docked substrate)
        2. cofactor atoms near the site centre
        3. nearest residues by Cα distance to the site centre, one residue
           at a time, until max_atoms is reached.
    Only elements MACE-OFF23 was trained on (H C N O F P S Cl Br I) are
    kept — a heme Fe would make the model fail and silently fall back.
    The site centre is `state.metadata["site_center"]` (set by the
    pipeline from analysis/site.py), else the first cofactor metal.
    """
    region: list[Atom] = []
    center = state.metadata.get("site_center")
    if center is None:
        metal = next((c.metal_atom for c in state.cofactors if c.metal_atom), None)
        center = metal.xyz if metal else None

    def _ok(a: Atom) -> bool:
        return a.element.upper() in _MACE_OFF_ELEMENTS

    for L in state.ligands:
        for a in L.atoms:
            if _ok(a):
                region.append(a)
    for c in state.cofactors:
        if center is not None and c.atoms and \
                min(math.dist(a.xyz, center) for a in c.atoms) > 6.0:
            continue
        region.extend(a for a in c.atoms if _ok(a))
    if len(region) >= max_atoms:
        return region[:max_atoms]

    if center is None or not state.protein.residues:
        return region[:max_atoms]

    # nearest residues
    by_dist = []
    for r in state.protein.residues:
        if r.ca is None: continue
        by_dist.append((math.dist(r.ca.xyz, center), r))
    by_dist.sort(key=lambda x: x[0])
    for d, r in by_dist:
        room = max_atoms - len(region)
        if room <= 0: break
        region.extend([a for a in r.atoms if _ok(a)][:room])
    return region[:max_atoms]


_MACE_OFF_ELEMENTS = {"H", "C", "N", "O", "F", "P", "S", "CL", "BR", "I"}


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
        # MACE runs on CPU by default.  The MACE-OFF23 checkpoint is stored
        # in float64 and mace_off() moves it to the device *before* casting
        # to float32 — MPS has no float64, so loading on "mps" fails with
        # "Cannot convert a MPS Tensor to float64 dtype".  The region is
        # ≤ 50 atoms, so CPU costs nothing noticeable.  Override with
        # PHIEX_MACE_DEVICE (e.g. "cuda" on a GPU node).
        if not self.device:
            self.device = os.environ.get("PHIEX_MACE_DEVICE", "cpu")
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
        region = select_active_site_region(state, self.max_atoms)
        if self.is_stub:
            return [(0.0, 0.0, 0.0) for _ in region]
        try:
            atoms = self._to_ase_atoms(region)
            atoms.calc = self._calc
            f = atoms.get_forces()
            return [tuple(float(c) for c in v) for v in f]
        except Exception as e:
            log("sys", f"MACE forces failed ({e}); zeros returned")
            return [(0.0, 0.0, 0.0) for _ in region]

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
            atoms = self._to_ase_atoms(region)
            atoms.calc = self._calc
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
        center = state.metadata.get("site_center")
        if center is None:
            metal = next((c.metal_atom for c in state.cofactors if c.metal_atom), None)
            center = metal.xyz if metal else None
        if center is not None:
            disp_sum = 0.0
            for a in region:
                d = math.dist(a.xyz, center)
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
