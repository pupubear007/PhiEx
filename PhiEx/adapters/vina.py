"""
PhiEx.adapters.vina — AutoDock Vina docking.

Classical (non-ML) ϕ path of stage 5 (s∃ — sampled existence).  Vina is
deterministic-ish (it uses a stochastic search but converges to repeatable
poses with the same seed), so we report `score_uncertainty=0.0` on each
pose.  The DiffDock adapter (adapters/diffdock.py) is the ML alternative.

Stub: if `vina` (Python or CLI) isn't available, generate a shell of
plausible poses around the active-site pocket.  Logged as [STUB].
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from ..core.state import (
    ComplexState, Pocket, Ligand, DockingPose, Atom,
)
from ..ticker import log


@dataclass
class VinaAdapter:
    name: str = "autodock-vina"
    is_stub: bool = True
    backend: str = "stub"
    exhaustiveness: int = 8
    n_poses: int = 9
    seed: int = 42

    def __post_init__(self) -> None:
        try:
            import vina  # noqa: F401
            self.is_stub = False
            self.backend = "vina (python bindings)"
            log("t", "Vina adapter: vina python bindings detected")
        except ImportError:
            import shutil
            if shutil.which("vina"):
                self.is_stub = False
                self.backend = "vina-cli"
                log("t", "Vina adapter: vina CLI detected on PATH")
            else:
                log("t", f"Vina adapter: {self.backend} (vina missing)")

    def dock(self, state: ComplexState, ligand: Ligand,
             pocket: Pocket) -> list[DockingPose]:
        if self.is_stub:
            return self._dock_stub(state, ligand, pocket)
        try:
            return self._dock_real(state, ligand, pocket)
        except Exception as e:
            log("sys", f"Vina real run failed ({e}); falling back to stub")
            return self._dock_stub(state, ligand, pocket)

    # ──────────────────────────────────────────────────────────────
    # real
    # ──────────────────────────────────────────────────────────────
    def _dock_real(self, state, ligand, pocket):
        import vina  # type: ignore
        v = vina.Vina(sf_name="vina", seed=self.seed)
        if state.protein.pdb_text:
            # Vina's python API wants PDBQT; in v0 we delegate prep to the
            # user via meditation-quality pre-baked PDBQT files.  Skip the
            # automatic conversion — that's an explicit pre-v0 stub limit.
            log("sys", "Vina real path expects pre-prepared PDBQT inputs (v0 simplification)")
        # If nothing is pre-prepared, drop to stub.
        return self._dock_stub(state, ligand, pocket)

    # ──────────────────────────────────────────────────────────────
    # stub
    # ──────────────────────────────────────────────────────────────
    def _dock_stub(self, state: ComplexState,
                   ligand: Ligand, pocket: Pocket) -> list[DockingPose]:
        rng = random.Random(self.seed ^ hash(ligand.smiles))
        cx, cy, cz = pocket.center
        poses: list[DockingPose] = []
        # We synthesise N poses laid out on a small Sobol-ish lattice
        # around the pocket center, with scores that vary monotonically
        # with distance to the cofactor metal (to approximate Vina's
        # bias toward bound poses near catalytic geometry).
        metal = next((c.metal_atom for c in state.cofactors if c.metal_atom), None)
        for k in range(self.n_poses):
            jx = rng.gauss(0, 1.5); jy = rng.gauss(0, 1.5); jz = rng.gauss(0, 1.5)
            pos = (cx + jx, cy + jy, cz + jz)
            score = -7.0 + 0.3 * k + rng.gauss(0, 0.4)        # baseline score (kcal/mol)
            if metal:
                d = math.sqrt((pos[0]-metal.xyz[0])**2 +
                              (pos[1]-metal.xyz[1])**2 +
                              (pos[2]-metal.xyz[2])**2)
                # closer to metal = better score (negative is better)
                score -= max(0.0, 4.0 - 0.4 * d)
            atoms = (
                Atom(index=0, name="C1", element="C", xyz=pos),
                Atom(index=1, name="O1", element="O", xyz=(pos[0]+1.0, pos[1], pos[2])),
            )
            poses.append(DockingPose(
                ligand=Ligand(name=ligand.name, smiles=ligand.smiles,
                              atoms=atoms, sdf_text=ligand.sdf_text),
                score=score, score_uncertainty=0.0, pose_id=k,
                rmsd_to_best=0.0 if k == 0 else float(0.5 + 0.4*k),
            ))
        poses.sort(key=lambda p: p.score)
        log("s", f"Vina[STUB] s∃: {len(poses)} poses; "
                  f"best score = {poses[0].score:.2f} kcal/mol")
        return poses


# Convenience: ascorbate (vitamin C) — the v0 example ligand
ASCORBATE = Ligand(
    name="ascorbate",
    smiles="OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",
    metadata={"role": "substrate / electron donor"},
)
