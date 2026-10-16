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

    box_size: float = 22.0           # Å, cubic search box around the site
    _why_stub: str = ""

    def __post_init__(self) -> None:
        missing = []
        for mod in ("vina", "meeko", "rdkit"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if missing:
            self.is_stub = True
            self._why_stub = f"{', '.join(missing)} not installed"
            self.backend = f"stub ({self._why_stub})"
            log("t", f"Vina adapter: {self.backend}")
        else:
            self.is_stub = False
            self.backend = "vina 1.2 + meeko (python)"
            log("t", f"Vina adapter: {self.backend}")

    def dock(self, state: ComplexState, ligand: Ligand,
             pocket: Pocket) -> list[DockingPose]:
        if self.is_stub:
            return self._dock_stub(state, ligand, pocket)
        try:
            return self._dock_real(state, ligand, pocket)
        except Exception as e:
            # Do NOT quietly substitute synthetic poses: a fake score next
            # to a real ligand name reads as a result.  Fail loudly.
            log("sys", f"Vina docking failed ({type(e).__name__}: {e})")
            raise RuntimeError(f"docking failed: {e}") from e

    # ──────────────────────────────────────────────────────────────
    # real: meeko receptor + ligand prep, Vina search in a box on the site
    # ──────────────────────────────────────────────────────────────
    def _dock_real(self, state: ComplexState, ligand: Ligand,
                   pocket: Pocket) -> list[DockingPose]:
        from vina import Vina  # type: ignore
        import tempfile, os
        center = [float(c) for c in pocket.center]
        rec = prepare_receptor_pdbqt(state, center)
        lig_pdbqt = prepare_ligand_pdbqt(ligand.smiles, seed=self.seed)
        log("phi", f"Vina: docking {ligand.name} into {pocket.id} "
                   f"(box {self.box_size:.0f} Å at {tuple(round(c,1) for c in center)}, "
                   f"exhaustiveness {self.exhaustiveness})")
        with tempfile.TemporaryDirectory() as td:
            rec_path = os.path.join(td, "receptor.pdbqt")
            with open(rec_path, "w") as fh:
                fh.write(rec)
            v = Vina(sf_name="vina", seed=self.seed, verbosity=0)
            v.set_receptor(rigid_pdbqt_filename=rec_path)
            v.set_ligand_from_string(lig_pdbqt)
            v.compute_vina_maps(center=center, box_size=[self.box_size]*3)
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
            energies = v.energies(n_poses=self.n_poses)
            poses_txt = v.poses(n_poses=self.n_poses)
        models = _split_pdbqt_models(poses_txt)
        poses: list[DockingPose] = []
        first_xyz = None
        for k, (model, e) in enumerate(zip(models, energies)):
            atoms = tuple(Atom(index=i, name=n, element=el, xyz=xyz)
                          for i, (n, el, xyz) in enumerate(model))
            if first_xyz is None:
                first_xyz = [a.xyz for a in atoms]
            rmsd = _rmsd_same_order([a.xyz for a in atoms], first_xyz)
            poses.append(DockingPose(
                ligand=Ligand(name=ligand.name, smiles=ligand.smiles, atoms=atoms,
                              metadata={**ligand.metadata, "pdbqt_model": k}),
                score=float(e[0]), score_uncertainty=0.0, pose_id=k,
                rmsd_to_best=rmsd))
        if not poses:
            raise RuntimeError("Vina returned no poses")
        log("s", f"Vina s∃: {len(poses)} poses; best = {poses[0].score:.2f} kcal/mol "
                 f"(range {poses[0].score:.2f} … {poses[-1].score:.2f})")
        return poses

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
        if self._why_stub:
            log("sys", f"Vina[STUB]: {self._why_stub} — poses below are synthetic")
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
                              atoms=atoms, sdf_text=ligand.sdf_text,
                              metadata={**ligand.metadata, "stub": True}),
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


# p-nitrophenyl acetate — the standard chromogenic esterase/lipase assay
# substrate; the default ligand when the site is a serine/cysteine hydrolase
PNP_ACETATE = Ligand(
    name="p-nitrophenyl acetate",
    smiles="CC(=O)Oc1ccc(cc1)[N+](=O)[O-]",
    metadata={"role": "model ester substrate (esterase assay)"},
)


def default_ligand_for_site(site) -> "Ligand | None":
    """Default substrate when the user gives no SMILES — only where the
    site itself says what a reasonable probe is:
        serine/cysteine hydrolase → p-nitrophenyl acetate (esterase assay)
        heme site                 → ascorbate (the APX reducing substrate)
    Anything else returns None: docking an arbitrary molecule into an
    unknown site produces numbers that look meaningful and are not."""
    kind = getattr(site, "kind", "") or ""
    if kind in ("serine-hydrolase", "cysteine-hydrolase"):
        return PNP_ACETATE
    if (getattr(site, "cofactor", "") or "").upper().startswith("HEME"):
        return ASCORBATE
    return None


# ────────────────────────────────────────────────────────────────────────
# preparation helpers (meeko + RDKit)
# ────────────────────────────────────────────────────────────────────────

_AD_COFACTOR_TYPES = {"C": "C", "N": "N", "O": "OA", "S": "SA", "P": "P",
                      "FE": "Fe", "ZN": "Zn", "MG": "Mg", "MN": "Mn",
                      "CA": "Ca", "CU": "Cu"}


def _near_chains(state: ComplexState, center, radius: float = 20.0) -> set:
    chains = set()
    for r in state.protein.residues:
        for a in r.atoms:
            if math.dist(a.xyz, center) <= radius:
                chains.add(r.chain); break
    return chains


def prepare_receptor_pdbqt(state: ComplexState, center) -> str:
    """Protein → PDBQT with meeko (adds polar H from residue templates).
    Only chains with atoms within 20 Å of the site are kept.  Real
    cofactors near the site (e.g. heme) are appended as rigid receptor
    atoms with AutoDock types, because meeko has no template for them and
    docking into a heme pocket without the heme would be meaningless."""
    from meeko import Polymer, ResidueChemTemplates, MoleculePreparation, PDBQTWriterLegacy
    keep = _near_chains(state, center)
    lines = []
    serial = 1
    for r in state.protein.residues:
        if keep and r.chain not in keep:
            continue
        for a in r.atoms:
            x, y, z = a.xyz
            lines.append(f"ATOM  {serial:>5d} {a.name:<4s} {r.name:<3s} {r.chain:>1s}"
                         f"{r.index:>4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                         f"          {a.element:>2s}")
            serial += 1
    lines.append("END")
    templates = ResidueChemTemplates.create_from_defaults()
    pol = Polymer.from_pdb_string("\n".join(lines) + "\n", templates,
                                  MoleculePreparation(), allow_bad_res=True)
    out = PDBQTWriterLegacy.write_from_polymer(pol)
    rigid = out[0] if isinstance(out, tuple) else out
    extra = []
    for c in state.cofactors:
        if c.metadata.get("stub") or not c.atoms:
            continue
        if min(math.dist(a.xyz, center) for a in c.atoms) > 12.0:
            continue
        for a in c.atoms:
            t = _AD_COFACTOR_TYPES.get(a.element.upper())
            if t is None:
                continue
            x, y, z = a.xyz
            extra.append(f"HETATM{serial:>5d} {a.name[:4]:<4s} {c.resname[:3]:<3s} X 999"
                         f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 {t:<2s}")
            serial += 1
    body = [l for l in rigid.splitlines() if l.strip() and not l.startswith("END")]
    return "\n".join(body + extra) + "\n"


def prepare_ligand_pdbqt(smiles: str, seed: int = 42) -> str:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"could not embed 3D coordinates for {smiles!r}")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError(f"meeko ligand prep failed: {err}")
    return pdbqt


def _split_pdbqt_models(text: str):
    """Vina pose PDBQT → [[(atom_name, element, xyz), ...] per model],
    heavy atoms only (Vina's H are united-atom bookkeeping)."""
    models, cur = [], None
    for l in text.splitlines():
        if l.startswith("MODEL"):
            cur = []
        elif l.startswith("ENDMDL"):
            if cur is not None:
                models.append(cur)
            cur = None
        elif l.startswith(("ATOM", "HETATM")) and cur is not None:
            adt = l[77:79].strip()
            if adt in ("H", "HD", "HS"):
                continue
            el = {"A": "C", "OA": "O", "NA": "N", "SA": "S", "HD": "H"}.get(adt, adt[:1] or "C")
            cur.append((l[12:16].strip(), el,
                        (float(l[30:38]), float(l[38:46]), float(l[46:54]))))
    if cur:
        models.append(cur)
    return models


def _rmsd_same_order(a, b) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return math.sqrt(sum(math.dist(p, q) ** 2 for p, q in zip(a, b)) / len(a))


# ────────────────────────────────────────────────────────────────────────
# ligand input: SMILES or a compound name
# ────────────────────────────────────────────────────────────────────────

_NAME_CACHE: dict = {}


def resolve_ligand_input(text: str, name: str = "") -> Ligand:
    """Turn what the user typed into a Ligand.

    * a valid SMILES (checked with RDKit) is used as-is;
    * otherwise the text is treated as a compound name and looked up on
      PubChem (e.g. "p-nitrophenyl butyrate", "tributyrin", "ascorbic acid");
    * otherwise ValueError with a message the UI can show.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("empty ligand")
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(text)
    except ImportError:
        mol = None
    if mol is not None and mol.GetNumAtoms() > 0:
        return Ligand(name=name or "custom ligand", smiles=text)
    smiles = _pubchem_smiles(text)
    if smiles:
        log("s", f"ligand: '{text}' resolved on PubChem → {smiles}")
        return Ligand(name=name or text, smiles=smiles,
                      metadata={"source": "pubchem", "query": text})
    raise ValueError(f"'{text}' is not a valid SMILES and was not found on PubChem "
                     f"by name — enter a SMILES (e.g. CCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]) "
                     f"or a compound name (e.g. p-nitrophenyl butyrate)")


def _pubchem_smiles(name: str) -> "str | None":
    key = name.lower()
    if key in _NAME_CACHE:
        return _NAME_CACHE[key]
    try:
        import httpx
        from urllib.parse import quote
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               f"{quote(name)}/property/SMILES,IsomericSMILES,CanonicalSMILES/JSON")
        r = httpx.get(url, timeout=10.0, headers={"User-Agent": "PhiEx/0.1"})
        if r.status_code != 200:
            _NAME_CACHE[key] = None
            return None
        props = (r.json().get("PropertyTable", {}).get("Properties") or [{}])[0]
        smi = (props.get("SMILES") or props.get("IsomericSMILES")
               or props.get("CanonicalSMILES"))
        _NAME_CACHE[key] = smi
        return smi
    except Exception as e:
        log("sys", f"PubChem lookup for '{name}' failed ({e})")
        return None
