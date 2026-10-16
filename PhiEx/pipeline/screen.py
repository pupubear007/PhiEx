"""
PhiEx.pipeline.screen — dock a list of ligands into one site and rank them.

Input is free text, one ligand per line (or separated by ';'):

    CCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]          a SMILES
    tributyrin                               a name (looked up on PubChem)
    pNP-octanoate: CCCCCCCC(=O)Oc1ccc(cc1)[N+](=O)[O-]   label: SMILES-or-name

Every ligand is docked with the same receptor, box and settings (receptor
preparation is cached), so scores are comparable within the screen.  Rows
report what a bench scientist needs to triage:

    score            Vina best-pose score, kcal/mol (more negative = better)
    efficiency       −score / heavy atoms — corrects Vina's bias toward big
                     molecules; ≥ 0.3 is typically considered good
    site_hits        defining site residues the pose touches (≤ 4 Å) — a pose
                     that scores well but misses the catalytic residues is
                     probably not a productive binding mode
    dist_to_site     closest ligand atom to the site centre, Å
    crystal_rmsd     only when re-docking the co-crystallized ligand

Scores from AutoDock Vina carry roughly ±2 kcal/mol error: use the ranking
and the contacts, not the absolute numbers.
"""

from __future__ import annotations
import math
import re
import time
from typing import Optional

from ..ticker import log
from ..core.state import ComplexState, Ligand, Pocket
from ..adapters.vina import VinaAdapter, resolve_ligand_input

MAX_LIGANDS = 50


def parse_ligand_list(text: str) -> list[tuple[str, str]]:
    """→ [(label, smiles_or_name), …]; blank lines and '#' comments skipped."""
    out = []
    for raw in re.split(r"[\n;]+", text or ""):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, value = "", line
        # "label: value" — but not a SMILES that merely contains ':' (atom maps)
        m = re.match(r"^([^:\[\]]{1,60}):\s*(\S.*)$", line)
        if m:
            label, value = m.group(1).strip(), m.group(2).strip()
        out.append((label, value))
    return out


def _heavy_atoms(smiles: str) -> Optional[int]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol is not None else None
    except ImportError:
        return None


def screen_ligands(state: ComplexState, pocket: Pocket, site,
                   entries: list[tuple[str, str]],
                   exhaustiveness: int = 8) -> dict:
    if len(entries) > MAX_LIGANDS:
        raise ValueError(f"{len(entries)} ligands given; the limit is {MAX_LIGANDS} per screen")
    if not entries:
        raise ValueError("no ligands given")
    vina = VinaAdapter(exhaustiveness=exhaustiveness)
    if vina.is_stub:
        raise RuntimeError(f"cannot screen: {vina.backend} — install vina, meeko and rdkit")
    site_res = set(getattr(site, "residues", ()) or ())
    center = tuple(site.center) if site is not None else tuple(pocket.center)
    chain = getattr(site, "chain", None)
    rows = []
    t0 = time.time()
    log("i", f"── ligand screen: {len(entries)} ligand(s) into {pocket.id} "
             f"({getattr(site, 'label', 'site')}) ──")
    for k, (label, value) in enumerate(entries, start=1):
        row = {"input": value, "name": label or value, "smiles": None,
               "score": None, "efficiency": None, "heavy_atoms": None,
               "contacts": [], "site_hits": [], "dist_to_site": None,
               "crystal_rmsd": None, "error": None}
        try:
            lig = resolve_ligand_input(value, label or "")
            if not label and lig.metadata.get("source") != "pubchem":
                lig.name = value if len(value) <= 24 else f"ligand {k}"
            row["name"], row["smiles"] = lig.name, lig.smiles
            row["heavy_atoms"] = _heavy_atoms(lig.smiles)
            poses = vina.dock(state, lig, pocket)
            best = poses[0]
            row["score"] = round(best.score, 2)
            if row["heavy_atoms"]:
                row["efficiency"] = round(-best.score / row["heavy_atoms"], 3)
            contacts = sorted({r.index for r in state.protein.residues
                               if (chain is None or r.chain == chain)
                               and any(math.dist(a.xyz, l.xyz) <= 4.0
                                       for a in r.atoms for l in best.ligand.atoms)})
            row["contacts"] = contacts
            row["site_hits"] = sorted(site_res & set(contacts))
            row["dist_to_site"] = round(min(math.dist(l.xyz, center)
                                            for l in best.ligand.atoms), 2)
            row["crystal_rmsd"] = best.ligand.metadata.get("crystal_rmsd")
            log("s", f"screen {k}/{len(entries)}: {row['name']}  "
                     f"score {row['score']:.2f}  LE {row['efficiency'] or 0:.2f}  "
                     f"site hits {row['site_hits'] or '—'}")
        except (ValueError, RuntimeError) as e:
            row["error"] = str(e)
            log("sys", f"screen {k}/{len(entries)}: {row['name']} — {e}")
        rows.append(row)
    ok = [r for r in rows if r["error"] is None]
    ok.sort(key=lambda r: r["score"])
    for i, r in enumerate(ok, start=1):
        r["rank"] = i
    failed = [r for r in rows if r["error"] is not None]
    secs = time.time() - t0
    log("i", f"── screen done: {len(ok)} docked, {len(failed)} failed, {secs:.0f} s ──")
    return {"rows": ok + failed, "n_ok": len(ok), "n_failed": len(failed),
            "seconds": round(secs, 1),
            "site": site.to_json() if hasattr(site, "to_json") else None,
            "pocket_id": pocket.id, "exhaustiveness": exhaustiveness}


def rows_to_csv(rows: list[dict]) -> str:
    cols = ["rank", "name", "score", "efficiency", "heavy_atoms", "site_hits",
            "contacts", "dist_to_site", "crystal_rmsd", "smiles", "input", "error"]
    def cell(v):
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        s = "" if v is None else str(v)
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(cell(r.get(c)) for c in cols))
    return "\n".join(lines) + "\n"
