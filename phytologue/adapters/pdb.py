"""
phytologue.adapters.pdb — fetch from RCSB and parse into core.state.Protein.

This is the classical (non-ML) ϕ path of stage 1:  retrieve the experimentally
solved structure of the target.  The ML path (ESMFold) is in adapters/esmfold.py.

Only Bio.PDB types live inside this module.  The public functions return
phytologue.core.state types and nothing else.

Stubbing policy:
    If httpx isn't available or the network is offline, fall back to a tiny
    bundled APX-like fake structure so the pipeline can run airgapped.  The
    ticker is logged with [STUB] in that path.
"""

from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional

from ..ticker import log
from ..core.state import Atom, Residue, Protein, Cofactor


# UniProt accession for plant ascorbate peroxidase (pea cytosolic APX1)
APX_DEFAULT_PDB = "1APX"
APX_DEFAULT_UNIPROT = "P48534"


def _fetch_text(url: str, timeout: float = 15.0) -> Optional[str]:
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url, headers={"User-Agent": "phytologue/0.1"})
            if r.status_code == 200:
                return r.text
            log("sys", f"PDB fetch HTTP {r.status_code} for {url}")
            return None
    except Exception as e:
        log("sys", f"PDB fetch failed: {e}")
        return None


def fetch_pdb_text(pdb_id: str, cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Return raw PDB text for `pdb_id` or None if unfetchable.  Caches to disk
    so re-runs are fast and the test target works offline after first run."""
    pdb_id = pdb_id.upper()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{pdb_id}.pdb"
    if cached.exists():
        log("phi", f"PDB {pdb_id}: loaded from cache")
        return cached.read_text()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    text = _fetch_text(url)
    if text:
        cached.write_text(text)
        log("phi", f"PDB {pdb_id}: fetched ({len(text)} bytes) and cached")
        return text
    return None


def fetch_uniprot_annotations(accession: str,
                              cache_dir: str | os.PathLike = "data/uniprot") -> dict:
    """Return a small dict of GO terms, EC number, function summary."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"{accession}.json"
    if cached.exists():
        import json
        log("s", f"UniProt {accession}: loaded from cache")
        return json.loads(cached.read_text())

    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    text = _fetch_text(url)
    if not text:
        log("sys", f"UniProt {accession}: unreachable, returning empty annotations [STUB]")
        return _stub_annotations(accession)
    import json
    data = json.loads(text)
    out = _condense_uniprot(data)
    cached.write_text(json.dumps(out, indent=2))
    log("s", f"UniProt {accession}: {len(out.get('go_terms', []))} GO terms")
    return out


def _condense_uniprot(data: dict) -> dict:
    go_terms = []
    ec = []
    function = ""
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            for prop in ref.get("properties", []):
                if prop.get("key") == "GoTerm":
                    go_terms.append(prop["value"])
    for c in data.get("comments", []):
        if c.get("commentType") == "FUNCTION":
            for tx in c.get("texts", []):
                function = tx.get("value", "")[:400]
    for ecn in data.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", []):
        ec.append(ecn.get("value"))
    return {"go_terms": go_terms[:25], "ec": ec, "function": function}


def _stub_annotations(accession: str) -> dict:
    if accession == APX_DEFAULT_UNIPROT:
        return {
            "go_terms": [
                "F:peroxidase activity",
                "F:heme binding",
                "F:metal ion binding",
                "P:response to oxidative stress",
                "P:hydrogen peroxide catabolic process",
                "C:cytosol",
            ],
            "ec": ["1.11.1.11"],
            "function": "Removes hydrogen peroxide using ascorbate as the "
                        "electron donor.  Ascorbate peroxidase is a heme "
                        "enzyme central to plant ROS scavenging.  [STUB]",
        }
    return {"go_terms": [], "ec": [], "function": "[STUB] no annotations"}


# ────────────────────────────────────────────────────────────────────────
# parsing
# ────────────────────────────────────────────────────────────────────────

# 3-letter to 1-letter
_AA3 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
    "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

# common cofactor residue names we want to keep
_COFACTORS = {"HEM": "HEME", "HEC": "HEME-C", "FAD": "FAD", "NAD": "NAD",
              "NDP": "NADPH", "FMN": "FMN"}


def parse_pdb_text(pdb_text: str, name: str = "") -> tuple[Protein, list[Cofactor]]:
    """Plain-text PDB parser, no Bio.PDB dependency required for v0.

    This is intentionally minimal — adapters are NOT supposed to ship
    parallel parsers.  But Bio.PDB's `Structure` objects are exactly the
    kind of tool-specific type we MUST NOT leak into core.  So the adapter
    either uses Bio.PDB internally (preferred when installed) or this
    fallback parser, and either way returns core.state types.
    """
    try:
        from Bio.PDB import PDBParser  # type: ignore
        return _parse_with_biopython(pdb_text, name)
    except Exception:
        return _parse_naive(pdb_text, name)


def _parse_with_biopython(pdb_text: str, name: str) -> tuple[Protein, list[Cofactor]]:
    from Bio.PDB import PDBParser  # type: ignore
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(name or "X", io.StringIO(pdb_text))

    residues_out: list[Residue] = []
    cofactors_out: list[Cofactor] = []
    seq_chars: list[str] = []
    atom_idx = 0
    for model in structure:
        for chain in model:
            for res in chain:
                hetflag, resseq, icode = res.id
                resname = res.get_resname()
                atoms = []
                for a in res:
                    coord = a.get_coord()
                    atoms.append(Atom(
                        index=atom_idx,
                        name=a.get_name(),
                        element=(a.element or a.get_name()[0]).strip().upper(),
                        xyz=(float(coord[0]), float(coord[1]), float(coord[2])),
                        bfactor=float(a.get_bfactor()),
                    ))
                    atom_idx += 1
                if hetflag == " " and resname in _AA3:
                    residues_out.append(Residue(
                        index=int(resseq), name=resname, chain=chain.id,
                        atoms=tuple(atoms)))
                    seq_chars.append(_AA3[resname])
                elif resname in _COFACTORS:
                    metal = None
                    for a in atoms:
                        if a.element in ("FE", "MG", "ZN", "CU", "MN"):
                            metal = a; break
                    cofactors_out.append(Cofactor(
                        name=_COFACTORS[resname], resname=resname,
                        atoms=tuple(atoms), metal_atom=metal,
                    ))
        break  # first model only
    protein = Protein(
        name=name, sequence="".join(seq_chars),
        residues=tuple(residues_out), source="pdb",
        pdb_text=pdb_text,
    )
    return protein, cofactors_out


def _parse_naive(pdb_text: str, name: str) -> tuple[Protein, list[Cofactor]]:
    """Tiny ATOM/HETATM parser — used when Bio.PDB is missing.  Handles
    the common columns; not robust to exotic records.  Adequate for 1APX
    and similar well-behaved entries."""
    residues: dict[tuple[str, int], list[Atom]] = {}
    res_meta: dict[tuple[str, int], str] = {}
    cof_atoms: dict[tuple[str, int], list[Atom]] = {}
    cof_resname: dict[tuple[str, int], str] = {}
    seq: list[tuple[int, str, str]] = []  # (resseq, chain, 1-letter)
    atom_idx = 0
    for line in pdb_text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21:22].strip() or "A"
            resseq = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            element = line[76:78].strip() or atom_name[0]
            try:
                bfac = float(line[60:66])
            except ValueError:
                bfac = 0.0
        except ValueError:
            continue
        a = Atom(index=atom_idx, name=atom_name, element=element.upper(),
                 xyz=(x, y, z), bfactor=bfac)
        atom_idx += 1
        if line.startswith("ATOM") and resname in _AA3:
            key = (chain, resseq)
            residues.setdefault(key, []).append(a)
            res_meta[key] = resname
            if not any(s[0] == resseq and s[1] == chain for s in seq):
                seq.append((resseq, chain, _AA3[resname]))
        elif line.startswith("HETATM") and resname in _COFACTORS:
            key = (chain, resseq)
            cof_atoms.setdefault(key, []).append(a)
            cof_resname[key] = resname

    residues_out = tuple(
        Residue(index=k[1], name=res_meta[k], chain=k[0], atoms=tuple(v))
        for k, v in sorted(residues.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    )
    seq_chars = "".join(c for _, _, c in sorted(seq))

    cofactors_out: list[Cofactor] = []
    for k, atoms in cof_atoms.items():
        metal = next((a for a in atoms if a.element in ("FE","MG","ZN","CU","MN")), None)
        cofactors_out.append(Cofactor(
            name=_COFACTORS.get(cof_resname[k], cof_resname[k]),
            resname=cof_resname[k],
            atoms=tuple(atoms), metal_atom=metal,
        ))
    return Protein(name=name, sequence=seq_chars, residues=residues_out,
                   source="pdb", pdb_text=pdb_text), cofactors_out


# ────────────────────────────────────────────────────────────────────────
# stubbed minimal APX (kept tiny; only used if there's no network on first run)
# ────────────────────────────────────────────────────────────────────────

_STUB_APX_SEQUENCE = (
    "MGKSYPTVSPDYQKAIEKAKRKLRGFIAEKKCAPLILRLAWHSAGTYDVSSKTGGPFGTIRHQAEL"
    "AHGANNGLDIAVRLLEPIKEQFPILSYADFYQLAGVVAVEVTGGPEVPFHPGREDKPEPPPEGRL"
    "PDATKGSDHLRDVFGKAMGLTDQDIVALSGGHTIGAAHKERSGFEGPWTSNPLIFDNSYFTELLT"
    "GEKEGLLQLPSDKALLTDPVFRPLVEKYAADEDAFFADYAEAHLKLSELGFADA"
)


def stub_apx_protein() -> Protein:
    """Sequence-only APX placeholder for offline first runs.  No structure;
    an ESMFold call is what fills it in.  Logged with [STUB] in callers."""
    return Protein(name="APX", sequence=_STUB_APX_SEQUENCE, source="stub")


def fetch_protein(pdb_id: str = APX_DEFAULT_PDB,
                  uniprot: str = APX_DEFAULT_UNIPROT) -> tuple[Protein, list[Cofactor], dict]:
    """Top-level convenience: fetch+parse PDB and pull UniProt annotations.

    Returns (protein, cofactors, annotations).  Caller is responsible for
    forming a ComplexState and routing further pipeline stages.
    """
    pdb_text = fetch_pdb_text(pdb_id)
    annotations = fetch_uniprot_annotations(uniprot)
    if pdb_text is None:
        log("sys", f"PDB fetch unavailable, using STUB sequence-only APX")
        protein = stub_apx_protein()
        return protein, [], annotations
    protein, cofactors = parse_pdb_text(pdb_text, name=pdb_id)
    log("phi", f"parsed {pdb_id}: {protein.n_residues} residues, "
                f"{len(cofactors)} cofactor(s) "
                f"({', '.join(c.name for c in cofactors) or 'none'})")
    return protein, cofactors, annotations
