"""
PhiEx.adapters.pdb — fetch from RCSB and parse into core.state.Protein.

This is the classical (non-ML) ϕ path of stage 1:  retrieve the experimentally
solved structure of the target.  The ML path (ESMFold) is in adapters/esmfold.py.

Only Bio.PDB types live inside this module.  The public functions return
PhiEx.core.state types and nothing else.

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
            r = c.get(url, headers={"User-Agent": "PhiEx/0.1"})
            if r.status_code == 200:
                return r.text
            log("sys", f"PDB fetch HTTP {r.status_code} for {url}")
            return None
    except Exception as e:
        log("sys", f"PDB fetch failed: {e}")
        return None


def list_pdbs_for_uniprot(uniprot_id: str, timeout: float = 10.0) -> list[dict]:
    """Look up every PDB entry cross-referenced from a UniProt accession.

    Returns a list of dicts like:
        [{"pdb_id": "2CYP", "method": "X-ray", "resolution": 1.7,
          "chains": "A=1-294", "raw": <full xref dict>}, ...]

    Sorted by resolution (best first), with non-X-ray entries last.  Empty
    list means no experimental structure is mapped to this UniProt ID —
    you can still run ESMFold-only and get a predicted structure.
    """
    try:
        import httpx, json as _json
    except ImportError:
        return []
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url, headers={"User-Agent": "PhiEx/0.1",
                                     "Accept": "application/json"})
            if r.status_code != 200:
                log("sys", f"UniProt lookup HTTP {r.status_code} for {uniprot_id}")
                return []
            data = r.json()
    except Exception as e:
        log("sys", f"UniProt lookup failed: {e}")
        return []
    out: list[dict] = []
    for x in data.get("uniProtKBCrossReferences", []):
        if x.get("database") != "PDB":
            continue
        props = {p["key"]: p["value"] for p in x.get("properties", [])}
        # resolution comes as e.g. "1.70 A" — extract the float
        res_str = props.get("Resolution", "") or ""
        res_val = None
        try:
            res_val = float(res_str.split()[0])
        except Exception:
            pass
        out.append({
            "pdb_id":     x.get("id", ""),
            "method":     props.get("Method", ""),
            "resolution": res_val,
            "chains":     props.get("Chains", ""),
        })
    # sort: x-ray first by resolution asc, then NMR / EM, then unknown
    def _key(r):
        method_rank = 0 if "X-ray" in (r["method"] or "") else 1
        res = r["resolution"] if r["resolution"] is not None else 1e6
        return (method_rank, res)
    out.sort(key=_key)

    # Append model-based fallbacks so the UI can offer them when no
    # experimental structure exists (very common for fungal proteins).
    # We probe quietly with HEAD/GET and only list what's actually there.
    sm = _swissmodel_summary(uniprot_id)
    if sm:
        out.append(sm)
    af = _alphafold_summary(uniprot_id)
    if af:
        out.append(af)

    log("phi", f"UniProt {uniprot_id}: found {len(out)} structural entries (incl. models)")
    return out


def _swissmodel_summary(uniprot_id: str, timeout: float = 6.0) -> Optional[dict]:
    """Probe SWISS-MODEL repository and return a list-entry dict if a model exists."""
    try:
        import httpx
    except ImportError:
        return None
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.head(url, headers={"User-Agent": "PhiEx/0.1"})
            if r.status_code == 200:
                return {
                    "pdb_id":     f"SWISS:{uniprot_id}",
                    "method":     "SWISS-MODEL (homology)",
                    "resolution": None,
                    "chains":     "",
                }
    except Exception:
        pass
    return None


def _alphafold_summary(uniprot_id: str, timeout: float = 6.0) -> Optional[dict]:
    """Probe AlphaFold DB and return a list-entry dict if a prediction exists."""
    try:
        import httpx
    except ImportError:
        return None
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.head(url, headers={"User-Agent": "PhiEx/0.1"})
            if r.status_code == 200:
                return {
                    "pdb_id":     f"AF:{uniprot_id}",
                    "method":     "AlphaFold (predicted)",
                    "resolution": None,
                    "chains":     "",
                }
    except Exception:
        pass
    return None


def fetch_pdb_text(pdb_id: str, cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Return raw PDB text for `pdb_id` or None if unfetchable.  Caches to disk
    so re-runs are fast and the test target works offline after first run.

    Special prefixes route to alternative structure sources:
        SWISS:<UniProt>  → SWISS-MODEL homology model for that accession
        AF:<UniProt>     → AlphaFold DB prediction for that accession
        AF2:<UniProt>    → alias for AF:
    Anything else is treated as an RCSB PDB code.
    """
    pdb_id = pdb_id.upper().strip()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    # Special prefixes
    if ":" in pdb_id:
        prefix, acc = pdb_id.split(":", 1)
        prefix = prefix.upper().strip()
        acc = acc.strip()
        if prefix == "SWISS":
            return fetch_swissmodel_pdb(acc, cache_dir=cache_dir)
        if prefix in ("AF", "AF2", "ALPHAFOLD"):
            return fetch_alphafold_pdb(acc, cache_dir=cache_dir)
        log("sys", f"unknown structure prefix '{prefix}' — falling back to RCSB lookup of {acc}")
        pdb_id = acc
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


def fetch_swissmodel_pdb(uniprot_id: str,
                         cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Fetch the best SWISS-MODEL homology model for a UniProt accession.

    SWISS-MODEL exposes its repository at
        https://swissmodel.expasy.org/repository/uniprot/{ID}.pdb
    which redirects to the highest-quality automated model (sorted by
    GMQE / coverage on the SWISS-MODEL side).  We just take whatever it
    returns and cache it as SWISS_<id>.pdb so it doesn't collide with
    RCSB cache entries of the same accession.
    """
    uniprot_id = uniprot_id.upper().strip()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"SWISS_{uniprot_id}.pdb"
    if cached.exists():
        log("phi", f"SWISS-MODEL {uniprot_id}: loaded from cache")
        return cached.read_text()
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb"
    text = _fetch_text(url)
    if text and text.lstrip().startswith(("HEADER", "ATOM", "REMARK", "TITLE")):
        cached.write_text(text)
        log("phi", f"SWISS-MODEL {uniprot_id}: fetched ({len(text)} bytes) and cached")
        return text
    log("sys", f"SWISS-MODEL {uniprot_id}: no model available")
    return None


def fetch_alphafold_pdb(uniprot_id: str,
                        cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Fetch the AlphaFold DB prediction for a UniProt accession.

    AlphaFold DB serves PDBs at predictable URLs:
        https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v4.pdb
    For most fungal proteins (and indeed most of UniProt) AlphaFold has a
    prediction, so this is the most reliable fallback when no PDB and no
    SWISS-MODEL exist.
    """
    uniprot_id = uniprot_id.upper().strip()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"AF_{uniprot_id}.pdb"
    if cached.exists():
        log("phi", f"AlphaFold {uniprot_id}: loaded from cache")
        return cached.read_text()
    # Try v4 first (current as of late 2023+); fall back to v3, v2.
    for v in ("v4", "v3", "v2"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{v}.pdb"
        text = _fetch_text(url)
        if text and text.lstrip().startswith(("HEADER", "ATOM", "REMARK", "TITLE")):
            cached.write_text(text)
            log("phi", f"AlphaFold {uniprot_id} ({v}): fetched ({len(text)} bytes) and cached")
            return text
    log("sys", f"AlphaFold {uniprot_id}: no prediction available")
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
