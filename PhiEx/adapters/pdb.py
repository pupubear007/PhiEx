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
    af_from_xref: Optional[dict] = None
    for x in data.get("uniProtKBCrossReferences", []):
        db = x.get("database")
        if db == "PDB":
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
        elif db == "AlphaFoldDB" and af_from_xref is None:
            # UniProt is the authoritative source for which AF prediction
            # exists for this accession. The xref id looks like the bare
            # accession (e.g. "P13006"); the full model id is AF-<acc>-F1.
            af_id = x.get("id") or uniprot_id
            af_from_xref = {
                "pdb_id":     f"AF:{af_id}",
                "method":     "AlphaFold (predicted)",
                "resolution": None,
                "chains":     "",
                "_source":    "uniprot_xref",
            }
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
    # AlphaFold: prefer the UniProt xref (no extra HTTP call, always-current
    # version). Fall back to the standalone probe if the xref is missing
    # (some accessions have an AF model but no xref yet).
    if af_from_xref is not None:
        af_from_xref.pop("_source", None)
        out.append(af_from_xref)
    else:
        af = _alphafold_summary(uniprot_id)
        if af:
            out.append(af)
    # AlphaFill — only meaningful if AF exists (it's an AF-derived model).
    # We probe its status endpoint quietly; if the user has no AF model
    # there's no AlphaFill either, so skip the call.
    if af_from_xref is not None or any(e["pdb_id"].startswith("AF:") for e in out):
        afill = _alphafill_summary(uniprot_id)
        if afill:
            out.append(afill)

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
    """Probe AlphaFold DB and return a list-entry dict if a prediction exists.

    Tries multiple model versions (v5 → v2) because AlphaFold DB bumps the
    version periodically and old probes silently start returning 404. Falls
    back to a small ranged GET when HEAD returns non-200, since the EBI
    CDN occasionally rejects HEAD on otherwise-fetchable static files.
    """
    try:
        import httpx
    except ImportError:
        return None
    headers = {"User-Agent": "PhiEx/0.1"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            for v in ("v5", "v4", "v3", "v2"):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{v}.pdb"
                # HEAD first (cheap)
                try:
                    r = c.head(url, headers=headers)
                    if r.status_code == 200:
                        return {
                            "pdb_id":     f"AF:{uniprot_id}",
                            "method":     f"AlphaFold (predicted, {v})",
                            "resolution": None,
                            "chains":     "",
                        }
                except Exception:
                    pass
                # GET fallback — only ask for the first 1KB so we don't pull the whole PDB
                try:
                    r = c.get(url, headers={**headers, "Range": "bytes=0-1023"})
                    if r.status_code in (200, 206):
                        return {
                            "pdb_id":     f"AF:{uniprot_id}",
                            "method":     f"AlphaFold (predicted, {v})",
                            "resolution": None,
                            "chains":     "",
                        }
                except Exception:
                    pass
    except Exception:
        pass
    return None


# Valid leading tokens for any well-formed PDB file. We use this both to
# accept newly-downloaded responses and to validate cache reads — earlier
# versions wrote ANY non-empty response (HTML errors, redirect pages) to
# the cache, then returned the same garbage forever on subsequent loads.
_PDB_VALID_HEADS = ("HEADER", "ATOM", "HETATM", "REMARK",
                    "TITLE", "COMPND", "CRYST", "MODEL")


def _is_valid_pdb_text(s: Optional[str], min_atoms: int = 1) -> bool:
    """True if `s` looks like a real PDB file with at least one ATOM record."""
    if not s or len(s) < 80:
        return False
    if not s.lstrip().startswith(_PDB_VALID_HEADS):
        return False
    # Quick sanity: count ATOM/HETATM lines without scanning the whole file
    atoms = 0
    for line in s.splitlines():
        if line.startswith(("ATOM ", "HETATM")):
            atoms += 1
            if atoms >= min_atoms:
                return True
    return atoms >= min_atoms


def _read_cache_or_purge(path: Path, label: str) -> Optional[str]:
    """Return cached PDB text iff valid; otherwise delete the cache and return None.

    Stops a single bad fetch from poisoning every subsequent load — which
    is the actual reason AlphaFold has been silently failing for users.
    """
    if not path.exists():
        return None
    text = path.read_text()
    if _is_valid_pdb_text(text):
        log("phi", f"{label}: loaded from cache")
        return text
    log("sys", f"{label}: cached file at {path} is malformed ({len(text)} bytes) — purging")
    try:
        path.unlink()
    except OSError:
        pass
    return None


def fetch_pdb_text(pdb_id: str, cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Return raw PDB text for `pdb_id` or None if unfetchable.  Caches to disk
    so re-runs are fast and the test target works offline after first run.

    Special prefixes route to alternative structure sources:
        SWISS:<UniProt>   → SWISS-MODEL homology model
        AF:<UniProt>      → AlphaFold DB prediction (PDB, with CIF fallback)
        AF2:<UniProt>     → alias for AF:
        AFILL:<UniProt>   → AlphaFill prediction (cofactor-transplanted AF model)
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
        if prefix in ("AFILL", "ALPHAFILL"):
            return fetch_alphafill_pdb(acc, cache_dir=cache_dir)
        log("sys", f"unknown structure prefix '{prefix}' — falling back to RCSB lookup of {acc}")
        pdb_id = acc
    cached_text = _read_cache_or_purge(cache / f"{pdb_id}.pdb", f"PDB {pdb_id}")
    if cached_text is not None:
        return cached_text
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    text = _fetch_text(url)
    if _is_valid_pdb_text(text):
        (cache / f"{pdb_id}.pdb").write_text(text)
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
    cached_text = _read_cache_or_purge(cached, f"SWISS-MODEL {uniprot_id}")
    if cached_text is not None:
        return cached_text
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.pdb"
    text = _fetch_text(url)
    if _is_valid_pdb_text(text):
        cached.write_text(text)
        log("phi", f"SWISS-MODEL {uniprot_id}: fetched ({len(text)} bytes) and cached")
        return text
    log("sys", f"SWISS-MODEL {uniprot_id}: no model available")
    return None


def fetch_alphafold_pdb(uniprot_id: str,
                        cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Fetch the AlphaFold DB prediction for a UniProt accession.

    AlphaFold DB serves models at predictable URLs:
        https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v{N}.pdb
        https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v{N}.cif
    Newer "extended" entries (AFDB Clusters / collaborator predictions)
    are CIF-only — when every PDB version 404s we fall back to CIF and
    convert it inline so the rest of the pipeline doesn't have to care.
    """
    uniprot_id = uniprot_id.upper().strip()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"AF_{uniprot_id}.pdb"
    cached_text = _read_cache_or_purge(cached, f"AlphaFold {uniprot_id}")
    if cached_text is not None:
        return cached_text
    # Try v5 first (current as of 2024+); fall back to v4, v3, v2.
    for v in ("v5", "v4", "v3", "v2"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{v}.pdb"
        text = _fetch_text(url)
        if _is_valid_pdb_text(text):
            cached.write_text(text)
            log("phi", f"AlphaFold {uniprot_id} ({v}): fetched ({len(text)} bytes) and cached")
            return text
    # CIF fallback for extended entries
    for v in ("v5", "v4", "v3", "v2"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{v}.cif"
        cif = _fetch_text(url)
        if cif and ("_atom_site." in cif or "loop_" in cif):
            pdb_text = _cif_to_pdb_text(cif)
            if _is_valid_pdb_text(pdb_text):
                cached.write_text(pdb_text)
                log("phi", f"AlphaFold {uniprot_id} ({v}, CIF→PDB): "
                           f"fetched ({len(cif)} bytes CIF, {len(pdb_text)} bytes PDB)")
                return pdb_text
    log("sys", f"AlphaFold {uniprot_id}: no prediction available (PDB and CIF both failed)")
    return None


# ────────────────────────────────────────────────────────────────────────
# AlphaFill — cofactor-transplanted AlphaFold models
# ────────────────────────────────────────────────────────────────────────

def fetch_alphafill_pdb(uniprot_id: str,
                        cache_dir: str | os.PathLike = "data/pdb") -> Optional[str]:
    """Fetch the AlphaFill prediction for a UniProt accession.

    AlphaFill (https://alphafill.eu) takes an AlphaFold model and
    transplants ligands/cofactors from homologous PDB entries. The API
    serves mmCIF only:
        https://alphafill.eu/v1/aff/{ID}
    We convert to PDB on the fly so the parser doesn't need a CIF code path.
    """
    uniprot_id = uniprot_id.upper().strip()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / f"AFILL_{uniprot_id}.pdb"
    cached_text = _read_cache_or_purge(cached, f"AlphaFill {uniprot_id}")
    if cached_text is not None:
        return cached_text
    url = f"https://alphafill.eu/v1/aff/{uniprot_id}"
    cif = _fetch_text(url, timeout=20.0)
    if not cif or ("_atom_site." not in cif and "loop_" not in cif):
        log("sys", f"AlphaFill {uniprot_id}: no model available")
        return None
    pdb_text = _cif_to_pdb_text(cif)
    if not _is_valid_pdb_text(pdb_text):
        log("sys", f"AlphaFill {uniprot_id}: CIF→PDB conversion produced no atoms")
        return None
    cached.write_text(pdb_text)
    log("phi", f"AlphaFill {uniprot_id}: fetched "
               f"({len(cif)} bytes CIF, {len(pdb_text)} bytes PDB) and cached")
    return pdb_text


def _alphafill_summary(uniprot_id: str, timeout: float = 6.0) -> Optional[dict]:
    """Probe AlphaFill and return a list-entry dict if a model exists."""
    try:
        import httpx
    except ImportError:
        return None
    headers = {"User-Agent": "PhiEx/0.1"}
    # AlphaFill exposes a JSON status endpoint that's cheaper than fetching
    # the full structure — use it for the existence probe.
    url = f"https://alphafill.eu/v1/aff/{uniprot_id}/status"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(url, headers=headers)
            if r.status_code == 200:
                return {
                    "pdb_id":     f"AFILL:{uniprot_id}",
                    "method":     "AlphaFill (cofactor-transplanted AF)",
                    "resolution": None,
                    "chains":     "",
                }
            # Some AlphaFill deployments return the model directly with no status route
            r = c.head(f"https://alphafill.eu/v1/aff/{uniprot_id}", headers=headers)
            if r.status_code == 200:
                return {
                    "pdb_id":     f"AFILL:{uniprot_id}",
                    "method":     "AlphaFill (cofactor-transplanted AF)",
                    "resolution": None,
                    "chains":     "",
                }
    except Exception:
        pass
    return None


# ────────────────────────────────────────────────────────────────────────
# minimal mmCIF → PDB converter
# ────────────────────────────────────────────────────────────────────────

def _cif_to_pdb_text(cif: str) -> str:
    """Convert an mmCIF file to PDB text by extracting the atom_site loop.

    Handles the standard PDBx/mmCIF atom_site fields; ignores everything
    else. Coordinates, occupancy, B-factor, chain, residue number, and
    element are preserved. This is intentionally minimal — we just need
    enough for the existing PDB parser to work.
    """
    lines = cif.splitlines()
    i = 0
    n = len(lines)
    out: list[str] = ["HEADER    CONVERTED FROM mmCIF BY PhiEx"]
    atom_serial = 1
    while i < n:
        line = lines[i].strip()
        if line == "loop_":
            # Collect column names for this loop
            cols: list[str] = []
            i += 1
            while i < n and lines[i].lstrip().startswith("_"):
                cols.append(lines[i].strip())
                i += 1
            # Is this the atom_site loop?
            if not any(c.startswith("_atom_site.") for c in cols):
                continue
            col_idx = {c.split(".",1)[1]: k for k, c in enumerate(cols)
                       if c.startswith("_atom_site.")}
            # Read data rows until we hit the next loop_/category or '#' / EOF
            while i < n:
                row = lines[i]
                stripped = row.strip()
                if (not stripped) or stripped.startswith("#") \
                        or stripped == "loop_" or stripped.startswith("_") \
                        or stripped.startswith("data_"):
                    break
                # mmCIF uses whitespace-separated tokens, with quoted strings
                # for values containing spaces. The atom_site loop never has
                # spaces inside its values, so a plain split is safe.
                toks = stripped.split()
                if len(toks) < len(col_idx):
                    i += 1
                    continue
                def g(name: str, default: str = ".") -> str:
                    k = col_idx.get(name)
                    return toks[k] if k is not None and k < len(toks) else default
                group = g("group_PDB", "ATOM").upper()
                if group not in ("ATOM", "HETATM"):
                    i += 1
                    continue
                atom_name = g("auth_atom_id") if "auth_atom_id" in col_idx else g("label_atom_id")
                atom_name = atom_name.strip('"').strip("'")
                element   = g("type_symbol", atom_name[:1]).strip(".")
                resname   = (g("auth_comp_id") if "auth_comp_id" in col_idx
                             else g("label_comp_id")).strip()
                chain     = (g("auth_asym_id") if "auth_asym_id" in col_idx
                             else g("label_asym_id"))[:1] or "A"
                resseq_s  = g("auth_seq_id") if "auth_seq_id" in col_idx else g("label_seq_id")
                try:
                    resseq = int(resseq_s)
                except (ValueError, TypeError):
                    i += 1
                    continue
                try:
                    x = float(g("Cartn_x")); y = float(g("Cartn_y")); z = float(g("Cartn_z"))
                except ValueError:
                    i += 1
                    continue
                try:
                    occ = float(g("occupancy", "1.00"))
                except ValueError:
                    occ = 1.0
                try:
                    bfac = float(g("B_iso_or_equiv", "0.00"))
                except ValueError:
                    bfac = 0.0
                # Format atom name to PDB convention (4-char field, element-aligned)
                if len(element) == 1 and len(atom_name) < 4:
                    aname = f" {atom_name:<3}"
                else:
                    aname = f"{atom_name:<4}"
                rec = "ATOM  " if group == "ATOM" else "HETATM"
                out.append(
                    f"{rec}{atom_serial:5d} {aname[:4]} {resname[:3]:>3} "
                    f"{chain}{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          "
                    f"{element[:2]:>2}"
                )
                atom_serial += 1
                i += 1
            continue
        i += 1
    out.append("END")
    return "\n".join(out) + "\n"


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
    # Surface the fetch attempt so the UI ticker shows progress for slow
    # AlphaFold / SWISS-MODEL pulls — otherwise the viewers just stay blank
    # for several seconds with no indication anything is happening.
    if pdb_id.upper().startswith(("AF:", "AF2:", "ALPHAFOLD:")):
        log("phi", f"AlphaFold {pdb_id}: fetching from EBI…")
    elif pdb_id.upper().startswith("SWISS:"):
        log("phi", f"SWISS-MODEL {pdb_id}: fetching from ExPASy…")
    pdb_text = fetch_pdb_text(pdb_id)
    annotations = fetch_uniprot_annotations(uniprot)
    if pdb_text is None:
        # Only fall back to the APX stub when the user is actually asking
        # for APX. For any other target, returning the APX sequence would
        # silently swap in a completely unrelated protein (an Aspergillus
        # niger glucose oxidase request would come back as pea ascorbate
        # peroxidase). Instead, return an empty Protein so the UI surfaces
        # a clear "structure unavailable" state.
        if uniprot.upper() == APX_DEFAULT_UNIPROT or pdb_id.upper() == APX_DEFAULT_PDB:
            log("sys", f"PDB fetch unavailable, using STUB sequence-only APX")
            return stub_apx_protein(), [], annotations
        log("sys", f"PDB {pdb_id} (UniProt {uniprot}): fetch failed — "
                   f"check network / verify the AlphaFold model exists for this accession")
        return Protein(name=pdb_id, sequence="", source="unavailable"), [], annotations
    protein, cofactors = parse_pdb_text(pdb_text, name=pdb_id)
    log("phi", f"parsed {pdb_id}: {protein.n_residues} residues, "
                f"{len(cofactors)} cofactor(s) "
                f"({', '.join(c.name for c in cofactors) or 'none'})")
    return protein, cofactors, annotations
