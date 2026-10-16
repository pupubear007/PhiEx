"""
PhiEx.analysis.seqtrack — per-residue tracks for the sequence viewer.

The AlphaFold DB entry page shows the sequence coloured by pLDDT above the
3D model, plus the predicted aligned error (PAE) map.  This module builds
the same picture for whatever structure is loaded, and adds the tracks
PhiEx itself knows about:

    plddt        model confidence (AlphaFold / AlphaFill models only)
    uniprot      signal peptide, transit peptide, transmembrane, domains,
                 regions, motifs, active/binding sites, disulfides,
                 glycosylation — from the UniProt entry
    site         residues defining the active site (anchor / cofactor /
                 catalytic triad) and the active pocket
    motif        catalytic motifs found in the sequence (G-x-S-x-G …)
    contacts     residues ≤ 4 Å from the best docked pose
    constraint   ESM-2 mutational intolerance: mean −LLR over the 19
                 substitutions at each position (high = evolution-like
                 constraint, i.e. the language model rejects changes there)
    variants     reserved for isolate variants (sequence differences between
                 isolates, mapped from RNA-seq) — filled by a later step

Numbering: AlphaFold models use UniProt numbering, so UniProt features map
1:1.  For experimental structures the chain sequence is aligned to the
UniProt sequence and features are carried across the aligned blocks.
"""

from __future__ import annotations
import difflib
import json
import math
import re
from pathlib import Path
from typing import Optional

from ..ticker import log

_AA1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}

# UniProt feature types worth drawing, and the track each goes on
_UNIPROT_TRACKS = {
    "Signal": "topology", "Transit peptide": "topology", "Propeptide": "topology",
    "Transmembrane": "topology", "Intramembrane": "topology",
    "Domain": "domains", "Region": "domains", "Repeat": "domains", "Zinc finger": "domains",
    "Coiled coil": "domains", "Compositional bias": "domains",
    "Motif": "sites", "Active site": "sites", "Binding site": "sites", "Site": "sites",
    "Metal binding": "sites", "Disulfide bond": "ptm", "Glycosylation": "ptm",
    "Modified residue": "ptm", "Lipidation": "ptm",
}


# ────────────────────────────────────────────────────────────────────────
# UniProt entry (features + canonical sequence), cached
# ────────────────────────────────────────────────────────────────────────

def fetch_uniprot_entry(acc: str) -> Optional[dict]:
    if not acc:
        return None
    path = Path("data/uniprot") / f"{acc}.entry.json"
    try:
        return json.loads(path.read_text())
    except Exception:
        pass
    try:
        import httpx
        r = httpx.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json", timeout=15.0,
                      headers={"User-Agent": "PhiEx/0.1"})
        if r.status_code != 200:
            log("sys", f"UniProt entry {acc}: HTTP {r.status_code}")
            return None
        data = r.json()
        slim = {"accession": acc,
                "sequence": (data.get("sequence") or {}).get("value", ""),
                "features": [{"type": f.get("type"), "description": f.get("description", ""),
                              "start": ((f.get("location") or {}).get("start") or {}).get("value"),
                              "end": ((f.get("location") or {}).get("end") or {}).get("value")}
                             for f in data.get("features", []) if f.get("type") in _UNIPROT_TRACKS]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(slim))
        return slim
    except Exception as e:
        log("sys", f"UniProt entry {acc} unavailable ({e})")
        return None


def uniprot_to_residue_map(uniprot_seq: str, chain_seq: str, resnums: list[int]) -> dict[int, int]:
    """UniProt position (1-based) → structure residue number, via the
    matching blocks of a difflib alignment.  Identity map when equal."""
    if not uniprot_seq or not chain_seq:
        return {}
    if uniprot_seq == chain_seq:
        return {i + 1: resnums[i] for i in range(len(resnums))}
    sm = difflib.SequenceMatcher(a=uniprot_seq, b=chain_seq, autojunk=False)
    out = {}
    for blk in sm.get_matching_blocks():
        if blk.size < 4:              # ignore tiny accidental matches
            continue
        for k in range(blk.size):
            out[blk.a + k + 1] = resnums[blk.b + k]
    return out


# ────────────────────────────────────────────────────────────────────────
# track builder
# ────────────────────────────────────────────────────────────────────────

def _segments(residue_numbers: list[int]) -> list[dict]:
    """[3,4,5,9] → [{start:3,end:5},{start:9,end:9}]"""
    out, run = [], []
    for n in sorted(set(residue_numbers)):
        if run and n == run[-1] + 1:
            run.append(n)
        else:
            if run:
                out.append({"start": run[0], "end": run[-1]})
            run = [n]
    if run:
        out.append({"start": run[0], "end": run[-1]})
    return out


def build_track(protein, uniprot: Optional[str] = None, site=None, active_pocket=None,
                esm2_pred=None, pose_contacts=None, catalytic_hits=None) -> dict:
    chain = protein.residues[0].chain if protein.residues else None
    res = [r for r in protein.residues if r.chain == chain]
    seq = "".join(_AA1.get(r.name.upper(), "X") for r in res)
    resnums = [r.index for r in res]
    is_model = (protein.name or "").upper().startswith(("AF:", "AFILL:"))

    plddt = None
    if is_model:
        vals = [(r.ca.bfactor if r.ca else None) for r in res]
        if any(v is not None for v in vals):
            scale = 100.0 if max(v for v in vals if v is not None) <= 1.0 else 1.0
            plddt = [None if v is None else round(v * scale, 1) for v in vals]

    tracks: list[dict] = []

    # UniProt features
    entry = fetch_uniprot_entry(uniprot) if uniprot else None
    if entry:
        pos_map = uniprot_to_residue_map(entry.get("sequence", ""), seq, resnums)
        groups: dict[str, list] = {}
        for f in entry.get("features", []):
            s, e = f.get("start"), f.get("end")
            if s is None or e is None:
                continue
            if f["type"] == "Disulfide bond":
                pts = [pos_map.get(s), pos_map.get(e)]
                feats = [{"start": p, "end": p} for p in pts if p is not None]
            else:
                mapped = [pos_map[p] for p in range(s, e + 1) if p in pos_map]
                feats = _segments(mapped)
            for seg in feats:
                groups.setdefault(_UNIPROT_TRACKS[f["type"]], []).append(
                    {**seg, "label": f["type"] + (f": {f['description']}" if f.get("description") else ""),
                     "type": f["type"]})
        for name in ("topology", "domains", "sites", "ptm"):
            if groups.get(name):
                tracks.append({"name": f"UniProt {name}", "kind": "features", "features": groups[name]})

    # PhiEx: site + pocket
    feats = []
    if site is not None:
        for n in getattr(site, "residues", ()) or ():
            feats.append({"start": n, "end": n, "label": f"site ({site.label})", "type": "site"})
    if active_pocket is not None:
        site_set = set(getattr(site, "residues", ()) or ())
        for seg in _segments([n for n in active_pocket.nearby_residues if n not in site_set]):
            feats.append({**seg, "label": f"pocket {active_pocket.id}", "type": "pocket"})
    if feats:
        tracks.append({"name": "PhiEx site / pocket", "kind": "features", "features": feats})

    # catalytic motifs in sequence + detected triads
    feats = []
    for m in re.finditer(r"G.S.G", seq):
        feats.append({"start": resnums[m.start()], "end": resnums[m.end() - 1],
                      "label": f"G-x-S-x-G nucleophile elbow ({seq[m.start():m.end()]})", "type": "motif"})
    for m in re.finditer(r"HG[GA]", seq):
        feats.append({"start": resnums[m.start()], "end": resnums[m.end() - 1],
                      "label": f"HGG/HGA oxyanion-hole motif ({seq[m.start():m.end()]})", "type": "motif"})
    try:
        from .site import detect_cofactor_motifs
        for h in detect_cofactor_motifs(protein, chain):
            rr = h["residues"]
            feats.append({"start": rr[0].index, "end": rr[-1].index,
                          "label": f"{h['label']} ({h['text']}) → {h['cofactor']}", "type": "motif"})
    except Exception:
        pass
    for h in (catalytic_hits or [])[:3]:
        trio = [h.nucleophile, h.histidine] + ([h.acid] if h.acid else [])
        for r in trio:
            feats.append({"start": r.index, "end": r.index, "label": f"catalytic: {h.label}", "type": "catalytic"})
    if feats:
        tracks.append({"name": "catalytic motifs", "kind": "features", "features": feats})

    # docked-pose contacts
    if pose_contacts:
        tracks.append({"name": "docked-ligand contacts", "kind": "features",
                       "features": [{**seg, "label": "≤ 4 Å of best pose", "type": "contact"}
                                    for seg in _segments(list(pose_contacts))]})

    # ESM-2 constraint
    lp = None
    if esm2_pred is not None and not (esm2_pred.meta or {}).get("stub"):
        lp = (esm2_pred.value or {}).get("aa_logprobs")
    if lp is not None and len(lp) == len(seq):
        from ..adapters.esm2 import AA20
        vals = []
        for i, wt in enumerate(seq):
            if wt not in AA20:
                vals.append(None)
                continue
            row = lp[i]; w = float(row[AA20.index(wt)])
            vals.append(round(sum(w - float(row[j]) for j in range(20) if AA20[j] != wt) / 19.0, 3))
        tracks.append({"name": "ESM-2 constraint", "kind": "values", "values": vals,
                       "label": "mean −LLR of substitutions (higher = less tolerant)"})

    # placeholder for the isolate-variant step
    tracks.append({"name": "isolate variants", "kind": "features", "features": [],
                   "label": "not yet computed — see the RNA-seq variant step"})

    return {"name": protein.name, "chain": chain, "uniprot": uniprot, "sequence": seq,
            "residues": resnums, "plddt": plddt, "is_model": is_model, "tracks": tracks}


# ────────────────────────────────────────────────────────────────────────
# predicted aligned error (AlphaFold DB)
# ────────────────────────────────────────────────────────────────────────

def fetch_pae(acc: str, max_bins: int = 400) -> Optional[dict]:
    """PAE matrix for the AlphaFold model of `acc`, downsampled (block max)
    to at most `max_bins` × `max_bins` for the browser."""
    if not acc:
        return None
    cache = Path("data/pdb") / f"AF_{acc}_pae.json"
    raw = None
    try:
        raw = json.loads(cache.read_text())
    except Exception:
        from ..adapters.pdb import _alphafold_api_entry
        entry, status = _alphafold_api_entry(acc)
        url = (entry or {}).get("paeDocUrl")
        if status != "ok" or not url:
            return None
        try:
            import httpx
            r = httpx.get(url, timeout=30.0, headers={"User-Agent": "PhiEx/0.1"})
            if r.status_code != 200:
                return None
            raw = r.json()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw))
        except Exception as e:
            log("sys", f"PAE for {acc} unavailable ({e})")
            return None
    return downsample_pae(raw, max_bins)


def downsample_pae(raw, max_bins: int = 400) -> Optional[dict]:
    doc = raw[0] if isinstance(raw, list) else raw
    mat = doc.get("predicted_aligned_error")
    if mat is None and "distance" in doc:          # pre-2022 format
        n = int(math.isqrt(len(doc["distance"])))
        mat = [doc["distance"][i * n:(i + 1) * n] for i in range(n)]
    if not mat:
        return None
    n = len(mat)
    mx = doc.get("max_predicted_aligned_error") or max(max(r) for r in mat)
    b = max(1, math.ceil(n / max_bins))
    if b == 1:
        small = [[round(float(v), 1) for v in row] for row in mat]
    else:
        m = math.ceil(n / b)
        small = [[round(max(float(mat[i][j]) for i in range(I * b, min(n, (I + 1) * b))
                            for j in range(J * b, min(n, (J + 1) * b))), 1)
                  for J in range(m)] for I in range(m)]
    return {"n": n, "bin": b, "max": mx, "matrix": small}
