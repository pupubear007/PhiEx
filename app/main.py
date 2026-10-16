"""
app.main — FastAPI backend for the PhiEx Sandbox.

Routes:
    GET  /                 → static frontend (index.html in static/)
    GET  /healthz          → liveness
    GET  /api/device       → describe selected device + OpenMM platform
    GET  /api/theories     → list of registered LearnedModels
    POST /api/structure    → fetch PDB + ESMFold, return both structures
    POST /api/annotate     → ESM-2 + Foldseek + UniProt → ϕ0
    POST /api/pockets      → P2Rank + active-site flag
    POST /api/dock         → Vina (default) or DiffDock
    POST /api/dynamics     → OpenMM short MD + MACE region energy
    POST /api/al/iterate   → one iteration of the active-learning loop
    POST /api/apx/run      → entire APX pipeline end-to-end

    GET  /events           → server-sent events stream of ticker entries

Streaming policy: every long-running route accepts `stream: true` and
yields SSE events with the same shape as the /events stream so the UI
gets uniform reasoning-ticker updates regardless of which endpoint is in
flight.
"""

from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from PhiEx.ticker import TICKER, log
from PhiEx.device import select_device, describe_device, select_openmm_platform
from PhiEx.core.theory import THEORIES
from PhiEx.pipeline.apx import (APXPipeline, APX_DEFAULT_PDB,
                                resolve_working_structure, working_chain,
                                log_site, pick_ligand, pose_contact_residues,
                                rank_site_residues, build_mutation_evaluator, _AA1)
from PhiEx.analysis.site import resolve_active_site, choose_active_pocket

# adapters used by the per-stage routes
from PhiEx.adapters.pdb import fetch_protein, parse_pdb_text, list_pdbs_for_uniprot
from PhiEx.adapters.esmfold import ESMFoldAdapter, ca_rmsd
from PhiEx.adapters.esm2 import ESM2Adapter, attention_residue_importance
from PhiEx.adapters.foldseek import FoldseekAdapter, aggregate_go_terms
from PhiEx.adapters.p2rank import P2RankAdapter
from PhiEx.adapters.vina import VinaAdapter, ASCORBATE
from PhiEx.adapters.diffdock import DiffDockAdapter
from PhiEx.adapters.openmm_calc import OpenMMSimulator
from PhiEx.adapters.mace import MACEAdapter, select_active_site_region
from PhiEx.adapters.surrogate import GBRSurrogate, encode_perturbation
from PhiEx.al import ActiveLearningLoop, make_acquisition, generate_mutation_panel

from PhiEx.core.state import (Atom, Residue, Protein, Cofactor,
                                   Ligand, ComplexState, Pocket)


app = FastAPI(title="PhiEx Sandbox v0",
              version="0.1.0",
              description="ϕ→∃ / s∃⟳tϕ workbench for in planta / in vitro biomolecular ML")


# ────────────────────────────────────────────────────────────────────────
# startup
# ────────────────────────────────────────────────────────────────────────

# session-level scratch so endpoints don't have to re-fetch on every call.
SESSION: dict[str, Any] = {}


@app.on_event("startup")
async def _startup() -> None:
    dev = select_device()
    info = describe_device(dev)
    plat, props = select_openmm_platform()
    log("sys", f"device selected: {dev}  info={info}")
    log("sys", f"openmm platform: {plat}  props={props}")
    SESSION["device"] = dev
    SESSION["openmm_platform"] = plat


# ────────────────────────────────────────────────────────────────────────
# errors — surface the cause instead of a bare "HTTP 500"
# ────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Log the failure to the ticker and the server console, and return it
    as {"error": ...} — the frontend's postJSON shows that field verbatim."""
    import traceback
    where = request.url.path
    msg = f"{type(exc).__name__}: {exc}"
    log("sys", f"{where} failed — {msg}")
    traceback.print_exception(exc)
    return JSONResponse({"error": f"{where}: {msg}"}, status_code=500)


# ────────────────────────────────────────────────────────────────────────
# static
# ────────────────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text())
    return HTMLResponse(
        "<h1>PhiEx Sandbox v0</h1><p>Frontend not built (static/index.html missing).</p>",
        status_code=200)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": SESSION.get("device")}


# ────────────────────────────────────────────────────────────────────────
# small data endpoints
# ────────────────────────────────────────────────────────────────────────

@app.get("/api/device")
async def api_device():
    dev = SESSION.get("device") or select_device()
    info = describe_device(dev)
    plat, props = select_openmm_platform()
    return {"device": dev, "info": info,
            "openmm_platform": plat, "openmm_properties": props}


@app.get("/api/theories")
async def api_theories():
    return {"theories": THEORIES.info()}


@app.get("/api/ticker/history")
async def api_ticker_history():
    return {"events": TICKER.history()}


@app.get("/api/uniprot/{uniprot_id}/pdbs")
def api_uniprot_pdbs(uniprot_id: str):
    """Return all PDB entries cross-referenced from a UniProt ID, best
    resolution first.  The frontend's "find PDB" button uses this."""
    entries = list_pdbs_for_uniprot(uniprot_id)
    return {"uniprot": uniprot_id, "n": len(entries), "entries": entries}


@app.post("/api/reset")
async def api_reset():
    """Wipe the per-session protein/cofactor/pocket/dock/MD/AL state, the
    learned-models registry, and the ticker replay buffer.  The frontend
    calls this from the ⊗ RESET ALL button to start fresh."""
    # preserve only the device + openmm platform we resolved at startup
    keep = {k: SESSION[k] for k in ("device", "openmm_platform") if k in SESSION}
    SESSION.clear()
    SESSION.update(keep)
    THEORIES.clear()
    TICKER.clear_history()
    log("sys", "session reset — fresh start")
    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────
# SSE — reasoning ticker stream
# ────────────────────────────────────────────────────────────────────────

@app.get("/events")
async def events(request: Request):
    async def gen():
        async for ev in TICKER.subscribe():
            if await request.is_disconnected():
                break
            yield f"data: {ev.to_json()}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────────────────
# pipeline routes
#
# These are plain `def` handlers on purpose: they do blocking work (network
# fetches, ESMFold, MD), and FastAPI runs sync handlers in its threadpool.
# Declaring them `async def` would stall the event loop — and with it the
# /events ticker stream — for the whole duration of each stage.
# ────────────────────────────────────────────────────────────────────────

class StructureRequest(BaseModel):
    pdb_id: str = APX_DEFAULT_PDB
    uniprot: str = "P48534"
    run_esmfold: bool = True


@app.post("/api/structure")
def api_structure(req: StructureRequest):
    protein, cofactors, ann = fetch_protein(req.pdb_id, req.uniprot)
    if not protein.residues and protein.source == "unavailable":
        # never leave the previous protein in place: every later stage would
        # silently run on it while the UI shows the new accession
        for k in ("protein", "cofactors", "uniprot", "site", "pockets", "active_pocket",
                  "pose_contacts", "esm2_pred", "best_pose"):
            SESSION.pop(k, None)
        return JSONResponse({"error": f"no structure could be downloaded for {req.pdb_id} "
                                      f"(UniProt {req.uniprot or '—'}); see the log"}, status_code=404)
    SESSION["annotations"] = ann
    out: dict[str, Any] = {
        "experimental": _protein_summary(protein),
        "annotations": ann,
    }
    predicted = None
    if req.run_esmfold and protein.sequence:
        esmfold = ESMFoldAdapter()
        if protein.residues:
            esmfold.set_reference(protein)
        THEORIES.register(esmfold)
        pred = esmfold.predict_chain(protein)
        predicted = pred.value
        SESSION["predicted"] = pred.value
        out["predicted"] = _protein_summary(pred.value)
        out["predicted"]["mean_plddt"] = pred.meta.get("mean_plddt")
        out["predicted"]["plddt_sd"] = pred.uncertainty
        out["predicted"]["pdb_text"] = pred.value.pdb_text
        if protein.residues and pred.value.residues:
            out["rmsd_pred_vs_exp"] = ca_rmsd(pred.value, protein)
    out["experimental"]["pdb_text"] = protein.pdb_text

    # same fallback as APXPipeline.run so the per-stage path and the
    # one-shot pipeline operate on the same working structure
    working, cofactors = resolve_working_structure(protein, cofactors, predicted)
    SESSION["protein"] = working
    acc = (req.uniprot or "").strip()
    used = protein.name or req.pdb_id          # may differ: AFILL → AF fallback
    if ":" in used and used.split(":", 1)[0].upper() in ("AF", "AFILL", "SWISS"):
        acc = used.split(":", 1)[1]
    SESSION["uniprot"] = acc
    for k in ("site", "pockets", "active_pocket", "pose_contacts", "esm2_pred", "best_pose"):
        SESSION.pop(k, None)              # stale per-structure state
    SESSION["cofactors"] = cofactors
    out["working_source"] = working.source
    out["residue_labels"] = {r.index: f"{r.name.title()}{r.index}"
                             for r in working_chain(working)[0]}
    out["cofactors"] = [{"name": c.name, "resname": c.resname,
                         "role": c.metadata.get("role", "cofactor"),
                         "metal": list(c.metal_atom.xyz) if c.metal_atom else None,
                         "stub": bool(c.metadata.get("stub", False))}
                        for c in cofactors]
    return out


class AnnotateRequest(BaseModel):
    use_session_protein: bool = True


@app.post("/api/annotate")
def api_annotate(req: AnnotateRequest):
    protein: Protein = SESSION.get("protein")
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    esm2 = ESM2Adapter(); THEORIES.register(esm2)
    fs = FoldseekAdapter(); THEORIES.register(fs)
    chain_res, chain_seq = working_chain(protein)
    pred = esm2.predict(chain_seq) if chain_seq else None
    fs_pred = fs.predict(protein, k=5)
    SESSION["esm2_pred"] = pred
    SESSION["foldseek"] = fs_pred
    return {
        "esm2_embedding_sd": float(pred.uncertainty) if pred else None,
        "foldseek_hits": fs_pred.value.get("hits", []),
        "go_aggregate": aggregate_go_terms(fs_pred),
        "go_uncertainty_log10": float(fs_pred.uncertainty),
    }


class PocketsRequest(BaseModel):
    anchor: str | None = None     # e.g. "S197,H351"; None = auto-detect


@app.post("/api/pockets")
def api_pockets(req: PocketsRequest | None = None):
    req = req or PocketsRequest()
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    try:
        site = resolve_active_site(protein, cofactors, req.anchor)
    except ValueError as e:
        return JSONResponse({"error": f"anchor: {e}"}, status_code=400)
    state = ComplexState(protein=protein, cofactors=tuple(cofactors))
    p2 = P2RankAdapter(); THEORIES.register(p2)
    pred = p2.predict(state)
    active, site, pockets = choose_active_pocket(protein, pred.value, site,
                                                 real_detector=not p2.is_stub)
    log_site(site, active)
    SESSION["pockets"] = pockets
    SESSION["active_pocket"] = active
    SESSION["site"] = site
    return {
        "pockets": [_pocket_json(p) for p in pockets],
        "score_sd": float(pred.uncertainty),
        "active_site_id": active.id if active else None,
        "site": site.to_json(),
    }


class DockRequest(BaseModel):
    pocket_id: str | None = None
    method: str = "vina"          # or "diffdock"
    smiles: str | None = None     # default: chosen from the site type
    ligand_name: str = ""


@app.post("/api/dock")
def api_dock(req: DockRequest):
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    pockets: list[Pocket] = SESSION.get("pockets") or []
    if not protein or not pockets:
        return JSONResponse({"error": "run /api/structure and /api/pockets first"}, status_code=400)

    pocket = (next((p for p in pockets if p.id == req.pocket_id), None)
              if req.pocket_id else next((p for p in pockets if p.is_active_site), pockets[0]))
    if pocket is None:
        return JSONResponse({"error": "no pocket selected"}, status_code=400)

    site = SESSION.get("site")
    try:
        ligand = pick_ligand(site, req.smiles, req.ligand_name)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if ligand is None:
        kind = site.kind if site else "unknown"
        return JSONResponse({"error": f"no ligand given and no default substrate for a "
                                      f"{kind} site — enter a SMILES or compound name in "
                                      f"'Ligand' (or paste a list under stage 4 and press 4b)"},
                            status_code=400)
    state = ComplexState(protein=protein, cofactors=tuple(cofactors), ligands=(ligand,),
                         metadata={"site_center": site.center} if site else {})

    if req.method == "diffdock":
        dd = DiffDockAdapter(); THEORIES.register(dd)
        poses = dd.predict(state, ligand, pocket).value
    else:
        try:
            poses = VinaAdapter().dock(state, ligand, pocket)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=422)
    SESSION["best_pose"] = poses[0] if poses else None
    contacts = []
    if poses and not poses[0].ligand.metadata.get("stub"):
        contacts = pose_contact_residues(protein, poses[0], site.chain if site else None)
        SESSION["pose_contacts"] = contacts
    return {"method": req.method,
            "ligand": {"name": ligand.name, "smiles": ligand.smiles},
            "n_poses": len(poses),
            "pose_contacts": contacts,
            "poses": [_pose_json(p) for p in poses]}


class ScreenRequest(BaseModel):
    ligands: str                  # one per line: SMILES, name, or "label: SMILES-or-name"
    exhaustiveness: int = 8


@app.post("/api/dock/batch")
def api_dock_batch(req: ScreenRequest):
    """Dock a list of ligands into the current site and rank them."""
    from PhiEx.pipeline.screen import parse_ligand_list, screen_ligands
    protein: Protein = SESSION.get("protein")
    pockets: list[Pocket] = SESSION.get("pockets") or []
    site = SESSION.get("site")
    if not protein or not pockets:
        return JSONResponse({"error": "run stages 1 and 3 (structure, site) first"},
                            status_code=400)
    pocket = next((p for p in pockets if p.is_active_site), pockets[0])
    state = ComplexState(protein=protein, cofactors=tuple(SESSION.get("cofactors") or []),
                         metadata={"site_center": site.center} if site else {})
    try:
        out = screen_ligands(state, pocket, site, parse_ligand_list(req.ligands),
                             exhaustiveness=max(1, min(32, req.exhaustiveness)))
    except (ValueError, RuntimeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    SESSION["screen"] = out
    return out


@app.get("/api/dock/batch.csv")
def api_dock_batch_csv():
    from PhiEx.pipeline.screen import rows_to_csv
    from fastapi.responses import Response
    out = SESSION.get("screen")
    if not out:
        return JSONResponse({"error": "no screen run yet"}, status_code=404)
    return Response(rows_to_csv(out["rows"]), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=phiex_ligand_screen.csv"})


# ────────────────────────────────────────────────────────────────────────
# candidate-gene triage (transcriptomics candidate list → structures, sites,
# ligand panels).  Runs in a background thread; the UI polls /status.
# ────────────────────────────────────────────────────────────────────────

class TriageRequest(BaseModel):
    text: str                          # TSV with a gene column, or one gene ID per line
    dock: bool = False                 # dock each dock-track gene's ligand panel (slow)
    exhaustiveness: int = 8
    organism: str = "all"              # all | pathogen | host
    tracks: list[str] = ["dock", "structure", "unsuited"]
    sources: str = ""                  # e.g. "TableS1.1,Table6" — keep rows whose sources contain any
    limit: int = 0                     # 0 = no limit
    use_cache: bool = True


@app.post("/api/candidates/start")
def api_candidates_start(req: TriageRequest):
    from PhiEx.candidates.triage import parse_candidates, TriageJob
    from PhiEx.candidates.classes import classify
    job = SESSION.get("triage")
    if job is not None and job.running:
        return JSONResponse({"error": "a triage is already running — stop it first"}, status_code=409)
    cands = parse_candidates(req.text)
    if req.organism == "pathogen":
        cands = [c for c in cands if not c.gene_id.startswith("Glyma")]
    elif req.organism == "host":
        cands = [c for c in cands if c.gene_id.startswith("Glyma")]
    if req.sources.strip():
        want = [s.strip() for s in req.sources.split(",") if s.strip()]
        cands = [c for c in cands if any(w in (c.sources or "") for w in want)]
    cands = [c for c in cands if classify(c.description, c.family).track in set(req.tracks)]
    if req.limit and req.limit > 0:
        cands = cands[:req.limit]
    if not cands:
        return JSONResponse({"error": "no candidate genes after filtering"}, status_code=400)
    job = TriageJob(candidates=cands, dock=req.dock,
                    exhaustiveness=max(1, min(32, req.exhaustiveness)), use_cache=req.use_cache)
    SESSION["triage"] = job
    job.start()
    return {"started": True, "total": len(cands)}


@app.get("/api/candidates/status")
def api_candidates_status():
    job = SESSION.get("triage")
    if job is None:
        return {"running": False, "done": 0, "total": 0, "rows": []}
    return job.status()


@app.post("/api/candidates/stop")
def api_candidates_stop():
    job = SESSION.get("triage")
    if job is not None:
        job.stop_requested = True
    return {"stopping": bool(job and job.running)}


@app.get("/api/candidates.csv")
def api_candidates_csv():
    from PhiEx.candidates.triage import rows_to_csv
    from fastapi.responses import Response
    job = SESSION.get("triage")
    if job is None or not job.rows:
        return JSONResponse({"error": "no triage results yet"}, status_code=404)
    return Response(rows_to_csv(job.status()["rows"]), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=phiex_candidate_triage.csv"})


@app.get("/api/candidates/local")
def api_candidates_local():
    """List candidate files kept locally under data/candidates/ (never in git)."""
    from pathlib import Path
    root = Path("data/candidates")
    files = sorted(p.name for p in root.glob("*.tsv")) + sorted(p.name for p in root.glob("*.csv")
                                                                if p.name != "triage_cache.csv")
    return {"files": files}


@app.get("/api/candidates/local/{name}")
def api_candidates_local_file(name: str):
    from pathlib import Path
    from fastapi.responses import PlainTextResponse
    p = Path("data/candidates") / Path(name).name
    if not p.exists() or p.suffix not in (".tsv", ".csv"):
        return JSONResponse({"error": "not found"}, status_code=404)
    return PlainTextResponse(p.read_text())


# ────────────────────────────────────────────────────────────────────────
# sequence viewer (AlphaFold-DB-style sequence strip + feature tracks + PAE)
# ────────────────────────────────────────────────────────────────────────

@app.get("/api/sequence-track")
def api_sequence_track():
    from PhiEx.analysis.seqtrack import build_track
    from PhiEx.analysis.site import detect_catalytic_sites
    protein: Protein = SESSION.get("protein")
    if not protein or not protein.residues:
        return JSONResponse({"error": "load a structure first (stage 1)"}, status_code=400)
    try:
        hits = detect_catalytic_sites(protein)
    except Exception:
        hits = []
    return build_track(protein, SESSION.get("uniprot") or None, SESSION.get("site"),
                       SESSION.get("active_pocket"), SESSION.get("esm2_pred"),
                       SESSION.get("pose_contacts"), hits)


@app.get("/api/pae")
def api_pae():
    from PhiEx.analysis.seqtrack import fetch_pae
    protein: Protein = SESSION.get("protein")
    acc = SESSION.get("uniprot")
    if not protein or not (protein.name or "").upper().startswith(("AF:", "AFILL:")):
        return JSONResponse({"error": "PAE exists only for AlphaFold / AlphaFill models"}, status_code=404)
    pae = fetch_pae(acc)
    if pae is None:
        return JSONResponse({"error": f"no PAE available for {acc}"}, status_code=404)
    return pae


class DynamicsRequest(BaseModel):
    total_ps: float = 10.0
    use_mace: bool = True


@app.post("/api/dynamics")
def api_dynamics(req: DynamicsRequest):
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    best = SESSION.get("best_pose")
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    ligs = (best.ligand,) if best else ()
    site = SESSION.get("site")
    state = ComplexState(protein=protein, cofactors=tuple(cofactors), ligands=ligs,
                         metadata={"site_center": site.center} if site else {})

    sim = OpenMMSimulator(total_ps=req.total_ps)
    traj = sim.run(state, label="api-dynamics")
    SESSION["traj"] = traj

    out = {
        "n_frames": len(traj.frames),
        "total_ps": traj.frames[-1].time_ps if traj.frames else 0.0,
        "energies_ev": [f.energy_ev for f in traj.frames],
    }
    if req.use_mace:
        mace = MACEAdapter(); THEORIES.register(mace)
        region = select_active_site_region(state, max_atoms=50)
        pred = mace.predict(state)
        out["mace"] = {
            "region_atoms": len(region),
            "energy_ev": float(pred.value),
            "energy_sd": float(pred.uncertainty or 0.0),
            "stub": bool(pred.meta.get("stub", False)),
        }

    # analysis
    from PhiEx.analysis import (ligand_residue_contacts, contact_frequency,
                                      rmsf_per_residue, residence_time)
    contacts = ligand_residue_contacts(traj)
    out["top_contacts"] = contact_frequency(contacts, top_k=10)
    out["rmsf"] = rmsf_per_residue(traj)
    res = None
    if site is not None and traj.metadata.get("ligand_simulated", True):
        res = residence_time(traj, site.center, cutoff_a=6.0)
    out["residence_ps_near_site"] = res
    out["residence_ps_near_heme_fe"] = res          # legacy key
    out["ligand_simulated"] = bool(traj.metadata.get("ligand_simulated", True))
    out["site"] = site.to_json() if site else None
    return out


class ALRequest(BaseModel):
    candidates: list[dict] | None = None
    n_iterations: int = 1
    acquisition: str = "ucb"


@app.post("/api/al/iterate")
def api_al(req: ALRequest):
    protein: Protein = SESSION.get("protein")
    pockets = SESSION.get("pockets") or []
    if not protein or not pockets:
        return JSONResponse({"error": "run /api/structure and /api/pockets first"}, status_code=400)
    active = next((p for p in pockets if p.is_active_site), pockets[0])
    site = SESSION.get("site")
    chain_res, chain_seq = working_chain(protein)
    ranked = (rank_site_residues(protein, active, site, limit=10) if site
              else list(active.nearby_residues)[:10])
    candidates = req.candidates or generate_mutation_panel(
        ranked, protein.sequence, ["A", "K"],
        residue_aa={r.index: _AA1.get(r.name.upper(), "X") for r in chain_res})

    surrogate = SESSION.get("surrogate") or GBRSurrogate(
        sequence_length=max(1, len(chain_seq)))
    THEORIES.register(surrogate)
    SESSION["surrogate"] = surrogate

    # evaluator: ESM-2 zero-shot mutation effect when stage 2 ran the real
    # model; otherwise the labelled synthetic heuristic
    best = SESSION.get("best_pose")
    docked = ComplexState(protein=protein, cofactors=tuple(SESSION.get("cofactors") or []),
                          ligands=(best.ligand,) if best else (),
                          metadata={"site_center": site.center} if site else {})
    evaluator, scoring = build_mutation_evaluator(SESSION.get("esm2_pred"), chain_res,
                                                  MACEAdapter(), docked)
    try:
        acquisition = make_acquisition(req.acquisition)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    loop = SESSION.get("al_loop")
    if loop is None or not getattr(surrogate, "_trained", False):
        log("t", f"AL objective: {scoring}")
        loop = ActiveLearningLoop(surrogate=surrogate, evaluator=evaluator,
                                   acquisition=acquisition)
        loop.seed(candidates[:5])
        loop.set_candidates(candidates[5:])
        SESSION["al_loop"] = loop
    elif loop.acquisition.name != acquisition.name:
        loop.acquisition = acquisition

    history: list[dict] = []
    for _ in range(req.n_iterations):
        if not loop.candidates: break
        r = loop.iterate()
        history.append(r.to_json())
    return {"history": history,
            "scoring": scoring,
            "remaining_candidates": len(loop.candidates),
            "iteration": loop.iteration}


# ────────────────────────────────────────────────────────────────────────
# whole pipeline (also streamed)
# ────────────────────────────────────────────────────────────────────────

class APXRunRequest(BaseModel):
    pdb_id: str = APX_DEFAULT_PDB
    uniprot: str | None = None        # None: taken from AF:<acc>, else APX default
    anchor: str | None = None         # e.g. "S197,H351"; None = auto-detect
    ligand_smiles: str | None = None  # None = default for the site type
    ligand_name: str = ""
    md_total_ps: float = 10.0
    al_iterations: int = 3
    use_diffdock: bool = False
    stream: bool = False


@app.post("/api/apx/run")
async def api_apx_run(req: APXRunRequest):
    uniprot = req.uniprot
    if not uniprot and ":" in req.pdb_id:
        uniprot = req.pdb_id.split(":", 1)[1]          # AF:A7F474 → A7F474
    pipeline = APXPipeline(
        pdb_id=req.pdb_id,
        **({"uniprot": uniprot} if uniprot else {}),
        anchor=(req.anchor or None),
        ligand_smiles=(req.ligand_smiles or None),
        ligand_name=req.ligand_name,
        md_total_ps=req.md_total_ps,
        al_iterations=req.al_iterations,
        use_diffdock=req.use_diffdock,
    )
    if not req.stream:
        # run on a worker thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pipeline.run)
        return _result_to_json(result)

    async def gen():
        # interleave the ticker stream with a final result event.  Wait on
        # the next ticker event and the pipeline future together: the last
        # event is usually logged *before* the future resolves, so waiting
        # on events alone would block forever once the pipeline goes quiet.
        loop = asyncio.get_running_loop()
        events = TICKER.subscribe(replay=False)
        next_ev = asyncio.ensure_future(events.__anext__())
        future = loop.run_in_executor(None, pipeline.run)
        try:
            while True:
                done, _ = await asyncio.wait({next_ev, future},
                                             return_when=asyncio.FIRST_COMPLETED)
                if next_ev in done:
                    yield f"data: {next_ev.result().to_json()}\n\n"
                    next_ev = asyncio.ensure_future(events.__anext__())
                elif future in done:
                    break
            result = await future
            payload = json.dumps({"event": "done", "result": _result_to_json(result)})
            yield f"data: {payload}\n\n"
        finally:
            # cancelling the pending __anext__ unwinds the subscription (its
            # finally unregisters the queue); wait for that before closing
            next_ev.cancel()
            try:
                await next_ev
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            await events.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────

def _protein_summary(p: Protein) -> dict:
    return {
        "name": p.name,
        "n_residues": p.n_residues,
        "sequence": p.sequence[:1500],
        "source": p.source,
        "plddt_per_residue": list(p.plddt) if p.plddt else None,
    }


def _pocket_json(p: Pocket) -> dict:
    return {
        "id": p.id, "center": list(p.center), "score": p.score,
        "is_active_site": p.is_active_site,
        "nearby_residues": list(p.nearby_residues),
    }


def _pose_json(p) -> dict:
    return {
        "pose_id": p.pose_id, "score": p.score,
        "stub": bool(p.ligand.metadata.get("stub")),
        "crystal_rmsd": p.ligand.metadata.get("crystal_rmsd"),
        "score_uncertainty": p.score_uncertainty,
        "rmsd_to_best": p.rmsd_to_best,
        "ligand": {"name": p.ligand.name, "smiles": p.ligand.smiles,
                   "atoms": [{"name": a.name, "element": a.element,
                              "xyz": list(a.xyz)} for a in p.ligand.atoms]},
    }


def _result_to_json(r) -> dict:
    from dataclasses import asdict
    out = asdict(r)
    # contacts/rmsf carry int keys → strings for JSON
    out["contacts"] = {str(k): v for k, v in out["contacts"].items()}
    out["rmsf"] = {str(k): v for k, v in out["rmsf"].items()}
    out["residue_labels"] = {str(k): v for k, v in out["residue_labels"].items()}
    return out
