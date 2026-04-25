"""
app.main — FastAPI backend for the Phytologue Sandbox.

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

from phytologue.ticker import TICKER, log
from phytologue.device import select_device, describe_device, select_openmm_platform
from phytologue.core.theory import THEORIES
from phytologue.pipeline.apx import APXPipeline, APX_DEFAULT_PDB

# adapters used by the per-stage routes
from phytologue.adapters.pdb import fetch_protein, parse_pdb_text
from phytologue.adapters.esmfold import ESMFoldAdapter, ca_rmsd
from phytologue.adapters.esm2 import ESM2Adapter, attention_residue_importance
from phytologue.adapters.foldseek import FoldseekAdapter, aggregate_go_terms
from phytologue.adapters.p2rank import P2RankAdapter
from phytologue.adapters.vina import VinaAdapter, ASCORBATE
from phytologue.adapters.diffdock import DiffDockAdapter
from phytologue.adapters.openmm_calc import OpenMMSimulator
from phytologue.adapters.mace import MACEAdapter, select_active_site_region
from phytologue.adapters.surrogate import GBRSurrogate, encode_perturbation
from phytologue.al import ActiveLearningLoop, UCBAcquisition, generate_mutation_panel

from phytologue.core.state import (Atom, Residue, Protein, Cofactor,
                                   Ligand, ComplexState, Pocket)


app = FastAPI(title="Phytologue Sandbox v0",
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
        "<h1>Phytologue Sandbox v0</h1><p>Frontend not built (static/index.html missing).</p>",
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
# ────────────────────────────────────────────────────────────────────────

class StructureRequest(BaseModel):
    pdb_id: str = APX_DEFAULT_PDB
    uniprot: str = "P48534"
    run_esmfold: bool = True


@app.post("/api/structure")
async def api_structure(req: StructureRequest):
    protein, cofactors, ann = fetch_protein(req.pdb_id, req.uniprot)
    SESSION["protein"] = protein
    SESSION["cofactors"] = cofactors
    SESSION["annotations"] = ann
    out: dict[str, Any] = {
        "experimental": _protein_summary(protein),
        "cofactors": [{"name": c.name, "resname": c.resname,
                       "metal": list(c.metal_atom.xyz) if c.metal_atom else None}
                      for c in cofactors],
        "annotations": ann,
    }
    if req.run_esmfold and protein.sequence:
        esmfold = ESMFoldAdapter()
        if protein.residues:
            esmfold.set_reference(protein)
        THEORIES.register(esmfold)
        pred = esmfold.predict(protein.sequence)
        SESSION["predicted"] = pred.value
        out["predicted"] = _protein_summary(pred.value)
        out["predicted"]["mean_plddt"] = pred.meta.get("mean_plddt")
        out["predicted"]["plddt_sd"] = pred.uncertainty
        out["predicted"]["pdb_text"] = pred.value.pdb_text
        if protein.residues and pred.value.residues:
            out["rmsd_pred_vs_exp"] = ca_rmsd(pred.value, protein)
    out["experimental"]["pdb_text"] = protein.pdb_text
    return out


class AnnotateRequest(BaseModel):
    use_session_protein: bool = True


@app.post("/api/annotate")
async def api_annotate(req: AnnotateRequest):
    protein: Protein = SESSION.get("protein")
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    esm2 = ESM2Adapter(); THEORIES.register(esm2)
    fs = FoldseekAdapter(); THEORIES.register(fs)
    pred = esm2.predict(protein.sequence) if protein.sequence else None
    fs_pred = fs.predict(protein, k=5)
    SESSION["esm2_pred"] = pred
    SESSION["foldseek"] = fs_pred
    return {
        "esm2_embedding_sd": float(pred.uncertainty) if pred else None,
        "foldseek_hits": fs_pred.value.get("hits", []),
        "go_aggregate": aggregate_go_terms(fs_pred),
        "go_uncertainty_log10": float(fs_pred.uncertainty),
    }


@app.post("/api/pockets")
async def api_pockets():
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    state = ComplexState(protein=protein, cofactors=tuple(cofactors))
    p2 = P2RankAdapter(); THEORIES.register(p2)
    pred = p2.predict(state)
    SESSION["pockets"] = pred.value
    return {
        "pockets": [_pocket_json(p) for p in pred.value],
        "score_sd": float(pred.uncertainty),
        "active_site_id": next((p.id for p in pred.value if p.is_active_site), None),
    }


class DockRequest(BaseModel):
    pocket_id: str | None = None
    method: str = "vina"          # or "diffdock"
    smiles: str | None = None     # default: ascorbate
    ligand_name: str = "ascorbate"


@app.post("/api/dock")
async def api_dock(req: DockRequest):
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    pockets: list[Pocket] = SESSION.get("pockets") or []
    if not protein or not pockets:
        return JSONResponse({"error": "run /api/structure and /api/pockets first"}, status_code=400)

    pocket = (next((p for p in pockets if p.id == req.pocket_id), None)
              if req.pocket_id else next((p for p in pockets if p.is_active_site), pockets[0]))
    if pocket is None:
        return JSONResponse({"error": "no pocket selected"}, status_code=400)

    ligand = ASCORBATE if not req.smiles else Ligand(name=req.ligand_name, smiles=req.smiles)
    state = ComplexState(protein=protein, cofactors=tuple(cofactors), ligands=(ligand,))

    if req.method == "diffdock":
        dd = DiffDockAdapter(); THEORIES.register(dd)
        poses = dd.predict(state, ligand, pocket).value
    else:
        poses = VinaAdapter().dock(state, ligand, pocket)
    SESSION["best_pose"] = poses[0] if poses else None
    return {"method": req.method,
            "n_poses": len(poses),
            "poses": [_pose_json(p) for p in poses]}


class DynamicsRequest(BaseModel):
    total_ps: float = 10.0
    use_mace: bool = True


@app.post("/api/dynamics")
async def api_dynamics(req: DynamicsRequest):
    protein: Protein = SESSION.get("protein")
    cofactors = SESSION.get("cofactors") or []
    best = SESSION.get("best_pose")
    if not protein:
        return JSONResponse({"error": "call /api/structure first"}, status_code=400)
    ligs = (best.ligand,) if best else ()
    state = ComplexState(protein=protein, cofactors=tuple(cofactors), ligands=ligs)

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
    from phytologue.analysis import (ligand_residue_contacts, contact_frequency,
                                      rmsf_per_residue, residence_time)
    contacts = ligand_residue_contacts(traj)
    out["top_contacts"] = contact_frequency(contacts, top_k=10)
    out["rmsf"] = rmsf_per_residue(traj)
    metal = next((c.metal_atom for c in cofactors if c.metal_atom), None)
    out["residence_ps_near_heme_fe"] = (
        residence_time(traj, metal.xyz, cutoff_a=6.0) if metal else None)
    return out


class ALRequest(BaseModel):
    candidates: list[dict] | None = None
    n_iterations: int = 1
    acquisition: str = "ucb"


@app.post("/api/al/iterate")
async def api_al(req: ALRequest):
    protein: Protein = SESSION.get("protein")
    pockets = SESSION.get("pockets") or []
    if not protein or not pockets:
        return JSONResponse({"error": "run /api/structure and /api/pockets first"}, status_code=400)
    active = next((p for p in pockets if p.is_active_site), pockets[0])
    candidates = req.candidates or generate_mutation_panel(
        list(active.nearby_residues)[:8], protein.sequence, ["A", "K"])

    surrogate = SESSION.get("surrogate") or GBRSurrogate(
        sequence_length=max(1, len(protein.sequence)))
    THEORIES.register(surrogate)
    SESSION["surrogate"] = surrogate

    # evaluator: cheap synthetic for the API path; real evaluator is wired
    # by run_apx_pipeline.  README documents the swap.
    from phytologue.runners.batch import synthetic_evaluator
    loop = SESSION.get("al_loop")
    if loop is None or not getattr(surrogate, "_trained", False):
        loop = ActiveLearningLoop(surrogate=surrogate, evaluator=synthetic_evaluator,
                                   acquisition=UCBAcquisition())
        loop.seed(candidates[:5])
        loop.set_candidates(candidates[5:])
        SESSION["al_loop"] = loop

    history: list[dict] = []
    for _ in range(req.n_iterations):
        if not loop.candidates: break
        r = loop.iterate()
        loop.candidates = [c for c in loop.candidates if c != r.chosen]
        history.append({
            "iteration": r.iteration, "chosen": r.chosen,
            "predicted": r.predicted, "predicted_sd": r.predicted_sd,
            "observed": r.observed, "observed_sd": r.observed_sd,
            "rmse": r.surrogate_rmse, "r2": r.surrogate_r2,
        })
    return {"history": history,
            "remaining_candidates": len(loop.candidates),
            "iteration": loop.iteration}


# ────────────────────────────────────────────────────────────────────────
# whole pipeline (also streamed)
# ────────────────────────────────────────────────────────────────────────

class APXRunRequest(BaseModel):
    pdb_id: str = APX_DEFAULT_PDB
    md_total_ps: float = 10.0
    al_iterations: int = 3
    use_diffdock: bool = False
    stream: bool = False


@app.post("/api/apx/run")
async def api_apx_run(req: APXRunRequest):
    pipeline = APXPipeline(
        pdb_id=req.pdb_id,
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
        # interleave the ticker stream with a final result event.
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, pipeline.run)
        async for ev in TICKER.subscribe(replay=False):
            yield f"data: {ev.to_json()}\n\n"
            if future.done():
                break
        result = await future
        payload = json.dumps({"event": "done", "result": _result_to_json(result)})
        yield f"data: {payload}\n\n"

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
    return out
