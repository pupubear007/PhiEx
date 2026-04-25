// ─────────────────────────────────────────────────────────────────────────
// PhiEx Sandbox v0 — frontend.
//
// Visual vocabulary preserved from the prototype:
//     ϕ / ∃ / s / tϕ / i  symbols labelling pipeline stages
//     blue (ϕ deductive), red (s∃ ⟳ tϕ inductive), accent yellow (theory),
//     dark background, monospace, reasoning ticker at the bottom.
//
// New for v0:
//     Mol* viewer for predicted (top, blue/pLDDT) and experimental (bottom),
//     uncertainty bars on every ML readout,
//     active-learning acquisition heatmap on residues,
//     stub-tag for any adapter that's running its synthetic fallback.
// ─────────────────────────────────────────────────────────────────────────

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const TICKER = $("#ticker");

function logLine(ev){
  const line = document.createElement("div");
  line.className = "log-line " + (ev.tag || "sys");
  const t = ev.t ? new Date(ev.t * 1000).toISOString().slice(11, 19) : "";
  const k = symbolFor(ev.tag);
  line.innerHTML = `<span class="t">${t}</span><span class="k">${k}</span><span class="m">${escapeHtml(ev.msg)}</span>`;
  TICKER.appendChild(line);
  TICKER.scrollTop = TICKER.scrollHeight;
}

function symbolFor(tag){
  switch(tag){
    case "phi":   return "ϕ→∃";
    case "exist": return " ∃ ";
    case "s":     return " s∃";
    case "t":     return " tϕ";
    case "i":     return " i ";
    default:      return "···";
  }
}
function escapeHtml(s){
  return String(s||"").replace(/[&<>"]/g, ch => (
    {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[ch]
  ));
}

// ─────────────────────────────────────────────────────────────────────────
// SSE — live ticker
// ─────────────────────────────────────────────────────────────────────────
let sse = null;
function connectTicker(){
  try { if (sse) sse.close(); } catch(_) {}
  sse = new EventSource("/events");
  sse.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.event === "done") {
        renderResult(msg.result);
        return;
      }
      logLine(msg);
    } catch (err) {
      console.warn("ticker parse error", err);
    }
  };
  sse.onerror = () => { /* let browser auto-reconnect */ };
}
connectTicker();

// ─────────────────────────────────────────────────────────────────────────
// Mol* viewers — top: predicted, bottom: experimental
//
// We use loadStructureFromData (string in, no Blob URL round-trip) because
// loadStructureFromUrl with an object-URL silently fails on some Mol*
// builds.  Every step logs to console and surfaces a sys-line in the
// on-screen ticker so you can tell from the UI whether structures arrived.
// ─────────────────────────────────────────────────────────────────────────
let predViewer = null, expViewer = null;
let viewersReady = null;          // resolves when both viewers are alive

function logSys(msg){
  logLine({tag: "sys", t: Date.now()/1000, msg});
}

async function initViewers(){
  if (typeof molstar === "undefined") {
    console.error("[viewer] Mol* global not present; CDN failed to load?");
    logSys("Mol* failed to load — 3D panes will be empty");
    return;
  }
  console.log("[viewer] Mol* version:", molstar.PLUGIN_VERSION || "(unknown)");
  const dark = {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    layoutShowRemoteState: false,
    viewportShowExpand: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
  };
  try {
    predViewer = await molstar.Viewer.create("viewer-pred", dark);
    expViewer  = await molstar.Viewer.create("viewer-exp",  dark);
    console.log("[viewer] both Mol* viewers initialised");
  } catch (e) {
    console.error("[viewer] Mol* init failed", e);
    logSys("Mol* init failed: " + (e.message || e));
  }
}
viewersReady = initViewers();

// Per-viewer load lock — without this, two consecutive runs race:
// run #1's clear() can be reordered relative to run #2's load() and
// the previous structure stays in the viewport, causing overlap.
const _loadLocks = new Map();   // keyed by container id (predViewer/expViewer get replaced)
function _withLock(key, fn){
  const prev = _loadLocks.get(key) || Promise.resolve();
  const next = prev.then(fn, fn);     // run regardless of prior outcome
  _loadLocks.set(key, next.catch(()=>{}));
  return next;
}

// Mol*'s viewer.clear() is unreliable across builds — sometimes it
// leaves structures resident.  The robust way to start fresh is to
// destroy the viewer and recreate it.  We do this for both panes.
async function _recreateViewer(containerId){
  if (typeof molstar === "undefined") return null;
  const dark = {
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    layoutShowRemoteState: false,
    viewportShowExpand: false,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
  };
  // Wipe the DOM children of the container so Mol* can mount cleanly.
  const el = document.getElementById(containerId);
  if (el) {
    // Tell any existing viewer to clean up.
    try {
      const cur = containerId === "viewer-pred" ? predViewer : expViewer;
      if (cur && cur.dispose) cur.dispose();
      else if (cur && cur.plugin && cur.plugin.dispose) cur.plugin.dispose();
    } catch (e) { console.warn("[viewer] dispose:", e); }
    el.innerHTML = "";
  }
  try {
    const v = await molstar.Viewer.create(containerId, dark);
    console.log(`[viewer] ${containerId} recreated`);
    return v;
  } catch (e) {
    console.error(`[viewer] recreate ${containerId} failed:`, e);
    return null;
  }
}

async function clearAllViewers(){
  if (viewersReady) await viewersReady;
  // Wait for any in-flight loads to finish before tearing down.
  await Promise.all([
    _loadLocks.get("viewer-pred") || Promise.resolve(),
    _loadLocks.get("viewer-exp")  || Promise.resolve(),
  ]);
  console.log("[viewer] clearing both viewers (destroy + recreate)");
  predViewer = await _recreateViewer("viewer-pred");
  expViewer  = await _recreateViewer("viewer-exp");
  // Reset the load locks so future loads don't queue behind stale promises.
  _loadLocks.delete("viewer-pred");
  _loadLocks.delete("viewer-exp");
  logSys("3D viewers cleared");
}

// loadStructureFromData accepts a raw PDB string directly. We hold a
// per-container lock so successive runs don't interleave clear/load and
// stack structures on top of each other.  The viewer reference can be
// replaced (by clearAllViewers) so we resolve it from the container id
// at execution time, not capture time.
async function loadPdbInto(containerId, pdbText, opts={}){
  if (!pdbText) {
    console.warn("[viewer] empty pdb_text — nothing to render");
    return;
  }
  // Stash the PDB so the highlight feature can filter it later without
  // having to re-fetch from the server.
  _pdbCache.set(containerId, pdbText);
  return _withLock(containerId, async () => {
    const viewer = (containerId === "viewer-pred") ? predViewer : expViewer;
    if (!viewer) {
      console.warn(`[viewer] ${containerId} not initialised`);
      return;
    }
    console.log(`[viewer] loading ${pdbText.length} bytes of PDB into ${opts.label||containerId}`);
    try {
      // Primary path: loadStructureFromData (string → trajectory, no fetch)
      await viewer.loadStructureFromData(pdbText, "pdb", false);
      console.log(`[viewer] ${opts.label||containerId} loaded OK`);
    } catch (e) {
      console.error(`[viewer] loadStructureFromData failed for ${opts.label||containerId}:`, e);
      // Fallback: blob-URL path (older Mol* builds)
      try {
        const blob = new Blob([pdbText], { type: "text/plain" });
        const url  = URL.createObjectURL(blob);
        await viewer.loadStructureFromUrl(url, "pdb", false);
        console.log(`[viewer] ${opts.label||containerId} loaded via fallback URL`);
      } catch (e2) {
        console.error(`[viewer] fallback also failed for ${opts.label||containerId}:`, e2);
        logSys(`3D load failed (${opts.label||containerId}): ${e2.message || e2}`);
      }
    }
  });
}

// pLDDT colouring: the backend writes pLDDT into the B-factor column
// of the predicted PDB.  We just load the structure; Mol*'s default
// preset already colours by B-factor when values are present, and the
// user can flip presets in the viewport menu if they want.
async function loadPredictedWithPlddt(pdbText){
  return loadPdbInto("viewer-pred", pdbText, {label: "predicted"});
}

// ─────────────────────────────────────────────────────────────────────────
// device tag
// ─────────────────────────────────────────────────────────────────────────
async function refreshDevice(){
  try {
    const r = await fetch("/api/device").then(r => r.json());
    $("#device-tag").innerHTML =
      `device <b>${r.device}</b> · backend <b>${r.info && r.info.backend || "—"}</b> · openmm <b>${r.openmm_platform}</b>`;
  } catch(_) {}
}
refreshDevice();
setInterval(refreshTheories, 4000);

async function refreshTheories(){
  try {
    const r = await fetch("/api/theories").then(r => r.json());
    if (!r.theories || !r.theories.length) {
      $("#theories-readout").innerHTML = "none loaded";
      return;
    }
    $("#theories-readout").innerHTML = r.theories.map(t => {
      const stub = t.stub ? '<span class="stub-tag">STUB</span>' : "";
      return `<div><b>${t.name}</b> <span style="color:var(--ink-dim)">on ${t.device}</span> ${stub}</div>
              <div style="color:var(--ink-dim);font-size:10px">${t.backend||""}</div>`;
    }).join("");
  } catch(_) {}
}

// ─────────────────────────────────────────────────────────────────────────
// stage actions
// ─────────────────────────────────────────────────────────────────────────
async function postJSON(url, body={}){
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
                              body: JSON.stringify(body)});
  if (!r.ok) {
    let err = null;
    try { err = await r.json(); } catch(_) {}
    throw new Error(err && err.error ? err.error : `HTTP ${r.status}`);
  }
  return r.json();
}

let lastStruct = null, lastPockets = null, lastPoses = null, lastDyn = null;
let lastAL = [];
let lastDiscoveries = [];   // disagreement residues from ESM-2 cross-check

// Cache the *raw PDB text* of whatever's currently in each viewer.  The
// highlight feature filters this down to a subset and loads the result as
// a SECOND structure on top — which dodges Mol*'s notoriously fragile
// expression-based selection API entirely.
const _pdbCache = new Map();          // containerId → pdb text
// Per-viewer cleanup handles for the highlight overlay.
const _highlightCells = new Map();    // containerId → array of state cells / refs

$("#run-structure").addEventListener("click", async () => {
  const pdb = $("#i-pdb").value.trim();
  const uni = $("#i-uniprot").value.trim();
  try {
    const r = await postJSON("/api/structure", {pdb_id: pdb, uniprot: uni, run_esmfold: true});
    lastStruct = r;
    if (r.predicted && r.predicted.pdb_text) {
      await loadPredictedWithPlddt(r.predicted.pdb_text);
      $("#v-plddt").textContent = (r.predicted.mean_plddt||0).toFixed(1)
                                + " ± " + (r.predicted.plddt_sd||0).toFixed(1);
    }
    if (r.experimental && r.experimental.pdb_text)
      await loadPdbInto("viewer-exp", r.experimental.pdb_text, {label: "experimental"});
    if (r.rmsd_pred_vs_exp != null)
      $("#v-rmsd").textContent = r.rmsd_pred_vs_exp.toFixed(2);
    $("#state-readout").innerHTML =
      `protein: <b>${r.experimental.name||"—"}</b><br>` +
      `cofactors: <b>${r.cofactors.map(c=>c.name).join(", ")||"—"}</b><br>` +
      `active site: <b>—</b>`;
  } catch(e){ alert("structure: "+e.message); }
});

$("#run-annotate").addEventListener("click", async () => {
  try {
    const r = await postJSON("/api/annotate", {});
    $("#v-emb").textContent = (r.esm2_embedding_sd||0).toFixed(3);
    const top = (r.go_aggregate || []).slice(0,5).map(([g,n]) =>
      `<div>${escapeHtml(g)} <span style="color:var(--ink-dim)">×${n}</span></div>`).join("");
    $("#phi0-readout").innerHTML =
      `<b>${(r.foldseek_hits||[]).length}</b> structural neighbours<br>` +
      `log10(E) σ: <b>${(r.go_uncertainty_log10||0).toFixed(2)}</b>` +
      `<div style="margin-top:6px">${top}</div>`;
  } catch(e){ alert("annotate: "+e.message); }
});

$("#run-pockets").addEventListener("click", async () => {
  try {
    const r = await postJSON("/api/pockets", {});
    lastPockets = r.pockets;
    const html = (r.pockets||[]).map(p =>
      `<div>${p.is_active_site?'<span style="color:var(--accent)">★ </span>':''}<b>${p.id}</b> score ${p.score.toFixed(2)} · ${p.nearby_residues.length} residues</div>`
    ).join("") || "no pockets";
    $("#pockets-readout").innerHTML = html + `<div style="color:var(--ink-dim);font-size:10px">σ ${(r.score_sd||0).toFixed(2)}</div>`;
    if (r.active_site_id)
      $("#state-readout").innerHTML = $("#state-readout").innerHTML.replace(/active site:.*/, `active site: <b>${r.active_site_id}</b>`);
  } catch(e){ alert("pockets: "+e.message); }
});

$("#run-dock").addEventListener("click", async () => {
  try {
    const r = await postJSON("/api/dock", {method: "vina"});
    lastPoses = r.poses;
    drawPoses(r.poses);
    $("#poses-readout").innerHTML =
      `<b>${r.n_poses}</b> poses · best <b>${(r.poses[0]||{}).score?.toFixed(2)}</b> ± <b>${((r.poses[0]||{}).score_uncertainty||0).toFixed(2)}</b> kcal/mol`;
  } catch(e){ alert("dock: "+e.message); }
});

$("#run-dynamics").addEventListener("click", async () => {
  try {
    const ps = parseFloat($("#i-ps").value || "10");
    const r = await postJSON("/api/dynamics", {total_ps: ps, use_mace: true});
    lastDyn = r;
    drawEnergies(r.energies_ev);
    $("#v-mace-n").textContent  = r.mace ? r.mace.region_atoms : "—";
    $("#v-mace").textContent    = r.mace ? r.mace.energy_ev.toFixed(3) : "—";
    $("#v-mace-sd").textContent = r.mace ? r.mace.energy_sd.toFixed(3) : "—";
    $("#v-res").textContent     = r.residence_ps_near_heme_fe != null
                                  ? r.residence_ps_near_heme_fe.toFixed(2) : "—";
  } catch(e){ alert("dynamics: "+e.message); }
});

$("#run-al").addEventListener("click", async () => {
  try {
    const n = parseInt($("#i-iter").value || "1", 10);
    const r = await postJSON("/api/al/iterate", {n_iterations: n});
    lastAL = lastAL.concat(r.history || []);
    drawALHeatmap(lastAL);
    const last = lastAL[lastAL.length-1] || {};
    $("#al-readout").innerHTML =
      `iteration <b>${last.iteration||"—"}</b><br>` +
      `tϕ suggests: <b>${renderPert(last.chosen)}</b><br>` +
      `predicted Δ = <b>${(last.predicted||0).toFixed(3)}</b> ± <b>${(last.predicted_sd||0).toFixed(3)}</b><br>` +
      `observed = <b>${(last.observed||0).toFixed(3)}</b> ± <b>${(last.observed_sd||0).toFixed(3)}</b><br>` +
      `RMSE <b>${last.rmse!=null?last.rmse.toFixed(3):"—"}</b> · R² <b>${last.r2!=null?last.r2.toFixed(3):"—"}</b>`;
  } catch(e){ alert("active-learning: "+e.message); }
});

// Find PDB(s) for a UniProt ID. Auto-fills the PDB field with the best
// match (highest-resolution X-ray) and lists alternates underneath.
$("#find-pdb").addEventListener("click", async () => {
  const uni = $("#i-uniprot").value.trim();
  const out = $("#pdb-candidates");
  if (!uni) { out.innerHTML = "enter a UniProt ID first"; return; }
  out.innerHTML = "looking up…";
  try {
    const r = await fetch(`/api/uniprot/${encodeURIComponent(uni)}/pdbs`).then(r => r.json());
    if (!r.entries || !r.entries.length) {
      out.innerHTML = `<span style="color:var(--s)">no PDB entries for ${uni} — try ESMFold-only (leave PDB blank or use any 4-letter placeholder; the predicted structure will still render)</span>`;
      return;
    }
    // Auto-fill the PDB field with the best (first) entry
    $("#i-pdb").value = r.entries[0].pdb_id;
    // Render the top 10 candidates so the user can pick a different one.
    // Model entries (SWISS-MODEL, AlphaFold) are tagged in a distinct colour
    // so it's obvious they're predicted, not experimental.
    const rows = r.entries.slice(0, 10).map(e => {
      const res = e.resolution != null ? e.resolution.toFixed(2) + " Å" : "—";
      const isBest = e === r.entries[0] ? '<b style="color:var(--accent)">★ </b>' : "";
      const isModel = /SWISS|AlphaFold|Modeling|Predicted/i.test(e.method || "")
                      || /^(SWISS|AF):/.test(e.pdb_id || "");
      const idColor = isModel ? "var(--stub)" : "var(--ink)";
      return `<div style="cursor:pointer;padding:2px 0" data-pdb="${e.pdb_id}">
        ${isBest}<b style="color:${idColor}">${e.pdb_id}</b>
        <span> · ${e.method||"—"} · ${res} · ${e.chains||""}</span>
      </div>`;
    }).join("");
    out.innerHTML = `<div style="margin-bottom:4px;color:var(--phi)">${r.n} PDB entries — click to pick</div>${rows}`;
    // Click-to-pick on the alternates
    out.querySelectorAll("[data-pdb]").forEach(el => {
      el.addEventListener("click", () => {
        $("#i-pdb").value = el.getAttribute("data-pdb");
        out.querySelectorAll("[data-pdb]").forEach(x =>
          x.style.background = x === el ? "var(--rule)" : "");
      });
    });
  } catch (e) {
    out.innerHTML = `<span style="color:var(--s)">lookup failed: ${e.message||e}</span>`;
  }
});

// Manual reset button — wipes EVERYTHING:
//   * both 3D viewers (destroy + recreate)
//   * all right-hand readouts (pockets, poses, MD, AL, cross-check)
//   * the bottom ticker
//   * frontend in-memory caches (lastStruct, lastPockets, lastPoses, lastDyn, lastAL)
//   * inline charts (poses bars, energy curve, AL heatmap)
//   * backend session state (POST /api/reset clears protein, cofactors,
//     pockets, traj, surrogate, AL loop, learned-models registry, ticker history)
async function resetAll(){
  // 1. backend
  try {
    await fetch("/api/reset", {method: "POST"});
    console.log("[reset] backend session cleared");
  } catch (e) {
    console.warn("[reset] /api/reset failed:", e);
  }

  // 2. 3D viewers
  await clearAllViewers();

  // 3. frontend caches
  lastStruct = null; lastPockets = null; lastPoses = null; lastDyn = null;
  lastAL = [];
  lastDiscoveries = [];
  _highlightCells.clear();
  _pdbCache.clear();

  // 4. side-panel readouts — restore initial empty messages
  $("#phi0-readout").innerHTML       = "no annotations yet — run stage 2";
  $("#state-readout").innerHTML      = "protein: <b>—</b><br>cofactors: <b>—</b><br>active site: <b>—</b>";
  $("#theories-readout").innerHTML   = "none loaded";
  $("#pockets-readout").innerHTML    = "no pockets yet";
  $("#poses-readout").innerHTML      = "no poses yet";
  $("#md-readout").innerHTML         = "MACE region: <b id=\"v-mace-n\">—</b> atoms<br>" +
                                        "MACE E: <b id=\"v-mace\">—</b> ± <b id=\"v-mace-sd\">—</b> eV<br>" +
                                        "residence near heme Fe: <b id=\"v-res\">—</b> ps";
  $("#al-readout").innerHTML         = "no iterations yet";
  $("#cross-readout").innerHTML      = "run AL first to populate";
  $("#struct-readout").innerHTML     = "mean pLDDT: <b id=\"v-plddt\">—</b><br>" +
                                        "Cα RMSD vs experimental: <b id=\"v-rmsd\">—</b> Å<br>" +
                                        "ESM-2 embed σ: <b id=\"v-emb\">—</b>";

  // 5. clear the inline canvases
  ["#poses-chart", "#energy-chart", "#al-heatmap"].forEach(sel => {
    const c = $(sel);
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
  });

  // 6. clear the ticker (keep only the header line)
  TICKER.querySelectorAll(".log-line").forEach(el => el.remove());

  console.log("[reset] full reset complete");
}
$("#clear-3d").addEventListener("click", resetAll);

// ─────────────────────────────────────────────────────────────────────────
// Highlight "discovery flags" — disagreement residues from the ESM-2
// cross-check — in both 3D viewers as red ball-and-stick.
//
// Mol*'s public API for adding a representation on a residue selection
// varies across 4.x point releases.  We use the most stable path:
//   1. grab the structure cell that's already loaded
//   2. build a MolScript expression selecting whole residues whose
//      auth_seq_id is in the discovery set
//   3. call plugin.builders.structure.representation.addRepresentation
//      with type=ball-and-stick and a uniform red color
//
// If anything goes wrong we log to console (for debugging) and surface
// a sys-line in the on-screen ticker so the user knows nothing happened.
// ─────────────────────────────────────────────────────────────────────────
// Filter a PDB text to only ATOM/HETATM lines whose residue sequence id
// is in the given set.  PDB column layout (1-based): cols 23-26 = resSeq.
// We also keep TER and END so the resulting fragment is a valid PDB.
function _filterPdbByResidues(pdbText, residueSet){
  const out = [];
  for (const line of pdbText.split(/\r?\n/)) {
    if (line.startsWith("ATOM") || line.startsWith("HETATM")) {
      const n = parseInt(line.substring(22, 26).trim(), 10);
      if (!isNaN(n) && residueSet.has(n)) out.push(line);
    } else if (line.startsWith("TER") || line.startsWith("END")) {
      // skip — we'll add a single END at the end
    }
  }
  if (!out.length) return null;
  out.push("END");
  return out.join("\n");
}

// Add a "discovery overlay" — a second tiny structure containing only the
// flagged residues, rendered as red ball-and-stick on top of the existing
// cartoon.  This sidesteps Mol*'s expression-based selection API by using
// the well-supported parseTrajectory → createModel → createStructure
// pipeline that's part of plugin.builders.structure.  Works on every
// Mol* 4.x build I've encountered.
async function _addDiscoveryHighlight(containerId, residues){
  const viewer = (containerId === "viewer-pred") ? predViewer : expViewer;
  if (!viewer || !viewer.plugin) {
    console.warn(`[highlight] ${containerId} viewer/plugin missing`);
    return false;
  }
  const plugin = viewer.plugin;
  const original = _pdbCache.get(containerId);
  if (!original) {
    console.warn(`[highlight] no cached PDB for ${containerId} — load a structure first`);
    return false;
  }
  const residueSet = new Set(residues.map(r => parseInt(r, 10)).filter(n => !isNaN(n)));
  if (!residueSet.size) {
    console.warn("[highlight] empty residue set");
    return false;
  }
  const filtered = _filterPdbByResidues(original, residueSet);
  if (!filtered) {
    console.warn(`[highlight] no atoms matched residues on ${containerId}; flagged=`, residues);
    return false;
  }
  console.log(`[highlight] ${containerId}: filtered overlay = ${filtered.split("\n").length} lines`);

  try {
    // Low-level structure-creation pipeline.  Each step returns a state
    // cell whose `.ref` we can later delete to remove the overlay.
    const data = await plugin.builders.data.rawData({
      data: filtered,
      label: "discovery-flags-data",
    });
    const trajectory = await plugin.builders.structure.parseTrajectory(data, "pdb");
    const model      = await plugin.builders.structure.createModel(trajectory);
    const structure  = await plugin.builders.structure.createStructure(model);

    // Attach a red ball-and-stick representation directly to the new
    // structure.  Because the structure ONLY contains the flagged
    // residues, no selection step is needed — every atom in this branch
    // is a discovery atom.
    await plugin.builders.structure.representation.addRepresentation(structure, {
      type: "ball-and-stick",
      color: "uniform",
      colorParams: { value: 0xe05a4a },     // red — matches --s
      sizeParams: { scale: 1.4 },
    });
    // A larger sphere makes it pop out from the cartoon.
    try {
      await plugin.builders.structure.representation.addRepresentation(structure, {
        type: "spacefill",
        color: "uniform",
        colorParams: { value: 0xe05a4a },
        sizeParams: { scale: 0.45 },
      });
    } catch(_) { /* not all builds support spacefill addition this way */ }

    // Track the root data cell — deleting it cascades to all children.
    const list = _highlightCells.get(containerId) || [];
    list.push(data);
    _highlightCells.set(containerId, list);
    console.log(`[highlight] ${containerId}: overlay attached`);
    return true;
  } catch (e) {
    console.error(`[highlight] overlay creation failed on ${containerId}:`, e);
    return false;
  }
}

async function _clearDiscoveryHighlights(containerId){
  const viewer = (containerId === "viewer-pred") ? predViewer : expViewer;
  if (!viewer || !viewer.plugin) return;
  const plugin = viewer.plugin;
  const cells = _highlightCells.get(containerId) || [];
  for (const cell of cells) {
    // The overlay's "data" cell is the root of a parseTrajectory chain.
    // Deleting it from the state tree cascades through all children.
    const ref = (cell && cell.ref) || (cell && cell.transform && cell.transform.ref) || null;
    if (!ref) {
      console.warn(`[highlight] no ref on cached cell for ${containerId}`);
      continue;
    }
    try {
      await plugin.state.data.build().delete(ref).commit();
      console.log(`[highlight] removed overlay ${ref} on ${containerId}`);
    } catch (e) {
      console.warn(`[highlight] state delete failed for ${ref}:`, e);
    }
  }
  _highlightCells.delete(containerId);
}

async function highlightDiscoveryResidues(){
  if (!lastDiscoveries || !lastDiscoveries.length) {
    logSys("no discovery flags yet — run end-to-end first");
    return;
  }
  if (viewersReady) await viewersReady;
  // Toggle: if either viewer already has highlights, clear both first.
  if (_highlightCells.has("viewer-pred") || _highlightCells.has("viewer-exp")) {
    console.log("[highlight] clearing existing highlights");
    await _clearDiscoveryHighlights("viewer-pred");
    await _clearDiscoveryHighlights("viewer-exp");
    logSys("discovery-flag highlights cleared");
    return;
  }
  console.log("[highlight] adding highlights for residues:", lastDiscoveries);
  const okPred = await _addDiscoveryHighlight("viewer-pred", lastDiscoveries);
  const okExp  = await _addDiscoveryHighlight("viewer-exp",  lastDiscoveries);
  if (okPred || okExp) {
    logSys(`★ discovery flags highlighted: R${lastDiscoveries.join(", R")}`);
  } else {
    logSys("highlight failed — see console (Mol* version mismatch?)");
  }
}
$("#highlight-disc").addEventListener("click", highlightDiscoveryResidues);

$("#run-all").addEventListener("click", async () => {
  // Streamed end-to-end run — server emits SSE events, the ticker shows
  // progress, and the final {event: "done", result} payload populates
  // the right-hand panels.
  const body = {
    pdb_id: $("#i-pdb").value.trim(),
    md_total_ps: parseFloat($("#i-ps").value || "10"),
    al_iterations: parseInt($("#i-iter").value || "3", 10),
    use_diffdock: false,
    stream: true,
  };
  // Stream by issuing a POST and reading the body line-by-line.
  const resp = await fetch("/api/apx/run", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream:true});
    let nl;
    while ((nl = buf.indexOf("\n\n")) !== -1) {
      const chunk = buf.slice(0, nl);
      buf = buf.slice(nl + 2);
      const line = chunk.replace(/^data:\s*/, "");
      if (!line) continue;
      try {
        const msg = JSON.parse(line);
        if (msg.event === "done") {
          renderResult(msg.result);
        } else {
          logLine(msg);
        }
      } catch (e) { /* skip non-JSON lines */ }
    }
  }
});

function renderPert(p){
  if (!p) return "—";
  if (p.type === "mutation") return `mutate ${p.from}${p.residue} → ${p.to}`;
  if (p.type === "ligand_variant") return `ligand ${(p.smiles||"").slice(0,20)}`;
  if (p.type === "temperature") return `T = ${p.K} K`;
  return JSON.stringify(p);
}

// ─────────────────────────────────────────────────────────────────────────
// charts (no library — keep prototype's hand-drawn aesthetic)
// ─────────────────────────────────────────────────────────────────────────
function setupCanvas(c){
  const r = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  c.width = r.width*dpr; c.height = r.height*dpr;
  const ctx = c.getContext("2d"); ctx.scale(dpr, dpr);
  return {ctx, w:r.width, h:r.height};
}

function drawPoses(poses){
  const c = $("#poses-chart"); const {ctx, w, h} = setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!poses || !poses.length) return;
  const minS = Math.min(...poses.map(p=>p.score));
  const maxS = Math.max(...poses.map(p=>p.score));
  const span = (maxS - minS) || 1;
  const bw = w / poses.length;
  poses.forEach((p, i) => {
    const v = (p.score - minS) / span;
    const bh = (1 - v) * (h - 20);
    ctx.fillStyle = "rgba(106,169,255,0.6)";
    ctx.fillRect(i*bw+1, h-10-bh, bw-2, bh);
    if (p.score_uncertainty) {
      const eh = Math.min(p.score_uncertainty * 5, 18);
      ctx.fillStyle = "var(--s)";
      ctx.fillRect(i*bw+bw/2-1, h-10-bh-eh, 2, eh*2);
    }
  });
  ctx.strokeStyle = "var(--rule)"; ctx.beginPath();
  ctx.moveTo(0,h-10); ctx.lineTo(w,h-10); ctx.stroke();
}

function drawEnergies(es){
  const c = $("#energy-chart"); const {ctx, w, h} = setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!es || !es.length) return;
  const xs = es.filter(v=>v!=null);
  if (!xs.length) return;
  const min = Math.min(...xs), max = Math.max(...xs), span = (max-min) || 1;
  ctx.strokeStyle = "rgba(106,169,255,0.8)"; ctx.lineWidth = 1.4;
  ctx.beginPath();
  xs.forEach((v, i) => {
    const x = i / Math.max(1,xs.length-1) * w;
    const y = h - 6 - ((v - min) / span) * (h - 12);
    if (i===0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawALHeatmap(history){
  const c = $("#al-heatmap"); const {ctx, w, h} = setupCanvas(c);
  ctx.clearRect(0,0,w,h);
  if (!history || !history.length) return;
  // assemble per-residue magnitude
  const mag = {};
  history.forEach(r => {
    const k = r.chosen && r.chosen.residue;
    if (!k) return;
    mag[k] = Math.max(mag[k]||0, Math.abs(r.observed));
  });
  const keys = Object.keys(mag).map(Number).sort((a,b)=>a-b);
  if (!keys.length) return;
  const maxM = Math.max(...Object.values(mag));
  const tile = w / keys.length;
  keys.forEach((k, i) => {
    const v = mag[k] / maxM;
    const r = Math.floor(224 * v + 30);
    const g = Math.floor(90  * v + 30);
    const b = Math.floor(74  * v + 30);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i*tile, 8, tile-1, h-22);
    ctx.fillStyle = "var(--ink-dim)"; ctx.font = "9px JetBrains Mono, monospace";
    ctx.textAlign = "center";
    ctx.fillText("R"+k, i*tile + tile/2, h-4);
  });
}

// ─────────────────────────────────────────────────────────────────────────
// streamed end-to-end result handler
// ─────────────────────────────────────────────────────────────────────────
function renderResult(r){
  console.log("[result] keys:", Object.keys(r));
  console.log("[result] predicted_pdb_text:",
              r.predicted_pdb_text ? r.predicted_pdb_text.length + " bytes" : "null");
  console.log("[result] experimental_pdb_text:",
              r.experimental_pdb_text ? r.experimental_pdb_text.length + " bytes" : "null");
  logSys(`structures received — predicted: ${r.predicted_pdb_text?r.predicted_pdb_text.length+" B":"null"}, experimental: ${r.experimental_pdb_text?r.experimental_pdb_text.length+" B":"null"}`);

  // Wait for both viewers to finish initialising before loading.
  const loadStructures = async () => {
    if (viewersReady) await viewersReady;
    if (r.predicted_pdb_text)    await loadPredictedWithPlddt(r.predicted_pdb_text);
    if (r.experimental_pdb_text) await loadPdbInto("viewer-exp", r.experimental_pdb_text, {label:"experimental"});
  };
  loadStructures();
  if (r.mean_plddt != null)
    $("#v-plddt").textContent = r.mean_plddt.toFixed(1) + " ± " + (r.plddt_sd||0).toFixed(1);
  if (r.rmsd_pred_vs_exp != null) $("#v-rmsd").textContent = r.rmsd_pred_vs_exp.toFixed(2);
  if (r.esm2_embedding_sd != null) $("#v-emb").textContent = r.esm2_embedding_sd.toFixed(3);

  // pockets / docking
  $("#pockets-readout").innerHTML = (r.pockets||[]).map(p =>
    `<div>${p.is_active_site?'<span style="color:var(--accent)">★ </span>':''}<b>${p.id}</b> score ${p.score.toFixed(2)} · ${p.nearby_residues.length} residues</div>`
  ).join("") || "no pockets";
  drawPoses(r.poses || []);
  $("#poses-readout").innerHTML =
    `<b>${(r.poses||[]).length}</b> poses · best <b>${(r.poses[0]||{}).score?.toFixed(2)}</b>`;

  // dynamics
  drawEnergies(r.energies_ev);
  $("#v-mace-n").textContent  = r.mace_region_size||"—";
  $("#v-mace").textContent    = r.mace_energy_mean!=null ? r.mace_energy_mean.toFixed(3) : "—";
  $("#v-mace-sd").textContent = r.mace_energy_sd!=null ? r.mace_energy_sd.toFixed(3) : "—";
  $("#v-res").textContent     = r.residence_ps_near_heme_fe!=null ? r.residence_ps_near_heme_fe.toFixed(2) : "—";

  // AL
  lastAL = r.al_history || [];
  drawALHeatmap(lastAL);
  if (lastAL.length){
    const last = lastAL[lastAL.length-1];
    $("#al-readout").innerHTML =
      `iteration <b>${last.iteration}</b> · tϕ suggests <b>${renderPert(last.chosen)}</b><br>` +
      `predicted <b>${last.predicted.toFixed(3)} ± ${last.predicted_sd.toFixed(3)}</b> · ` +
      `observed <b>${last.observed.toFixed(3)} ± ${last.observed_sd.toFixed(3)}</b><br>` +
      `converged residues: ${(r.al_converged_residues||[]).map(x=>`<span class="residue-pill">R${x}</span>`).join("")}`;
  }
  // cross-check
  $("#cross-readout").innerHTML =
    `agreement: ${(r.agreement_residues||[]).map(x=>`<span class="residue-pill agree">R${x}</span>`).join("") || "—"}<br>` +
    `disagreement (discovery flags): ${(r.disagreement_residues||[]).map(x=>`<span class="residue-pill disagree">R${x}</span>`).join("") || "—"}`;

  // remember the discovery-flag residues so the user can light them up in 3D
  lastDiscoveries = (r.disagreement_residues || []).slice();
  console.log("[discovery] flagged residues:", lastDiscoveries);

  refreshTheories();
}

// initial header refresh once theories begin populating
setTimeout(refreshTheories, 500);
