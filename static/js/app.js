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

async function runStructure() {
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
      `hetero groups: <b>${summarizeHet(r.cofactors)}</b><br>` +
      `active site: <b>—</b>`;
    residueLabels = r.residue_labels || {};
    refreshSequence(true);
  } catch(e){ alert("structure: "+e.message); }
}
$("#run-structure").addEventListener("click", () => runStructure());

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
    refreshSequence();
  } catch(e){ alert("annotate: "+e.message); }
});

async function runPockets() {
  try {
    const r = await postJSON("/api/pockets", {anchor: $("#i-anchor").value.trim() || null});
    lastPockets = r.pockets;
    lastSite = r.site || null;
    const html = (r.pockets||[]).map(p =>
      `<div>${p.is_active_site?'<span style="color:var(--accent)">★ </span>':''}<b>${p.id}</b> score ${p.score.toFixed(2)} · ${p.nearby_residues.length} residues</div>`
    ).join("") || "no pockets";
    $("#pockets-readout").innerHTML = siteHTML(r.site) + html + `<div style="color:var(--ink-dim);font-size:10px">σ ${(r.score_sd||0).toFixed(2)}</div>`;
    if (r.active_site_id)
      $("#state-readout").innerHTML = $("#state-readout").innerHTML.replace(/active site:.*/,
        `active site: <b>${r.site ? r.site.label : r.active_site_id}</b> (${r.active_site_id})`);
    refreshSequence();
  } catch(e){ alert("pockets: "+e.message); }
}
$("#run-pockets").addEventListener("click", () => runPockets());

$("#run-dock").addEventListener("click", async () => {
  try {
    const smi = $("#i-smiles").value.trim();
    const r = await postJSON("/api/dock", {method: "vina", smiles: smi || null});
    lastPoses = r.poses;
    drawPoses(r.poses);
    refreshSequence();
    $("#poses-readout").innerHTML =
      `<b>${r.ligand ? r.ligand.name : "ligand"}</b>: <b>${r.n_poses}</b> poses · best <b>${(r.poses[0]||{}).score?.toFixed(2)}</b> kcal/mol` +
      (((r.poses||[])[0]||{}).stub ? ` <span style="color:var(--warn, #c66)">[STUB — synthetic poses, Vina not installed]</span>` : "") +
      ((r.pose_contacts||[]).length ? `<br>contacts: ${r.pose_contacts.map(pill).join("")}` : "");
  } catch(e){ alert("dock: "+e.message); }
});

$("#run-screen").addEventListener("click", async () => {
  const text = $("#i-ligands").value.trim();
  if (!text) { alert("screen: paste one ligand per line first"); return; }
  const n = text.split(/[\n;]+/).filter(s => s.trim() && !s.trim().startsWith("#")).length;
  $("#screen-readout").innerHTML = `docking ${n} ligand(s)… progress in the ticker`;
  try {
    const r = await postJSON("/api/dock/batch", {ligands: text});
    renderScreen(r);
  } catch(e){
    $("#screen-readout").innerHTML = `<span style="color:var(--warn, #c66)">screen failed — ${e.message}</span>`;
  }
});

function summarizeHet(list){
  if (!list || !list.length) return "—";
  const n = {};
  list.forEach(c => { const k = `${c.name}${c.role && c.role !== "cofactor" ? " (" + c.role + ")" : ""}`; n[k] = (n[k]||0) + 1; });
  return Object.entries(n).map(([k, v]) => v > 1 ? `${k} ×${v}` : k).join(", ");
}

function renderScreen(r){
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const showRmsd = r.rows.some(x => x.crystal_rmsd != null);
  const head = `<tr><th>#</th><th>ligand</th><th>score</th><th>LE</th><th>site hits</th><th>dist</th>${showRmsd ? "<th>RMSD</th>" : ""}</tr>`;
  const body = r.rows.map(x => x.error
    ? `<tr class="err"><td>—</td><td title="${esc(x.input)}">${esc(x.name)}</td><td colspan="${showRmsd ? 5 : 4}" title="${esc(x.error)}">${esc(x.error.split(" — ")[0])}</td></tr>`
    : `<tr><td>${x.rank}</td><td title="${esc(x.smiles)}">${esc(x.name)}</td>` +
      `<td class="num">${x.score.toFixed(2)}</td><td class="num">${x.efficiency != null ? x.efficiency.toFixed(2) : "—"}</td>` +
      `<td>${(x.site_hits||[]).map(n => resLabel(n)).join(" ") || "<span style='color:var(--warn, #c66)'>none</span>"}</td>` +
      `<td class="num">${x.dist_to_site != null ? x.dist_to_site.toFixed(1) : "—"}</td>` +
      (showRmsd ? `<td class="num">${x.crystal_rmsd != null ? x.crystal_rmsd.toFixed(2) : "—"}</td>` : "") + `</tr>`
  ).join("");
  $("#screen-readout").innerHTML =
    `<div style="margin-bottom:4px">${r.n_ok} docked · ${r.n_failed} failed · ${r.seconds}s · site <b>${esc((r.site||{}).label || r.pocket_id)}</b></div>` +
    `<table class="screen-table">${head}${body}</table>` +
    `<div style="color:var(--ink-dim);font-size:10px;margin-top:4px">score: Vina kcal/mol (±~2; rank, don't trust absolutes) · LE: −score/heavy atom (≥0.3 good) · ` +
    `site hits: defining site residues within 4 Å · dist: closest atom to site centre, Å</div>` +
    `<a href="/api/dock/batch.csv" download style="font-size:11px">⤓ download CSV</a>`;
}

$("#run-dynamics").addEventListener("click", async () => {
  try {
    const ps = parseFloat($("#i-ps").value || "10");
    const r = await postJSON("/api/dynamics", {total_ps: ps, use_mace: true});
    lastDyn = r;
    drawEnergies(r.energies_ev);
    $("#v-mace-n").textContent  = r.mace ? r.mace.region_atoms : "—";
    $("#v-mace").textContent    = r.mace ? r.mace.energy_ev.toFixed(3) : "—";
    $("#v-mace-sd").textContent = r.mace ? r.mace.energy_sd.toFixed(3) : "—";
    $("#v-res").textContent     = r.residence_ps_near_site != null
                                  ? r.residence_ps_near_site.toFixed(2)
                                  : (r.ligand_simulated === false ? "n/a (ligand not simulated)" : "—");
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
      surrogateQuality(last);
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
      const isModel = /SWISS|AlphaFold|AlphaFill|Modeling|Predicted/i.test(e.method || "")
                      || /^(SWISS|AF|AFILL|ALPHAFILL):/i.test(e.pdb_id || "");
      const idColor = isModel ? "var(--stub)" : "var(--ink)";
      return `<div style="cursor:pointer;padding:2px 0" data-pdb="${e.pdb_id}">
        ${isBest}<b style="color:${idColor}">${e.pdb_id}</b>
        <span> · ${e.method||"—"} · ${res} · ${e.chains||""}</span>
      </div>`;
    }).join("");
    out.innerHTML = `<div style="margin-bottom:4px;color:var(--phi)">${r.n} PDB entries — click to pick</div>${rows}`;
    // Click-to-pick on the alternates. For predicted-model entries
    // (AF: / SWISS:) we also auto-trigger the STRUCTURE stage, since
    // those rows have no resolution / chain metadata to inspect — the
    // user almost certainly clicked them to render the model, not to
    // stage a different fetch. For experimental PDBs we leave the
    // explicit STRUCTURE button click in place (cheaper to be deliberate).
    out.querySelectorAll("[data-pdb]").forEach(el => {
      el.addEventListener("click", () => {
        const pid = el.getAttribute("data-pdb");
        $("#i-pdb").value = pid;
        out.querySelectorAll("[data-pdb]").forEach(x =>
          x.style.background = x === el ? "var(--rule)" : "");
        if (/^(AF|AF2|ALPHAFOLD|SWISS|AFILL|ALPHAFILL):/i.test(pid)) {
          $("#run-structure").click();
        }
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
  $("#state-readout").innerHTML      = "protein: <b>—</b><br>hetero groups: <b>—</b><br>active site: <b>—</b>";
  $("#theories-readout").innerHTML   = "none loaded";
  $("#pockets-readout").innerHTML    = "no pockets yet";
  $("#poses-readout").innerHTML      = "no poses yet";
  $("#md-readout").innerHTML         = "MACE region: <b id=\"v-mace-n\">—</b> atoms<br>" +
                                        "MACE E: <b id=\"v-mace\">—</b> ± <b id=\"v-mace-sd\">—</b> eV<br>" +
                                        "residence near site: <b id=\"v-res\">—</b> ps";
  $("#al-readout").innerHTML         = "no iterations yet";
  $("#cross-readout").innerHTML      = "run AL first to populate";
  $("#screen-readout").innerHTML     = "paste ligands under stage 4 and press 4b";
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
    logSys(`★ discovery flags highlighted: ${lastDiscoveries.map(resLabel).join(", ")}`);
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
    uniprot: $("#i-uniprot").value.trim() || null,
    anchor: $("#i-anchor").value.trim() || null,
    ligand_smiles: $("#i-smiles").value.trim() || null,
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

// residue labels ("Ser197") come from the server; fall back to "#197"
let residueLabels = {};
let lastSite = null;
function resLabel(n){ return residueLabels[n] || residueLabels[String(n)] || `#${n}`; }
function pill(n, cls=""){ return `<span class="residue-pill ${cls}">${resLabel(n)}</span>`; }
function siteHTML(site){
  if (!site) return "";
  const conf = {user:"var(--accent)", high:"var(--accent)", medium:"var(--ink)", low:"var(--warn, #c66)"}[site.confidence] || "var(--ink)";
  const notes = (site.notes||[]).map(n=>`<div style="color:var(--ink-dim);font-size:10px">• ${n}</div>`).join("");
  return `<div style="margin-bottom:6px">site: <b>${site.label}</b> ` +
         `<span style="color:${conf}">[${site.source}, ${site.confidence}]</span>${notes}</div>`;
}

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

// Out-of-sample surrogate quality.  In-sample (training) RMSE/R² are not
// shown: a boosted-tree fit drives them to ≈ 0 / ≈ 1 regardless of skill.
function surrogateQuality(h){
  const f = (v, d=3) => v!=null ? v.toFixed(d) : "—";
  return `prospective RMSE <b>${f(h.prospective_rmse)}</b> · ` +
         `LOO R² <b>${f(h.loo_r2)}</b> · ` +
         `±1σ coverage <b>${h.coverage_1sigma!=null ? Math.round(100*h.coverage_1sigma)+"%" : "—"}</b>`;
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

  // site / pockets / docking
  residueLabels = r.residue_labels || {};
  lastSite = r.site || null;
  if (r.site && r.site.label)
    $("#state-readout").innerHTML = $("#state-readout").innerHTML.replace(/active site:.*/,
      `active site: <b>${r.site.label}</b> (${r.active_site_id||"—"})`);
  $("#pockets-readout").innerHTML = siteHTML(r.site) + (r.pockets||[]).map(p =>
    `<div>${p.is_active_site?'<span style="color:var(--accent)">★ </span>':''}<b>${p.id}</b> score ${p.score.toFixed(2)} · ${p.nearby_residues.length} residues</div>`
  ).join("") || "no pockets";
  drawPoses(r.poses || []);
  $("#poses-readout").innerHTML = (r.poses||[]).length
    ? `<b>${(r.ligand||{}).name || "ligand"}</b>${(r.ligand||{}).default ? ' <span style="color:var(--ink-dim)">(default for this site)</span>' : ""}: ` +
      `<b>${r.poses.length}</b> poses · best <b>${r.poses[0].score.toFixed(2)}</b> kcal/mol` +
      (r.poses[0].stub ? ` <span style="color:var(--warn, #c66)">[STUB — synthetic poses]</span>` : "") +
      ((r.pose_contacts||[]).length ? `<br>contacts: ${r.pose_contacts.map(pill).join("")}` : "")
    : (r.docking_error
        ? `<span style="color:var(--warn, #c66)">no ligand docked — ${r.docking_error}</span>`
        : `no ligand docked — no default substrate for this site type; enter a SMILES or compound name in <b>Ligand</b>`);

  // dynamics
  drawEnergies(r.energies_ev);
  $("#v-mace-n").textContent  = r.mace_region_size||"—";
  $("#v-mace").textContent    = r.mace_energy_mean!=null ? r.mace_energy_mean.toFixed(3) : "—";
  $("#v-mace-sd").textContent = r.mace_energy_sd!=null ? r.mace_energy_sd.toFixed(3) : "—";
  $("#v-res").textContent     = r.residence_ps_near_site!=null ? r.residence_ps_near_site.toFixed(2)
                                : "n/a (ligand not simulated)";

  // AL
  lastAL = r.al_history || [];
  drawALHeatmap(lastAL);
  if (lastAL.length){
    const last = lastAL[lastAL.length-1];
    $("#al-readout").innerHTML =
      `iteration <b>${last.iteration}</b> · tϕ suggests <b>${renderPert(last.chosen)}</b><br>` +
      `predicted <b>${last.predicted.toFixed(3)} ± ${last.predicted_sd.toFixed(3)}</b> · ` +
      `observed <b>${last.observed.toFixed(3)} ± ${last.observed_sd.toFixed(3)}</b><br>` +
      surrogateQuality(last) + `<br>` +
      (r.al_scoring ? `<span style="color:var(--ink-dim);font-size:10px">objective: ${r.al_scoring}</span><br>` : "") +
      `converged residues: ${(r.al_converged_residues||[]).map(x=>pill(x)).join("")}`;
  }
  // cross-check
  $("#cross-readout").innerHTML = r.cross_check_skipped
    ? `cross-check skipped: ${r.cross_check_skipped}`
    : `agreement: ${(r.agreement_residues||[]).map(x=>pill(x,"agree")).join("") || "—"}<br>` +
      `discovery flags (structure/simulation only): ${(r.disagreement_residues||[]).map(x=>pill(x,"disagree")).join("") || "—"}<br>` +
      `attention only: ${(r.attention_only_residues||[]).map(x=>pill(x)).join("") || "—"}`;

  // remember the discovery-flag residues so the user can light them up in 3D
  lastDiscoveries = (r.disagreement_residues || []).slice();
  console.log("[discovery] flagged residues:", lastDiscoveries);

  refreshTheories();
}

// initial header refresh once theories begin populating
setTimeout(refreshTheories, 500);

// ════════════════════════════════════════════════════════════════════
// Candidate-gene triage
// ════════════════════════════════════════════════════════════════════
let triagePoll = null;
let triageRows = [];

async function refreshCandidateFiles(){
  try {
    const r = await (await fetch("/api/candidates/local")).json();
    const sel = $("#i-cand-file");
    sel.innerHTML = `<option value="">— local candidate file —</option>` +
      (r.files||[]).map(f => `<option value="${f}">${f}</option>`).join("");
  } catch(_) {}
}
refreshCandidateFiles();

$("#i-cand-file").addEventListener("change", async e => {
  const name = e.target.value; if (!name) return;
  const r = await fetch(`/api/candidates/local/${encodeURIComponent(name)}`);
  if (r.ok) { $("#i-cands").value = await r.text(); logSys(`candidate file loaded: ${name}`); }
});
$("#cand-upload-btn").addEventListener("click", () => $("#i-cand-upload").click());
$("#i-cand-upload").addEventListener("change", async e => {
  const f = e.target.files[0]; if (!f) return;
  $("#i-cands").value = await f.text();
  logSys(`candidate file loaded from disk: ${f.name}`);
});

$("#run-triage").addEventListener("click", async () => {
  const text = $("#i-cands").value.trim();
  if (!text) { alert("triage: load or paste a candidate list first"); return; }
  const body = {
    text, dock: $("#i-cand-dock").checked,
    organism: $("#i-cand-org").value,
    sources: $("#i-cand-src").value.trim(),
    tracks: $("#i-cand-track").value.split(","),
    limit: parseInt($("#i-cand-limit").value || "0", 10),
  };
  try {
    const r = await postJSON("/api/candidates/start", body);
    $("#triage-readout").innerHTML = `triaging ${r.total} gene(s)… progress in the ticker`;
    if (triagePoll) clearInterval(triagePoll);
    triagePoll = setInterval(pollTriage, 3000);
  } catch(e) { alert("triage: " + e.message); }
});
$("#stop-triage").addEventListener("click", () => postJSON("/api/candidates/stop", {}).catch(()=>{}));

async function pollTriage(){
  try {
    const s = await (await fetch("/api/candidates/status")).json();
    triageRows = s.rows || [];
    renderTriage(s);
    if (!s.running && triagePoll) { clearInterval(triagePoll); triagePoll = null; }
  } catch(_) {}
}

function renderTriage(s){
  const esc = v => String(v ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
  const fc = v => v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(1);
  const rows = triageRows.map((r, i) => {
    const struct = r.structure ? `${esc(r.structure.replace(/^AF:|^AFILL:/, m => m === "AFILL:" ? "AFill " : "AF "))}` +
      (r.mean_plddt != null ? ` <span style="color:var(--ink-dim)">${r.mean_plddt.toFixed(0)}</span>` : "") : "—";
    const site = r.site ? `${esc(r.site)} <span style="color:var(--ink-dim)">${esc(r.site_confidence)}</span>` : "—";
    const dockc = r.error ? `<span style="color:var(--warn, #c66)" title="${esc(r.error)}">error</span>`
      : r.best_ligand ? `${esc(r.best_ligand)} ${r.best_score.toFixed(1)}` +
        (r.margin != null ? ` <b title="best binder − negative control (kcal/mol)">Δ${r.margin.toFixed(1)}</b>` : "")
      : (r.track === "dock" ? `<span style="color:var(--ink-dim)">panel ready</span>` : "");
    const warn = (r.warnings||[]).length ? ` <span title="${esc(r.warnings.join("\n"))}">⚠</span>` : "";
    const full = $("#triage-readout").classList.contains("tri-full");
    return `<tr class="tri-row" data-i="${i}" title="${esc(r.class_title)} — ${esc(r.description)}">` +
      `<td>${esc(r.gene_id)}${warn}</td><td><span class="trk ${r.track}">${r.track}</span> ${esc(full ? r.class_title : r.class)}` +
      (full ? `<div style="color:var(--ink-dim);font-size:10px">${esc((r.description||"").slice(0, 90))}</div>` : "") + `</td>` +
      `<td class="num">${fc(r.peak_log2fc)}</td><td>${struct}</td><td>${site}</td><td>${dockc}</td></tr>`;
  }).join("");
  const counts = {dock:0, structure:0, unsuited:0};
  triageRows.forEach(r => counts[r.track] = (counts[r.track]||0) + 1);
  $("#triage-readout").innerHTML =
    `<div style="margin-bottom:4px"><button class="btn" id="tri-expand" style="width:auto;padding:2px 8px;margin:0 6px 0 0;font-size:10px">${$("#triage-readout").classList.contains("tri-full") ? "✕ close" : "⤢ expand"}</button>` +
    `${s.running ? "running" : "done"} · ${s.done}/${s.total} genes · ${s.elapsed}s · ` +
    `<span class="trk dock">dock ${counts.dock}</span> <span class="trk structure">structure ${counts.structure}</span> ` +
    `<span class="trk unsuited">unsuited ${counts.unsuited}</span></div>` +
    `<div class="tri-wrap"><table class="screen-table"><tr><th>gene</th><th>class</th><th>log2FC</th><th>structure (pLDDT)</th><th>site</th><th>dock</th></tr>${rows}</table></div>` +
    `<div style="color:var(--ink-dim);font-size:10px;margin-top:4px">click a row to open it: structure + site load, and its ligand panel fills the screen list (press 4b). ` +
    `Δ = best binder − negative control, kcal/mol (larger = clearer).</div>` +
    `<a href="/api/candidates.csv" download style="font-size:11px">⤓ download CSV</a>`;
  document.querySelectorAll("#triage-readout tr.tri-row").forEach(tr =>
    tr.addEventListener("click", () => {
      $("#triage-readout").classList.remove("tri-full");
      openCandidate(triageRows[+tr.dataset.i]);
    }));
  const ex = $("#tri-expand");
  if (ex) ex.addEventListener("click", ev => {
    ev.stopPropagation();
    $("#triage-readout").classList.toggle("tri-full");
    renderTriage(s);
  });
}

async function openCandidate(r){
  if (!r || !r.structure) { alert(`${r ? r.gene_id : "candidate"}: no structure available${r && r.error ? " — " + r.error : ""}`); return; }
  $("#i-pdb").value = r.structure;
  $("#i-uniprot").value = r.uniprot || "";
  $("#i-anchor").value = "";
  $("#i-smiles").value = "";
  $("#i-ligands").value = (r.panel||[]).map(p => `${p.label}: ${p.value}`).join("\n");
  logSys(`opening ${r.gene_id} (${r.class_title}) — ${r.structure}`);
  if (r.note) logSys(`note: ${r.note}`);
  (r.warnings||[]).forEach(w => logSys(`⚠ ${r.gene_id}: ${w}`));
  await runStructure();
  await runPockets();
  if (r.screen) renderScreen({rows: r.screen, n_ok: r.screen.filter(x=>!x.error).length,
                              n_failed: r.screen.filter(x=>x.error).length, seconds: "cached",
                              site: {label: r.site}, pocket_id: ""});
}

// ════════════════════════════════════════════════════════════════════
// Sequence viewer — AlphaFold-DB-style sequence strip, feature tracks,
// PAE map; click/drag highlights residues in the 3D structure.
// ════════════════════════════════════════════════════════════════════
const SEQ = { data: null, cellW: 11, sel: null, dragFrom: null, overlay: [], pae: null };
const PLDDT_BANDS = [[90, "#0053D6", "very high (>90)"], [70, "#65CBF3", "high (70–90)"],
                     [50, "#FFDB13", "low (50–70)"], [-1, "#FF7D45", "very low (<50)"]];
const FEAT_COLORS = {
  "Signal": "#c478b3", "Transit peptide": "#a070c0", "Propeptide": "#8a6aa8", "Transmembrane": "#9aa0a6",
  "Intramembrane": "#7d8288", "Domain": "#3d6fb0", "Region": "#5a6f8a", "Repeat": "#4e7f9a",
  "Zinc finger": "#6a8fbf", "Coiled coil": "#556b7f", "Compositional bias": "#3a4552",
  "Active site": "#ff4d4d", "Binding site": "#e69a3a", "Metal binding": "#d4c27a", "Site": "#e0c050",
  "Motif": "#d4c27a", "Disulfide bond": "#e8e070", "Glycosylation": "#8fbf7f", "Modified residue": "#6fbfa0",
  "Lipidation": "#bf9f6f", site: "#e05a4a", pocket: "#e69a3a", motif: "#d4c27a", catalytic: "#ff4d4d",
  contact: "#8fbf7f", variant: "#ff8ad8",
};
const plddtColor = v => v == null ? "#2a2d33" : PLDDT_BANDS.find(b => v > b[0])[1];

$("#seq-toggle").addEventListener("click", () => {
  $("#seq-panel").classList.toggle("collapsed");
  $("#seq-toggle").textContent = ($("#seq-panel").classList.contains("collapsed") ? "▸" : "▾") + " sequence";
});
$("#seq-refresh").addEventListener("click", () => refreshSequence(true));
$("#seq-clear").addEventListener("click", () => clearSeqHighlight());
$("#seq-pae").addEventListener("click", () => togglePae());

async function refreshSequence(open=false){
  try {
    const r = await fetch("/api/sequence-track");
    if (!r.ok) return;
    SEQ.data = await r.json();
    SEQ.sel = null;
    renderSequence();
    if (open && $("#seq-panel").classList.contains("collapsed")) $("#seq-toggle").click();
    if ($("#pae-box").classList.contains("on")) loadPae();
  } catch(e) { console.warn("[seq] refresh failed", e); }
}

function renderSequence(){
  const d = SEQ.data; if (!d) return;
  const W = SEQ.cellW, n = d.sequence.length;
  const mean = d.plddt ? (d.plddt.filter(v => v != null).reduce((a, b) => a + b, 0) / d.plddt.filter(v => v != null).length) : null;
  $("#seq-title").innerHTML = `<b>${d.name}</b> chain ${d.chain} · ${n} aa` +
    (d.uniprot ? ` · UniProt <b>${d.uniprot}</b>` : "") +
    (mean != null ? ` · mean pLDDT <b>${mean.toFixed(1)}</b>` : " · experimental (no pLDDT)");
  $("#seq-legend").innerHTML = d.plddt ? PLDDT_BANDS.map(b => `<span style="background:${b[1]}"></span>${b[2]}`).join("") : "";
  $("#seq-pae").style.display = d.is_model ? "" : "none";

  const ROW_NUM = 11, ROW_SEQ = 14, ROW_TRK = 9, GAP = 3;
  const tracks = d.tracks || [];
  const H = ROW_NUM + ROW_SEQ + GAP + tracks.length * (ROW_TRK + GAP) + 2;
  const width = n * W;
  const idxOf = {}; d.residues.forEach((r, i) => idxOf[r] = i);
  const x = resnum => (idxOf[resnum] ?? -1) * W;

  let s = `<svg width="${width}" height="${H}" id="seq-svg">`;
  for (let i = 0; i < n; i++) {
    const rn = d.residues[i];
    if (rn % 10 === 0) s += `<text x="${i * W + W / 2}" y="9" font-size="8" fill="#8a8a80" text-anchor="middle">${rn}</text>`;
    s += `<rect x="${i * W}" y="${ROW_NUM}" width="${W - 1}" height="${ROW_SEQ}" fill="${plddtColor(d.plddt ? d.plddt[i] : null)}"/>`;
    const dark = d.plddt && d.plddt[i] != null && d.plddt[i] > 50 && d.plddt[i] <= 90 ? "#0b0d10" : "#e8e6df";
    s += `<text x="${i * W + W / 2}" y="${ROW_NUM + 11}" font-size="10" fill="${dark}" text-anchor="middle">${d.sequence[i]}</text>`;
  }
  let y = ROW_NUM + ROW_SEQ + GAP;
  const labels = [];
  for (const t of tracks) {
    labels.push([t.name, y, t.label || ""]);
    s += `<rect x="0" y="${y}" width="${width}" height="${ROW_TRK}" fill="#14171c"/>`;
    if (t.kind === "values" && t.values) {
      const vals = t.values.filter(v => v != null);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      t.values.forEach((v, i) => {
        if (v == null) return;
        const f = hi > lo ? (v - lo) / (hi - lo) : 0;
        const c = `rgb(${Math.round(27 + f * 185)},${Math.round(39 + f * 155)},${Math.round(51 + f * 71)})`;
        s += `<rect x="${i * W}" y="${y}" width="${W - 1}" height="${ROW_TRK}" fill="${c}"/>`;
      });
    } else {
      for (const f of (t.features || [])) {
        const x0 = x(f.start), x1 = x(f.end);
        if (x0 < 0 || x1 < 0) continue;
        const c = FEAT_COLORS[f.type] || "#888";
        s += `<rect x="${x0}" y="${y}" width="${Math.max(W - 1, x1 - x0 + W - 1)}" height="${ROW_TRK}" fill="${c}" opacity="0.9"/>`;
      }
    }
    y += ROW_TRK + GAP;
  }
  s += `<rect id="seq-selrect" x="0" y="0" width="0" height="${H}" fill="#ffffff22" stroke="#ffd400" stroke-width="1" visibility="hidden"/>`;
  s += `</svg>`;
  $("#seq-scroll").innerHTML = s;

  // fixed track-label column
  let lab = document.getElementById("seq-labels");
  if (!lab) {
    lab = document.createElement("div"); lab.id = "seq-labels";
    lab.style.cssText = "flex:0 0 auto;position:relative;font-size:9px;color:#8a8a80;width:118px;";
    $("#seq-body").insertBefore(lab, $("#seq-scroll"));
  }
  lab.style.height = H + "px";
  lab.innerHTML = `<div style="position:absolute;top:${ROW_NUM + 1}px">sequence${d.plddt ? " · pLDDT" : ""}</div>` +
    labels.map(([nm, yy, tip]) => `<div title="${tip}" style="position:absolute;top:${yy - 1}px;white-space:nowrap;overflow:hidden;width:118px">${nm}</div>`).join("");

  const svg = $("#seq-svg");
  const idxAt = ev => {
    const rect = svg.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.floor((ev.clientX - rect.left) / W)));
  };
  svg.addEventListener("mousemove", ev => {
    const i = idxAt(ev);
    showSeqTip(ev, i);
    if (SEQ.dragFrom != null) drawSel(Math.min(SEQ.dragFrom, i), Math.max(SEQ.dragFrom, i));
  });
  svg.addEventListener("mouseleave", () => { $("#seq-tip").style.display = "none"; });
  svg.addEventListener("mousedown", ev => { SEQ.dragFrom = idxAt(ev); drawSel(SEQ.dragFrom, SEQ.dragFrom); });
  SEQ.idxAt = idxAt;
  if (!SEQ.mouseupBound) {
    SEQ.mouseupBound = true;
    window.addEventListener("mouseup", ev => {
      if (SEQ.dragFrom == null || !SEQ.data) return;
      const i = SEQ.idxAt(ev), a = Math.min(SEQ.dragFrom, i), b = Math.max(SEQ.dragFrom, i);
      SEQ.dragFrom = null;
      const res = SEQ.data.residues.slice(a, b + 1);
      highlightSeqResidues(res, a === b ? `${resName(a)}` : `${resName(a)}–${resName(b)} (${b - a + 1} aa)`);
    });
  }
  SEQ.H = H; fitSeqPanel();
}

function fitSeqPanel(){
  const pae = $("#pae-box").classList.contains("on") ? 225 : 0;
  $("#seq-panel").style.maxHeight = (Math.max(SEQ.H || 0, pae) + 46) + "px";
}

function resName(i){ const d = SEQ.data; return `${d.sequence[i]}${d.residues[i]}`; }

function drawSel(a, b){
  const r = $("#seq-selrect"); if (!r) return;
  r.setAttribute("x", a * SEQ.cellW); r.setAttribute("width", (b - a + 1) * SEQ.cellW);
  r.setAttribute("visibility", "visible");
}

function showSeqTip(ev, i){
  const d = SEQ.data, rn = d.residues[i];
  const lines = [`<b>${resName(i)}</b>` + (d.plddt && d.plddt[i] != null ? ` · pLDDT ${d.plddt[i].toFixed(1)}` : "")];
  for (const t of d.tracks || []) {
    if (t.kind === "values" && t.values && t.values[i] != null) lines.push(`${t.name}: ${t.values[i].toFixed(2)}`);
    for (const f of (t.features || [])) if (rn >= f.start && rn <= f.end) lines.push(`${f.label}`);
  }
  const tip = $("#seq-tip");
  tip.innerHTML = lines.join("<br>");
  tip.style.display = "block";
  tip.style.left = Math.min(window.innerWidth - 370, ev.clientX + 12) + "px";
  tip.style.top = (ev.clientY - 10 - tip.offsetHeight) + "px";
}

// 3D: overlay the selected residues (chain-restricted) on the structure in
// the lower viewer, as yellow sticks, and fly the camera to them.
async function highlightSeqResidues(resnums, label){
  const d = SEQ.data; if (!d || !resnums.length) return;
  await clearSeqHighlight();
  const viewer = expViewer;
  const pdb = _pdbCache.get("viewer-exp");
  if (!viewer || !viewer.plugin || !pdb) { logSys("sequence: no structure in the 3D viewer to highlight"); return; }
  const want = new Set(resnums);
  const lines = pdb.split(/\r?\n/).filter(l => l.startsWith("ATOM") &&
    (l.substring(21, 22).trim() || "A") === (d.chain || "A") && want.has(parseInt(l.substring(22, 26), 10)));
  if (!lines.length) { logSys(`sequence: residues not found in the loaded model (${label})`); return; }
  const plugin = viewer.plugin;
  try {
    const data = await plugin.builders.data.rawData({ data: lines.join("\n") + "\nEND\n", label: "seq-selection" });
    const traj = await plugin.builders.structure.parseTrajectory(data, "pdb");
    const model = await plugin.builders.structure.createModel(traj);
    const structure = await plugin.builders.structure.createStructure(model);
    await plugin.builders.structure.representation.addRepresentation(structure, {
      type: "ball-and-stick", color: "uniform", colorParams: { value: 0xffd400 }, sizeParams: { scale: 1.3 } });
    SEQ.overlay.push(data);
    try {
      const st = structure.cell && structure.cell.obj && structure.cell.obj.data;
      if (st) plugin.managers.camera.focusLoci({ kind: "structure-loci", structure: st });
    } catch(_) {}
    logSys(`sequence → 3D: ${label}`);
  } catch(e) { console.warn("[seq] highlight failed", e); }
}

async function clearSeqHighlight(){
  const viewer = expViewer;
  if (!viewer || !viewer.plugin) { SEQ.overlay = []; return; }
  for (const cell of SEQ.overlay) {
    const ref = (cell && cell.ref) || (cell && cell.transform && cell.transform.ref);
    if (ref) { try { await viewer.plugin.state.data.build().delete(ref).commit(); } catch(_) {} }
  }
  SEQ.overlay = [];
  const r = $("#seq-selrect"); if (r) r.setAttribute("visibility", "hidden");
}

// PAE map (AlphaFold DB): green = confident relative placement
async function togglePae(){
  const box = $("#pae-box");
  box.classList.toggle("on");
  fitSeqPanel();
  if (box.classList.contains("on")) await loadPae();
}

async function loadPae(){
  try {
    const r = await fetch("/api/pae");
    const j = await r.json();
    if (!r.ok) { logSys(`PAE: ${j.error || j.detail}`); $("#pae-box").classList.remove("on"); fitSeqPanel(); return; }
    SEQ.pae = j; drawPae();
  } catch(e) { console.warn("[pae]", e); }
}

function drawPae(sel){
  const p = SEQ.pae; if (!p) return;
  const cv = $("#pae-canvas"), m = p.matrix.length;
  const size = Math.min(200, Math.max(120, m));
  cv.width = cv.height = size;
  const ctx = cv.getContext("2d"), px = size / m;
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) {
    const f = Math.min(1, p.matrix[i][j] / (p.max || 31.75));
    const g = Math.round(68 + f * 187), rr = Math.round(f * 255), b = Math.round(27 + f * 228);
    ctx.fillStyle = `rgb(${rr},${g},${b})`;
    ctx.fillRect(j * px, i * px, Math.ceil(px), Math.ceil(px));
  }
  if (sel) { ctx.strokeStyle = "#ffd400"; ctx.strokeRect(sel[0], sel[1], sel[2] - sel[0], sel[3] - sel[1]); }
  let start = null;
  cv.onmousedown = ev => { const r = cv.getBoundingClientRect(); start = [ev.clientX - r.left, ev.clientY - r.top]; };
  cv.onmouseup = ev => {
    if (!start) return;
    const r = cv.getBoundingClientRect(), end = [ev.clientX - r.left, ev.clientY - r.top];
    const toRes = v => Math.max(1, Math.min(p.n, Math.floor(v / px) * p.bin + 1));
    const xs = [toRes(Math.min(start[0], end[0])), toRes(Math.max(start[0], end[0])) + p.bin - 1];
    const ys = [toRes(Math.min(start[1], end[1])), toRes(Math.max(start[1], end[1])) + p.bin - 1];
    drawPae([Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.max(start[0], end[0]), Math.max(start[1], end[1])]);
    start = null;
    const set = new Set();
    for (let k = xs[0]; k <= Math.min(p.n, xs[1]); k++) set.add(k);
    for (let k = ys[0]; k <= Math.min(p.n, ys[1]); k++) set.add(k);
    highlightSeqResidues([...set], `PAE box: ${ys[0]}–${ys[1]} × ${xs[0]}–${xs[1]}`);
  };
}
