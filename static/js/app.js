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
// ─────────────────────────────────────────────────────────────────────────
let predViewer = null, expViewer = null;
async function initViewers(){
  if (typeof molstar === "undefined") {
    console.warn("Mol* not available; structure panes will be empty");
    return;
  }
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
  } catch (e) {
    console.error("Mol* init failed", e);
  }
}
initViewers();

async function loadPdbInto(viewer, pdbText, opts={}){
  if (!viewer || !pdbText) return;
  await viewer.clear();
  const blob = new Blob([pdbText], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  await viewer.loadStructureFromUrl(url, "pdb", false, {
    representationParams: { theme: { globalName: opts.coloring || "chain-id" } }
  });
}

// pLDDT colouring: write pLDDT into B-factor column then ask Mol* to colour
// by B-factor.  This is the standard trick.  The backend already wrote
// pLDDT into the b-factor column in the predicted PDB text.
async function loadPredictedWithPlddt(pdbText){
  if (!predViewer) return;
  await predViewer.clear();
  const blob = new Blob([pdbText], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  await predViewer.loadStructureFromUrl(url, "pdb", false, {
    representationParams: { theme: { globalName: "atom-property" } }
  });
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
      await loadPdbInto(expViewer, r.experimental.pdb_text);
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
  if (r.predicted_pdb_text) loadPredictedWithPlddt(r.predicted_pdb_text);
  if (r.experimental_pdb_text) loadPdbInto(expViewer, r.experimental_pdb_text);
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

  refreshTheories();
}

// initial header refresh once theories begin populating
setTimeout(refreshTheories, 500);
