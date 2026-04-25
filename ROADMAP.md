# Phytologue — Roadmap

The v0 sandbox is a vertical slice through every stage of the pipeline,
correctly architected and stubbed where the production tool is too heavy
for v0 compute. Everything below is a localised extension — each item
should be a small, surgical change because the adapter layer is what
already absorbs the surface area.

Items are listed in roughly the order I expect to tackle them. Numbering
is not a priority ranking; (★) marks items that unlock new science rather
than just better engineering.

## 1.  ESMFold-first path for genuinely novel sequences (★)

When the user uploads a sequence with no PDB hit, skip stage 1 (a) and
let ESMFold be the only structural input. The pipeline already supports
this — the `Protein.source` field can be `"esmfold"` only — so the work
is UI: a "structure: novel sequence" toggle and a clearly-flagged
provenance badge in the predicted-structure pane. `adapters/pdb.py` and
`adapters/esmfold.py` already do not assume each other's output.

## 2.  `ReaDDyAdapter` for cell-compartment reaction-diffusion (★)

Add `phytologue/adapters/readdy.py` wearing the `Calculator` coat at the
particle/CG level: positions are particle-CoMs, "energies" are reaction
events, "forces" are particle-particle interaction kernels.  The
`ComplexState` extension is to allow `entities: list[Particle]` alongside
the existing protein/cofactor/ligand fields — `core/state.py` already
keeps them as separate tuples, so adding a fifth field is mechanical.
This is the cell-scale layer.

## 3.  QM/MM for the actual peroxidase catalytic step (★)

The MACE adapter's region selector (`select_active_site_region`) gets
swapped for a smaller, electron-transfer-aware region; the `Calculator`
on that region becomes a real PySCF QM Calculator (not MACE-OFF23,
which is force-field-style).  Replace MACEAdapter with `PySCFCalculator`
inside the same region selector.  The rest of the pipeline does not
change.

## 4.  Plant-cell membrane context via MARTINI

Add `adapters/martini.py` with a `Calculator` shape that reads MARTINI
3 force-field tables and produces forces over coarse-grained beads.
Use it as the "bulk" Calculator instead of OpenMM-AMBER for runs where
membrane embedding matters.  Adapter shape:
`MartiniCalculator(state) -> energy, forces` plus a thin
`MartiniSimulator` cousin to OpenMMSimulator.

## 5.  Generative / inverse direction — function → sequence (★)

This is **polarity ↓** in the framework.  v0 runs ϕ → ∃ exclusively;
inverse runs ∃ → ϕ where the desired ∃ (e.g. higher activity, broader
substrate scope) drives a generative pass over sequence space.  Two
candidate adapters:

* `adapters/protein_lm.py` — masked-LM decoding from ESM-2 (cheap, in-env).
* `adapters/proteindiffusion.py` — protein diffusion model (heavier, GPU).

Both implement `LearnedModel` returning `Prediction(value=sequence,
uncertainty=per-position log-prob entropy)`.  The AL loop's `evaluator`
becomes a forward-pass simulator over the generated sequence.

## 6.  Multi-objective active learning across activity, stability, solubility

Replace the scalar `evaluator: dict -> (y, sigma)` with a vector
evaluator `dict -> (np.ndarray[k], np.ndarray[k])`.  UCB becomes a
hypervolume-improvement acquisition (or Chebyshev scalarisation for a
simpler v1).  `acquisition.py`'s protocol takes a `Prediction` whose
`value` is a vector — already supported by the protocol's `Any` typing.
Pareto-front rendering goes in the right-hand panel as a 2-D scatter
with uncertainty crosses.

## 7.  Add CUDA device branch in `device.py`

The current rule is "MPS on Mac, CPU on Linux".  When MSI GPU nodes are
ready, add a CUDA arm to `select_device()` gated on
`PHYTOLOGUE_DEVICE=cuda` *or* a positive `PHYTOLOGUE_GPU=1` flag.  Add
an `environment-cuda.yml` (separate from the cross-platform
`environment.yml`) that pins `pytorch-cuda` to the cluster's driver.

## 8.  Slurm wrappers for scaled MSI runs

Each runner already takes JSON on stdin and writes JSON on stdout, so
the wrappers themselves are tiny:

```bash
# scripts/slurm_dock.sh
#!/usr/bin/env bash
#SBATCH ...
cat $1 | python -m phytologue.runners.docker > $2
```

Write three wrappers — one per runner.  Add a `scripts/submit.py` that
takes a list of perturbations and emits the SLURM array job description.

## 9.  Real DiffDock and Foldseek

Drop-in replacements for the v0 stubs.  DiffDock requires
`torch_geometric` and ~2 GB of weights; Foldseek requires the AlphaFoldDB
or PDB databases locally (~700 GB for the full AFDB cluster, much less
for `afdb50_v4`).  Each replacement is a single-file change inside the
existing adapter — the FastAPI routes and the AL loop see no diff.

## 10.  Calibrated uncertainty everywhere

v0 uncertainties are heuristic (quantile heads on the surrogate, ensemble
SD on MACE, embedding-norm SD on ESM-2).  Replace with:

* surrogate → conformal prediction over the existing GBR
* MACE → real ensemble of 3 MACE-OFF23 members, averaged
* ESM-2 → MC-dropout or last-layer Bayesian linear head

Each is local to one adapter.

## 11.  GNN surrogate with on-the-fly fingerprints

Swap `GBRSurrogate` for a GNN over a small molecular graph (residue
context + ligand subgraph).  Implement in `adapters/gnn_surrogate.py`
with the same `(fit, predict)` shape so the AL loop is unchanged.  This
unlocks much larger candidate panels (1k+ instead of 50).

## 12.  Memory of past sessions

Persist `theories: list[LearnedModel]` and the AL history to a JSON or
SQLite file under `data/sessions/`.  Add a session picker to the UI.
This makes "iteration i carries forward across days" an honest claim
rather than a single-process artifact.

## 13.  Docs site (separate)

A small Vitepress / mdBook site that surfaces the framework essay
(Wang 2025) alongside the engineering README.  The framework vocabulary
deserves a fixed home; the README should remain the engineering surface.
