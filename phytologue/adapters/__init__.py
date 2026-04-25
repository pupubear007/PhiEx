"""
phytologue.adapters — every external tool / ML model wears the same coat.

Architectural rule (the most important one in this repo):
    Each adapter is the MINIMUM code that translates between an external
    tool's API and `phytologue.core.state` + `phytologue.core.calculator`
    + `phytologue.core.learned`.  Tool-specific types (Bio.PDB.Structure,
    openmm.app.Topology, esm.ESM2 module, …) MUST NOT leak past this
    package.

Adapter map (v0):

    pdb.py          Classical: fetch RCSB PDB by id; UniProt by accession.
    esmfold.py      ML:        ESMFold inference  (LearnedModel)
    esm2.py         ML:        ESM-2 embeddings + attention  (LearnedModel)
    foldseek.py     ML+stub:   structural homology search    (LearnedModel)
    p2rank.py       ML+stub:   pocket detection              (LearnedModel)
    vina.py         Classical: AutoDock Vina docking
    diffdock.py     ML+stub:   DiffDock pose generator       (LearnedModel)
    openmm_calc.py  Classical: OpenMM Calculator
    mace.py         ML:        MACE-OFF23 Calculator + LearnedModel
    surrogate.py    ML:        gradient-boosting surrogate   (LearnedModel)

Stubs raise ModelLoadError if real backends are missing AND the env var
PHYTOLOGUE_STRICT=1 is set; otherwise they fall through to a clearly-marked
synthetic implementation that returns plausible structure-shaped data so
the rest of the pipeline can still be exercised end-to-end.  This is the
v0 stubbing policy and is documented in the README.
"""
