"""
phytologue.pipeline — end-to-end pipelines.

The v0 example is APX (apx.py).  Each pipeline composes adapters and
runners but adds NO new science of its own — it only wires.  Adding a
new pipeline (e.g. plant chitinase) is mostly a copy of apx.py with the
target's PDB id and ligand swapped.
"""

from .apx import APXPipeline, APXPipelineResult, run_apx_pipeline

__all__ = ["APXPipeline", "APXPipelineResult", "run_apx_pipeline"]
