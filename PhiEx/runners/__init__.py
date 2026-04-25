"""
PhiEx.runners — Slurm-ready batch runners.

Each runner is callable two ways:
    (a) in-process from FastAPI for local M4 runs
    (b) as a standalone script reading JSON from stdin and writing JSON
        to stdout, so a Slurm wrapper can submit it on MSI

Architectural rule: runners hold NO in-memory-only state in their core
job logic.  They take a serialisable payload and emit a serialisable
result.  The FastAPI layer is the only place that holds session state.

v0 runners:
    runners.docker      Docker payload runner (Vina or DiffDock)
    runners.simulator   Simulator payload runner (OpenMM or MACE-MM)
    runners.batch       Active-learning batch runner (n iterations)
"""

from .docker import run_docking_payload
from .simulator import run_simulator_payload
from .batch import run_batch_payload

__all__ = ["run_docking_payload", "run_simulator_payload", "run_batch_payload"]
