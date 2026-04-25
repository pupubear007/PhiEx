"""
phytologue.device — compute device selection.

This is a thin re-export of the project's authoritative device.py at the
repository root.  Import from here in package code:

    from phytologue.device import select_device, describe_device

The root-level device.py (delivered as scaffolding) is the single source of
truth so a freshly cloned repo behaves identically whether the user imports
`device` (top-level) or `phytologue.device` (package).

Choice order:
    1.  explicit override via PHYTOLOGUE_DEVICE env var ("cpu", "mps", "cuda")
    2.  MPS  if available (Apple Silicon)
    3.  CPU  (Linux, including MSI login / compute nodes)

We deliberately do NOT auto-select CUDA even if it's available.  For v0 the
rule is: MPS on Mac, CPU on Linux.  Flip this later when you want GPU nodes.

The MPS path sets PYTORCH_ENABLE_MPS_FALLBACK=1 so ops that haven't yet
landed on MPS fall through to CPU instead of raising.
"""

from __future__ import annotations
import os
import platform
import logging

log = logging.getLogger("phytologue.device")


def select_device(override: str | None = None) -> str:
    """Return a torch device string: 'cpu', 'mps', or (if forced) 'cuda'."""
    forced = override or os.environ.get("PHYTOLOGUE_DEVICE")
    if forced:
        forced = forced.lower()
        if forced not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"PHYTOLOGUE_DEVICE={forced!r} not in cpu/mps/cuda")
        log.info("device forced via env: %s", forced)
        return forced

    # Lazy torch import so importing this module is cheap and works in
    # environments where torch is not yet installed (CI on docs-only branches).
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - torch missing is recoverable
        log.info("torch not importable; defaulting device=cpu")
        return "cpu"

    import torch  # type: ignore
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        log.info("device auto-selected: mps (Apple Silicon %s)", platform.machine())
        return "mps"

    log.info("device auto-selected: cpu (%s)", platform.platform())
    return "cpu"


def describe_device(device: str) -> dict:
    """Human-readable info about the chosen device.  Goes into the reasoning
    ticker so every ϕ → ∃ call logs which substrate it ran on."""
    info: dict = {"device": device}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        info["torch"] = "missing"
        return info

    if device == "mps":
        info["backend"] = "Metal Performance Shaders"
        info["fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    elif device == "cuda":
        info["backend"] = f"CUDA {getattr(torch.version, 'cuda', '?')}"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["backend"] = "CPU"
        info["threads"] = torch.get_num_threads()
    return info


def select_openmm_platform():
    """Pick the best OpenMM platform for this machine.

    OpenMM has its own platform layer separate from PyTorch.  v0 rule:
    Metal on Mac if available, else CPU.  CUDA is intentionally not chosen
    by default, mirroring the PyTorch policy.

    Returns (platform_name: str, properties: dict).  The properties dict
    is ready to hand to openmm.app.Simulation.
    """
    try:
        from openmm import Platform  # type: ignore
    except Exception:
        return "CPU", {}

    avail = {Platform.getPlatform(i).getName()
             for i in range(Platform.getNumPlatforms())}
    if "Metal" in avail:
        return "Metal", {}
    if "CUDA" in avail and os.environ.get("PHYTOLOGUE_DEVICE", "").lower() == "cuda":
        return "CUDA", {"Precision": "mixed"}
    return "CPU", {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = select_device()
    print(f"selected device: {dev}")
    for k, v in describe_device(dev).items():
        print(f"  {k}: {v}")
    plat, props = select_openmm_platform()
    print(f"openmm platform: {plat}  props={props}")
