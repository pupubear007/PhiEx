"""
phytologue.device — compute device selection.

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
    import torch   # imported lazily so this module is cheap to import

    forced = override or os.environ.get("PHYTOLOGUE_DEVICE")
    if forced:
        forced = forced.lower()
        if forced not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"PHYTOLOGUE_DEVICE={forced!r} not in cpu/mps/cuda")
        log.info("device forced via env: %s", forced)
        return forced

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        log.info("device auto-selected: mps (Apple Silicon %s)", platform.machine())
        return "mps"

    log.info("device auto-selected: cpu (%s)", platform.platform())
    return "cpu"


def describe_device(device: str) -> dict:
    """Human-readable info about the chosen device.  Goes into the reasoning
    ticker so every ϕ → ∃ call logs which substrate it ran on."""
    import torch
    info = {"device": device, "torch": torch.__version__}
    if device == "mps":
        info["backend"] = "Metal Performance Shaders"
        info["fallback"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    elif device == "cuda":
        info["backend"] = f"CUDA {torch.version.cuda}"
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
    else:
        info["backend"] = "CPU"
        info["threads"] = torch.get_num_threads()
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dev = select_device()
    print(f"selected device: {dev}")
    for k, v in describe_device(dev).items():
        print(f"  {k}: {v}")
