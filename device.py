"""
device — top-level shim for `PhiEx.device`.

`PhiEx/device.py` is the single source of truth for device selection
(PyTorch device + OpenMM platform).  This file only re-exports it so that
`import device` from the repository root keeps working.

    python device.py        # print the selected device and OpenMM platform
"""

from PhiEx.device import (  # noqa: F401
    select_device, describe_device, select_openmm_platform,
)

if __name__ == "__main__":
    import runpy
    runpy.run_module("PhiEx.device", run_name="__main__")
