import os
from pathlib import Path

# Canonical deployment root. Override with PXDESIGN_ROOT env var if the package
# is installed somewhere other than the default /pxdesign layout.
_PXDESIGN_ROOT = Path(os.environ.get("PXDESIGN_ROOT", "/pxdesign"))

if os.environ.get("PROTENIX_DATA_ROOT_DIR") is None:
    os.environ["PROTENIX_DATA_ROOT_DIR"] = str(
        _PXDESIGN_ROOT / "release_data" / "ccd_cache"
    )

if os.environ.get("TOOL_WEIGHTS_ROOT") is None:
    os.environ["TOOL_WEIGHTS_ROOT"] = str(_PXDESIGN_ROOT / "tool_weights")
