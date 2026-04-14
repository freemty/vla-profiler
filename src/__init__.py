"""vlla — VLM/VLA profiling framework."""

import os
import sys

# Add the core submodule root to sys.path so that `probe_core`
# is importable (the submodule's internal imports use `from probe_core.xxx`).
_CORE_DIR = os.path.join(os.path.dirname(__file__), "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)
