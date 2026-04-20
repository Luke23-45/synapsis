"""Workspace-local Python startup customization.

Ensures this repository's SYNAPSIS package is importable before any sibling
workspace package with the same top-level name.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent

# Add project root to path for SYNAPSIS imports
root_str = str(_ROOT)
try:
    sys.path.remove(root_str)
except ValueError:
    pass
sys.path.insert(0, root_str)
