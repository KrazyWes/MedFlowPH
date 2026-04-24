"""Expose PhilGEPS step 01 source directory for imports."""

from __future__ import annotations

import sys
from pathlib import Path

_PHILGEPS_SRC = Path(__file__).resolve().parents[1] / "source_code" / "PhilGEPS"
if str(_PHILGEPS_SRC) not in sys.path:
    sys.path.insert(0, str(_PHILGEPS_SRC))
