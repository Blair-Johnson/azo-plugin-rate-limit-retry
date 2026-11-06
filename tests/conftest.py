from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in (
    ROOT / "src",
    Path(os.environ.get("AGENT_UTILS_SRC", ROOT.parent / "agent-utils" / "src")),
):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
