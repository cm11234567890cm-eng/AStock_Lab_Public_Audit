from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    # Deliberately loopback-only. Do not replace with 0.0.0.0 without user approval.
    uvicorn.run("src.dashboard.app:app", host="127.0.0.1", port=8000, reload=False, access_log=True)
