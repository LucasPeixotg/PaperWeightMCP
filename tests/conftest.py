"""Put `src/` on sys.path.

Imports in this repo are rooted at src/ (`from config import settings`), and
there is no pyproject.toml to install the package from, so the tests reproduce
what running `python src/server.py` does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
