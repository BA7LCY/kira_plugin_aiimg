"""Make the plugin package importable when tests run from the tests dir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
