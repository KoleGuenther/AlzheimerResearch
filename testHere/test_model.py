"""Compatibility wrapper for the previous quick-start script path."""

__test__ = False

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.alzheimer_research.train_baseline import main


if __name__ == "__main__":
    main()
