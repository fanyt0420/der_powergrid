"""Run the unified multiyear weather benchmark; no training or QSTS changes."""
from pathlib import Path
import os
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.unified_benchmark import main

if __name__ == "__main__":
    main()
