"""
Centralized paths to keep figures/outputs organized.
Adjust ROOT if you run this outside the sandbox.
"""
from pathlib import Path

# Root of the project (change if needed)
ROOT = Path(".")

# Folders
SRC = ROOT / "src"
DATA = ROOT / "data"
FIGURES = SRC / "figures"
OUTPUTS = SRC / "outputs"

# Ensure folders exist when imported
for p in [SRC, DATA, FIGURES, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)