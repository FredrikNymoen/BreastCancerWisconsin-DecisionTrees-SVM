"""
Centralized paths to keep figures/outputs organized.
Adjust ROOT if you run this outside the sandbox.
"""
from pathlib import Path

ROOT = Path.cwd().parents[0] 

# Folders
SRC = ROOT / "src"
FIGURES = SRC / "figures"
OUTPUTS = SRC / "outputs"

# Ensure folders exist when imported
for p in [SRC, FIGURES, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)