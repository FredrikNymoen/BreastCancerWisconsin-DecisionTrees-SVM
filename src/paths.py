"""
Centralized paths to keep figures/outputs organized.
Adjust ROOT if you run this outside the sandbox.
"""
from pathlib import Path

ROOT = Path.cwd().parents[1] 

# Folders
SRC = ROOT / "src"
FIGURES = SRC / "figures"
OUTPUTS = SRC / "outputs"
