# check_shapes.py
from pathlib import Path
import numpy as np

root = Path("data")          # adapt if you moved the data folder
for p in sorted(root.iterdir()):
    if not p.is_dir():
        continue
    try:
        mag   = np.load(p / "mag.npy")   # (D, H, W)
        phase = np.load(p / "phase.npy")
        mask  = np.load(p / "mask.npy")
        print(f"{p.name:<35}  mag{mag.shape}  phase{phase.shape}  mask{mask.shape}")
    except Exception as e:
        print(f"{p.name:<35}  !!! error: {e}")
