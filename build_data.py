#!/usr/bin/env python
# build_data.py
#
# Create a tidy data/ folder from the raw export where:
#   └─ <patient>/
#        ├─ FFE/   (magnitude slices: imgXXX_##.npy)
#        ├─ PHASE/ (phase slices)
#        ├─ mask.npy
#        └─ match.json   (optional – metadata)
#
# ----------------------------------------------------
# Example (PowerShell, keep the ^ line-continuations):
# > python build_data.py ^
#       --src "C:\Users\Machine Learning\Documents\GitHub\neuro_segmentation\output" ^
#       --dst "C:\Users\Machine Learning\Documents\GitHub\csf_pc_mri\data"
#
# Result:
# data/
# └─ Patient_…/
#      ├─ mag.npy    (D, H, W)
#      ├─ phase.npy
#      ├─ mask.npy
#      └─ meta.json  (copied from match.json, if present)
# ----------------------------------------------------

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np

# Regex to pull the numeric slice index from img701_00.npy, img701_12.npy, …
SLICE_RE = re.compile(r'img\d+_(\d+)\.npy$', re.IGNORECASE)


def sorted_npy_list(folder: Path) -> List[Path]:
    """Return *.npy files sorted by their numeric suffix (00, 01, 02…)."""
    files = [f for f in folder.glob('*.npy') if f.is_file()]
    return sorted(files, key=lambda p: int(SLICE_RE.search(p.name).group(1)))


def stack_slices(files: List[Path]) -> np.ndarray:
    """Load slices and stack into a (D, H, W) volume."""
    slices = [np.load(f) for f in files]
    return np.stack(slices, axis=0)


def process_patient(patient_dir: Path, dst_root: Path) -> None:
    pid = patient_dir.name
    ffe_dir, phase_dir = patient_dir / 'FFE', patient_dir / 'PHASE'

    if not (ffe_dir.exists() and phase_dir.exists()):
        print(f'[WARN] {pid}: missing FFE or PHASE folder – skipped')
        return

    mag   = stack_slices(sorted_npy_list(ffe_dir))
    phase = stack_slices(sorted_npy_list(phase_dir))
    mask_file = patient_dir / 'mask.npy'
    mask = np.load(mask_file) if mask_file.exists() else None

    out_dir = dst_root / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'mag.npy', mag)
    np.save(out_dir / 'phase.npy', phase)
    if mask is not None:
        np.save(out_dir / 'mask.npy', mask)

    # copy match.json → meta.json (optional but handy)
    src_meta = patient_dir / 'match.json'
    if src_meta.exists():
        with open(src_meta) as f:
            meta = json.load(f)
        with open(out_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    print(f'[OK] {pid:<30} slices={mag.shape[0]}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Re-package raw PC-MRI slices into neat volumes.')
    parser.add_argument('--src', required=True, type=Path, help='Path to the raw “output” folder')
    parser.add_argument('--dst', required=True, type=Path, help='Destination path for the new data/ folder')
    args = parser.parse_args()

    patients = [p for p in args.src.iterdir() if p.is_dir()]
    print(f'Found {len(patients)} patient folders under {args.src}')
    args.dst.mkdir(parents=True, exist_ok=True)

    for p in patients:
        process_patient(p, args.dst)


if __name__ == '__main__':
    main()
