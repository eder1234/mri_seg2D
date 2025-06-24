# viz_overlay.py
# Usage:
#   python viz_overlay.py --subj data/Patient_X --mask outputs/Patient_X.nii.gz
#   (accepts .npy or .nii/.nii.gz)

import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib

def load_mask(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path)                    # (D, H, W)
    elif ext in (".nii", ".gz"):
        return nib.load(path).get_fdata().astype(bool)  # (D, H, W)
    else:
        raise ValueError(f"Unrecognised mask format: {path}")

def main(subject_dir, mask_path):
    subj = Path(subject_dir)
    mag  = np.load(subj / "mag.npy")            # (D, H, W)
    mask = load_mask(Path(mask_path))           # (D, H, W)

    d = mag.shape[0]                            # number of slices
    cols = 8
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i in range(d):
        ax = axes.flatten()[i]
        ax.imshow(mag[i], cmap="gray")
        ax.imshow(np.ma.masked_where(mask[i]==0, mask[i]),
                  cmap="autumn", alpha=0.4)
        ax.axis("off")
        ax.set_title(f"z={i}")
    for j in range(i+1, rows*cols):
        axes.flatten()[j].axis("off")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subj", required=True)
    p.add_argument("--mask", required=True)
    args = p.parse_args()
    main(args.subj, args.mask)
