# viz_overlay_dual.py  (optional)
# --- top of viz_overlay_dual.py  (fix) -----------------
from pathlib import Path
import nibabel as nib
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_mask(path):
    return (nib.load(path).get_fdata() if path.suffix in (".nii", ".gz")
            else np.load(path)).astype(bool)

def main(subj, mask_path, mode="both"):
    subj = Path(subj)
    mag   = np.load(subj / "mag.npy")      # (D,H,W)
    phase = np.load(subj / "phase.npy")
    mask  = load_mask(Path(mask_path))     # (D,H,W)

    vols = {"mag": mag, "phase": phase}
    imgs = [vols["mag"]] if mode == "mag" else \
           [vols["phase"]] if mode == "phase" else \
           [mag, phase]                         # mode == "both"

    rows   = len(imgs)
    slices = mag.shape[0]
    cols   = 8
    r_grid = rows * int(np.ceil(slices / cols))

    fig, axes = plt.subplots(r_grid, cols, figsize=(cols*2, r_grid*2))
    axes = axes.flatten()

    for v, vol in enumerate(imgs):
        for z in range(slices):
            ax = axes[v*cols*int(np.ceil(slices/cols)) + z]
            ax.imshow(vol[z], cmap="gray")
            ax.imshow(np.ma.masked_where(mask[z]==0, mask[z]),
                      cmap="autumn", alpha=0.4)
            ax.set_title(f"{['mag','phase'][v]} z={z}")
            ax.axis("off")

    for ax in axes[len(imgs)*slices:]:
        ax.axis("off")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subj", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--mode", choices=["mag", "phase", "both"], default="both")
    args = p.parse_args()
    main(args.subj, args.mask, args.mode)
