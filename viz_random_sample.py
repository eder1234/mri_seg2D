# viz_random_sample.py
"""
Visualise the predicted CSF mask for a random validation subject
of a chosen cross-validation fold (0-4).

Usage examples
--------------
# multiframe model
python viz_random_sample.py --fold 1 --config configs/unet2d_multiframe.yaml

# phase-only model
python viz_random_sample.py --fold 1 --config configs/unet2d_phase.yaml

# list the validation subjects only
python viz_random_sample.py --fold 1 --config configs/unet2d_phase.yaml --list-val
"""
from pathlib import Path
import argparse, random, yaml, time
import numpy as np
import torch, matplotlib.pyplot as plt

from datasets import _split_ids
from models   import get_model
from utils.postproc import largest_component


# --------------------------------------------------------------------- helpers
def load_patient(folder: Path, modality: str):
    """Return np.ndarray(channels, H, W) according to the modality flag."""
    mag   = np.load(folder / "mag.npy")      # (32, H, W)
    phase = np.load(folder / "phase.npy")    # (32, H, W)

    if modality == "mag":
        return mag
    if modality == "phase":
        return phase
    # "both"
    return np.concatenate([mag, phase], axis=0)   # (64, H, W)


def predict_mask(model, vol4d, device):
    model.eval()
    with torch.no_grad():
        logits = model(vol4d.to(device))
        prob   = torch.softmax(logits, 1)[0, 1]   # (H, W)
        mask   = (prob.cpu().numpy() > 0.5).astype(np.uint8)
    return largest_component(mask)


def overlay_grid(x, mask, modality):
    """
    If modality=="both"  → 64-panel grid   (mag0, phase0, …)
    Otherwise            → 32-panel grid   (single channel set)
    """
    n_src = x.shape[0]        # 32 or 64
    rows, cols = 8, n_src // 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in range(n_src):
        ax = axes[i]
        ax.imshow(x[i], cmap="gray")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.3)
        lab = f"{['M','P'][i%2]} {i//2}" if modality == "both" else f"{i}"
        ax.set_title(lab, fontsize=7)
        ax.axis("off")

    for ax in axes[n_src:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------- main
def main(cfg_path: str, fold: int, list_only: bool):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    data_root = Path(cfg["data"]["root"])
    modality  = cfg["data"].get("modality", "both")  # mag / phase / both

    # pick a validation subject for this fold
    _, val_ids = _split_ids(data_root, cfg["folds"], fold, cfg["seed"])

    if list_only:
        print(f"Validation subjects (fold {fold}):")
        print("\n".join(val_ids))
        return

    random.seed(time.time())
    sid = random.choice(val_ids)
    print(f"Selected validation subject: {sid}")

    # build checkpoint path from experiment name in YAML
    exp = cfg["experiment"]
    ckpt = Path(f"checkpoints/{exp}_fold{fold}_best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # load input tensor
    x_np = load_patient(data_root / sid, modality)         # (C, 64, 64)
    x_t  = torch.from_numpy(np.ascontiguousarray(x_np)).unsqueeze(0).float()

    # predict & visualise
    mask = predict_mask(model, x_t, device)
    overlay_grid(x_np, mask, modality)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",   type=int, choices=range(5), required=True)
    ap.add_argument("--config", type=str,
                    default="configs/unet2d_multiframe.yaml",
                    help="YAML config path")
    ap.add_argument("--list-val", action="store_true",
                    help="List validation IDs and quit")
    args = ap.parse_args()
    main(args.config, args.fold, args.list_val)
