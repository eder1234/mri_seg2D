from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt
from subject_dataset import SingleSubjectDataset, pad_to_full, overlay

# Parameters – edit as needed
CONFIG_YAML = "config.yaml"
WEIGHTS = "outputs/checkpoints/best.pt"
OUT_DIR = "outputs/preds_single"     # *_pred.npy and *_probs.npy
PROB_THRESH = 0.5
cfg = load_yaml(CONFIG_YAML)
crop_size = cfg["data"]["crop_size"]

# list the folder names in data/test
import os

folder_names = os.listdir("data/test")
print("Available subjects:", folder_names)

for subject_id in folder_names:
    SUBJECT_ID = subject_id
    FIG_PATH = f"outputs/figures/{SUBJECT_ID}_single.png"
    ds = SingleSubjectDataset(Path(cfg["data"]["root"]) / cfg["data"]["test_dir"],
                            SUBJECT_ID, crop_size)
    sample = ds[0]
    img = sample["image"].unsqueeze(0)            # (1,64,80,80)
    gt_crop = sample["mask"].squeeze().numpy()
    gt_full = pad_to_full(gt_crop, crop_size)
    phase0_full = np.load(ds.subj_path / "phase.npy")[0]
    mag0_full   = np.load(ds.subj_path / "mag.npy")[0]
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(in_channels=cfg["model"]["in_channels"],
                out_channels=cfg["model"]["out_channels"],
                base_channels=cfg["model"]["base_channels"]).to(device)
    ckpt = load_ckpt(WEIGHTS, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Predict
    with torch.no_grad():
        logits = model(img.to(device))
        probs_crop = torch.sigmoid(logits).cpu().numpy()[0,0]
    probs_full = pad_to_full(probs_crop, crop_size)
    pred_mask = (probs_full >= PROB_THRESH).astype(np.uint8)

    # Save outputs
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{SUBJECT_ID}_probs.npy", probs_full.astype(np.float32))
    np.save(out_dir / f"{SUBJECT_ID}_pred.npy", pred_mask.astype(np.uint8))
    print("Saved probability and binary mask to", out_dir)
