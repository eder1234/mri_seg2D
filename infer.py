# infer.py
import argparse
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
import yaml
from monai.inferers import SlidingWindowInferer

from models import get_model
from utils.postproc import largest_component


def load_volume(patient_dir: Path):
    mag = np.load(patient_dir / "mag.npy")
    phase = np.load(patient_dir / "phase.npy")
    x = np.stack([mag, phase], axis=0)  # (2, D, H, W)
    return torch.from_numpy(x).unsqueeze(0).float()  # (1, 2, D, H, W)


def save_nifti(mask: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)   # NEW
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
    nib.save(nii, out_path)

def main(ckpt_path, config_path, patient_dir, out_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        vol = load_volume(Path(patient_dir)).to(device)
        inferer = SlidingWindowInferer(roi_size=cfg["data"]["patch_size"], overlap=0.5)
        logits = inferer(vol, model)
        prob = torch.softmax(logits, 1)[0, 1]  # foreground prob (D, H, W)
        mask = (prob.cpu().numpy() > 0.5).astype(np.uint8)
        mask = largest_component(mask)

    save_nifti(mask, Path(out_path))
    print(f"Saved mask to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True, help="Patient folder")
    p.add_argument("--out", required=True, help="Output NIfTI path")
    args = p.parse_args()
    main(args.ckpt, args.config, args.input, args.out)
