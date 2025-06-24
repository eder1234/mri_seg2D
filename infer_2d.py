# infer_2d.py   (new 2-D inference script)
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
import nibabel as nib

from models import get_model
from utils.postproc import largest_component


def load_patient(folder: Path):
    mag = np.load(folder / "mag.npy")      # (16, H, W)
    phase = np.load(folder / "phase.npy")  # (16, H, W)
    x = np.concatenate([mag, phase], axis=0)  # (32, H, W)
    return torch.from_numpy(x).unsqueeze(0).float()  # (1, 32, H, W)


def main(cfg_path, ckpt_path, patient_dir, out_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    with torch.no_grad():
        vol = load_patient(Path(patient_dir)).to(device)
        logits = model(vol)          # (1, 2, H, W)
        prob = torch.softmax(logits, 1)[0, 1]  # (H, W)
        mask = (prob.cpu().numpy() > 0.5).astype(np.uint8)
        mask = largest_component(mask)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(mask, affine=np.eye(4))
    nib.save(nii, out_path)
    print(f"Saved 2-D mask to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    main(args.config, args.ckpt, args.input, args.out)
