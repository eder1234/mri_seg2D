# ensemble.py
import argparse
from pathlib import Path
import numpy as np
import torch
import yaml
from monai.inferers import SlidingWindowInferer

from models import get_model
from utils.postproc import largest_component


def load_volume(patient_dir: Path):
    mag = np.load(patient_dir / "mag.npy")
    phase = np.load(patient_dir / "phase.npy")
    x = np.stack([mag, phase], axis=0)
    return torch.from_numpy(x).unsqueeze(0).float()


def predict_prob(model, vol, roi_size):
    inferer = SlidingWindowInferer(roi_size=roi_size, overlap=0.5)
    with torch.no_grad():
        logits = inferer(vol, model)
        return torch.softmax(logits, 1)[0, 1].cpu().numpy()  # (D, H, W)


def main(cfg_path, patient_dir, out_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roi_size = cfg["data"]["patch_size"]

    acc = None
    for ckpt in cfg["ensemble"]["ckpts"]:
        model = get_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        vol = load_volume(Path(patient_dir)).to(device)
        prob = predict_prob(model, vol, roi_size)
        acc = prob if acc is None else acc + prob

    prob_avg = acc / len(cfg["ensemble"]["ckpts"])
    mask = (prob_avg > 0.5).astype(np.uint8)
    mask = largest_component(mask)
    np.save(out_path, mask)
    print(f"Ensembled mask saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/ensemble.yaml")
    parser.add_argument("--input", required=True, help="Patient folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.config, args.input, args.out)
