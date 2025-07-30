from pathlib import Path
from src import CSFVolumeDataset, UNet2D
from src.utils.misc import load_yaml, load_ckpt
import numpy as np

# Single-subject dataset wrapper reusing preprocessing
class SingleSubjectDataset(CSFVolumeDataset):
    def __init__(self, root_dir: Path, subject_id: str, crop_size: int):
        super().__init__(root_dir=root_dir, split="test", crop_size=crop_size)
        idxs = [i for i, p in enumerate(self.subjects) if p.name == subject_id]
        if not idxs:
            raise FileNotFoundError(f"No subject '{subject_id}' under {root_dir}")
        self.keep = idxs[0]
        self.subj_path = self.subjects[self.keep]
    def __len__(self): return 1
    def __getitem__(self, idx): return super().__getitem__(self.keep)

def pad_to_full(mask_crop: np.ndarray, crop_size: int, full_size: int = 240) -> np.ndarray:
    pad = (full_size - crop_size) // 2
    full = np.zeros((full_size, full_size), dtype=mask_crop.dtype)
    full[pad: pad + crop_size, pad: pad + crop_size] = mask_crop
    return full

def overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = .4) -> np.ndarray:
    img_norm = (img - img.min()) / (img.ptp() + 1e-8)
    rgb = np.stack([img_norm]*3, -1)
    mask_gt = gt > .5
    rgb[..., 0][mask_gt] = (1-alpha)*rgb[...,0][mask_gt] + alpha
    rgb[..., 1][mask_gt] = (1-alpha)*rgb[...,1][mask_gt] + alpha
    mask_pred = pred > .5
    rgb[..., 2][mask_pred] = (1-alpha)*rgb[...,2][mask_pred] + alpha
    return np.clip(rgb, 0, 1)