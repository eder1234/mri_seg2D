# datasets/__init__.py
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from .pcmri_csf import PCMRICSF
from utils.seed import set_determinism
from datasets.pcmri_csf import PCMRICSF

def _split_ids(root, folds, fold_idx, seed=42):
    subjects = sorted([p.name for p in Path(root).iterdir() if p.is_dir()])
    set_determinism(seed)
    fold_size = len(subjects) // folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size
    val_ids = subjects[val_start:val_end]
    train_ids = subjects[:val_start] + subjects[val_end:]
    return train_ids, val_ids


def get_dataloaders(cfg, fold_idx=0):
    data_cfg = cfg["data"]
    train_ids, val_ids = _split_ids(
        data_cfg["root"], cfg["folds"], fold_idx, cfg.get("seed", 42)
    )

    train_ds = PCMRICSF(data_cfg["root"], train_ids,
                        augment=True,  modality=data_cfg.get("modality", "both"))
    val_ds   = PCMRICSF(data_cfg["root"], val_ids,
                        augment=False, modality=data_cfg.get("modality", "both"))

    loader_args = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    return train_loader, val_loader
