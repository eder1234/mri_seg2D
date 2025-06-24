# datasets/pcmri_csf.py  (replace current file)
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import random

# --- replace the helper at the top ---------------------------------
def random_flip(x, y):
    """
    Horizontal = left–right  (axis=-1)
    Vertical   = up–down     (axis=-2)
    Works for:
        x : (C, H, W)
        y : (H, W)
    """
    # horizontal
    if random.random() < 0.5:
        x = np.flip(x, axis=-1)   # width
        y = np.flip(y, axis=-1)
    # vertical
    if random.random() < 0.5:
        x = np.flip(x, axis=-2)   # height
        y = np.flip(y, axis=-2)
    return x.copy(), y.copy()     # drop negative strides

def random_translate(x, y, max_shift=8):
    """
    Randomly translate image and mask by up to max_shift pixels.
    """
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    x = np.roll(x, shift=shift_x, axis=-1)
    y = np.roll(y, shift=shift_x, axis=-1)
    x = np.roll(x, shift=shift_y, axis=-2)
    y = np.roll(y, shift=shift_y, axis=-2)
    return x, y

def random_rotate(x, y, angles=[0, 90, 180, 270]):
    """
    Randomly rotate image and mask by one of the specified angles.
    """
    angle = random.choice(angles)
    k = angle // 90
    x = np.rot90(x, k, axes=(-2, -1))
    y = np.rot90(y, k, axes=(-2, -1))
    return x, y

def random_zoom(x, y, zoom_range=(0.9, 1.1)):
    """
    Randomly zoom image and mask.
    """
    from scipy.ndimage import zoom as scipy_zoom  # <-- move import here
    zoom_factor = random.uniform(*zoom_range)
    if zoom_factor == 1.0:
        return x, y
    

    # Zoom x (C, H, W)
    zoomed_x = []
    for c in range(x.shape[0]):
        zoomed = scipy_zoom(x[c], zoom_factor, order=1)
        # Center crop or pad to original size
        if zoomed.shape[0] > x.shape[1]:
            start = (zoomed.shape[0] - x.shape[1]) // 2
            zoomed = zoomed[start:start + x.shape[1], start:start + x.shape[2]]
        else:
            pad = (x.shape[1] - zoomed.shape[0]) // 2
            zoomed = np.pad(zoomed, ((pad, x.shape[1] - zoomed.shape[0] - pad),
                                     (pad, x.shape[2] - zoomed.shape[1] - pad)), mode='constant')
        zoomed_x.append(zoomed)
    x = np.stack(zoomed_x, axis=0)

    # Zoom y (H, W)
    zoomed_y = scipy_zoom(y, zoom_factor, order=0)
    if zoomed_y.shape[0] > y.shape[0]:
        start = (zoomed_y.shape[0] - y.shape[0]) // 2
        zoomed_y = zoomed_y[start:start + y.shape[0], start:start + y.shape[1]]
    else:
        pad = (y.shape[0] - zoomed_y.shape[0]) // 2
        zoomed_y = np.pad(zoomed_y, ((pad, y.shape[0] - zoomed_y.shape[0] - pad),
                                     (pad, y.shape[1] - zoomed_y.shape[1] - pad)), mode='constant')
    y = zoomed_y

    return x, y


class PCMRICSF(Dataset):
    """
    One sample per patient:
        x : (32, H=64, W=64)  – 16 mag + 16 phase frames
        y : (H,  W)           – 2-D mask
    """

    def __init__(self, root, id_list, augment=True, modality="both"):
        self.root = Path(root)
        self.ids = id_list
        self.augment = augment
        self.modality = modality

    def _load_subject(self, pid):
        p = self.root / pid
        mag   = np.load(p / "mag.npy")      # (32, 64, 64)
        phase = np.load(p / "phase.npy")    # (32, 64, 64)
        mask  = np.load(p / "mask.npy")     # (64, 64)

        if self.modality == "mag":
            x = mag                         # (32, 64, 64)
        elif self.modality == "phase":
            x = phase
        else:                               # "both"
            x = np.concatenate([mag, phase], axis=0)  # (64, 64, 64)
        return x, mask

    # datasets/pcmri_csf.py   (patch __getitem__)
    def __getitem__(self, idx):
        x_np, y_np = self._load_subject(self.ids[idx])

        if self.augment:
            x_np, y_np = random_flip(x_np, y_np)

        # ensure positive strides even if no aug happened
        x_np = np.ascontiguousarray(x_np)
        y_np = np.ascontiguousarray(y_np)

        x = torch.from_numpy(x_np).float()   # (32 or 64, 64, 64)
        y = torch.from_numpy(y_np).long()    # (64, 64)
        return x, y


    def __len__(self):
        return len(self.ids)
