# utils/seed.py
import random
import numpy as np
import torch
import monai


def set_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    monai.utils.set_determinism(seed)
