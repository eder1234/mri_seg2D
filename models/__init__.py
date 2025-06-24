# models/__init__.py
from .unet3d import build_unet3d
from .unet2d import build_unet2d


def get_model(cfg):
    name = cfg["model"]["name"].lower()
    if name == "unet3d":
        return build_unet3d(cfg)
    if name == "unet2d":
        return build_unet2d(cfg)
    raise ValueError(f"Unknown model: {name}")
