# models/unet2d.py
import monai


def build_unet2d(cfg):
    p = cfg["model"]
    return monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=p["in_channels"],
        out_channels=p["out_channels"],
        channels=tuple(p["channels"]),
        strides=tuple(p["strides"]),
        num_res_units=p.get("num_res_units", 2),
        norm="INSTANCE",
    )
