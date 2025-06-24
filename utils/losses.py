# utils/losses.py
import torch
import monai


def get_loss(cfg):
    if cfg["loss"]["type"] == "dice_ce":
        return monai.losses.DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            lambda_dice=cfg["loss"].get("dice_weight", 1.0),
            lambda_ce=cfg["loss"].get("ce_weight", 1.0),
        )
    raise ValueError("Unknown loss type")
