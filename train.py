# train.py
import argparse
import yaml
from pathlib import Path
import torch
from tqdm import tqdm

from datasets import get_dataloaders
from models import get_model
from utils.losses import get_loss
from utils.metrics import DiceMeter
from utils.logger import TBLogger
from utils.seed import set_determinism


def train_one_epoch(model, loader, loss_fn, optim, scaler, device):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = loss_fn(logits, y.unsqueeze(1))
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def validate(model, loader, dice_meter, device):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = torch.softmax(model(x), 1)
            preds = (logits > 0.5).int()
            dice_meter.update(preds, y.unsqueeze(1))
    return dice_meter.compute()


def main(cfg_path, fold):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_determinism(cfg.get("seed", 42))

    train_loader, val_loader = get_dataloaders(cfg, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)

    loss_fn = get_loss(cfg)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg["train"]["scheduler_T_max"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])
    dice_meter = DiceMeter()
    logger = TBLogger(f"runs/{cfg['experiment']}/fold{fold}")

    best_dice = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(cfg["train"]["epochs"]):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optim, scaler, device)
        scheduler.step()
        if epoch % cfg["logging"]["interval"] == 0:
            dice = validate(model, val_loader, dice_meter, device)
            logger.log_scalar("loss/train", tr_loss, epoch)
            logger.log_scalar("dice/val", dice, epoch)
            if dice > best_dice:
                best_dice = dice
                torch.save(
                    model.state_dict(),
                    ckpt_dir / f"{cfg['experiment']}_fold{fold}_best.pt",
                )
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    main(args.config, args.fold)
