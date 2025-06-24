# utils/logger.py
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class TBLogger:
    def __init__(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.w.add_scalar(tag, value, step)

    def close(self):
        self.w.close()
