# utils/metrics.py
import monai


class DiceMeter:
    def __init__(self):
        self.metric = monai.metrics.DiceMetric(
            include_background=False, reduction="mean_batch", get_not_nans=False
        )

    def update(self, preds, targets):
        self.metric(preds, targets)

    def compute(self):
        v = self.metric.aggregate().item()
        self.metric.reset()
        return v
