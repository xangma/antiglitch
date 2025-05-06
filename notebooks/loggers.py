# loggers.py ──────────────────────────────────────────────────────────────
import wandb
from torch.utils.tensorboard import SummaryWriter

class DualLogger:
    """
    Mirrors every add_scalar / add_scalars call to both TensorBoard
    and W&B so you can keep your existing TB dashboards and still get
    W&B’s UI, sweeps, etc.
    """
    def __init__(self, run, tb_writer: SummaryWriter):
        self.run = run
        self.tb  = tb_writer

    # ---- scalar helpers --------------------------------------------------
    def add_scalar(self, tag, value, step):
        self.tb.add_scalar(tag, value, step)
        self.run.log({tag: value}, step=step)

    def add_scalars(self, tag, tag_scalar_dict, step):
        # TensorBoard expects the parent key *outside* the dict, W&B doesn’t,
        # so flatten:  metrics/foo=value  →  {"metrics/foo": value}
        self.tb.add_scalars(tag, tag_scalar_dict, step)
        self.run.log({f"{tag}/{k}" if "/" not in tag else f"{tag}.{k}": v
                      for k, v in tag_scalar_dict.items()}, step=step)

    # ---- generic passthroughs you already rely on -----------------------
    def __getattr__(self, name):
        return getattr(self.tb, name)