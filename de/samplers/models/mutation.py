import torch
from functools import partial
from lightning import LightningModule
from typing import Tuple
from torch.optim import AdamW, Adam
from ...trainers.optimizers import Lion


class MutationModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optim_type: str,
                 scheduler: torch.optim.lr_scheduler = None,
                 lr: float = 1e-4,
                 betas: Tuple[float, float] = (0.95, 0.98),
                 weight_decay: float = 0.01):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"])

        self.net = net
        self.optim_type = optim_type
        self.scheduler = scheduler

    def forward(self, x):
        return self.net(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.net(**batch).loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.net(**batch).loss
        self.log("valid/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.net(**batch).loss
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        part_optim = self.load_optimizer_partially()
        optimizer = part_optim(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def load_optimizer_partially(self):
        optim_kwargs = {
            "lr": self.hparams.lr,
            "betas": self.hparams.betas,
            "weight_decay": self.hparams.weight_decay
        }
        if self.hparams.optim_type == "lion":
            partial_optimizer = partial(Lion, **optim_kwargs)
        elif self.hparams.optim_type == "adamw":
            partial_optimizer = partial(AdamW, **optim_kwargs)
        else:
            partial_optimizer = partial(Adam, **optim_kwargs)

        return partial_optimizer
