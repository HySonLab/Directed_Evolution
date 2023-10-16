import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.mae import MeanAbsoluteError
from typing import Any
from .decoder import Decoder
from transformers import EsmModel, AutoTokenizer


class ESM2_Attention(nn.Module):
    def __init__(self,
                 pretrained_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D",
                 hidden_dim: int = 512):
        super().__init__()
        self.esm = EsmModel.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        input_dim = self.esm.config.hidden_size
        self.decoder = Decoder(input_dim, hidden_dim)

    def freeze_encoder(self):
        for param in self.esm.parameters():
            param.requires_grad = False

    def forward(self, x):
        enc_out = self.esm(x).last_hidden_state
        output = self.decoder(enc_out)
        return output


class ESM2DecoderModule(LightningModule):
    def __init__(self,
                 net: torch.nn.Module,
                 optimizer: torch.optim.Optimizer):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging error
        self.train_mae = MeanAbsoluteError()
        self.valid_mae = MeanAbsoluteError()
        self.valid_mse = MeanSquaredError()

        # averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far
        self.val_mae_best = MinMetric()
        self.val_mse_best = MinMetric()

    def forward(self, x):
        return self.net(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.valid_mae.reset()
        self.valid_mse.reset()
        self.val_mse_best.reset()
        self.val_mae_best.reset()

    def model_step(self, batch):
        x, y = batch["input_ids"], batch["fitness"]
        y = y.unsqueeze(1)
        pred = self.forward(x)
        loss = self.criterion(pred, y)
        return loss, pred, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)

        # return loss
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.valid_mae(preds, targets)
        self.valid_mse(preds, targets)
        self.log("val_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        mae = self.valid_mae.compute()  # get current mae
        mse = self.valid_mse.compute()  # get current mse
        self.val_mae_best(mae)
        self.val_mse_best(mse)
        self.log("val_mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val_mse_best", self.val_mse_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}

    def infer_fitness(self, representation: torch.Tensor):
        fitness = self.net.decoder(representation)
        return fitness
