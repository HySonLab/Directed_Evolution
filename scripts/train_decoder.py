import argparse
import os
import torch
from functools import partial
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks
from torch.optim import Adam
from de.dataio.proteins import ProteinsDataModule
from de.predictors.attention.module import ESM2_Attention, ESM2DecoderModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train decoder.")
    parser.add_argument("--data_file",
                        type=str,
                        default="/home/thanhtvt1/workspace/Directed_Evolution/data/proteins/avGFP/avGFP.csv",
                        help="Path to data directory.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="avGFP",
                        help="Name of trained dataset.")
    parser.add_argument("--pretrained_encoder",
                        type=str,
                        default="facebook/esm2_t12_35M_UR50D",
                        help="Path to pretrained encoder.")
    parser.add_argument("--dec_hidden_dim",
                        type=int,
                        default=1024,
                        help="Hidden dim of decoder.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        help="Checkpoint of model.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="../exps",
                        help="Path to output directory.")
    parser.add_argument("--grad_accum_steps",
                        type=int,
                        default=1,
                        help="No. updates steps to accumulate the gradient.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=30,
                        help="Number of epochs.")
    parser.add_argument("--wandb_project",
                        type=str,
                        default="directed_evolution",
                        help="WandB project's name.")
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--set_seed_only",
                        action="store_true",
                        help="Whether to not set deterministic flag.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=64,
                        help="No. workers.")
    parser.add_argument("--num_ckpts",
                        type=int,
                        default=5,
                        help="Maximum no. checkpoints can be saved.")
    parser.add_argument("--log_interval",
                        type=int,
                        default=100,
                        help="How often to log within steps.")
    parser.add_argument("--precision",
                        type=str,
                        choices=["highest", "high", "medium"],
                        default="highest",
                        help="Internal precision of float32 matrix multiplications.")
    args = parser.parse_args()
    return args


def init_model(pretrained_encoder, hidden_dim):
    model = ESM2_Attention(pretrained_encoder, hidden_dim)
    tokenizer = model.tokenizer
    model.freeze_encoder()
    return model, tokenizer


def train(args):
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)
    accelerator = "cpu" if args.devices == "-1" else "gpu"

    # Load model
    model, tokenizer = init_model(args.pretrained_encoder, args.dec_hidden_dim)
    # Init optimizer
    optim = partial(Adam, lr=args.lr)

    # ================== #
    # ====== Data ====== #
    # ================== #
    datamodule = ProteinsDataModule(
        csv_file=args.data_file,
        tokenizer=tokenizer,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # ==================== #
    #  ====== Model ====== #
    # ==================== #
    module = ESM2DecoderModule(model, optim)

    # ====================== #
    # ====== Training ====== #
    # ====================== #
    logger_list = [
        loggers.CSVLogger(args.output_dir),
        loggers.WandbLogger(save_dir=args.output_dir,
                            project=args.wandb_project)
    ]
    callback_list = [
        callbacks.RichModelSummary(),
        callbacks.RichProgressBar(),
        callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename=f"esm_dec-{args.dataset_name}_" + "{epoch:02d}-{val_loss:.3f}",
            monitor="val_loss",
            verbose=True,
            save_top_k=args.num_ckpts,
            save_weights_only=False,
            every_n_epochs=1,
        )
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=[int(d) for d in args.devices.split(",")],
        max_epochs=args.num_epochs,
        log_every_n_steps=args.log_interval,
        accumulate_grad_batches=args.grad_accum_steps,
        deterministic=not args.set_seed_only,
        default_root_dir=args.output_dir,
        logger=logger_list,
        callbacks=callback_list,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
