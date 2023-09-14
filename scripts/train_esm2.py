import argparse
import os
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks
from transformers import EsmForMaskedLM, EsmTokenizer
from typing import List
from de.common.constants import all_possible_kmers
from de.common.utils import save_vocabulary
from de.samplers.models.mutation import MutationModule
from de.trainers.dataio import UniRefDataModule
from configs import get_esm2_config


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train ESM2 uing MLM.")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/home/thanhtvt1/workspace/Directed_Evolution/data/uniref50",
                        help="Path to data directory.")
    parser.add_argument("--vocab_dir",
                        type=str,
                        default="/home/thanhtvt1/workspace/Directed_Evolution/data/vocabs",
                        help="Path to vocab file directory.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size.")
    parser.add_argument("--k",
                        type=int,
                        default=1,
                        help="Split sequence into multiple tokens with length `k`.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        help="Checkpoint of model.")
    parser.add_argument("--max_length",
                        type=int,
                        default=1024,
                        help="Maximum acceptable length of sequence.")
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
    parser.add_argument("--optim_type",
                        type=str,
                        choices=["adam", "adamw", "lion"],
                        default="lion",
                        help="Optimizer algorithm.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate (Default value is suitable for Lion).")
    parser.add_argument("--beta1",
                        type=float,
                        default=0.95,
                        help="Beta 1 in optimizer (Default value is suitable for Lion).")
    parser.add_argument("--beta2",
                        type=float,
                        default=0.98,
                        help="Beta 2 in optimizer (Default value is suitable for Lion).")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.01,
                        help="Weight decay in optimizer (Default value is suitable for Lion).")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=30,
                        help="Number of epochs.")
    parser.add_argument("--mlm_prob",
                        type=float,
                        default=0.15,
                        help="Mask probability.")
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
    parser.add_argument("--num_ckpts",
                        type=int,
                        default=5,
                        help="Maximum no. checkpoints can be saved.")
    parser.add_argument("--num_workers",
                        type=int,
                        default=128,
                        help="No. workers.")
    parser.add_argument("--log_interval",
                        type=int,
                        default=100,
                        help="How often to log within steps.")
    parser.add_argument("--precision",
                        type=str,
                        choices=["highest", "high", "medium"],
                        default="high",
                        help="Internal precision of float32 matrix multiplications.")
    args = parser.parse_args()
    return args


def init_model(vocab: List[str], args):
    if args.ckpt_path is not None:
        raise ValueError("Current version not support loading checkpoint.")
        # model = EsmForMaskedLM.from_pretrained(args.ckpt_path)
    else:
        config = get_esm2_config(vocab)
        model = EsmForMaskedLM(config)

        vocab_filepath = os.path.join(args.vocab_dir, f"vocab_{args.k}.txt")
        tokenizer = EsmTokenizer(vocab_filepath)

    return model, tokenizer


def train(args):
    """Pre-training model"""

    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.precision)
    accelerator = "cpu" if args.devices == "-1" else "gpu"

    # Process data
    standard_toks = []
    for km in range(1, args.k + 1):
        standard_toks.extend(all_possible_kmers(km))
    vocabs = save_vocabulary(args.vocab_dir, standard_toks, args.k)

    # Load model and tokenizer
    net, tokenizer = init_model(vocabs, args)

    # ================== #
    # ====== Data ====== #
    # ================== #
    data_module = UniRefDataModule(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_length=args.max_length,
        pad_to_max_length=False,
        overwrite_cache=False,
        mlm_probability=args.mlm_prob,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
        preprocess_num_workers=args.num_workers,
        dataloader_num_workers=args.num_workers,
    )

    # ==================== #
    #  ====== Model ====== #
    # ==================== #
    model = MutationModule(
        net,
        optim_type=args.optim_type,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

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
        callbacks.Timer(),
        callbacks.ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="esm2-{epoch:02d}-{val_loss:.3f}",
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

    trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
