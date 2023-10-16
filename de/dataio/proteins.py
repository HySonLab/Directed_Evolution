import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer
from typing import Dict, Tuple


class ProteinDataset(Dataset):

    def __init__(self, csv_file: str, tokenizer: PreTrainedTokenizer, max_length: int = None):
        """
        Args:
            csv_file (str): Path to the csv file.
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequences = self.data.iloc[idx, 0]
        fitnesses = self.data.iloc[idx, 1]
        if isinstance(sequences, pd.Series):
            sequences = sequences.tolist()
            fitnesses = fitnesses.tolist()
        input_ids = self.tokenizer(sequences,
                                   add_special_tokens=True,
                                   truncation=True,
                                   max_length=self.max_length)["input_ids"]
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "fitness": torch.tensor(fitnesses, dtype=torch.float32)}


class ProteinsDataModule(LightningDataModule):

    def __init__(self,
                 csv_file: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = None,
                 train_val_split: Tuple[float, float] = (0.9, 0.1),
                 train_batch_size: int = 32,
                 valid_batch_size: int = 32,
                 num_workers: int = 64,
                 seed: int = 0):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage):
        datasets = ProteinDataset(self.hparams.csv_file,
                                  self.hparams.tokenizer,
                                  self.hparams.max_length)
        self.train_dataset, self.valid_dataset = random_split(
            dataset=datasets,
            lengths=self.hparams.train_val_split,
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.hparams.valid_batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
