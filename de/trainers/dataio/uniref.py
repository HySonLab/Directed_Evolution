import pandas as pd
import os
import torch
from datasets import load_dataset, Features, Sequence, Value
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Dict
# from ..batch_samplers import BucketBatchSampler


class UniRefDataset(Dataset):

    def __init__(self,
                 csv_file: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024,):
        """
        Args:
            csv_file (str): Path to the csv file.
        """
        uniref = pd.read_csv(csv_file)
        sequences = uniref["text"].tolist()

        batch_encoding = tokenizer(sequences,
                                   add_special_tokens=True,
                                   truncation=True,
                                   max_length=max_length)
        examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.examples[idx]


class UniRefDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 1024,
                 pad_to_max_length: bool = True,
                 overwrite_cache: bool = False,
                 mlm_probability: float = 0.15,
                 train_batch_size: int = 32,
                 valid_batch_size: int = 32,
                 test_batch_size: int = 32,
                 preprocess_num_workers: int = 128,
                 dataloader_num_workers: int = 128,):
        super().__init__()
        self.data_dir = data_dir
        self.train_csv = os.path.join(data_dir, "train.csv")
        self.val_csv = os.path.join(data_dir, "valid.csv")
        self.test_csv = os.path.join(data_dir, "test.csv")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.overwrite_cache = overwrite_cache
        self.mlm_probability = mlm_probability
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.preprocess_num_workers = preprocess_num_workers
        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage):
        data_files = {
            "train": self.train_csv,
            "validation": self.val_csv,
            "test": self.test_csv
        }
        datasets = load_dataset(self.data_dir, data_files=data_files)

        padding = "max_length" if self.pad_to_max_length else "longest"
        features = Features({
            "input_ids": Sequence(Value("uint32")),
            "token_type_ids": Sequence(Value("uint8")),
            "attention_mask": Sequence(Value("uint8")),
            "special_tokens_mask": Sequence(Value("uint8")),
        })

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=self.max_seq_length,
                return_special_tokens_mask=True
            )

        # tokenized_datasets = datasets.map(
        #     tokenize_function,
        #     batched=True,
        #     num_proc=self.preprocess_num_workers,
        #     remove_columns=["text"],
        #     load_from_cache_file=not self.overwrite_cache,
        #     features=features,
        # )
        tokenized_datasets = datasets.with_transform(tokenize_function)

        self.train_dataset = tokenized_datasets["train"]
        self.val_dataset = tokenized_datasets["validation"]
        self.test_dataset = tokenized_datasets["test"]
        self.data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, mlm_probability=self.mlm_probability
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.dataloader_num_workers,
            shuffle=False,
        )
