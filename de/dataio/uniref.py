import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict


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
