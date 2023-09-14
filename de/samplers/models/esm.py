import torch
from transformers import AutoTokenizer, EsmForMaskedLM, BatchEncoding, EsmConfig, EsmTokenizer
from typing import List


class ESM2(torch.nn.Module):
    def __init__(self,
                 vocab_file: str,
                 config: EsmConfig = None,
                 pretrained_model_name_or_path: str = None,
                 freeze_until_layer: str = None,
                 device: str = "cpu"):
        """
        Args:
            vocab_file (str): Path to vocabulary file.
            config (EsmConfig): `EsmConfig` object.
                One of `config` or `pretrained_model_name_or_path` must be specified.
            pretrained_model_name_or_path (str): Pre-trained model to load.
                One of `config` or `pretrained_model_name_or_path` must be specified.
            freeze_until_layer (str): Freeze the model from bottom to `freeze_until_layer`.
            device (str): device to be used.
        """
        super(ESM2, self).__init__()
        if config is None and pretrained_model_name_or_path is None:
            raise ValueError("`config` or `pretrained_model_name_or_path` must be specified.")

        self.device = device
        if pretrained_model_name_or_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            self.model = EsmForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        else:
            self.tokenizer = EsmTokenizer(vocab_file)
            self.model = EsmForMaskedLM(config)

        if pretrained_model_name_or_path and freeze_until_layer:
            self._freeze_model(freeze_until_layer)
        self.model.to(self.device)

    def _freeze_model(self, freeze_until_layer: str):
        """Freeze layers for fine-tuning"""
        done_freeze = False
        for name, params in self.model.named_parameters():
            if name == freeze_until_layer:
                done_freeze = True
            if not done_freeze:
                params.requires_grad = False
            else:
                params.requires_grad = True

    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [population].

        Returns:
            encoded_inputs (BatchEncoding): a BatchEncoding object.
        """
        encoded_inputs = self.tokenizer(inputs, add_special_tokens=False, return_tensors="pt")
        return encoded_inputs.to(self.device)

    def decode(self, tokens: torch.Tensor) -> List[str]:
        """Decode predicted tokens into alphabet characters

        Args:
            tokens (torch.Tensor): Predicted tokens of shape [batch, sequence_length]

        Returns:
            (List[str]): Predicted characters."""
        return self.tokenizer.batch_decode(tokens)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """Forward pass of ESM2 model

        Args:
            inputs (BatchEncoding): Output of tokenizer.

        Returns:
            logits (torch.Tensor): Logits.
        """
        results = self.model(**inputs)
        return results
