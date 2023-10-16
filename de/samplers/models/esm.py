import torch
from transformers import AutoTokenizer, EsmForMaskedLM, BatchEncoding
from typing import List


class ESM2(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path: str = "facebook/esm2_t12_35M_UR50D"):
        """
        Args:
            pretrained_model_name_or_path (str): Pre-trained model to load.
        """
        super(ESM2, self).__init__()
        assert pretrained_model_name_or_path is not None
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = EsmForMaskedLM.from_pretrained(pretrained_model_name_or_path)

    def tokenize(self, inputs: List[str]) -> BatchEncoding:
        """Convert inputs to a format suitable for the model.

        Args:
            inputs (List[str]): A list of protein sequence strings of len [population].

        Returns:
            encoded_inputs (BatchEncoding): a BatchEncoding object.
        """
        encoded_inputs = self.tokenizer(inputs,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        padding=True)
        return encoded_inputs

    def decode(self, tokens: torch.Tensor) -> List[str]:
        """Decode predicted tokens into alphabet characters

        Args:
            tokens (torch.Tensor): Predicted tokens of shape [batch, sequence_length]

        Returns:
            (List[str]): Predicted characters.
        """
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def forward(self, inputs: BatchEncoding) -> torch.Tensor:
        """Forward pass of ESM2 model

        Args:
            inputs (BatchEncoding): Output of tokenizer.

        Returns:
            logits (torch.Tensor): Logits.
        """
        results = self.model(output_hidden_states=True, **inputs)
        return results
