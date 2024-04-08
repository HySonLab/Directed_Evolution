import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from typing import List, Union
# from .attention.decoder import Decoder
from de.common.utils import get_mutants
from de.predictors.attention.decoder import Decoder


class ESM1b_Attention1d(nn.Module):

    def __init__(self):
        super(ESM1b_Attention1d, self).__init__()
        self.encoder = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        self.decoder = Decoder(input_dim=1280, hidden_dim=512)

    def forward(self, inputs):
        x = self.encoder(**inputs).last_hidden_state
        x = self.decoder(x)
        return x


class ESM1b_Landscape:
    """
        An ESM-based oracle model to simulate protein fitness landscape.
    """

    def __init__(self, task: str, device: Union[str, torch.device]):
        task_dir_path = os.path.join('./landscape_params/esm1b_landscape', task)
        task_dir_path = os.path.abspath(task_dir_path)
        assert os.path.exists(os.path.join(task_dir_path, 'decoder.pt'))
        self.model = ESM1b_Attention1d()
        self.model.decoder.load_state_dict(
            torch.load(os.path.join(task_dir_path, 'decoder.pt'))
        )
        with open(os.path.join(task_dir_path, 'starting_sequence.json')) as f:
            self.starting_sequence = json.load(f)

        self.tokenizer = self.model.tokenizer
        self.device = device
        self.model.to(self.device)

    def infer_fitness(self, sequences: List[str], batch_size: int = 16, device=None):
        # Input:  - sequences:      [query_batch_size, sequence_length]
        # Output: - fitness_scores: [query_batch_size]

        self.model.eval()
        fitness_scores = []
        seqs = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]
        for seq in seqs:
            inputs = self.tokenizer(seq, return_tensors="pt").to(self.device)
            fitness = self.model(inputs).cpu().tolist()
            fitness_scores.extend(fitness)
            # fitness_scores.append(self.model(inputs).item())
        return fitness_scores


class ESM1v:

    def __init__(self, model_name: str, device, method: str, offset_idx: int):
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.model = EsmModel.from_pretrained(f"facebook/{model_name}")
        self.model.eval()
        self.model = self.model.to(device)
        self.device = device
        self.method = method
        self.offset_idx = offset_idx

    def compute_pppl(self, variants: List[str]):
        log_probs = []
        mask_id = self.tokenizer._token_to_id["<mask>"]
        inputs = self.tokenizer(variants, return_tensors="pt").to(self.device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

        for i in range(1, len(variants[0]) - 1):
            token_ids = input_ids[:, i].unsqueeze(1)
            batch_token_masked = input_ids.clone()
            batch_token_masked[:, i] = mask_id

            with torch.inference_mode():
                logits = self.model(batch_token_masked, attention_mask).last_hidden_state
                token_probs = torch.log_softmax(logits, dim=-1)[:, i]
                token_probs = torch.gather(token_probs, dim=1, index=token_ids)

            log_probs.append(token_probs)

        return torch.sum(torch.concat(log_probs, dim=1), dim=1).cpu().tolist()

    def compute_masked_marginals(self, wt_seq: str, mutants: List[str]):
        all_token_probs = []
        mask_id = self.tokenizer._token_to_id["<mask>"]
        inputs = self.tokenizer(wt_seq, return_tensors="pt").to(self.device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        for i in range(input_ids.size(1)):
            batch_token_masked = input_ids.clone()
            batch_token_masked[:, i] = mask_id

            with torch.inference_mode():
                logits = self.model(batch_token_masked, attention_mask).last_hidden_state
                token_probs = torch.log_softmax(logits, dim=-1)[:, i]

            all_token_probs.append(token_probs)

        token_probs = torch.cat(all_token_probs, dim=0)
        scores = []
        for mutant in mutants:
            ms = mutant.split(":")
            score = 0
            for row in ms:
                if len(row) == 0:
                    continue
                wt, idx, mt = row[0], int(row[1:-1]) - self.offset_idx, row[-1]
                assert wt_seq[idx] == wt

                wt_encoded, mt_encoded = self.tokenizer._token_to_id[wt], self.tokenizer._token_to_id[mt]
                mt_score = token_probs[1 + idx, mt_encoded] - token_probs[1 + idx, wt_encoded]
                score = score + mt_score.item()

            scores.append(score)

        return scores

    def infer_fitness(self, sequences: List[str], wt_seq: str = None, device=None):
        if self.method == "pseudo":
            scores = self.compute_pppl(sequences)
        elif self.method == "masked":
            assert wt_seq is not None, "wt_seq must be provided when using masked marginal."
            mutants = [get_mutants(wt_seq, seq, self.offset_idx) for seq in sequences]
            scores = self.compute_masked_marginals(wt_seq, mutants)
        else:
            raise ValueError("method is not supported")
        return scores


if __name__ == "__main__":
    import sys
    import pandas as pd
    from de.common.utils import get_mutated_sequence

    csv_file = sys.argv[1]

    device = torch.device("cuda:0")
    landscape = ESM1b_Landscape("AAV", device)

    df = pd.read_csv(csv_file)
    df["mutated"] = df.apply(lambda x: get_mutated_sequence(x["WT"], x.mutants), axis=1)
    opt_score = df["score"].tolist()
    mutated_seqs = df["mutated"].tolist()

    scores = landscape.infer_fitness(mutated_seqs)
    results = {"mutated": mutated_seqs, "opt_score": opt_score, "eval_score": scores}
    df = pd.DataFrame.from_dict(results)
    target_path = os.path.join(os.path.dirname(csv_file), "tmp.csv")
    df.to_csv(target_path)
