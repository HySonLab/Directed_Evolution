import numpy as np
import pandas as pd
import re
import torch

from datasets import Dataset
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.utils.data.sampler import SequentialSampler
from transformers import DataCollatorForLanguageModeling, GPT2PreTrainedModel
from typing import Dict, List


AA_vocab = "ACDEFGHIKLMNPQRSTVWY"


def get_mutated_sequence(focus_seq: str,
                         mutant: str,
                         start_idx: int = 1,
                         AA_vocab: str = AA_vocab) -> str:
    """Mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).

    Args:
        focus_seq (str): Input sequence.
        mutant (str): list of mutants applied to input sequence (e.g., "B12F:A83M").
        start_idx (int): Index to start indexing.
        AA_vocab (str): Amino acids.

    Returns:
        (str): mutated sequence.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(
                mutation[1:-1]), mutation[-1]
        except ValueError:
            print("Issue with mutant: " + str(mutation))
        relative_position = position - start_idx
        assert from_AA == focus_seq[relative_position], \
            f"Invalid from_AA or mutant position: {str(mutation)} from_AA {str(str(from_AA))} " \
            f"relative pos: {str(relative_position)} focus_seq: {str(focus_seq)}"
        assert to_AA in AA_vocab, f"Mutant to_AA is invalid: {str(mutation)}"
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def nansum(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs)


def get_optimal_window(mutation_position_relative: int,
                       seq_len_wo_special: int,
                       model_window: int):
    """Selects an optimal sequence window that fits the maximum model context size.

    Args:
        mutation_position_relative (int)
        seq_len_wo_special (int)
        model_window (int)

    Returns:
        (List[int, int])
    """
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0, seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0, model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [
            max(0, mutation_position_relative - half_model_window),
            min(seq_len_wo_special, mutation_position_relative + half_model_window)
        ]


def sequence_replace_single(sequence: str, char_to_replace: str, char_replacements: str):
    char_replacements = list(char_replacements)
    positions = [m.start() for m in re.finditer(char_to_replace, sequence)]
    replacements = np.random.choice(a=char_replacements,
                                    size=len(positions),
                                    replace=True)
    sequence = list(sequence)
    for idx, position in enumerate(positions):
        sequence[position] = replacements[idx]
    return ''.join(sequence)


def sequence_replace(sequences: List[str], char_to_replace: str, char_replacements: str):
    """Replaces all Amino Acids passsed in via char_to_replace (string of AAs)
    with Amino Acids sampled from char_replacements (string of eligible AAs)
    """
    return [sequence_replace_single(sequence, char_to_replace, char_replacements)
            for sequence in sequences]


def update_scores_df(mutated_df: pd.DataFrame, scores: Dict, index: int, batch_size: int):
    mutated_sequence = np.array(mutated_df['mutated_sequence'][index:index + batch_size])
    scores['mutated_sequence'] += list(mutated_sequence)
    mutant = np.array(mutated_df['mutant'][index:index + batch_size])
    scores['mutant'] += list(mutant)
    sliced_mutated_sequence = np.array(
        mutated_df['sliced_mutated_sequence']
        [index:index + batch_size]
    )
    scores['sliced_mutated_sequence'] += list(sliced_mutated_sequence)
    window_start = np.array(mutated_df['window_start'][index:index + batch_size])
    scores['window_start'] += list(window_start)
    window_end = np.array(mutated_df['window_end'][index:index + batch_size])
    scores['window_end'] += list(window_end)

    return scores, (mutated_sequence, window_start, window_end)


def get_tranception_scores_mutated_sequences(model: GPT2PreTrainedModel,
                                             mutated_sequence_df: pd.DataFrame,
                                             batch_size_inference: int,
                                             score_var_name: str,
                                             target_seq: str,
                                             num_workers: int = 10,
                                             reverse: bool = False,
                                             indel_mode: bool = False):
    """Takes input as a set of mutated sequences and returns scores for each mutation.
    If `target_seq` is not None, returns the delta log likelihood w.r.t that target sequence.
    Otherwise returns the log likelihood of the protein sequences.
    """
    scores = {}
    scores['mutated_sequence'] = []
    scores['mutant'] = []
    scores['sliced_mutated_sequence'] = []
    scores['window_start'] = []
    scores['window_end'] = []
    scores['score'] = []
    with torch.no_grad():
        ds = Dataset.from_pandas(mutated_sequence_df)
        ds.set_transform(model.encode_batch)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=model.config.tokenizer, mlm=False
        )
        sampler = SequentialSampler(ds)
        ds_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size_inference,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        mutant_index = 0
        # for encoded_batch in tqdm.tqdm(ds_loader):
        for encoded_batch in ds_loader:
            full_batch_length = len(encoded_batch['input_ids'])
            scores, variables = update_scores_df(
                mutated_sequence_df, scores, mutant_index, full_batch_length
            )
            for k, v in encoded_batch.items():
                if isinstance(v, torch.Tensor):
                    encoded_batch[k] = v.to(model.device)
            shift_labels = encoded_batch['labels'][..., 1:].contiguous()
            if hasattr(model.config, "retrieval_aggregation_mode") and \
                    model.config.retrieval_aggregation_mode is not None:
                if reverse:
                    encoded_batch['flip'] = torch.tensor([1] * full_batch_length)
                encoded_batch['start_slice'] = variables[1]
                encoded_batch['end_slice'] = variables[2]
                # Only mutated_sequence is flipped if the scoring_mirror branch of score_mutants.
                # No need to flip mutated_sequence for MSA re-aligning
                encoded_batch['mutated_sequence'] = variables[0]
                fused_shift_log_probas = model(
                    **encoded_batch, return_dict=True).fused_shift_log_probas
                loss_fct = NLLLoss(reduction='none')
                loss = -loss_fct(
                    input=fused_shift_log_probas.view(-1, fused_shift_log_probas.size(-1)),
                    target=shift_labels.view(-1)
                ).view(fused_shift_log_probas.shape[0], fused_shift_log_probas.shape[1])
            else:
                lm_logits = model(**encoded_batch, return_dict=True).logits
                shift_logits = lm_logits[..., :-1, :].contiguous()
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = -loss_fct(
                    input=shift_logits.view(-1, shift_logits.size(-1)),
                    target=shift_labels.view(-1)
                ).view(shift_logits.shape[0], shift_logits.shape[1])
            mask = encoded_batch['attention_mask'][..., 1:].float()
            mask[mask == 0] = float('nan')
            loss *= mask
            loss = nansum(loss, dim=1)
            scores_batch = list(loss.cpu().numpy())
            full_batch_length = len(encoded_batch['input_ids'])
            scores['score'] += scores_batch
            mutant_index += full_batch_length
    scores = pd.DataFrame(scores)
    if model.config.scoring_window == "sliding":
        # We need to aggregate scores when using sliding mode
        scores = scores[[
            'mutated_sequence', 'score'
        ]].groupby('mutated_sequence').sum().reset_index()
    scores['score'] = scores['score'] / scores['mutated_sequence'].map(lambda x: len(x))
    scores_mutated_seq = scores[scores.mutated_sequence != target_seq]
    scores_wt = scores[scores.mutated_sequence == target_seq]
    merge_delta = 'mutated_sequence' if model.config.scoring_window == "sliding" \
        else 'window_start'
    if model.config.scoring_window == "optimal":
        delta_scores = pd.merge(scores_mutated_seq,
                                scores_wt,
                                how='left',
                                on=[merge_delta],
                                suffixes=('', '_wt'))
        delta_scores[score_var_name] = delta_scores['score'] - delta_scores['score_wt']
    elif model.config.scoring_window == "sliding":
        delta_scores = scores_mutated_seq.copy()
        # In sliding mode there is a single reference window for the WT
        delta_scores[score_var_name] = delta_scores['score'] - list(scores_wt['score'])[0]

    delta_scores.drop_duplicates(subset="mutated_sequence", inplace=True)
    delta_scores["mutant"] = delta_scores["mutant"].replace(np.nan, '', regex=True)
    return delta_scores[['mutated_sequence', 'mutant', score_var_name]]


def get_sequence_slices(df: pd.DataFrame,
                        target_seq: str,
                        model_context_len: int,
                        start_idx: int = 1,
                        scoring_window: str = "optimal",
                        indel_mode: bool = False):
    """Processes a DataFrame `df` containing mutant triples (or indels) for scoring
    by slicing sequences to fit the model's maximum context window.
    Note: when scoring indels for sequences that would be longer than the model max context length,
          it is preferable to use the "sliding" scoring_window. Use "optimal" otherwise.

    Args:
        df (pd.DataFrame): Input dataframe to be processed.
        target_seq (string): Full reference sequence (wild type) that is mutated in the DMS assay.
        model_context_len (int): Maximum context size for the model.
        start_idx (int): Integer to move to 0-indexing (from 1-indexing) of positions.
        scoring_window (string): Method to slice sequences longer than maximum context size:
            - `optimal`: selects a single window as large as possible (Default).
            - `sliding`: splits the full sequence in contiguous (non-overlapping) chunks
                         that are equal to the max context (last chunk which may be shorter)
        indel_mode (bool): Flag to be used when scoring insertions and deletions.
    """
    len_target_seq = len(target_seq)
    num_mutants = len(df['mutated_sequence'])
    df = df.reset_index(drop=True)
    if scoring_window == "optimal":
        if not indel_mode:
            df['mutation_barycenter'] = df['mutant'].apply(lambda x: int(np.array(
                [int(mutation[1:-1]) - start_idx for mutation in x.split(':')]
            ).mean()))
            df['scoring_optimal_window'] = df['mutation_barycenter'].apply(
                lambda x: get_optimal_window(x, len_target_seq, model_context_len)
            )
        else:
            df['mutation_barycenter'] = df['mutated_sequence'].apply(lambda x: len(x) // 2)
            df['scoring_optimal_window'] = df['mutated_sequence'].apply(lambda x: (0, len(x)))

        df['sliced_mutated_sequence'] = [
            df['mutated_sequence'][index]
            [df['scoring_optimal_window'][index][0]:
             df['scoring_optimal_window'][index][1]]
            for index in range(num_mutants)
        ]
        df['window_start'] = df['scoring_optimal_window'].map(lambda x: x[0])
        df['window_end'] = df['scoring_optimal_window'].map(lambda x: x[1])
        df = df.drop(columns=["scoring_optimal_window", "mutation_barycenter"],
                     errors="ignore")

        df_wt = df.copy()
        df_wt["mutant"] = ""
        df_wt['mutated_sequence'] = [target_seq] * num_mutants
        # For indels, we set the wild type reference to be always the same (full length) sequence.
        # We assume here that the length is lower than model context size
        # (otherwise "Sliding" mode should be used)
        if indel_mode:
            df_wt['window_end'] = df_wt['mutated_sequence'].map(lambda x: len(x))

        df_wt['sliced_mutated_sequence'] = [
            target_seq[df_wt['window_start'][index]:df_wt['window_end'][index]]
            for index in range(num_mutants)
        ]
        df = pd.concat([df, df_wt], axis=0)
        df.drop_duplicates(inplace=True)
    elif scoring_window == "sliding":
        num_windows = 1 + int(len_target_seq / model_context_len)
        df_list = []
        start = 0
        for window_index in range(1, num_windows + 1):
            df_sliced = df.copy()
            df_sliced["mutant"] = ""
            df_sliced['sliced_mutated_sequence'] = df_sliced[
                'mutated_sequence'].map(
                    lambda x: x[start:start + model_context_len])
            df_sliced['window_start'] = [start] * num_mutants
            df_sliced['window_end'] = df_sliced['mutated_sequence'].map(
                lambda x: min(len(x), start + model_context_len))
            df_sliced_wt = df_sliced.copy()
            df_sliced_wt['mutated_sequence'] = [target_seq] * num_mutants
            df_sliced_wt['sliced_mutated_sequence'] = df_sliced_wt[
                'mutated_sequence'].map(
                    lambda x: x[start:start + model_context_len])
            df_sliced_wt['window_end'] = df_sliced_wt['mutated_sequence'].map(
                lambda x: min(len(x), start + model_context_len)
            )  # Need to adjust end index if WT and sequence are not same full length
            df_list.append(df_sliced)
            df_list.append(df_sliced_wt)
            start += model_context_len
        df_final = pd.concat(df_list, axis=0)
        df = df_final.drop_duplicates()

    return df.reset_index(drop=True)
