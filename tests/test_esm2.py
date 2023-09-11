import os
from typing import List
from de.common.io_utils import read_fasta
from de.common.utils import save_vocabulary
from de.common.constants import CANONICAL_ALPHABET
from de.samplers.models.esm import ESM2
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer


fasta_file = "/home/thanhtvt1/workspace/Directed_Evolution/data/uniprot_sprot.fasta"
vocab_file = "/home/thanhtvt1/workspace/Directed_Evolution/data/vocabs/vocab_1.txt"
save_dir = "/home/thanhtvt1/workspace/Directed_Evolution/data/vocabs"


class ESMArguments:
    num_layers: int = 30
    embed_dim: int = 640
    attention_heads: int = 20
    intermediate_size: int = 2560
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1026
    position_embedding_type: str = "rotary"
    token_dropout: str = True


esmconf = ESMArguments()


def init_model(vocab: List[str], vocab_dir: str, k: int):
    config = EsmConfig(
        len(vocab),
        mask_token_id=vocab.index("<mask>"),
        pad_token_id=vocab.index("<pad>"),
        hidden_size=esmconf.embed_dim,
        num_hidden_layers=esmconf.num_layers,
        num_attention_heads=esmconf.attention_heads,
        intermediate_size=esmconf.intermediate_size,
        hidden_dropout_prob=esmconf.hidden_dropout_prob,
        attention_probs_dropout_prob=esmconf.attention_probs_dropout_prob,
        max_position_embeddings=esmconf.max_position_embeddings,
        position_embedding_type=esmconf.position_embedding_type,
        token_dropout=esmconf.token_dropout,
    )
    model = EsmForMaskedLM(config)

    vocab_filepath = os.path.join(vocab_dir, f"vocab_{k}.txt")
    tokenizer = EsmTokenizer(vocab_filepath)

    return model, tokenizer


def get_sequence_from_fasta(fasta_file: str, max_seq_len: int) -> List[str]:
    seqs = read_fasta(fasta_file, max_seq_length=max_seq_len)
    seqs = seqs.values()
    return list(seqs)


seqs = get_sequence_from_fasta(fasta_file, 1024)
all_toks = save_vocabulary(save_dir, CANONICAL_ALPHABET, k=3)
# esm = ESM2(vocab_file,
#            pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D")

print(f"len(all_toks) = {len(all_toks)}")

model, tokenizer = init_model(all_toks, save_dir, 3)

seq = seqs[0]

print("tokenizing")

tokens = tokenizer([seq[:31]], add_special_tokens=False, return_tensors="pt")
print("Sequence:", seq[:31])
print('\n----------------------------------------------\n')
print("Token:", tokens)

print("Decoded Sequence:", tokenizer.decode(tokens["input_ids"].squeeze()))
