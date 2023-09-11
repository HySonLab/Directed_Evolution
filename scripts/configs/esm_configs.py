from typing import List
from transformers import EsmConfig


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


esm_config = ESMArguments()


def get_esm2_config(vocabs: List[str]):
    config = EsmConfig(
        len(vocabs),
        mask_token_id=vocabs.index("<mask>"),
        pad_token_id=vocabs.index("<pad>"),
        hidden_size=esm_config.embed_dim,
        num_hidden_layers=esm_config.num_layers,
        num_attention_heads=esm_config.attention_heads,
        intermediate_size=esm_config.intermediate_size,
        hidden_dropout_prob=esm_config.hidden_dropout_prob,
        attention_probs_dropout_prob=esm_config.attention_probs_dropout_prob,
        max_position_embeddings=esm_config.max_position_embeddings,
        position_embedding_type=esm_config.position_embedding_type,
        token_dropout=esm_config.token_dropout,
    )
    return config
