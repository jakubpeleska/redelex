from typing import Optional
from functools import lru_cache

import torch

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    embedding_dim: int

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        always_use_cache: bool = False,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.always_use_cache = always_use_cache

    def __call__(self, sentences: list[str], use_cache: bool = False) -> torch.Tensor:
        if use_cache or self.always_use_cache:
            return torch.stack([self.cached_call(s) for s in sentences])
        return self.model.encode(sentences, convert_to_tensor=True)

    @lru_cache(maxsize=10000, typed=False)
    def cached_call(self, sentence: str) -> torch.Tensor:
        return self.model.encode(sentence, convert_to_tensor=True)


class GloveTextEmbedder(TextEmbedder):
    embedding_dim = 300

    def __init__(self, **kwargs):
        super().__init__(
            "sentence-transformers/average_word_embeddings_glove.6B.300d", **kwargs
        )


class PotionTextEmbedder(TextEmbedder):
    """
    Text embedder using the Potion multilingual model.
    """

    embedding_dim = 256

    def __init__(self, **kwargs):
        super().__init__("minishlab/potion-multilingual-128M", **kwargs)
