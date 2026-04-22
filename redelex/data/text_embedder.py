from typing import Optional

import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    embedding_dim: int

    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, sentences: list[str], use_cache: bool = False) -> torch.Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)


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
