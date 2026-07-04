import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

_LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "model_cache"
_BI_ENCODER_NAME = "all-MiniLM-L6-v2"
_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EmbeddingService:

    _bi_encoder = None
    _cross_encoder = None

    def __init__(self):
        if EmbeddingService._bi_encoder is None:
            if _LOCAL_MODEL_DIR.exists() and any(_LOCAL_MODEL_DIR.iterdir()):
                EmbeddingService._bi_encoder = SentenceTransformer(str(_LOCAL_MODEL_DIR))
            else:
                # Fallback only if the bundled model is missing
                EmbeddingService._bi_encoder = SentenceTransformer(_BI_ENCODER_NAME)
        self.bi_encoder = EmbeddingService._bi_encoder

    @property
    def cross_encoder(self):
        # Loaded lazily — only the /recommend/talent path uses it.
        if EmbeddingService._cross_encoder is None:
            EmbeddingService._cross_encoder = CrossEncoder(_CROSS_ENCODER_NAME)
        return EmbeddingService._cross_encoder

    def encode(self, text: str):
        return self.bi_encoder.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

    def encode_batch(self, texts: list[str]):
        return self.bi_encoder.encode(
            texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True
        )

    def cosine_similarity(self, vec1, vec2):
        # Vectors are L2-normalized, so dot product == cosine similarity.
        return float(np.dot(vec1, vec2))

    def cross_score(self, pairs):
        return self.cross_encoder.predict(pairs)
