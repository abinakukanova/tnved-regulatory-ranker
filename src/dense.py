import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_path, batch_size=16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Dense device: {self.device}")

        self.model = SentenceTransformer(
            str(model_path),
            device=self.device,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.batch_size = batch_size
        self.embeddings = None

    def fit(self, texts):
        self.embeddings = self.encode(texts)

    def encode(self, text):

        return self.model.encode(
            text,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def search(self, query, k):
        q = self.encode(query)

        scores = self.embeddings @ q

        ids = np.argsort(-scores)[:k]

        return [(int(i), float(scores[i])) for i in ids]