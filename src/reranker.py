import numpy as np
import torch
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_path, batch_size=64):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Reranker device: {self.device}")

        self.model = CrossEncoder(
            str(model_path),
            device=self.device,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.batch_size = batch_size

    def predict(self, pairs):

        if not pairs:
            return np.array([], dtype=np.float32)

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(
            scores,
            dtype=np.float32
        ).reshape(-1)

        return scores