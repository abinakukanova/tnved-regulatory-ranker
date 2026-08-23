import torch
from sentence_transformers import SentenceTransformer

print("START", flush=True)

print("CUDA:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True
    )

print("Loading BGE-M3...", flush=True)

model = SentenceTransformer(
    "./models/bge-m3",
    device="cuda",
    local_files_only=True,
)

print("MODEL LOADED", flush=True)

texts = [
    "свинина мороженая ребра домашних свиней",
    "лекарственный препарат витамин C таблетки",
    "женские трикотажные брюки хлопок",
]

print("Encoding...", flush=True)

emb = model.encode(
    texts,
    batch_size=1,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)

print("ENCODING DONE", flush=True)
print(emb.shape, flush=True)
print(emb.dtype, flush=True)

print("SUCCESS", flush=True)