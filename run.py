from pathlib import Path
import argparse
from src.pipeline import HybridRanker

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("./models"))
    p.add_argument("--bm25-k", type=int, default=60)
    p.add_argument("--dense-k", type=int, default=60)
    p.add_argument("--candidate-k", type=int, default=80)
    p.add_argument("--rerank-batch-size", type=int, default=16)
    args = p.parse_args()

    ranker = HybridRanker(
        data_dir=args.data,
        model_dir=args.models,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        candidate_k=args.candidate_k,
        rerank_batch_size=args.rerank_batch_size,
    )
    print(f"Saved: {ranker.run(args.out)}")

if __name__ == "__main__":
    main()
