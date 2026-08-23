import csv
from pathlib import Path

from .data import load_data
from .text import (
    declaration_text,
    regulation_text
)

from .tnved import TNVEDKnowledge

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .reranker import Reranker

from .fusion import (
    rrf,
    minmax
)



class HybridRanker:


    def __init__(
        self,
        data_dir,
        model_dir,
        bm25_k=60,
        dense_k=60,
        candidate_k=80,
        rerank_batch_size=16
    ):


        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.candidate_k = candidate_k


        (
            self.declarations,
            self.regulations,
            tnved_file
        ) = load_data(data_dir)



        print(
            "Loading TNVED structure..."
        )


        self.tnved = TNVEDKnowledge(
            tnved_file
        )


        print(
            "TNVED nodes:",
            len(self.tnved.nodes)
        )



        self.regulation_texts = []


        for row in self.regulations:


            context = self.tnved.get_context(
                row.get("code")
            )


            text = regulation_text(row)


            if context:

                text += (
                    "\n\n"
                    "ТН ВЭД иерархия: "
                    + context
                )


            self.regulation_texts.append(
                text
            )



        print(
            "Building BM25..."
        )

        self.regulation_bm25 = BM25Retriever(
            self.regulation_texts
        )



        print(
            "Building dense index..."
        )


        self.regulation_dense = DenseRetriever(
            model_dir / "bge-m3",
            batch_size=rerank_batch_size
        )


        self.regulation_dense.fit(
            self.regulation_texts
        )

        print("Encoding declaration queries...")

        self.declaration_queries = [
        declaration_text(d)
        for d in self.declarations
        ]

        self.declaration_embeddings = (
        self.regulation_dense.encode(
        self.declaration_queries
        )
        )

        print("Declaration embeddings ready")



        print(
            "Loading reranker..."
        )


        self.reranker = Reranker(
            model_dir / "bge-reranker-v2-m3",
            batch_size=rerank_batch_size
        )


        print(
            "Pipeline ready"
        )


    def rank_one(
        self,
        query,
        query_embedding,
        rerank_top_k=40
    ):

        bm25_results = self.regulation_bm25.search(
            query,
            self.bm25_k
        )

        scores = (
            self.regulation_dense.embeddings
            @ query_embedding
        )

        dense_ids = (
            scores
            .argsort()[-self.dense_k:][::-1]
        )

        dense_results = [
            (
                int(i),
                float(scores[i])
            )
            for i in dense_ids
        ]

        candidates = rrf(
            bm25_results,
            dense_results,
            top_k=self.candidate_k
        )

        return candidates

    def rerank_all(
        self,
        prepared_queries,
        rerank_top_k=40
    ):

        all_pairs = []
        metadata = []

        for item in prepared_queries:

            query = item["query"]
            candidates = item["candidates"]

            candidates = candidates[:rerank_top_k]

            for idx, rrf_score in candidates:

                all_pairs.append(
                    (
                        query,
                        self.regulation_texts[idx]
                    )
                )

                metadata.append(
                    (
                        item["declaration_id"],
                        idx,
                        rrf_score
                    )
                )

        print(
            f"Reranking {len(all_pairs)} pairs..."
        )

        rerank_scores = self.reranker.predict(
            all_pairs
        )

        results = {}

        for meta, rerank_score in zip(
            metadata,
            rerank_scores
        ):

            declaration_id, idx, rrf_score = meta

            if declaration_id not in results:
                results[declaration_id] = []

            results[declaration_id].append(
                (
                    idx,
                    float(rerank_score),
                    float(rrf_score)
                )
            )

        return results

    def run(self, out_dir):

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output = (
            out_dir /
            "predictions.csv"
        )

        print("Retrieving candidates...")

        prepared_queries = []

        for declaration_idx, declaration in enumerate(
            self.declarations
        ):

            declaration_id = declaration[
                "declaration_id"
            ]

            query = self.declaration_queries[
                declaration_idx
            ]

            query_embedding = (
                self.declaration_embeddings[
                    declaration_idx
                ]
            )

            candidates = self.rank_one(
                query,
                query_embedding
            )

            prepared_queries.append(
                {
                    "declaration_id": declaration_id,
                    "query": query,
                    "candidates": candidates
                }
            )

        print(
            "Candidate retrieval completed."
        )

        # Rerank only top 40 RRF candidates.
        # 40 is a compromise between quality and runtime.
        reranked = self.rerank_all(
            prepared_queries,
            rerank_top_k=40
        )

        rows = []

        for declaration in self.declarations:

            declaration_id = declaration[
                "declaration_id"
            ]

            candidates = reranked[
                declaration_id
            ]

            candidates.sort(
                key=lambda x: -x[1]
            )

            top10 = candidates[:10]

            for rank, (
                idx,
                rerank_score,
                rrf_score
            ) in enumerate(
                top10,
                1
            ):

                rows.append(
                    {
                        "declaration_id":
                            declaration_id,

                        "rank":
                            rank,

                        "regulation_id":
                            self.regulations[idx][
                                "regulation_id"
                            ],

                        "score":
                            f"{rerank_score:.8f}"
                    }
                )

        with output.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "declaration_id",
                    "rank",
                    "regulation_id",
                    "score"
                ]
            )

            writer.writeheader()
            writer.writerows(rows)

        print(
            f"Predictions saved to {output}"
        )

        return output