from rank_bm25 import BM25Okapi
from .text import tokenize

class BM25Retriever:
    def __init__(self, texts):
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query, k):
        scores = self.index.get_scores(tokenize(query))
        ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in ids]
