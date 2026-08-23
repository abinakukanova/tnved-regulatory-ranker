from collections import defaultdict

def rrf(*result_lists, rrf_k=60, top_k=80):
    scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results, 1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

def minmax(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.5] * len(values)
    return [(x - lo) / (hi - lo) for x in values]
