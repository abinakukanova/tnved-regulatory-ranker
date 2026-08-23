## Стэк

- BM25
- BAAI/bge-m3
- Reciprocal Rank Fusion
- BAAI/bge-reranker-v2-m3

## Описание подхода

Используется гибридный подход: BM25 + bge-m3

```
                    Декларация
                         |
              +----------+----------+
              |                     |
             BM25                 BGE-M3
              |                     |
          lexical              semantic
          retrieval             retrieval
              |                     |
              +----------+----------+
                         |
                        RRF
                         |
                 candidate set
                         |
                 BGE-reranker-v2-m3
                         |
                       Top-10
```

`tnved_knowledge.txt` используется как доп. источник информации, добавляется к тексту регуляций.


## Запуск

0. Скачайте модели с Hugging face и положите в одну папку:

```text
models/
├── bge-m3/
└── bge-reranker-v2-m3/
```

1. Установите зависимости
```bash
pip install -r requirements.txt
```

2. Запустите ранкер

```bash
python run.py --data ./data --out ./out
```

Полная команда:

```bash
python run.py \
  --data ./data \
  --out ./out \
  --models ./models \
  --bm25-k 60 \
  --dense-k 60 \
  --candidate-k 80 \
  --rerank-batch-size 16
```