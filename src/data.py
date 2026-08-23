import json
from pathlib import Path

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{n}") from e
    return rows

def load_data(data_dir: Path):
    declarations = read_jsonl(data_dir / "declarations.jsonl")
    regulations = read_jsonl(data_dir / "regulations.jsonl")
    knowledge = data_dir / "tnved_knowledge.txt"
    return declarations, regulations, knowledge
