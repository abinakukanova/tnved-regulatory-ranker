import re
import unicodedata

TOKEN_RE = re.compile(r"[a-zа-я0-9]+(?:[-./][a-zа-я0-9]+)*", re.I)

def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text):
    return TOKEN_RE.findall(normalize_text(text))

def declaration_text(row):
    return " ".join(
        x for x in [normalize_text(row.get("G31_1")),
                    normalize_text(row.get("desc_extention"))] if x
    )

def regulation_text(row):
    return " ".join(
        x for x in [
            normalize_text(row.get("code")),
            normalize_text(row.get("description")),
            normalize_text(row.get("notes")),
            normalize_text(row.get("explanation")),
        ] if x
    )
