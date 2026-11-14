"""Small text/path helpers that do not load models."""

import hashlib
import html
import re
from pathlib import Path

TAG_RE = re.compile(r"<\|[^|]+\|>")


def sanitize_example_id(value) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._") or "sample"


def report_name(word: str) -> str:
    safe = sanitize_example_id(word)
    if safe != word:
        safe += "-" + hashlib.sha256(word.encode()).hexdigest()[:8]
    return safe


def resolve_path(value, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def clean_subtitle_markup(text: str) -> str:
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\s+</?(?:font|i|b|u)(?:\s[^>]*)?$", "", text, flags=re.I)
    text = re.sub(r"\{\\[^}]*\}", "", text)
    return " ".join(html.unescape(text).replace(r"\N", " ").split())


def drop_text_fragments(text: str, fragments=None) -> str:
    for fragment in fragments or []:
        if fragment:
            text = re.sub(re.escape(fragment), "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split())).strip()
