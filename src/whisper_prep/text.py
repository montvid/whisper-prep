"""Small text/path helpers that do not load models."""

import hashlib
import html
import re
import unicodedata
from pathlib import Path

TAG_RE = re.compile(r"<\|[^|]+\|>")


def sanitize_example_id(value) -> str:
    value = unicodedata.normalize("NFC", str(value).strip())
    # Python's \w excludes combining marks, including Indic vowel signs.
    safe = "".join(
        char if char in "._-" or unicodedata.category(char)[0] in "LNM" else "_" for char in value
    )
    return re.sub(r"_+", "_", safe).strip("._") or "sample"


def report_name(word: str) -> str:
    safe = sanitize_example_id(word)
    if safe != word:
        safe += "-" + hashlib.sha256(word.encode()).hexdigest()[:8]
    return safe


def resolve_path(value, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def clean_subtitle_markup(text: str) -> str:
    # Only subtitle markup is disposable. Comparisons and literal <words>
    # can be spoken content, so arbitrary angle-bracketed text must survive.
    text = re.sub(
        r"</?(?:font|i|b|u|c|v|lang|ruby|rt)(?:\.[\w.-]+)?(?:\s[^<>]*)?\s*>",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"<(?:\d+:)?\d{2}:\d{2}\.\d{3}>", "", text)
    text = re.sub(r"\s+</?(?:font|i|b|u)(?:\s[^>]*)?$", "", text, flags=re.I)
    text = re.sub(r"\{\\[^}]*\}", "", text)
    return " ".join(html.unescape(text).replace(r"\N", " ").split())


def drop_text_fragments(text: str, fragments=None) -> str:
    for fragment in fragments or []:
        if fragment:
            # Remove space introduced next to a comma/period by this deletion
            # only. Preserve the transcript's other punctuation and spacing.
            text = re.sub(
                r"(?<=\S)[ \t]*" + re.escape(fragment) + r"[ \t]*(?=[,.])",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(re.escape(fragment), "", text, flags=re.IGNORECASE)
    return " ".join(text.split())
