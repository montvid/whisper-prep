from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

from whisper_prep.text import TAG_RE, drop_text_fragments

if TYPE_CHECKING:
    import pysubs2

NETFLIX_CHAR = 42
NETFLIX_DUR = 7


def fastlid(text):
    from whisper_prep.language import identify_language as detect

    return detect(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=config_path, required=True)
    return parser.parse_args()


def config_path(path: str):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config path:{path} is not a valid path.")


def get_compression_ratio(text: str):
    """
    Compute compression ratio after removing timestamp tokens <|...|> to avoid skew from time markers.
    """
    # Strip timestamp tokens before compression measurement
    cleaned = TAG_RE.sub("", text)
    encoded = cleaned.encode("utf-8")
    compression_ratio = len(encoded) / len(zlib.compress(encoded))
    return compression_ratio


# ---------------------------------------------------------------------------
# 1. compile once – these are the micro-second French heuristics
# ---------------------------------------------------------------------------

FR_WORDS = (
    r"\b(?:que|qui|dans|avec|pour|chez|entre|sans|leur(?:s)?|une?|des?|du|la|le|les"
    r"|il|elle|ils|elles|ça|est|sera|sont)\b"
)

FR_APOS = r"\b(?:[ldcjtmnsq]\s*'|qu\s*')"  # l' d' qu' …

HEURISTIC_RE = re.compile(f"(?:{FR_WORDS}|{FR_APOS})", re.I)

# English lexical heuristics
EN_WORDS = (
    r"\b(?:the|and|that|with|this|from|for|you|your|are|was|were|have|has"
    r"|not|but|they|their|what|when|where|would|could|should)\b"
)
EN_APOS = r"\b(?:i|you|we|they|it|he|she)\s*'"  # i'm / you're / it's ...
EN_HEURISTIC_RE = re.compile(f"(?:{EN_WORDS}|{EN_APOS})", re.I)


# Language identification is loaded on demand and keeps the full label set.
def is_french(line: str, prob_th: float = 0.85) -> bool:
    """
    If regexes matches, double check with model.
    """
    # normalise & de-tag
    txt = TAG_RE.sub("", line).replace("’", "'").strip()

    # 1️⃣ ultra-cheap lexical heuristic
    if HEURISTIC_RE.search(txt):
        try:
            lang, prob = fastlid(txt)
        except (IndexError, ValueError):  # no label came back
            return False
        return lang == "fr" and prob >= prob_th

    return False


def is_english(line: str, prob_th: float = 0.85) -> bool:
    """
    If regexes matches, double check with model.
    """
    txt = TAG_RE.sub("", line).replace("’", "'").strip()

    if EN_HEURISTIC_RE.search(txt):
        try:
            lang, prob = fastlid(txt)
        except (IndexError, ValueError):
            return False
        return lang == "en" and prob >= prob_th

    return False


def fuse_until_limits(
    subs: pysubs2.SSAFile,
    max_chars: int = NETFLIX_CHAR,
    max_duration: float = NETFLIX_DUR,
    skip_words: list = None,
    drop_text: list = None,
) -> bool:
    """Rewrite *subs* in‑place, merging cues while combined cue stays <= limits.

    Cues containing any word in skip_words will not be merged with others.
    Phrases in drop_text are removed from cue text before merging.

    Returns True if *subs* was modified.
    """
    import pysubs2

    if not subs:
        return False

    skip_words = skip_words or []
    drop_text = drop_text or []

    text_changed = False
    for cue in subs:
        cleaned_text = drop_text_fragments(cue.text, drop_text)
        if cleaned_text != cue.text:
            text_changed = True
        cue.text = cleaned_text

    def contains_skip_word(text: str) -> bool:
        return any(word.lower() in text.lower() for word in skip_words)

    merged = pysubs2.SSAFile()
    current = subs[0].copy()

    # If first cue contains skip word, don't merge with anything
    current_has_skip = contains_skip_word(current.text)

    for cue in subs[1:]:
        cue_has_skip = contains_skip_word(cue.text)
        combined_text = f"{current.text.rstrip()} {cue.text.lstrip()}"
        combined_dur = (max(cue.end, current.end) - current.start) / 1000

        plain_length = len((current.plaintext + " " + cue.plaintext).replace("\n", " "))
        within_limits = plain_length <= max_chars and combined_dur <= max_duration

        # Don't merge if either cue contains a skip word
        monotone = current.start < current.end <= cue.start < cue.end
        can_merge = monotone and within_limits and not current_has_skip and not cue_has_skip

        if can_merge:
            # Safe to merge: extend current cue
            current.end = cue.end
            current.text = combined_text
        else:
            # Would exceed limits or contains skip word -> flush current and start anew
            merged.append(current)
            current = cue.copy()
            current_has_skip = cue_has_skip

    merged.append(current)

    # Detect change by comparing lengths or any differing fields
    changed = (
        text_changed
        or len(merged) != len(subs)
        or any(
            m.start != s.start or m.end != s.end or m.text != s.text
            for m, s in zip(merged, subs, strict=False)
        )
    )

    if changed:
        subs.clear()
        subs.extend(merged)
    return changed


def netflix_normalize_file(path, skip_words=None, drop_text=None, output_path=None) -> Path:
    """Write normalized subtitles to a separate file, preserving SRT/VTT format."""
    import pysubs2

    from whisper_prep.subtitling.parse import read_cues

    path = Path(path)
    destination = (
        Path(output_path)
        if output_path
        else path.with_name(path.stem + ".normalized" + path.suffix)
    )
    if destination.resolve() == path.resolve():
        raise ValueError("Normalized subtitles must not overwrite the source")
    cues = read_cues(path)
    subs = pysubs2.SSAFile()
    subs.extend(pysubs2.SSAEvent(start=cue.start, end=cue.end, text=cue.text) for cue in cues)
    fuse_until_limits(subs, skip_words=skip_words, drop_text=drop_text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subs.save(str(destination), format_=path.suffix.lower().lstrip("."))
    return destination


def netflix_normalize_all_srts_in_folder(
    folder=".", skip_words=None, drop_text=None, output_dir=None
):
    folder = Path(folder)
    destination = Path(output_dir) if output_dir else folder / "normalized"
    return [
        netflix_normalize_file(path, skip_words, drop_text, destination / path.name)
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in {".srt", ".vtt"}
    ]


def save_hu_dataset_locally(config, audio_dir, transcript_dir):
    from whisper_prep.dataset.download import save_hu_dataset_locally as download

    return download(config, audio_dir, transcript_dir)


def __getattr__(name):
    if name == "DownloadedDatasetPaths":
        from whisper_prep.dataset.download import DownloadedDatasetPaths

        return DownloadedDatasetPaths
    raise AttributeError(name)
