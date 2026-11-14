"""Read SRT and WebVTT cues without depending on index or blank-line offsets."""

import logging
import re
from pathlib import Path

from whisper_prep.generation.typing import Utterance
from whisper_prep.text import clean_subtitle_markup

TIMESTAMP = r"(?:\d+:)?\d{2}:\d{2}[.,]\d{1,3}"
TIMING_LINE = re.compile(rf"^\s*({TIMESTAMP})\s+-->\s+({TIMESTAMP})(?:\s+.*)?$")


def milliseconds(value: str) -> int:
    if not re.fullmatch(TIMESTAMP, value):
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    clock, fraction = re.split(r"[.,]", value)
    parts = [int(part) for part in clock.split(":")]
    if len(parts) == 2:
        parts.insert(0, 0)
    hours, minutes, seconds = parts
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + int(fraction.ljust(3, "0"))


def read_subtitle_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logging.getLogger(__name__).warning("Reading %s as Windows-1252", path)
        return raw.decode("cp1252")


def read_cues(path: str | Path) -> list[Utterance]:
    text = read_subtitle_text(path).replace("\r\n", "\n").replace("\r", "\n")
    if Path(path).suffix.lower() == ".vtt":
        # Metadata blocks are separated by blank lines in WebVTT.
        text = "\n\n".join(
            block
            for block in re.split(r"\n[ \t]*\n", text)
            if not re.match(r"^(?:NOTE(?:\s|$)|STYLE(?:\s|$)|REGION(?:\s|$))", block.lstrip())
        )
    lines = text.splitlines()
    indices = [(i, TIMING_LINE.fullmatch(line)) for i, line in enumerate(lines)]
    indices = [(i, match) for i, match in indices if match is not None]
    cues = []
    for position, (index, match) in enumerate(indices):
        stop = indices[position + 1][0] if position + 1 < len(indices) else len(lines)
        body = lines[index + 1 : stop]
        # A separator ends cue text; a following optional identifier belongs
        # to the next cue. Missing separators are supported for numeric SRT IDs.
        separated = False
        for blank, line in enumerate(body):
            if not line.strip():
                body = body[:blank]
                separated = True
                break
        if (
            not separated
            and position + 1 < len(indices)
            and len(body) > 1
            and body[-1].strip().isdigit()
        ):
            body.pop()
        cue_text = clean_subtitle_markup(" ".join(body))
        if cue_text and cue_text != ".":
            cues.append(Utterance(cue_text, milliseconds(match[1]), milliseconds(match[2])))
    if not indices and text.strip() and not text.strip().startswith("WEBVTT"):
        raise ValueError(f"No valid subtitle timing lines in {path}")
    return cues
