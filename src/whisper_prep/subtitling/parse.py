"""Read SRT and WebVTT cues without depending on index or blank-line offsets."""

import logging
import re
from pathlib import Path

from whisper_prep.generation.typing import Utterance
from whisper_prep.text import clean_subtitle_markup

TIMESTAMP = r"(?:\d+:)?\d{2}:\d{2}[.,]\d{1,3}"
TIMING_LINE = re.compile(rf"^\s*({TIMESTAMP})\s*-->\s*({TIMESTAMP})(?:\s+.*)?$")


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
    is_vtt = Path(path).suffix.lower() == ".vtt"
    separator = r"\n\n" if is_vtt else r"\n[ \t]*\n"

    def empty_line(line):
        return line == "" if is_vtt else not line.strip()

    if is_vtt:
        # Metadata blocks are separated by blank lines in WebVTT.
        text = "\n\n".join(
            block
            for block in re.split(separator, text)
            if not re.match(r"^(?:NOTE(?:\s|$)|STYLE(?:\s|$)|REGION(?:\s|$))", block.lstrip())
        )
    for block in re.split(separator, text):
        if not block.strip() or any(TIMING_LINE.fullmatch(line) for line in block.splitlines()):
            continue
        if is_vtt and re.fullmatch(
            r"WEBVTT(?:[ \t][^\n]*)?(?:\n[ \t]*[A-Za-z][\w-]*[=:][^\n]*)*", block.strip()
        ):
            continue  # Empty VTTs may still contain a signature and header metadata.
        raise ValueError(f"No valid subtitle timing lines in {path}: {block.strip()!r}")
    lines = text.splitlines()
    indices = []
    in_cue = False
    for index, line in enumerate(lines):
        match = TIMING_LINE.fullmatch(line)
        if match:
            indices.append((index, match))
            in_cue = True
        elif empty_line(line):
            in_cue = False
        elif "-->" in line and (not in_cue or re.match(r"\s*-?(?:\d+:)?\d{2}:\d{2}[.,]", line)):
            # Preserve ordinary arrows in cue text; reject broken timing lines,
            # including cues whose blank separator is missing.
            raise ValueError(f"Invalid subtitle timing line in {path}:{index + 1}: {line!r}")
    cues = []
    for position, (index, match) in enumerate(indices):
        stop = indices[position + 1][0] if position + 1 < len(indices) else len(lines)
        body = lines[index + 1 : stop]
        # A separator ends cue text; a following optional identifier belongs
        # to the next cue. Missing separators are supported for numeric SRT IDs.
        separated = False
        for blank, line in enumerate(body):
            if empty_line(line):
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
    return cues
