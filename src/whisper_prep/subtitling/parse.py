"""Read SRT and WebVTT cues without depending on index or blank-line offsets."""

import logging
import re
from pathlib import Path

from whisper_prep.generation.typing import Utterance
from whisper_prep.text import VTT_WORD_TIMESTAMP, clean_subtitle_markup, render_subtitle_tags

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


def _display_lines(body: list[str], cue_text: str) -> tuple[list[str], int | None]:
    if len(body) == 1 and "<" not in body[0] and r"\N" not in body[0]:
        return [cue_text], None  # A single line cannot carry text above a new line.
    raw_lines = render_subtitle_tags("\n".join(body)).replace(r"\N", "\n").splitlines()
    if len(raw_lines) < 2:
        return [cue_text], None
    lines = []
    first_word_timing = None
    for line in raw_lines:
        if first_word_timing is None and VTT_WORD_TIMESTAMP.search(line):
            first_word_timing = len(lines)
        visible = clean_subtitle_markup(line)
        if visible:
            lines.append(visible)
    return lines, first_word_timing


def _carried_lines(previous: list[str], current: list[str]) -> int:
    """Count complete trailing lines redisplayed above a new final line."""
    if not previous or len(current) < 2 or current == previous:
        return 0
    for count in range(min(len(previous), len(current) - 1), 0, -1):
        if previous[-1] == current[count - 1] and previous[-count:] == current[:count]:
            return count if any(line != previous[-1] for line in current[count:]) else 0
    return 0


def _rolling_error(path, line: int, is_vtt: bool) -> ValueError:
    kind = "WebVTT" if is_vtt else "SRT"
    return ValueError(
        f"Detected rolling {kind} captions in {path}:{line}. "
        "Convert them to non-rolling subtitles before preparing this file."
    )


def read_cues(path: str | Path) -> list[Utterance]:
    text = read_subtitle_text(path).replace("\r\n", "\n").replace("\r", "\n")
    is_vtt = Path(path).suffix.lower() == ".vtt"
    separator = r"\n\n" if is_vtt else r"\n[ \t]*\n"

    def empty_line(line):
        return line == "" if is_vtt else not line.strip()

    if is_vtt:
        # Metadata blocks are separated by blank lines in WebVTT.
        text = "\n\n".join(
            "\n" * block.count("\n")
            if re.match(r"^(?:NOTE(?:\s|$)|STYLE(?:\s|$)|REGION(?:\s|$))", block.lstrip())
            else block
            for block in re.split(separator, text)
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
    previous_lines = []
    carry_positions = []
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
        if not separated and position + 1 < len(indices) and len(body) > 1:
            if body[-1].strip().isdigit():
                body.pop()
            elif (
                is_vtt
                and not body[-2].strip()
                and body[-1].strip()
                and any(line.strip() for line in body[:-2])
            ):
                raise ValueError(
                    f"Ambiguous WebVTT cue identifier in {path}:{stop}. Separate cues with an empty line."
                )
        cue_text = clean_subtitle_markup(" ".join(body))
        if not cue_text or cue_text == ".":
            previous_lines = []
            continue
        cue = Utterance(cue_text, milliseconds(match[1]), milliseconds(match[2]))
        display, first_timing = _display_lines(body, cue_text)
        if previous_lines and len(display) > 1:
            previous = cues[-1]
            gap = abs(cue.start - previous.end)
            # Allow a few video frames at a display change. A word timing on the
            # supposedly carried text gives that line its own speech alignment.
            carried = _carried_lines(previous_lines, display) if gap <= 100 else 0
            if carried and (first_timing is None or first_timing >= carried):
                transition = 0 < previous.end - previous.start < 20
                word_timed = first_timing is not None and first_timing < len(display)
                if gap <= 20 and (transition or word_timed):
                    raise _rolling_error(path, index + 1, is_vtt)
                if carry_positions and carry_positions[-1] == position - 1:
                    raise _rolling_error(path, indices[carry_positions[-1]][0] + 1, is_vtt)
                carry_positions.append(position)
        previous_lines = display
        cues.append(cue)
    # Dialogue may accumulate in separate pairs rather than in one continuous run.
    if carry_positions and 2 * len(carry_positions) > len(cues) - 1:
        raise _rolling_error(path, indices[carry_positions[0]][0] + 1, is_vtt)
    return cues
