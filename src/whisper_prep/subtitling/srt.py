from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path


@dataclass
class Caption:
    start_second: float
    end_second: float
    text: str
    offset: float = 0

    def start_timestamp(self) -> str:
        return format_delta(timedelta(seconds=self.start_second + self.offset), "{h}:{m}:{s},{ms}")

    def end_timestamp(self) -> str:
        return format_delta(timedelta(seconds=self.end_second + self.offset), "{h}:{m}:{s},{ms}")


def format_delta(delta: timedelta, pattern: str) -> str:
    total_ms = round(delta.total_seconds() * 1000)
    if total_ms < 0:
        raise ValueError("Subtitle timestamps must be nonnegative")
    seconds, ms = divmod(total_ms, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return pattern.format(h=f"{hours:02d}", m=f"{minutes:02d}", s=f"{seconds:02d}", ms=f"{ms:03d}")


def can_trim_overlap(start, end, next_start):
    """A repaired cue must keep at least 200 ms and half its speech interval."""
    retained = next_start - start
    return retained >= 0.2 and retained >= (end - start) / 2


def generate_srt(captions: list[Caption], save_path: str | Path) -> int:
    repaired = []
    merged = 0
    for original in captions:
        cue = replace(
            original,
            start_second=original.start_second + original.offset,
            end_second=original.end_second + original.offset,
            text=" ".join(original.text.split()),
            offset=0,
        )
        if cue.start_second < 0 or cue.end_second <= cue.start_second:
            raise ValueError(f"Invalid generated caption: {original}")
        if repaired and cue.start_second < repaired[-1].end_second:
            previous = repaired[-1]
            if can_trim_overlap(previous.start_second, previous.end_second, cue.start_second):
                previous.end_second = cue.start_second
            else:
                previous.text += " " + cue.text
                previous.end_second = max(previous.end_second, cue.end_second)
                merged += 1
                continue
        repaired.append(cue)
    content = "".join(
        f"{i}\n{cue.start_timestamp()} --> {cue.end_timestamp()}\n{cue.text}\n\n"
        for i, cue in enumerate(repaired, 1)
    )
    Path(save_path).write_text(content, encoding="utf-8")
    return merged
