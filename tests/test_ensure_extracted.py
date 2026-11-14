"""Tests for empty transcript segmentation."""

from __future__ import annotations

import csv
import shutil
import struct
import tempfile
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_silent_wav(path: Path, dur_s: float = 2.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(dur_s * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


class TestEmptyTranscriptSegments:
    """Verify empty transcript clips are written through the normal segment path."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp)

    def _run_dp(self, audio_path: Path, dur_s: float) -> list[dict]:
        """Run DataProcessor in TSV mode with one clip, return parsed records."""
        import json

        from whisper_prep.generation.data_processor import DataProcessor

        srt_path = self.tmp / f"{audio_path.stem}.srt"
        srt_path.write_text("", encoding="utf-8")

        tsv_path = self.tmp / "transcripts.tsv"
        with tsv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "id",
                    "audio_path",
                    "srt_path",
                    "language",
                    "duration_seconds",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                dict(
                    id=audio_path.stem,
                    audio_path=str(audio_path),
                    srt_path=str(srt_path),
                    language="en",
                    duration_seconds=f"{dur_s:.6f}",
                )
            )

        output = self.tmp / "data.ljson"
        dump = self.tmp / "dump"
        dp = DataProcessor(
            audio_dir=str(self.tmp),
            transcript_dir=str(self.tmp),
            language="en",
            output=str(output),
            dump_dir=str(dump),
            transcripts_tsv=str(tsv_path),
            keep_empty_chance=1.0,
        )
        dp.run()

        records = []
        with output.open() as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def test_short_empty_clip_creates_dump_segment(self):
        """Short clip with an empty SRT creates a normal dumped segment."""
        audio = self.tmp / "short_clip.wav"
        _make_silent_wav(audio, dur_s=5.0)

        records = self._run_dp(audio, dur_s=5.0)

        assert len(records) == 1
        assert records[0]["audio_path"] != str(audio.absolute())
        assert (self.tmp / records[0]["audio_path"]).exists()
        assert records[0]["text"] == ""

    def test_long_empty_clip_is_split_into_dump_segments(self):
        """Clips longer than 30 s are split and never point back to the source."""
        audio = self.tmp / "long_clip.wav"
        _make_silent_wav(audio, dur_s=35.0)

        records = self._run_dp(audio, dur_s=35.0)

        assert len(records) >= 2
        for r in records:
            assert r["audio_path"] != str(audio.absolute())
            assert (self.tmp / r["audio_path"]).exists()
