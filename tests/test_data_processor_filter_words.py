import json
import re
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from whisper_prep.generation.data_processor import SAMPLE_RATE, DataProcessor


class TestDataProcessorFilterWords(unittest.TestCase):
    def test_audio_cut_keeps_silence_until_next_utterance_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            transcript_dir = root / "transcripts"
            dump_dir = root / "dump"
            output = root / "data.ljson"
            audio_dir.mkdir()
            transcript_dir.mkdir()

            audio_path = audio_dir / "sample.wav"
            audio = torch.zeros(1, SAMPLE_RATE * 40)
            torchaudio.save(audio_path, audio, SAMPLE_RATE)

            (transcript_dir / "sample.srt").write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:28,170",
                        "Hello until just before the window end",
                        "",
                        "2",
                        "00:00:28,190 --> 00:00:32,070",
                        "Next speech belongs to the next segment",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = DataProcessor(
                audio_dir=audio_dir,
                transcript_dir=transcript_dir,
                output=output,
                dump_dir=dump_dir,
            )
            processor.run()

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(Path(records[0]["audio_path"]).name, "0.flac")
            self.assertEqual(Path(records[1]["audio_path"]).name, "28190.flac")
            self.assertIn("Hello until just before the window end", records[0]["text"])
            self.assertNotIn("Next speech belongs", records[0]["text"])
            self.assertIn("Next speech belongs", records[1]["text"])

            segment_audio, sample_rate = torchaudio.load(output.parent / records[0]["audio_path"])
            duration_ms = round(segment_audio.size(1) * 1000 / sample_rate)

            self.assertEqual(sample_rate, SAMPLE_RATE)
            self.assertTrue(records[0]["text"].endswith("<|28.20|>"))
            self.assertLessEqual(abs(duration_ms - 30000), 1)
            max_timestamp_ms = max(
                round(float(value) * 1000)
                for value in re.findall(r"<\|([0-9]+\.[0-9]+)\|>", records[0]["text"])
            )
            self.assertLessEqual(max_timestamp_ms, duration_ms)

    def test_filter_words_cut_audio_and_text_from_folder_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            transcript_dir = root / "transcripts"
            dump_dir = root / "dump"
            output = root / "data.ljson"
            audio_dir.mkdir()
            transcript_dir.mkdir()

            audio_path = audio_dir / "sample.wav"
            audio = torch.zeros(1, SAMPLE_RATE * 25)
            torchaudio.save(audio_path, audio, SAMPLE_RATE)

            (transcript_dir / "sample.srt").write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:10,000",
                        "Hello before",
                        "",
                        "2",
                        "00:00:10,000 --> 00:00:20,000",
                        "SECRET pii should not survive",
                        "",
                        "3",
                        "00:00:20,000 --> 00:00:25,000",
                        "After safe",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = DataProcessor(
                audio_dir=audio_dir,
                transcript_dir=transcript_dir,
                output=output,
                dump_dir=dump_dir,
                filter_segment_words=["SECRET"],
            )
            processor.run()

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all("SECRET" not in record["text"] for record in records))
            self.assertEqual(
                [Path(record["audio_path"]).name for record in records],
                ["0.flac", "20000.flac"],
            )

            first_audio, first_rate = torchaudio.load(output.parent / records[0]["audio_path"])
            second_audio, second_rate = torchaudio.load(output.parent / records[1]["audio_path"])
            self.assertEqual(first_rate, SAMPLE_RATE)
            self.assertEqual(second_rate, SAMPLE_RATE)
            self.assertEqual(round(first_audio.size(1) * 1000 / first_rate), 10000)
            self.assertEqual(round(second_audio.size(1) * 1000 / second_rate), 5000)
            self.assertFalse((dump_dir / "sample" / "10000.flac").exists())

    def test_repeated_hallucination_blocks_audio_not_empty_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            transcript_dir = root / "transcripts"
            dump_dir = root / "dump"
            output = root / "data.ljson"
            audio_dir.mkdir()
            transcript_dir.mkdir()

            audio_path = audio_dir / "sample.wav"
            audio = torch.zeros(1, SAMPLE_RATE * 70)
            torchaudio.save(audio_path, audio, SAMPLE_RATE)

            (transcript_dir / "sample.srt").write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:01,000",
                        "Real speech before",
                        "",
                        "2",
                        "00:00:10,000 --> 00:00:12,000",
                        "Repeated credit",
                        "",
                        "3",
                        "00:00:12,000 --> 00:00:20,000",
                        "Repeated credit",
                        "",
                        "4",
                        "00:00:20,000 --> 00:00:30,000",
                        "Repeated credit",
                        "",
                        "5",
                        "00:00:45,000 --> 00:00:46,000",
                        "Real speech after",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = DataProcessor(
                audio_dir=audio_dir,
                transcript_dir=transcript_dir,
                output=output,
                dump_dir=dump_dir,
                keep_empty_chance=1.0,
            )
            processor.run()

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            names = [Path(record["audio_path"]).name for record in records]

            self.assertIn("0.flac", names)
            self.assertIn("1000.flac", names)  # true silence before hallucination
            self.assertIn("30000.flac", names)
            self.assertNotIn("10000.flac", names)
            self.assertTrue(all("Repeated credit" not in record["text"] for record in records))

            hallucination_report = root / "filtered_repeated_hallucination_segments.tsv"
            self.assertTrue(hallucination_report.exists())
            self.assertIn(
                "Repeated credit",
                hallucination_report.read_text(encoding="utf-8"),
            )

    def test_drop_text_keeps_audio_but_hard_filter_drops_segment_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio_dir = root / "audio"
            transcript_dir = root / "transcripts"
            dump_dir = root / "dump"
            output = root / "data.ljson"
            audio_dir.mkdir()
            transcript_dir.mkdir()

            audio_path = audio_dir / "sample.wav"
            audio = torch.zeros(1, SAMPLE_RATE * 18)
            torchaudio.save(audio_path, audio, SAMPLE_RATE)

            (transcript_dir / "sample.srt").write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:06,000",
                        "Live-Untertitel Willkommen zum Film",
                        "",
                        "2",
                        "00:00:06,000 --> 00:00:12,000",
                        "SECRET pii should not survive",
                        "",
                        "3",
                        "00:00:12,000 --> 00:00:18,000",
                        "Danach geht es weiter",
                        "",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            processor = DataProcessor(
                audio_dir=audio_dir,
                transcript_dir=transcript_dir,
                output=output,
                dump_dir=dump_dir,
                filter_segment_words=["SECRET"],
                drop_text=["Live-Untertitel"],
            )
            processor.run()

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertIn("Willkommen zum Film", records[0]["text"])
            self.assertNotIn("Live-Untertitel", records[0]["text"])
            self.assertTrue(all("SECRET" not in record["text"] for record in records))
            self.assertEqual(
                [Path(record["audio_path"]).name for record in records],
                ["0.flac", "12000.flac"],
            )

            first_audio, first_rate = torchaudio.load(output.parent / records[0]["audio_path"])
            second_audio, second_rate = torchaudio.load(output.parent / records[1]["audio_path"])
            self.assertEqual(round(first_audio.size(1) * 1000 / first_rate), 6000)
            self.assertEqual(round(second_audio.size(1) * 1000 / second_rate), 6000)
            self.assertFalse((dump_dir / "sample" / "6000.flac").exists())

    def test_filter_words_cut_real_fixture_audio_lengths_and_json_output(self):
        fixture_id = "filter_words_fixture"
        fixture_dir = Path(__file__).parent / "assets/filter_words"
        fixture_srt = fixture_dir / f"{fixture_id}.srt"
        transcripts_tsv = fixture_dir / "transcripts.tsv"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dump_dir = root / "dump"
            output = root / "data.ljson"

            filter_words = ["TV-Zuschauer", "pragmatisch"]
            filtered_out = []
            DataProcessor.read_utterances_from_srt(
                fixture_srt,
                filter_segment_words=filter_words,
                filtered_out=filtered_out,
                source_id=fixture_id,
            )
            self.assertGreater(len(filtered_out), 0)
            blocked_intervals = [(record["start_ms"], record["end_ms"]) for record in filtered_out]

            processor = DataProcessor(
                audio_dir="",
                transcript_dir="",
                output=output,
                dump_dir=dump_dir,
                filter_segment_words=filter_words,
                transcripts_tsv=transcripts_tsv,
            )
            processor.run()

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertGreater(len(records), 0)
            for record in records:
                self.assertNotIn("TV-Zuschauer", record["text"])
                self.assertNotIn("pragmatisch", record["text"])

            self.assertTrue(any(Path(record["audio_path"]).name != "0.flac" for record in records))

            for record in records:
                audio, sample_rate = torchaudio.load(output.parent / record["audio_path"])
                segment_start = int(Path(record["audio_path"]).stem)
                duration_ms = round(audio.size(1) * 1000 / sample_rate)
                segment_end = segment_start + duration_ms
                self.assertEqual(sample_rate, SAMPLE_RATE)
                self.assertLess(duration_ms, 30000)
                for blocked_start, blocked_end in blocked_intervals:
                    self.assertTrue(
                        segment_end <= blocked_start or segment_start >= blocked_end,
                        f"{record['audio_path']} overlaps filtered interval "
                        f"{blocked_start}-{blocked_end}",
                    )

            emitted_names = {path.name for path in (dump_dir / fixture_id).glob("*.flac")}
            for blocked_start, _ in blocked_intervals:
                self.assertNotIn(f"{blocked_start}.flac", emitted_names)
