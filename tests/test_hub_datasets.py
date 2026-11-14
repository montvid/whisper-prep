"""Integration tests: prepare public Hub noise audio as Whisper examples.

The former i4ds exports are unavailable. Pinned UrbanSound8K and MUSAN
sources now exercise DataProcessor and the project's HF conversion instead.
Downloads use the HF cache; network or processing failures fail the tests.
"""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.hub


URBANSOUND_REPO = "danavery/urbansound8K"
URBANSOUND_REVISION = "8aa9177a0c5a6949ee4ee4b7fcabb01dfd4ae466"
URBANSOUND_SHARD = "data/train-00000-of-00016-e478d7cccca6a095.parquet"

FREESOUNDS_REPO = "bilguun/musan-noise"
FREESOUNDS_REVISION = "d179de7b0c6adf58bf295495b481386b8fdb97c1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _urbansound_paths(directory: Path, n: int = 10) -> list[Path]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    # Cache one shard (~434 MB), not the full dataset. Read the local file
    # synchronously: early termination of remote Parquet streaming caused
    # interpreter-shutdown crashes with datasets 4.8.5 / pyarrow 24.0.0.
    shard = hf_hub_download(
        URBANSOUND_REPO,
        filename=URBANSOUND_SHARD,
        revision=URBANSOUND_REVISION,
        repo_type="dataset",
        token=False,
    )
    paths = []
    with pq.ParquetFile(shard) as parquet:
        for batch in parquet.iter_batches(
            batch_size=32,
            columns=["audio", "slice_file_name", "class"],
            use_threads=False,
        ):
            for row in batch.to_pylist():
                if row["class"] in {"children_playing", "street_music"}:
                    continue
                path = directory / Path(row["slice_file_name"]).name
                path.write_bytes(row["audio"]["bytes"])
                paths.append(path)
                if len(paths) == n:
                    return paths
    raise AssertionError(f"Expected {n} UrbanSound clips, found {len(paths)}")


def _freesound_paths(n: int = 5) -> list[Path]:
    from huggingface_hub import hf_hub_download

    return [
        Path(
            hf_hub_download(
                FREESOUNDS_REPO,
                filename=f"noise/free-sound/noise-free-sound-{index:04d}.wav",
                revision=FREESOUNDS_REVISION,
                repo_type="dataset",
                token=False,
            )
        )
        for index in range(n)
    ]


def _prepare_examples(
    audio_paths: list[Path],
    directory: Path,
    expected_segments: int,
    rejected_sources: set[str] | None = None,
) -> list[dict]:
    from datasets import Audio, load_from_disk

    from whisper_prep.dataset.convert import ljson_to_pandas, pandas_to_hf_dataset
    from whisper_prep.generation.data_processor import DataProcessor

    transcript_dir = directory / "transcripts"
    transcript_dir.mkdir()
    mapping = directory / "transcripts.tsv"
    with mapping.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "audio_path", "srt_path", "language"],
            delimiter="\t",
        )
        writer.writeheader()
        for audio_path in audio_paths:
            srt_path = transcript_dir / f"{audio_path.stem}.srt"
            srt_path.write_text("", encoding="utf-8")
            writer.writerow(
                {
                    "id": audio_path.stem,
                    "audio_path": str(audio_path),
                    "srt_path": str(srt_path),
                    "language": "en",
                }
            )

    output = directory / "data.ljson"
    DataProcessor(
        audio_dir=str(directory),
        transcript_dir=str(transcript_dir),
        transcripts_tsv=str(mapping),
        output=str(output),
        dump_dir=str(directory / "dump"),
        language="en",
        keep_empty_chance=1.0,
        validate_empty_with_vad=True,
        empty_vad_max_speech_ratio=0.0,
        n_jobs=1,
    ).run()

    records = ljson_to_pandas(output)
    # Check the retained output, including expected VAD rejections. Empty
    # or partially processed batches must not pass the assertions below.
    assert len(records) == expected_segments
    expected_sources = {path.stem for path in audio_paths} - (rejected_sources or set())
    assert {Path(path).parent.name for path in records["audio"]} == expected_sources

    dataset_dir = directory / "hf"
    pandas_to_hf_dataset(records).save_to_disk(dataset_dir)
    dataset = load_from_disk(dataset_dir)["train"].cast_column("audio", Audio(decode=False))
    examples = []
    for example in dataset:
        storage = example["audio"]
        source = BytesIO(storage["bytes"]) if storage["bytes"] is not None else storage["path"]
        # Inspect the encoded output's native rate. Audio(sampling_rate=16000)
        # decoding would resample it and conceal incorrectly stored audio.
        samples, sample_rate = sf.read(source, dtype="float32")
        example["audio"] = {"array": samples, "sampling_rate": sample_rate}
        examples.append(example)
    assert len(examples) == expected_segments
    return examples


# ---------------------------------------------------------------------------
# danavery/urbansound8K -> DataProcessor -> HF dataset
# ---------------------------------------------------------------------------


class TestUrbansoundHub:
    @pytest.fixture(scope="class")
    def examples(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("urbansound-hub")
        return _prepare_examples(_urbansound_paths(directory), directory, expected_segments=10)

    def test_has_expected_columns(self, examples):
        required = {"audio", "text", "language", "prompt"}
        for example in examples:
            assert required.issubset(example.keys()), (
                f"Missing columns: {required - example.keys()}"
            )

    def test_text_is_empty_silence(self, examples):
        non_empty = [e["text"] for e in examples if e["text"].strip()]
        assert not non_empty, f"Found non-empty text: {non_empty}"

    def test_language_is_english(self, examples):
        langs = {e["language"] for e in examples}
        assert langs == {"en"}, f"Unexpected languages: {langs}"

    def test_audio_is_16khz(self, examples):
        for ex in examples:
            assert ex["audio"]["sampling_rate"] == 16000
            assert ex["audio"]["array"].ndim == 1
            assert np.isfinite(ex["audio"]["array"]).all()

    def test_audio_duration_at_most_30s(self, examples):
        for ex in examples:
            arr = ex["audio"]["array"]
            sr = ex["audio"]["sampling_rate"]
            dur = len(arr) / sr
            assert dur <= 30.5, f"Segment too long: {dur:.2f}s"
            assert dur >= 0.1, f"Segment too short: {dur:.2f}s"

    def test_audio_has_non_zero_rms(self, examples):
        """Real background audio — should not be digital silence."""
        for ex in examples:
            arr = np.array(ex["audio"]["array"])
            rms = np.sqrt(np.mean(arr**2))
            assert rms > 1e-4, f"Audio looks like dead silence (RMS={rms:.6f})"


# ---------------------------------------------------------------------------
# bilguun/musan-noise -> DataProcessor -> HF dataset
# ---------------------------------------------------------------------------


class TestFreeSoundsHub:
    @pytest.fixture(scope="class")
    def examples(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("freesounds-hub")
        # The two segments from 0001 trigger speech detection and must not
        # become empty-transcript examples. The other four clips yield six
        # segments, including three from the 71-second recording 0002.
        return _prepare_examples(
            _freesound_paths(),
            directory,
            expected_segments=6,
            rejected_sources={"noise-free-sound-0001"},
        )

    def test_has_expected_columns(self, examples):
        required = {"audio", "text", "language", "prompt"}
        for example in examples:
            assert required.issubset(example.keys())

    def test_text_is_empty_silence(self, examples):
        non_empty = [e["text"] for e in examples if e["text"].strip()]
        assert not non_empty, f"Found non-empty text: {non_empty}"

    def test_language_is_english(self, examples):
        langs = {e["language"] for e in examples}
        assert langs == {"en"}, f"Unexpected languages: {langs}"

    def test_audio_is_16khz(self, examples):
        for ex in examples:
            assert ex["audio"]["sampling_rate"] == 16000
            assert ex["audio"]["array"].ndim == 1
            assert np.isfinite(ex["audio"]["array"]).all()

    def test_audio_duration_at_most_30s(self, examples):
        for ex in examples:
            arr = ex["audio"]["array"]
            dur = len(arr) / ex["audio"]["sampling_rate"]
            assert dur <= 30.5, f"Segment too long: {dur:.2f}s"
            assert dur >= 0.1, f"Segment too short: {dur:.2f}s"
