"""Contracts for the seven findings in docs/review-additional.md."""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from whisper_prep import main
from whisper_prep.audio.io import read_audio
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.subtitling.parse import read_cues


def wav(path, seconds=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(seconds * 16000) / 16000
    sf.write(path, 0.1 * np.sin(2 * np.pi * 220 * t), 16000)
    return path


def sentence_tsv(path, rows, column="path", dialect="csv"):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(
            stream,
            delimiter="\t",
            quoting=csv.QUOTE_MINIMAL if dialect == "csv" else csv.QUOTE_NONE,
        )
        writer.writerow([column, "sentence", "client_id"])
        writer.writerows(rows)
    return path


def sentence_config(tmp_path, tsv, **options):
    return dict(
        out_folder_base=str(tmp_path / "output"),
        dataset_name="review",
        split_name="train",
        tsv_paths=[str(tsv)],
        clips_folders=[str(tmp_path)],
        language="en",
        n_jobs=1,
        vad_chance=0,
        overlap_chance=0,
        min_text_words=0,
        **options,
    )


@pytest.mark.parametrize("source_dir", ["source", "dump"])
def test_silence_subsampling_preserves_original_audio(tmp_path, source_dir):
    sources = [wav(tmp_path / source_dir / f"source-{index}.wav") for index in range(3)]
    originals = [path.read_bytes() for path in sources]
    metadata = tmp_path / "sentences.tsv"
    metadata.write_text("".join(f"{path}\t\n" for path in sources), encoding="utf-8")
    processor = DataProcessor(
        audio_dir="",
        transcript_dir="",
        with_timestamps=False,
        data_file=str(metadata),
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        subsampling_factor_for_silence=2,
    )
    processor.run()
    assert len(DataProcessor.read_records(tmp_path / "data.ljson")) == 2
    assert [path.read_bytes() for path in sources] == originals


@pytest.mark.parametrize("separator", ["\n\n", "\r\n\r\n", "\u2029"])
def test_multiline_sentence_keeps_all_spoken_text(tmp_path, separator):
    wav(tmp_path / "clip.wav")
    text = f"First spoken sentence.{separator}Second spoken sentence."
    tsv = sentence_tsv(tmp_path / "sentences.tsv", [("clip.wav", text, "speaker")])
    dataset = main(sentence_config(tmp_path, tsv))
    texts = " ".join(dataset["train"]["text"])
    assert "First spoken sentence. Second spoken sentence." in texts


@pytest.mark.parametrize(
    "dialect,column,reference",
    [
        ("csv", "path", "relative"),
        ("none", "clip_path", "relative"),
        ("csv", "path", "absolute"),
        ("csv", "clip_path", "symlink"),
    ],
)
def test_overwrite_checks_sentence_tsv_audio_references(tmp_path, dialect, column, reference):
    source = wav(tmp_path / "output/review/train/audios/original.wav")
    original = source.read_bytes()
    path = source if reference == "absolute" else source.relative_to(tmp_path)
    if reference == "symlink":
        alias = tmp_path / "linked.wav"
        alias.symlink_to(source)
        path = alias.name
    tsv = sentence_tsv(
        tmp_path / "sentences.tsv", [(path, "A valid sentence.", "speaker")], column, dialect
    )
    with pytest.raises(ValueError, match="inside generated output"):
        main(sentence_config(tmp_path, tsv, overwrite_output=True, tsv_quoting=dialect))
    assert source.read_bytes() == original


def test_overwrite_cannot_delete_sentence_metadata_named_like_a_report(tmp_path):
    wav(tmp_path / "clip.wav")
    output = tmp_path / "output/review/train"
    output.mkdir(parents=True)
    tsv = sentence_tsv(output / "bad_examples.tsv", [("clip.wav", "A sentence.", "speaker")])
    original = tsv.read_bytes()
    with pytest.raises(ValueError, match="inside generated output"):
        main(sentence_config(tmp_path, tsv, overwrite_output=True))
    assert tsv.read_bytes() == original


@pytest.mark.parametrize("suffix", ["srt", "vtt"])
@pytest.mark.parametrize(
    "separator,timing",
    [
        ("\n\n", "00:00:01,500 --> BAD"),
        ("\n", "00:00:01,500 --> BAD"),
        ("\n\n", "00:00:01,500 -> 00:00:02,000"),
    ],
)
def test_malformed_timing_is_not_silently_ignored(tmp_path, suffix, separator, timing):
    path = tmp_path / f"source.{suffix}"
    header = "WEBVTT\n\n" if suffix == "vtt" else ""
    path.write_text(
        header
        + "1\n00:00:00,000 --> 00:00:01,000\nValid sentence."
        + separator
        + f"2\n{timing}\nLost sentence.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="subtitle timing"):
        read_cues(path)


@pytest.mark.parametrize(
    "suffix,content",
    [
        ("srt", ""),
        ("vtt", "WEBVTT\n\n"),
        (
            "vtt",
            "WEBVTT\nX-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:0\n\nNOTE QA\nignore --> this\n",
        ),
    ],
)
def test_valid_empty_subtitles_remain_supported(tmp_path, suffix, content):
    path = tmp_path / f"empty.{suffix}"
    path.write_text(content, encoding="utf-8")
    assert read_cues(path) == []


@pytest.mark.parametrize(
    "content",
    [
        "WEBVTT\n\n00:00.000 --> INVALID\nReal speech.\n",
        "WEBVTT\n\nCue with missing timing and real speech.\n",
    ],
)
def test_invalid_vtt_is_not_treated_as_silence(tmp_path, content):
    path = tmp_path / "source.vtt"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="subtitle timing"):
        read_cues(path)


@pytest.mark.parametrize(
    "predicate,text",
    [
        ("is_english", "This is the English sentence that should be filtered."),
        ("is_french", "C'est une phrase avec des mots français."),
    ],
)
def test_corrupt_language_model_is_not_silently_ignored(tmp_path, monkeypatch, predicate, text):
    from whisper_prep import language, utils

    cache = tmp_path / "bad.ftz"
    cache.write_bytes(b"corrupt cache")
    monkeypatch.setenv("WHISPER_PREP_LANGUAGE_MODEL", str(cache))
    language.get_language_model.cache_clear()
    try:
        with pytest.raises(ValueError, match="checksum"):
            getattr(utils, predicate)(text)
    finally:
        language.get_language_model.cache_clear()


@pytest.mark.parametrize("suffix", ["srt", "vtt"])
def test_clock_text_with_an_arrow_is_not_a_broken_timing_line(tmp_path, suffix):
    text = "12:45 --> 13:30 is the scheduled interval."
    path = tmp_path / f"source.{suffix}"
    header = "WEBVTT\n\n" if suffix == "vtt" else ""
    path.write_text(header + f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n", encoding="utf-8")
    assert [cue.text for cue in read_cues(path)] == [text]


def test_successful_language_prediction_without_labels_is_unknown(monkeypatch):
    from whisper_prep import language, utils

    class Model:
        def predict(self, text, k):
            return [], []

    monkeypatch.setattr(language, "get_language_model", Model)
    assert not utils.is_english("This is the test sentence.")
    assert not utils.is_french("Une phrase avec des mots.")


def test_sentence_fusion_accepts_opus_audio(tmp_path):
    from pydub import AudioSegment

    source = wav(tmp_path / "clip.wav")
    opus = tmp_path / "clip.opus"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-c:a", "libopus", str(opus)], check=True
    )
    assert AudioSegment.from_file(opus).duration_seconds > 2.9
    assert read_audio(opus, resample_rate=16000).duration_seconds > 2.9
    tsv = sentence_tsv(tmp_path / "sentences.tsv", [(opus.name, "Opus spoken text.", "speaker")])
    dataset = main(sentence_config(tmp_path, tsv))
    assert "Opus spoken text." in " ".join(dataset["train"]["text"])


def mapping(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["id", "audio_path", "srt_path"])
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("normalize,allowed_rate", [(False, 0.5), (True, 0.5), (True, 0.0)])
def test_normalization_honors_allowed_file_error_rate(tmp_path, normalize, allowed_rate, n_jobs):
    audio = wav(tmp_path / "source.wav")
    subtitle = tmp_path / "valid.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:03,000\nA useful spoken sentence.\n")
    missing = tmp_path / "missing.srt"
    tsv = mapping(
        tmp_path / "mapping.tsv", [("valid", audio, subtitle), ("missing", audio, missing)]
    )
    config = dict(
        out_folder_base=str(tmp_path / "output"),
        dataset_name="review",
        split_name="train",
        transcripts_tsv=str(tsv),
        language="en",
        min_text_words=0,
        n_jobs=n_jobs,
        netflix_normalize=normalize,
        max_error_rate=allowed_rate,
    )
    if allowed_rate == 0:
        with pytest.raises(RuntimeError, match="Failed to process 1/2 files"):
            main(config)
    else:
        dataset = main(config)
        assert len(dataset["train"]) == 1
    created = tmp_path / "output/review/train/created_dataset"
    summary = json.loads((created / "data.summary.json").read_text())
    assert summary["files"] == 2 and summary["failed_files"] == 1
    errors = json.loads((created / "data.errors.json").read_text())
    assert len(errors) == 1 and errors[0]["type"] == "FileNotFoundError"
    assert errors[0]["audio_path"] == str(audio)
    assert errors[0]["transcript_path"] == str(missing)
    if normalize:
        assert errors[0]["stage"] == "normalization"
    assert not list((created / "dump/missing").glob("*.mp3"))


def test_all_normalization_failures_can_export_an_empty_dataset(tmp_path):
    audio = wav(tmp_path / "source.wav")
    tsv = mapping(tmp_path / "mapping.tsv", [("missing", audio, tmp_path / "missing.srt")])
    dataset = main(
        dict(
            out_folder_base=str(tmp_path / "output"),
            dataset_name="review",
            split_name="train",
            transcripts_tsv=str(tsv),
            netflix_normalize=True,
            max_error_rate=1,
        )
    )
    assert len(dataset["train"]) == 0
    summary = json.loads(
        (tmp_path / "output/review/train/created_dataset/data.summary.json").read_text()
    )
    assert summary["files"] == summary["failed_files"] == 1


def test_malformed_vtt_cannot_export_real_speech_as_silence(tmp_path):
    from whisper_prep.audio.vad import silero_speech_ratio

    fixture = Path(__file__).parent / "assets/empty_vad/empty_with_speech.mp3"
    assert silero_speech_ratio(fixture) > 0.06
    audio = tmp_path / "speech.mp3"
    shutil.copyfile(fixture, audio)
    subtitle = tmp_path / "speech.vtt"
    subtitle.write_text("WEBVTT\n\n00:00.000 --> INVALID\nSpoken words in the audio.\n")
    tsv = mapping(tmp_path / "mapping.tsv", [("speech", audio, subtitle)])
    config = dict(
        out_folder_base=str(tmp_path / "output"),
        dataset_name="review",
        split_name="train",
        transcripts_tsv=str(tsv),
        language="en",
        min_text_words=0,
        keep_empty_chance=1,
    )
    with pytest.raises(RuntimeError, match="Failed to process 1/1 files"):
        main(config)
    created = tmp_path / "output/review/train/created_dataset"
    assert not (created / "data.ljson").read_text()
    assert not list((created / "dump").rglob("*.mp3"))
    summary = json.loads((created / "data.summary.json").read_text())
    assert summary["failed_files"] == 1
