"""Regression contracts for the confirmed review.md findings."""

import copy
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from whisper_prep import main, utils
from whisper_prep.audio import vad
from whisper_prep.config import PipelineConfig
from whisper_prep.dataset import download
from whisper_prep.dataset.convert import (
    combine_tsvs_to_dataframe,
    ljson_to_pandas,
    pandas_to_hf_dataset,
)
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import _ordered_samples, generate_fold
from whisper_prep.generation.text_normalizer import SwissNumberConverter, normalize_text
from whisper_prep.generation.typing import PromptNode
from whisper_prep.pipeline import _apply_basic_text_filters
from whisper_prep.subtitling.srt import Caption, generate_srt
from whisper_prep.upload_asr_dataset import dataset_from_tsv


def wav(path, seconds=3, frequency=220):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(round(16000 * seconds)) / 16000
    sf.write(path, 0.1 * np.sin(2 * np.pi * frequency * t), 16000)
    return path


def stamp(ms):
    seconds, fraction = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{fraction:03d}"


def subtitles(path, cues):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(
            f"{index}\n{stamp(start)} --> {stamp(end)}\n{text}"
            for index, (start, end, text) in enumerate(cues, 1)
        ),
        encoding="utf-8",
    )
    return path


def process(root, cues, seconds, **options):
    wav(root / "audio/example.wav", seconds)
    subtitles(root / "subtitles/example.srt", cues)
    processor = DataProcessor(
        audio_dir=root / "audio",
        transcript_dir=root / "subtitles",
        output=root / "data.ljson",
        dump_dir=root / "dump",
        **options,
    )
    processor.run()
    return processor, DataProcessor.read_records(root / "data.ljson")


def mapping(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, delimiter="\t", fieldnames=["id", "audio_path", "srt_path", "language"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_overlap_does_not_discard_following_windows(tmp_path):
    cues = [
        (i * 3000 + 100, i * 3000 + 2900, f"cue{i:03d} has a unique spoken sentence")
        for i in range(60)
    ]
    cues[3] = (cues[2][1] - 300, cues[3][1], cues[3][2])
    processor, records = process(tmp_path, cues, 181)
    text = " ".join(record.text for record in records)
    assert all(text.count(f"cue{i:03d}") == 1 for i in range(60))
    assert processor.stats["repaired_overlaps"] == 1
    assert processor.stats["invalid_windows"] == 0


def test_dense_lithuanian_is_split_to_token_budget(tmp_path):
    cues = [
        (
            i * 2500,
            i * 2500 + 2400,
            f"Numeris {i:03d}. Lietuvos gyventojai šiandien diskutuoja apie švietimo sistemos ateitį ir naujas galimybes.",
        )
        for i in range(24)
    ]
    processor, records = process(tmp_path, cues, 61, language="lt")
    text = " ".join(record.text for record in records)
    assert all(text.count(f"Numeris {i:03d}.") == 1 for i in range(24))
    assert len(records) > 2
    for record in records:
        assert len(processor.tokenizer.encode(record.text, allowed_special="all")) <= 219
        assert all(0 <= float(value) <= 30 for value in re.findall(r"<\|([0-9.]+)\|>", record.text))


def test_exact_budget_and_prompt_limit(tmp_path):
    from whisper.tokenizer import get_tokenizer

    text = "A fully preserved sentence"
    budget = len(get_tokenizer(multilingual=True).encode(" " + text)) + 2
    processor, records = process(
        tmp_path, [(0, 2000, text), (3000, 5000, "A second sentence")], 6, max_tokens_length=budget
    )
    assert text in " ".join(record.text for record in records)
    assert all(
        len(processor.tokenizer.encode(record.text, allowed_special="all")) <= budget
        for record in records
    )
    nodes = deque([PromptNode("old", 60), PromptNode("middle", 150), PromptNode("new", 150)])
    assert processor._get_prompt(nodes) == "new"
    assert sum(node.num_tokens for node in nodes) <= processor.max_prompt_length


@pytest.mark.parametrize(
    "last_end,reason", [(8000, "invalid_timestamps"), (14000, "outside_audio")]
)
def test_bad_final_cue_is_reported_without_losing_good_cues(tmp_path, last_end, reason):
    processor, records = process(
        tmp_path,
        [
            (0, 3000, "first valid sentence"),
            (4000, 7000, "second valid sentence"),
            (9000, last_end, "bad final sentence"),
        ],
        12,
        keep_empty_chance=1,
    )
    text = " ".join(record.text for record in records)
    assert "first valid sentence" in text and "second valid sentence" in text
    assert "bad final sentence" not in text
    assert processor.stats[f"dropped_{reason}"] == 1
    assert list(tmp_path.glob(f"filtered_{reason}_segments.tsv"))


def test_small_audio_overrun_clamped_and_no_false_silence(tmp_path):
    processor, records = process(
        tmp_path,
        [
            (0, 3000, "first sentence"),
            (4000, 8000, "second sentence"),
            (9000, 12400, "final sentence"),
        ],
        12,
        keep_empty_chance=1,
    )
    assert "final sentence" in " ".join(record.text for record in records)
    assert processor.stats["clamped_cues"] == 1
    assert all(record.text for record in records)


def test_repetition_threshold_and_filter_filename(tmp_path):
    processor, records = process(
        tmp_path,
        [
            (0, 1000, "keep first"),
            (1200, 2200, "repeat this"),
            (2300, 3300, "repeat this"),
            (4000, 5000, "N/A marker"),
            (5500, 6500, "keep final"),
        ],
        7,
        rep_threshold=2,
        filter_segment_words=["N/A"],
    )
    text = " ".join(record.text for record in records)
    assert "keep first" in text and "keep final" in text
    assert "repeat this" not in text and "N/A" not in text
    assert processor.stats["dropped_repeated_hallucination"] == 1
    assert len(list(tmp_path.glob("filtered_*_segments.tsv"))) == 2


def test_duplicate_ids_rejected_before_writing_audio(tmp_path):
    rows = [
        dict(id="same", audio_path="first.wav", srt_path="first.srt", language="en"),
        dict(id="same", audio_path="second.wav", srt_path="second.srt", language="en"),
    ]
    path = mapping(tmp_path / "mapping.tsv", rows)
    processor = DataProcessor(
        "", "", transcripts_tsv=path, output=tmp_path / "data.ljson", dump_dir=tmp_path / "dump"
    )
    with pytest.raises(ValueError, match="Duplicate output id"):
        processor.run()
    assert not any(path.is_file() for path in (tmp_path / "dump").rglob("*"))


def test_tsv_paths_ids_and_languages_are_validated(tmp_path, monkeypatch):
    wav(tmp_path / "source.wav")
    subtitles(tmp_path / "source.srt", [(0, 2000, "safe text")])
    path = mapping(
        tmp_path / "mapping.tsv",
        [dict(id="../../escape", audio_path="source.wav", srt_path="source.srt", language="en")],
    )
    monkeypatch.chdir(tmp_path.parent)
    processor = DataProcessor(
        "", "", transcripts_tsv=path, output=tmp_path / "data.ljson", dump_dir=tmp_path / "dump"
    )
    processor.run()
    records = DataProcessor.read_records(tmp_path / "data.ljson")
    assert records and all(
        Path(record.audio_path).is_relative_to(tmp_path / "dump") for record in records
    )
    assert not Path(
        json.loads((tmp_path / "data.ljson").read_text().splitlines()[0])["audio_path"]
    ).is_absolute()
    path = mapping(
        tmp_path / "invalid.tsv",
        [dict(id="x", audio_path="source.wav", srt_path="source.srt", language="invalid")],
    )
    with pytest.raises(ValueError, match="Unsupported language"):
        DataProcessor(
            "",
            "",
            transcripts_tsv=path,
            output=tmp_path / "invalid.ljson",
            dump_dir=tmp_path / "dump2",
        ).run()


def test_parallel_order_rng_and_empty_mapping(tmp_path):
    rows = []
    for i in range(5):
        wav(tmp_path / f"audio{i}.wav", 62)
        subtitles(tmp_path / f"audio{i}.srt", [])
        rows.append(
            dict(id=f"id{i}", audio_path=f"audio{i}.wav", srt_path=f"audio{i}.srt", language="en")
        )
    path = mapping(tmp_path / "mapping.tsv", rows)
    results = []
    for workers in (1, 2):
        output = tmp_path / f"run{workers}"
        DataProcessor(
            "",
            "",
            transcripts_tsv=path,
            output=output / "data.ljson",
            dump_dir=output / "dump",
            n_jobs=workers,
            seed=123,
            keep_empty_chance=0.5,
        ).run()
        results.append((output / "data.ljson").read_text())
        assert not (output / "_parts").exists()
    assert results[0] == results[1]
    empty = mapping(tmp_path / "empty.tsv", [])
    DataProcessor(
        "",
        "",
        transcripts_tsv=empty,
        output=tmp_path / "empty.ljson",
        dump_dir=tmp_path / "empty-dump",
        n_jobs=2,
    ).run()
    assert DataProcessor.read_records(tmp_path / "empty.ljson") == []


def test_missing_audio_fails_with_traceback_and_totals(tmp_path, caplog):
    subtitles(tmp_path / "source.srt", [(0, 1000, "valid transcript")])
    path = mapping(
        tmp_path / "mapping.tsv",
        [dict(id="x", audio_path="missing.wav", srt_path="source.srt", language="en")],
    )
    with pytest.raises(RuntimeError, match="1/1"):
        DataProcessor(
            "", "", transcripts_tsv=path, output=tmp_path / "data.ljson", dump_dir=tmp_path / "dump"
        ).run()
    assert "missing.wav" in caplog.text
    assert any(record.exc_info for record in caplog.records)
    assert json.loads((tmp_path / "data.summary.json").read_text())["failed_files"] == 1


@pytest.mark.parametrize(
    "suffix,content,expected",
    [
        (
            ".srt",
            "1\n00:00:00,000 --> 00:00:01,000\nFirst\n2\n00:00:01,000 --> 00:00:02,000\nSecond",
            ["First", "Second"],
        ),
        (
            ".srt",
            "00:00:00,000 --> 00:00:01,000\nFirst\n\n00:00:01,000 --> 00:00:02,000\nSecond",
            ["First", "Second"],
        ),
        (
            ".vtt",
            "WEBVTT\n\n00:00.000 --> 00:01.000\n我\n\nNOTE checked by QA\nignore\n\nSTYLE\n::cue { color: red }\n\nREGION\nid=1\n\n00:01.000 --> 00:02.000\nText --> still text",
            ["我", "Text --> still text"],
        ),
        (".srt", "1\n00:00:00,000 --> 00:00:01,000\n我", ["我"]),
    ],
)
def test_subtitle_parser_layouts(tmp_path, suffix, content, expected):
    path = tmp_path / ("sample" + suffix)
    path.write_text(content, encoding="utf-8")
    assert [cue.text for cue in DataProcessor.read_utterances_from_srt(path)] == expected


def test_subtitle_encoding_and_normalization_preserve_source(tmp_path):
    path = tmp_path / "source.srt"
    original = "1\n00:00:00,000 --> 00:00:01,000\nGrüße Müller".encode("cp1252")
    path.write_bytes(original)
    assert DataProcessor.read_utterances_from_srt(path)[0].text == "Grüße Müller"
    output = utils.netflix_normalize_file(path)
    assert output != path and output.exists() and path.read_bytes() == original


def sentence_tsv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["path", "sentence", "client_id"])
        writer.writerows(rows)
    return path


def test_tsv_preserves_quotes_missing_value_words_and_speakers(tmp_path):
    texts = ['"Hallo", sagte er.', "Quoted\ttab and\nnewline", "NA", "null", "007", ""]
    path = sentence_tsv(
        tmp_path / "sentences.tsv", [(f"{i}.wav", text, "") for i, text in enumerate(texts)]
    )
    data = combine_tsvs_to_dataframe([path], [tmp_path], [1])
    assert data.sentence.tolist() == texts
    assert data.client_id.tolist() == ["0"] * len(texts)
    with pytest.raises(ValueError, match="equal lengths"):
        combine_tsvs_to_dataframe([path, path], [tmp_path, tmp_path], [1])


def test_raw_tsv_dialect_is_explicit_and_subsampling_seeded(tmp_path):
    path = tmp_path / "raw.tsv"
    path.write_text('path\tsentence\tclient_id\na.wav\t"Unclosed quote\t\nb.wav\tNA\t\n')
    with pytest.raises(ValueError, match="tsv_quoting"):
        combine_tsvs_to_dataframe([path], [tmp_path], [1])
    raw = combine_tsvs_to_dataframe([path], [tmp_path], [1], tsv_quoting="none")
    assert raw.sentence.tolist() == ['"Unclosed quote', "NA"]
    path = sentence_tsv(tmp_path / "many.tsv", [(f"{i}.wav", str(i), "speaker") for i in range(30)])
    first = combine_tsvs_to_dataframe([path], [tmp_path], [0.5], seed=27)
    np.random.seed(999)
    second = combine_tsvs_to_dataframe([path], [tmp_path], [0.5], seed=27)
    assert first.equals(second)


@pytest.mark.parametrize(
    "text",
    [
        "Im Jahr 2021 zogen wir nach 8004 Zürich.",
        "Der Laden ist 24/7 offen.",
        "Das war am 12.05.2021.",
        "Baden-Württemberg und Nord-Süd-Achse.",
        "Schick mir eine E-Mail, Ex-Präsident.",
        "Fr. Müller wohnt in EUROPA.",
        "Agent 007 ruft 0041 an.",
        "Er sagte (leise) nein.",
        "About 50 % of the COUNTRY uses PHP.",
        "Apie 50 % žmonių, t. y. 1 000 € per metus 2024.",
        "McDonald and YouTube opened in 1999.",
        "Seite 3/0 fehlt.",
    ],
)
def test_multilingual_normalization_preserves_spoken_content(text):
    assert normalize_text(text) == text


def test_explicit_swiss_normalization_does_not_corrupt_identifiers():
    converter = SwissNumberConverter()
    assert converter.combine_numbers("150 000") == "150000"
    assert converter.convert("007 0041 2021 8004 24/7 3/0") == "007 0041 2021 8004 24/7 3/0"
    assert converter.convert("1⁄2 3⁄0") == "0.5 3⁄0"
    assert (
        converter.convert("Fr. Müller COUNTRY EUROPA PHP 50% 10 EUR")
        == "Fr. Müller COUNTRY EUROPA PHP 50 Prozent 10 Euro"
    )


def test_language_probability_compression_and_word_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "fastlid", lambda text: ("fr", 0.01))
    assert not utils.is_french("une phrase avec des mots", 0.85)
    monkeypatch.setattr(utils, "fastlid", lambda text: ("en", 0.99))
    assert utils.is_english("the person with the book", 0.85)
    phrase = "ქართული Ελληνικά русский Lietuva " * 12
    assert utils.get_compression_ratio(phrase) == len(phrase.encode()) / len(
        zlib.compress(phrase.encode())
    )
    frame = pd.DataFrame(
        {
            "text": [
                "<|0.00|> one two three four five six seven<|2.00|>",
                "<|0.00|> one two three four five six seven eight<|2.00|>",
                "<|0.00|>这是一个很好的中文测试句子<|2.00|>",
            ],
            "language": ["en", "en", "zh"],
        }
    )
    result = _apply_basic_text_filters(frame, tmp_path)
    assert result.index.tolist() == [1, 2]


def test_package_and_helpers_import_without_models_or_downloads():
    code = """import sys, importlib.util
import whisper_prep
import whisper_prep.generation.text_normalizer
import whisper_prep.download_hf_dataset
import whisper_prep.push_to_hub
assert not {'torch', 'datasets', 'fastlid', 'silero_vad', 'pandas'}.intersection(sys.modules)
assert whisper_prep.__version__ == '0.1.0'
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    result = subprocess.run(
        [sys.executable, "-c", "from whisper_prep import main; main()"],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 2 and "--config" in result.stderr and "Traceback" not in result.stderr
    )


def test_vad_uses_sample_timestamps(monkeypatch):
    calls = []
    monkeypatch.setattr(vad, "get_vad_model", lambda: object())

    def detect(audio, model, **kwargs):
        calls.append(kwargs)
        return [{"start": 123, "end": 4567}]

    monkeypatch.setattr(vad, "get_speech_timestamps", detect)
    start, end = vad.silero_vad_collector(np.zeros(16000, dtype=np.float32))
    assert (start, end) == (123 / 16000, 4567 / 16000)
    assert calls[0]["return_seconds"] is False


def test_fusion_seed_independent_of_worker_and_caller_rng(tmp_path):
    rows = []
    for index in range(8):
        wav(tmp_path / f"clip{index}.wav", 1, frequency=220 + 20 * index)
        rows.append(
            (f"clip{index}.wav", f"Distinct sentence number {index}", f"speaker{index % 3}")
        )
    path = sentence_tsv(tmp_path / "sentences.tsv", rows)
    outputs = []
    for workers in (1, 2):
        random.seed(workers * 999)
        output = tmp_path / f"fusion{workers}"
        generate_fold(
            [path],
            [tmp_path],
            [1],
            output,
            n_samples_per_srt=2,
            n_jobs=workers,
            seed=19,
            vad_chance=0,
            audio_format="wav",
        )
        outputs.append(
            {
                str(file.relative_to(output)): hashlib.sha256(file.read_bytes()).hexdigest()
                for file in output.rglob("*")
                if file.is_file()
            }
        )
    assert outputs[0] == outputs[1]


def test_active_speaker_sampler_visits_every_row():
    frame = pd.DataFrame(
        {
            "sentence": [str(i) for i in range(10000)],
            "audio_file_path": ["x"] * 10000,
            "client_id": [str(i) for i in range(2000)] + ["large"] * 8000,
        }
    )
    rows = list(_ordered_samples(frame, 0.5, random.Random(42)))
    assert len(rows) == 10000 and len({row.sentence for row in rows}) == 10000


def test_generated_srt_overlap_never_reverses_cues(tmp_path):
    cues = [Caption(0, 1, "first"), Caption(0.995, 1.005, "tiny")]
    path = tmp_path / "sample.srt"
    generate_srt(cues, path)
    assert cues[1].start_second == 0.995
    parsed = DataProcessor.read_utterances_from_srt(path)
    assert len(parsed) == 2 and all(cue.start < cue.end for cue in parsed)
    assert "," in path.read_text().splitlines()[1]


def pipeline_config(root):
    audio = wav(root / "source.wav", 35)
    subs = subtitles(root / "source.srt", [(1000, 2000, "A valid source sentence")])
    path = mapping(
        root / "mapping.tsv",
        [dict(id="source", audio_path=str(audio), srt_path=str(subs), language="en")],
    )
    return dict(
        out_folder_base=str(root / "output"),
        dataset_name="test",
        split_name="train",
        transcripts_tsv=str(path),
        min_text_words=0,
        keep_empty_chance=1,
    )


def test_output_preflight_config_copy_and_overwrite(tmp_path):
    config = pipeline_config(tmp_path)
    before = copy.deepcopy(config)
    main(config)
    assert config == before
    folder = tmp_path / "output/test/train"
    files_before = {
        str(file.relative_to(folder)): file.stat().st_size
        for file in folder.rglob("*")
        if file.is_file()
    }
    with pytest.raises(FileExistsError):
        main(config)
    assert files_before == {
        str(file.relative_to(folder)): file.stat().st_size
        for file in folder.rglob("*")
        if file.is_file()
    }
    main({**config, "overwrite_output": True})
    assert not list(folder.glob("*.stale.*"))
    assert len(DataProcessor.read_records(folder / "created_dataset/data.ljson")) == 3


def test_empty_filtered_export_and_orphan_cleanup(tmp_path):
    config = pipeline_config(tmp_path)
    result = main({**config, "min_text_words": 999, "drop_empty_text": True})
    assert len(result["train"]) == 0
    folder = tmp_path / "output/test/train/created_dataset"
    assert not any(path.is_file() for path in (folder / "dump").rglob("*"))
    assert DataProcessor.read_records(folder / "data.ljson") == []
    assert len(pandas_to_hf_dataset(ljson_to_pandas(folder / "data.ljson"))["train"]) == 0
    from datasets import Audio, load_from_disk

    restored = load_from_disk(str(folder.parent / "hf"))
    assert len(restored["train"]) == 0
    assert isinstance(restored["train"].features["audio"], Audio)


def test_configuration_errors_before_side_effects(tmp_path):
    config = pipeline_config(tmp_path)
    for extra, message in [
        ({"unknown_key": 42}, "Unknown"),
        ({"n_jobs": 0}, "positive"),
        ({"upload_to_hu": True}, "hu_repo"),
        ({"max_tokens_length": 0}, "positive"),
    ]:
        with pytest.raises(ValueError, match=message):
            main({**config, **extra})
    assert not (tmp_path / "output").exists()
    parsed = PipelineConfig.from_dict({**config, "upload_to_hu": True, "hu_repo": "owner/repo"})
    assert parsed.hu_private is True
    with pytest.raises(ValueError, match="exactly one"):
        PipelineConfig.from_dict(
            dict(out_folder_base=str(tmp_path), dataset_name="test", split_name="train")
        )


def test_pipeline_subtitle_normalization_leaves_inputs_unchanged(tmp_path):
    config = pipeline_config(tmp_path)
    source = tmp_path / "source.srt"
    before = source.read_bytes()
    main({**config, "netflix_normalize": True})
    assert source.read_bytes() == before
    assert (tmp_path / "output/test/train/normalized_transcripts/source.srt").exists()


def test_upload_helper_keeps_each_rows_subtitles(tmp_path):
    rows = []
    for i in range(2):
        wav(tmp_path / f"{i}.wav")
        subtitles(tmp_path / f"{i}.srt", [(0, 2000, f"Transcript number {i}")])
        rows.append({"path": f"{i}.wav", "srt_path": f"{i}.srt"})
    path = tmp_path / "upload.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    from datasets import Audio

    result = dataset_from_tsv(path)["train"].cast_column("audio", Audio(decode=False))
    assert result[0]["sentence"] == "Transcript number 0"
    assert result[1]["sentence"] == "Transcript number 1"
    assert result[0]["srt"] != result[1]["srt"]


def test_downloader_mixed_text_schemas_and_id_collisions(tmp_path, monkeypatch):
    from tests.test_download_hf_dataset import FakeDataset, fake_audio

    sources = {
        "first": FakeDataset([{"id": "a/b", "sentence": "NA", "audio": fake_audio()}]),
        "second": FakeDataset([{"id": "a_b", "text": "null", "audio": fake_audio()}]),
    }
    monkeypatch.setattr(download, "load_dataset", lambda name, split: sources[name])
    result = download.save_hu_dataset_locally(
        dict(out_folder=tmp_path, hu_datasets=list(sources)),
        tmp_path / "audio",
        tmp_path / "transcripts",
    )
    with Path(result.sentence_tsvs[0]).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["sentence"] for row in rows] == ["NA", "null"]
    assert len({row["path"] for row in rows}) == 2
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 2


def test_unusable_cue_blocks_speech_and_resets_prompt(tmp_path):
    cues = [
        (0, 2000, "before rejected cue"),
        (3000, 34000, "oversized utterance"),
        (35000, 37000, "after rejected cue"),
    ]
    processor, records = process(tmp_path, cues, 40, keep_empty_chance=1)
    assert processor.stats["dropped_cue_duration"] == 1
    assert all("oversized utterance" not in record.text for record in records)
    after = next(record for record in records if "after rejected cue" in record.text)
    assert after.prompt == ""
    for record in records:
        start = int(Path(record.audio_path).stem)
        length = sf.info(record.audio_path).duration * 1000
        assert start >= 34000 or start + length <= 3001


def test_oversized_token_cue_is_reported_and_following_cue_survives(tmp_path):
    processor, records = process(
        tmp_path,
        [
            (0, 1000, "previous"),
            (1500, 3000, "unusually lengthy sentence " * 20),
            (4000, 6000, "following"),
        ],
        7,
        max_tokens_length=12,
    )
    assert processor.stats["dropped_cue_token_budget"] == 1
    assert "following" in " ".join(record.text for record in records)


def test_nested_overlaps_preserve_both_transcripts(tmp_path):
    processor, records = process(
        tmp_path,
        [
            (0, 5000, "outer speaker"),
            (1000, 3000, "inner speaker"),
            (6000, 8000, "following speaker"),
        ],
        9,
    )
    text = " ".join(record.text for record in records)
    assert all(
        text.count(value) == 1 for value in ("outer speaker", "inner speaker", "following speaker")
    )
    assert processor.stats["repaired_overlaps"] == 1


def test_folder_ignores_metadata_and_rejects_colliding_stems(tmp_path):
    wav(tmp_path / "input/sample.wav")
    subtitles(tmp_path / "input/sample.srt", [(0, 1000, "one letter 我")])
    (tmp_path / "input/readme.txt").write_text("ignore")
    (tmp_path / "input/subfolder").mkdir()
    processor = DataProcessor(
        tmp_path / "input",
        tmp_path / "input",
        output=tmp_path / "data.ljson",
        dump_dir=tmp_path / "dump",
    )
    processor.run()
    assert processor.stats["files"] == 1
    # A second supported extension with the same stem would overwrite output.
    (tmp_path / "input/sample.mp3").write_bytes(b"preflight must reject before decoding")
    with pytest.raises(ValueError, match="Duplicate output id"):
        DataProcessor(
            tmp_path / "input",
            tmp_path / "input",
            output=tmp_path / "second.ljson",
            dump_dir=tmp_path / "dump2",
        ).run()


def test_portable_metadata_and_silence_cleanup(tmp_path):
    root = tmp_path / "original"
    _, records = process(root, [], 95, keep_empty_chance=1, subsampling_factor_for_silence=2)
    assert len(records) == len(list((root / "dump").rglob("*.flac"))) == 2
    destination = tmp_path / "moved"
    root.rename(destination)
    assert all(
        Path(record.audio_path).is_file()
        for record in DataProcessor.read_records(destination / "data.ljson")
    )
    assert all(
        Path(path).is_file() for path in ljson_to_pandas(destination / "data.ljson")["audio"]
    )


def test_sentence_pipeline_rerun_preflight_happens_before_fusion(tmp_path):
    wav(tmp_path / "clip.wav", 2)
    path = sentence_tsv(tmp_path / "sentences.tsv", [("clip.wav", "A valid transcript", "speaker")])
    config = dict(
        out_folder_base=str(tmp_path / "output"),
        dataset_name="fusion",
        split_name="train",
        tsv_paths=[str(path)],
        clips_folders=[str(tmp_path)],
        partials=[1],
        vad_chance=0,
        n_jobs=1,
        min_text_words=0,
        language="en",
    )
    main(config)
    folder = tmp_path / "output/fusion/train"
    snapshot = {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        main(config)
    assert snapshot == {
        str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in folder.rglob("*")
        if path.is_file()
    }
    main({**config, "overwrite_output": True})
    assert len(list((folder / "audios").iterdir())) == 1
    assert len(list((folder / "transcripts").iterdir())) == 1
    assert len(DataProcessor.read_records(folder / "created_dataset/data.ljson")) == 1


def test_fusion_reports_input_errors(tmp_path, caplog):
    path = sentence_tsv(
        tmp_path / "sentences.tsv", [("missing.wav", "valid transcript", "speaker")]
    )
    with pytest.raises(FileNotFoundError):
        generate_fold([path], [tmp_path], [1], tmp_path / "fusion", n_jobs=1, vad_chance=0)
    assert "Failed to fuse sample" in caplog.text
    assert any(record.exc_info for record in caplog.records)


def test_normalizing_vtt_preserves_format_and_invalid_cues(tmp_path):
    path = tmp_path / "source.vtt"
    original = "WEBVTT\n\n00:00.000 --> 00:02.000\nFirst\n\n00:03.000 --> 00:01.000\nInvalid\n"
    path.write_text(original)
    output = utils.netflix_normalize_file(path)
    assert path.read_text() == original
    assert output.read_text().startswith("WEBVTT")
    assert len(DataProcessor.read_utterances_from_srt(output)) == 2


def test_netflix_counts_visible_characters():
    import pysubs2

    subs = pysubs2.SSAFile()
    subs.extend(
        [
            pysubs2.SSAEvent(start=0, end=1000, text=r"{\i1}Hello{\i0}"),
            pysubs2.SSAEvent(start=1100, end=2000, text=r"{\b1}world{\b0}"),
        ]
    )
    assert utils.fuse_until_limits(subs, max_chars=11)
    assert len(subs) == 1 and subs[0].plaintext == "Hello world"


def test_overwrite_cannot_delete_audio_referenced_by_mapping(tmp_path):
    config = pipeline_config(tmp_path)
    folder = tmp_path / "output/test/train/created_dataset"
    audio = wav(folder / "source.wav")
    original = audio.read_bytes()
    mapping(
        tmp_path / "mapping.tsv",
        [
            dict(
                id="source",
                audio_path=str(audio),
                srt_path=str(tmp_path / "source.srt"),
                language="en",
            )
        ],
    )
    with pytest.raises(ValueError, match="inside generated output"):
        main({**config, "overwrite_output": True})
    assert audio.read_bytes() == original


def test_custom_timestamp_resolution_stays_in_whisper_range(tmp_path):
    _, records = process(tmp_path, [(0, 29990, "valid long cue")], 30, timestamp_resolution=40000)
    assert records[0].text.endswith("<|30.00|>")


def test_language_model_uses_all_labels_and_full_confidence(monkeypatch):
    from whisper_prep import language

    class Model:
        def predict(self, text, k):
            assert text == "Lietuvos žmonės" and k == 1
            return ["__label__lt"], [0.84999]

    monkeypatch.setattr(language, "get_language_model", Model)
    assert language.identify_language("Lietuvos\nžmonės") == ("lt", 0.84999)


def test_language_model_checks_explicit_cached_file(tmp_path, monkeypatch):
    from whisper_prep import language

    path = tmp_path / "model.ftz"
    path.write_bytes(b"verified fixture")
    monkeypatch.setenv("WHISPER_PREP_LANGUAGE_MODEL", str(path))
    monkeypatch.setattr(language, "MODEL_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    assert language._model_path() == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        language._model_path()


def test_multiline_numeric_subtitle_text_is_not_an_index(tmp_path):
    path = tmp_path / "numeric.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nRoom\n101\n\n2\n00:00:01,000 --> 00:00:02,000\nNext\n"
    )
    assert [cue.text for cue in DataProcessor.read_utterances_from_srt(path)] == [
        "Room 101",
        "Next",
    ]


@pytest.mark.parametrize("text", ["", '""'])
def test_legacy_sentence_preparation_accepts_empty_text(text):
    from whisper_prep.dataset.convert import _prepare_dataset

    assert _prepare_dataset({"sentence": text})["sentence"] == ""
