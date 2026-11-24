import importlib.util
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

spec = importlib.util.spec_from_file_location(
    "liepa3_runner", Path(__file__).parents[1] / "tools/prepare_liepa3.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_previous_evaluation_shards_are_preserved():
    read = list(range(1350))
    assert runner.held_out(read, "read") == list(range(1338, 1350))
    assert runner.held_out(list(range(136)), "phon") == [
        5,
        16,
        27,
        38,
        49,
        60,
        71,
        82,
        93,
        104,
        115,
        126,
    ]
    assert runner.held_out(list(range(2264)), "spon") == [
        94,
        282,
        470,
        658,
        846,
        1034,
        1222,
        1410,
        1598,
        1786,
        1974,
        2162,
    ]
    assert runner.SUBSETS == ("read", "spon", "phon")


def test_incomplete_corpus_cannot_silently_change_evaluation_split():
    with pytest.raises(ValueError, match="more than 12 shards"):
        runner.held_out(list(range(12)), "read")


def audio_bytes(seconds, *, rate=16000, channels=1, subtype="PCM_16"):
    stream = io.BytesIO()
    sf.write(
        stream, np.zeros((round(seconds * rate), channels)), rate, format="FLAC", subtype=subtype
    )
    return stream.getvalue()


def test_export_counts_filtering_and_keeps_transcript_and_speaker(tmp_path):
    pytest.importorskip("soxr")
    path = tmp_path / "train-00000.parquet"
    text = 'Lietuviškas tekstas su "kabutėmis"\nir kita eilute.'
    table = pa.Table.from_pylist(
        [
            {"audio": {"bytes": audio_bytes(0.5)}, "text": text, "speaker_id": "007"},
            {"audio": {"bytes": audio_bytes(0.1)}, "text": "Too short", "speaker_id": "007"},
            {"audio": {"bytes": audio_bytes(26)}, "text": "Too long", "speaker_id": "008"},
            {"audio": {"bytes": b"unused"}, "text": " ", "speaker_id": "008"},
        ]
    )
    pq.write_table(table, path)
    clips = tmp_path / "clips"
    clips.mkdir()
    rows, stats, seconds = runner.export_shard((path, clips, 0.4, 25))
    assert stats == {
        "source_rows": 4,
        "exported_clips": 1,
        "too_short": 1,
        "too_long": 1,
        "empty_text": 1,
    }
    assert seconds == 0.5
    assert rows == [("train-00000/000000.flac", text, "007")]
    info = sf.info(clips / rows[0][0])
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")


def test_corrupt_source_audio_stops_export(tmp_path):
    pytest.importorskip("soxr")
    path = tmp_path / "train-00001.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"audio": {"bytes": b"broken"}, "text": "Real transcript", "speaker_id": "007"}]
        ),
        path,
    )
    clips = tmp_path / "clips"
    clips.mkdir()
    with pytest.raises(sf.LibsndfileError):
        runner.export_shard((path, clips, 0.4, 25))
    assert not list(clips.rglob("*.flac"))


@pytest.mark.parametrize(
    "text,prompt",
    [
        ("<|1.00|> word<|1.00|>", False),
        ("<|1.00|> word<|2.00|><|3.00|>", True),
        ("<|0.01|> word<|1.00|>", False),
        ("<|0.00|> word<|31.00|>", False),
        ("<|1.00|> word<|3.00|><|2.00|> next<|4.00|>", False),
    ],
)
def test_export_validation_rejects_invalid_labels(text, prompt):
    with pytest.raises(ValueError):
        runner.validate_labels(text, prompt=prompt)


def test_export_validation_distinguishes_partial_targets_from_completed_prompts():
    target = "<|0.00|> word<|1.00|><|29.00|>"
    assert runner.validate_labels(target)["cues"] == 1
    assert runner.validate_labels("", prompt=True) == {}
    # Prompt context from earlier windows may restart its relative clock.
    assert (
        runner.validate_labels("<|29.00|> previous<|30.00|><|0.00|> next<|1.00|>", prompt=True)[
            "cues"
        ]
        == 2
    )


def test_export_checks_every_flac_header_and_sliced_array_offsets():
    good = {"bytes": audio_bytes(1), "path": None}
    bad = {"bytes": b"not FLAC", "path": "external.wav"}
    values = pa.array([good, good, bad])
    runner.validate_flac_headers(values.slice(1, 1))
    with pytest.raises(ValueError, match="embedded FLAC"):
        runner.validate_flac_headers(values.slice(1, 2))


@pytest.mark.parametrize("settings", [{"rate": 8000}, {"channels": 2}, {"subtype": "PCM_24"}])
def test_export_rejects_valid_flac_with_the_wrong_audio_format(settings):
    values = pa.array([{"bytes": audio_bytes(1, **settings)}])
    with pytest.raises(ValueError, match="mono 16 kHz PCM16 FLAC"):
        runner.validate_flac_headers(values)


def test_batch_counts_account_for_skips_and_merges_before_acceptance():
    counts = dict(
        input_clips=10,
        failed_clips=0,
        skipped_empty_text=1,
        skipped_placeholder_text=1,
        skipped_no_speech=2,
        skipped_unlabelled_speech=1,
        kept_speech_clips=5,
        kept_empty_clips=0,
        merged_captions=1,
        output_cues=4,
    )
    runner.validate_counts({"exported_clips": 10}, counts, {"input_cues": 4})
    with pytest.raises(RuntimeError, match="do not reconcile"):
        runner.validate_counts({"exported_clips": 9}, counts, {"input_cues": 4})
    for corrupted in (
        {**counts, "skipped_no_speech": 1},
        {**counts, "merged_captions": 0},
        {**counts, "output_cues": 3},
    ):
        with pytest.raises(RuntimeError, match="do not reconcile"):
            runner.validate_counts({"exported_clips": 10}, corrupted, {"input_cues": 4})
