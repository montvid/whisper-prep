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


def audio_bytes(seconds):
    stream = io.BytesIO()
    sf.write(stream, np.zeros(round(seconds * 16000)), 16000, format="FLAC", subtype="PCM_16")
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
