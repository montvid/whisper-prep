"""Real inference, lossless encoding, and worker propagation contracts."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, load_from_disk

from whisper_prep import main
from whisper_prep.audio import vad
from whisper_prep.audio.io import read_audio, save_audio_segment
from whisper_prep.generation.data_processor import DataProcessor

ASSETS = Path(__file__).parent / "assets"


@pytest.mark.parametrize("backend", ["onnx", "onnx-sequence"])
@pytest.mark.parametrize(
    "fixture",
    [
        "empty_vad/empty_with_speech.mp3",
        "empty_vad/empty_without_speech.mp3",
        "empty_vad/empty_without_speech_2.mp3",
        "filter_words/filter_words_fixture.mp3",
        "punctuation_hallucination/punctuation_hallucination_slice.mp3",
    ],
)
def test_onnx_matches_torch_on_real_audio(backend, fixture):
    pytest.importorskip("onnxruntime")
    audio = vad.read_audio(str(ASSETS / fixture))
    assert vad.silero_vad_collector(audio, backend=backend) == vad.silero_vad_collector(audio)
    assert vad.silero_speech_ratio(audio, backend=backend) == vad.silero_speech_ratio(audio)


@pytest.mark.parametrize("samples", [0, 1, 511, 512, 513, 16000 * 35 + 13])
def test_sequence_handles_silence_and_partial_frame_blocks(samples):
    pytest.importorskip("onnxruntime")
    audio = np.zeros(samples, dtype=np.float32)
    assert vad.silero_vad_collector(audio, backend="onnx-sequence") == (0, 0)
    assert vad.silero_speech_ratio(audio, backend="onnx-sequence") == 0


def test_sequence_resets_between_recordings():
    pytest.importorskip("onnxruntime")
    audio = vad.read_audio(str(ASSETS / "empty_vad/empty_with_speech.mp3"))
    first = vad.silero_vad_collector(audio, backend="onnx-sequence")
    vad.silero_vad_collector(np.zeros(16000), backend="onnx-sequence")
    assert vad.silero_vad_collector(audio, backend="onnx-sequence") == first


@pytest.mark.parametrize("channels", [1, 2])
def test_flac_io_preserves_every_pcm16_sample(tmp_path, channels):
    samples = np.random.default_rng(9).integers(-32768, 32768, (4096, channels), dtype=np.int16)
    source, destination = tmp_path / "input.flac", tmp_path / "output.flac"
    sf.write(source, samples, 44100, subtype="PCM_16")
    segment = read_audio(source, normalize=False)
    save_audio_segment(segment, destination, "flac")
    actual, rate = sf.read(destination, dtype="int16", always_2d=True)
    np.testing.assert_array_equal(actual, samples)
    assert rate == 44100
    assert sf.info(destination).subtype == "PCM_16"


@pytest.mark.parametrize("route", ["sentences", "subtitles"])
@pytest.mark.parametrize("workers", [1, 2])
def test_flac_onnx_pipeline_embeds_portable_lossless_audio(tmp_path, route, workers):
    pytest.importorskip("onnxruntime")
    source = ASSETS / "empty_vad/empty_with_speech.mp3"
    text = "This is a spoken test sentence with enough words for the training example."
    config = dict(
        out_folder_base=str(tmp_path / "output"),
        dataset_name="flac",
        split_name="train",
        language="en",
        audio_format="flac",
        vad_backend="onnx-sequence",
        n_jobs=workers,
        min_text_words=0,
        overlap_chance=0,
        n_samples_per_srt=1,
    )
    if route == "sentences":
        tsv = tmp_path / "sentences.tsv"
        tsv.write_text(f"path\tsentence\n{source}\t{text}\n{source}\t{text}\n")
        config.update(tsv_paths=[str(tsv)], clips_folders=[str(tmp_path)])
    else:
        srt = tmp_path / "source.srt"
        srt.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n")
        tsv = tmp_path / "mapping.tsv"
        tsv.write_text(f"id\taudio_path\tsrt_path\na\t{source}\t{srt}\nb\t{source}\t{srt}\n")
        config.update(transcripts_tsv=str(tsv))
    main(config)
    folder = tmp_path / "output/flac/train"
    records = DataProcessor.read_records(folder / "created_dataset/data.ljson")
    assert records
    assert not list(folder.rglob("*.mp3"))
    dataset = load_from_disk(folder / "hf")["train"].cast_column("audio", Audio(decode=False))
    assert len(dataset) == len(records)
    for row in dataset:
        blob = row["audio"]["bytes"]
        assert blob.startswith(b"fLaC")
        info = sf.info(io.BytesIO(blob))
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
        assert 0 < info.duration <= 30
        assert row["language"] == "en"
    for record in records:
        path = folder / "created_dataset" / record.audio_path
        assert path.suffix == ".flac"
        assert path.is_file()


@pytest.mark.parametrize("audio_format", ["flac", "wav"])
def test_failed_file_removes_generated_lossless_segments(tmp_path, monkeypatch, audio_format):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(32000), 16000)
    (tmp_path / "clip.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nSpoken text\n")
    processor = DataProcessor(
        audio_dir=str(tmp_path),
        transcript_dir=str(tmp_path),
        output=str(tmp_path / "data.ljson"),
        dump_dir=str(tmp_path / "dump"),
        audio_format=audio_format,
    )
    original = processor._save_segment_audio

    def fail_after_saving(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("simulated failure after encoding")

    monkeypatch.setattr(processor, "_save_segment_audio", fail_after_saving)
    with pytest.raises(RuntimeError, match="Failed to process"):
        processor.run()
    assert not list((tmp_path / "dump").rglob(f"*.{audio_format}"))
    assert json.loads((tmp_path / "data.summary.json").read_text())["failed_files"] == 1


def test_missing_onnx_fails_before_overwriting_output(tmp_path, monkeypatch):
    tsv = tmp_path / "sentences.tsv"
    tsv.write_text("path\tsentence\n")
    previous = tmp_path / "output/corpus/train/audios/keep.flac"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"previous output")
    vad.get_vad_model.cache_clear()
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match="whisper-prep\\[onnx\\]"):
        main(
            dict(
                out_folder_base=str(tmp_path / "output"),
                dataset_name="corpus",
                split_name="train",
                tsv_paths=[str(tsv)],
                clips_folders=[str(tmp_path)],
                vad_backend="onnx-sequence",
                overwrite_output=True,
            )
        )
    assert previous.read_bytes() == b"previous output"


def test_unknown_vad_backend_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="vad_backend"):
        main(
            dict(
                out_folder_base=str(tmp_path / "output"),
                dataset_name="corpus",
                split_name="train",
                source_audio_dir=str(tmp_path),
                source_transcript_dir=str(tmp_path),
                vad_backend="cuda",
            )
        )
    assert not (tmp_path / "output").exists()
