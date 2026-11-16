"""Observable contracts for N1–N14 in the follow-up review."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.test_review_regressions import process, sentence_tsv, subtitles, wav
from whisper_prep import main
from whisper_prep.audio.io import read_audio
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import _generate, generate_fold
from whisper_prep.generation.text_normalizer import normalize_text
from whisper_prep.subtitling.parse import read_cues
from whisper_prep.subtitling.srt import Caption, generate_srt
from whisper_prep.text import clean_subtitle_markup, drop_text_fragments, sanitize_example_id
from whisper_prep.upload_asr_dataset import dataset_from_tsv

TIMES = re.compile(r"<\|(\d+\.\d+)\|>")


@pytest.mark.parametrize(
    "names",
    [("pokalbis_ą", "pokalbis_č"), ("запись_а", "запись_б"), ("कला", "कली"), ("音声", "声音")],
)
def test_unicode_ids_keep_distinguishing_letters_and_marks(names):
    assert [sanitize_example_id(name) for name in names] == list(names)
    assert sanitize_example_id("a\u0328") == "ą"
    assert "/" not in sanitize_example_id("../../" + names[0])


@pytest.mark.parametrize(
    "text", ["Comment ça va ?", "Il dit : « Non ! »", "Score 3 : 2 pour nous", "Wait ... what"]
)
def test_punctuation_spacing_survives_normalization_and_fragment_removal(text):
    assert drop_text_fragments(text) == text
    assert drop_text_fragments("[noise] " + text, ["[noise]"]) == text
    assert normalize_text(text) == text


def test_fragment_cleanup_only_changes_adjacent_comma_space():
    assert (
        drop_text_fragments("Hallo [Musik] , comment ça va ?", ["[Musik]"])
        == "Hallo, comment ça va ?"
    )


@pytest.mark.parametrize("text", ["5 < 6 und 7 > 3", "Use <placeholder> in this sentence"])
def test_literal_angle_brackets_are_transcript_content(text):
    assert clean_subtitle_markup(text) == text
    assert clean_subtitle_markup("<v Anna><i>" + text + "</i></v>") == text


@pytest.mark.parametrize("text", ["Im Jahr 2021 300 Leute", "Tel. 044 123 45 67"])
def test_swiss_normalizer_does_not_join_years_or_zero_prefixed_identifiers(text):
    assert normalize_text(text, language="de-CH") == text


@pytest.mark.parametrize("suffix", ["wav", "flac"])
@pytest.mark.parametrize("rate,frequency", [(48000, 10000), (32000, 9000), (44100, 12000)])
def test_downsampling_removes_out_of_band_tones(tmp_path, suffix, rate, frequency):
    for tone in (1000, frequency):
        source = tmp_path / f"tone-{tone}.{suffix}"
        audio = 0.2 * np.sin(2 * np.pi * tone * np.arange(rate) / rate)
        sf.write(source, audio, rate, subtype="PCM_16")
        segment = read_audio(source, resample_rate=16000, normalize=False)
        assert segment.frame_rate == 16000 and len(segment) == 1000
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float64) / 32768
        rms = np.sqrt(np.mean(samples[1600:-1600] ** 2))
        assert (0.13 < rms < 0.15) if tone == 1000 else rms < 0.0014


def test_short_overlapping_subtitles_merge_without_zero_duration_labels(tmp_path):
    processor, records = process(
        tmp_path, [(10000, 10400, "Taip."), (10005, 12000, "Ne, palauk.")], 15
    )
    assert processor.stats["repaired_overlaps"] == 1
    text = " ".join(record.text for record in records)
    assert "Taip. Ne, palauk." in text
    times = [float(t) for t in TIMES.findall(text)]
    assert all(end > start for start, end in zip(times[::2], times[1::2]))


def test_generated_short_overlaps_keep_both_transcripts(tmp_path):
    path = tmp_path / "short.srt"
    merged = generate_srt([Caption(10, 10.4, "Taip."), Caption(10.005, 12, "Ne.")], path)
    cues = read_cues(path)
    assert merged == 1 and len(cues) == 1
    assert cues[0].text == "Taip. Ne." and (cues[0].start, cues[0].end) == (10000, 12000)


def test_short_fused_speech_is_not_squeezed_by_maximum_overlap(tmp_path):
    clips = []
    for i in range(8):
        path = tmp_path / f"clip-{i}.wav"
        sf.write(path, np.ones(6400) * 0.1, 16000)
        clips.append({"path": path, "sentence": f"word{i}"})
    audios, transcripts = tmp_path / "audio", tmp_path / "subtitles"
    audios.mkdir()
    transcripts.mkdir()
    assert _generate(clips, audios, transcripts, 1, 1, 0.5, 0, 0, audio_format="flac")
    cues = read_cues(transcripts / "00000000.srt")
    assert len(cues) == 8
    assert all(cue.end - cue.start >= 280 for cue in cues)
    assert [cue.text for cue in cues] == [f"word{i}" for i in range(8)]


def test_rounding_half_steps_does_not_erase_a_full_timestamp_tick(tmp_path):
    _, records = process(tmp_path, [(30, 50, "Brief"), (100, 200, "Following")], 1)
    times = [float(t) for t in TIMES.findall(records[0].text)]
    assert all(end > start for start, end in zip(times[::2], times[1::2]))


def folder_config(root, source, **options):
    return dict(
        out_folder_base=str(root / "output"),
        dataset_name="case",
        split_name="train",
        source_audio_dir=str(source),
        source_transcript_dir=str(source),
        language="en",
        audio_format="flac",
        min_text_words=0,
        **options,
    )


@pytest.mark.parametrize("broken", ["missing", "malformed"])
def test_subtitle_preflight_fails_before_encoding_any_good_file(tmp_path, broken):
    source = tmp_path / "source"
    wav(source / "a.wav", 3)
    wav(source / "b.wav", 3)
    subtitles(source / "a.srt", [(0, 2000, "Good speech")])
    if broken == "malformed":
        (source / "b.srt").write_text("Bad subtitle timing\n")
    with pytest.raises(RuntimeError, match="1/2 files during preflight"):
        main(folder_config(tmp_path, source))
    assert not list((tmp_path / "output").rglob("*.flac"))
    summary = json.loads(
        (tmp_path / "output/case/train/created_dataset/data.summary.json").read_text()
    )
    assert summary["failed_files"] == 1 and summary["preflight_skipped_files"] == 1


def test_language_model_is_checked_before_audio_processing(tmp_path, monkeypatch):
    from whisper_prep import language

    source = tmp_path / "source"
    wav(source / "a.wav", 3)
    subtitles(source / "a.srt", [(0, 2000, "Good speech")])

    def unavailable():
        raise FileNotFoundError("unavailable model")

    monkeypatch.setattr(language, "get_language_model", unavailable)
    with pytest.raises(FileNotFoundError, match="unavailable model"):
        main(folder_config(tmp_path, source, filter_french=True))
    assert not list((tmp_path / "output").rglob("*.flac"))


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("allowed", [0.2, 0.4])
def test_fusion_error_budget_reports_failed_clips(tmp_path, workers, allowed):
    for i in range(2):
        wav(tmp_path / f"{i}.wav", 1)
    (tmp_path / "bad.wav").write_bytes(b"broken")
    tsv = sentence_tsv(
        tmp_path / "sentences.tsv",
        [(f"{i}.wav", f"Sentence {i}", "speaker") for i in (0, 1, "bad")],
    )

    def run():
        return generate_fold(
            [tsv],
            [tmp_path],
            [1],
            tmp_path / "fusion",
            n_jobs=workers,
            vad_chance=0,
            audio_format="flac",
            max_error_rate=allowed,
        )

    if allowed < 1 / 3:
        with pytest.raises(RuntimeError, match="1/3 fusion clips"):
            run()
    else:
        run()
    counts = json.loads((tmp_path / "fusion/fusion.summary.json").read_text())
    assert counts["input_clips"] == 3 and counts["failed_clips"] == 1
    assert counts["kept_speech_clips"] == 2
    errors = json.loads((tmp_path / "fusion/fusion.errors.json").read_text())
    assert len(errors) == 1 and Path(errors[0]["path"]).name == "bad.wav"


@pytest.mark.parametrize("vad_chance", [0, 1])
def test_unlabelled_speech_is_never_kept_as_empty_audio(tmp_path, monkeypatch, vad_chance):
    from whisper_prep.generation import generate

    monkeypatch.setattr(
        generate,
        "silero_vad_collector",
        lambda waveform, **kwargs: (0, len(waveform) / 16000) if np.any(waveform) else (0, 0),
    )
    speech = wav(tmp_path / "speech.wav", 1)
    silence = tmp_path / "silence.wav"
    sf.write(silence, np.zeros(16000), 16000)
    audios, transcripts = tmp_path / "audio", tmp_path / "subtitles"
    audios.mkdir()
    transcripts.mkdir()
    counts = Counter()
    _generate(
        [
            {"path": speech, "sentence": "Spoken text"},
            {"path": speech, "sentence": ""},
            {"path": silence, "sentence": ""},
        ],
        audios,
        transcripts,
        0,
        0,
        0,
        vad_chance,
        1,
        audio_format="flac",
        stats=counts,
    )
    assert counts["input_clips"] == 3 and counts["skipped_unlabelled_speech"] == 1
    assert counts["kept_speech_clips"] == counts["kept_empty_clips"] == 1
    assert sf.info(audios / "00000000.flac").duration == 2
    assert [cue.text for cue in read_cues(transcripts / "00000000.srt")] == ["Spoken text"]


def test_webvtt_space_only_payload_line_is_not_a_separator(tmp_path):
    path = tmp_path / "space.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n \nSpoken words\n\n00:00:03.000 --> 00:00:04.000\nOther words\n"
    )
    assert [cue.text for cue in read_cues(path)] == ["Spoken words", "Other words"]


def test_subtitle_timing_arrow_accepts_missing_spaces(tmp_path):
    path = tmp_path / "arrow.srt"
    path.write_text("1\n00:00:01,000-->00:00:02,000\nSpoken words\n")
    assert [cue.text for cue in read_cues(path)] == ["Spoken words"]


@pytest.mark.parametrize("bad_start", [15000, 50000])
def test_ambiguous_reversed_cue_is_a_file_error_before_audio_is_written(tmp_path, bad_start):
    with pytest.raises(RuntimeError, match="during preflight"):
        process(
            tmp_path,
            [
                (bad_start, 10000, "Reversed"),
                (12000, 13000, "Valid first"),
                (90000, 91000, "Valid last"),
            ],
            120,
        )
    summary = json.loads((tmp_path / "data.summary.json").read_text())
    errors = json.loads((tmp_path / "data.errors.json").read_text())
    assert summary["failed_files"] == 1 and summary["output_records"] == 0
    assert "Ambiguous reversed" in errors[0]["message"]
    assert not list((tmp_path / "dump").rglob("*.mp3"))


@pytest.mark.parametrize("word", ["0.0", "2.5", "|>"])
def test_record_filters_match_transcript_not_timestamp_tokens(tmp_path, word):
    source = tmp_path / "source"
    wav(source / "a.wav", 3)
    subtitles(source / "a.srt", [(0, 2500, "Useful spoken words")])
    assert len(main(folder_config(tmp_path, source, filter_words=[word]))["train"]) == 1


def test_upload_tsv_rejects_malformed_quotes_and_preserves_valid_multiline(tmp_path):
    audio = wav(tmp_path / "a.wav", 1)
    tsv = tmp_path / "sentences.tsv"
    tsv.write_text(f'path\tsentence\n{audio}\t"first\nsecond"trailing\n')
    with pytest.raises(ValueError, match="Invalid quoted TSV"):
        dataset_from_tsv(tsv)
    tsv.write_text(f'path\tsentence\n{audio}\t"first\nsecond"\n')
    dataset = dataset_from_tsv(tsv)
    assert list(dataset["train"]["sentence"]) == ["first\nsecond"]


def test_existing_output_is_rejected_without_scanning_sentence_paths(tmp_path, monkeypatch):
    from whisper_prep import pipeline

    (tmp_path / "output/case/train/audios").mkdir(parents=True)
    (tmp_path / "unused.tsv").write_text("path\tsentence\n" + "unused.wav\tSpeech\n" * 1000)

    def unexpected(*args):
        raise AssertionError("Existing-output preflight must not traverse the corpus")

    monkeypatch.setattr(pipeline, "_sentence_audio_paths", unexpected)
    with pytest.raises(FileExistsError):
        main(
            dict(
                out_folder_base=str(tmp_path / "output"),
                dataset_name="case",
                split_name="train",
                tsv_paths=[str(tmp_path / "unused.tsv")],
                clips_folders=[str(tmp_path)],
                vad_chance=0,
            )
        )


def test_overlapping_filtered_cue_excludes_the_whole_affected_speech(tmp_path):
    processor, records = process(
        tmp_path,
        [
            (0, 1000, "Before"),
            (2000, 5000, "Crossing speech"),
            (3000, 4000, "FILTER"),
            (6000, 7000, "After"),
        ],
        8,
        filter_segment_words=["FILTER"],
        keep_empty_chance=1,
    )
    assert processor.stats["dropped_blocked_boundary"] == 1
    for record in records:
        start = int(Path(record.audio_path).stem)
        end = start + sf.info(record.audio_path).duration * 1000
        assert end <= 2001 or start >= 5000


def test_defensive_invalid_window_resumes_at_partial_cue_and_clears_prompt(tmp_path, monkeypatch):
    original = DataProcessor._is_valid_utterances

    def invalid_first(self, cues, start):
        return start != 0 and original(self, cues, start)

    monkeypatch.setattr(DataProcessor, "_is_valid_utterances", invalid_first)
    processor, records = process(tmp_path, [(0, 1000, "First"), (29000, 31000, "Following")], 35)
    assert processor.stats["invalid_windows"] == 1
    following = next(record for record in records if "Following" in record.text)
    assert following.prompt == "" and Path(following.audio_path).stem == "29000"


def test_partial_segment_timestamp_is_target_only_not_previous_prompt(tmp_path):
    _, records = process(tmp_path, [(0, 1000, "First"), (29000, 31000, "Following")], 35)
    assert records[0].text.endswith("<|29.00|>")
    following = next(record for record in records if "Following" in record.text)
    assert following.prompt == "<|0.00|> First<|1.00|>"
    assert all(len(TIMES.findall(record.prompt)) % 2 == 0 for record in records)


def test_coarse_grid_falls_back_to_native_ticks_instead_of_erasing_speech(tmp_path):
    processor, records = process(
        tmp_path, [(1000, 2000, "Short"), (3000, 4000, "Next")], 5, timestamp_resolution=40000
    )
    assert processor.stats["fine_timestamp_windows"] == 1
    assert records[0].text == "<|1.00|> Short<|2.00|><|3.00|> Next<|4.00|>"


def test_unicode_named_recordings_produce_distinct_usable_output(tmp_path):
    source = tmp_path / "source"
    for name in ("pokalbis_ą", "pokalbis_č", "कला", "कली"):
        wav(source / (name + ".wav"), 1)
        subtitles(source / (name + ".srt"), [(0, 1000, name)])
    dataset = main(folder_config(tmp_path, source))
    assert len(dataset["train"]) == 4
    assert len(list((tmp_path / "output").rglob("*.flac"))) == 4


def test_fusion_collects_multiple_decode_failures_in_one_run(tmp_path):
    audio = wav(tmp_path / "good.wav", 1)
    tsv = sentence_tsv(
        tmp_path / "sentences.tsv",
        [
            (audio.name, "Words", "s"),
            ("missing1.wav", "Words", "s"),
            ("missing2.wav", "Words", "s"),
        ],
    )
    counts = generate_fold(
        [tsv],
        [tmp_path],
        [1],
        tmp_path / "fusion",
        n_jobs=1,
        vad_chance=0,
        audio_format="flac",
        max_error_rate=0.7,
    )
    assert counts["failed_clips"] == 2 and counts["kept_speech_clips"] == 1
    errors = json.loads((tmp_path / "fusion/fusion.errors.json").read_text())
    assert {Path(error["path"]).name for error in errors} == {"missing1.wav", "missing2.wav"}
