"""Regression tests for the follow-up reviews."""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.test_review_regressions import mapping, process, sentence_tsv, subtitles, wav
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


@pytest.mark.parametrize("punctuation", ["?", "!", ":", ";"])
def test_glued_fragment_preserves_the_original_punctuation_boundary(punctuation):
    fragment = "[Musik]"
    assert drop_text_fragments(f"Was ist das [MUSIK]{punctuation}", [fragment]) == (
        f"Was ist das{punctuation}"
    )
    spaced = f"Ça va [Musik] {punctuation} Et toi ?"
    assert drop_text_fragments(spaced, [fragment]) == f"Ça va {punctuation} Et toi ?"


def test_converted_subtitle_html_preserves_word_boundaries_and_literal_text(tmp_path):
    path = subtitles(
        tmp_path / "converted.srt",
        [
            (0, 1000, "Line one<BR/>line two<br>three"),
            (
                1000,
                2000,
                '<div><span style="color:#ff0" title="x > y">Ge<em>lb</em></span></div><p>und <s>weg</s></p>',
            ),
            (2000, 3000, "<strong>5 < 6</strong> und 7 > 3, <smile> &lt;span&gt;"),
        ],
    )
    assert [cue.text for cue in read_cues(path)] == [
        "Line one line two three",
        "Gelb und weg",
        "5 < 6 und 7 > 3, <smile> <span>",
    ]


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


def test_subtick_cues_are_reported_and_a_full_tick_survives(tmp_path):
    processor, records = process(
        tmp_path,
        [(100, 101, "One"), (200, 219, "Nineteen"), (300, 320, "Twenty")],
        1,
        audio_format="flac",
    )
    assert processor.stats["dropped_cue_timestamp_resolution"] == 2
    assert [(r["start_ms"], r["end_ms"]) for r in processor.filtered_segment_records] == [
        (100, 101),
        (200, 219),
    ]
    assert len(records) == 1 and "Twenty" in records[0].text
    assert "One" not in records[0].text and "Nineteen" not in records[0].text
    assert int(Path(records[0].audio_path).stem) == 219
    start, end = map(float, TIMES.findall(records[0].text))
    assert end - start == pytest.approx(0.02)


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


def test_missing_mapped_audio_fails_preflight_before_encoding_other_files(tmp_path):
    wav(tmp_path / "good.wav", 1)
    subtitles(tmp_path / "words.srt", [(0, 1000, "Spoken words")])
    tsv = mapping(
        tmp_path / "inputs.tsv",
        [
            {"id": name, "audio_path": f"{name}.wav", "srt_path": "words.srt", "language": "en"}
            for name in ("good", "missing")
        ],
    )
    processor = DataProcessor(
        "",
        "",
        transcripts_tsv=tsv,
        output=tmp_path / "data.ljson",
        dump_dir=tmp_path / "dump",
        audio_format="flac",
    )
    with pytest.raises(RuntimeError, match="1/2 files during preflight"):
        processor.run()
    (error,) = json.loads((tmp_path / "data.errors.json").read_text())
    assert error["stage"] == "preflight" and error["type"] == "FileNotFoundError"
    assert Path(error["message"]).name == "missing.wav"
    assert processor.stats["preflight_skipped_files"] == 1
    assert not list((tmp_path / "dump").rglob("*.flac"))


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
            {"path": speech, "sentence": "."},
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
    assert counts["input_clips"] == 4 and counts["skipped_unlabelled_speech"] == 1
    assert counts["skipped_placeholder_text"] == 1
    assert counts["kept_speech_clips"] == counts["kept_empty_clips"] == 1
    assert sf.info(audios / "00000000.flac").duration == 2
    assert [cue.text for cue in read_cues(transcripts / "00000000.srt")] == ["Spoken text"]


def test_webvtt_space_only_payload_line_is_not_a_separator(tmp_path):
    path = tmp_path / "space.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n \nSpoken words\n\n00:00:03.000 --> 00:00:04.000\nOther words\n"
    )
    assert [cue.text for cue in read_cues(path)] == ["Spoken words", "Other words"]


@pytest.mark.parametrize("bridge,word_timing", [(True, True), (True, False), (False, True)])
def test_rolling_webvtt_is_rejected_in_preflight(tmp_path, bridge, word_timing):
    source = tmp_path / "source"
    wav(source / "roll.wav", 5)
    transition = "00:02.230 --> 00:02.240\nhello and welcome\n \n\n" if bridge else ""
    start = "00:02.240" if bridge else "00:02.230"
    previous = "hello<00:00:00.320><c> and welcome</c>" if word_timing else "hello and welcome"
    current = "to<00:00:03.000><c> the channel</c>" if word_timing else "to the channel"
    path = source / "roll.vtt"
    contents = (
        f"WEBVTT\n\nNOTE generated captions\nin English\n\n00:00.080 --> 00:02.230\n \n{previous}\n\n{transition}"
        f"{start} --> 00:04.470 align:start position:0%\nhello and welcome\n{current}\n\n"
        "00:04.470 --> 00:04.480\nto the channel\n \n"
    )
    path.write_text(contents)
    with pytest.raises(RuntimeError, match="during preflight"):
        main(folder_config(tmp_path, source))
    (error,) = json.loads(
        (tmp_path / "output/case/train/created_dataset/data.errors.json").read_text()
    )
    line = contents[: contents.index(f"{start} --> 00:04.470")].count("\n") + 1
    assert "rolling WebVTT" in error["message"] and f"{path}:{line}" in error["message"]
    assert not list((tmp_path / "output").rglob("*.flac"))


def test_ordinary_repetition_and_word_timed_webvtt_keep_their_spoken_text(tmp_path):
    path = tmp_path / "dialogue.vtt"
    path.write_text(
        "WEBVTT\n\n00:00.000 --> 00:01.000\nAgain\n\n"
        "00:01.000 --> 00:02.000\nAgain\n \nPlease\n\n"
        "00:02.000 --> 00:03.000\n<00:02.000>Please\nSay<00:02.500> it\n\n"
        "00:05.000 --> 00:06.000\nSay it\nOnce<00:05.500> more\n"
    )
    assert [cue.text for cue in read_cues(path)] == [
        "Again",
        "Again Please",
        "Please Say it",
        "Say it Once more",
    ]


def test_whitespace_only_webvtt_separator_cannot_leak_a_cue_identifier(tmp_path):
    path = tmp_path / "ids.vtt"
    path.write_text(
        "WEBVTT\n\nfirst-cue\n00:00.000 --> 00:01.000\nHello\n \nsecond-cue\n"
        "00:01.000 --> 00:02.000\nGoodbye\n"
    )
    with pytest.raises(ValueError, match="empty line"):
        read_cues(path)


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


def test_reversed_filtered_cue_audio_is_excluded_from_the_saved_flacs(tmp_path):
    processor, records = process(
        tmp_path,
        [(1000, 2000, "Good one"), (20500, 20000, "FILTERME words"), (24000, 25000, "Good two")],
        60,
        filter_segment_words=["FILTERME"],
        audio_format="flac",
    )
    (dropped,) = processor.filtered_segment_records
    assert (dropped["matched_word"], dropped["start_ms"], dropped["end_ms"]) == (
        "FILTERME",
        20500,
        20000,
    )
    spans = []
    for record in records:
        start = int(Path(record.audio_path).stem)
        info = sf.info(record.audio_path)
        spans.append((start, start + info.frames * 1000 // info.samplerate))
    assert spans == [(0, 20000), (20500, 50500)]
    assert "Good one" in records[0].text and "Good two" in records[1].text


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


def test_distinct_recording_names_preserve_unicode_and_literal_underscores(tmp_path):
    source = tmp_path / "source"
    names = ("pokalbis_ą", "pokalbis_č", "कला", "कली", "rec_01", "rec__01")
    for name in names:
        wav(source / (name + ".wav"), 1)
        subtitles(source / (name + ".srt"), [(0, 1000, name)])
    dataset = main(folder_config(tmp_path, source))
    assert len(dataset["train"]) == len(names)
    assert {p.parent.name for p in (tmp_path / "output").rglob("*.flac")} == set(names)


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
