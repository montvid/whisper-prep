import json

import pytest

from tests.test_followup_review_regressions import folder_config
from tests.test_review_regressions import stamp, wav
from whisper_prep import main
from whisper_prep.subtitling.parse import read_cues


def write_track(path, cues):
    vtt = path.suffix == ".vtt"
    blocks = []
    for index, (start, end, body) in enumerate(cues, 1):
        timing = f"{stamp(start)} --> {stamp(end)}"
        blocks.append(f"{index}\n{timing.replace(',', '.') if vtt else timing}\n{body}")
    path.write_text(("WEBVTT\n\n" if vtt else "") + "\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("suffix", ["srt", "vtt"])
@pytest.mark.parametrize(
    "first,following",
    [
        ("HELLO AND WELCOME", "HELLO AND WELCOME\nTO THE EVENING NEWS"),
        ("- Kur eini?", "<i>- Kur eini?</i>\n- Namo."),
    ],
)
def test_markerless_rolling_captions_fail_before_audio_is_written(
    tmp_path, suffix, first, following
):
    source = tmp_path / "source"
    wav(source / "news.wav", 3)
    path = write_track(source / f"news.{suffix}", [(0, 1000, first), (1040, 2600, following)])
    with pytest.raises(RuntimeError, match="during preflight"):
        main(folder_config(tmp_path, source))
    (error,) = json.loads(
        (tmp_path / "output/case/train/created_dataset/data.errors.json").read_text()
    )
    assert "rolling" in error["message"] and "non-rolling" in error["message"]
    assert f"{path}:{8 if suffix == 'vtt' else 6}" in error["message"]
    assert not list((tmp_path / "output").rglob("*.flac"))


@pytest.mark.parametrize("line_break", ["\n", "<br/>", "</p><p>", r"\N"])
def test_sliding_and_accumulating_displays_are_detected_inside_an_ordinary_track(
    tmp_path, line_break
):
    displays = [
        ["Intro"],
        ["Title"],
        ["Presenter"],
        ["One", "Two", "Three"],
        ["Two", "Three", "Four"],
        ["Two", "Three", "Four", "Five"],
        ["Outro"],
    ]
    path = write_track(
        tmp_path / "news.srt",
        [(i * 1000, (i + 1) * 1000, line_break.join(lines)) for i, lines in enumerate(displays)],
    )
    with pytest.raises(ValueError, match="rolling SRT"):
        read_cues(path)


def test_disjoint_accumulating_dialogue_pairs_can_dominate_a_track(tmp_path):
    bodies = [
        "- Where?",
        "- Where?\n- Home.",
        "- Today?",
        "- Today?\n- Yes.",
        "- Alone?",
        "- Alone?\n- No.",
    ]
    path = write_track(
        tmp_path / "dialogue.vtt",
        [(i * 1000, (i + 1) * 1000, body) for i, body in enumerate(bodies)],
    )
    with pytest.raises(ValueError, match="rolling WebVTT"):
        read_cues(path)


def test_cleaned_youtube_srt_keeps_its_short_transition_as_evidence(tmp_path):
    path = write_track(
        tmp_path / "converted.srt",
        [
            (0, 1000, "First phrase"),
            (1000, 1010, "First phrase"),
            (1010, 2000, "First phrase\nNew phrase"),
            (2000, 3000, "Only once"),
        ],
    )
    with pytest.raises(ValueError, match="rolling SRT"):
        read_cues(path)


@pytest.mark.parametrize("suffix", ["srt", "vtt"])
def test_real_repetitions_and_unrelated_multiline_cues_survive(tmp_path, suffix):
    cues = [
        (0, 1000, "Again"),
        (1000, 2000, "Again"),
        (2000, 3000, "Again\nPlease"),  # One isolated echo in a longer track.
        (3000, 4000, "Again\nPlease"),  # The complete display is repeated.
        (4500, 5500, "Please\nStay here"),  # A pause before the repeated phrase.
        (5500, 6500, "Stay here with me\nTomorrow"),  # Only part of a line matches.
    ]
    path = write_track(tmp_path / f"ordinary.{suffix}", cues)
    assert [cue.text for cue in read_cues(path)] == [body.replace("\n", " ") for _, _, body in cues]


def test_fresh_word_timings_on_a_carried_line_are_evidence_of_new_speech(tmp_path):
    path = write_track(
        tmp_path / "lyrics.vtt",
        [
            (0, 1000, "Sing\nWith me"),
            (1000, 2000, "Sing\nWith<00:01.500> me\nAgain"),
        ],
    )
    assert [cue.text for cue in read_cues(path)] == ["Sing With me", "Sing With me Again"]


@pytest.mark.parametrize(
    "start,second,expected",
    [
        (1101, "Oh yes\nHere we go", "Oh yes Here we go"),
        (1000, "Oh yes\nOh yes", "Oh yes Oh yes"),
        (1000, '<span title="<br/>">Oh yes</span> here we go', "Oh yes here we go"),
    ],
)
def test_two_cues_need_a_visible_continuous_roll(tmp_path, start, second, expected):
    path = write_track(tmp_path / "song.srt", [(0, 1000, "Oh yes"), (start, 2000, second)])
    assert [cue.text for cue in read_cues(path)] == ["Oh yes", expected]
