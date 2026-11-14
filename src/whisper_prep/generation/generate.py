"""Deterministic sentence fusion with bounded per-speaker queues."""

import logging
import random
from collections import deque
from functools import partial
from inspect import signature
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm

from whisper_prep.audio.io import read_audio, save_audio_segment
from whisper_prep.audio.vad import silero_vad_collector
from whisper_prep.dataset.convert import combine_tsvs_to_dataframe
from whisper_prep.generation.text_normalizer import normalize_text as normalize_text_
from whisper_prep.subtitling.srt import Caption, generate_srt

LOGGER = logging.getLogger(__name__)


def _init_fusion_worker():
    import torch

    torch.set_num_threads(1)


def _execute_parallel_process(func, args, n_jobs: int) -> list:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if n_jobs == 1:
        return [func(arg) for arg in args]
    with get_context("spawn").Pool(n_jobs, initializer=_init_fusion_worker) as executor:
        return list(tqdm(executor.imap(func, args), desc="Fusing sentences"))


def _ordered_samples(data, maintain_speaker_chance, rng):
    groups = {}
    for row in data.itertuples(index=False):
        groups.setdefault(row.client_id, deque()).append(row)
    active = list(groups)
    positions = {speaker: index for index, speaker in enumerate(active)}
    previous = None
    while active:
        if previous in positions and rng.random() < maintain_speaker_chance:
            speaker = previous
        else:
            speaker = rng.choice(active)
        row = groups[speaker].popleft()
        if not groups[speaker]:
            position = positions.pop(speaker)
            last = active.pop()
            if position < len(active):
                active[position] = last
                positions[last] = position
        previous = speaker
        yield row


def _generate_wrapper(sample, *, seed=42, **kwargs):
    index, clips = sample
    try:
        return _generate(clips, seed=seed + index, file_name=f"{index:08d}", **kwargs)
    except Exception:
        LOGGER.exception("Failed to fuse sample %s (%d input clips)", index, len(clips))
        raise


def _generate(
    constructed_samples,
    audios_folder,
    transcripts_folder,
    overlap_chance,
    max_overlap_chance,
    max_overlap_duration,
    vad_chance,
    keep_empty_chance,
    audio_format="mp3",
    seed=42,
    file_name="00000000",
):
    rng = random.Random(seed)
    offset = 0.0
    combined_audio = AudioSegment.empty()
    captions = []
    previous_space_after_speech = 0.0
    for segment in constructed_samples:
        sentence = (segment["sentence"] or "").strip()
        audio_segment = (
            read_audio(segment["path"], resample_rate=16000).set_channels(1).set_sample_width(2)
        )
        duration = audio_segment.duration_seconds
        if rng.random() < vad_chance:
            # Use the decoded PCM already in memory, rather than decoding twice.
            waveform = np.asarray(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768
            start, end = silero_vad_collector(waveform)
            found_speech = end is not None and end > start
        else:
            start, end, found_speech = 0.0, duration, True
        if not sentence or not found_speech:
            if rng.random() >= keep_empty_chance:
                continue
            sentence, start, end = "", 0.0, duration
        overlap = 0.0
        if len(combined_audio) and rng.random() < overlap_chance:
            limit = min(
                previous_space_after_speech + start + max_overlap_duration,
                combined_audio.duration_seconds,
                duration,
            )
            overlap = limit if rng.random() < max_overlap_chance else rng.uniform(0, limit)
            overlap_ms = round(overlap * 1000)
            overlap = overlap_ms / 1000
            combined_audio = (
                combined_audio.overlay(
                    audio_segment[:overlap_ms], position=len(combined_audio) - overlap_ms
                )
                + audio_segment[overlap_ms:]
            )
        else:
            combined_audio += audio_segment
        if sentence:
            captions.append(Caption(start - overlap + offset, end - overlap + offset, sentence))
        offset += duration - overlap
        previous_space_after_speech = duration - end
    if not len(combined_audio):
        return False
    audio_path = Path(audios_folder, f"{file_name}.{audio_format}")
    transcript_path = Path(transcripts_folder, f"{file_name}.srt")
    if audio_path.exists() or transcript_path.exists():
        raise FileExistsError(f"Refusing to overwrite fused sample {file_name}")
    save_audio_segment(combined_audio, audio_path, format=audio_format)
    generate_srt(captions, transcript_path)
    return True


def generate_fold_from_yaml(config: dict):
    from dataclasses import fields

    from whisper_prep.config import PipelineConfig

    unknown = set(config) - {field.name for field in fields(PipelineConfig)} - {"out_folder"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    parameters = signature(generate_fold).parameters
    return generate_fold(**{key: value for key, value in config.items() if key in parameters})


def generate_fold(
    tsv_paths,
    clips_folders,
    partials,
    out_folder,
    maintain_speaker_chance=0.5,
    n_samples_per_srt=16,
    normalize_text=False,
    overlap_chance=0.5,
    max_overlap_chance=0.2,
    max_overlap_duration=0.2,
    vad_chance=1.0,
    keep_empty_chance=0.0,
    audio_format="mp3",
    n_jobs=4,
    seed=42,
    tsv_quoting="csv",
    normalizer_language=None,
):
    for name, value in {
        "maintain_speaker_chance": maintain_speaker_chance,
        "overlap_chance": overlap_chance,
        "max_overlap_chance": max_overlap_chance,
        "vad_chance": vad_chance,
        "keep_empty_chance": keep_empty_chance,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if n_samples_per_srt < 1 or n_jobs < 1 or max_overlap_duration < 0:
        raise ValueError(
            "n_samples_per_srt and n_jobs must be positive; max_overlap_duration must be nonnegative"
        )
    if audio_format not in {"mp3", "wav", "flac"}:
        raise ValueError("audio_format must be mp3, wav or flac")
    audios_folder, transcripts_folder = Path(out_folder, "audios"), Path(out_folder, "transcripts")
    for folder in (audios_folder, transcripts_folder):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Fusion output is not empty: {folder}")
    data = combine_tsvs_to_dataframe(tsv_paths, clips_folders, partials, seed, tsv_quoting)
    data = data.sample(frac=1, random_state=seed)
    for folder in (audios_folder, transcripts_folder):
        folder.mkdir(parents=True, exist_ok=True)

    def batches():
        sequence = []
        index = 0
        for row in _ordered_samples(data, maintain_speaker_chance, random.Random(seed)):
            text = (
                normalize_text_(row.sentence, language=normalizer_language)
                if normalize_text
                else row.sentence
            )
            sequence.append(
                {"path": row.audio_file_path, "sentence": text, "speaker_id": row.client_id}
            )
            if len(sequence) == n_samples_per_srt:
                yield index, sequence
                index += 1
                sequence = []
        if sequence:
            yield index, sequence

    worker = partial(
        _generate_wrapper,
        seed=seed,
        audios_folder=audios_folder,
        transcripts_folder=transcripts_folder,
        overlap_chance=overlap_chance,
        max_overlap_chance=max_overlap_chance,
        max_overlap_duration=max_overlap_duration,
        vad_chance=vad_chance,
        keep_empty_chance=keep_empty_chance,
        audio_format=audio_format,
    )
    results = _execute_parallel_process(worker, batches(), n_jobs)
    LOGGER.info(
        "Fusion: %d inputs, %d files created, %d empty batches",
        len(data),
        sum(results),
        results.count(False),
    )
