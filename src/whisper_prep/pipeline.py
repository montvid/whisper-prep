"""Validated local-input pipeline with isolated generated artifacts."""

import csv
import json
import logging
import shutil
from itertools import chain
from multiprocessing import get_context
from pathlib import Path

from whisper_prep.config import PipelineConfig
from whisper_prep.text import TAG_RE, report_name, resolve_path
from whisper_prep.utils import (
    get_compression_ratio,
    is_english,
    is_french,
    netflix_normalize_file,
    parse_args,
)

LOGGER = logging.getLogger(__name__)


def _sentence_audio_paths(config):
    from whisper_prep.dataset.convert import iter_sentence_rows

    for tsv, clips in zip(config.tsv_paths, config.clips_folders, strict=True):
        for row in iter_sentence_rows(tsv, config.tsv_quoting):
            yield Path(clips, row["path"])


def _resolve_input_sources(config, audio_dir, transcript_dir):
    if config.get("hu_datasets"):
        raise ValueError("Download datasets first with whisper_prep_download_hf")
    sentence = bool(config.get("tsv_paths") or config.get("clips_folders"))
    folders = bool(config.get("source_audio_dir") or config.get("source_transcript_dir"))
    mapping = config.get("transcripts_tsv")
    if sum((sentence, folders, bool(mapping))) != 1:
        raise ValueError("Configure exactly one input route")
    if folders:
        if not (config.get("source_audio_dir") and config.get("source_transcript_dir")):
            raise ValueError("Both source_audio_dir and source_transcript_dir are required")
        return Path(config["source_audio_dir"]), Path(config["source_transcript_dir"]), None, False
    return audio_dir, transcript_dir, mapping, sentence


def _apply_basic_text_filters(
    df, out_folder, min_text_words=8, drop_empty_text=True, language="de"
):
    clean = df["text"].fillna("").map(lambda text: TAG_RE.sub("", text).strip())
    nonempty = clean != ""
    bad = (df["text"].map(get_compression_ratio) >= 2.4) & nonempty
    if min_text_words is not None and min_text_words > 0:
        languages = df["language"] if "language" in df else [language] * len(df)
        counts = [
            sum(character.isalnum() for character in text)
            if lang in {"zh", "ja", "th", "lo", "my"}
            else len(text.split())
            for text, lang in zip(clean, languages, strict=True)
        ]
        import pandas as pd

        bad |= (pd.Series(counts, index=df.index, dtype="int64") < min_text_words) & nonempty
    if drop_empty_text:
        bad |= ~nonempty
    if bad.any():
        df[bad].to_csv(Path(out_folder, "bad_examples.tsv"), sep="\t", index=False)
    return df[~bad].copy()


def _normalize_row(task):
    row, destination, skip, drop = task
    try:
        normalized = netflix_normalize_file(row["srt_path"], skip, drop, destination)
    except Exception as exc:
        LOGGER.exception(
            "Failed normalizing subtitles %s for audio %s", row["srt_path"], row["audio_path"]
        )
        return {
            **row,
            "_normalization_error": {
                "audio_path": row["audio_path"],
                "transcript_path": row["srt_path"],
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "normalization",
            },
        }
    return {**row, "srt_path": str(normalized.resolve())}


def _normalize_inputs(processor, out_folder, config, skip, drop):
    rows = processor._input_rows()
    destination = out_folder / "normalized_transcripts"
    destination.mkdir()
    tasks = [
        (row, destination / (row["id"] + Path(row["srt_path"]).suffix.lower()), skip, drop)
        for row in rows
    ]
    jobs = config.netflix_normalize_n_jobs or config.transcripts_tsv_n_jobs or config.n_jobs
    if jobs > 1 and tasks:
        with get_context("spawn").Pool(min(jobs, len(tasks))) as pool:
            normalized = pool.map(_normalize_row, tasks)
    else:
        normalized = list(map(_normalize_row, tasks))
    mapping = destination / "transcripts.tsv"
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            delimiter="\t",
            fieldnames=["id", "audio_path", "srt_path", "language"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(normalized)
    processor.transcripts_tsv = str(mapping)
    return normalized


def main(config=None):
    if config is None:
        import yaml

        args = parse_args()
        with open(args.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    config = PipelineConfig.from_dict(config)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg must be installed before processing audio")
    out_folder = (Path(config.out_folder_base) / config.dataset_name / config.split_name).resolve()
    audio_dir, transcript_dir = out_folder / "audios", out_folder / "transcripts"
    output_dir, hf_folder = out_folder / "created_dataset", out_folder / "hf"
    output_file = output_dir / "data.ljson"
    source_audio, source_transcript, mapping, sentence_route = _resolve_input_sources(
        config.to_dict(), audio_dir, transcript_dir
    )
    generated = [output_dir, hf_folder, out_folder / "normalized_transcripts"]
    if sentence_route:
        generated += [audio_dir, transcript_dir]
    # A rejected rerun needs no corpus traversal or model initialization.
    existing = [path for path in generated if path.exists()]
    if existing and not config.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {existing[0]}; set overwrite_output=true to replace generated artifacts"
        )
    if sentence_route or config.validate_empty_with_vad:
        from whisper_prep.audio.vad import get_vad_model

        # Unlabelled sentence clips always require VAD, even if vad_chance=0.
        get_vad_model(config.vad_backend)
    if config.filter_french or config.filter_english:
        from whisper_prep.language import get_language_model

        get_language_model()
    reports = [
        path
        for pattern in (
            "bad_examples.tsv",
            "french_examples.tsv",
            "english_examples.tsv",
            "filtered_*_segments.tsv",
            "filtered_*_records.tsv",
            "fusion.*.json",
        )
        for path in out_folder.glob(pattern)
    ]
    # Never allow overwrite cleanup to remove a configured source location.
    inputs = [config.transcripts_tsv, config.source_audio_dir, config.source_transcript_dir]
    inputs += (config.tsv_paths or []) + (config.clips_folders or [])
    if config.transcripts_tsv:
        base = Path(config.transcripts_tsv).resolve().parent
        with open(config.transcripts_tsv, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t", strict=True):
                inputs.extend(
                    resolve_path(row[key], base)
                    for key in ("audio_path", "srt_path")
                    if row.get(key)
                )
    if sentence_route:
        inputs = chain(inputs, _sentence_audio_paths(config))
    artifacts = generated + reports
    for value in filter(None, inputs):
        source = Path(value).resolve()
        if any(source == artifact or artifact in source.parents for artifact in artifacts):
            raise ValueError(
                f"Input {source} is inside generated output; choose a separate output directory"
            )
    for path in existing:
        if path.is_symlink():
            raise ValueError(f"Refusing to remove symlinked output: {path}")
    if config.overwrite_output:
        for path in existing:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        for path in reports:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    from whisper_prep.dataset.convert import ljson_to_pandas, pandas_to_hf_dataset, save_hf_dataset
    from whisper_prep.generation.data_processor import DataProcessor

    if sentence_route:
        from whisper_prep.generation.generate import generate_fold_from_yaml

        generate_fold_from_yaml({**config.to_dict(), "out_folder": out_folder})
    skip = (
        config.drop_segments_containing
        if config.drop_segments_containing is not None
        else config.filter_words
    )
    processor = DataProcessor(
        audio_dir=str(source_audio),
        transcript_dir=str(source_transcript),
        transcripts_tsv=mapping,
        output=str(output_file),
        dump_dir=str(output_dir / "dump"),
        language=config.language,
        cut_initial_audio=config.cut_initial_audio,
        filter_segment_words=skip,
        drop_text=config.drop_text,
        keep_empty_chance=config.keep_empty_chance,
        subsampling_factor_for_silence=config.subsampling_factor_for_silence,
        validate_empty_with_vad=config.validate_empty_with_vad,
        empty_vad_max_speech_ratio=config.empty_vad_max_speech_ratio,
        n_jobs=config.transcripts_tsv_n_jobs or config.n_jobs,
        seed=config.seed,
        max_tokens_length=config.max_tokens_length,
        max_prompt_length=config.max_prompt_length,
        rep_threshold=config.rep_threshold,
        timestamp_resolution=config.timestamp_resolution,
        normalize_unicode=config.normalize_unicode,
        tokenizer_type=config.tokenizer_type,
        max_error_rate=config.max_error_rate,
        audio_format=config.audio_format,
        vad_backend=config.vad_backend,
    )
    input_rows = (
        _normalize_inputs(processor, out_folder, config, skip, config.drop_text)
        if config.netflix_normalize
        else None
    )
    processor.run(input_rows=input_rows)
    df = ljson_to_pandas(output_file)
    original_paths = set(df["audio"])
    df = _apply_basic_text_filters(
        df,
        out_folder,
        config.min_text_words,
        config.drop_empty_text
        if config.drop_empty_text is not None
        else config.keep_empty_chance <= 0,
        config.language,
    )
    for enabled, predicate, name in (
        (config.filter_french, is_french, "french"),
        (config.filter_english, is_english, "english"),
    ):
        if enabled:
            rejected = df["text"].map(predicate).astype(bool)
            if rejected.any():
                df[rejected].to_csv(out_folder / f"{name}_examples.tsv", sep="\t", index=False)
                df = df[~rejected]
    for word in skip or []:
        if not word:
            continue
        rejected = (
            df["text"]
            .map(lambda text: TAG_RE.sub("", text))
            .str.contains(word, case=False, regex=False, na=False)
        )
        if rejected.any():
            df[rejected].to_csv(
                out_folder / f"filtered_{report_name(word)}_records.tsv", sep="\t", index=False
            )
            df = df[~rejected]
    for path in original_paths - set(df["audio"]):
        Path(path).unlink(missing_ok=True)
    # Keep portable metadata consistent with the exported (filtered) data.
    with output_file.open("w", encoding="utf-8") as stream:
        for row in df.to_dict("records"):
            row["audio_path"] = str(Path(row.pop("audio")).relative_to(output_dir))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset = pandas_to_hf_dataset(df, config.split_name)
    save_hf_dataset(dataset, hf_folder)
    LOGGER.info("Exported %d records to %s", len(df), hf_folder)
    if config.upload_to_hu:
        dataset.push_to_hub(config.hu_repo, private=config.hu_private)
    return dataset
