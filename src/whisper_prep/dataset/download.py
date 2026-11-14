"""Download heterogeneous Hub sources without losing text or overwriting audio."""

import csv
from dataclasses import dataclass
from pathlib import Path

from whisper_prep.text import sanitize_example_id


@dataclass
class DownloadedDatasetPaths:
    sentence_tsvs: list[str]
    transcripts_tsv: str | None = None


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as load

    return load(*args, **kwargs)


def save_hu_dataset_locally(config, audio_dir, transcript_dir):
    import numpy as np
    import soundfile as sf
    from tqdm.auto import tqdm

    audio_dir, transcript_dir = Path(audio_dir).resolve(), Path(transcript_dir).resolve()
    out_folder = Path(config["out_folder"]).resolve()
    split = config.get("hu_input_split", config.get("split_name", "train"))
    mode = config.get("hf_input_format", "auto")
    if mode not in {"auto", "srt", "sentences"}:
        raise ValueError("hf_input_format must be auto, srt or sentences")
    names = config["hu_datasets"]
    if not names:
        raise ValueError("At least one Hugging Face dataset is required")
    for directory in (audio_dir, transcript_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Download destination is not empty: {directory}")
    for filename in ("hf_sentences.tsv", "transcripts_mapping.tsv"):
        if (out_folder / filename).exists():
            raise FileExistsError(out_folder / filename)
    sources = [(name, load_dataset(name, split=split)) for name in names]
    ids_file = config.get("hu_episode_ids_file")
    allowed_ids = set(Path(ids_file).read_text().splitlines()) if ids_file else None
    # Validate each schema before any files are written; never intersect away
    # sentence/text columns belonging to different source datasets.
    for name, dataset in sources:
        columns = set(dataset.column_names)
        if "audio" not in columns:
            raise ValueError(f"{name} has no audio column")
        if allowed_ids is not None and "id" not in columns:
            raise ValueError(f"{name} has no id column for the episode filter")
        if mode == "srt" and "srt" not in columns:
            raise ValueError(f"{name} has no srt column")
        if (mode == "sentences" or "srt" not in columns) and not columns.intersection(
            {"sentence", "text"}
        ):
            raise ValueError(f"{name} requires a sentence or text column")
    for directory in (out_folder, audio_dir, transcript_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sentences, subtitles = [], []
    for source_index, (name, dataset) in enumerate(sources):
        for index, example in enumerate(tqdm(dataset, desc=f"Downloading {name}")):
            if allowed_ids is not None and str(example.get("id")) not in allowed_ids:
                continue
            identifier = f"{source_index}-{index}-{sanitize_example_id(example.get('id', index))}"
            srt = example.get("srt")
            use_srt = mode != "sentences" and srt is not None
            text = example.get("sentence")
            if text is None:
                text = example.get("text")
            if mode == "srt" and srt is None:
                raise ValueError(f"Missing SRT for {name} row {index}")
            if not use_srt and text is None:
                raise ValueError(f"Missing sentence/text for {name} row {index}")
            audio = example["audio"]
            if hasattr(audio, "get_all_samples"):
                samples = audio.get_all_samples()
                array = samples.data.detach().cpu().numpy().T
                sample_rate = samples.sample_rate
            else:
                array, sample_rate = np.asarray(audio["array"]), audio["sampling_rate"]
            if array.dtype not in (np.float32, np.float64, np.int16, np.int32):
                array = array.astype(np.float32)
            path = audio_dir / f"{identifier}.wav"
            sf.write(path, array, int(sample_rate), format="WAV")
            if use_srt:
                transcript = transcript_dir / f"{identifier}.srt"
                transcript.write_text(srt, encoding="utf-8")
                subtitles.append(
                    {
                        "id": identifier,
                        "audio_path": str(path),
                        "srt_path": str(transcript),
                        "language": example.get("language") or config.get("language", ""),
                    }
                )
            else:
                speaker = example.get("client_id") or example.get("speaker_id") or "0"
                sentences.append({"path": path.name, "sentence": text, "client_id": str(speaker)})
    sentence_paths, transcript_path = [], None
    for rows, filename, fields in (
        (sentences, "hf_sentences.tsv", ["path", "sentence", "client_id"]),
        (subtitles, "transcripts_mapping.tsv", ["srt_path", "audio_path", "language", "id"]),
    ):
        if rows:
            path = out_folder / filename
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            if rows is sentences:
                sentence_paths.append(str(path))
            else:
                transcript_path = str(path)
    return DownloadedDatasetPaths(sentence_paths, transcript_path)
