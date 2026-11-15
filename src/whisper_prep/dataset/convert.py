import csv
import json
import os
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Features, Value, load_dataset
from pydub import AudioSegment
from tqdm import tqdm


def ljson_to_dataframe(json_path: Union[str, Path]) -> pd.DataFrame:
    data = []

    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data, columns=["audio_path", "text", "language", "prompt"])


def ljson_to_pandas(json_path: Union[str, Path]) -> pd.DataFrame:
    train_meta_file = ljson_to_dataframe(json_path)
    train_meta_file["audio_path"] = train_meta_file["audio_path"].map(
        lambda value: str((Path(json_path).resolve().parent / value).resolve())
    )
    train_meta_file.rename(columns={"audio_path": "audio"}, inplace=True)

    return train_meta_file


def pandas_to_hf_dataset(
    train_meta_file: pd.DataFrame, split_name: str = "train", sampling_rate: int = 16000
):
    train_meta_file = train_meta_file.copy()
    # Datasets>=4 with recent pyarrow can fail casting audio from large_string.
    # Build the expected Audio storage struct explicitly.
    train_meta_file["audio"] = train_meta_file["audio"].apply(
        lambda p: {"path": str(p), "bytes": None}
    )
    if train_meta_file.empty:
        features = Features(
            {
                name: Audio(sampling_rate=sampling_rate) if name == "audio" else Value("string")
                for name in train_meta_file.columns
            }
        )
        dataset_train = Dataset.from_dict({name: [] for name in features}, features=features)
    else:
        dataset_train = Dataset.from_pandas(train_meta_file, preserve_index=False)

    dataset = DatasetDict()
    dataset[split_name] = dataset_train

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    return dataset


def save_hf_dataset(dataset, path):
    # datasets 4.x estimates Audio storage with a division by row count.
    # An explicit single empty shard bypasses that estimator and preserves
    # the typed schema for load_from_disk, without adding a dummy example.
    dataset.save_to_disk(
        str(path), num_shards={name: 1 for name, split in dataset.items() if not len(split)}
    )


def iter_sentence_rows(dataset_tsv_path, tsv_quoting="csv"):
    """Share TSV validation without buffering the whole corpus during preflight."""
    if tsv_quoting not in {"csv", "none"}:
        raise ValueError("tsv_quoting must be 'csv' or 'none'")
    with open(dataset_tsv_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(
            stream,
            delimiter="\t",
            strict=True,
            quoting=csv.QUOTE_MINIMAL if tsv_quoting == "csv" else csv.QUOTE_NONE,
        )
        try:
            columns = reader.fieldnames or []
            if not {"path", "clip_path"}.intersection(columns) or "sentence" not in columns:
                raise ValueError(
                    f"{dataset_tsv_path} requires path (or clip_path) and sentence columns"
                )
            path_column = "path" if "path" in columns else "clip_path"
            for row in reader:
                if None in row or row.get(path_column) is None or row.get("sentence") is None:
                    raise ValueError(f"Inconsistent TSV column count in {dataset_tsv_path}")
                yield {
                    "path": row[path_column],
                    "sentence": row["sentence"],
                    "client_id": (row.get("client_id") or "").strip() or "0",
                }
        except csv.Error as exc:
            raise ValueError(
                f"Invalid quoted TSV {dataset_tsv_path}: {exc}. "
                "For raw unquoted TSV, set tsv_quoting='none'."
            ) from exc


def combine_tsvs_to_dataframe(
    tsv_paths: list[Union[str, Path]],
    clips_folders: list[Union[str, Path]],
    partials: list[float],
    seed: int = 42,
    tsv_quoting: str = "csv",
) -> pd.DataFrame:
    combined_dataset = []
    if not (len(tsv_paths) == len(clips_folders) == len(partials)):
        raise ValueError("tsv_paths, clips_folders and partials must have equal lengths")
    if tsv_quoting not in {"csv", "none"}:
        raise ValueError("tsv_quoting must be 'csv' or 'none'")

    for dataset_tsv_path, clips_folder, partial in zip(
        tsv_paths, clips_folders, partials, strict=True
    ):
        if not 0 <= partial <= 1:
            raise ValueError(f"partials values must be between 0 and 1, got {partial}")
        data = pd.DataFrame(
            iter_sentence_rows(dataset_tsv_path, tsv_quoting),
            columns=["path", "sentence", "client_id"],
        )

        if partial < 1.0:
            data = data.sample(frac=partial, random_state=seed)

        for row in tqdm(pd.DataFrame.itertuples(data), total=len(data)):
            audio_file_path = Path(clips_folder, row.path).resolve()
            combined_dataset.append((row.sentence, audio_file_path, row.client_id))

    return pd.DataFrame(data=combined_dataset, columns=["sentence", "audio_file_path", "client_id"])


def _prepare_dataset(batch):
    """Function to preprocess the dataset with the .map method. Recommended by HU"""
    transcription = batch["sentence"]
    if not transcription:
        return batch

    if transcription.startswith('"') and transcription.endswith('"'):
        # we can remove trailing quotation marks as they do not affect the transcription
        transcription = transcription[1:-1]

    if transcription and transcription[-1] not in [".", "?", "!"]:
        # append a full-stop to sentences that do not end in punctuation
        transcription = transcription + "."

    batch["sentence"] = transcription

    return batch


def _process_record(example, base_path, clips_path):
    """Process each record to convert audio files to MP3 and update the path."""
    audio_path = example["path"]
    audio_id = os.path.splitext(os.path.basename(audio_path))[0]
    mp3_path = os.path.join(clips_path, f"{audio_id}.mp3")

    # Convert and save audio as MP3
    audio = AudioSegment.from_file(os.path.join(base_path, audio_path))
    audio.export(mp3_path, format="mp3")

    # Update the path in the record to the new MP3 filename
    example["path"] = mp3_path
    return example


def hf_dataset_to_tsv(
    dataset_name: str, language: str, split: str, base_path: Union[str, Path]
) -> None:
    base_path = Path(base_path, language)
    os.makedirs(base_path, exist_ok=True)
    clips_path = Path(base_path, "clips")
    os.makedirs(clips_path, exist_ok=True)

    dataset = load_dataset(dataset_name, language, split=split)
    dataset = dataset.map(_prepare_dataset, num_proc=os.cpu_count())

    dataset = dataset.map(
        lambda batch: _process_record(batch, base_path, clips_path),
        num_proc=os.cpu_count(),
        desc=f"Processing {split} records",
        remove_columns="audio",
    )

    # Convert the dataset to a pandas DataFrame
    df = pd.DataFrame.from_dict(dataset)

    # Adjust the path to store only the basename (filename) instead of the full path
    df["path"] = df["path"].apply(os.path.basename)

    # Save to TSV
    df.to_csv(os.path.join(base_path, f"{split}.tsv"), sep="\t", index=False)
