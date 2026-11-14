#!/usr/bin/env python3
"""
Generate a Hugging Face dataset from subtitle/audio folders organized by series.

This script traverses an input directory containing subdirectories for each series (or movie),
loads each .srt file and its matching .mp3 audio file, and creates a dataset with the following fields:
  - audio: the path to the .mp3 audio file (cast to Audio feature)
  - sentence: the subtitle text without timestamps (concatenated cues)
  - series: the name of the series (subdirectory name)
  - srt: the raw .srt file content (including timestamps)
  - language: the language code (e.g., 'de')

The resulting dataset is saved to disk in Hugging Face format and can optionally be pushed to the Hub.
"""

import argparse
from pathlib import Path

from whisper_prep.subtitling.parse import read_cues, read_subtitle_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Hugging Face dataset from subtitle/audio files by series"
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Directory containing series subfolders with .srt and matching audio files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory where the Hugging Face dataset will be saved",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="de",
        help="Language code for all examples (default: de)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target audio sampling rate for the dataset (default: 16000)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Name of the dataset split (default: train)",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional Hugging Face repo identifier to push the dataset to the Hub",
    )
    parser.add_argument(
        "--public", action="store_true", help="Publish publicly; default is private"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import pandas as pd

    from whisper_prep.dataset.convert import pandas_to_hf_dataset, save_hf_dataset

    records = []
    # Traverse each series subfolder
    for series_dir in sorted(args.input_folder.iterdir()):
        if not series_dir.is_dir():
            continue
        series = series_dir.name
        for srt_path in sorted(series_dir.glob("*.srt")):
            matches = [
                path
                for path in series_dir.iterdir()
                if path.stem == srt_path.stem
                and path.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one matching audio file for {srt_path}, found {len(matches)}"
                )
            audio_path = matches[0]
            srt_text = read_subtitle_text(srt_path)
            # Concatenate subtitle cues into a single sentence without timestamps
            subs = read_cues(srt_path)
            sentence = " ".join(cue.text for cue in subs)

            records.append(
                {
                    "audio": str(audio_path.resolve()),
                    "sentence": sentence,
                    "series": series,
                    "srt": srt_text,
                    "language": args.language,
                }
            )

    # Build Hugging Face dataset
    frame = pd.DataFrame(records, columns=["audio", "sentence", "series", "srt", "language"])
    ds_dict = pandas_to_hf_dataset(frame, args.split_name, args.sampling_rate)

    # Save to disk
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_hf_dataset(ds_dict, args.output_dir)

    # Optionally push to the Hugging Face Hub
    if args.push_to_hub:
        ds_dict.push_to_hub(args.push_to_hub, private=not args.public)


if __name__ == "__main__":
    main()
