"""Convert sentence/subtitle TSV metadata and optionally publish the result."""

import argparse
from pathlib import Path


def dataset_from_tsv(tsv, audio_column="path", split="train", sampling_rate=16000):
    import pandas as pd
    import pysubs2

    from whisper_prep.dataset.convert import pandas_to_hf_dataset
    from whisper_prep.subtitling.parse import read_cues
    from whisper_prep.text import resolve_path

    tsv = Path(tsv).resolve()
    frame = pd.read_csv(tsv, sep="\t", keep_default_na=False, dtype=str)
    if audio_column not in frame:
        raise ValueError(f"Missing audio column {audio_column!r}")
    frame = frame.rename(columns={audio_column: "audio"})
    frame["audio"] = frame["audio"].map(lambda value: str(resolve_path(value, tsv.parent)))
    for path in frame["audio"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if "srt_path" in frame:
        make_sentence = "sentence" not in frame and "text" not in frame
        if make_sentence:
            frame["sentence"] = ""
        frame["srt"] = ""
        for index, row in frame.iterrows():
            cues = read_cues(resolve_path(row["srt_path"], tsv.parent))
            subs = pysubs2.SSAFile()
            subs.extend(
                pysubs2.SSAEvent(start=cue.start, end=cue.end, text=cue.text) for cue in cues
            )
            frame.at[index, "srt"] = subs.to_string(format_="srt")
            if make_sentence:
                frame.at[index, "sentence"] = " ".join(cue.text for cue in cues)
        frame = frame.drop(columns="srt_path")
    elif "sentence" not in frame and "text" not in frame:
        raise ValueError("TSV requires sentence, text, or srt_path")
    return pandas_to_hf_dataset(frame, split, sampling_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--repo_id", help="Publish to owner/dataset (requires credentials)")
    parser.add_argument("--output-dir", type=Path, help="Save a local HF dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio_column", default="path")
    parser.add_argument("--sampling_rate", default=16000, type=int)
    parser.add_argument(
        "--public", action="store_true", help="Publish publicly; default is private"
    )
    args = parser.parse_args()
    if not args.repo_id and not args.output_dir:
        parser.error("Provide --output-dir or --repo_id")
    dataset = dataset_from_tsv(args.tsv, args.audio_column, args.split, args.sampling_rate)
    if args.output_dir:
        from whisper_prep.dataset.convert import save_hf_dataset

        save_hf_dataset(dataset, args.output_dir)
    if args.repo_id:
        dataset.push_to_hub(args.repo_id, private=not args.public)
