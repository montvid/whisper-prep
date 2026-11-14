"""Publish explicitly selected local dataset splits."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="SPLIT=PATH")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    from datasets import DatasetDict, load_from_disk

    dataset = DatasetDict()
    for item in args.dataset:
        if "=" not in item:
            parser.error("--dataset must be SPLIT=PATH")
        split, path = item.split("=", 1)
        if split in dataset:
            parser.error(f"Duplicate split {split}")
        loaded = load_from_disk(path)
        dataset[split] = loaded[split] if isinstance(loaded, DatasetDict) else loaded
    dataset.push_to_hub(args.repo_id, private=not args.public)
