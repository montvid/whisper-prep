#!/usr/bin/env python3
"""Compatibility wrapper for whisper_prep_upload_asr."""

from whisper_prep.upload_asr_dataset import dataset_from_tsv as dataset_from_tsv
from whisper_prep.upload_asr_dataset import main

if __name__ == "__main__":
    main()
