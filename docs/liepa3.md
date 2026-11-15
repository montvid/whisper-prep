# Rebuilding local LIEPA-3 data

`tools/prepare_liepa3.py` stages local Parquet shards, exports sentence clips,
and runs the fixed sentence-fusion pipeline in bounded batches. Install the
CPU PyTorch pair as described in the README, then `pip install -e '.[liepa3]'`.

```bash
python tools/prepare_liepa3.py \
  --source /home/kvb/vietinis/liepa3 \
  --output /home/kvb/vietinis_rw/liepa3_prep \
  --work /home/kvb/liepa3-reprep-20261115/work \
  --git-commit "$(git rev-parse HEAD)" \
  --shards-per-batch 16 --workers 24 --export-workers 12
```

The output must be empty or contain this runner's matching manifest. The
runner never deletes an existing dataset automatically. It excludes `dial`,
reserves the last 12 `read` shards, and reserves 12 evenly spaced shards from
each of `spon` and `phon`, matching the previous run's evaluation split.

Source clips outside 0.4–25 seconds or with empty text are counted and excluded,
matching the previous export policy. Audio decoding failures stop the run.
Clips are resampled with SoXR to 16 kHz, normalized during sentence fusion,
and saved as mono FLAC PCM_16. Fused recordings, final training windows and
embedded Hugging Face audio all use FLAC. VAD uses Silero's CPU ONNX sequence
backend; Lithuanian text uses conservative `basic` normalization.

Each batch is processed on local disk, validated, copied to a hidden temporary
NAS directory, verified by SHA-256, and renamed into `full/<batch>/` only when
complete. Local scratch files are then removed. Finished batches contain:

- `train/hf/`: the portable Hugging Face training dataset with embedded audio.
- `config.json` and `source_shards.json`: configuration and input provenance.
- `export_summary.json`, `prep_summary.json` and filtering reports.
- `complete.json`: row count, file size, metadata digest and completion receipt.

The root `run_manifest.json` records the code revision, package versions and
exact training/evaluation shard lists. `status.json` records the current stage
and totals, including failures. Repeating the same command validates completed
batches and resumes incomplete ones. Do not run two copies against the same
output. Use `--smoke --shards-per-batch 1` with separate work/output directories
to exercise one source shard from each included subset first.

Rebuilding is not byte-for-byte compatible with the historical output: fixed
normalization, timestamp handling, token packing, filtering and seeded fusion
can change rows and labels. Changing batch size also changes fusion groupings.
