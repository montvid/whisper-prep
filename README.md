# whisper-prep

Prepare audio and timestamped transcripts for fine-tuning Whisper. The pipeline
can fuse sentence clips into longer recordings or split existing SRT/VTT audio
into windows of at most 30 seconds. It exports portable JSON-lines metadata and
a Hugging Face `DatasetDict` with 16 kHz mono audio.

## Installation

Use Python **3.11 or 3.12** and install **FFmpeg** with your operating system's
package manager. Python 3.13+ is not supported because pydub uses `audioop`.

```bash
python -m pip install -e '.[dev]'
```

For a CPU-only Linux installation, install the matching PyTorch pair first:

```bash
python -m pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

The package constrains PyTorch/TorchAudio to 2.10 and TorchCodec to 0.10. `uv.lock`
records the development resolution; `uv sync --extra dev --locked` uses it.
The lock's default Linux PyTorch distribution includes CUDA dependencies.
Importing `whisper_prep` does not initialize models or download data. Language
filtering loads fastText on first use and may download its language model.

The language model is [fastText lid.176.ftz](https://fasttext.cc/docs/en/language-identification.html),
licensed under CC BY-SA 3.0. Its checksum is verified before loading. Offline
runs can set `WHISPER_PREP_LANGUAGE_MODEL` to a previously downloaded copy.
Otherwise it is cached under `$XDG_CACHE_HOME/whisper-prep` (default
`~/.cache/whisper-prep`); an existing `~/.cache/lid.176.ftz` is also recognized.

## Input and configuration

Choose exactly one input route. Example files are
[config_sentence_fusion.yaml](config_sentence_fusion.yaml) and
[config_srt_splitting.yaml](config_srt_splitting.yaml). All options and defaults
are listed in [the configuration reference](docs/configuration.md).

For existing subtitles:

```yaml
dataset_name: my_dataset
split_name: train
out_folder_base: ./datasets
language: en
transcripts_tsv: ./data/transcripts.tsv
min_text_words: 8
seed: 42
n_jobs: 1
```

`transcripts.tsv` has `srt_path` and `audio_path` columns, with optional `id` and
`language`. Relative paths resolve against the **TSV's directory**. IDs must be
unique after sanitization; duplicates are rejected before audio is written.
A row's language overrides the configured fallback. Both are validated against
Whisper's supported languages.

Alternatively, replace `transcripts_tsv` with `source_audio_dir` and
`source_transcript_dir`. Files are paired by stem; only supported audio files
are processed. SRT and VTT, missing numeric SRT indexes, common missing blank
separators, UTF-8/BOM and Windows-1252 text are supported. Encoding fallback is
reported in the log.

For sentence fusion:

```yaml
dataset_name: fused_sentences
split_name: train
out_folder_base: ./datasets
language: en
tsv_paths: [./data/sentences.tsv]
clips_folders: [./data/clips]
partials: [1.0]
n_samples_per_srt: 16
normalize_text: false
seed: 42
```

Sentence TSVs require `path` (or `clip_path`) and `sentence`. Audio paths are
relative to the corresponding `clips_folders` entry. Missing/blank `client_id`
uses speaker `0`. Empty text, `NA`, `null`, and leading zeros remain strings.
Standard CSV quoting with tab delimiters is the default, including escaped
quotes and embedded tabs/newlines. For deliberately unquoted TSV data, set
`tsv_quoting: none`. The three input lists must have equal lengths; omitted
`partials` defaults to all `1.0`.

```bash
whisper_prep -c config.yaml
```

Configuration paths resolve against the current working directory. Unknown
options and invalid values fail before processing. Existing generated output
fails before sentence fusion; `overwrite_output: true` replaces generated
artifacts. Keep source files separate from the output directory.
Overwrite preflight checks sentence TSV audio references as well as transcript
mappings before removing generated files.

## Processing behavior

- Dense subtitles are packed within `max_tokens_length` rather than dropping
  an entire 30-second window. A cue exceeding the token budget or 30 seconds
  is reported and its audio is excluded. Small overlaps are repaired; nested
  overlaps retain both texts in one interval.
- Final cues up to one second beyond the audio are clamped. Larger overruns,
  reversed timestamps and repeated hallucinations are reported and excluded.
  Malformed timing lines or untimed cue blocks are reported as file errors;
  they cannot silently turn a recording into an empty transcript.
- `normalize_text: true` performs conservative Unicode/whitespace cleanup.
  Language-specific expansion requires `normalizer_language: de-CH`; it is
  intended only for Swiss German transcripts.
- `netflix_normalize: true` merges eligible consecutive captions within 42
  visible characters and seven seconds. It writes separate normalized files
  and leaves source subtitles unchanged. It can merge across silence.
- `min_text_words` is a minimum, excluding timestamp tokens. Languages without
  word spaces use an alphanumeric-character count. Compression filtering uses
  UTF-8 bytes. French/English filters require a matching model prediction with
  confidence of at least 0.85.
- `filter_words` / `drop_segments_containing` remove cue audio intervals;
  `drop_text` removes text fragments. Filter reports have distinct filenames.
- `keep_empty_chance` retains a seeded fraction of untranscribed spans. Optional
  `validate_empty_with_vad` checks the actual encoded audio for speech.
- Given the same input and software versions, fixed seeds produce the same
  ordering, sampling, fusion names and metadata across worker counts.
- File failures log tracebacks and write error/summary JSON. By default any
  file failure makes the run fail; `max_error_rate` permits an explicit fraction,
  including failures during subtitle normalization.

## Output

```text
out_folder_base/dataset_name/split_name/
├── audios/                         # Fused input audio (sentence route only)
├── transcripts/                    # Generated SRTs (sentence route only)
├── normalized_transcripts/         # Optional copies and mapping TSV
├── created_dataset/
│   ├── data.ljson                  # audio_path, text, language, prompt
│   ├── data.summary.json           # File, cue and output counts
│   ├── data.errors.json            # Present when files fail
│   └── dump/<id>/<start_ms>.mp3
├── hf/                             # Saved Hugging Face DatasetDict
├── bad_examples.tsv                # Quality-filtered records, when present
├── french_examples.tsv             # Likewise english_examples.tsv
├── filtered_<word>_segments.tsv     # Dropped subtitle intervals
└── filtered_<word>_records.tsv      # Residual record-level matches
```

JSON audio paths are relative to `data.ljson`; `DataProcessor.read_records` and
`ljson_to_pandas` resolve them. The directory can be moved as a unit. Audio
removed by final filtering/subsampling is deleted, and JSON metadata matches
HF output. An entirely filtered dataset is saved with its empty typed schema.

## Downloading and publishing

Download Hub input separately, then use the resulting local TSV:

```bash
whisper_prep_download_hf --dataset owner/dataset --output-dir ./data/downloaded --split train --format auto
```

`--format srt` requires `srt`; `sentences` uses `sentence` or `text`. `auto`
chooses per source. Downloads namespace filenames by source and row to prevent
ID collisions, and refuse existing data. Use `transcripts_mapping.tsv` for SRT
input, or `hf_sentences.tsv` with the downloaded `audios` directory for fusion.

Installed helper commands also work outside a source checkout:

```bash
whisper_prep_upload_asr --tsv ./data.tsv --output-dir ./asr-dataset
whisper_prep_srt_to_hf ./series-folder ./hf-dataset --language en
whisper_prep_push_to_hub --dataset train=./hf-dataset --repo-id owner/dataset
```

`whisper_prep_upload_asr` accepts `path` plus `sentence`, `text`, or `srt_path`;
relative paths resolve against the TSV. It can publish with `--repo_id`.
Publishing requires your Hub credentials and is private by default; helper
CLIs accept `--public`. Pipeline upload requires `upload_to_hu: true` and
`hu_repo: owner/dataset`; `hu_private` defaults to true. Root scripts remain
thin compatibility wrappers for the installed commands.

## Tests and development

```bash
python -m pytest -ra                       # Excludes the network Hub group
python -m pytest -m '' -ra                 # Full suite, including Hub tests
python -m pytest -m hub tests/test_hub_datasets.py -ra
ruff check .
ruff format --check .
check-manifest
python -m build
python -m twine check dist/*
```

The 11 Hub tests use pinned revisions of
[`danavery/urbansound8K`](https://huggingface.co/datasets/danavery/urbansound8K)
and [`bilguun/musan-noise`](https://huggingface.co/datasets/bilguun/musan-noise).
The initial UrbanSound download caches one approximately 434 MB Parquet shard;
the tests select ten clips from it and five MUSAN clips. They run the real pipeline and inspect
empty text, audio metadata and VAD results. Unavailable fixtures fail explicitly
instead of silently skipping. The initial run needs public Hub access.

GitHub Actions runs local-fixture tests, lint and packaging on Python 3.11/3.12.
Its manual `hub` option enables the network group. Regression coverage and
intentional compatibility changes are recorded in [review fixes](docs/review-fixes.md).

## License and credits

[MIT](LICENSE). Original project: [i4Ds/whisper-prep](https://github.com/i4Ds/whisper-prep),
Claudio Paonessa and Vincenzo Timmel, FHNW.
