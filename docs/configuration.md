# Configuration reference

`whisper_prep -c config.yaml` accepts these keys. Unknown keys are rejected.
Paths in YAML resolve against the current working directory. Paths inside a
transcript TSV resolve against that TSV's directory. Sentence TSV audio paths
resolve against the corresponding `clips_folders` entry.

## Dataset and inputs

| Key | Default | Meaning |
| --- | --- | --- |
| `out_folder_base` | required | Output parent directory. |
| `dataset_name` | required | Safe single directory name. |
| `split_name` | required | Safe split name, e.g. `train`. |
| `language` | `de` | Whisper language code/name; set explicitly for your data. Transcript TSV rows may override it. |
| `tsv_paths` | null | Sentence TSV list; requires `clips_folders`. |
| `clips_folders` | null | One audio base directory per sentence TSV. |
| `partials` | all `1.0` | One sampling fraction (0–1) per TSV, chosen using `seed`. |
| `tsv_quoting` | `csv` | Standard tab-delimited CSV quoting; `none` selects raw unquoted fields. |
| `transcripts_tsv` | null | Mapping with `audio_path`, `srt_path`, optional `id` and `language`. |
| `source_audio_dir` | null | Folder of audio paired by stem with `source_transcript_dir`. |
| `source_transcript_dir` | null | Folder of matching SRT/VTT subtitles. |

Exactly one route must be present: sentence TSVs, a transcript TSV, or the two
source folders. Download Hub sources with `whisper_prep_download_hf` first;
`hu_datasets` is not a pipeline input option.

## Fusion

These options affect the sentence route only. Probabilities must be in [0, 1].

| Key | Default | Meaning |
| --- | --- | --- |
| `n_samples_per_srt` | `16` | Positive number of input clips per fused recording. |
| `maintain_speaker_chance` | `0.5` | Probability of retaining the previous speaker when clips remain. |
| `normalize_text` | `false` | Enable conservative text cleanup. |
| `normalizer_language` | null | `basic` (also null) or explicit Swiss German `de-CH` expansion. |
| `vad_chance` | `1.0` | Probability of detecting speech boundaries for each clip. |
| `overlap_chance` | `0.5` | Probability of overlapping clips. |
| `max_overlap_chance` | `0.2` | Probability of choosing the maximum allowed overlap. |
| `max_overlap_duration` | `0.2` | Nonnegative seconds of speech overlap beyond detected silence. |
| `audio_format` | `mp3` | Format of both fused recordings and final training segments: `mp3`, `wav`, or `flac`. FLAC/WAV segments use 16-bit PCM. Applies to subtitle inputs too. |

## Segmentation, normalization and filtering

| Key | Default | Meaning |
| --- | --- | --- |
| `netflix_normalize` | `false` | Merge eligible captions to 42 visible characters / 7 seconds; write separate copies. |
| `cut_initial_audio` | `false` | Start at most one second before the first cue. |
| `keep_empty_chance` | `0.0` | Probability of retaining empty/no-speech clips and untranscribed spans. |
| `drop_empty_text` | null | Drop final empty records; null follows whether `keep_empty_chance` is zero. |
| `validate_empty_with_vad` | `false` | Check retained empty segments for speech using their encoded audio. |
| `vad_backend` | `torch` | Silero inference: `torch`, `onnx`, or `onnx-sequence`. Applies to fusion and empty-segment validation. ONNX backends require the `onnx` extra and run on CPU with one thread per worker. The sequence backend processes blocks of frames to reduce Python overhead. |
| `empty_vad_max_speech_ratio` | `0.06` | Maximum accepted speech ratio (0–1) for empty records. |
| `subsampling_factor_for_silence` | `1` | Keep every Nth empty record; delete unused generated segment audio. Original audio in the legacy timestamp-free API is preserved. Positive integer. |
| `max_tokens_length` | `219` | Packed transcription budget including timestamps; minimum 3. |
| `max_prompt_length` | `223` | Maximum previous-context tokens; zero disables prompts. |
| `rep_threshold` | `3` | Minimum repeated identical consecutive cues to reject; minimum 2. |
| `timestamp_resolution` | `20` | Timestamp rounding in ms, a positive multiple of 20. A window falls back to 20 ms if its configured grid would give a cue zero duration. Cues shorter than 20 ms are reported and excluded with their audio interval. |
| `normalize_unicode` | `false` | NFKC normalization of source subtitle text. |
| `tokenizer_type` | `multilingual` | `multilingual` or `english` Whisper tokenizer. |
| `min_text_words` | `8` | Minimum nonempty transcript length; null/zero disables. Timestamp tokens excluded. CJK/Thai/Lao/Burmese count alphanumeric characters. |
| `filter_french` | `false` | Drop French predictions meeting confidence 0.85 after lexical prefiltering. |
| `filter_english` | `false` | Likewise for English. |
| `filter_words` | null | Case-insensitive literal terms whose cue intervals are excluded. |
| `drop_segments_containing` | null | If supplied, replaces `filter_words`, including with an empty list. |
| `drop_text` | null | Case-insensitive literal text fragments to erase without excluding audio. |

Compression filtering always uses the UTF-8 byte ratio, with threshold 2.4.
Rejected cue intervals break prompt continuity and cannot become empty-text
training audio. Unusable cues and VAD rejections are counted separately.

## Execution and publishing

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | `42` | Integer seed for partial sampling, fusion and empty-segment retention. |
| `n_jobs` | `1` | Positive worker count. |
| `transcripts_tsv_n_jobs` | null | Override segmentation workers for both mapping and folder input. |
| `netflix_normalize_n_jobs` | null | Override normalization workers; otherwise uses segmentation worker setting. |
| `max_error_rate` | `0.0` | Allowed fraction of failed input files (0–1), including subtitle normalization and file preflight failures. Also applies independently to failed sentence-clip decodes in fusion. Zero stops on the first fusion decode error; a positive value collects decode failures and enforces the budget after fusion. Schema errors and model failures remain fatal. |
| `overwrite_output` | `false` | Replace generated artifacts on a rerun. Existing output otherwise fails before fusion. |
| `upload_to_hu` | `false` | Publish the resulting dataset to the Hub. |
| `hu_repo` | null | Required `owner/dataset` when uploading. |
| `hu_private` | `true` | Keep uploaded dataset private unless explicitly disabled. |

Changing worker counts preserves ordering and seeded choices for the same input
and library versions. MP3 bitstreams need not be identical across FFmpeg or
TorchCodec versions. Worker failures include traceback logs, source paths,
exception type, `data.errors.json` and `data.summary.json` totals.

Malformed subtitle timing lines and untimed cue blocks are file errors, not
empty transcripts. Valid empty SRTs and VTT headers/metadata remain supported.
Convert rolling WebVTT captions to non-rolling cues before preparation. Preflight
rejects a repeated display line when adjacent cues also contain a short transition
or word timings on the new line. Separate identified WebVTT cues with an empty
line so the identifier cannot become training text.
Language-model initialization, checksum and prediction errors stop filtering
instead of silently accepting the affected records.

Subtitle file existence, syntax, and ambiguous reversed timestamps are checked
before any segmentation audio is encoded. Predictable failures exceeding the
file budget write the usual error/summary reports and stop before processing
valid files. Enabled language filters load their model before fusion or output
replacement. An existing output is rejected before traversing sentence paths.

Fusion writes `fusion.summary.json` with mutually exclusive counts for failed
decodes, skipped empty text, bare-period placeholders, VAD-rejected speech text,
unlabelled speech, retained labelled clips, and retained silence. It also counts
merged captions and written cues. Allowed decode failures are listed in
`fusion.errors.json`. Unlabelled clips always require VAD to be retained, even
with `vad_chance: 0`; detected speech without a transcript is always excluded.

Synthetic speech overlap is limited to 30% of each neighbouring speech interval.
Subtitle overlap repair merges captions when trimming would leave less than
200 ms or half the earlier interval. Only completed cues enter previous-context
prompts; a partial start timestamp remains a target-only token.
