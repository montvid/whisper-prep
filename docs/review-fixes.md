# Review remediation record

This change addresses the confirmed defects from the repository review and
remote reproduction report. Regression tests use real audio processing where
it matters; network datasets are a separate, explicitly selected test group.

| Findings | Change | Regression coverage |
| --- | --- | --- |
| C1 | Repair small overlaps and merge nested captions; advance safely past invalid windows. | `test_overlap_does_not_discard_following_windows`, `test_nested_overlaps_preserve_both_transcripts` |
| C2 | Pack cues within the transcription token budget; preserve dense multilingual transcripts. Oversized individual cues are reported and their audio excluded. | `test_dense_lithuanian_is_split_to_token_budget`, `test_exact_budget_and_prompt_limit`, `test_oversized_token_cue_is_reported_and_following_cue_survives` |
| C3 | Sanitize output IDs and reject duplicates before segment writes; namespace download filenames by source/row. | `test_duplicate_ids_rejected_before_writing_audio`, `test_folder_ignores_metadata_and_rejects_colliding_stems`, `test_downloader_mixed_text_schemas_and_id_collisions` |
| C4 | Validate input-list lengths, preserve quoted TSV fields and string values, default missing speakers, seed partial selection. | `test_tsv_preserves_quotes_missing_value_words_and_speakers`, `test_raw_tsv_dialect_is_explicit_and_subsampling_seeded` |
| C5 | Use conservative normalization by default; make Swiss expansion explicit; preserve case, dates, identifiers and slash expressions; bound currency replacements. | `test_multilingual_normalization_preserves_spoken_content`, `test_explicit_swiss_normalization_does_not_corrupt_identifiers`, `test_text_normalizer.py` |
| H1 | Check output before fusion; deterministic filenames; explicit replacement of generated artifacts without stale directories. | `test_sentence_pipeline_rerun_preflight_happens_before_fusion`, `test_output_preflight_config_copy_and_overwrite`, `test_overwrite_cannot_delete_audio_referenced_by_mapping` |
| H2 | Local seeded RNGs per file/batch, sorted input, ordered parent writes, seeded pandas sampling. | `test_fusion_seed_independent_of_worker_and_caller_rng`, `test_parallel_order_rng_and_empty_mapping` |
| H3 | Obtain integer VAD sample indices before converting to seconds. | `test_vad_uses_sample_timestamps` |
| H4 | Report reversed final cues, honor repetition threshold, clamp small final overruns, exclude unusable speech spans, sanitize report names. | `test_bad_final_cue_is_reported_without_losing_good_cues`, `test_small_audio_overrun_clamped_and_no_false_silence`, `test_repetition_threshold_and_filter_filename`, `test_unusable_cue_blocks_speech_and_resets_prompt` |
| H5 | Share a timing-line-based SRT/VTT reader; remove metadata blocks, support encoding fallback, preserve source files and output format. Count visible subtitle characters. | `test_subtitle_parser_layouts`, `test_subtitle_encoding_and_normalization_preserve_source`, `test_normalizing_vtt_preserves_format_and_invalid_cues`, `test_netflix_counts_visible_characters`, `test_netflix_normalize.py` |
| H6 | Honor model confidence and full language label set; compute compression using bytes; exclude timestamps from minimum-length counting; handle languages without spaces. Keep interval and final-record reports separate. | `test_language_probability_compression_and_word_counts`, `test_basic_text_filters.py`, `test_data_processor_filter_words.py` |
| H7 | Log source paths and tracebacks, write error/summary JSON, fail by default on file errors, propagate fusion errors. | `test_missing_audio_fails_with_traceback_and_totals`, `test_fusion_reports_input_errors` |
| M1 | Replace repeated whole-dataframe scans with active-speaker queues. | `test_active_speaker_sampler_visits_every_row` |
| M2 | Reuse NumPy waveform storage, reuse decoded fusion PCM for VAD, limit worker Torch threads, spawn workers without inheriting threaded model state. | Parallel determinism tests and real pipeline/Hub tests. |
| M3 | Lazy package entrypoints, language identification and Silero model loading; remove script import side effects. | `test_package_and_helpers_import_without_models_or_downloads` |
| M4 | Validate a typed configuration before effects, require CLI `-c`, reject unknown options, preserve caller configuration, expose/document processing controls. | `test_configuration_errors_before_side_effects`, `test_package_and_helpers_import_without_models_or_downloads`, `test_output_preflight_config_copy_and_overwrite` |
| M5 | Discover normalizer tests, correct stale audio-length assertions, isolate test output, replace unavailable Hub fixtures, add CI and narrow fixture ignores. | Complete pytest collection; 11 real Hub integration checks; Python 3.11/3.12 CI configuration. |
| M6 | Declare direct dependencies, compatible audio-library versions and supported Python versions; replace Python-incompatible fastlid with lazy direct fastText loading; remove duplicate fasttext distribution and runtime pytest; refresh lock and source manifest. | Remote installation/dependency checks, source/wheel builds, manifest and metadata checks. |
| L1 | Fix row-wise SRT assignment and generated sentence columns; make helpers import-safe installed CLIs; support matching non-MP3 audio; expose publishing privacy. | `test_upload_helper_keeps_each_rows_subtitles`, import/CLI smoke checks. |
| L2 | Remove unused fusion state and in-place module execution; fix empty legacy sentence handling and remove remote-code flag. Retain optional legacy APIs for compatibility. | `test_package_and_helpers_import_without_models_or_downloads`; normalizer unit tests. |
| L3 | Share row processing, worker setup, subtitle parsing, text cleanup and dataset conversion; move downloader into dataset package. | Folder, TSV, downloader and parser tests use these shared paths. |
| L4 | Bound prompts, handle header-only mappings, remove shard debris, generate valid comma-separated SRT times, validate row languages, resolve TSV-relative paths, remove orphan audio, keep one-character cues, use portable JSON paths and lint. | `test_exact_budget_and_prompt_limit`, `test_parallel_order_rng_and_empty_mapping`, `test_generated_srt_overlap_never_reverses_cues`, `test_tsv_paths_ids_and_languages_are_validated`, `test_portable_metadata_and_silence_cleanup`, parser tests and Ruff. |
| Additional remote findings | Preserve text from heterogeneous Hub schemas, build compatible Audio storage, and save/reload an entirely filtered dataset with its schema. | `test_downloader_mixed_text_schemas_and_id_collisions`, `test_upload_helper_keeps_each_rows_subtitles`, `test_empty_filtered_export_and_orphan_cleanup` |

Most named regressions are in `tests/test_review_regressions.py`.

## Compatibility and retained choices

- Standard quoted TSV remains the default. The suggested blanket `QUOTE_NONE`
  change would break valid quoted fields; raw TSV is an explicit alternative.
- Empty-segment VAD still checks the actual encoded MP3. A pre-encoding-only
  optimization changed decisions on a real MUSAN clip, so retaining that
  check is necessary for the existing empty-label contract. Fusion avoids
  its separate redundant decode, and array-to-tensor copies are removed.
- The optional legacy WebRTC detector and conversion helpers remain callable.
  Unused APIs and design preferences alone are not confirmed bugs.
- Review L5's codec, caption-gap, short-text and speaker-sampling policies remain
  configuration/design choices. This change does not claim to resolve them.
- Subtitle normalization now returns a new path instead of modifying its input.
  Invalid files raise errors. JSON paths are relative to their metadata file;
  readers resolve them. Config typos and duplicate output IDs now fail early.
- Publishing is private by default and requires an explicit repository. No
  dataset publication or GitHub push is part of local verification.

## Verification

Run the commands in the README. The full remote test run includes the 11 Hub
checks; the default local-fixture selection excludes them by marker, not skip.
Build/manifest checks cover the distribution's tests, audio fixtures, example
configs, helper wrappers, documentation and workflow. GitHub Actions is added
as configuration; its hosted run requires a subsequent push.
