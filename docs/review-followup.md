# Follow-up review corrections

This change addresses N1–N14 in the 2026-11-16 review of commit `6795f80`.
The original review's Appendix A is not applied verbatim: it loses Unicode
combining marks and still rewrites punctuation when deleting a fragment. Its
suggested pydub read parameters also fail to resample WAV input.

| Finding | Resolution |
| --- | --- |
| N1: Unicode IDs | Preserve NFC letters, numbers and combining marks, including Indic vowel signs. Retain path sanitization and explicit duplicate rejection. |
| N2: Punctuation | Preserve punctuation spacing in ordinary cleanup and basic normalization. Fragment deletion tidies only an adjacent comma/period, leaving unrelated punctuation alone. |
| N3: Shortened cues | Cap synthetic speech overlap by both clips' speech lengths; merge overlapping captions that cannot retain 200 ms and half the earlier duration. Round half steps consistently, exclude sub-20 ms cues explicitly, and fall back to the native grid when a coarse grid erases a cue. |
| N4: Late predictable failures | Load language-filter models early; preflight every subtitle input before encoding. Preserve per-file error budgets and error reports. |
| N5: Fusion decode errors | Apply `max_error_rate` to clip decoding as a separate stage. Positive budgets collect all failures, preserve usable inputs, and write a per-clip error report. Strict zero remains fail-fast. |
| N6: Fusion accounting and unlabelled speech | Count every input outcome and caption merge. Retain empty-transcript audio only after VAD finds no speech; bare-period placeholders are excluded and counted. Persist and reconcile these counts in the LIEPA-3 runner. |
| N7: Resampling aliasing | Use FFmpeg output resampling when rates differ, preserving the fast path for already-correct audio. Tests measure suppression of out-of-band tones in WAV and FLAC. |
| N8: Subtitle compatibility | Accept compact timing arrows. Preserve whitespace-only WebVTT payload lines. Malformed timings and untimed content still produce explicit file errors. |
| N9: Reversed timestamps | Fail preflight if a reversed cue would block more than 30 seconds or intersect valid cues. Do not guess its speech interval and silently discard neighbouring transcripts. A bounded invalid final cue can still be reported without losing earlier valid cues. |
| N10: Record filters | Match literal filter terms against transcript text with timestamp tokens removed. |
| N11: Upload TSV helper | Use strict CSV parsing, preserve valid quoted multiline fields, reject malformed quotes/column counts, and expose `--tsv-quoting none` for raw TSV. |
| N12: Expensive rejected reruns | Reject existing output before iterating sentence paths. When replacement is requested, retain every source/symlink containment check. |
| N13: Coverage | Add behavioral regression tests for blocked-boundary audio exclusion and defensive invalid-window recovery. CI installs and imports the ONNX/SoXR extras. |
| N14: Prompts, text and policy | Remove partial starts from prompts; preserve mathematical comparisons and unknown literal angle-bracket text; avoid joining years/zero-prefixed IDs into Swiss numbers. Keep the documented minimum-word/compression policy. |

Regression coverage is in `tests/test_followup_review_regressions.py`, with
additional output-validation contracts in `tests/test_liepa3_runner.py`.
The rebuild validator checks every label, prompt and embedded audio header,
plus copied-file checksums and source/fusion/segmentation count reconciliation.
Passing these checks establishes those contracts; it does not measure downstream
recognition or hallucination rates, nor establish that all possible bugs are gone.

## Validation on 2026-11-16

All runtime checks ran on `kvb@172.16.6.44`. Python 3.11 and 3.12 each passed
235 tests, including all 11 public Hub integrations and the optional ONNX/SoXR
tests. Ruff, the lockfile check, package dependency checks, source-manifest
validation, wheel/sdist builds and distribution metadata validation passed.

A smoke rebuild of one source shard from each included LIEPA-3 subset produced
1,758 records. Every complete cue had positive duration, every prompt contained
only completed cues, and all FLAC headers and NAS copy checksums passed. The
`phon` shard's 7,711 exported inputs reconcile to 7,707 retained speech clips and
four VAD rejections, with no unexplained gap or caption merges. These checks
gate the full rebuild; its completion requires separate production evidence.
