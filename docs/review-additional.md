# Additional whisper-prep review

Reviewed local `main` at `1bf290f759e70f4372f658cbcef443a2e9ca6b98`.
This records seven findings beyond the earlier [review fixes](review-fixes.md).
All seven are fixed and verified; the descriptions below preserve the original
failure conditions, and the remediation table records the resulting behavior.
The review covered the main source modules and existing tests; it was not an
exhaustive execution of every input, configuration, dependency or failure path.
Another broad scan is not a prerequisite for fixing these confirmed defects.

All reproductions ran on `kvb@172.16.6.44`, using Python 3.12.14, PyTorch and
TorchAudio 2.10.0 CPU, TorchCodec 0.10.0, datasets 4.8.5, and FFmpeg 7.1.5.
The original 114 tests passed, including all 11 Hub tests. Ten additional
contract checks failed, covering the seven findings below; one control passed.
Deletion probes used disposable fixtures, not user recordings.

## Findings and required behavior

### R1 — High: silence subsampling deletes original audio

In `generation/data_processor.py`, `_subsample_silence()` unconditionally
unlinks discarded recordings. With `with_timestamps=False`, records reference
original audio rather than generated segments. Three source WAVs and a
subsampling factor of two reproduced deletion of the second original WAV.

Only generated segments may be deleted. The legacy sentence-only route must
retain its source recordings even when their metadata records are discarded.

### R2 — High: overwrite protection misses sentence TSV references

In `pipeline.py`, overwrite preflight checks audio referenced by transcript
TSVs, but not audio referenced by sentence TSVs. If a sentence points into a
generated directory, `overwrite_output=True` deletes that source and then
fails with `FileNotFoundError`.

Resolve sentence paths with the same TSV dialect and path-column rules as
fusion, and reject source/output overlap before cleanup. Preserve the input
even when its path is relative or reached through a symlink.

### R3 — High: malformed subtitles can label speech as silence

In `subtitling/parse.py`, unmatched timing lines are ignored. A file containing
one valid and one malformed cue silently loses the malformed cue. An invalid
VTT containing real speech exported an empty label with `keep_empty_chance=1`;
the summary reported zero input cues and no file error.

Reject malformed timing lines and untimed cue blocks with a useful error.
Keep valid empty subtitles, VTT metadata, literal arrows in cue text, and the
existing supported SRT/VTT layouts working.

### R4 — Medium: multiline sentence text is silently truncated

In `subtitling/srt.py`, generated captions preserve blank lines from a quoted
sentence TSV field. The parser treats these as cue boundaries. Input
`First spoken sentence.\n\nSecond spoken sentence.` exported only the first
sentence under the default `normalize_text=False` setting.

Collapse caption whitespace when writing generated SRTs, independently of
optional linguistic normalization, so all spoken text survives the round trip.

### R5 — Medium: language-model corruption silently disables filtering

In `utils.py`, the English and French predicates catch `ValueError`, including
the checksum failure raised while loading fastText. A corrupt model file
therefore produced a `False` filter decision rather than an error.

Propagate model initialization and prediction failures. A successful prediction
with no labels may still return the explicit unknown-language result.

### R6 — Medium: normalization bypasses the file-error allowance

In `pipeline.py`, normalization exceptions escape before the processor can
account for failed files. With one missing subtitle among two inputs and
`max_error_rate=0.5`, processing succeeded without normalization but failed
when `netflix_normalize=True`.

Count normalization failures in the same denominator and error budget as
segmentation failures. Preserve source paths, traceback logging, error JSON
and summary JSON for both serial and parallel execution.

### R7 — Medium: sentence fusion rejects valid Opus files

In `audio/io.py`, the extension is forced as an FFmpeg input format. A valid
Ogg Opus file failed with `Unknown input format: 'opus'`, although automatic
format detection decoded the same file successfully.

Let the decoder detect the input container instead of treating an extension
as a demuxer name.

## Evidence and remediation

Original reproduction scripts, logs, versions and commands are retained outside
the repository in `../prep-review2-validation/`. The remote review workspace is
`/tmp/whisper-prep-review2.8F5dz9`.

The fixes have permanent coverage in
[`tests/test_additional_review_regressions.py`](../tests/test_additional_review_regressions.py).

| Finding | Implemented change | Regression coverage |
| --- | --- | --- |
| R1 | Delete discarded audio only in the route that generates segment files. | Original WAVs remain byte-identical, including when stored under the configured dump directory; earlier tests still check generated-audio cleanup. |
| R2 | Share a streaming sentence TSV parser between preflight and fusion; check resolved row paths and report files before cleanup. | Relative, absolute and symlinked audio paths; both path column names and TSV dialects; source metadata named like a report. |
| R3 | Reject malformed timing lines and untimed cue blocks instead of treating them as silence. | SRT/VTT errors, missing separators, invalid arrows, real speech, valid empty subtitles, metadata, and literal arrows following ordinary clock text. |
| R4 | Collapse whitespace when serializing generated captions. | LF, CRLF and Unicode paragraph separators retain all spoken words through the full pipeline. |
| R5 | Propagate model loading and prediction exceptions from language predicates. | Corrupt caches raise for English and French; successful predictions without labels remain unknown. |
| R6 | Carry normalization errors into the processor's ordinary per-file result accounting. | Serial and parallel runs, allowed and exceeded error budgets, original source paths in error JSON, and an entirely failed input exported as empty when explicitly allowed. |
| R7 | Use automatic audio container detection. | A real Opus file decodes and completes sentence fusion and dataset export. |

Final verification ran on the same remote host in
`/tmp/whisper-prep-fix2.OrcMrx`:

- Python 3.11.16: **149 passed**, no failures or skips.
- Python 3.12.14: **149 passed**, no failures or skips.
- Each full run includes all **11 Hub tests** and **35 additional regressions**.
- The original external reproduction suite now reports **11 passed**.
- Ruff lint and formatting checks passed.
- Source manifest, source distribution and wheel build, and Twine checks passed.

The full test command was `PYTHONPATH=src python -m pytest -q -m ""`, using
the corresponding remote environment for each Python version. The empty marker
expression explicitly includes the Hub integration group. Final logs are
retained in the sibling directory `../prep-review2-validation/fixed/`.

These results verify the reported defects and tested compatibility cases. They
do not establish the absence of bugs in untested inputs, dependency versions or
execution paths; that would require a precise specification and substantially
stronger, explicitly scoped verification assumptions.
