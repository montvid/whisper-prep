import csv
import inspect
import json
import logging
import os
import random
import re
import unicodedata
import warnings
from collections import Counter, defaultdict, deque
from multiprocessing import get_context
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from tqdm import tqdm
from whisper.audio import load_audio
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import format_timestamp

from whisper_prep.audio.vad import silero_speech_ratio
from whisper_prep.generation.typing import PromptNode, Record, Utterance
from whisper_prep.subtitling.parse import milliseconds, read_cues
from whisper_prep.text import (
    clean_subtitle_markup,
    drop_text_fragments,
    report_name,
    resolve_path,
    sanitize_example_id,
)

DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)
_WORKER_PROCESSOR = None
LOGGER = logging.getLogger(__name__)
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".wma"}


def _init_transcripts_worker(processor_kwargs: dict) -> None:
    global _WORKER_PROCESSOR
    torch.set_num_threads(1)
    kwargs = dict(processor_kwargs)
    # Workers return metadata to the parent; no shard files are created.
    kwargs["output"] = f"{kwargs['output']}.worker-{os.getpid()}"
    _WORKER_PROCESSOR = DataProcessor(**kwargs)


def _process_transcripts_tsv_row(row: dict) -> dict:
    if _WORKER_PROCESSOR is None:
        raise RuntimeError("Transcript worker was not initialized")
    return _WORKER_PROCESSOR._process_row(row)


class DataProcessor:
    def __init__(
        self,
        audio_dir: str,
        transcript_dir: str,
        with_timestamps: bool = True,
        data_file: Optional[str] = None,
        transcript_formats: Optional[List[str]] = None,
        language: str = "de",
        output: str = "data.json",
        dump_dir: str = "dump",
        timestamp_resolution: int = 20,
        max_prompt_length: int = 223,  # 223 tokens and some extra for the time stamps.
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        keep_empty_chance: float = 0.0,
        rep_threshold: int = 3,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
        cut_initial_audio: bool = False,
        filter_segment_words: Optional[List[str]] = None,
        drop_text: Optional[List[str]] = None,
        transcripts_tsv: Optional[str] = None,
        validate_empty_with_vad: bool = False,
        empty_vad_max_speech_ratio: float = 0.06,
        n_jobs: int = 1,
        seed: int = 42,
        max_error_rate: float = 0.0,
        audio_format: str = "mp3",
        vad_backend: str = "torch",
    ) -> None:
        self.with_timestamps = with_timestamps
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.data_file = data_file
        self.transcript_formats = list(transcript_formats or ["{id}.srt", "{id}.vtt"])
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.timestamp_resolution = timestamp_resolution
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.subsampling_factor_for_silence = subsampling_factor_for_silence
        self.keep_empty_chance = keep_empty_chance
        self.rep_threshold = rep_threshold
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode
        self.cut_initial_audio = cut_initial_audio
        self.filter_segment_words = filter_segment_words
        self.drop_text = drop_text
        self.transcripts_tsv = transcripts_tsv
        self.validate_empty_with_vad = validate_empty_with_vad
        self.empty_vad_max_speech_ratio = empty_vad_max_speech_ratio
        self.n_jobs = n_jobs
        self.seed = seed
        self.max_error_rate = max_error_rate
        self.audio_format = audio_format
        self.vad_backend = vad_backend
        self.rng = random.Random(seed)
        self.stats = Counter()
        self.filtered_segment_records: List[dict] = []

        self._verify_args()

        self.tokenizer = get_tokenizer(multilingual=(self.tokenizer_type == "multilingual"))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        from whisper_prep.audio.vad import VAD_BACKENDS

        if self.audio_format not in {"mp3", "wav", "flac"}:
            raise ValueError("audio_format must be mp3, wav or flac")
        if self.vad_backend not in VAD_BACKENDS:
            raise ValueError("vad_backend must be torch, onnx or onnx-sequence")
        if self.with_timestamps:
            if not self.transcripts_tsv:
                if not self.audio_dir or not self.transcript_dir:
                    raise ValueError(
                        "`audio_dir` and `transcript_dir` must be set when `with_timestamps` is True and no transcripts_tsv provided"
                    )
            if self.timestamp_resolution <= 0 or self.timestamp_resolution % 20 != 0:
                raise ValueError(
                    "`timestamps_resolution` must be multiples of 20ms. "
                    f"Got {self.timestamp_resolution}"
                )
        else:
            if not self.data_file:
                raise ValueError("`data_file` must be set when `with_timestamps` is False")

        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

        if not 0 <= self.keep_empty_chance <= 1:
            raise ValueError(
                f"keep_empty_chance must be between 0 and 1, got {self.keep_empty_chance}"
            )

        if not 0 <= self.empty_vad_max_speech_ratio <= 1:
            raise ValueError(
                "empty_vad_max_speech_ratio must be between 0 and 1, "
                f"got {self.empty_vad_max_speech_ratio}"
            )

        if self.max_tokens_length < 3 or self.max_prompt_length < 0:
            raise ValueError("max_tokens_length must be >= 3 and max_prompt_length >= 0")
        if self.rep_threshold < 2 or self.subsampling_factor_for_silence < 1:
            raise ValueError("rep_threshold must be >= 2 and subsampling_factor_for_silence >= 1")
        if not 0 <= self.max_error_rate <= 1:
            raise ValueError("max_error_rate must be between 0 and 1")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be at least 1, got {self.n_jobs}")

    def run(self, *, input_rows=None) -> None:
        try:
            if self.with_timestamps:
                self._process_with_timestamps(input_rows)
            else:
                self._process_without_timestamps()
            if self.subsampling_factor_for_silence > 1:
                self._subsample_silence()
        finally:
            self._write_filtered_segments()
            summary_path = Path(self.output).with_suffix(".summary.json")
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(dict(self.stats), indent=2) + "\n", encoding="utf-8")
        LOGGER.info("Processing totals: %s", dict(self.stats))
        if self.stats["failed_files"] / max(1, self.stats["files"]) > self.max_error_rate:
            raise RuntimeError(
                f"Failed to process {self.stats['failed_files']}/{self.stats['files']} files; "
                f"see {Path(self.output).with_suffix('.errors.json')}"
            )

    def _process_without_timestamps(self) -> None:
        records = []
        with open(self.data_file, encoding="utf-8") as f:
            for line in f:
                audio_path, text = line.rstrip("\n").split("\t", 1)
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(audio_path=audio_path, text=text, language=self.language)
                records.append(record)

        self.write_records(records, self.output)

    def _drop_repeated_utterances_with_records(
        self, utterances: List[Utterance], threshold: int = 3
    ) -> tuple[List[Utterance], List[dict]]:
        if not utterances:
            return [], []

        # This function assumes utterances are already sorted by start time.
        result_utterances = []
        dropped_records = []
        repeat_count = 1
        last_text = None

        def flush_group(end_index: int) -> None:
            group = utterances[end_index - repeat_count : end_index]
            if repeat_count < threshold:
                result_utterances.extend(group)
                return

            dropped_records.append(
                {
                    "start_ms": min(utterance.start for utterance in group),
                    "end_ms": max(utterance.end for utterance in group),
                    "text": group[0].text,
                    "matched_word": "repeated_hallucination",
                }
            )

        for i in range(len(utterances)):
            current_text = utterances[i].text

            if i == 0:
                last_text = current_text
                continue

            if current_text == last_text:
                repeat_count += 1
            else:
                flush_group(i)
                # Reset count and update last_text
                repeat_count = 1
                last_text = current_text

        # Check the last sequence at the end of the list
        flush_group(len(utterances))

        return result_utterances, dropped_records

    def _drop_single_letter_utterance(self, utterances: List[Utterance]) -> List[Utterance]:
        sanitized_utterances = []
        for utterance in utterances:
            if len(utterance.text.strip()) > 0:
                sanitized_utterances.append(utterance)

        return sanitized_utterances

    @staticmethod
    def _is_punctuation_hallucination(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if len(compact) < 3:
            return False
        has_word_character = re.search(r"\w", compact, flags=re.UNICODE)
        return has_word_character is None

    def _drop_punctuation_hallucinations_with_records(
        self, utterances: List[Utterance]
    ) -> tuple[List[Utterance], List[dict]]:
        kept = []
        dropped_records = []
        for utterance in utterances:
            if self._is_punctuation_hallucination(utterance.text):
                dropped_records.append(
                    {
                        "start_ms": utterance.start,
                        "end_ms": utterance.end,
                        "text": utterance.text,
                        "matched_word": "punctuation_hallucination",
                    }
                )
            else:
                kept.append(utterance)
        return kept, dropped_records

    def _sanitize_utterances(
        self, utterances, filtered_out=None, source_id=None, transcript_path=None
    ) -> List[Utterance]:
        filtered_out = filtered_out if filtered_out is not None else []
        valid = []
        for cue in utterances:
            if cue.start is None or cue.end is None or cue.start < 0 or cue.start >= cue.end:
                filtered_out.append(
                    {
                        "speech_id": source_id or "",
                        "transcript_path": str(transcript_path or ""),
                        "start_ms": max(0, min(cue.start or 0, cue.end or 0)),
                        "end_ms": max(cue.start or 0, cue.end or 0),
                        "text": cue.text,
                        "matched_word": "invalid_timestamps",
                    }
                )
                continue
            if cue.text and cue.text.strip():
                valid.append(Utterance(cue.text, cue.start, cue.end))
        valid.sort(key=lambda cue: (cue.start, cue.end))
        valid, repeats = self._drop_repeated_utterances_with_records(valid, self.rep_threshold)
        valid, punctuation = self._drop_punctuation_hallucinations_with_records(valid)
        for record in repeats + punctuation:
            filtered_out.append(
                {
                    "speech_id": source_id or "",
                    "transcript_path": str(transcript_path or ""),
                    **record,
                }
            )
        repaired = []
        for cue in valid:
            if repaired and repaired[-1].end > cue.start:
                previous = repaired[-1]
                self.stats["repaired_overlaps"] += 1
                if previous.start < cue.start and previous.end - cue.start <= 500:
                    previous.end = cue.start
                else:
                    # Simultaneous/nested cues cannot have monotone timestamps;
                    # retain both texts inside their combined audio interval.
                    previous.text += " " + cue.text
                    previous.end = max(previous.end, cue.end)
                    continue
            repaired.append(cue)
        return repaired

    @staticmethod
    def _language_code(value):
        value = TO_LANGUAGE_CODE.get(value, value)
        if value not in LANGUAGES:
            raise ValueError(f"Unsupported language: {value}")
        return value

    def _input_rows(self):
        if self.transcripts_tsv:
            base = Path(self.transcripts_tsv).resolve().parent
            with open(self.transcripts_tsv, encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t", strict=True)
                if not {"audio_path", "srt_path"}.issubset(reader.fieldnames or []):
                    raise ValueError("transcripts_tsv requires audio_path and srt_path columns")
                rows = list(reader)
        else:
            base = Path.cwd()
            rows = []
            for audio in sorted(Path(self.audio_dir).iterdir()):
                if not audio.is_file() or audio.suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                transcripts = [
                    Path(self.transcript_dir) / fmt.format(id=audio.stem)
                    for fmt in self.transcript_formats
                ]
                transcript = next((path for path in transcripts if path.is_file()), transcripts[0])
                rows.append(
                    {
                        "id": audio.stem,
                        "audio_path": str(audio.resolve()),
                        "srt_path": str(transcript.resolve()),
                    }
                )
        seen = {}
        for row in rows:
            if not row.get("audio_path") or not row.get("srt_path") or None in row:
                raise ValueError("Invalid row in transcripts_tsv")
            row["audio_path"] = str(resolve_path(row["audio_path"], base))
            row["srt_path"] = str(resolve_path(row["srt_path"], base))
            raw_id = row.get("id") or Path(row["audio_path"]).stem
            row["id"] = sanitize_example_id(raw_id)
            if row["id"] in seen:
                raise ValueError(
                    f"Duplicate output id {row['id']!r}: {seen[row['id']]} and {row['audio_path']}"
                )
            seen[row["id"]] = row["audio_path"]
            row["language"] = self._language_code(row.get("language") or self.language)
        return rows

    def _processor_kwargs(self):
        kwargs = {name: getattr(self, name) for name in inspect.signature(type(self)).parameters}
        kwargs.update(n_jobs=1, subsampling_factor_for_silence=1)
        return kwargs

    def _process_with_timestamps(self, rows=None) -> None:
        # Supplied rows were validated before optional subtitle normalization.
        if rows is None:
            rows = self._input_rows()
        Path(self.output).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output).touch()
        if self.n_jobs > 1 and rows:
            with get_context("spawn").Pool(
                min(self.n_jobs, len(rows)),
                initializer=_init_transcripts_worker,
                initargs=(self._processor_kwargs(),),
            ) as pool:
                self._consume_results(pool.imap(_process_transcripts_tsv_row, rows), len(rows))
        else:
            self._consume_results(map(self._process_row, rows), len(rows))

    def _consume_results(self, results, total):
        summary = Counter(files=total)
        filtered = []
        errors = []
        for result in tqdm(results, total=total, desc="Processing audio and subtitles"):
            summary.update(result["stats"])
            filtered.extend(result["filtered_records"])
            if result["error"]:
                errors.append(result["error"])
                summary["failed_files"] += 1
            else:
                self.write_records(result["records"], self.output)
                summary["output_records"] += len(result["records"])
        self.stats = summary
        self.filtered_segment_records = filtered
        if errors:
            Path(self.output).with_suffix(".errors.json").write_text(
                json.dumps(errors, indent=2) + "\n", encoding="utf-8"
            )

    def _process_row(self, row):
        if row.get("_normalization_error"):
            return {
                "records": [],
                "filtered_records": [],
                "error": row["_normalization_error"],
                "stats": {},
            }
        original_language = self.language
        self.language = row["language"]
        self.rng = random.Random(f"{self.seed}:{row['id']}")
        self.stats = Counter()
        self.filtered_segment_records = []
        records, error = [], None
        try:
            transcript = Path(row["srt_path"])
            if transcript.suffix.lower() not in {".srt", ".vtt"}:
                raise ValueError(f"Unsupported transcript format: {transcript.suffix}")
            cues = self.read_utterances_from_srt(
                transcript,
                self.normalize_unicode,
                self.filter_segment_words,
                self.drop_text,
                self.filtered_segment_records,
                row["id"],
            )
            self.stats["input_cues"] = len(cues) + len(self.filtered_segment_records)
            cues = self._sanitize_utterances(
                cues, self.filtered_segment_records, row["id"], transcript
            )
            blocked = [(r["start_ms"], r["end_ms"]) for r in self.filtered_segment_records]
            records = self._create_records_with_timestamps(
                cues,
                Path(row["audio_path"]),
                row["id"],
                blocked_intervals=blocked,
            )
        except Exception as exc:
            LOGGER.exception(
                "Failed processing audio %s with subtitles %s", row["audio_path"], row["srt_path"]
            )
            error = {
                "audio_path": row["audio_path"],
                "transcript_path": row["srt_path"],
                "type": type(exc).__name__,
                "message": str(exc),
            }
            # A failed file must not leave segments that look like accepted data.
            for path in (Path(self.dump_dir) / row["id"]).glob(f"*.{self.audio_format}"):
                path.unlink()
        finally:
            self.language = original_language
        self.stats["dropped_cues"] += len(self.filtered_segment_records)
        for record in self.filtered_segment_records:
            self.stats["dropped_" + record["matched_word"]] += 1
        return {
            "records": records,
            "filtered_records": self.filtered_segment_records,
            "error": error,
            "stats": dict(self.stats),
        }

    @staticmethod
    def read_utterances_from_srt(
        transcript_path,
        normalize_unicode=False,
        filter_segment_words=None,
        drop_text=None,
        filtered_out=None,
        source_id=None,
    ) -> List[Utterance]:
        kept = []
        for cue in read_cues(transcript_path):
            text = unicodedata.normalize("NFKC", cue.text) if normalize_unicode else cue.text
            text = drop_text_fragments(text, drop_text)
            if not text:
                continue
            word = next(
                (
                    word
                    for word in filter_segment_words or []
                    if word and word.lower() in text.lower()
                ),
                None,
            )
            if word is not None:
                if filtered_out is not None:
                    filtered_out.append(
                        {
                            "speech_id": source_id or Path(transcript_path).stem,
                            "transcript_path": str(transcript_path),
                            "start_ms": cue.start,
                            "end_ms": cue.end,
                            "text": text,
                            "matched_word": word,
                        }
                    )
            else:
                kept.append(Utterance(text, cue.start, cue.end))
        return kept

    read_utterances_from_vtt = read_utterances_from_srt
    _clean_subtitle_markup = staticmethod(clean_subtitle_markup)
    _drop_text_fragments = staticmethod(drop_text_fragments)

    @staticmethod
    def _parse_timestamp_line(line: str) -> tuple[int, int]:
        start_time, end_time = line.strip().split(" --> ", 1)
        end_time = end_time.split()[0]
        return (
            DataProcessor.str_to_milliseconds(start_time),
            DataProcessor.str_to_milliseconds(end_time),
        )

    def _write_filtered_segments(self) -> None:
        if not self.filtered_segment_records:
            return

        out_folder = Path(self.output).parent
        if out_folder.name == "created_dataset":
            out_folder = out_folder.parent

        grouped = defaultdict(list)
        for record in self.filtered_segment_records:
            grouped[record.get("matched_word", "")].append(record)

        for word, records in grouped.items():
            if not records:
                continue
            out_path = out_folder / f"filtered_{report_name(word)}_segments.tsv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "speech_id",
                        "transcript_path",
                        "start_ms",
                        "end_ms",
                        "text",
                        "matched_word",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(records)

    def _create_records_with_timestamps(
        self,
        utterances: List[Utterance],
        audio_path: Path,
        speech_id: Optional[str] = None,
        blocked_intervals: Optional[List[tuple]] = None,
    ) -> List[Record]:
        audio = torch.from_numpy(load_audio(audio_path))
        dump_dir = Path(self.dump_dir) / sanitize_example_id(speech_id or audio_path.stem)
        dump_dir.mkdir(parents=True, exist_ok=True)
        audio_duration_ms = int(audio.size(0) * 1000 / SAMPLE_RATE)
        blocked = list(blocked_intervals or [])
        prepared = []
        for cue in sorted(utterances, key=lambda u: u.start):
            cue = Utterance(cue.text, cue.start, cue.end)
            reason = None
            if cue.end > audio_duration_ms:
                if cue.start < audio_duration_ms and cue.end - audio_duration_ms <= 1000:
                    LOGGER.warning("Clamping final cue in %s to the audio duration", audio_path)
                    cue.end = audio_duration_ms
                    self.stats["clamped_cues"] += 1
                else:
                    reason = "outside_audio"
            if cue.end - cue.start > DURATION:
                reason = "cue_duration"
            if (
                len(self.tokenizer.encode(self._add_leading_space(cue.text))) + 2
                > self.max_tokens_length
            ):
                reason = "cue_token_budget"
            if any(cue.start < end and cue.end > start for start, end in blocked):
                reason = "blocked_boundary"
            if reason:
                blocked.append((cue.start, cue.end))
                self.filtered_segment_records.append(
                    {
                        "speech_id": speech_id or audio_path.stem,
                        "transcript_path": "",
                        "start_ms": cue.start,
                        "end_ms": cue.end,
                        "text": cue.text,
                        "matched_word": reason,
                    }
                )
                LOGGER.warning("Dropping cue from %s (%s): %s", audio_path, reason, cue.text)
            else:
                prepared.append(cue)
        utterances = prepared
        safe_spans = self._get_safe_spans(audio_duration_ms, blocked)
        records = []
        utterances = sorted(utterances, key=lambda u: u.start)
        span_cursor = 0
        for i, (span_start, span_end) in enumerate(safe_spans):
            if span_start >= span_end:
                continue
            # Restart aggregation across filtered-word gaps.
            prompt_buffer: Deque[PromptNode] = deque()

            span_utterances = []
            while span_cursor < len(utterances):
                utterance = utterances[span_cursor]
                if utterance.end is None or utterance.start is None:
                    span_cursor += 1
                    continue
                if utterance.end <= span_start:
                    span_cursor += 1
                    continue
                if utterance.start >= span_end:
                    break
                # Defensive: if a subtitle intersects a blocked interval boundary, drop it.
                if utterance.start < span_start or utterance.end > span_end:
                    span_cursor += 1
                    continue
                span_utterances.append(utterance)
                span_cursor += 1

            if not span_utterances:
                records.extend(
                    self._create_empty_records_for_span(
                        audio, audio_path, dump_dir, span_start, span_end
                    )
                )
                continue

            # Optionally trim initial audio only for the first safe span.
            if i == 0 and self.cut_initial_audio:
                segment_start = max(span_start, span_utterances[0].start - 1000)
            else:
                segment_start = span_start

            idx = 0
            while idx < len(span_utterances):
                segment_end = min(segment_start + DURATION, span_end)
                if segment_start >= segment_end:
                    break

                # If the utterance is included in the segment and longer than the segment, skip it.
                if (
                    span_utterances[idx].start < segment_end
                    and span_utterances[idx].start + DURATION < span_utterances[idx].end
                ):
                    # The skipped utterance breaks the continuity of the transcript, so
                    # the buffered prompt must not be carried into the following
                    # segments -- the paper conditions on "the transcript text preceding
                    # the current audio segment". The invalid-utterance skip path below
                    # already clears it.
                    prompt_buffer.clear()
                    segment_start = span_utterances[idx].end
                    idx += 1
                    continue

                prompt = self._get_prompt(prompt_buffer)

                segment_utterances = []
                next_segment_start = None
                packed_tokens = 0
                while idx < len(span_utterances) and span_utterances[idx].start < segment_end:
                    if span_utterances[idx].end > segment_end:
                        next_segment_start = span_utterances[idx].start
                        break
                    cost = (
                        len(
                            self.tokenizer.encode(
                                self._add_leading_space(span_utterances[idx].text)
                            )
                        )
                        + 2
                    )
                    # Leave room for the next cue's start timestamp. A single
                    # cue that exactly fills the budget gets a shorter audio
                    # window below, without an extra dangling token.
                    if segment_utterances and packed_tokens + cost + 1 > self.max_tokens_length:
                        next_segment_start = span_utterances[idx].start
                        break
                    segment_utterances.append(span_utterances[idx])
                    packed_tokens += cost
                    idx += 1

                if packed_tokens == self.max_tokens_length and next_segment_start is not None:
                    segment_end = segment_utterances[-1].end
                    next_segment_start = None

                if not self._is_valid_utterances(segment_utterances, segment_start):
                    tqdm.write(
                        f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                        f"{format_timestamp(segment_end / 1000)}) because it contains invalid "
                        f"utterances: {segment_utterances}"
                    )
                    prompt_buffer.clear()
                    self.stats["invalid_windows"] += 1
                    segment_start = (
                        next_segment_start if next_segment_start is not None else segment_end
                    )
                    continue

                tokens_length = 0
                segment_text = []
                for utterance in segment_utterances:
                    start_token = self._get_time_token(utterance.start, segment_start, audio_path)
                    end_token = self._get_time_token(utterance.end, segment_start, audio_path)
                    utterance_text = self._add_leading_space(utterance.text)
                    segment_text.extend([start_token, utterance_text, end_token])
                    new_prompt_length = len(self.tokenizer.encode(utterance_text)) + 2
                    new_prompt_node = PromptNode(
                        start_token + utterance_text + end_token, new_prompt_length
                    )
                    tokens_length += new_prompt_length

                    prompt_buffer.append(new_prompt_node)

                # An utterance that straddles the window edge contributes only its
                # start timestamp, with no text and no end timestamp. Whisper uses
                # this dangling token to decide where the next 30 s window begins
                # (paper section 2.3: "For segments that are only partially included
                # in the current 30-second window, we predict only their start time
                # token ... to indicate that the subsequent decoding should be
                # performed on an audio window aligned with that time").
                if next_segment_start is not None:
                    dangling_token = self._get_time_token(
                        next_segment_start, segment_start, audio_path
                    )
                    segment_text.append(dangling_token)
                    prompt_buffer.append(PromptNode(dangling_token, 1))
                    tokens_length += 1

                if tokens_length > self.max_tokens_length:
                    tqdm.write(
                        f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                        f"{format_timestamp(segment_end / 1000)}) because it is too long "
                        f"({tokens_length} tokens)"
                    )
                elif not segment_utterances and self.rng.random() >= self.keep_empty_chance:
                    pass
                else:
                    # Always store the full window. Truncating at the last
                    # complete utterance made every training sample end in silence,
                    # which is not what the model sees at inference time.
                    segment_audio_path = self._save_segment_audio(
                        audio, segment_start, segment_end, dump_dir
                    )
                    if segment_utterances or self._empty_segment_passes_vad(segment_audio_path):
                        record = Record(
                            audio_path=segment_audio_path,
                            language=self.language,
                            text="".join(segment_text),
                            prompt=prompt,
                        )
                        records.append(record)

                if next_segment_start is not None:
                    segment_start = next_segment_start
                elif len(segment_utterances) == 0:
                    segment_start += DURATION
                else:
                    segment_start = segment_utterances[-1].end

            if segment_start < span_end:
                records.extend(
                    self._create_empty_records_for_span(
                        audio, audio_path, dump_dir, segment_start, span_end
                    )
                )

        return records

    def _create_empty_records_for_span(
        self,
        audio: torch.Tensor,
        audio_path: Path,
        dump_dir: Path,
        span_start: int,
        span_end: int,
    ) -> List[Record]:
        records = []
        segment_start = span_start
        while segment_start < span_end:
            segment_end = min(segment_start + DURATION, span_end)
            if segment_start >= segment_end:
                break
            if self.rng.random() < self.keep_empty_chance:
                segment_audio_path = self._save_segment_audio(
                    audio, segment_start, segment_end, dump_dir
                )
                if not self._empty_segment_passes_vad(segment_audio_path):
                    segment_start = segment_end
                    continue
                records.append(
                    Record(
                        audio_path=segment_audio_path,
                        language=self.language,
                        text="",
                        prompt="",
                    )
                )
            segment_start = segment_end
        return records

    def _empty_segment_passes_vad(self, segment_audio_path: str) -> bool:
        if not self.validate_empty_with_vad:
            return True

        # Validate the actual encoded training audio. MP3 encoding can alter
        # VAD decisions; checking only the pre-encoding PCM is insufficient.
        speech_ratio = silero_speech_ratio(segment_audio_path, backend=self.vad_backend)
        if speech_ratio <= self.empty_vad_max_speech_ratio:
            return True

        tqdm.write(
            f"Skipping empty-text segment {segment_audio_path} because VAD found "
            f"{speech_ratio:.2%} speech"
        )
        Path(segment_audio_path).unlink(missing_ok=True)
        self.stats["vad_rejected_segments"] += 1
        return False

    def _save_segment_audio(
        self, audio: torch.Tensor, segment_start: int, segment_end: int, dump_dir: Path
    ) -> str:
        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        audio_end_idx = int(segment_end * SAMPLE_RATE / 1000)
        segment_audio_path = str((dump_dir / f"{segment_start}.{self.audio_format}").absolute())
        segment_audio = audio[
            audio_start_idx : min(
                audio_end_idx, audio_start_idx + DURATION_IN_SAMPLES, audio.size(0)
            )
        ]
        if self.audio_format in {"flac", "wav"}:
            import soundfile as sf

            sf.write(
                segment_audio_path,
                segment_audio.detach().cpu().numpy(),
                SAMPLE_RATE,
                format=self.audio_format.upper(),
                subtype="PCM_16",
            )
            return segment_audio_path
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'encoding' parameter is not fully supported by TorchCodec AudioEncoder.",
                category=UserWarning,
            )
            torchaudio.save(
                segment_audio_path,
                segment_audio.unsqueeze(0),
                SAMPLE_RATE,
                encoding="mp3",
            )
        return segment_audio_path

    @staticmethod
    def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
        if not intervals:
            return []
        merged = []
        for start, end in sorted(intervals):
            if start is None or end is None:
                continue
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _get_safe_spans(
        self, audio_duration_ms: int, blocked_intervals: List[tuple]
    ) -> List[tuple]:
        blocked = self._merge_intervals(blocked_intervals)
        if not blocked:
            return [(0, audio_duration_ms)]

        safe_spans = []
        cursor = 0
        for start, end in blocked:
            start = max(0, min(start, audio_duration_ms))
            end = max(0, min(end, audio_duration_ms))
            if cursor < start:
                safe_spans.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < audio_duration_ms:
            safe_spans.append((cursor, audio_duration_ms))
        return safe_spans

    def _is_valid_utterances(self, utterances: List[Utterance], segment_start: int) -> bool:
        if len(utterances) == 0:
            return True

        for utterance in utterances:
            # Check the utterances' start times are in the segment
            if utterance.start < segment_start:
                return False
            if utterance.start > utterance.end:
                return False
            if self._is_punctuation_hallucination(utterance.text):
                return False

        # Check the utterances do not overlap
        for i in range(len(utterances) - 1):
            if utterances[i].end > utterances[i + 1].start:
                return False

        # Check for repeated words three or more times consecutively
        last_text = utterances[0].text
        repeat_count = 1
        for i in range(1, len(utterances)):
            if utterances[i].text == last_text:
                repeat_count += 1
                if repeat_count >= self.rep_threshold:
                    return False
            else:
                last_text = utterances[i].text
                repeat_count = 1

        return True

    def _add_leading_space(self, text: str) -> str:
        """
        Add a leading space to the text if the language uses spaces to separate words.
        For languages that do not use spaces, namely Chinese, Japanese, Thai, Lao, and
        Burmese, return the text as is.
        """
        if self.language in ["zh", "ja", "th", "lo", "my"]:
            return text
        else:
            return " " + text

    str_to_milliseconds = staticmethod(milliseconds)

    def _get_time_token(self, time: int, segment_start: int, audio_path: Path) -> str:
        """
        Get the time token for the given time.

        Args:
            time: Time in milliseconds
            segment_start: Start time of the segment in milliseconds

        Returns:
            Time token (e.g. self._get_time_token(1200, 1000) -> "<|0.20|>")
        """
        if time < segment_start or segment_start + DURATION < time:
            raise ValueError(
                f"Time {format_timestamp(time / 1000)} is out of the segment "
                f"({format_timestamp(segment_start / 1000)} - "
                f"{format_timestamp((segment_start + DURATION) / 1000)}) of {audio_path}"
            )

        time_in_segment = time - segment_start
        nearest_timestamp = min(
            DURATION,
            (round(time_in_segment / self.timestamp_resolution) * self.timestamp_resolution),
        )  # in milliseconds; coarse custom resolutions must not exceed 30 s.
        time_token = f"<|{nearest_timestamp / 1000:.2f}|>"
        return time_token

    def _get_prompt(self, prompt_buffer: Deque[PromptNode]) -> str:
        prompt_length = sum(node.num_tokens for node in prompt_buffer)
        while prompt_buffer and prompt_length > self.max_prompt_length:
            prompt_length -= prompt_buffer.popleft().num_tokens
        return "".join(node.text for node in prompt_buffer)

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=str(resolve_path(data["audio_path"], Path(path).resolve().parent)),
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                data = {
                    "audio_path": os.path.relpath(record.audio_path, Path(path).resolve().parent),
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _subsample_silence(self) -> None:
        records = self.read_records(self.output)

        silence_records = filter(lambda record: record.text == "", records)
        non_silence_records = filter(lambda record: record.text != "", records)
        filtered_records = (
            list(non_silence_records)
            + list(silence_records)[:: self.subsampling_factor_for_silence]
        )

        # The timestamp-free route references original recordings; only the
        # timestamped route creates disposable segment audio for these records.
        if self.with_timestamps:
            retained = {record.audio_path for record in filtered_records}
            for record in records:
                if record.audio_path not in retained:
                    Path(record.audio_path).unlink(missing_ok=True)
        Path(self.output).unlink()
        self.write_records(filtered_records, self.output)
