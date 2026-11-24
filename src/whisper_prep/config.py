"""Validate pipeline options before generating or removing any files."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_AUDIO_FORMAT = "flac"


@dataclass
class PipelineConfig:
    out_folder_base: str
    dataset_name: str
    split_name: str
    language: str = "de"
    tsv_paths: list[str] | None = None
    clips_folders: list[str] | None = None
    partials: list[float] | None = None
    transcripts_tsv: str | None = None
    source_audio_dir: str | None = None
    source_transcript_dir: str | None = None
    maintain_speaker_chance: float = 0.5
    n_samples_per_srt: int = 16
    normalize_text: bool = False
    normalizer_language: str | None = None
    tsv_quoting: str = "csv"
    overlap_chance: float = 0.5
    max_overlap_chance: float = 0.2
    max_overlap_duration: float = 0.2
    vad_chance: float = 1.0
    vad_backend: str = "torch"
    keep_empty_chance: float = 0.0
    audio_format: str = DEFAULT_AUDIO_FORMAT
    seed: int = 42
    n_jobs: int = 1
    transcripts_tsv_n_jobs: int | None = None
    netflix_normalize_n_jobs: int | None = None
    netflix_normalize: bool = False
    cut_initial_audio: bool = False
    filter_french: bool = False
    filter_english: bool = False
    filter_words: list[str] | None = None
    drop_segments_containing: list[str] | None = None
    drop_text: list[str] | None = None
    validate_empty_with_vad: bool = False
    empty_vad_max_speech_ratio: float = 0.06
    subsampling_factor_for_silence: int = 1
    min_text_words: int | None = 8
    drop_empty_text: bool | None = None
    overwrite_output: bool = False
    max_tokens_length: int = 219
    max_prompt_length: int = 223
    rep_threshold: int = 3
    timestamp_resolution: int = 20
    normalize_unicode: bool = False
    tokenizer_type: str = "multilingual"
    max_error_rate: float = 0.0
    upload_to_hu: bool = False
    hu_repo: str | None = None
    hu_private: bool = True

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a mapping")
        if values.get("hu_datasets"):
            raise ValueError("Download datasets first with whisper_prep_download_hf")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        try:
            config = cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self):
        from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

        from whisper_prep.text import sanitize_example_id

        self.language = TO_LANGUAGE_CODE.get(self.language, self.language)
        if self.language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {self.language}")
        for name in ("dataset_name", "split_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or sanitize_example_id(value) != value:
                raise ValueError(f"{name} must be a nonempty safe path component")
        if not isinstance(self.out_folder_base, (str, Path)) or not str(self.out_folder_base):
            raise ValueError("out_folder_base must be a directory path")
        for name in (
            "maintain_speaker_chance",
            "overlap_chance",
            "max_overlap_chance",
            "vad_chance",
            "keep_empty_chance",
            "empty_vad_max_speech_ratio",
            "max_error_rate",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a number between 0 and 1")
        for name in (
            "n_jobs",
            "transcripts_tsv_n_jobs",
            "netflix_normalize_n_jobs",
            "n_samples_per_srt",
            "subsampling_factor_for_silence",
            "max_tokens_length",
            "rep_threshold",
            "timestamp_resolution",
        ):
            value = getattr(self, name)
            if value is None and name.endswith("_n_jobs"):
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("max_prompt_length", "min_text_words"):
            value = getattr(self, name)
            if value is None and name == "min_text_words":
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.max_tokens_length < 3 or self.rep_threshold < 2 or self.timestamp_resolution % 20:
            raise ValueError(
                "Token budget must be >= 3, repetition threshold >= 2, timestamp resolution a multiple of 20 ms"
            )
        if not isinstance(self.max_overlap_duration, (int, float)) or self.max_overlap_duration < 0:
            raise ValueError("max_overlap_duration must be nonnegative")
        for name in (
            "normalize_text",
            "netflix_normalize",
            "cut_initial_audio",
            "filter_french",
            "filter_english",
            "validate_empty_with_vad",
            "overwrite_output",
            "normalize_unicode",
            "upload_to_hu",
            "hu_private",
            "drop_empty_text",
        ):
            value = getattr(self, name)
            if value is None and name == "drop_empty_text":
                continue
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "filter_words",
            "drop_segments_containing",
            "drop_text",
            "tsv_paths",
            "clips_folders",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, list)
                or not all(isinstance(item, (str, Path)) for item in value)
            ):
                raise ValueError(f"{name} must be a list")
        if self.audio_format not in {"mp3", "wav", "flac"} or self.tsv_quoting not in {
            "csv",
            "none",
        }:
            raise ValueError("Unsupported audio_format or tsv_quoting")
        if self.normalizer_language not in {None, "basic", "de-CH"}:
            raise ValueError("normalizer_language must be basic or de-CH")
        from whisper_prep.audio.vad import VAD_BACKENDS

        if self.vad_backend not in VAD_BACKENDS:
            raise ValueError("vad_backend must be torch, onnx or onnx-sequence")
        if self.tokenizer_type not in {"multilingual", "english"}:
            raise ValueError("tokenizer_type must be multilingual or english")
        sentence = bool(self.tsv_paths or self.clips_folders)
        folders = bool(self.source_audio_dir or self.source_transcript_dir)
        if sum((sentence, folders, bool(self.transcripts_tsv))) != 1:
            raise ValueError("Configure exactly one input route")
        if sentence:
            if not self.tsv_paths or not self.clips_folders:
                raise ValueError("Both tsv_paths and clips_folders are required")
            if self.partials is None:
                self.partials = [1.0] * len(self.tsv_paths)
            if not isinstance(self.partials, list) or not (
                len(self.tsv_paths) == len(self.clips_folders) == len(self.partials)
            ):
                raise ValueError("tsv_paths, clips_folders and partials must have equal lengths")
            if any(
                type(value) not in (int, float) or not 0 <= value <= 1 for value in self.partials
            ):
                raise ValueError("partials values must be numbers between 0 and 1")
        if folders and not (self.source_audio_dir and self.source_transcript_dir):
            raise ValueError("Both source_audio_dir and source_transcript_dir are required")
        for name in ("transcripts_tsv", "source_audio_dir", "source_transcript_dir"):
            value = getattr(self, name)
            if value and not Path(value).exists():
                raise FileNotFoundError(value)
        for value in (self.tsv_paths or []) + (self.clips_folders or []):
            if not Path(value).exists():
                raise FileNotFoundError(value)
        if self.upload_to_hu and (not self.hu_repo or len(self.hu_repo.split("/")) != 2):
            raise ValueError("upload_to_hu requires hu_repo='owner/dataset'")

    def to_dict(self):
        return asdict(self)
