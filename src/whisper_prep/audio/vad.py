import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

import numpy as np

try:
    from webrtcvad import Vad
except ImportError:  # optional dependency; only needed for VoiceActivityDetector
    Vad = None


VAD_BACKENDS = {"torch", "onnx", "onnx-sequence"}


@lru_cache(maxsize=3)
def get_vad_model(backend="torch"):
    if backend not in VAD_BACKENDS:
        raise ValueError(f"Unsupported VAD backend: {backend}")
    from silero_vad import load_silero_vad

    if backend != "torch":
        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ONNX VAD requires the optional extra: pip install 'whisper-prep[onnx]'"
            ) from exc
        return load_silero_vad(onnx=True, sequence=backend == "onnx-sequence")
    return load_silero_vad()


def read_audio(path):
    from silero_vad import read_audio as decode

    return decode(path, sampling_rate=16000)


def get_speech_timestamps(*args, **kwargs):
    from silero_vad import get_speech_timestamps as detect

    return detect(*args, **kwargs)


def _detect(audio, backend, **kwargs):
    # Both ONNX implementations explicitly use Silero's single-threaded CPU
    # provider. Cache one model per backend in each spawned worker.
    if backend == "onnx-sequence":
        from silero_vad import get_speech_timestamps_sequence

        return get_speech_timestamps_sequence(audio, get_vad_model(backend), **kwargs)
    model = get_vad_model() if backend == "torch" else get_vad_model(backend)
    return get_speech_timestamps(audio, model, **kwargs)


def _waveform(audio):
    from pathlib import Path

    import torch

    if isinstance(audio, (str, Path)):
        return read_audio(str(audio))
    return torch.as_tensor(audio, dtype=torch.float32)


def __getattr__(name):
    if name == "SILERO_MODEL":  # Compatibility for callers of the old module constant.
        return get_vad_model()
    raise AttributeError(name)


@dataclass
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


class VoiceActivityDetector:
    def __init__(
        self,
        aggressiveness: int,
        frame_duration_ms: int,
        padding_duration_ms: int,
        rate_voiced_frames_threshold: float = 0.9,
        rate_unvoiced_frames_threshold: float = 0.9,
    ) -> None:
        if Vad is None:
            raise ImportError(
                "VoiceActivityDetector requires 'webrtcvad' (or 'webrtcvad-wheels'). "
                "Default pipeline uses silero_vad_collector and does not need it."
            )
        self.vad = Vad(aggressiveness)

        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)

        self.rate_voiced_frames_threshold = rate_voiced_frames_threshold
        self.rate_unvoiced_frames_threshold = rate_unvoiced_frames_threshold

    def frame_generator(self, audio: bytes, sample_rate: int) -> Iterator[Frame]:
        n = int(sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset : offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def return_start_end_of_audio(
        self,
        audio: np.array,
        sample_rate: int,
    ) -> tuple[float, float]:
        ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        triggered = False

        start_second = None
        end_second = None

        # Get raw data
        raw_data = audio.raw_data

        for frame in self.frame_generator(raw_data, sample_rate):
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])

                if num_voiced > self.rate_voiced_frames_threshold * ring_buffer.maxlen:
                    triggered = True

                    if start_second is None:
                        start_second = ring_buffer[0][0].timestamp

                    ring_buffer.clear()
            else:
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                if num_unvoiced > self.rate_unvoiced_frames_threshold * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
                    end_second = frame.timestamp

        if triggered:
            end_second = (len(raw_data) / sample_rate) / 2.0

        if start_second is None:
            start_second = 0

        return start_second, end_second


def silero_vad_collector(
    path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 1024,
    speech_pad_ms: int = 30,
    backend: str = "torch",
) -> tuple[float, float]:

    audio = _waveform(path)

    # Get speech timestamps
    speech_timestamps = _detect(
        audio,
        backend,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    if not speech_timestamps:
        return 0.0, 0.0

    start_second = speech_timestamps[0]["start"] / 16000
    end_second = speech_timestamps[-1]["end"] / 16000

    return start_second, end_second


def silero_speech_ratio(
    path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 1024,
    speech_pad_ms: int = 30,
    backend: str = "torch",
) -> float:
    audio = _waveform(path)
    if len(audio) == 0:
        return 0.0

    speech_timestamps = _detect(
        audio,
        backend,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    speech_samples = sum(
        max(0, timestamp["end"] - timestamp["start"]) for timestamp in speech_timestamps
    )
    return speech_samples / len(audio)
