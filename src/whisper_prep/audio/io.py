from pathlib import Path
from typing import Optional, Union

from pydub import AudioSegment, effects


def read_audio(
    path: Union[str, Path], resample_rate: Optional[int] = None, normalize: bool = True
) -> AudioSegment:
    # Extensions such as .opus name a codec, not an FFmpeg input container.
    audio_segment = None
    if Path(path).suffix.lower() == ".flac":
        import soundfile as sf

        with sf.SoundFile(path) as stream:
            if stream.subtype == "PCM_16":
                samples = stream.read(dtype="int16", always_2d=True)
                audio_segment = AudioSegment(
                    samples.tobytes(),
                    sample_width=2,
                    frame_rate=stream.samplerate,
                    channels=stream.channels,
                )
    if audio_segment is None:
        audio_segment = AudioSegment.from_file(path)

    if resample_rate is not None and audio_segment.frame_rate != resample_rate:
        # set_frame_rate uses audioop.ratecv, which aliases frequencies above
        # the new Nyquist limit. Export parameters actually reach FFmpeg's
        # output resampler; from_file(parameters=...) does not do this for WAV.
        with audio_segment.export(format="wav", parameters=["-ar", str(resample_rate)]) as stream:
            audio_segment = AudioSegment.from_wav(stream)

    if normalize:
        audio_segment = effects.normalize(audio_segment)

    return audio_segment


def save_audio_segment(audio_segment: AudioSegment, path: Union[str, Path], format: str) -> None:
    if format == "flac" and audio_segment.sample_width == 2:
        import numpy as np
        import soundfile as sf

        samples = np.asarray(audio_segment.get_array_of_samples()).reshape(
            -1, audio_segment.channels
        )
        sf.write(path, samples, audio_segment.frame_rate, format="FLAC", subtype="PCM_16")
        return
    audio_segment.export(path, format=format)
