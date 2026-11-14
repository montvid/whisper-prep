"""Whisper data preparation. Importing the package does not load any models."""

__version__ = "0.1.0"


def main(config=None):
    from whisper_prep.pipeline import main as run

    return run(config)


def _resolve_input_sources(*args, **kwargs):
    from whisper_prep.pipeline import _resolve_input_sources as resolve

    return resolve(*args, **kwargs)


def _apply_basic_text_filters(*args, **kwargs):
    from whisper_prep.pipeline import _apply_basic_text_filters as apply

    return apply(*args, **kwargs)


def __getattr__(name):
    if name == "DataProcessor":
        from whisper_prep.generation.data_processor import DataProcessor

        return DataProcessor
    raise AttributeError(name)
