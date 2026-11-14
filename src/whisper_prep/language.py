"""Load the standard fastText language model only when filtering is requested."""

import hashlib
import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
MODEL_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"


def _verify_model(data):
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Invalid lid.176.ftz checksum; remove the corrupt cached file and retry")


def _model_path():
    configured = os.environ.get("WHISPER_PREP_LANGUAGE_MODEL")
    if configured:
        path = Path(configured).expanduser()
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        path = cache / "whisper-prep" / "lid.176.ftz"
        legacy = Path.home() / ".cache" / "lid.176.ftz"
        if not path.exists() and legacy.is_file():
            path = legacy
        if not path.exists():
            logging.getLogger(__name__).info("Downloading language model from %s", MODEL_URL)
            with urlopen(MODEL_URL, timeout=30) as response:
                data = response.read(2 * 1024 * 1024)
            _verify_model(data)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(data)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    _verify_model(path.read_bytes())
    return path


@lru_cache(maxsize=1)
def get_language_model():
    import fasttext

    return fasttext.load_model(str(_model_path()))


def identify_language(text):
    labels, probabilities = get_language_model().predict(" ".join(text.split()), k=1)
    if not labels:
        return "", 0.0
    return labels[0].removeprefix("__label__"), float(probabilities[0])
