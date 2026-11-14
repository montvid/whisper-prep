from pathlib import Path

from datasets import load_from_disk

from whisper_prep import main
from whisper_prep.generation.data_processor import DataProcessor
from whisper_prep.generation.generate import generate_fold

ASSETS = Path(__file__).parent / "assets/tsv-data-example"


def test_generate_data(tmp_path):
    inputs = dict(
        tsv_paths=[str(ASSETS / "sentences.tsv")],
        clips_folders=[str(ASSETS / "clips")],
        partials=[1.0],
        normalize_text=True,
        n_samples_per_srt=8,
        n_jobs=1,
        language="de",
    )
    direct = tmp_path / "direct"
    generate_fold(
        **{key: value for key, value in inputs.items() if key != "language"}, out_folder=direct
    )
    processor = DataProcessor(
        audio_dir=direct / "audios",
        transcript_dir=direct / "transcripts",
        output=direct / "data.ljson",
        dump_dir=direct / "dump",
    )
    processor.run()
    assert DataProcessor.read_records(direct / "data.ljson")
    for split in ("train", "val"):
        main(
            {
                **inputs,
                "out_folder_base": str(tmp_path),
                "dataset_name": "corpus",
                "split_name": split,
            }
        )
        dataset = load_from_disk(tmp_path / "corpus" / split / "hf")[split]
        assert len(dataset) > 0
        assert {"audio", "text", "language", "prompt"}.issubset(dataset.column_names)
