"""Prepare local LIEPA-3 Parquet shards in bounded, resumable batches.

Run from an installed checkout with the ``liepa3`` extra. Source Parquet files
are read-only; only the explicitly selected work/output directories are written.
The held-out shard rule matches the previous LIEPA-3 training run.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

SUBSETS = ("read", "spon", "phon")
GIB = 2**30


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def held_out(shards, subset, count=12):
    if len(shards) <= count:
        raise ValueError(
            f"{subset} needs more than {count} shards to preserve the evaluation split"
        )
    if subset == "read":
        return shards[-count:]
    step = len(shards) // count
    return [shards[min(i * step + step // 2, len(shards) - 1)] for i in range(count)]


def select_shards(source, subset):
    shards = sorted((Path(source) / subset).glob("*.parquet"))
    reserved = held_out(shards, subset)
    excluded = set(reserved)
    return [p for p in shards if p not in excluded], reserved


def check_space(path, required):
    free = shutil.disk_usage(path).free
    if free < required:
        raise RuntimeError(
            f"Insufficient space at {path}: {free / GIB:.1f} GiB free; "
            f"{required / GIB:.1f} GiB required"
        )


def export_shard(task):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import soxr

    shard, clips, min_s, max_s = task
    shard, clips = Path(shard), Path(clips)
    bucket = clips / shard.stem
    bucket.mkdir()
    rows, stats = [], Counter()
    seconds = 0.0
    parquet = pq.ParquetFile(shard)
    for batch in parquet.iter_batches(batch_size=32, columns=["audio", "text", "speaker_id"]):
        for row in batch.to_pylist():
            index = stats["source_rows"]
            stats["source_rows"] += 1
            text = row["text"]
            if not isinstance(text, str) or not text.strip():
                stats["empty_text"] += 1
                continue
            # Decoding failures stop the batch; they are never silently skipped.
            audio, rate = sf.read(
                io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True
            )
            audio = audio.mean(axis=1)
            if not np.isfinite(audio).all():
                raise ValueError(f"Nonfinite audio in {shard.name}, row {index}")
            if rate != 16000:
                audio = soxr.resample(audio, rate, 16000)
            duration = len(audio) / 16000
            if duration < min_s:
                stats["too_short"] += 1
                continue
            if duration > max_s:
                stats["too_long"] += 1
                continue
            relative = Path(shard.stem) / f"{index:06d}.flac"
            sf.write(clips / relative, audio, 16000, format="FLAC", subtype="PCM_16")
            rows.append((str(relative), text.strip(), str(row["speaker_id"] or "unk")))
            stats["exported_clips"] += 1
            seconds += duration
    return rows, dict(stats), seconds


def export_batch(shards, work, workers, min_s, max_s):
    stage, clips = work / "stage", work / "clips"
    stage.mkdir()
    clips.mkdir()
    check_space(work, sum(p.stat().st_size for p in shards) + 10 * GIB)

    def copy(path):
        destination = stage / path.name
        shutil.copyfile(path, destination)
        if destination.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"Incomplete staging copy: {path}")
        return destination

    with ThreadPoolExecutor(max_workers=4) as pool:
        staged = list(pool.map(copy, shards))
    stats, seconds = Counter(), 0.0
    with (work / "sentences.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["path", "sentence", "client_id"])
        tasks = [(p, clips, min_s, max_s) for p in staged]
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=get_context("spawn")
        ) as pool:
            # Input order is stable regardless of worker completion order.
            for rows, counts, duration in pool.map(export_shard, tasks):
                writer.writerows(rows)
                stats.update(counts)
                seconds += duration
    shutil.rmtree(stage)
    if not stats["exported_clips"]:
        raise RuntimeError("No usable source clips in this batch")
    result = {**stats, "exported_hours": seconds / 3600}
    write_json(work / "export_summary.json", result)
    return result


def validate_export(hf):
    import pyarrow as pa
    import soundfile as sf

    split = hf / "train"
    state = json.loads((split / "state.json").read_text())
    rows, bytes_total = 0, 0
    digest = hashlib.sha256()
    for entry in state["_data_files"]:
        shard = split / entry["filename"]
        with pa.memory_map(str(shard), "r") as stream:
            reader = pa.ipc.open_stream(stream)
            for batch in reader:
                rows += batch.num_rows
                # Check every metadata row, without decoding all audio again.
                for lang in batch.column("language").to_pylist():
                    if lang != "lt":
                        raise RuntimeError(f"Unexpected language in {shard}")
                for column in ("text", "prompt"):
                    for value in batch.column(column).to_pylist():
                        digest.update(value.encode("utf-8"))
                        digest.update(b"\0")
                if batch.num_rows:
                    audio = batch.column("audio")[0].as_py()
                    blob = audio.get("bytes")
                    if not blob or not blob.startswith(b"fLaC"):
                        raise RuntimeError(f"Expected embedded FLAC in {shard}")
                    info = sf.info(io.BytesIO(blob))
                    if (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
                        raise RuntimeError(f"Unexpected audio format in {shard}")
                    if not 0 < info.duration <= 30:
                        raise RuntimeError(f"Unexpected audio duration in {shard}")
        bytes_total += shard.stat().st_size
    if not rows:
        raise RuntimeError(f"No training rows in {hf}")
    return {
        "rows": rows,
        "arrow_files": len(state["_data_files"]),
        "bytes": bytes_total,
        "text_prompt_sha256": digest.hexdigest(),
    }


def pipeline_config(work, name, workers):
    return dict(
        dataset_name=name,
        split_name="train",
        out_folder_base=str(work / "dataset"),
        tsv_paths=[str(work / "sentences.tsv")],
        clips_folders=[str(work / "clips")],
        partials=[1.0],
        language="lt",
        normalize_text=True,
        normalizer_language="basic",
        maintain_speaker_chance=0.6,
        n_samples_per_srt=16,
        vad_chance=1.0,
        vad_backend="onnx-sequence",
        audio_format="flac",
        keep_empty_chance=0.0,
        overlap_chance=0.4,
        max_overlap_chance=0.2,
        max_overlap_duration=0.5,
        n_jobs=workers,
        seed=42,
        max_error_rate=0.0,
    )


def build_plan(args):
    plan, splits = [], {}
    for subset in args.subsets:
        training, evaluation = select_shards(args.source, subset)
        splits[subset] = {
            "training": [p.name for p in training],
            "evaluation": [p.name for p in evaluation],
        }
        for start in range(0, len(training), args.shards_per_batch):
            end = min(start + args.shards_per_batch, len(training))
            plan.append((f"liepa_{subset}_{start:05d}_{end:05d}", training[start:end]))
            if args.smoke:
                break
    return plan, splits


def prepare(args):
    from whisper_prep import main as prepare_dataset
    from whisper_prep.audio.vad import get_vad_model

    source, output, workspace = args.source.resolve(), args.output.resolve(), args.work.resolve()
    for a, b in ((source, output), (source, workspace), (output, workspace)):
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("Source, output, and work directories must be separate")
    if args.output.is_symlink() or args.work.is_symlink():
        raise ValueError("Output and work must not be symlinks")
    get_vad_model("onnx-sequence")
    plan, splits = build_plan(args)
    output.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args, plan, splits, source, output, workspace, prepare_dataset)


def run_locked(args, plan, splits, source, output, workspace, prepare_dataset):
    manifest = {
        "source": str(source),
        "subsets": args.subsets,
        "excluded_subsets": ["dial"],
        "splits": splits,
        "shards_per_batch": args.shards_per_batch,
        "min_source_seconds": args.min_s,
        "max_source_seconds": args.max_s,
        "workers": args.workers,
        "export_workers": args.export_workers,
        "git_commit": args.git_commit,
        "smoke": args.smoke,
        "versions": {
            name: version(name)
            for name in (
                "whisper-prep",
                "silero-vad",
                "onnxruntime",
                "soxr",
                "soundfile",
                "datasets",
                "torch",
            )
        },
    }
    marker = output / "run_manifest.json"
    if marker.exists():
        if json.loads(marker.read_text()) != manifest:
            raise RuntimeError("Existing run manifest differs; use a new output directory")
    elif any(output.iterdir()):
        raise RuntimeError("Output is not empty and has no matching run manifest")
    else:
        write_json(marker, manifest)
    started, completed, total_rows = time.time(), [], 0
    status = {"state": "running", "pid": os.getpid(), "total_batches": len(plan)}

    def update(**values):
        status.update(
            values,
            updated_utc=datetime.now(timezone.utc).isoformat(),
            completed_batches=len(completed),
            rows=total_rows,
            elapsed_seconds=round(time.time() - started),
        )
        write_json(output / "status.json", status)

    update()
    try:
        for number, (name, shards) in enumerate(plan, 1):
            final = output / "full" / name
            receipt = final / "complete.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                checked = validate_export(final / "train/hf")
                if saved["dataset"] != checked:
                    raise RuntimeError(f"Completed batch validation changed: {name}")
                completed.append(name)
                total_rows += saved["dataset"]["rows"]
                update(current_batch=name, stage="resumed")
                continue
            if final.exists():
                raise RuntimeError(f"Unmarked output batch already exists: {final}")
            work = workspace / name
            partial = output / "full" / f".{name}.partial"
            # Only private, incomplete scratch paths from this run are replaced.
            for path in (work, partial):
                if path.is_symlink():
                    raise ValueError(f"Refusing symlinked scratch path: {path}")
                if path.exists():
                    shutil.rmtree(path)
            check_space(workspace, 20 * GIB)
            check_space(output, 25 * GIB)
            work.mkdir()
            print(
                f"[{number}/{len(plan)}] {name}: exporting {len(shards)} source shards", flush=True
            )
            update(current_batch=name, stage="export")
            start = time.time()
            exported = export_batch(shards, work, args.export_workers, args.min_s, args.max_s)
            config = pipeline_config(work, name, args.workers)
            write_json(work / "config.json", config)
            write_json(
                work / "source_shards.json",
                [
                    {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                    for p in shards
                ],
            )
            print(f"[{number}/{len(plan)}] {name}: {exported}; preparing", flush=True)
            update(stage="prepare", current_export=exported)
            prepare_dataset(config)
            dataset = work / "dataset" / name / "train"
            checked = validate_export(dataset / "hf")
            summary = json.loads((dataset / "created_dataset/data.summary.json").read_text())
            if summary.get("failed_files", 0):
                raise RuntimeError(f"Batch contains file errors: {name}")
            update(stage="copy", current_dataset=checked)
            check_space(output, checked["bytes"] + 10 * GIB)
            shutil.copytree(dataset / "hf", partial / "train/hf")
            # Compare every copied shard by streaming checksum, not just size.
            for original in sorted((dataset / "hf").rglob("*")):
                if original.is_file():
                    copied = partial / "train/hf" / original.relative_to(dataset / "hf")
                    with original.open("rb") as a, copied.open("rb") as b:
                        if (
                            hashlib.file_digest(a, "sha256").digest()
                            != hashlib.file_digest(b, "sha256").digest()
                        ):
                            raise RuntimeError(f"NAS copy checksum mismatch: {copied}")
            for name_ in ("config.json", "export_summary.json", "source_shards.json"):
                shutil.copyfile(work / name_, partial / name_)
            shutil.copyfile(
                dataset / "created_dataset/data.summary.json", partial / "prep_summary.json"
            )
            for report in dataset.glob("*.tsv"):
                shutil.copyfile(report, partial / report.name)
            write_json(
                partial / "complete.json",
                {
                    "dataset": checked,
                    "export": exported,
                    "prep": summary,
                    "seconds": round(time.time() - start),
                    "git_commit": args.git_commit,
                },
            )
            partial.rename(final)
            shutil.rmtree(work)
            completed.append(name)
            total_rows += checked["rows"]
            print(f"[{number}/{len(plan)}] {name}: COMPLETE {checked}", flush=True)
            update(stage="batch_complete")
        update(state="complete", stage="complete")
        print(f"COMPLETE: {len(completed)} batches, {total_rows} training rows", flush=True)
    except BaseException as exc:
        update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--shards-per-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--export-workers", type=int, default=12)
    parser.add_argument("--min-s", type=float, default=0.4)
    parser.add_argument("--max-s", type=float, default=25.0)
    parser.add_argument("--smoke", action="store_true", help="Only the first batch of each subset")
    args = parser.parse_args()
    if min(args.shards_per_batch, args.workers, args.export_workers) < 1:
        parser.error("Batch size and worker counts must be positive")
    if not 0 <= args.min_s < args.max_s:
        parser.error("Require 0 <= min-s < max-s")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s"
    )
    prepare(args)


if __name__ == "__main__":
    main()
