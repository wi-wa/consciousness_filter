#!/usr/bin/env python3
"""Tag the full FineWeb-Edu corpus with every embedding-model output head.

Example:

    python full_pretrain_filter/tag_with_embedding_model.py

The model name and batch size come from ``filter_settings`` in
``full_pretrain_filter/config.json``. The configured model resolves only
beneath ``embedding_model/checkpoints``. When it is a run directory, its
``best`` checkpoint is preferred, falling back to ``final``. Each input row is
preserved and receives a nested ``tags`` mapping of
``filter -> judge -> probability``.

Documents are globally sorted by their stored token count, divided into
length-homogeneous batches, and those batches are deterministically shuffled.
Predictions are spooled by source-row index so the output retains input order.

Output is written alongside the input as:

    fineweb_edu_{document_count}_{model_name}.jsonl

An interrupted run leaves a ``.partial`` file that can be continued with
``--resume``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import sys
import time
from array import array
from collections import deque
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from embedding_model.embedding_handlabel_rater import (  # noqa: E402
    CHECKPOINT_ROOT,
    TargetLayout,
    choose_device_and_dtype,
    find_run_directory,
    parse_target_layout,
    read_json_object,
    resolve_checkpoint,
    resolve_saved_path,
)


DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
INFERENCE_ATTENTION_IMPLEMENTATION = "flex_attention"


@dataclass(frozen=True)
class FilterConfig:
    input_jsonl: Path
    model_name: str
    batch_size: int
    batch_shuffle_seed: int


@dataclass(frozen=True)
class DocumentIndex:
    offsets: array
    token_counts: array

    def __len__(self) -> int:
        return len(self.offsets)


@dataclass(frozen=True)
class InferenceModel:
    torch: Any
    tokenizer: Any
    encoder: Any
    prediction_head: Any
    device: Any
    model_dtype: Any | None
    max_length: int
    batch_size: int
    target_names: tuple[str, ...]
    layout: TargetLayout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Full-pretrain config (default: {DEFAULT_CONFIG})",
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing .partial output after validating its alignment.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Start again and atomically replace an existing output after completion.",
    )
    return parser.parse_args(argv)


def validate_model_name(model_name: str) -> str:
    if not MODEL_NAME_RE.fullmatch(model_name):
        raise SystemExit(
            "config.filter_settings.model must be one top-level checkpoint "
            "directory name containing only letters, numbers, '.', '_', or '-'."
        )
    return model_name


def load_config(path: Path) -> FilterConfig:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError:
        raise SystemExit(f"Config does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")

    download_settings = raw_config.get("download_settings")
    filter_settings = raw_config.get("filter_settings")
    if not isinstance(download_settings, dict):
        raise SystemExit("config.download_settings must be an object.")
    if not isinstance(filter_settings, dict):
        raise SystemExit("config.filter_settings must be an object.")

    raw_input = download_settings.get("output_jsonl")
    if not isinstance(raw_input, str) or not raw_input:
        raise SystemExit(
            "config.download_settings.output_jsonl must be a non-empty path string."
        )
    input_jsonl = Path(raw_input)
    if not input_jsonl.is_absolute():
        input_jsonl = REPO_ROOT / input_jsonl

    raw_model_name = filter_settings.get("model")
    if not isinstance(raw_model_name, str):
        raise SystemExit("config.filter_settings.model must be a string.")
    model_name = validate_model_name(raw_model_name)
    batch_size = filter_settings.get("batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise SystemExit(
            "config.filter_settings.batch_size must be an integer greater than zero."
        )
    batch_shuffle_seed = filter_settings.get("batch_shuffle_seed", 42)
    if not isinstance(batch_shuffle_seed, int) or isinstance(
        batch_shuffle_seed, bool
    ):
        raise SystemExit(
            "config.filter_settings.batch_shuffle_seed must be an integer."
        )
    return FilterConfig(
        input_jsonl=input_jsonl,
        model_name=model_name,
        batch_size=batch_size,
        batch_shuffle_seed=batch_shuffle_seed,
    )


def checkpoint_for_model(model_name: str) -> Path:
    return resolve_checkpoint(CHECKPOINT_ROOT / model_name)


def index_documents(path: Path) -> DocumentIndex:
    try:
        input_file = path.open("rb")
    except FileNotFoundError:
        raise SystemExit(f"Input corpus does not exist: {path}") from None

    offsets = array("Q")
    token_counts = array("I")
    file_size = path.stat().st_size
    with input_file, tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc="Indexing document lengths",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as progress:
        pending_bytes = 0
        line_number = 0
        while True:
            offset = input_file.tell()
            line = input_file.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                raise SystemExit(f"Blank input row at {path}:{line_number}.")
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            token_count = row.get("token_count") if isinstance(row, dict) else None
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count <= 0
                or token_count > 0xFFFFFFFF
            ):
                raise SystemExit(
                    f"{path}:{line_number} has no positive integer 'token_count'."
                )
            offsets.append(offset)
            token_counts.append(token_count)
            pending_bytes += len(line)
            if line_number % 4096 == 0:
                progress.update(pending_bytes)
                pending_bytes = 0
        progress.update(pending_bytes)
    if not offsets:
        raise SystemExit(f"Input corpus is empty: {path}")
    return DocumentIndex(offsets=offsets, token_counts=token_counts)


def output_path_for(
    input_path: Path, model_name: str, document_count: int
) -> Path:
    return input_path.parent / f"fineweb_edu_{document_count}_{model_name}.jsonl"


def parse_input_row(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not isinstance(row, dict):
        raise SystemExit(f"{path}:{line_number} is not a JSON object.")
    if not isinstance(row.get("text"), str) or not row["text"]:
        raise SystemExit(f"{path}:{line_number} has no non-empty string 'text'.")
    if "tags" in row:
        raise SystemExit(
            f"{path}:{line_number} already has a 'tags' field; refusing to replace it."
        )
    return row


def expected_tag_keys(layout: TargetLayout) -> dict[str, set[str]]:
    return {
        filter_name: set(layout.judge_names) for filter_name in layout.filter_names
    }


def validate_tags(
    tags: Any,
    expected_keys: dict[str, set[str]],
    path: Path,
    line_number: int,
) -> None:
    if not isinstance(tags, dict) or set(tags) != set(expected_keys):
        raise SystemExit(f"Invalid filter tags at {path}:{line_number}.")
    for filter_name, judge_names in expected_keys.items():
        judge_tags = tags.get(filter_name)
        if not isinstance(judge_tags, dict) or set(judge_tags) != judge_names:
            raise SystemExit(
                f"Invalid judge tags for {filter_name!r} at {path}:{line_number}."
            )
        for probability in judge_tags.values():
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not math.isfinite(float(probability))
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise SystemExit(
                    f"Invalid tag probability at {path}:{line_number}: "
                    f"{probability!r}."
                )


def validate_partial_alignment(
    input_path: Path,
    partial_path: Path,
    layout: TargetLayout,
) -> int:
    processed = 0
    expected_keys = expected_tag_keys(layout)
    with input_path.open("r", encoding="utf-8") as input_file, partial_path.open(
        "r", encoding="utf-8"
    ) as partial_file:
        for line_number, partial_line in enumerate(partial_file, start=1):
            input_line = input_file.readline()
            if not input_line:
                raise SystemExit(
                    f"Partial output has more rows than its input: {partial_path}."
                )
            source_row = parse_input_row(input_path, line_number, input_line)
            try:
                output_row = json.loads(partial_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON at {partial_path}:{line_number}: {exc}. "
                    "Restart with --overwrite."
                ) from exc
            if not isinstance(output_row, dict):
                raise SystemExit(f"{partial_path}:{line_number} is not a JSON object.")
            validate_tags(
                output_row.get("tags"), expected_keys, partial_path, line_number
            )
            untagged_output = dict(output_row)
            untagged_output.pop("tags")
            if untagged_output != source_row:
                raise SystemExit(
                    f"Partial output is not aligned with the input at line "
                    f"{line_number}; restart with --overwrite."
                )
            processed += 1
    return processed


def make_length_batch_plan(
    document_index: DocumentIndex,
    *,
    first_row: int,
    batch_size: int,
    shuffle_seed: int,
) -> tuple[list[int], list[int]]:
    sorted_rows = list(range(first_row, len(document_index)))
    sorted_rows.sort(key=document_index.token_counts.__getitem__)
    batch_count = math.ceil(len(sorted_rows) / batch_size)
    batch_order = list(range(batch_count))
    random.Random(shuffle_seed).shuffle(batch_order)
    return sorted_rows, batch_order


def rows_for_batch(
    input_file: Any,
    input_path: Path,
    document_index: DocumentIndex,
    row_indexes: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Source-order reads reduce seek distance; order within a batch has no effect.
    row_indexes.sort()
    for row_index in row_indexes:
        input_file.seek(document_index.offsets[row_index])
        raw_line = input_file.readline()
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(
                f"Invalid UTF-8 at {input_path}:{row_index + 1}: {exc}"
            ) from exc
        rows.append(parse_input_row(input_path, row_index + 1, line))
    return rows


def load_inference_model(
    checkpoint_directory: Path,
    batch_size: int,
) -> InferenceModel:
    try:
        import torch
        import torch.nn as nn
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Embedding inference dependencies are missing. Install requirements.txt. "
            f"Original import error: {exc}"
        ) from exc

    run_directory = find_run_directory(checkpoint_directory)
    checkpoint = read_json_object(
        checkpoint_directory / "checkpoint.json", "checkpoint metadata"
    )
    train_config = read_json_object(
        run_directory / "train_config.json", "training config"
    )
    raw_target_names = checkpoint.get("target_names")
    layout = parse_target_layout(raw_target_names)
    if not isinstance(raw_target_names, list):
        raise SystemExit("checkpoint.json target_names must be a list.")
    target_names = tuple(raw_target_names)

    model_config = train_config.get("model")
    runtime = train_config.get("runtime")
    if not isinstance(model_config, dict) or not isinstance(runtime, dict):
        raise SystemExit(
            f"Saved training config is missing model/runtime sections: {run_directory}"
        )

    max_length = model_config.get("max_length")
    if not isinstance(max_length, int) or max_length <= 0:
        raise SystemExit("Saved model.max_length must be a positive integer.")
    if batch_size <= 0:
        raise SystemExit(
            "config.filter_settings.batch_size must be greater than zero."
        )

    device, model_dtype, precision = choose_device_and_dtype(torch, runtime)
    base_model_path = resolve_saved_path(
        checkpoint.get("base_model_path"), "base model"
    )
    if not base_model_path.is_dir():
        raise SystemExit(
            f"Saved base-model directory does not exist: {base_model_path}"
        )

    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": INFERENCE_ATTENTION_IMPLEMENTATION,
    }
    if model_dtype is not None:
        load_kwargs["dtype"] = model_dtype
    print(
        f"Loading embedding model on {device} ({precision}, "
        f"attention={INFERENCE_ATTENTION_IMPLEMENTATION})"
    )
    base_model = AutoModel.from_pretrained(base_model_path, **load_kwargs)
    encoder = PeftModel.from_pretrained(
        base_model,
        checkpoint_directory / "adapter",
        is_trainable=False,
        local_files_only=True,
    )

    head_spec = checkpoint.get("prediction_head")
    if not isinstance(head_spec, dict):
        raise SystemExit("checkpoint.json has no prediction_head specification.")
    input_features = head_spec.get("input_features")
    output_features = head_spec.get("output_features")
    use_bias = head_spec.get("bias")
    if (
        not isinstance(input_features, int)
        or input_features <= 0
        or output_features != len(target_names)
        or not isinstance(use_bias, bool)
    ):
        raise SystemExit("checkpoint.json prediction_head dimensions are inconsistent.")
    if head_spec.get("pooling") != "token_position_zero":
        raise SystemExit("Only token_position_zero checkpoints are supported.")

    prediction_head = nn.Linear(input_features, output_features, bias=use_bias)
    try:
        head_state = torch.load(
            checkpoint_directory / "prediction_head.pt",
            map_location="cpu",
            weights_only=True,
        )
        prediction_head.load_state_dict(head_state, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Could not load the prediction head: {exc}") from exc

    encoder.to(device)
    prediction_head.to(device)
    encoder.eval()
    prediction_head.eval()
    print(
        f"Loaded {len(layout.filter_names)} filters × {len(layout.judge_names)} "
        f"judges ({len(target_names)} probability heads); batch size {batch_size}."
    )
    return InferenceModel(
        torch=torch,
        tokenizer=tokenizer,
        encoder=encoder,
        prediction_head=prediction_head,
        device=device,
        model_dtype=model_dtype,
        max_length=max_length,
        batch_size=batch_size,
        target_names=target_names,
        layout=layout,
    )


def autocast_context(model: InferenceModel) -> Any:
    if model.model_dtype is None:
        return nullcontext()
    return model.torch.autocast(
        device_type=model.device.type, dtype=model.model_dtype
    )


def build_tags(
    target_names: tuple[str, ...], probabilities: Sequence[float]
) -> dict[str, dict[str, float]]:
    if len(target_names) != len(probabilities):
        raise ValueError("Probability count does not match checkpoint target names.")
    tags: dict[str, dict[str, float]] = {}
    for target_name, probability in zip(target_names, probabilities, strict=True):
        filter_name, judge_name = target_name.split("::", 1)
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"Model produced invalid probability {value!r}.")
        tags.setdefault(filter_name, {})[judge_name] = value
    return tags


def predict_batch(
    model: InferenceModel, rows: list[dict[str, Any]]
) -> list[list[float]]:
    encoded = model.tokenizer(
        [row["text"] for row in rows],
        padding=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(model.device, non_blocking=True)
        for key, value in encoded.items()
    }
    with model.torch.inference_mode(), autocast_context(model):
        hidden = model.encoder(**encoded).last_hidden_state[:, 0, :]
        probabilities = model.prediction_head(hidden).float().sigmoid().cpu().tolist()
    return probabilities


def prediction_artifact_paths(partial_path: Path) -> tuple[Path, Path]:
    prediction_path = partial_path.with_name(partial_path.name + ".predictions.f32")
    state_path = partial_path.with_name(partial_path.name + ".state.json")
    return prediction_path, state_path


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def initialize_prediction_state(
    *,
    input_path: Path,
    partial_path: Path,
    document_count: int,
    materialized_rows: int,
    model_name: str,
    model: InferenceModel,
    shuffle_seed: int,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, Path, dict[str, Any]]:
    prediction_path, state_path = prediction_artifact_paths(partial_path)
    if overwrite:
        prediction_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    artifacts_exist = prediction_path.exists(), state_path.exists()
    if any(artifacts_exist) and not all(artifacts_exist):
        raise SystemExit(
            "The length-batching resume artifacts are incomplete. Restart with "
            "--overwrite."
        )

    record_size = 4 * len(model.target_names)
    if all(artifacts_exist):
        if not resume and not overwrite:
            raise SystemExit(
                "Prediction resume artifacts already exist. Use --resume to "
                "continue or --overwrite to start again."
            )
        try:
            with state_path.open("r", encoding="utf-8") as state_file:
                state = json.load(state_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Could not read prediction resume state {state_path}: {exc}. "
                "Restart with --overwrite."
            ) from exc
        if not isinstance(state, dict):
            raise SystemExit(
                f"Invalid prediction resume state {state_path}; restart with --overwrite."
            )
    else:
        if resume and not partial_path.exists():
            raise SystemExit(f"Partial output does not exist: {partial_path}")
        stat = input_path.stat()
        prediction_first_row = materialized_rows
        total_batches = math.ceil(
            (document_count - prediction_first_row) / model.batch_size
        )
        state = {
            "version": 1,
            "input_path": str(input_path.resolve()),
            "input_size": stat.st_size,
            "input_mtime_ns": stat.st_mtime_ns,
            "document_count": document_count,
            "model_name": model_name,
            "batch_size": model.batch_size,
            "batch_shuffle_seed": shuffle_seed,
            "target_names": list(model.target_names),
            "prediction_first_row": prediction_first_row,
            "total_batches": total_batches,
            "completed_batches": 0,
        }
        with prediction_path.open("wb") as prediction_file:
            prediction_file.truncate(document_count * record_size)
            prediction_file.flush()
            os.fsync(prediction_file.fileno())
        write_json_atomic(state_path, state)

    stat = input_path.stat()
    expected = {
        "version": 1,
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "document_count": document_count,
        "model_name": model_name,
        "batch_size": model.batch_size,
        "batch_shuffle_seed": shuffle_seed,
        "target_names": list(model.target_names),
    }
    mismatches = [key for key, value in expected.items() if state.get(key) != value]
    if mismatches:
        raise SystemExit(
            "Prediction resume state does not match this run ("
            + ", ".join(mismatches)
            + "); restart with --overwrite."
        )
    prediction_first_row = state.get("prediction_first_row")
    completed_batches = state.get("completed_batches")
    total_batches = state.get("total_batches")
    if (
        not isinstance(prediction_first_row, int)
        or not 0 <= prediction_first_row <= materialized_rows
        or not isinstance(completed_batches, int)
        or not isinstance(total_batches, int)
        or not 0 <= completed_batches <= total_batches
        or total_batches
        != math.ceil((document_count - prediction_first_row) / model.batch_size)
    ):
        raise SystemExit(
            f"Invalid progress in {state_path}; restart with --overwrite."
        )
    if prediction_path.stat().st_size != document_count * record_size:
        raise SystemExit(
            f"Prediction spool has the wrong size: {prediction_path}. Restart "
            "with --overwrite."
        )
    return prediction_path, state_path, state


def write_prediction_batch(
    prediction_file: Any,
    row_indexes: list[int],
    probabilities: list[list[float]],
    record: struct.Struct,
) -> None:
    if len(row_indexes) != len(probabilities):
        raise ValueError("Prediction count does not match batch size.")
    for row_index, values in sorted(
        zip(row_indexes, probabilities, strict=True), key=lambda item: item[0]
    ):
        try:
            packed = record.pack(*values)
        except struct.error as exc:
            raise SystemExit(f"Invalid embedding-model prediction shape: {exc}") from exc
        prediction_file.seek(row_index * record.size)
        prediction_file.write(packed)
    prediction_file.flush()
    os.fsync(prediction_file.fileno())


def prepare_output(
    input_path: Path,
    output_path: Path,
    *,
    resume: bool,
    overwrite: bool,
    layout: TargetLayout,
) -> tuple[Path, int, str]:
    partial_path = output_path.with_name(output_path.name + ".partial")
    if resume:
        if output_path.exists():
            raise SystemExit(f"Completed output already exists: {output_path}")
        if not partial_path.exists():
            raise SystemExit(f"Partial output does not exist: {partial_path}")
        processed = validate_partial_alignment(input_path, partial_path, layout)
        return partial_path, processed, "a"

    if output_path.exists() and not overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if partial_path.exists() and not overwrite:
        raise SystemExit(
            f"Partial output already exists: {partial_path}. Use --resume to "
            "continue or --overwrite to start again."
        )
    return partial_path, 0, "w"


def tag_corpus(
    input_path: Path,
    model_name: str,
    model: InferenceModel,
    document_index: DocumentIndex,
    shuffle_seed: int,
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    document_count = len(document_index)
    output_path = output_path_for(input_path, model_name, document_count)
    partial_path, processed, mode = prepare_output(
        input_path,
        output_path,
        resume=resume,
        overwrite=overwrite,
        layout=model.layout,
    )
    if processed > document_count:
        raise SystemExit("Partial output contains more rows than the input corpus.")
    if processed == document_count:
        os.replace(partial_path, output_path)
        prediction_path, state_path = prediction_artifact_paths(partial_path)
        prediction_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        print(f"Finalized already-complete partial output: {output_path}")
        return output_path

    # Establish/truncate the materialized prefix before creating resume state.
    with partial_path.open(mode, encoding="utf-8", newline="\n"):
        pass
    prediction_path, state_path, state = initialize_prediction_state(
        input_path=input_path,
        partial_path=partial_path,
        document_count=document_count,
        materialized_rows=processed,
        model_name=model_name,
        model=model,
        shuffle_seed=shuffle_seed,
        resume=resume,
        overwrite=overwrite,
    )

    prediction_first_row = int(state["prediction_first_row"])
    sorted_rows, batch_order = make_length_batch_plan(
        document_index,
        first_row=prediction_first_row,
        batch_size=model.batch_size,
        shuffle_seed=shuffle_seed,
    )
    completed_batches = int(state["completed_batches"])
    print(
        f"Length batching: {len(batch_order):,} globally sorted batches, "
        f"shuffled with seed {shuffle_seed}."
    )

    def batch_bounds(batch_number: int) -> tuple[int, int]:
        start = batch_number * model.batch_size
        return start, min(start + model.batch_size, len(sorted_rows))

    completed_documents = sum(
        batch_bounds(batch_number)[1] - batch_bounds(batch_number)[0]
        for batch_number in batch_order[:completed_batches]
    )
    record = struct.Struct(f"<{len(model.target_names)}f")
    recent_batch_seconds: deque[float] = deque(maxlen=16)
    with input_path.open("rb") as input_file, prediction_path.open(
        "r+b"
    ) as prediction_file:

        def run_batch(plan_position: int) -> tuple[int, float]:
            started_at = time.perf_counter()
            batch_number = batch_order[plan_position]
            start, end = batch_bounds(batch_number)
            row_indexes = sorted_rows[start:end]
            rows = rows_for_batch(
                input_file, input_path, document_index, row_indexes
            )
            probabilities = predict_batch(model, rows)
            write_prediction_batch(
                prediction_file, row_indexes, probabilities, record
            )
            elapsed = time.perf_counter() - started_at
            state["completed_batches"] = plan_position + 1
            write_json_atomic(state_path, state)
            return len(rows), elapsed

        # FlexAttention compiles its kernels on the first forward in each process.
        # Use a real batch and persist its predictions, but exclude compilation from
        # the steady-state timing statistics.
        if completed_batches < len(batch_order):
            print("Compiling FlexAttention kernels with one warm-up batch...")
            warmup_documents, warmup_seconds = run_batch(completed_batches)
            completed_batches += 1
            completed_documents += warmup_documents
            print(
                f"FlexAttention warm-up finished in {warmup_seconds:.2f}s; "
                "starting timed batches."
            )

        with tqdm(
            total=len(sorted_rows),
            initial=completed_documents,
            unit="doc",
            unit_scale=True,
            desc=f"Length-batched tags ({model_name})",
            dynamic_ncols=True,
            smoothing=0.05,
            mininterval=0.5,
        ) as progress:
            for plan_position in range(completed_batches, len(batch_order)):
                batch_documents, elapsed = run_batch(plan_position)
                recent_batch_seconds.append(elapsed)
                progress.update(batch_documents)
                progress.set_postfix_str(
                    f"s/batch={elapsed:.3f}, "
                    f"ma16bps={sum(recent_batch_seconds) / len(recent_batch_seconds):.3f}"
                )

    # Predictions were computed out of order; materialize them in source order so
    # the final JSONL remains line-for-line aligned with the input corpus.
    written = processed
    with input_path.open("r", encoding="utf-8") as input_file, prediction_path.open(
        "rb"
    ) as prediction_file, partial_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as output_file, tqdm(
        total=document_count,
        initial=processed,
        unit="doc",
        unit_scale=True,
        desc="Writing tagged JSONL",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as progress:
        for row_index, line in enumerate(input_file):
            if row_index < processed:
                continue
            row = parse_input_row(input_path, row_index + 1, line)
            prediction_file.seek(row_index * record.size)
            packed = prediction_file.read(record.size)
            if len(packed) != record.size:
                raise SystemExit(
                    f"Missing spooled prediction for input row {row_index + 1}."
                )
            tags = build_tags(model.target_names, record.unpack(packed))
            output_row = dict(row)
            output_row["tags"] = tags
            output_file.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
            progress.update()
            if written % model.batch_size == 0:
                output_file.flush()

    if written != document_count:
        raise SystemExit(
            f"Input document count changed during inference: expected "
            f"{document_count:,}, wrote {written:,}. Partial output remains at "
            f"{partial_path}."
        )
    os.replace(partial_path, output_path)
    prediction_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    print(f"Wrote {document_count:,} tagged documents to {output_path}")
    return output_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    checkpoint_directory = checkpoint_for_model(config.model_name)
    document_index = index_documents(config.input_jsonl)
    document_count = len(document_index)
    output_path = output_path_for(
        config.input_jsonl, config.model_name, document_count
    )
    print(f"Input: {config.input_jsonl} ({document_count:,} documents)")
    print(f"Checkpoint: {checkpoint_directory}")
    print(f"Output: {output_path}")

    # Resolve output conflicts before allocating model memory.
    partial_path = output_path.with_name(output_path.name + ".partial")
    if args.resume and output_path.exists():
        raise SystemExit(f"Completed output already exists: {output_path}")
    if args.resume and not partial_path.exists():
        raise SystemExit(f"Partial output does not exist: {partial_path}")
    if not args.resume and not args.overwrite:
        if output_path.exists():
            raise SystemExit(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        if partial_path.exists():
            raise SystemExit(
                f"Partial output already exists: {partial_path}. Use --resume to "
                "continue or --overwrite to start again."
            )

    model = load_inference_model(checkpoint_directory, config.batch_size)
    tag_corpus(
        config.input_jsonl,
        config.model_name,
        model,
        document_index,
        config.batch_shuffle_seed,
        resume=args.resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
