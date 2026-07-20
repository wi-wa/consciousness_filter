#!/usr/bin/env python3
"""Rate the hand-annotated documents with a saved embedding-model checkpoint.

Usage:

    python embedding_model/embedding_handlabel_rater.py \
        embedding_model/checkpoints/run_YYYYMMDD_HHMMSS_utc/best

The positional path must resolve inside ``embedding_model/checkpoints``. It may
name a ``best``/``final`` checkpoint directory, its ``adapter`` directory, or a
run directory (in which case ``best`` is selected). The hand annotations and
live LLM-rated JSONL are always read-only. Hand rows are exact-matched, then
200-character-prefix-matched, to the live rated rows so both kinds of judge
score refer to the same document text. Results atomically replace:

    llm_judge/data/hand_annotated_embedding_ratings.jsonl

For each filter, the script applies sigmoid to every distilled judge head,
averages those probabilities, and multiplies by 10 for the viewer's scale.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = REPO_ROOT / "embedding_model/checkpoints"
HAND_ANNOTATIONS_PATH = REPO_ROOT / "llm_judge/data/hand_annotated_samples.jsonl"
LIVE_RATED_PATH = REPO_ROOT / "llm_judge/data/fineweb_edu_88k_rated.jsonl"
OUTPUT_PATH = REPO_ROOT / "llm_judge/data/hand_annotated_embedding_ratings.jsonl"
PREFIX_MATCH_CHARS = 200


@dataclass(frozen=True)
class TargetLayout:
    filter_names: list[str]
    judge_names: list[str]
    indexes_by_filter: dict[str, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help=(
            "Checkpoint inside embedding_model/checkpoints: a run directory, "
            "best/final directory, or adapter directory."
        ),
    )
    return parser.parse_args()


def read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing {description}: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a JSON object in {description}: {path}")
    return value


def resolve_checkpoint(raw_path: Path) -> Path:
    checkpoint_root = CHECKPOINT_ROOT.resolve()
    candidate = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
    candidate = candidate.resolve()
    try:
        candidate.relative_to(checkpoint_root)
    except ValueError:
        raise SystemExit(
            f"Checkpoint must be inside {checkpoint_root}; received {candidate}."
        ) from None

    if candidate.name == "adapter" and (candidate.parent / "checkpoint.json").is_file():
        candidate = candidate.parent
    elif not (candidate / "checkpoint.json").is_file():
        if (candidate / "best/checkpoint.json").is_file():
            candidate = candidate / "best"
            print(f"Run directory supplied; selected its best checkpoint: {candidate}")
        elif (candidate / "final/checkpoint.json").is_file():
            candidate = candidate / "final"
            print(f"Run directory supplied; selected its final checkpoint: {candidate}")

    required_paths = [
        candidate / "checkpoint.json",
        candidate / "prediction_head.pt",
        candidate / "adapter/adapter_config.json",
        candidate / "adapter/adapter_model.safetensors",
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            f"Not a complete embedding checkpoint: {candidate}\nMissing: "
            + ", ".join(str(path) for path in missing)
        )
    return candidate


def find_run_directory(checkpoint_directory: Path) -> Path:
    for candidate in (checkpoint_directory, *checkpoint_directory.parents):
        if candidate == CHECKPOINT_ROOT.resolve():
            break
        if (candidate / "train_config.json").is_file() and (
            candidate / "metadata.json"
        ).is_file():
            return candidate
    raise SystemExit(
        f"Could not find train_config.json and metadata.json above {checkpoint_directory}."
    )


def load_hand_annotations(path: Path) -> list[dict[str, Any]]:
    try:
        annotation_file = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Hand-annotation file does not exist: {path}") from None

    rows: list[dict[str, Any]] = []
    with annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise SystemExit(f"Missing non-empty 'text' at {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No hand-annotated documents found in {path}.")
    return rows


def load_live_rated_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rated_file = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Live rated file does not exist: {path}") from None

    rows: list[dict[str, Any]] = []
    with rated_file:
        for line_number, raw_line in enumerate(rated_file, start=1):
            if not raw_line.strip():
                raise SystemExit(f"Blank line at {path}:{line_number} would break alignment.")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise SystemExit(f"Missing non-empty 'text' at {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No live rated documents found in {path}.")
    return rows


def sync_annotations_to_live_rows(
    annotations: list[dict[str, Any]], live_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int, int]:
    """Return annotation copies whose text exactly matches the live rated JSONL."""
    exact: dict[str, int] = {}
    prefix: dict[str, int] = {}
    for index, row in enumerate(live_rows):
        text = row["text"]
        exact.setdefault(text, index)
        prefix.setdefault(text[:PREFIX_MATCH_CHARS], index)

    synced: list[dict[str, Any]] = []
    used_live_indexes: set[int] = set()
    exact_matches = 0
    prefix_matches = 0
    unmatched_lines: list[int] = []
    for annotation_line, annotation in enumerate(annotations, start=1):
        annotation_text = annotation["text"]
        live_index = exact.get(annotation_text)
        if live_index is not None:
            exact_matches += 1
        else:
            live_index = prefix.get(annotation_text[:PREFIX_MATCH_CHARS])
            if live_index is not None:
                prefix_matches += 1
        if live_index is None:
            unmatched_lines.append(annotation_line)
            continue
        if live_index in used_live_indexes:
            raise SystemExit(
                f"Multiple hand annotations match live rated line {live_index + 1}; "
                "refusing to create ambiguous embedding ratings."
            )
        used_live_indexes.add(live_index)
        synced_annotation = dict(annotation)
        # Inference and viewer joining both use the exact live-rated document text.
        synced_annotation["text"] = live_rows[live_index]["text"]
        synced.append(synced_annotation)

    if unmatched_lines:
        preview = ", ".join(str(line) for line in unmatched_lines[:10])
        suffix = "..." if len(unmatched_lines) > 10 else ""
        raise SystemExit(
            f"{len(unmatched_lines)} hand annotations did not match the live rated file "
            f"(hand lines {preview}{suffix}). No output was written."
        )
    return synced, exact_matches, prefix_matches


def parse_target_layout(target_names: Any) -> TargetLayout:
    if not isinstance(target_names, list) or not target_names:
        raise SystemExit("checkpoint.json must contain a non-empty target_names list.")

    filter_names: list[str] = []
    indexes_by_filter: dict[str, list[int]] = {}
    judges_by_filter: dict[str, list[str]] = {}
    for index, target_name in enumerate(target_names):
        if not isinstance(target_name, str) or "::" not in target_name:
            raise SystemExit(f"Invalid target name at output index {index}: {target_name!r}")
        filter_name, judge_name = target_name.split("::", 1)
        if not filter_name or not judge_name:
            raise SystemExit(f"Invalid target name at output index {index}: {target_name!r}")
        if filter_name not in indexes_by_filter:
            filter_names.append(filter_name)
            indexes_by_filter[filter_name] = []
            judges_by_filter[filter_name] = []
        indexes_by_filter[filter_name].append(index)
        judges_by_filter[filter_name].append(judge_name)

    judge_names = judges_by_filter[filter_names[0]]
    if len(set(judge_names)) != len(judge_names):
        raise SystemExit("The checkpoint contains duplicate judge heads for a filter.")
    for filter_name in filter_names[1:]:
        if judges_by_filter[filter_name] != judge_names:
            raise SystemExit(
                "Every filter must contain the same judge heads in the same order; "
                f"{filter_name!r} differs."
            )
    return TargetLayout(
        filter_names=filter_names,
        judge_names=judge_names,
        indexes_by_filter=indexes_by_filter,
    )


def choose_device_and_dtype(torch: Any, runtime: dict[str, Any]) -> tuple[Any, Any | None, str]:
    requested_device = str(runtime.get("device", "cuda")).lower()
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "This checkpoint requests CUDA, but CUDA is unavailable. Change the saved "
                "runtime device or run in a CUDA environment."
            )
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        raise SystemExit("Saved runtime.device must be 'cuda' or 'cpu'.")

    precision = str(runtime.get("precision", "auto")).lower()
    if precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device.type == "cuda":
            precision = "fp16"
        else:
            precision = "fp32"
    dtype_by_name = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtype_by_name:
        raise SystemExit("Saved runtime.precision must be auto, fp32, fp16, or bf16.")
    if device.type == "cpu" and precision == "fp16":
        raise SystemExit("fp16 inference on CPU is unsupported; use fp32 or bf16.")
    return device, dtype_by_name[precision], precision


def resolve_saved_path(raw_path: Any, description: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise SystemExit(f"Missing non-empty {description} path in saved checkpoint metadata.")
    path = Path(raw_path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def predict_filter_probabilities(
    checkpoint_directory: Path,
    run_directory: Path,
    annotations: list[dict[str, Any]],
    checkpoint: dict[str, Any],
    train_config: dict[str, Any],
    layout: TargetLayout,
) -> list[dict[str, float]]:
    try:
        import torch
        import torch.nn as nn
        from peft import PeftModel
        from tqdm import tqdm
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Embedding inference dependencies are missing. Install them with "
            "`uv pip install -r requirements.txt`. "
            f"Original import error: {exc}"
        ) from exc

    model_config = train_config.get("model")
    runtime = train_config.get("runtime")
    optimization = train_config.get("optimization")
    if not isinstance(model_config, dict) or not isinstance(runtime, dict):
        raise SystemExit(f"Saved training config is missing model/runtime sections: {run_directory}")
    if not isinstance(optimization, dict):
        raise SystemExit(f"Saved training config is missing optimization: {run_directory}")

    device, model_dtype, precision = choose_device_and_dtype(torch, runtime)
    base_model_path = resolve_saved_path(checkpoint.get("base_model_path"), "base model")
    if not base_model_path.is_dir():
        raise SystemExit(f"Saved base-model directory does not exist: {base_model_path}")
    max_length = model_config.get("max_length")
    batch_size = optimization.get("batch_size")
    if not isinstance(max_length, int) or max_length <= 0:
        raise SystemExit("Saved model.max_length must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise SystemExit("Saved optimization.batch_size must be a positive integer.")

    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": str(model_config.get("attention_implementation", "sdpa")),
    }
    if model_dtype is not None:
        load_kwargs["dtype"] = model_dtype
    print(f"Loading base model on {device} ({precision})")
    base_model = AutoModel.from_pretrained(base_model_path, **load_kwargs)
    encoder = PeftModel.from_pretrained(
        base_model,
        checkpoint_directory / "adapter",
        is_trainable=False,
        local_files_only=True,
    )

    prediction_head_spec = checkpoint.get("prediction_head")
    if not isinstance(prediction_head_spec, dict):
        raise SystemExit("checkpoint.json has no prediction_head specification.")
    input_features = prediction_head_spec.get("input_features")
    output_features = prediction_head_spec.get("output_features")
    use_bias = prediction_head_spec.get("bias")
    if (
        not isinstance(input_features, int)
        or input_features <= 0
        or output_features != sum(len(v) for v in layout.indexes_by_filter.values())
        or not isinstance(use_bias, bool)
    ):
        raise SystemExit("checkpoint.json prediction_head dimensions are inconsistent.")
    if prediction_head_spec.get("pooling") != "token_position_zero":
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

    def autocast_context() -> Any:
        if model_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=model_dtype)

    predictions: list[dict[str, float]] = []
    with torch.inference_mode():
        for offset in tqdm(
            range(0, len(annotations), batch_size),
            desc="Embedding hand rating",
            unit="batch",
        ):
            batch = annotations[offset : offset + batch_size]
            encoded = tokenizer(
                [row["text"] for row in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            with autocast_context():
                hidden = encoder(**encoded).last_hidden_state[:, 0, :]
                logits = prediction_head(hidden)
            probabilities = logits.float().sigmoid()
            for row_index in range(probabilities.shape[0]):
                predictions.append(
                    {
                        filter_name: float(
                            probabilities[
                                row_index, layout.indexes_by_filter[filter_name]
                            ].mean().item()
                        )
                        for filter_name in layout.filter_names
                    }
                )
    return predictions


def build_output_rows(
    annotations: list[dict[str, Any]],
    predictions: list[dict[str, float]],
    layout: TargetLayout,
    model_name: str,
) -> list[dict[str, Any]]:
    if len(annotations) != len(predictions):
        raise ValueError("Every hand annotation must have exactly one prediction row.")

    rows: list[dict[str, Any]] = []
    for annotation, probabilities in zip(annotations, predictions, strict=True):
        row = dict(annotation)
        row["ratings"] = {
            filter_name: [
                {
                    "model": model_name,
                    "rating": round(probabilities[filter_name] * 10.0, 6),
                    "explanation": (
                        f"Embedding probability {probabilities[filter_name] * 100.0:.2f}% "
                        f"(mean of {len(layout.judge_names)} distilled judge heads)."
                    ),
                    "quote": "",
                }
            ]
            for filter_name in layout.filter_names
        }
        rows.append(row)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=path.name + ".",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        for row in rows:
            temporary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    checkpoint_directory = resolve_checkpoint(args.checkpoint)
    run_directory = find_run_directory(checkpoint_directory)
    checkpoint = read_json_object(
        checkpoint_directory / "checkpoint.json", "checkpoint metadata"
    )
    train_config = read_json_object(run_directory / "train_config.json", "training config")
    layout = parse_target_layout(checkpoint.get("target_names"))
    annotations = load_hand_annotations(HAND_ANNOTATIONS_PATH)
    live_rows = load_live_rated_rows(LIVE_RATED_PATH)
    synced_annotations, exact_matches, prefix_matches = sync_annotations_to_live_rows(
        annotations, live_rows
    )
    relative_checkpoint = checkpoint_directory.relative_to(CHECKPOINT_ROOT.resolve())
    model_name = f"embedding::{relative_checkpoint.as_posix()}"

    print(f"Checkpoint: {checkpoint_directory}")
    print(f"Viewer judge: {model_name}")
    print(f"Hand annotations (read only): {HAND_ANNOTATIONS_PATH}")
    print(f"Live LLM ratings (read only): {LIVE_RATED_PATH}")
    print(
        f"Synced hand documents: {len(synced_annotations)} "
        f"({exact_matches} exact, {prefix_matches} prefix)"
    )
    print(
        f"Outputs: {len(layout.filter_names)} filters × "
        f"mean of {len(layout.judge_names)} judge heads"
    )
    predictions = predict_filter_probabilities(
        checkpoint_directory,
        run_directory,
        synced_annotations,
        checkpoint,
        train_config,
        layout,
    )
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for prediction in predictions
        for probability in prediction.values()
    ):
        raise SystemExit("Model produced a non-finite or out-of-range probability.")
    output_rows = build_output_rows(
        synced_annotations, predictions, layout, model_name
    )
    atomic_write_jsonl(OUTPUT_PATH, output_rows)
    print(f"Wrote {len(output_rows)} embedding-rated rows atomically to {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
