#!/usr/bin/env python3
"""Distill the configured LLM judges into a LoRA-tuned ModernBERT model.

The script intentionally keeps the data and training loop small and explicit:

* targets are the configured (filter, judge) ratings divided by 10;
* hand-annotated documents are validation-only, but their LLM ratings are used;
* another uniform random fraction of the rated corpus is held out;
* ModernBERT is frozen except for LoRA adapters on its embedding, attention,
  and MLP matrices;
* a zero-initialized linear head reads token position zero;
* logs, adapter weights, head weights, and target metadata are written beneath
  embedding_model/checkpoints (or the configured checkpoint directory).

This file does not train anything merely by being imported. Run it explicitly:

    python embedding_model/train.py

Use --validate-data-only to inspect the split without importing torch or loading
the tokenizer/model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "embedding_model/configs/train.json"

HUMAN_LABEL_FIELDS = {
    "philosophy_of_mind": ("pom-rating", "PoM"),
    "reified_experience": ("reification-rating", "Reification"),
    "experience_descriptions": ("experience-rating", "Experience"),
}
UPSAMPLE_FILTER_ALIASES = {
    "pom": "philosophy_of_mind",
    "reification": "reified_experience",
    "experience": "experience_descriptions",
}
UPSAMPLE_NORMALIZED_RATING_THRESHOLD = 0.2


@dataclass(frozen=True)
class Example:
    text: str
    targets: tuple[float, ...]
    source_line: int


@dataclass(frozen=True)
class HandAnnotation:
    text: str
    labels: tuple[int | None, ...]


@dataclass(frozen=True)
class PreparedData:
    train: list[Example]
    validation: list[Example]
    target_names: list[str]
    filter_names: list[str]
    judge_names: list[str]
    rated_rows: int
    incomplete_rows: int
    matched_hand_rows: int
    unmatched_hand_rows: int
    random_validation_rows: int
    human_labels_by_source_line: dict[int, tuple[int | None, ...]] = field(
        default_factory=dict
    )
    unique_training_examples: int = 0
    upsampled_source_rows: int = 0
    upsample_weights: dict[str, int] = field(default_factory=dict)


@dataclass
class AccuracyTally:
    overall_total: int = 0
    overall_score_sum: float = 0.0
    positive_total: int = 0
    positive_score_sum: float = 0.0
    negative_total: int = 0
    negative_score_sum: float = 0.0


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    mae: float
    human_accuracy: dict[str, AccuracyTally] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Training config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate targets and print the split without loading ML libraries.",
    )
    return parser.parse_args()


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SystemExit(f"Missing {key!r} in {context}.")
    return mapping[key]


def resolve_path(value: str, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"Expected a non-empty path string for {context}.")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        raise SystemExit(f"Training config does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"Training config must be a JSON object: {path}")
    for section in ("data", "model", "lora", "optimization", "logging", "runtime"):
        if not isinstance(config.get(section), dict):
            raise SystemExit(f"Training config section {section!r} must be an object.")
    return config


def load_target_layout(rating_config_path: Path) -> tuple[list[str], list[str], list[str]]:
    try:
        with rating_config_path.open("r", encoding="utf-8") as rating_config_file:
            rating_config = json.load(rating_config_file)
    except FileNotFoundError:
        raise SystemExit(f"Rating config does not exist: {rating_config_path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {rating_config_path}: {exc}") from exc

    raw_filters = require(rating_config, "filters", str(rating_config_path))
    raw_judges = require(rating_config, "models", str(rating_config_path))
    if not isinstance(raw_filters, list) or not raw_filters:
        raise SystemExit("Rating config 'filters' must be a non-empty list.")
    if not isinstance(raw_judges, list) or not raw_judges:
        raise SystemExit("Rating config 'models' must be a non-empty list.")

    filter_names: list[str] = []
    for item in raw_filters:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise SystemExit("Every rating config filter must have a string 'name'.")
        filter_names.append(item["name"])
    if not all(isinstance(judge, str) and judge for judge in raw_judges):
        raise SystemExit("Every rating config model must be a non-empty string.")
    judge_names = list(raw_judges)
    if len(set(filter_names)) != len(filter_names):
        raise SystemExit("Rating config contains duplicate filter names.")
    if len(set(judge_names)) != len(judge_names):
        raise SystemExit("Rating config contains duplicate model names.")

    target_names = [
        f"{filter_name}::{judge_name}"
        for filter_name in filter_names
        for judge_name in judge_names
    ]
    return filter_names, judge_names, target_names


def extract_targets(
    row: dict[str, Any], filter_names: list[str], judge_names: list[str]
) -> tuple[float, ...] | None:
    ratings = row.get("ratings")
    if not isinstance(ratings, dict):
        return None

    targets: list[float] = []
    for filter_name in filter_names:
        entries = ratings.get(filter_name)
        if not isinstance(entries, list):
            return None
        by_judge: dict[str, float] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
                continue
            rating = entry.get("rating")
            if (
                isinstance(rating, (int, float))
                and not isinstance(rating, bool)
                and math.isfinite(float(rating))
                and 0.0 <= float(rating) <= 10.0
            ):
                by_judge[entry["model"]] = float(rating) / 10.0
        for judge_name in judge_names:
            if judge_name not in by_judge:
                return None
            targets.append(by_judge[judge_name])
    return tuple(targets)


def load_upsample_weights(
    data_config: dict[str, Any], filter_names: list[str]
) -> dict[str, int]:
    """Load per-filter integer multiplicities, accepting configured short aliases."""
    raw_weights = data_config.get("upsample_mult")
    if raw_weights is None:
        return {filter_name: 1 for filter_name in filter_names}
    if not isinstance(raw_weights, dict):
        raise SystemExit("data.upsample_mult must be an object of filter: integer weight.")

    known_filters = set(filter_names)
    weights: dict[str, int] = {}
    for configured_name, weight in raw_weights.items():
        if not isinstance(configured_name, str) or not configured_name:
            raise SystemExit("Every data.upsample_mult key must be a non-empty string.")
        filter_name = UPSAMPLE_FILTER_ALIASES.get(configured_name, configured_name)
        if filter_name not in known_filters:
            raise SystemExit(
                f"Unknown filter {configured_name!r} in data.upsample_mult. "
                f"Configured filters: {', '.join(filter_names)}"
            )
        if filter_name in weights:
            raise SystemExit(
                f"data.upsample_mult specifies filter {filter_name!r} more than once."
            )
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise SystemExit(
                f"data.upsample_mult[{configured_name!r}] must be an integer >= 1."
            )
        weights[filter_name] = weight

    missing_filters = [name for name in filter_names if name not in weights]
    if missing_filters:
        raise SystemExit(
            "data.upsample_mult must include every configured filter; missing: "
            + ", ".join(missing_filters)
        )
    return weights


def example_upsample_weight(
    example: Example,
    filter_names: list[str],
    judge_names: list[str],
    upsample_weights: dict[str, int],
) -> int:
    """Return total training multiplicity, using the max weight of qualifying filters."""
    judge_count = len(judge_names)
    expected_target_count = len(filter_names) * judge_count
    if judge_count == 0 or len(example.targets) != expected_target_count:
        raise ValueError("Example targets do not match the configured filter/judge layout.")

    multiplicity = 1
    for filter_index, filter_name in enumerate(filter_names):
        start = filter_index * judge_count
        mean_rating = sum(example.targets[start : start + judge_count]) / judge_count
        if mean_rating >= UPSAMPLE_NORMALIZED_RATING_THRESHOLD:
            multiplicity = max(multiplicity, upsample_weights[filter_name])
    return multiplicity


def upsample_training_examples(
    examples: list[Example],
    filter_names: list[str],
    judge_names: list[str],
    upsample_weights: dict[str, int],
) -> tuple[list[Example], int]:
    """Repeat each qualifying row to its total multiplicity; return rows affected too."""
    result: list[Example] = []
    upsampled_source_rows = 0
    for example in examples:
        multiplicity = example_upsample_weight(
            example,
            filter_names,
            judge_names,
            upsample_weights,
        )
        result.extend([example] * multiplicity)
        upsampled_source_rows += int(multiplicity > 1)
    return result, upsampled_source_rows


def read_hand_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        hand_file = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Hand-annotation file does not exist: {path}") from None
    with hand_file:
        for line_number, line in enumerate(hand_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise SystemExit(f"Missing non-empty 'text' at {path}:{line_number}.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No hand-annotated documents found in {path}.")
    return rows


def read_hand_texts(path: Path) -> list[str]:
    return [row["text"] for row in read_hand_rows(path)]


def read_hand_annotations(path: Path, filter_names: list[str]) -> list[HandAnnotation]:
    annotations: list[HandAnnotation] = []
    for row in read_hand_rows(path):
        labels: list[int | None] = []
        for filter_name in filter_names:
            field_spec = HUMAN_LABEL_FIELDS.get(filter_name)
            raw_label = None if field_spec is None else row.get(field_spec[0])
            if (
                isinstance(raw_label, int)
                and not isinstance(raw_label, bool)
                and raw_label >= 0
            ):
                labels.append(raw_label)
            else:
                # Match the viewer: malformed, negative, and absent labels are ignored.
                labels.append(None)
        annotations.append(HandAnnotation(text=row["text"], labels=tuple(labels)))
    return annotations


def match_hand_rows(
    hand_texts: list[str], rated_texts: list[str], prefix_chars: int
) -> tuple[set[int], int]:
    """Match hand texts like the viewer/rerater: exact first, then prefix."""
    exact: dict[str, int] = {}
    prefix: dict[str, int] = {}
    for index, text in enumerate(rated_texts):
        exact.setdefault(text, index)
        prefix.setdefault(text[:prefix_chars], index)

    matched: set[int] = set()
    unmatched = 0
    for hand_text in hand_texts:
        index = exact.get(hand_text)
        if index is None:
            index = prefix.get(hand_text[:prefix_chars])
        if index is None:
            unmatched += 1
        else:
            matched.add(index)
    return matched, unmatched


def match_hand_annotations(
    annotations: list[HandAnnotation], rated_texts: list[str], prefix_chars: int
) -> tuple[dict[int, tuple[int | None, ...]], int]:
    """Join annotations to rated rows using the viewer's exact/prefix matching."""
    exact: dict[str, int] = {}
    prefix: dict[str, int] = {}
    for index, text in enumerate(rated_texts):
        exact.setdefault(text, index)
        prefix.setdefault(text[:prefix_chars], index)

    matched_labels: dict[int, tuple[int | None, ...]] = {}
    unmatched = 0
    for annotation in annotations:
        index = exact.get(annotation.text)
        if index is None:
            index = prefix.get(annotation.text[:prefix_chars])
        if index is None:
            unmatched += 1
        else:
            matched_labels.setdefault(index, annotation.labels)
    return matched_labels, unmatched


def prepare_data(config: dict[str, Any]) -> PreparedData:
    data_config = config["data"]
    rated_path = resolve_path(require(data_config, "rated_path", "data"), "data.rated_path")
    hand_path = resolve_path(require(data_config, "hand_annotations_path", "data"), "data.hand_annotations_path")
    rating_config_path = resolve_path(
        require(data_config, "rating_config_path", "data"), "data.rating_config_path"
    )
    validation_fraction = float(require(data_config, "validation_fraction", "data"))
    prefix_chars = int(require(data_config, "prefix_match_chars", "data"))
    seed = int(require(config["runtime"], "seed", "runtime"))
    if not 0.0 <= validation_fraction < 1.0:
        raise SystemExit("data.validation_fraction must be in [0, 1).")
    if prefix_chars <= 0:
        raise SystemExit("data.prefix_match_chars must be positive.")

    filter_names, judge_names, target_names = load_target_layout(rating_config_path)
    upsample_weights = load_upsample_weights(data_config, filter_names)
    hand_annotations = read_hand_annotations(hand_path, filter_names)
    rated_texts: list[str] = []
    examples_by_row: dict[int, Example] = {}
    incomplete_rows = 0

    try:
        rated_file = rated_path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Rated data file does not exist: {rated_path}") from None
    with rated_file:
        for line_number, line in enumerate(rated_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {rated_path}:{line_number}: {exc}") from exc
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str) or not text:
                raise SystemExit(f"Missing non-empty 'text' at {rated_path}:{line_number}.")
            rated_texts.append(text)
            row_index = len(rated_texts) - 1
            targets = extract_targets(row, filter_names, judge_names)
            if targets is None:
                incomplete_rows += 1
                continue
            examples_by_row[row_index] = Example(text=text, targets=targets, source_line=line_number)

    if not rated_texts:
        raise SystemExit(f"No rated documents found in {rated_path}.")
    if not examples_by_row:
        raise SystemExit("No rows have a complete rating for every configured filter and judge.")

    hand_labels_by_row, unmatched_hand_rows = match_hand_annotations(
        hand_annotations, rated_texts, prefix_chars
    )
    hand_rows = set(hand_labels_by_row)
    complete_hand_rows = hand_rows & examples_by_row.keys()
    incomplete_hand_rows = len(hand_rows) - len(complete_hand_rows)
    if incomplete_hand_rows:
        print(
            f"WARNING: {incomplete_hand_rows} matched hand documents have incomplete "
            "judge targets and cannot be used for validation.",
            file=sys.stderr,
        )
    if unmatched_hand_rows:
        print(
            f"WARNING: {unmatched_hand_rows} hand documents are not present in the rated file yet.",
            file=sys.stderr,
        )

    eligible_rows = sorted(examples_by_row)
    random_validation_count = round(len(eligible_rows) * validation_fraction)
    if validation_fraction > 0.0 and random_validation_count == 0 and len(eligible_rows) > 1:
        random_validation_count = 1
    random_validation_rows = set(
        random.Random(seed).sample(eligible_rows, k=random_validation_count)
    )
    validation_rows = complete_hand_rows | random_validation_rows
    train_rows = set(eligible_rows) - validation_rows
    if not train_rows:
        raise SystemExit("The validation split leaves no training examples.")
    if not validation_rows:
        raise SystemExit("The validation split contains no complete examples.")

    unique_training_examples = [
        examples_by_row[index] for index in sorted(train_rows)
    ]
    training_examples, upsampled_source_rows = upsample_training_examples(
        unique_training_examples,
        filter_names,
        judge_names,
        upsample_weights,
    )

    return PreparedData(
        train=training_examples,
        validation=[examples_by_row[index] for index in sorted(validation_rows)],
        target_names=target_names,
        filter_names=filter_names,
        judge_names=judge_names,
        rated_rows=len(rated_texts),
        incomplete_rows=incomplete_rows,
        matched_hand_rows=len(complete_hand_rows),
        unmatched_hand_rows=unmatched_hand_rows + incomplete_hand_rows,
        random_validation_rows=len(random_validation_rows),
        human_labels_by_source_line={
            examples_by_row[index].source_line: hand_labels_by_row[index]
            for index in complete_hand_rows
        },
        unique_training_examples=len(unique_training_examples),
        upsampled_source_rows=upsampled_source_rows,
        upsample_weights=upsample_weights,
    )


def print_data_summary(data: PreparedData) -> None:
    print(f"Rated rows read:       {data.rated_rows}")
    print(f"Complete target rows:  {data.rated_rows - data.incomplete_rows}")
    print(f"Incomplete rows skipped: {data.incomplete_rows}")
    unique_training_examples = data.unique_training_examples or len(data.train)
    print(f"Training examples:     {len(data.train)} (after upsampling)")
    print(f"  unique training rows:{unique_training_examples:>5}")
    print(
        f"  duplicate copies:    "
        f"{len(data.train) - unique_training_examples:>5}"
    )
    print(f"  upsampled rows:       {data.upsampled_source_rows:>5}")
    if data.upsample_weights:
        formatted_weights = ", ".join(
            f"{HUMAN_LABEL_FIELDS.get(name, (None, name))[1]}={weight}"
            for name, weight in data.upsample_weights.items()
        )
        print(f"  upsample weights:    {formatted_weights}")
    print(f"Validation examples:   {len(data.validation)}")
    print(f"  matched hand rows:   {data.matched_hand_rows}")
    print(f"  random validation rows:{data.random_validation_rows:>5}")
    print(f"Unusable hand rows:    {data.unmatched_hand_rows}")
    print(f"Prediction targets:    {len(data.target_names)}")
    for index, name in enumerate(data.target_names):
        print(f"  [{index}] {name}")


def make_run_directory(checkpoint_root: Path) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    run_directory = checkpoint_root / f"run_{timestamp}"
    suffix = 2
    while run_directory.exists():
        run_directory = checkpoint_root / f"run_{timestamp}_{suffix}"
        suffix += 1
    run_directory.mkdir()
    return run_directory


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def import_training_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with "
            "`python -m pip install -r requirements.txt` before training. "
            f"Original import error: {exc}"
        ) from exc
    return torch, nn, functional, (LoraConfig, get_peft_model), (AutoModel, AutoTokenizer)


def choose_device_and_precision(torch: Any, runtime: dict[str, Any]) -> tuple[Any, Any | None, str]:
    requested_device = str(require(runtime, "device", "runtime")).lower()
    requested_precision = str(require(runtime, "precision", "runtime")).lower()
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "runtime.device is 'cuda', but CUDA is unavailable. Refusing to fall back "
                "to a resource-heavy CPU training run; set runtime.device explicitly if desired."
            )
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        raise SystemExit("runtime.device must be 'cuda' or 'cpu'.")

    if requested_precision == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif device.type == "cuda":
            precision = "fp16"
        else:
            precision = "fp32"
    else:
        precision = requested_precision
    dtype_by_name = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtype_by_name:
        raise SystemExit("runtime.precision must be one of: auto, fp32, fp16, bf16.")
    if device.type == "cpu" and precision == "fp16":
        raise SystemExit("fp16 CPU training is unsupported; use fp32 or bf16.")
    return device, dtype_by_name[precision], precision


def scheduler_multiplier(
    current_step: int, total_steps: int, warmup_fraction: float, decay_fraction: float
) -> float:
    warmup_steps = max(1, round(total_steps * warmup_fraction))
    decay_steps = max(1, round(total_steps * decay_fraction))
    decay_start = total_steps - decay_steps
    if current_step < warmup_steps:
        return (current_step + 1) / warmup_steps
    if current_step < decay_start:
        return 1.0
    progress = min(1.0, (current_step - decay_start) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def progress_timing(
    start_time: float,
    completed_steps: int,
    total_steps: int,
    current_time: float | None = None,
) -> tuple[float, float | None]:
    """Return elapsed seconds and a step-rate ETA in seconds."""
    now = time.monotonic() if current_time is None else current_time
    elapsed = max(0.0, now - start_time)
    if completed_steps <= 0:
        return elapsed, None
    remaining_steps = max(0, total_steps - completed_steps)
    return elapsed, elapsed * remaining_steps / completed_steps


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    whole_seconds = max(0, round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def absolute_probability_errors(logits: Any, targets: Any) -> Any:
    """Elementwise |sigmoid(logit) - target probability| in float32."""
    return (logits.float().sigmoid() - targets.float()).abs()


def update_human_accuracy(
    tallies: dict[str, AccuracyTally],
    mean_probabilities: list[list[float]],
    source_lines: list[int],
    human_labels_by_source_line: dict[int, tuple[int | None, ...]],
    filter_names: list[str],
) -> None:
    """Tally probabilistic correctness: p for label 1 and 1-p for label 0."""
    if len(mean_probabilities) != len(source_lines):
        raise ValueError("Each validation prediction row must have a source line.")

    for probabilities, source_line in zip(mean_probabilities, source_lines):
        labels = human_labels_by_source_line.get(source_line)
        if labels is None:
            continue
        if len(probabilities) != len(filter_names) or len(labels) != len(filter_names):
            raise ValueError("Human labels and probabilities must align with filter_names.")

        for filter_index, filter_name in enumerate(filter_names):
            label = labels[filter_index]
            tally = tallies.get(filter_name)
            if label is None or tally is None:
                continue

            probability = probabilities[filter_index]
            if label == 1:
                correctness_score = probability
                tally.positive_total += 1
                tally.positive_score_sum += correctness_score
            elif label == 0:
                correctness_score = 1.0 - probability
                tally.negative_total += 1
                tally.negative_score_sum += correctness_score
            else:
                # This metric is defined only for the binary hand labels 0 and 1.
                continue
            tally.overall_total += 1
            tally.overall_score_sum += correctness_score


def mean_accuracy(score_sum: float, total: int) -> float | None:
    return None if total == 0 else score_sum / total


def serialize_human_accuracy(
    tallies: dict[str, AccuracyTally],
) -> dict[str, dict[str, int | float | None]]:
    return {
        filter_name: {
            "positive_accuracy": mean_accuracy(
                tally.positive_score_sum, tally.positive_total
            ),
            "positive_score_sum": tally.positive_score_sum,
            "positive_total": tally.positive_total,
            "negative_accuracy": mean_accuracy(
                tally.negative_score_sum, tally.negative_total
            ),
            "negative_score_sum": tally.negative_score_sum,
            "negative_total": tally.negative_total,
            "overall_accuracy": mean_accuracy(
                tally.overall_score_sum, tally.overall_total
            ),
            "overall_score_sum": tally.overall_score_sum,
            "overall_total": tally.overall_total,
        }
        for filter_name, tally in tallies.items()
    }


def format_accuracy_percentage(score_sum: float, total: int) -> str:
    return "–" if total == 0 else f"{100.0 * score_sum / total:.1f}%"


def format_human_accuracy_lines(
    tallies: dict[str, AccuracyTally], filter_names: list[str]
) -> list[str]:
    lines: list[str] = []
    for filter_name in filter_names:
        field_spec = HUMAN_LABEL_FIELDS.get(filter_name)
        tally = tallies.get(filter_name)
        if field_spec is None or tally is None:
            continue
        display_name = field_spec[1]
        lines.append(
            f"{display_name:<13}—  Postive Acc : "
            f"{format_accuracy_percentage(tally.positive_score_sum, tally.positive_total)} "
            f"— Negative Acc : "
            f"{format_accuracy_percentage(tally.negative_score_sum, tally.negative_total)} "
            f"— Overall Acc "
            f"{format_accuracy_percentage(tally.overall_score_sum, tally.overall_total)}"
        )
    return lines


def write_validation_loss_plot(
    path: Path,
    points: list[tuple[int, float]],
    total_steps: int,
) -> None:
    """Atomically write a dependency-free SVG validation-loss curve."""
    finite_points = [(step, loss) for step, loss in points if math.isfinite(loss)]
    if not finite_points:
        return

    width, height = 900, 520
    left, right, top, bottom = 85, 35, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom
    losses = [loss for _, loss in finite_points]
    loss_min = min(losses)
    loss_max = max(losses)
    loss_span = loss_max - loss_min
    padding = max(0.01, loss_span * 0.1, loss_max * 0.05)
    y_min = max(0.0, loss_min - padding)
    y_max = loss_max + padding
    if y_max <= y_min:
        y_max = y_min + 0.01
    x_max = max(1, total_steps, max(step for step, _ in finite_points))

    def x_position(step: int) -> float:
        return left + plot_width * max(0, step) / x_max

    def y_position(loss: float) -> float:
        return top + plot_height * (y_max - loss) / (y_max - y_min)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="22" font-weight="bold">Validation Loss</text>',
    ]

    tick_count = 5
    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        y = top + plot_height * fraction
        value = y_max - (y_max - y_min) * fraction
        svg.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
            f'y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="monospace" font-size="12" fill="#374151">{value:.4f}</text>'
        )

    for tick in range(tick_count + 1):
        fraction = tick / tick_count
        x = left + plot_width * fraction
        step = round(x_max * fraction)
        svg.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
            f'font-family="monospace" font-size="12" fill="#374151">{step}</text>'
        )

    coordinates = " ".join(
        f"{x_position(step):.2f},{y_position(loss):.2f}"
        for step, loss in finite_points
    )
    svg.extend(
        [
            f'<polyline points="{coordinates}" fill="none" stroke="#2563eb" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>',
            *[
                f'<circle cx="{x_position(step):.2f}" cy="{y_position(loss):.2f}" '
                'r="4" fill="#2563eb"><title>'
                f'step {step}: {loss:.6f}</title></circle>'
                for step, loss in finite_points
            ],
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}" stroke="#111827" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
            'stroke="#111827" stroke-width="2"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" '
            'text-anchor="middle" font-family="sans-serif" font-size="15">Optimizer step</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
            'font-family="sans-serif" font-size="15" '
            f'transform="rotate(-90 20 {top + plot_height / 2:.2f})">BCE loss</text>',
            f'<text x="{left + plot_width}" y="45" text-anchor="end" '
            'font-family="monospace" font-size="13" fill="#1d4ed8">'
            f'latest: {finite_points[-1][1]:.6f} at step {finite_points[-1][0]}</text>',
            "</svg>",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def build_peft_lora_config(
    lora_config_class: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    use_rslora: bool,
) -> Any:
    return lora_config_class(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        use_rslora=use_rslora,
        bias="none",
        target_modules=target_modules,
    )


def train(config: dict[str, Any], data: PreparedData, config_path: Path) -> Path:
    torch, nn, functional, peft_api, transformers_api = import_training_dependencies()
    LoraConfig, get_peft_model = peft_api
    AutoModel, AutoTokenizer = transformers_api
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    model_config = config["model"]
    lora_config = config["lora"]
    optimization = config["optimization"]
    logging_config = config["logging"]
    runtime = config["runtime"]

    use_rslora = require(lora_config, "use_rslora", "lora")
    if not isinstance(use_rslora, bool):
        raise SystemExit("lora.use_rslora must be true or false.")

    seed = int(require(runtime, "seed", "runtime"))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device, model_dtype, precision = choose_device_and_precision(torch, runtime)
    base_model_path = resolve_path(
        require(model_config, "base_model_path", "model"), "model.base_model_path"
    )
    if not base_model_path.exists():
        raise SystemExit(f"Base model directory does not exist: {base_model_path}")
    checkpoint_root = resolve_path(
        require(logging_config, "checkpoint_dir", "logging"), "logging.checkpoint_dir"
    )
    run_directory = make_run_directory(checkpoint_root)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": str(base_model_path),
        "training_config_path": str(config_path.resolve()),
        "filter_names": data.filter_names,
        "judge_names": data.judge_names,
        "target_names": data.target_names,
        "target_scale": "rating / 10",
        "pooling": "token_position_zero",
        "adapter_type": "rslora" if use_rslora else "lora",
        "validation_loss_plot": "validation_loss.svg",
        "validation_metrics": {
            "loss": "mean binary cross-entropy with logits",
            "mae": "mean(abs(sigmoid(logit) - target_probability))",
            "human_accuracy": (
                "probabilistic correctness from the mean judge probability: "
                "p for hand label 1 and 1-p for hand label 0; reported overall "
                "and by positive/negative hand-label class"
            ),
        },
        "upsampling": {
            "threshold": "mean configured-judge rating >= 2",
            "weight_semantics": "total multiplicity; max qualifying filter weight",
            "weights": data.upsample_weights,
            "unique_training_examples": data.unique_training_examples,
            "upsampled_source_rows": data.upsampled_source_rows,
            "duplicate_copies_added": len(data.train) - data.unique_training_examples,
            "training_examples_after_upsampling": len(data.train),
        },
        "split": {
            "rated_rows": data.rated_rows,
            "incomplete_rows_skipped": data.incomplete_rows,
            "train_examples": len(data.train),
            "validation_examples": len(data.validation),
            "matched_hand_validation_rows": data.matched_hand_rows,
            "random_validation_rows_before_union": data.random_validation_rows,
            "unusable_hand_rows": data.unmatched_hand_rows,
        },
    }
    write_json(run_directory / "train_config.json", config)
    write_json(run_directory / "metadata.json", metadata)
    log_path = run_directory / "training_log.jsonl"

    print(f"Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    max_length = int(require(model_config, "max_length", "model"))
    if max_length <= 0:
        raise SystemExit("model.max_length must be positive.")

    class TextTargetDataset:
        def __init__(self, examples: list[Example]) -> None:
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Example:
            return self.examples[index]

    def collate(examples: list[Example]) -> tuple[dict[str, Any], Any, list[int]]:
        encoded = tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        targets = torch.tensor([example.targets for example in examples], dtype=torch.float32)
        source_lines = [example.source_line for example in examples]
        return dict(encoded), targets, source_lines

    batch_size = int(require(optimization, "batch_size", "optimization"))
    epochs = int(require(optimization, "epochs", "optimization"))
    num_workers = int(require(runtime, "num_workers", "runtime"))
    if batch_size <= 0 or epochs <= 0 or num_workers < 0:
        raise SystemExit("batch_size and epochs must be positive; num_workers cannot be negative.")
    loader_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        TextTargetDataset(data.train), shuffle=True, generator=loader_generator, **loader_options
    )
    validation_loader = DataLoader(
        TextTargetDataset(data.validation), shuffle=False, **loader_options
    )

    print(f"Loading ModernBERT from {base_model_path} on {device} ({precision})")
    load_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "attn_implementation": str(require(model_config, "attention_implementation", "model")),
    }
    if model_dtype is not None:
        load_kwargs["torch_dtype"] = model_dtype
    encoder = AutoModel.from_pretrained(base_model_path, **load_kwargs)
    if bool(require(model_config, "gradient_checkpointing", "model")):
        encoder.gradient_checkpointing_enable()

    rank = int(require(lora_config, "rank", "lora"))
    alpha = int(require(lora_config, "alpha", "lora"))
    dropout = float(require(lora_config, "dropout", "lora"))
    target_modules = require(lora_config, "target_modules", "lora")
    if rank <= 0 or alpha <= 0 or not 0.0 <= dropout < 1.0:
        raise SystemExit("LoRA rank/alpha must be positive and dropout must be in [0, 1).")
    if not isinstance(target_modules, list) or not all(
        isinstance(name, str) and name for name in target_modules
    ):
        raise SystemExit("lora.target_modules must be a list of module names.")
    encoder = get_peft_model(
        encoder,
        build_peft_lora_config(
            LoraConfig,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            use_rslora=use_rslora,
        ),
    )
    unexpected_trainable = [
        name
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if unexpected_trainable:
        raise SystemExit(
            "PEFT left non-LoRA encoder parameters trainable: "
            + ", ".join(unexpected_trainable[:10])
        )

    hidden_size = int(encoder.config.hidden_size)
    prediction_head = nn.Linear(hidden_size, len(data.target_names), bias=False)
    nn.init.zeros_(prediction_head.weight)
    encoder.to(device)
    prediction_head.to(device)
    encoder.print_trainable_parameters()

    trainable_parameters = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    trainable_parameters.extend(prediction_head.parameters())
    learning_rate = float(require(optimization, "learning_rate", "optimization"))
    betas = require(optimization, "betas", "optimization")
    weight_decay = float(require(optimization, "weight_decay", "optimization"))
    max_grad_norm = float(require(optimization, "max_grad_norm", "optimization"))
    warmup_fraction = float(require(optimization, "warmup_fraction", "optimization"))
    decay_fraction = float(require(optimization, "cosine_decay_fraction", "optimization"))
    if (
        learning_rate <= 0.0
        or not isinstance(betas, list)
        or len(betas) != 2
        or not all(0.0 <= float(beta) < 1.0 for beta in betas)
        or weight_decay < 0.0
        or max_grad_norm <= 0.0
        or not 0.0 <= warmup_fraction < 1.0
        or not 0.0 < decay_fraction <= 1.0
        or warmup_fraction + decay_fraction > 1.0
    ):
        raise SystemExit("Invalid optimizer or schedule settings in the training config.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        betas=(float(betas[0]), float(betas[1])),
        weight_decay=weight_decay,
    )
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: scheduler_multiplier(
            step, total_steps, warmup_fraction, decay_fraction
        ),
    )
    use_fp16_scaler = precision == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_scaler)
    autocast_enabled = model_dtype is not None

    def autocast_context() -> Any:
        if not autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=model_dtype)

    def forward_logits(encoded: dict[str, Any]) -> Any:
        outputs = encoder(**encoded)
        return prediction_head(outputs.last_hidden_state[:, 0, :])

    def validation_metrics() -> ValidationMetrics:
        encoder.eval()
        prediction_head.eval()
        loss_sum = 0.0
        absolute_error_sum = 0.0
        element_count = 0
        human_accuracy = {
            filter_name: AccuracyTally()
            for filter_name in data.filter_names
            if filter_name in HUMAN_LABEL_FIELDS
        }
        with torch.no_grad():
            for encoded, targets, source_lines in validation_loader:
                encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                targets = targets.to(device, non_blocking=True)
                with autocast_context():
                    logits = forward_logits(encoded)
                    losses = functional.binary_cross_entropy_with_logits(
                        logits, targets, reduction="none"
                    )
                absolute_errors = absolute_probability_errors(logits, targets)
                loss_sum += losses.float().sum().item()
                absolute_error_sum += absolute_errors.float().sum().item()
                element_count += losses.numel()
                mean_probabilities = (
                    logits.float()
                    .sigmoid()
                    .reshape(
                        logits.shape[0],
                        len(data.filter_names),
                        len(data.judge_names),
                    )
                    .mean(dim=2)
                    .cpu()
                    .tolist()
                )
                update_human_accuracy(
                    human_accuracy,
                    mean_probabilities,
                    source_lines,
                    data.human_labels_by_source_line,
                    data.filter_names,
                )
        encoder.train()
        prediction_head.train()
        return ValidationMetrics(
            loss=loss_sum / element_count,
            mae=absolute_error_sum / element_count,
            human_accuracy=human_accuracy,
        )

    def append_log(event: dict[str, Any]) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(), **event}
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            log_file.flush()

    def save_weights(
        directory: Path,
        step: int,
        metrics: ValidationMetrics | None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        encoder.save_pretrained(
            directory / "adapter",
            safe_serialization=True,
            save_embedding_layers=False,
        )
        torch.save(prediction_head.state_dict(), directory / "prediction_head.pt")
        write_json(
            directory / "checkpoint.json",
            {
                "global_step": step,
                "validation_loss": None if metrics is None else metrics.loss,
                "validation_mae": None if metrics is None else metrics.mae,
                "human_accuracy": (
                    None
                    if metrics is None
                    else serialize_human_accuracy(metrics.human_accuracy)
                ),
                "target_names": data.target_names,
                "base_model_path": str(base_model_path),
                "prediction_head": {
                    "input_features": hidden_size,
                    "output_features": len(data.target_names),
                    "bias": False,
                    "pooling": "token_position_zero",
                },
            },
        )

    log_every = int(require(logging_config, "log_every", "logging"))
    val_every = int(require(logging_config, "val_every", "logging"))
    if log_every <= 0 or val_every <= 0:
        raise SystemExit("logging.log_every and logging.val_every must be positive.")

    print(f"Run directory: {run_directory}")
    print(f"Training for {epochs} epochs / {total_steps} optimizer steps")
    append_log(
        {
            "event": "start",
            "epochs": epochs,
            "total_steps": total_steps,
            "train_examples": len(data.train),
            "unique_training_examples": data.unique_training_examples,
            "upsampled_source_rows": data.upsampled_source_rows,
            "duplicate_copies_added": len(data.train) - data.unique_training_examples,
            "upsample_weights": data.upsample_weights,
            "validation_examples": len(data.validation),
            "precision": precision,
        }
    )

    encoder.train()
    prediction_head.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    running_loss = 0.0
    running_count = 0
    best_validation_loss = math.inf
    mae_at_best_validation_loss: float | None = None
    latest_validation_metrics: ValidationMetrics | None = None
    validation_history: list[tuple[int, float]] = []
    last_validation_step = -1
    training_start_time = time.monotonic()
    progress_bar = tqdm(
        total=total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    for epoch in range(1, epochs + 1):
        for encoded, targets, _source_lines in train_loader:
            global_step += 1
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            targets = targets.to(device, non_blocking=True)
            with autocast_context():
                loss = functional.binary_cross_entropy_with_logits(
                    forward_logits(encoded), targets, reduction="none"
                ).mean()

            if use_fp16_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            if use_fp16_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().float().item()
            running_count += 1
            learning_rate_now = scheduler.get_last_lr()[0]
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                lr=f"{learning_rate_now:.2e}",
            )
            if global_step % log_every == 0:
                mean_loss = running_loss / running_count
                elapsed_seconds, eta_seconds = progress_timing(
                    training_start_time,
                    global_step,
                    total_steps,
                )
                progress_bar.write(
                    f"step {global_step}/{total_steps} epoch {epoch} "
                    f"loss={mean_loss:.6f} grad_norm={float(grad_norm):.4f} "
                    f"lr={learning_rate_now:.3e} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"eta={format_duration(eta_seconds)}"
                )
                append_log(
                    {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": mean_loss,
                        "grad_norm": float(grad_norm),
                        "learning_rate": learning_rate_now,
                        "elapsed_seconds": elapsed_seconds,
                        "eta_seconds": eta_seconds,
                    }
                )
                running_loss = 0.0
                running_count = 0

            if global_step % val_every == 0:
                progress_bar.set_description("Validating")
                current_validation_metrics = validation_metrics()
                progress_bar.set_description("Training")
                latest_validation_metrics = current_validation_metrics
                validation_history.append(
                    (global_step, current_validation_metrics.loss)
                )
                write_validation_loss_plot(
                    run_directory / "validation_loss.svg",
                    validation_history,
                    total_steps,
                )
                last_validation_step = global_step
                elapsed_seconds, eta_seconds = progress_timing(
                    training_start_time,
                    global_step,
                    total_steps,
                )
                validation_line = (
                    f"step {global_step}/{total_steps} "
                    f"validation_loss={current_validation_metrics.loss:.6f} "
                    f"validation_mae={current_validation_metrics.mae:.6f} "
                    f"elapsed={format_duration(elapsed_seconds)} "
                    f"eta={format_duration(eta_seconds)}"
                )
                progress_bar.write("")
                progress_bar.write(validation_line)
                for accuracy_line in format_human_accuracy_lines(
                    current_validation_metrics.human_accuracy,
                    data.filter_names,
                ):
                    progress_bar.write(accuracy_line)
                progress_bar.write("")
                append_log(
                    {
                        "event": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": current_validation_metrics.loss,
                        "mae": current_validation_metrics.mae,
                        "human_accuracy": serialize_human_accuracy(
                            current_validation_metrics.human_accuracy
                        ),
                        "elapsed_seconds": elapsed_seconds,
                        "eta_seconds": eta_seconds,
                    }
                )
                if current_validation_metrics.loss < best_validation_loss:
                    best_validation_loss = current_validation_metrics.loss
                    mae_at_best_validation_loss = current_validation_metrics.mae
                    save_weights(
                        run_directory / "best",
                        global_step,
                        current_validation_metrics,
                    )

    progress_bar.close()

    if last_validation_step != global_step:
        current_validation_metrics = validation_metrics()
        latest_validation_metrics = current_validation_metrics
        validation_history.append((global_step, current_validation_metrics.loss))
        write_validation_loss_plot(
            run_directory / "validation_loss.svg",
            validation_history,
            total_steps,
        )
        elapsed_seconds, eta_seconds = progress_timing(
            training_start_time,
            global_step,
            total_steps,
        )
        validation_line = (
            f"step {global_step}/{total_steps} "
            f"validation_loss={current_validation_metrics.loss:.6f} "
            f"validation_mae={current_validation_metrics.mae:.6f} "
            f"elapsed={format_duration(elapsed_seconds)} "
            f"eta={format_duration(eta_seconds)}"
        )
        print()
        print(validation_line)
        for accuracy_line in format_human_accuracy_lines(
            current_validation_metrics.human_accuracy,
            data.filter_names,
        ):
            print(accuracy_line)
        print()
        append_log(
            {
                "event": "validation",
                "step": global_step,
                "epoch": epochs,
                "loss": current_validation_metrics.loss,
                "mae": current_validation_metrics.mae,
                "human_accuracy": serialize_human_accuracy(
                    current_validation_metrics.human_accuracy
                ),
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            }
        )
        if current_validation_metrics.loss < best_validation_loss:
            best_validation_loss = current_validation_metrics.loss
            mae_at_best_validation_loss = current_validation_metrics.mae
            save_weights(
                run_directory / "best",
                global_step,
                current_validation_metrics,
            )

    save_weights(run_directory / "final", global_step, latest_validation_metrics)
    total_elapsed_seconds, _ = progress_timing(
        training_start_time,
        global_step,
        total_steps,
    )
    append_log(
        {
            "event": "complete",
            "step": global_step,
            "best_validation_loss": best_validation_loss,
            "mae_at_best_validation_loss": mae_at_best_validation_loss,
            "latest_validation_loss": (
                None if latest_validation_metrics is None else latest_validation_metrics.loss
            ),
            "latest_validation_mae": (
                None if latest_validation_metrics is None else latest_validation_metrics.mae
            ),
            "latest_human_accuracy": (
                None
                if latest_validation_metrics is None
                else serialize_human_accuracy(latest_validation_metrics.human_accuracy)
            ),
            "elapsed_seconds": total_elapsed_seconds,
        }
    )
    print(f"Training complete. Best validation loss: {best_validation_loss:.6f}")
    print(f"Weights and logs: {run_directory}")
    return run_directory


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data = prepare_data(config)
    print_data_summary(data)
    if args.validate_data_only:
        print("Data validation complete; model was not loaded and training was not started.")
        return
    if data.unmatched_hand_rows:
        raise SystemExit(
            f"Refusing to train: {data.unmatched_hand_rows} hand-annotated documents do not "
            "yet have complete LLM-judge targets in the rated file. Finish rating those "
            "documents, then rerun --validate-data-only."
        )
    train(config, data, config_path)


if __name__ == "__main__":
    main()
