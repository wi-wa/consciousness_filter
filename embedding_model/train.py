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
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "embedding_model/configs/train.json"


@dataclass(frozen=True)
class Example:
    text: str
    targets: tuple[float, ...]
    source_line: int


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


def read_hand_texts(path: Path) -> list[str]:
    texts: list[str] = []
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
            texts.append(text)
    if not texts:
        raise SystemExit(f"No hand-annotated documents found in {path}.")
    return texts


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
    hand_texts = read_hand_texts(hand_path)
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

    hand_rows, unmatched_hand_rows = match_hand_rows(hand_texts, rated_texts, prefix_chars)
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

    return PreparedData(
        train=[examples_by_row[index] for index in sorted(train_rows)],
        validation=[examples_by_row[index] for index in sorted(validation_rows)],
        target_names=target_names,
        filter_names=filter_names,
        judge_names=judge_names,
        rated_rows=len(rated_texts),
        incomplete_rows=incomplete_rows,
        matched_hand_rows=len(complete_hand_rows),
        unmatched_hand_rows=unmatched_hand_rows + incomplete_hand_rows,
        random_validation_rows=len(random_validation_rows),
    )


def print_data_summary(data: PreparedData) -> None:
    print(f"Rated rows read:       {data.rated_rows}")
    print(f"Complete target rows:  {data.rated_rows - data.incomplete_rows}")
    print(f"Incomplete rows skipped: {data.incomplete_rows}")
    print(f"Training examples:     {len(data.train)}")
    print(f"Validation examples:   {len(data.validation)}")
    print(f"  matched hand rows:   {data.matched_hand_rows}")
    print(f"  random 5% candidates:{data.random_validation_rows:>5}")
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


def train(config: dict[str, Any], data: PreparedData, config_path: Path) -> Path:
    torch, nn, functional, peft_api, transformers_api = import_training_dependencies()
    LoraConfig, get_peft_model = peft_api
    AutoModel, AutoTokenizer = transformers_api
    from torch.utils.data import DataLoader

    model_config = config["model"]
    lora_config = config["lora"]
    optimization = config["optimization"]
    logging_config = config["logging"]
    runtime = config["runtime"]

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

    def collate(examples: list[Example]) -> tuple[dict[str, Any], Any]:
        encoded = tokenizer(
            [example.text for example in examples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        targets = torch.tensor([example.targets for example in examples], dtype=torch.float32)
        return dict(encoded), targets

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
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_modules,
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

    def validation_loss() -> float:
        encoder.eval()
        prediction_head.eval()
        loss_sum = 0.0
        element_count = 0
        with torch.no_grad():
            for encoded, targets in validation_loader:
                encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
                targets = targets.to(device, non_blocking=True)
                with autocast_context():
                    losses = functional.binary_cross_entropy_with_logits(
                        forward_logits(encoded), targets, reduction="none"
                    )
                loss_sum += losses.float().sum().item()
                element_count += losses.numel()
        encoder.train()
        prediction_head.train()
        return loss_sum / element_count

    def append_log(event: dict[str, Any]) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(), **event}
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            log_file.flush()

    def save_weights(directory: Path, step: int, val_loss: float | None) -> None:
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
                "validation_loss": val_loss,
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
    latest_validation_loss: float | None = None
    last_validation_step = -1

    for epoch in range(1, epochs + 1):
        for encoded, targets in train_loader:
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
            if global_step % log_every == 0:
                mean_loss = running_loss / running_count
                learning_rate_now = scheduler.get_last_lr()[0]
                print(
                    f"step {global_step}/{total_steps} epoch {epoch} "
                    f"loss={mean_loss:.6f} grad_norm={float(grad_norm):.4f} "
                    f"lr={learning_rate_now:.3e}"
                )
                append_log(
                    {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": mean_loss,
                        "grad_norm": float(grad_norm),
                        "learning_rate": learning_rate_now,
                    }
                )
                running_loss = 0.0
                running_count = 0

            if global_step % val_every == 0:
                current_validation_loss = validation_loss()
                latest_validation_loss = current_validation_loss
                last_validation_step = global_step
                print(f"step {global_step}/{total_steps} validation_loss={current_validation_loss:.6f}")
                append_log(
                    {
                        "event": "validation",
                        "step": global_step,
                        "epoch": epoch,
                        "loss": current_validation_loss,
                    }
                )
                if current_validation_loss < best_validation_loss:
                    best_validation_loss = current_validation_loss
                    save_weights(run_directory / "best", global_step, current_validation_loss)

    if last_validation_step != global_step:
        current_validation_loss = validation_loss()
        latest_validation_loss = current_validation_loss
        print(f"step {global_step}/{total_steps} validation_loss={current_validation_loss:.6f}")
        append_log(
            {
                "event": "validation",
                "step": global_step,
                "epoch": epochs,
                "loss": current_validation_loss,
            }
        )
        if current_validation_loss < best_validation_loss:
            best_validation_loss = current_validation_loss
            save_weights(run_directory / "best", global_step, current_validation_loss)

    save_weights(run_directory / "final", global_step, latest_validation_loss)
    append_log(
        {
            "event": "complete",
            "step": global_step,
            "best_validation_loss": best_validation_loss,
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
