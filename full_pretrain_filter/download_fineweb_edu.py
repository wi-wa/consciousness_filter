#!/usr/bin/env python3
"""Download a token-bounded FineWeb-Edu corpus disjoint from prior corpora.

FineWeb-Edu publishes a GPT-2 ``token_count`` for each document. This script
uses that field for both the per-document bounds and the total token target, so
the expensive source documents do not need to be tokenized locally.

The destination is written to ``<output>.partial`` and atomically renamed only
after the target is reached. An interrupted run can be continued with
``--resume``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.json")


@dataclass(frozen=True)
class DownloadConfig:
    repo_id: str
    revision: str
    sample_prefix: str
    output_jsonl: Path
    exclude_jsonl: tuple[Path, ...]
    target_tokens: int
    min_document_tokens: int
    max_document_tokens: int
    parquet_batch_size: int


@dataclass(frozen=True)
class ExistingOutput:
    digests: set[bytes]
    documents: int
    tokens: int


@dataclass(frozen=True)
class CorpusAudit:
    documents: int
    tokens: int
    excluded_documents: int
    excluded_tokens: int


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
        help="Continue from an existing .partial output file.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing complete or partial output after success.",
    )
    output_mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify that the completed output is disjoint from every exclusion.",
    )
    output_mode.add_argument(
        "--repair-overlap",
        action="store_true",
        help=(
            "Remove excluded documents from a completed output and download "
            "replacements until target_tokens is restored."
        ),
    )
    return parser.parse_args(argv)


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SystemExit(f"Missing {key!r} in {context}.")
    return mapping[key]


def require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{context} must be a non-empty string.")
    return value.strip()


def require_positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SystemExit(f"{context} must be an integer greater than zero.")
    return value


def resolve_repo_path(value: Any, context: str) -> Path:
    raw_path = require_nonempty_string(value, context)
    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> DownloadConfig:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except FileNotFoundError:
        raise SystemExit(f"Config does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    download_settings = require(raw_config, "download_settings", str(path))
    if not isinstance(download_settings, dict):
        raise SystemExit("config.download_settings must be an object.")
    dataset = require(
        download_settings, "dataset", "config.download_settings"
    )
    if not isinstance(dataset, dict):
        raise SystemExit("config.download_settings.dataset must be an object.")

    raw_exclusions = require(
        download_settings, "exclude_jsonl", "config.download_settings"
    )
    if not isinstance(raw_exclusions, list) or not raw_exclusions:
        raise SystemExit(
            "config.download_settings.exclude_jsonl must be a non-empty list of paths."
        )

    config = DownloadConfig(
        repo_id=require_nonempty_string(
            require(dataset, "repo_id", "config.download_settings.dataset"),
            "config.download_settings.dataset.repo_id",
        ),
        revision=require_nonempty_string(
            require(dataset, "revision", "config.download_settings.dataset"),
            "config.download_settings.dataset.revision",
        ),
        sample_prefix=require_nonempty_string(
            require(
                dataset, "sample_prefix", "config.download_settings.dataset"
            ),
            "config.download_settings.dataset.sample_prefix",
        ).strip("/"),
        output_jsonl=resolve_repo_path(
            require(
                download_settings, "output_jsonl", "config.download_settings"
            ),
            "config.download_settings.output_jsonl",
        ),
        exclude_jsonl=tuple(
            resolve_repo_path(
                value, f"config.download_settings.exclude_jsonl[{index}]"
            )
            for index, value in enumerate(raw_exclusions)
        ),
        target_tokens=require_positive_int(
            require(
                download_settings, "target_tokens", "config.download_settings"
            ),
            "config.download_settings.target_tokens",
        ),
        min_document_tokens=require_positive_int(
            require(
                download_settings,
                "min_document_tokens",
                "config.download_settings",
            ),
            "config.download_settings.min_document_tokens",
        ),
        max_document_tokens=require_positive_int(
            require(
                download_settings,
                "max_document_tokens",
                "config.download_settings",
            ),
            "config.download_settings.max_document_tokens",
        ),
        parquet_batch_size=require_positive_int(
            require(
                download_settings,
                "parquet_batch_size",
                "config.download_settings",
            ),
            "config.download_settings.parquet_batch_size",
        ),
    )
    if config.min_document_tokens > config.max_document_tokens:
        raise SystemExit(
            "config.download_settings.min_document_tokens cannot exceed "
            "config.download_settings.max_document_tokens."
        )
    if config.output_jsonl in config.exclude_jsonl:
        raise SystemExit(
            "config.download_settings.output_jsonl cannot also be an exclusion path."
        )
    return config


def normalized_text(text: str) -> str:
    """Match the normalization used by the embedding-data downloader."""
    return " ".join(text.split())


def text_digest(text: str) -> bytes:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).digest()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        input_file = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Required JSONL file does not exist: {path}") from None

    with input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object.")
            yield line_number, row


def load_exclusion_digests(paths: Sequence[Path]) -> set[bytes]:
    digests: set[bytes] = set()
    for path in paths:
        rows = 0
        for line_number, row in iter_jsonl(path):
            text = row.get("text")
            if not isinstance(text, str):
                raise SystemExit(f"{path}:{line_number} has no string 'text' field.")
            digests.add(text_digest(text))
            rows += 1
        print(f"Loaded {rows:,} exclusion rows from {path}")
    return digests


def load_partial_output(
    path: Path, *, min_document_tokens: int, max_document_tokens: int
) -> ExistingOutput:
    digests: set[bytes] = set()
    tokens = 0
    documents = 0
    for line_number, row in iter_jsonl(path):
        text = row.get("text")
        token_count = row.get("token_count")
        if not isinstance(text, str):
            raise SystemExit(f"{path}:{line_number} has no string 'text' field.")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= 0
        ):
            raise SystemExit(
                f"{path}:{line_number} has no positive integer 'token_count' field."
            )
        if not min_document_tokens <= token_count <= max_document_tokens:
            raise SystemExit(
                f"{path}:{line_number} has token_count {token_count}, outside the "
                f"configured [{min_document_tokens}, {max_document_tokens}] range."
            )
        digest = text_digest(text)
        if digest in digests:
            raise SystemExit(f"Duplicate text found in partial output at {path}:{line_number}.")
        digests.add(digest)
        tokens += token_count
        documents += 1
    return ExistingOutput(digests=digests, documents=documents, tokens=tokens)


def list_parquet_files(config: DownloadConfig) -> list[str]:
    prefix = config.sample_prefix + "/"
    files = HfApi().list_repo_files(
        config.repo_id,
        repo_type="dataset",
        revision=config.revision,
    )
    parquet_files = sorted(
        path for path in files if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(
            f"No Parquet files found under {config.repo_id}@{config.revision}/{prefix}"
        )
    return parquet_files


def iter_documents(config: DownloadConfig) -> Iterator[tuple[str, int]]:
    filesystem = HfFileSystem()
    for parquet_path in list_parquet_files(config):
        tqdm.write(f"Reading {parquet_path}")
        remote_path = (
            f"datasets/{config.repo_id}@{config.revision}/{parquet_path}"
        )
        with filesystem.open(remote_path, "rb") as parquet_file:
            reader = pq.ParquetFile(parquet_file)
            required_columns = {"text", "token_count"}
            missing_columns = required_columns.difference(reader.schema.names)
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise RuntimeError(f"{parquet_path} is missing columns: {missing}")

            for batch in reader.iter_batches(
                columns=["text", "token_count"],
                batch_size=config.parquet_batch_size,
            ):
                texts = batch.column("text").to_pylist()
                token_counts = batch.column("token_count").to_pylist()
                for text, token_count in zip(texts, token_counts, strict=True):
                    if not isinstance(text, str) or not text:
                        continue
                    if (
                        not isinstance(token_count, int)
                        or isinstance(token_count, bool)
                        or token_count <= 0
                    ):
                        continue
                    yield text, token_count


def partial_path_for(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".partial")


def audit_corpus(
    config: DownloadConfig, path: Path, excluded_digests: set[bytes]
) -> CorpusAudit:
    documents = 0
    tokens = 0
    excluded_documents = 0
    excluded_tokens = 0
    for line_number, row in iter_jsonl(path):
        text = row.get("text")
        token_count = row.get("token_count")
        if not isinstance(text, str):
            raise SystemExit(f"{path}:{line_number} has no string 'text' field.")
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or not config.min_document_tokens
            <= token_count
            <= config.max_document_tokens
        ):
            raise SystemExit(
                f"{path}:{line_number} has invalid token_count {token_count!r}; "
                f"expected [{config.min_document_tokens}, "
                f"{config.max_document_tokens}]."
            )
        documents += 1
        tokens += token_count
        if text_digest(text) in excluded_digests:
            excluded_documents += 1
            excluded_tokens += token_count
    return CorpusAudit(
        documents=documents,
        tokens=tokens,
        excluded_documents=excluded_documents,
        excluded_tokens=excluded_tokens,
    )


def verify_completed_output(config: DownloadConfig) -> CorpusAudit:
    if not config.output_jsonl.exists():
        raise SystemExit(f"Completed output does not exist: {config.output_jsonl}")
    excluded_digests = load_exclusion_digests(config.exclude_jsonl)
    audit = audit_corpus(config, config.output_jsonl, excluded_digests)
    print(
        f"Audited {audit.documents:,} documents and {audit.tokens:,} tokens in "
        f"{config.output_jsonl}."
    )
    if audit.excluded_documents:
        raise SystemExit(
            f"Disjointness check failed: {audit.excluded_documents:,} documents "
            f"({audit.excluded_tokens:,} tokens) occur in configured exclusions."
        )
    print("Disjointness check passed: 0 documents occur in configured exclusions.")
    return audit


def write_repaired_partial(
    config: DownloadConfig,
    excluded_digests: set[bytes],
    destination: Path,
) -> CorpusAudit:
    documents = 0
    tokens = 0
    excluded_documents = 0
    excluded_tokens = 0
    with destination.open("x", encoding="utf-8", newline="\n") as output_file:
        for line_number, row in iter_jsonl(config.output_jsonl):
            text = row.get("text")
            token_count = row.get("token_count")
            if not isinstance(text, str):
                raise SystemExit(
                    f"{config.output_jsonl}:{line_number} has no string 'text' field."
                )
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or not config.min_document_tokens
                <= token_count
                <= config.max_document_tokens
            ):
                raise SystemExit(
                    f"{config.output_jsonl}:{line_number} has invalid token_count "
                    f"{token_count!r}; expected [{config.min_document_tokens}, "
                    f"{config.max_document_tokens}]."
                )
            if text_digest(text) in excluded_digests:
                excluded_documents += 1
                excluded_tokens += token_count
                continue
            output_file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            documents += 1
            tokens += token_count
    audit = CorpusAudit(
        documents=documents,
        tokens=tokens,
        excluded_documents=excluded_documents,
        excluded_tokens=excluded_tokens,
    )
    return audit


def repair_completed_output(config: DownloadConfig) -> None:
    output_path = config.output_jsonl
    partial_path = partial_path_for(output_path)
    repairing_path = output_path.with_name(output_path.name + ".repairing")
    backup_path = output_path.with_name(output_path.name + ".pre-disjoint-repair.bak")
    if not output_path.exists():
        raise SystemExit(f"Completed output does not exist: {output_path}")
    if repairing_path.exists() or backup_path.exists():
        raise SystemExit(
            f"Repair scratch path already exists: "
            f"{repairing_path if repairing_path.exists() else backup_path}"
        )

    excluded_digests = load_exclusion_digests(config.exclude_jsonl)
    if partial_path.exists():
        existing = load_partial_output(
            partial_path,
            min_document_tokens=config.min_document_tokens,
            max_document_tokens=config.max_document_tokens,
        )
        overlap = excluded_digests.intersection(existing.digests)
        if overlap:
            raise SystemExit(
                f"Existing repair partial still contains {len(overlap):,} excluded "
                "documents; remove it before starting a new repair."
            )
        print(
            f"Continuing prior repair partial with {existing.documents:,} documents "
            f"and {existing.tokens:,} tokens."
        )
    else:
        audit = write_repaired_partial(config, excluded_digests, repairing_path)
        if not audit.excluded_documents:
            repairing_path.unlink()
            print("No excluded documents found; the completed output is already disjoint.")
            return
        os.replace(repairing_path, partial_path)
        print(
            f"Removed {audit.excluded_documents:,} excluded documents "
            f"({audit.excluded_tokens:,} tokens); refilling from "
            f"{audit.tokens:,} retained tokens."
        )

    os.replace(output_path, backup_path)
    try:
        download(config, resume=True, overwrite=False)
    except BaseException:
        if not output_path.exists() and backup_path.exists():
            os.replace(backup_path, output_path)
        print(
            "Repair refill did not finish. The original completed output was "
            "restored; rerun --repair-overlap to continue the clean partial."
        )
        raise
    verify_completed_output(config)
    backup_path.unlink()


def prepare_output(
    config: DownloadConfig, *, resume: bool, overwrite: bool
) -> tuple[Path, ExistingOutput, str]:
    output_path = config.output_jsonl
    partial_path = partial_path_for(output_path)
    if resume:
        if output_path.exists():
            raise SystemExit(
                f"Completed output already exists: {output_path}. Nothing to resume."
            )
        if not partial_path.exists():
            raise SystemExit(f"Partial output does not exist: {partial_path}")
        existing = load_partial_output(
            partial_path,
            min_document_tokens=config.min_document_tokens,
            max_document_tokens=config.max_document_tokens,
        )
        return partial_path, existing, "a"

    if not overwrite and output_path.exists():
        raise SystemExit(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    if not overwrite and partial_path.exists():
        raise SystemExit(
            f"Partial output already exists: {partial_path}. "
            "Use --resume to continue it or --overwrite to start again."
        )
    return partial_path, ExistingOutput(set(), 0, 0), "w"


def download(config: DownloadConfig, *, resume: bool, overwrite: bool) -> None:
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    partial_path, existing, mode = prepare_output(
        config, resume=resume, overwrite=overwrite
    )
    excluded_digests = load_exclusion_digests(config.exclude_jsonl)
    partial_exclusions = excluded_digests.intersection(existing.digests)
    if partial_exclusions:
        raise SystemExit(
            f"Partial output contains {len(partial_exclusions):,} document(s) from "
            "the configured exclusions. Start again with --overwrite."
        )
    seen_digests = excluded_digests | existing.digests
    total_tokens = existing.tokens
    documents = existing.documents

    if total_tokens >= config.target_tokens:
        os.replace(partial_path, config.output_jsonl)
        print(
            f"Finalized {documents:,} documents ({total_tokens:,} tokens) at "
            f"{config.output_jsonl}"
        )
        return

    rejected_short = 0
    rejected_long = 0
    rejected_seen = 0
    with partial_path.open(mode, encoding="utf-8", newline="\n") as output_file:
        with tqdm(
            total=config.target_tokens,
            initial=total_tokens,
            unit="tok",
            unit_scale=True,
            desc="FineWeb-Edu",
        ) as progress:
            for text, token_count in iter_documents(config):
                if token_count < config.min_document_tokens:
                    rejected_short += 1
                    continue
                if token_count > config.max_document_tokens:
                    rejected_long += 1
                    continue

                digest = text_digest(text)
                if digest in seen_digests:
                    rejected_seen += 1
                    continue

                output_file.write(
                    json.dumps(
                        {"text": text, "token_count": token_count},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                output_file.write("\n")
                seen_digests.add(digest)
                total_tokens += token_count
                documents += 1
                progress.update(token_count)
                if total_tokens >= config.target_tokens:
                    break

    if total_tokens < config.target_tokens:
        raise SystemExit(
            f"Dataset source was exhausted at {total_tokens:,} of "
            f"{config.target_tokens:,} target tokens. Partial output remains at "
            f"{partial_path}."
        )

    os.replace(partial_path, config.output_jsonl)
    print(
        f"Wrote {documents:,} documents and {total_tokens:,} tokens to "
        f"{config.output_jsonl} (target overshoot: "
        f"{total_tokens - config.target_tokens:,} tokens)."
    )
    print(
        "Rejected documents: "
        f"{rejected_short:,} below minimum, {rejected_long:,} above maximum, "
        f"{rejected_seen:,} excluded or duplicate."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    if args.verify:
        verify_completed_output(config)
        return
    if args.repair_overlap:
        repair_completed_output(config)
        return
    download(config, resume=args.resume, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
