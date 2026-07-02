#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from tqdm import tqdm


REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_SAMPLE_PREFIX = "sample/10BT"
DEFAULT_COUNT = 100_000
DEFAULT_OUTPUT = Path("data/fineweb_edu_100k.jsonl")
DEFAULT_BATCH_SIZE = 1_024


def normalize_one_row(text: str) -> str:
    return " ".join(text.split())


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_existing_text_digests(path: Path) -> set[str]:
    digests: set[str] = set()
    if not path.exists():
        return digests

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object.")

            text = row.get("text")
            if not isinstance(text, str):
                raise SystemExit(f"{path}:{line_number} has no string 'text' field.")

            digests.add(text_digest(normalize_one_row(text)))

    return digests


def list_parquet_files(sample_prefix: str) -> list[str]:
    prefix = sample_prefix.strip("/") + "/"
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    parquet_files = sorted(
        path for path in files if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"No parquet files found under {REPO_ID}/{prefix}")

    return parquet_files


def iter_document_texts(
    sample_prefix: str,
    batch_size: int,
) -> Iterator[str]:
    filesystem = HfFileSystem()

    for parquet_path in list_parquet_files(sample_prefix):
        remote_path = f"datasets/{REPO_ID}/{parquet_path}"
        tqdm.write(f"Reading {parquet_path}")

        with filesystem.open(remote_path, "rb") as parquet_file:
            reader = pq.ParquetFile(parquet_file)
            for batch in reader.iter_batches(columns=["text"], batch_size=batch_size):
                for text in batch.column("text").to_pylist():
                    if not isinstance(text, str):
                        continue

                    text = normalize_one_row(text)
                    if not text:
                        continue

                    yield text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FineWeb-Edu document bodies as one serialized JSON object "
            "per row."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of non-empty documents to write. Default: {DEFAULT_COUNT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON Lines file. Default: {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--sample-prefix",
        default=DEFAULT_SAMPLE_PREFIX,
        help=f"Dataset sample shard prefix. Default: {DEFAULT_SAMPLE_PREFIX}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet batch size. Default: {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new unique documents to the output JSONL instead of overwriting it.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip documents whose normalized text already exists in the output JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing_digests = (
        load_existing_text_digests(args.output)
        if args.append or args.skip_existing
        else set()
    )

    written = 0
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8", newline="\n") as output_file:
        texts = iter_document_texts(
            sample_prefix=args.sample_prefix,
            batch_size=args.batch_size,
        )
        with tqdm(total=args.count, unit="doc", desc="FineWeb-Edu") as progress:
            for text in texts:
                digest = text_digest(text)
                if digest in existing_digests:
                    continue

                output_file.write(json.dumps({"text": text}, ensure_ascii=False))
                output_file.write("\n")
                existing_digests.add(digest)
                written += 1
                progress.update()
                if written >= args.count:
                    break

    if written != args.count:
        raise SystemExit(f"Expected {args.count} documents but only wrote {written}.")

    print(f"Wrote {written} documents to {args.output}")


if __name__ == "__main__":
    main()
