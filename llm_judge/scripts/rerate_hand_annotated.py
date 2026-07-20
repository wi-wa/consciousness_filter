#!/usr/bin/env python3
"""Fill missing OpenRouter ratings for hand-annotated documents already rated.

The hand-annotation JSONL is always read-only. This script only considers
documents present in both that file and the existing rated JSONL; unlike the
main rater, run.max_documents does not restrict its scope.

Default behavior fills only missing configured (filter, model) entries.
``--hard-refresh`` first removes the entire ``ratings`` value from every
matched rated row, persists that removal, and then refills configured entries.
Both mutating modes immediately overwrite ``<rated-output>.bak`` before making
changes and persist each completed API batch to the live rated JSONL.

``--check`` is strictly read-only and prints only aggregate missing counts.
It is mutually exclusive with ``--hard-refresh``.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rate_filters_openrouter as rater

PREFIX_MATCH_CHARS = 200


def read_annotation_texts(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Hand-annotation file does not exist: {path}")
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise SystemExit(
                    f"Line {line_number} of {path} has no string 'text' field."
                )
            texts.append(row["text"])
    if not texts:
        raise SystemExit(f"No usable hand-annotated documents found in {path}.")
    return texts


def match_annotations_to_rows(
    annotation_texts: list[str], rated_rows: list[dict[str, Any]]
) -> tuple[list[int], int]:
    """Return unique rated-row indexes plus the number of unmatched annotations."""
    exact: dict[str, int] = {}
    prefix: dict[str, int] = {}
    for index, row in enumerate(rated_rows):
        text = row["text"]
        exact.setdefault(text, index)
        prefix.setdefault(text[:PREFIX_MATCH_CHARS], index)

    matched: set[int] = set()
    unmatched = 0
    for text in annotation_texts:
        row_index = exact.get(text)
        if row_index is None:
            row_index = prefix.get(text[:PREFIX_MATCH_CHARS])
        if row_index is None:
            unmatched += 1
        else:
            matched.add(row_index)
    return sorted(matched), unmatched


def load_hand_scope(
    config: rater.Config,
) -> tuple[rater.RatedDataset, list[int], int]:
    dataset = rater.read_rated_dataset(config.output_jsonl)
    if not dataset.rows:
        raise SystemExit(
            f"Rated file has no document rows: {config.output_jsonl}. The hand rater "
            "never creates or extends it."
        )
    annotation_texts = read_annotation_texts(config.hand_annotations_jsonl)
    matched_indexes, unmatched = match_annotations_to_rows(annotation_texts, dataset.rows)
    return dataset, matched_indexes, unmatched


def hand_stats(
    config: rater.Config,
    dataset: rater.RatedDataset,
    matched_indexes: list[int],
) -> rater.MissingStats:
    return rater.missing_stats(config, [dataset.rows[index] for index in matched_indexes])


def hard_refresh(
    config: rater.Config,
    dataset: rater.RatedDataset,
    matched_indexes: list[int],
) -> int:
    """Remove every rating from matched rows and persist before any API calls."""
    replacements: dict[int, dict[str, Any]] = {}
    cleared = 0
    for index in matched_indexes:
        if "ratings" not in dataset.rows[index]:
            continue
        row = copy.deepcopy(dataset.rows[index])
        row.pop("ratings", None)
        replacements[index] = row
        cleared += 1
    rater.persist_replacements(config, dataset, replacements)
    return cleared


async def fill_hand_ratings(
    config: rater.Config,
    dataset: rater.RatedDataset,
    matched_indexes: list[int],
) -> None:
    await rater.rate_selected_rows(
        config,
        dataset,
        matched_indexes,
        progress_description="Hand rating",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=rater.DEFAULT_CONFIG,
        help=f"Rating config (default: {rater.DEFAULT_CONFIG})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--hard-refresh",
        action="store_true",
        help="Delete all ratings on matched hand rows, persist, then refill them.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Print aggregate hand-scope missing counts without any writes.",
    )
    args = parser.parse_args()

    config = rater.load_config(args.config.resolve())

    if args.check:
        dataset, matched_indexes, _ = load_hand_scope(config)
        print(rater.format_missing_stats(hand_stats(config, dataset, matched_indexes)))
        return

    # The backup intentionally happens before reading annotations, matching the
    # requested startup semantics for every mutating invocation.
    rater.create_startup_backup(config)
    dataset, matched_indexes, unmatched = load_hand_scope(config)
    if unmatched:
        print(
            f"Skipped {unmatched} hand annotations not currently present in "
            f"{config.output_jsonl}; the hand rater never extends that file."
        )

    if args.hard_refresh:
        cleared = hard_refresh(config, dataset, matched_indexes)
        print(
            f"Removed every rating from {cleared} matched hand rows and saved the "
            "live rated file; beginning refill."
        )

    stats_before = hand_stats(config, dataset, matched_indexes)
    if stats_before.missing_ratings:
        asyncio.run(fill_hand_ratings(config, dataset, matched_indexes))
    print(rater.format_missing_stats(hand_stats(config, dataset, matched_indexes)))


if __name__ == "__main__":
    main()
