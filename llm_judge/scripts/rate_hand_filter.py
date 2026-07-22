#!/usr/bin/env python3
"""Rate the hand-annotated documents into their own standalone JSONL.

This is the hand-label counterpart of rate_filters_openrouter.py, and it is
fully independent of the corpus rated output: it reads documents from
``hand_annotations_jsonl`` and writes ``hand_rated_jsonl``, never touching
``output_jsonl``. There is no matching step and no max_documents boundary, so
every hand-annotated document is rated regardless of where it sits in the
corpus.

Rows are copied verbatim from the hand-annotation file, which keeps the human
label fields (``pom-rating`` and friends) next to the judge ratings:

    {
      "text": "...",
      "experience-rating": 1, "pom-rating": 1, "reification-rating": 1,
      "ratings": {"philosophy_of_mind": [{"model": "...", "rating": 7, ...}, ...]}
    }

That single file backs both the viewer's hand-label page and the embedding
model's validation split, so neither has to join two files at load time.

The hand-annotation file is always read-only. New annotations appended to it
are added to the rated output on the next run and then rated; blank lines in it
are skipped. Editing or reordering existing annotations is refused, because the
rated output is aligned to the annotation file line for line.

Default behavior fills only missing configured (filter, model) entries.
``--hard-refresh`` first removes the entire ``ratings`` value from every row,
persists that removal, and then refills configured entries. Both mutating modes
immediately overwrite ``<hand-rated>.bak`` before making changes and persist
each completed API batch to the live rated JSONL.

``--check`` is strictly read-only and prints only aggregate missing counts.
It is mutually exclusive with ``--hard-refresh``.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rate_filters_openrouter as rater


def hand_config(config: rater.Config) -> rater.Config:
    """Point the shared rater machinery at the hand annotation/rated pair."""
    if config.hand_rated_jsonl == config.output_jsonl:
        raise SystemExit(
            "hand_rated_jsonl must differ from output_jsonl; the hand rater owns "
            "its own file and never writes the corpus rated output."
        )
    if config.hand_rated_jsonl == config.hand_annotations_jsonl:
        raise SystemExit(
            "hand_rated_jsonl must differ from hand_annotations_jsonl; the "
            "annotation file is always read-only."
        )
    return replace(
        config,
        input_jsonl=config.hand_annotations_jsonl,
        output_jsonl=config.hand_rated_jsonl,
        # The hand set is small and is always rated in full.
        max_documents=None,
    )


def load_hand_scope(config: rater.Config) -> rater.MainScope:
    return rater.load_main_scope(config, allow_blank_input_lines=True)


def scope_rows(scope: rater.MainScope) -> list[dict[str, Any] | None]:
    return rater.main_scope_rows(scope)


def hard_refresh(config: rater.Config, dataset: rater.RatedDataset) -> int:
    """Remove every rating from the hand rows and persist before any API calls."""
    replacements: dict[int, dict[str, Any]] = {}
    cleared = 0
    for index, row in enumerate(dataset.rows):
        if "ratings" not in row:
            continue
        refreshed = copy.deepcopy(row)
        refreshed.pop("ratings", None)
        replacements[index] = refreshed
        cleared += 1
    rater.persist_replacements(config, dataset, replacements)
    return cleared


async def rate_hand_documents(config: rater.Config, scope: rater.MainScope) -> None:
    await rater.rate_selected_rows(
        config,
        scope.dataset,
        list(range(scope.document_count)),
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
        help="Delete all ratings in the hand rated file, persist, then refill them.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Print aggregate hand-scope missing counts without any writes.",
    )
    args = parser.parse_args()

    config = hand_config(rater.load_config(args.config.resolve()))

    if args.check:
        scope = load_hand_scope(config)
        print(rater.format_missing_stats(rater.missing_stats(config, scope_rows(scope))))
        return

    # The backup intentionally happens before reading annotations, matching the
    # startup semantics of every other mutating invocation.
    rater.create_startup_backup(config)
    scope = load_hand_scope(config)

    added_rows = rater.extend_output_to_scope(config, scope)
    if added_rows:
        print(
            f"Added {added_rows} hand document row(s) to {config.output_jsonl}; "
            "existing rows were preserved."
        )

    if args.hard_refresh:
        cleared = hard_refresh(config, scope.dataset)
        print(
            f"Removed every rating from {cleared} hand row(s) and saved the live "
            "rated file; beginning refill."
        )

    if rater.missing_stats(config, scope_rows(scope)).missing_ratings:
        asyncio.run(rate_hand_documents(config, scope))
    print(rater.format_missing_stats(rater.missing_stats(config, scope_rows(scope))))


if __name__ == "__main__":
    main()
