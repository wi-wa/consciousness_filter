#!/usr/bin/env python3
"""Re-rate only the hand-annotated samples, replacing their old model ratings.

Matches every sample in the hand-annotation JSONL to its row in the rated
output JSONL (exact text match, falling back to a 200-character prefix match
like the viewer), requests a fresh judgement for every (filter, model) pair in
the config, and rewrites the rated file with those rows' "ratings" replaced
wholesale. Everything else is preserved:

  * The hand-annotation file is only ever opened for reading; the hand labels
    ("experience-rating", "pom-rating", "reification-rating") live there and
    are never touched.
  * If the rated file is missing, or an annotated document lies beyond its
    last row, unrated stub rows are seeded from the input JSONL (raw lines,
    appended only at the end) so the file stays the line-aligned prefix of
    the input that the main rater's resume logic requires. The main rater
    later rates stubs like any other pending document.
  * Unmatched rows of the rated file are copied byte-for-byte.
  * All rating calls happen before the rated file is touched, and the rewrite
    goes through a temp file + atomic replace, with the previous version kept
    at <output>.pre_rerate.bak.

Usage:
    python scripts/rerate_hand_annotated.py [--config config.json]
        [--annotations data/hand_annotated_samples.jsonl] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rate_filters_openrouter as rater

PREFIX_MATCH_CHARS = 200  # same fallback the viewer uses


def read_annotation_texts(annotations_path: Path) -> list[str]:
    if not annotations_path.exists():
        raise SystemExit(f"Annotation file does not exist: {annotations_path}")

    texts: list[str] = []
    with annotations_path.open("r", encoding="utf-8") as annotations_file:
        for line_number, line in enumerate(annotations_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping unparseable annotation line {line_number}: {exc}")
                continue
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                print(f"WARNING: skipping annotation line {line_number}: no string 'text' field.")
                continue
            texts.append(row["text"])
    return texts


def match_annotations_to_rows(
    annotation_texts: list[str], rated_rows: list[dict[str, Any] | None]
) -> tuple[dict[int, int], list[int]]:
    """Map rated-file line index -> annotation index; also return unmatched annotations."""
    exact = {row["text"]: i for i, row in enumerate(rated_rows) if row is not None}
    prefix: dict[str, int] = {}
    for i, row in enumerate(rated_rows):
        if row is not None:
            prefix.setdefault(row["text"][:PREFIX_MATCH_CHARS], i)

    matched: dict[int, int] = {}
    unmatched: list[int] = []
    for annotation_index, text in enumerate(annotation_texts):
        row_index = exact.get(text)
        if row_index is None:
            row_index = prefix.get(text[:PREFIX_MATCH_CHARS])
        if row_index is None:
            unmatched.append(annotation_index)
        elif row_index in matched:
            print(
                f"WARNING: annotations {matched[row_index] + 1} and {annotation_index + 1} "
                f"match the same rated row {row_index + 1}; re-rating it once."
            )
        else:
            matched[row_index] = annotation_index
    return matched, unmatched


def locate_annotations_in_input(
    input_path: Path, annotation_texts: list[str]
) -> int | None:
    """Return the highest input line number needed to cover the annotations.

    Mirrors match_annotations_to_rows: each annotation resolves to its first
    exact-text match in the input, falling back to its first 200-character
    prefix match (for hand-edited tails). Annotations found nowhere are
    ignored here; the matching step warns about them later. Stops reading
    once every annotation has an exact match, so the common case never scans
    far past the annotated region.
    """
    exact_targets: dict[str, list[int]] = {}
    prefix_targets: dict[str, list[int]] = {}
    for index, text in enumerate(annotation_texts):
        exact_targets.setdefault(text, []).append(index)
        prefix_targets.setdefault(text[:PREFIX_MATCH_CHARS], []).append(index)

    exact_line: dict[int, int] = {}
    prefix_line: dict[int, int] = {}
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            for index in exact_targets.get(text, ()):
                exact_line.setdefault(index, line_number)
            for index in prefix_targets.get(text[:PREFIX_MATCH_CHARS], ()):
                prefix_line.setdefault(index, line_number)
            if len(exact_line) == len(annotation_texts):
                break

    resolved = [
        exact_line.get(index, prefix_line.get(index))
        for index in range(len(annotation_texts))
    ]
    lines_needed = [line_number for line_number in resolved if line_number is not None]
    return max(lines_needed) if lines_needed else None


def read_rated_rows_seeding_stubs(
    config: rater.Config, annotation_texts: list[str]
) -> list[str]:
    """Read the rated file's raw lines, seeding stubs to cover the annotations.

    When the rated file is missing or ends before some annotated document's
    input line, the input's raw lines are appended (in memory) as unrated stub
    rows up to the furthest annotated document, keeping the file a
    line-aligned prefix of the input. Nothing is written here: the stubs reach
    disk in the final atomic rewrite, after all rating calls have finished.
    """
    output_path = config.output_jsonl
    raw_lines: list[str] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as output_file:
            raw_lines = output_file.readlines()

    rows_needed = locate_annotations_in_input(config.input_jsonl, annotation_texts)
    if rows_needed is None or rows_needed <= len(raw_lines):
        return raw_lines

    stub_lines: list[str] = []
    with config.input_jsonl.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line_number > rows_needed:
                break
            if line_number > len(raw_lines):
                stub_lines.append(line if line.endswith("\n") else line + "\n")

    if raw_lines and not raw_lines[-1].endswith("\n"):
        raw_lines[-1] += "\n"

    action = "extended" if raw_lines else "created"
    print(
        f"Rated file will be {action} with {len(stub_lines)} unrated stub rows "
        f"(input lines {len(raw_lines) + 1}-{rows_needed}) so every annotated "
        "document has a row. The main rater rates stubs like any other pending "
        "document."
    )
    return raw_lines + stub_lines


def scrub_salvage_sidecar(config: rater.Config, texts: list[str]) -> None:
    """Drop the re-rated documents from any leftover salvage sidecar.

    Otherwise a sidecar left by an interrupted main run could re-inject the
    old, deleted ratings on the next run.
    """
    salvage_path = rater.salvage_path_for(config)
    if not salvage_path.exists():
        return
    salvage = rater.load_salvage(salvage_path)
    digests = {rater.text_digest(text) for text in texts}
    kept = {digest: entry for digest, entry in salvage.items() if digest not in digests}
    if len(kept) == len(salvage):
        return
    if kept:
        rater.write_salvage(salvage_path, kept)
    else:
        salvage_path.unlink()
    print(
        f"Removed {len(salvage) - len(kept)} stale entries for re-rated documents "
        f"from {salvage_path}."
    )


def rewrite_rated_file(
    output_path: Path, raw_lines: list[str], replacements: dict[int, dict[str, Any]]
) -> None:
    backup_path: Path | None = None
    if output_path.exists():
        backup_path = output_path.with_name(output_path.name + ".pre_rerate.bak")
        shutil.copy2(output_path, backup_path)

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as tmp_file:
        for line_index, raw_line in enumerate(raw_lines):
            row = replacements.get(line_index)
            if row is None:
                tmp_file.write(raw_line)
                if not raw_line.endswith("\n"):
                    tmp_file.write("\n")
            else:
                tmp_file.write(json.dumps(row, ensure_ascii=False))
                tmp_file.write("\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    os.replace(tmp_path, output_path)
    if backup_path is not None:
        print(f"Updated {output_path} (previous version saved to {backup_path}).")
    else:
        print(f"Created {output_path}.")


async def rerate(config: rater.Config, annotations_path: Path, dry_run: bool) -> None:
    annotation_texts = read_annotation_texts(annotations_path)
    if not annotation_texts:
        raise SystemExit(f"No usable samples found in {annotations_path}.")

    output_path = config.output_jsonl
    raw_lines = read_rated_rows_seeding_stubs(config, annotation_texts)
    if not raw_lines:
        raise SystemExit(
            f"{output_path} has no rows and no annotated document could be "
            f"located in {config.input_jsonl}; nothing to re-rate."
        )

    rated_rows: list[dict[str, Any] | None] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            row = None
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            print(f"WARNING: rated file line {line_number} is not a valid document row.")
            row = None
        rated_rows.append(row)

    matched, unmatched = match_annotations_to_rows(annotation_texts, rated_rows)
    for annotation_index in unmatched:
        snippet = annotation_texts[annotation_index][:60].replace("\n", " ")
        print(
            f"WARNING: annotation {annotation_index + 1} has no matching row in "
            f"{output_path}; skipping it. Text starts: {snippet!r}"
        )
    if not matched:
        raise SystemExit("No annotated sample matched a rated row; nothing to do.")

    pairs = [(spec, model) for spec in config.filters for model in config.models]
    print(
        f"Re-rating {len(matched)} of {len(annotation_texts)} hand-annotated samples: "
        f"{len(pairs)} (filter, model) pairs each, {len(matched) * len(pairs)} calls total."
    )
    if dry_run:
        print("Dry run: no API calls made, no files changed.")
        return

    # Fresh copies with the old model ratings deleted; every pair is requested.
    batch: list[tuple[int, dict[str, Any], list[tuple[rater.FilterSpec, str]]]] = []
    for row_index in sorted(matched):
        row = copy.deepcopy(rated_rows[row_index])
        row.pop("ratings", None)
        batch.append((row_index + 1, row, list(pairs)))

    api_key = rater.get_api_key(config)
    headers = rater.make_headers(config, api_key)
    semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    timeout = httpx.Timeout(config.request_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with tqdm(total=len(batch) * len(pairs), unit="call", desc="Re-rating") as progress:
            rated_rows_new, failed_pairs = await rater.rate_batch(
                client=client,
                headers=headers,
                config=config,
                semaphore=semaphore,
                batch=batch,
                progress=progress,
            )

    replacements = {
        line_number - 1: row
        for (line_number, _, _), row in zip(
            batch, rated_rows_new, strict=True
        )
    }
    for (line_number, _, _), row in zip(batch, rated_rows_new, strict=True):
        original = rated_rows[line_number - 1]
        if row.get("text") != original.get("text"):
            raise SystemExit(
                f"Refusing to write: re-rated row for line {line_number} does not "
                "match the original text."
            )

    rewrite_rated_file(output_path, raw_lines, replacements)
    scrub_salvage_sidecar(config, [row.get("text", "") for row in rated_rows_new])

    print(f"Replaced model ratings on {len(replacements)} rows.")
    if failed_pairs:
        print(
            f"WARNING: {failed_pairs} (filter, model) pairs failed all retries and "
            "were left unrated on their rows. Run this script again to redo those "
            "documents (all pairs are always re-requested)."
        )
    print(f"Hand annotations in {annotations_path} were not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the JSON config file (default: config.json)",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/hand_annotated_samples.jsonl"),
        help="Hand-annotation JSONL (read-only; default: data/hand_annotated_samples.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-rated without calling the API or writing files.",
    )
    args = parser.parse_args()

    config = rater.load_config(args.config)
    asyncio.run(rerate(config, args.annotations, args.dry_run))


if __name__ == "__main__":
    main()
