from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

# File/scoping tests should run without installing network dependencies.
if importlib.util.find_spec("httpx") is None:
    httpx = types.ModuleType("httpx")
    httpx.HTTPError = Exception
    httpx.HTTPStatusError = Exception
    httpx.AsyncClient = object
    httpx.Timeout = object
    sys.modules["httpx"] = httpx
if importlib.util.find_spec("tqdm") is None:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = object
    sys.modules["tqdm"] = tqdm_module

import rate_filters_openrouter as rater
import rerate_hand_annotated as hand_rater


def json_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False) + "\n"


def complete_entry(model: str, rating: int = 5) -> dict:
    return {
        "model": model,
        "rating": rating,
        "explanation": "complete",
        "quote": "",
    }


class RatingWorkflowTests(unittest.TestCase):
    def make_config(
        self,
        directory: Path,
        *,
        max_documents: int | None = 2,
    ) -> rater.Config:
        return rater.Config(
            input_jsonl=directory / "input.jsonl",
            output_jsonl=directory / "rated.jsonl",
            hand_annotations_jsonl=directory / "hand.jsonl",
            filters=[
                rater.FilterSpec(
                    name="filter",
                    prompt_path=directory / "prompt.txt",
                    prompt_template="{document}",
                )
            ],
            grading_instruction="",
            models=["model-a", "model-b"],
            api_key_env="TEST_API_KEY",
            chat_url="https://example.invalid",
            app_title="test",
            http_referer=None,
            reasoning_effort="medium",
            max_output_tokens=100,
            max_retries=1,
            retry_base_delay_seconds=0.0,
            request_timeout_seconds=1.0,
            batch_size=2,
            max_concurrent_requests=2,
            max_documents=max_documents,
        )

    def write_input(self, config: rater.Config, count: int) -> list[dict]:
        rows = [{"text": f"document {index}"} for index in range(1, count + 1)]
        config.input_jsonl.write_text("".join(map(json_line, rows)), encoding="utf-8")
        return rows

    def test_main_check_counts_absent_output_rows_as_fully_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory, max_documents=3)
            rows = self.write_input(config, 3)
            first = copy.deepcopy(rows[0])
            first["ratings"] = {
                "filter": [complete_entry("model-a"), complete_entry("model-b")]
            }
            config.output_jsonl.write_text(json_line(first), encoding="utf-8")

            scope = rater.load_main_scope(config)
            stats = rater.missing_stats(config, rater.main_scope_rows(scope))

            self.assertEqual(stats.missing_documents, 2)
            self.assertEqual(stats.total_documents, 3)
            self.assertEqual(stats.missing_ratings, 4)
            self.assertEqual(stats.total_ratings, 6)

    def test_lower_max_documents_preserves_existing_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory, max_documents=2)
            rows = self.write_input(config, 3)
            original = "".join(map(json_line, rows))
            config.output_jsonl.write_text(original, encoding="utf-8")

            scope = rater.load_main_scope(config)
            added = rater.extend_output_to_scope(config, scope)

            self.assertEqual(added, 0)
            self.assertEqual(config.output_jsonl.read_text(encoding="utf-8"), original)
            self.assertEqual(len(scope.dataset.rows), 3)

    def test_short_output_extends_only_to_max_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory, max_documents=2)
            rows = self.write_input(config, 3)
            config.output_jsonl.write_text(json_line(rows[0]), encoding="utf-8")

            scope = rater.load_main_scope(config)
            added = rater.extend_output_to_scope(config, scope)

            self.assertEqual(added, 1)
            self.assertEqual(len(scope.dataset.rows), 2)
            self.assertEqual(scope.dataset.rows[1]["text"], "document 2")

    def test_fixed_backup_is_overwritten_instead_of_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory)
            config.output_jsonl.write_text("first version\n", encoding="utf-8")
            backup = rater.backup_path_for(config)
            backup.write_text("obsolete backup\n", encoding="utf-8")

            rater.create_startup_backup(config)
            self.assertEqual(backup.read_text(encoding="utf-8"), "first version\n")
            config.output_jsonl.write_text("second version\n", encoding="utf-8")
            rater.create_startup_backup(config)

            self.assertEqual(backup.read_text(encoding="utf-8"), "second version\n")
            self.assertEqual(list(directory.glob("rated.jsonl.bak*")), [backup])

    def test_merge_appends_without_changing_existing_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            config = self.make_config(Path(directory_name))
            row = {
                "text": "document",
                "ratings": {
                    "filter": [
                        {
                            **complete_entry("legacy-model"),
                            "extra": {"must": "survive"},
                        },
                        {"model": "model-a", "rating": "malformed but preserved"},
                    ],
                    "legacy-filter": [{"arbitrary": ["data"]}],
                },
            }
            old_ratings = copy.deepcopy(row["ratings"])

            rater.merge_ratings(
                row,
                config,
                {
                    ("filter", "model-a"): {
                        "rating": 8,
                        "explanation": "new",
                        "quote": "evidence",
                    }
                },
            )

            self.assertEqual(row["ratings"]["filter"][:-1], old_ratings["filter"])
            self.assertEqual(row["ratings"]["legacy-filter"], old_ratings["legacy-filter"])
            self.assertEqual(row["ratings"]["filter"][-1]["model"], "model-a")

    def test_hand_scope_ignores_max_documents_and_hard_refresh_clears_all_ratings(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory, max_documents=1)
            rows = self.write_input(config, 3)
            rated_rows = copy.deepcopy(rows)
            rated_rows[2]["ratings"] = {
                "filter": [complete_entry("model-a"), complete_entry("model-b")],
                "legacy-filter": [{"model": "old", "rating": 99}],
            }
            config.output_jsonl.write_text(
                "".join(map(json_line, rated_rows)), encoding="utf-8"
            )
            annotation_bytes = json_line({"text": rows[2]["text"], "human": 1}).encode()
            config.hand_annotations_jsonl.write_bytes(annotation_bytes)

            rater.create_startup_backup(config)
            dataset, indexes, unmatched = hand_rater.load_hand_scope(config)
            cleared = hand_rater.hard_refresh(config, dataset, indexes)

            self.assertEqual(indexes, [2])
            self.assertEqual(unmatched, 0)
            self.assertEqual(cleared, 1)
            self.assertNotIn("ratings", dataset.rows[2])
            self.assertEqual(config.hand_annotations_jsonl.read_bytes(), annotation_bytes)
            backup_rows = [
                json.loads(line)
                for line in rater.backup_path_for(config).read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("legacy-filter", backup_rows[2]["ratings"])

    def test_concurrent_file_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            path = directory / "rated.jsonl"
            path.write_text('{"text":"new"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "changed while this process"):
                rater.atomic_replace_lines(
                    path,
                    ['{"text":"old"}\n'],
                    ['{"text":"replacement"}\n'],
                )
            self.assertEqual(path.read_text(encoding="utf-8"), '{"text":"new"}\n')

    def test_completed_batches_are_persisted_to_live_file(self) -> None:
        class FakeClient:
            def __init__(self, **_: object) -> None:
                pass

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *_: object) -> None:
                return None

        class FakeProgress:
            def __init__(self, **_: object) -> None:
                pass

            def __enter__(self) -> "FakeProgress":
                return self

            def __exit__(self, *_: object) -> None:
                return None

        async def fake_rate_batch(**kwargs: object) -> tuple[list[dict], int]:
            config = kwargs["config"]
            batch = kwargs["batch"]
            rated_rows: list[dict] = []
            for _, row, pairs in batch:
                new_entries = {
                    (filter_spec.name, model): {
                        "rating": 6,
                        "explanation": "new",
                        "quote": "",
                    }
                    for filter_spec, model in pairs
                }
                rater.merge_ratings(row, config, new_entries)
                rated_rows.append(row)
            return rated_rows, 0

        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            config = self.make_config(directory, max_documents=3)
            rows = self.write_input(config, 3)
            original = "".join(map(json_line, rows))
            config.output_jsonl.write_text(original, encoding="utf-8")
            rater.create_startup_backup(config)

            with (
                mock.patch.object(rater, "get_api_key", return_value="test-key"),
                mock.patch.object(rater.httpx, "Timeout", return_value=object()),
                mock.patch.object(rater.httpx, "AsyncClient", FakeClient),
                mock.patch.object(rater, "tqdm", FakeProgress),
                mock.patch.object(rater, "rate_batch", side_effect=fake_rate_batch),
                mock.patch.object(
                    rater,
                    "persist_replacements",
                    wraps=rater.persist_replacements,
                ) as persist,
            ):
                asyncio.run(rater.rate_jsonl(config))

            rated = rater.read_rated_dataset(config.output_jsonl)
            self.assertTrue(all(not rater.pending_pairs(row, config) for row in rated.rows))
            self.assertEqual(persist.call_count, 2)  # batch_size=2 for three documents
            self.assertEqual(
                rater.backup_path_for(config).read_text(encoding="utf-8"), original
            )


if __name__ == "__main__":
    unittest.main()
