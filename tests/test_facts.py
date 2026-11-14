"""Fact lookup uses real local persistence and refuses unsupported citations."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase

import asyncio
import importlib
import json
import sys


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_fact_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
PhotographyEngine = importlib.import_module(PACKAGE + ".workflow").PhotographyEngine
Settings = importlib.import_module(PACKAGE + ".config").Settings


class FactEngine(PhotographyEngine):
    def __init__(self, directory: Path) -> None:
        super().__init__(SimpleNamespace(paths=SimpleNamespace(data_dir=directory)), Settings(daily_enabled=False))
        self.reader = SimpleNamespace(search=self.search, page=self.page)
        self.calls: List[str] = []
        self.model_inputs: List[Dict[str, Any]] = []
        self.search_results = [{"url": "https://publisher.example/book", "title": "出版资料"}]
        self.redirects: Dict[str, str] = {}
        self.judgement = {"resolved": True, "conclusion": "两种标题在这份出版资料中指向同一本书。",
                          "source_ids": None}

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        self.calls.append(query)
        return self.search_results[:limit]

    async def page(self, url: str) -> Dict[str, Any]:
        return {"url": self.redirects.get(url, url), "title": "出版资料", "text": "原名与译名的出版说明"}

    async def _model(self, purpose: str, data: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
        assert purpose == "fact_check"
        self.model_inputs.append(data)
        result = dict(self.judgement)
        if "source_ids" in result and result["source_ids"] is None:
            result["source_ids"] = [data["pages"][0]["source_id"]]
        return result


class FactTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.engine = FactEngine(Path(self.temp.name))
        await asyncio.to_thread(self.engine.store.initialize)
        self.job = {"id": "fact-job", "kind": "fact_check", "status": "running", "errors": []}
        self.item = {"question": "某影集的两个名字是否对应？", "search_query": "artist book publisher"}

    async def test_resolved_lookup_retains_sources_and_is_not_repeated(self) -> None:
        await self.engine._queue_fact_checks([self.item, self.item], self.job)
        await self.engine._check_facts(self.job)
        records = await self.engine.db("list", "fact_check")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "resolved")
        self.assertEqual(records[0]["sources"], ["https://publisher.example/book"])
        self.assertEqual(records[0]["source_ids"], [self.engine.model_inputs[0]["pages"][0]["source_id"]])
        self.assertEqual(records[0]["checked_job_id"], self.job["id"])
        await self.engine._queue_fact_checks([self.item], self.job)
        await self.engine._check_facts(self.job)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(await self.engine.db("list", "profile"), [])
        self.assertEqual(await self.engine.db("list", "diary"), [])

    async def test_guessed_source_is_a_visible_failure(self) -> None:
        self.engine.judgement["source_ids"] = ["source-unknown"]
        await self.engine._queue_fact_checks([self.item], self.job)
        with self.assertRaisesRegex(ValueError, "未知编号"):
            await self.engine._check_facts(self.job)
        record = (await self.engine.db("list", "fact_check"))[0]
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["error"])
        self.assertEqual(record["sources"], [])

    async def test_unresolved_facts_remain_pending_for_a_later_day(self) -> None:
        self.engine.judgement = {"resolved": False, "conclusion": "出版资料尚不足以确认译名对应。", "source_ids": []}
        await self.engine._queue_fact_checks([self.item], self.job)
        await self.engine._check_facts(self.job)
        record = (await self.engine.db("list", "fact_check"))[0]
        self.assertEqual(record["status"], "unresolved")
        await self.engine._check_facts(self.job)
        self.assertEqual(len(self.engine.calls), 1)

    async def test_source_ids_map_to_opened_redirect_url_without_model_rewriting(self) -> None:
        actual = "https://publisher.example/Book?edition=1%2F2"
        self.engine.redirects["https://publisher.example/book"] = actual
        await self.engine._queue_fact_checks([self.item], self.job)
        await self.engine._check_facts(self.job)
        record = (await self.engine.db("list", "fact_check"))[0]
        self.assertEqual(record["sources"], [actual])
        first_id = record["source_ids"][0]
        await self.engine._queue_fact_checks([{"question": "Another edition?", "search_query": "another edition"}], self.job)
        await self.engine._check_facts(self.job)
        self.assertEqual(self.engine.model_inputs[-1]["pages"][0]["source_id"], first_id)

    async def test_old_url_output_and_empty_resolved_citations_fail_explicitly(self) -> None:
        await self.engine._queue_fact_checks([self.item], self.job)
        self.engine.judgement = {"resolved": True, "conclusion": "Claim", "sources": ["https://publisher.example/book"]}
        with self.assertRaisesRegex(ValueError, "source_ids"):
            await self.engine._check_facts(self.job)
        await self.engine._prepare_fact_retry(self.job)
        self.engine.judgement = {"resolved": True, "conclusion": "Claim", "source_ids": []}
        with self.assertRaisesRegex(ValueError, "实际来源"):
            await self.engine._check_facts(self.job)

    async def test_recent_observations_supply_provenance_without_asserting_unknown_authors(self) -> None:
        for number in range(37):
            await self.engine.db("put", "observation", {
                "id": f"observation-{number:02}", "job_id": "earlier-viewing", "author": "Unconfirmed attribution",
                "work": "Tentative title", "title": "A child on sand", "identity_verified": False,
                "source_url": f"https://museum.example/item/{number}", "image_url": f"https://museum.example/photo/{number}.jpg",
                "observations": "A child raises both hands on sand.", "raw_prompt": "private model input", "image_data": "base64 bytes",
            })
        await self.engine._queue_fact_checks([self.item], self.job)
        await self.engine._check_facts(self.job)
        context = self.engine.model_inputs[0]["recent_observations"]
        self.assertEqual(len(context), 36)
        self.assertNotIn("observation-00", {item["id"] for item in context})
        self.assertEqual(context[0]["page_url"], "https://museum.example/item/36")
        self.assertEqual(context[0]["image_url"], "https://museum.example/photo/36.jpg")
        self.assertFalse(context[0]["identity_verified"])
        self.assertEqual(context[0]["author"], "Unconfirmed attribution")
        self.assertNotIn("raw_prompt", context[0])
        self.assertNotIn("image_data", context[0])

    async def test_workflow_retry_scopes_legacy_failed_and_unresolved_checks(self) -> None:
        checked = datetime.now(timezone.utc).isoformat()
        self.job = {"id": "facts-review-source", "kind": "fact_check", "status": "failed", "errors": []}
        await self.engine.db("put", "job", self.job)
        cases = [("unrelated-pending", "other-review", "pending"), ("same-source-pending", "review-source", "pending"),
                 ("legacy-failed", "review-source", "failed"), ("legacy-unresolved", "review-source", "unresolved"),
                 ("already-resolved", "review-source", "resolved"), ("unrelated-failed", "other-review", "failed")]
        for ident, source, status in cases:
            await self.engine.db("put", "fact_check", {"id": ident, "job_id": source, "status": status,
                "question": ident, "search_query": ident, "attempts": 1, "checked_at": "" if status == "pending" else checked,
                "conclusion": "Earlier result", "sources": [], "error": ""})
        # retry() must use the legacy facts-<source> relationship, not require a new checked_job_id field.
        await self.engine.retry(self.job["id"])
        retried = await self.engine.db("get", "job", self.job["id"])
        self.assertEqual(set(retried["retry_fact_ids"]), {"legacy-failed", "legacy-unresolved"})
        await self.engine._check_facts(retried)
        self.assertEqual(set(self.engine.calls), {"legacy-failed", "legacy-unresolved"})
        for ident in ("legacy-failed", "legacy-unresolved"):
            fact = await self.engine.db("get", "fact_check", ident)
            self.assertEqual(fact["status"], "resolved")
            self.assertEqual(fact["attempts"], 2)
        self.assertEqual((await self.engine.db("get", "fact_check", "unrelated-failed"))["attempts"], 1)
        self.assertEqual((await self.engine.db("get", "fact_check", "already-resolved"))["attempts"], 1)
        self.assertEqual((await self.engine.db("get", "fact_check", "unrelated-pending"))["status"], "pending")
        self.assertEqual((await self.engine.db("get", "fact_check", "same-source-pending"))["status"], "pending")
        self.assertEqual(retried["retry_fact_ids"], [])

    async def test_explicit_retry_bypasses_cooldown_only_once_for_current_task(self) -> None:
        self.engine.judgement = {"resolved": False, "conclusion": "Still insufficient", "source_ids": []}
        await self.engine._queue_fact_checks([self.item], self.job)
        await self.engine._check_facts(self.job)
        self.job["status"] = "failed"
        await self.engine.db("put", "job", self.job)
        await self.engine.retry(self.job["id"])
        retry_job = await self.engine.db("get", "job", self.job["id"])
        await self.engine._check_facts(retry_job)
        await self.engine._check_facts(retry_job)
        self.assertEqual(len(self.engine.calls), 2)

    def test_all_language_prompts_use_source_ids_and_observation_context(self) -> None:
        prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
        for language in ("zh-CN", "en", "ja"):
            self.assertIn("source_ids", prompts[language]["fact_check"])
            self.assertIn("recent_observations", prompts[language]["fact_check"])
            self.assertIn("identity_verified=false", prompts[language]["fact_check"])
