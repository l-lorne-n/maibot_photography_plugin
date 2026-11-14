"""Offline image embedding cache and same-space retrieval behavior."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase, TestCase

from PIL import Image

import importlib
import json
import sys


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_visual_memory_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
module = importlib.import_module(PACKAGE + ".visual_memory")
Store = importlib.import_module(PACKAGE + ".store").Store
ModelFailure = importlib.import_module(PACKAGE + ".errors").ModelFailure


def picture(color: str) -> Dict[str, Any]:
    output = BytesIO()
    Image.new("RGB", (90, 140), color).save(output, format="PNG")
    return {"data": output.getvalue(), "mime_type": "image/png", "width": 90, "height": 140}


def response(vector: List[float], **changes: Any) -> Dict[str, Any]:
    result = {
        "success": True, "embedding": vector, "model_name": "qwen3-vl-embedding",
        "model_identifier": "qwen3-vl-embedding", "api_provider": "Alibaba",
        "preprocess_version": module.PREPROCESS_VERSION, "request_protocol_hash": "native-dashscope-v1",
        "embedding_fingerprint": {
            "version": "embedding-model-v1", "hash": "configured-space-1", "model": "qwen3-vl-embedding",
            "provider": "Alibaba", "model_identifier": "qwen3-vl-embedding", "dimension": len(vector),
            "dimension_request_mode": "native", "source": "image_embedding",
        },
    }
    result.update(changes)
    return result


class FakeLLM:
    def __init__(self, results: List[Dict[str, Any]]) -> None:
        self.results = deepcopy(results)
        self.calls: List[Dict[str, Any]] = []

    async def embed(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        if not self.results:
            raise AssertionError("Unexpected embedding request")
        return self.results.pop(0)


class FakeStore:
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.aliases: Dict[str, str] = {}
        self.requested_limits: List[int] = []

    def resolve_image_id(self, work_id: str) -> str:
        return self.aliases.get(work_id, work_id)

    def get(self, kind: str, key: str) -> Any:
        return deepcopy(self.records.get(kind, {}).get(key))

    def put(self, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
        self.records.setdefault(kind, {})[record["id"]] = deepcopy(record)
        return deepcopy(record)

    def count(self, kind: str) -> int:
        return len(self.records.get(kind, {}))

    def list(self, kind: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.requested_limits.append(limit)
        return list(self.records.get(kind, {}).values())[:limit]


def cached_record(record_id: str, work_id: str, vector: List[float], **identity_changes: Any) -> Dict[str, Any]:
    value = response(vector, **identity_changes)
    identity = module._space(value, len(vector))
    return {**identity, "id": record_id, "embedding": module.normalized_vector(vector), "work_id": work_id,
            "work_ids": [work_id], "variant_id": "variant-" + record_id, "variant_ids": ["variant-" + record_id],
            "input_hash": "input-" + record_id, "metadata_by_work": {work_id: {"author": "Archived author"}}}


class VectorValidationTests(TestCase):
    def test_only_finite_nonzero_dense_vectors(self) -> None:
        self.assertEqual(module.normalized_vector([3, 4]), [0.6, 0.8])
        for value in ([], [0, 0], [True, 1], [float("nan")], [float("inf")], ["1"], None, (1, 2)):
            with self.subTest(value=value), self.assertRaises(ModelFailure):
                module.normalized_vector(value)


class CacheTests(IsolatedAsyncioTestCase):
    async def test_cache_uses_prepared_image_and_current_space_with_actual_store(self) -> None:
        with TemporaryDirectory() as directory:
            store = Store(Path(directory))
            store.initialize()
            llm = FakeLLM([response([3, 4]), response([3, 4])])
            memory = module.VisualMemory(llm, store)
            first = await memory.embed_image(picture("red"), work_id="a" * 64, variant_id="1" * 64,
                metadata={"author": "Alec", "observations": "red", "api_key": "do-not-store", "messages": "private"})
            second = await memory.embed_image(picture("red"), work_id="a" * 64, variant_id="1" * 64)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(len(llm.calls), 1)
            self.assertEqual(second["metadata_by_work"]["a" * 64]["author"], "Alec")
            self.assertEqual(llm.calls[0]["task_name"], "image_embedding")
            self.assertEqual(llm.calls[0]["mime_type"], "image/jpeg")
            self.assertNotIn("text", llm.calls[0])
            self.assertEqual(store.count(module.RECORD_KIND), 1)
            serialized = json.dumps(store.list(module.RECORD_KIND))
            self.assertNotIn("do-not-store", serialized)
            self.assertNotIn("private", serialized)
            self.assertNotIn("image_base64", serialized)
            await memory.begin_run()
            third = await memory.embed_image(picture("red"), work_id="a" * 64)
            self.assertFalse(third["cache_hit"])
            self.assertEqual(len(llm.calls), 2)
            self.assertEqual(third["space_id"], first["space_id"])

    async def test_space_change_is_rejected_mid_run_and_allowed_at_next_run(self) -> None:
        changed = response([1, 0])
        changed["embedding_fingerprint"]["hash"] = "new-provider-settings"
        llm = FakeLLM([response([1, 0]), changed, changed])
        store = FakeStore()
        memory = module.VisualMemory(llm, store)
        old = await memory.embed_image(picture("red"), work_id="red")
        with self.assertRaisesRegex(ModelFailure, "空间已改变"):
            await memory.embed_image(picture("blue"), work_id="blue")
        self.assertEqual(store.count(module.RECORD_KIND), 1)
        await memory.begin_run()
        new = await memory.embed_image(picture("blue"), work_id="blue")
        self.assertNotEqual(old["space_id"], new["space_id"])
        self.assertEqual(await memory.search(new), [])

    async def test_failed_wrong_model_and_wrong_protocol_are_not_cached(self) -> None:
        wrong_model = response([1, 0], model_name="text-model")
        wrong_model["embedding_fingerprint"]["model"] = "text-model"
        for result in ({"success": False, "error": "provider rejected"}, wrong_model,
                       response([1, 0], preprocess_version="different-preprocessor"), response([0, 0])):
            with self.subTest(result=result):
                store = FakeStore()
                memory = module.VisualMemory(FakeLLM([result]), store)
                with self.assertRaises(ModelFailure):
                    await memory.embed_image(picture("red"), work_id="red")
                self.assertEqual(store.count(module.RECORD_KIND), 0)

    async def test_alias_resolved_before_association(self) -> None:
        store = FakeStore()
        store.aliases["old"] = "canonical"
        memory = module.VisualMemory(FakeLLM([response([1, 0])]), store)
        record = await memory.embed_image(picture("red"), work_id="old")
        self.assertEqual(record["work_id"], "canonical")
        self.assertEqual(record["work_ids"], ["canonical"])
        self.assertEqual(await memory.search(record), [])


class RetrievalTests(IsolatedAsyncioTestCase):
    async def test_same_space_full_cosine_ranks_deduplicates_aliases_and_excludes_self(self) -> None:
        store = FakeStore()
        store.aliases.update({"old-query": "query", "old-a": "a"})
        query = cached_record("q", "old-query", [1, 0])
        for record in [query, cached_record("a1", "old-a", [0.8, 0.6]), cached_record("a2", "a", [0.6, 0.8]),
                       cached_record("b", "b", [-1, 0]), cached_record("x", "x", [1, 0], api_provider="other",
                           embedding_fingerprint={**response([1, 0])["embedding_fingerprint"], "provider": "other"}),
                       cached_record("p", "p", [1, 0], preprocess_version="different"),
                       cached_record("r", "r", [1, 0], request_protocol_hash="different")]:
            store.put(module.RECORD_KIND, record)
        store.put("image", {"id": "a", "author": "Current author", "observations": "Current observation"})
        snapshot = deepcopy(store.records)
        memory = module.VisualMemory(FakeLLM([]), store)
        matches = await memory.search(query, limit=10)
        self.assertEqual([item["work_id"] for item in matches], ["a", "b"])
        self.assertAlmostEqual(matches[0]["similarity"], 0.8)
        self.assertEqual(matches[0]["metadata"]["author"], "Current author")
        self.assertEqual(matches[0]["variant_id"], "variant-a1")
        self.assertEqual(store.records, snapshot)
        self.assertEqual(await memory.search(query, exclude_work_ids=["old-a"], min_similarity=0), [])

    async def test_search_scans_beyond_default_and_ten_thousand_rows(self) -> None:
        store = FakeStore()
        prototype = cached_record("ordinary", "ordinary", [0, 1])
        for index in range(10001):
            record = {**prototype, "id": str(index)}
            store.records.setdefault(module.RECORD_KIND, {})[str(index)] = record
        store.put(module.RECORD_KIND, cached_record("last", "best", [1, 0]))
        query = cached_record("q", "query", [1, 0])
        memory = module.VisualMemory(FakeLLM([]), store)
        matches = await memory.search(query, limit=1)
        self.assertEqual(matches[0]["work_id"], "best")
        self.assertEqual(store.requested_limits, [10002])

    async def test_corrupted_same_space_fails_instead_of_scoring(self) -> None:
        store = FakeStore()
        query = cached_record("q", "query", [1, 0])
        corrupt = cached_record("c", "candidate", [1, 0])
        corrupt["embedding"] = [1, 0, 0]
        store.put(module.RECORD_KIND, corrupt)
        memory = module.VisualMemory(FakeLLM([]), store)
        with self.assertRaises(ModelFailure):
            await memory.search(query)
        for kwargs in ({"limit": 0}, {"limit": True}, {"min_similarity": float("nan")}, {"min_similarity": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await memory.search(query, **kwargs)
