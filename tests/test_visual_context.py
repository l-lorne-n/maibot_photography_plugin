"""Historical real images reach visual comparison, reflection, and the index job."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from PIL import Image

import asyncio
import importlib
import importlib.util
import sys


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_visual_context_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
context_module = importlib.import_module(PACKAGE + ".visual_context")
memory_module = importlib.import_module(PACKAGE + ".visual_memory")
PhotographyEngine = importlib.import_module(PACKAGE + ".workflow").PhotographyEngine
fixture_spec = importlib.util.spec_from_file_location("photography_browsing_fixtures", ROOT / "tests" / "test_browsing.py")
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)


def picture(color: str) -> Dict[str, Any]:
    output = BytesIO()
    Image.new("RGB", (90, 140), color).save(output, format="PNG")
    return {"data": output.getvalue(), "mime_type": "image/png", "width": 90, "height": 140}


def vector(record_id: str, work_id: str, similarity: float) -> Dict[str, Any]:
    identity = memory_module._space({
        "model_name": "qwen3-vl-embedding", "model_identifier": "qwen3-vl-embedding", "api_provider": "Alibaba",
        "preprocess_version": memory_module.PREPROCESS_VERSION, "request_protocol_hash": "native",
        "embedding_fingerprint": {"hash": "space", "model": "qwen3-vl-embedding", "provider": "Alibaba",
                                  "model_identifier": "qwen3-vl-embedding", "dimension": 2},
    }, 2)
    return {**identity, "id": record_id, "embedding": [similarity, (1 - similarity ** 2) ** 0.5],
            "work_id": work_id, "work_ids": [work_id], "variant_id": record_id,
            "metadata_by_work": {work_id: {"author": "Historical photographer"}}}


class ContextStore:
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.aliases = {"old-history": "history", "old-query": "query", "old-current": "current"}
        self.pictures = {"history": picture("red"), "second": picture("blue"), "current": picture("green")}
        self.limits: Dict[str, int] = {}

    def resolve_image_id(self, identifier: str) -> str:
        return self.aliases.get(identifier, identifier)

    def resolve_image_ids(self, identifiers: List[str]) -> List[str]:
        return list(dict.fromkeys(self.resolve_image_id(identifier) for identifier in identifiers))

    def identify_image(self, data: bytes, mime_type: str) -> Dict[str, Any]:
        with Image.open(BytesIO(data)) as image:
            image.load()
        return {"work_id": "query", "fingerprint": {"pixel_hash": "pixels"}}

    def image_bytes(self, identifier: str):
        image = self.pictures[self.resolve_image_id(identifier)]
        return image["data"], image["mime_type"]

    def get(self, kind: str, identifier: str) -> Any:
        if kind == "image":
            identifier = self.resolve_image_id(identifier)
        return deepcopy(self.records.get(kind, {}).get(identifier))

    def list(self, kind: str, limit: int = 100) -> List[Dict[str, Any]]:
        self.limits[kind] = limit
        return list(self.records.get(kind, {}).values())[:limit]

    def count(self, kind: str) -> int:
        return len(self.records.get(kind, {}))

    def put(self, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
        self.records.setdefault(kind, {})[record["id"]] = deepcopy(record)
        return deepcopy(record)


class FakeMemory(memory_module.VisualMemory):
    def __init__(self, llm: Any, store: ContextStore, **kwargs: Any) -> None:
        super().__init__(llm, store, **kwargs)
        self.requests: List[Any] = []
        self.started = 0

    async def begin_run(self) -> None:
        self.started += 1

    async def embed_image(self, image: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        self.requests.append((image, kwargs))
        return vector("query-vector", "query", 1)


class ContextHarness(context_module.VisualContextMixin):
    def __init__(self) -> None:
        self.store = ContextStore()
        self.ctx = SimpleNamespace(llm=None)
        self.settings = SimpleNamespace(image_embedding_task="image_embedding")
        self.cancelled = set()
        self.saved: List[Dict[str, Any]] = []
        self.visual_memory = FakeMemory(None, self.store)

    async def db(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        methods = {"get": self.store.get, "list": self.store.list, "count": self.store.count, "put": self.store.put}
        return await asyncio.to_thread(methods[operation], *args, **kwargs)

    async def save_job_progress(self, job: Dict[str, Any]) -> None:
        self.saved.append(deepcopy(job))

    def add_image(self, identifier: str, *, status: str = "available", job: str = "old") -> None:
        self.store.put("image", {"id": identifier, "local_status": status, "job_id": job, "width": 90, "height": 140,
                                 "author": identifier, "work": "Book", "observations": "Past observation"})


class ContextTests(IsolatedAsyncioTestCase):
    async def test_eligible_history_ranked_before_limit_and_alias_diary_attached(self) -> None:
        engine = ContextHarness()
        for identifier in ("query", "history", "second"):
            engine.add_image(identifier)
        engine.add_image("evicted", status="evicted")
        with patch.object(context_module, "VisualMemory", FakeMemory):
            await engine._begin_visual_memory()
        engine.add_image("current", job="today")
        # An old collection can be observed or upgraded in today's job and is
        # still historical context. Only today's first collections are new.
        engine.add_image("history", job="today")
        for index in range(8):
            engine.store.put(memory_module.RECORD_KIND, vector(str(index), "uncollected-" + str(index), 0.99))
        for identifier, similarity in (("old-query", 1), ("old-current", 0.98), ("evicted", 0.97),
                                       ("old-history", 0.8), ("second", 0.7)):
            engine.store.put(memory_module.RECORD_KIND, vector(identifier, identifier, similarity))
        engine.store.put("diary", {"id": "old-diary", "title": "Color", "content": "A feeling worth remembering",
                                   "image_ids": ["old-history"]})
        notes, images = await engine._historical_visual_context(picture("black"), {"id": "preview", "author": "New"},
            {"id": "today", "image_ids": ["history", "current"],
             "viewed_work_ids": {"old-history": "p0", "old-current": "p1"},
             "preview_work_ids": {"old-history": "p0", "old-current": "p1"}})
        self.assertEqual([item["image_id"] for item in notes], ["history", "second"])
        self.assertEqual(notes[0]["diaries"][0]["id"], "old-diary")
        self.assertEqual([item["id"] for item in images], ["history", "second"])
        for image in images:
            with Image.open(BytesIO(image["data"])) as decoded:
                decoded.load()
        link = engine.store.get("visual_link", "preview")
        self.assertEqual([item["work_id"] for item in link["neighbours"]], ["history", "second"])
        self.assertEqual(engine.store.limits["image"], engine.store.count("image"))
        self.assertEqual(engine.store.limits["diary"], engine.store.count("diary"))

    async def test_index_every_available_image_once_with_source_metadata(self) -> None:
        engine = ContextHarness()
        engine.add_image("history")
        engine.add_image("second")
        engine.add_image("gone", status="evicted")
        job = {"id": "index"}
        with patch.object(context_module, "VisualMemory", FakeMemory):
            await engine._index_visual_memory(job)
        self.assertEqual(engine.visual_memory.started, 1)
        self.assertEqual([request[1]["work_id"] for request in engine.visual_memory.requests], ["history", "second"])
        self.assertEqual(engine.visual_memory.requests[0][1]["metadata"]["author"], "history")
        self.assertEqual(job["indexed_count"], 2)
        self.assertEqual(job["status"], "done")
        self.assertEqual(engine.store.limits["image"], 3)

    async def test_index_cancellation_prevents_model_work(self) -> None:
        engine = ContextHarness()
        engine.add_image("history")
        engine.cancelled.add("index")
        with patch.object(context_module, "VisualMemory", FakeMemory):
            await engine._index_visual_memory({"id": "index"})
        self.assertEqual(engine.visual_memory.requests, [])
        self.assertEqual(engine.saved[-1]["status"], "cancelled")


class ComparisonEngine(fixtures.FakeEngine):
    def __init__(self, directory: Path) -> None:
        super().__init__(directory, preview_min=1, preview_max=1, detail_min=1, detail_max=1)
        self.bad_comparison = False
        self.vision_images: List[Dict[str, Any]] = []
        self.reflection_material: Dict[str, Any] = {}

    async def _historical_visual_context(self, actual: Dict[str, Any], preview: Dict[str, Any], job: Dict[str, Any]):
        return ([{"image_id": "history1", "diaries": [{"content": "Past feeling"}]}, {"image_id": "history2"}],
                [{**picture("red"), "id": "history1"}, {**picture("blue"), "id": "history2"}])

    async def _model(self, purpose: str, data: Dict[str, Any], job: Dict[str, Any], image=None, images=None):
        if purpose == "vision":
            self.vision_images = images
            assert images and len(images) == 3 and image is None
            for actual in images:
                with Image.open(BytesIO(actual["data"])) as decoded:
                    decoded.load()
            result = await super()._model(purpose, data, job, image=images[0], images=images)
            result["visual_comparisons"] = [{"image_id": "invented" if self.bad_comparison else note["image_id"],
                "composition": "The figure is similarly off-centre", "color": "This one is warmer",
                "visual_style": "A shared stillness", "preference_insight": "I prefer the extra surrounding space"}
                for note in data["historical_neighbours"]]
            return result
        if purpose == "reflection":
            self.reflection_material = data
            return {"title": "Spacing", "content": "I prefer the quieter frame.", "summary": "More surrounding space"}
        return await super()._model(purpose, data, job, image=image, images=images)


class BrowsingComparisonTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.engine = ComparisonEngine(Path(temporary.name))
        page = self.engine.reader.add_page(0, 1)
        await self.engine.db("put", "source", {"id": page, "url": page, "name": "Photos"})
        self.job = {"id": "today", "kind": "manual", "intent": "Look", "model_calls": 0, "errors": [],
                    "image_ids": [], "observation_ids": [], "status": "running"}

    async def test_current_and_two_history_images_reach_vision_then_validated_comparisons_reach_reflection(self) -> None:
        self.engine.collect = True
        await self.engine._explore(self.job, None)
        self.assertEqual([image["id"] for image in self.engine.vision_images[1:]], ["history1", "history2"])
        call = next(call for call in self.engine.model_inputs if call["purpose"] == "vision")
        self.assertEqual(call["data"]["historical_neighbours"][0]["diaries"][0]["content"], "Past feeling")
        observation = await self.engine.db("get", "observation", self.job["observation_ids"][0])
        self.assertEqual(len(observation["visual_comparisons"]), 2)
        collected = await self.engine.db("get", "image", observation["image_id"])
        self.assertEqual(collected["visual_comparisons"], observation["visual_comparisons"])
        await PhotographyEngine._reflect(self.engine, self.job)
        self.assertEqual(self.engine.reflection_material["observations"][0]["visual_comparisons"],
                         observation["visual_comparisons"])
        self.assertEqual(await self.engine.db("count", "diary"), 1)

    async def test_unknown_comparison_id_is_not_saved_as_observation(self) -> None:
        self.engine.bad_comparison = True
        with self.assertRaisesRegex(fixtures.ModelFailure, "未知或重复"):
            await self.engine._explore(self.job, None)
        self.assertEqual(await self.engine.db("count", "observation"), 0)
        self.assertEqual(self.job["last_model_error"]["purpose"], "vision")
