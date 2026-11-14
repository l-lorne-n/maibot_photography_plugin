"""Offline behavioral checks for bounded real-image preview and detail browsing."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase

import asyncio
import importlib
import json
import sys

from PIL import Image, ImageDraw


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_browsing_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
BrowsingMixin = importlib.import_module(PACKAGE + ".browsing").BrowsingMixin
ModelFailure = importlib.import_module(PACKAGE + ".errors").ModelFailure
reader_module = importlib.import_module(PACKAGE + ".reader")
Store = importlib.import_module(PACKAGE + ".store").Store


def picture(index: int, width: int = 120, height: int = 180, compression: int = 6) -> Dict[str, Any]:
    output = BytesIO()
    Image.new("RGB", (width, height), (index % 255, (index * 13) % 255, (index * 37) % 255)).save(
        output, format="PNG", compress_level=compression)
    return {"data": output.getvalue(), "mime_type": "image/png", "width": width, "height": height}


class FakeReader:
    def __init__(self) -> None:
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.images: Dict[str, Any] = {}
        self.calls: List[Any] = []
        self.changed_on_detail = False
        self.image_calls: Dict[str, int] = {}

    def add_page(self, number: int, count: int, *, duplicate: bool = False, logos: bool = False,
                 next_page: str = "") -> str:
        page_url = f"https://photos.example/page/{number}"
        candidates = []
        for index in range(count):
            image_url = f"https://photos.example/{'logo' if logos else 'work'}/{number}-{index}.png"
            candidates.append({"url": image_url, "alt": "Picture"})
            self.images[image_url] = picture(1 if duplicate else 1 + number * 40 + index, compression=1 if index % 2 else 9)
        self.pages[page_url] = {"url": page_url, "title": f"Artist {number} — Book {number}",
                               "text": f"All these works are photographs by Artist {number}, from Book {number}.",
                               "links": [{"url": next_page, "title": "Next photographer"}] if next_page else [],
                               "images": candidates}
        return page_url

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        self.calls.append(("search", query))
        return [{"url": url, "title": page["title"], "snippet": "A real result"} for url, page in self.pages.items()][:limit]

    async def page(self, url: str) -> Dict[str, Any]:
        self.calls.append(("open", url))
        return self.pages[url]

    async def image(self, url: str) -> Dict[str, Any]:
        self.calls.append(("image", url))
        self.image_calls[url] = self.image_calls.get(url, 0) + 1
        item = self.images[url]
        if isinstance(item, Exception):
            raise item
        if self.changed_on_detail and self.image_calls[url] > 1:
            return {**picture(250), "url": url}
        return {**item, "url": url}


class FakeEngine(BrowsingMixin):
    def __init__(self, directory: Path, **targets: int) -> None:
        settings = {"preview_min": 30, "preview_max": 60, "detail_min": 10, "detail_max": 20,
                    "max_images": 700, "max_mb": 16, "idle_days": 4, "daily_authors": 3,
                    "exploration_ratio": 0.2, **targets}
        self.settings = SimpleNamespace(**settings, model_dump=lambda: dict(settings))
        self.store = Store(directory)
        self.store.initialize()
        self.reader = FakeReader()
        self.collection_lock = asyncio.Lock()
        self.cancelled = set()
        self.decisions: List[Dict[str, Any]] = []
        self.model_inputs: List[Dict[str, Any]] = []
        self.collect = False
        self.bad_preview = False
        self.bad_work_id = False
        self.nullable_title = False
        self.known_author = True
        self.cancel_after_batch = False
        self.interrupt_preview_commit = False
        self.interrupt_observation_commit = False
        self.interrupt_pending_detail = False
        self.vision_override: Dict[str, Any] = {}
        self.vision_results: List[Dict[str, Any]] = []
        self.omit_observations = False
        self.source_evidence = []

    @staticmethod
    def source_weight(source: Dict[str, Any]) -> float:
        return 0.5

    async def db(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        methods = {"get": self.store.get, "put": self.store.put, "list": self.store.list, "count": self.store.count}
        value = await asyncio.to_thread(methods[operation], *args, **kwargs)
        if operation == "put" and args[0] == "preview":
            if self.interrupt_preview_commit:
                self.interrupt_preview_commit = False
                raise asyncio.CancelledError()
            if self.cancel_after_batch:
                self.cancelled.add(args[1].get("job_id", "test-job"))
        if operation == "put" and args[0] == "observation" and self.interrupt_observation_commit:
            self.interrupt_observation_commit = False
            raise asyncio.CancelledError()
        if operation == "put" and args[0] == "job" and self.interrupt_pending_detail and args[1].get("state", {}).get("pending_detail_result"):
            self.interrupt_pending_detail = False
            raise asyncio.CancelledError()
        return value

    async def _source_observed(self, url: str, interest: float, evidence: str, directions: List[str]) -> None:
        self.source_evidence.append(evidence)

    async def save_job_progress(self, job: Dict[str, Any]) -> None:
        await self.db("put", "job", job)

    async def _historical_visual_context(self, actual: Dict[str, Any], preview: Dict[str, Any], job: Dict[str, Any]):
        return [], []

    async def _model(self, purpose: str, data: Dict[str, Any], job: Dict[str, Any],
                     image: Dict[str, Any] = None, images: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        job["model_calls"] += 1
        self.model_inputs.append({"purpose": purpose, "data": data, "image": image, "images": images})
        if purpose == "decision":
            if self.decisions:
                return self.decisions.pop(0)
            if data["pending_images"]:
                return {"action": "preview", "reason": "Continue actual candidate images"}
            if data["candidate_links"]:
                return {"action": "open", "url": data["candidate_links"][0]["url"], "reason": "Next discovered page"}
            return {"action": "finish", "reason": "Discovered candidates are exhausted"}
        if purpose == "preview":
            assert images and len(images) <= 6
            for actual in images:
                with Image.open(BytesIO(actual["data"])) as decoded:
                    decoded.load()
            if self.bad_preview:
                return {"items": []}
            return {"model": "fake-vlm", "items": [{"id": item["id"], "visual_note": "A visible quiet patch of color.",
                "interest": 0.55, "is_photo": "/logo/" not in item["image_url"], "detail": False,
                "reason": "Worth comparing, but collection is undecided."} for item in data["images"]]}
        if purpose == "vision":
            assert image and image["data"]
            seed = data["seed"]
            ids = [seed["work_items"][0]["id"]] if seed and seed["work_items"] else []
            result = {"author": (seed["author"] if seed else "One Artist") if self.known_author else None, "work": "Salt Ponds",
                    "title": None if self.nullable_title else "A photograph", "identity_verified": self.known_author,
                    "identity_evidence": "The opened page names the author." if self.known_author else None,
                    "matches_seed": bool(seed), "matched_work_ids": ["made-up-id"] if self.bad_work_id else ids,
                    "observations": "A muted color field with little local detail at this resolution.",
                    "feeling": "I want to compare its spacing with the other photographs.", "reason": "A concrete visual comparison.",
                    "directions": ["quiet color"], "bot_interest": 0.55, "collect": self.collect, "model": "fake-vlm"}
            result.update(self.vision_override)
            if self.vision_results:
                result.update(self.vision_results.pop(0))
            if self.omit_observations:
                result.pop("observations", None)
            return result
        raise AssertionError("Unexpected purpose " + purpose)


class BrowsingTests(IsolatedAsyncioTestCase):
    async def test_cross_batch_clearer_variant_updates_one_preview_and_existing_collection(self) -> None:
        engine, job = await self.engine_job(preview_min=7, preview_max=7, detail_min=1, detail_max=1)
        page = engine.reader.add_page(0, 8)
        await self.source(engine, page)
        urls = [item["url"] for item in engine.reader.pages[page]["images"]]
        engine.reader.images[urls[0]] = picture(1, 160, 240)
        engine.reader.images[urls[6]] = picture(1, 640, 960)
        existing = await asyncio.to_thread(engine.store.save_image, engine.reader.images[urls[0]]["data"], "image/png",
                                            {"bot_interest": 0.8, "observations": "Existing collection observation"})
        await engine._explore(job, None)
        previews = await engine.db("list", "preview", limit=20)
        self.assertEqual(job["preview_count"], 7)
        self.assertEqual(sum(len(call["images"]) for call in engine.model_inputs if call["purpose"] == "preview"), 7)
        variants = [item for item in previews if item["work_id"] == existing["id"]]
        self.assertEqual(len(variants), 1)
        self.assertEqual((variants[0]["image_url"], variants[0]["width"]), (urls[6], 640))
        upgraded = await engine.db("get", "image", existing["id"])
        self.assertEqual((upgraded["width"], upgraded["version"]), (640, 2))
        self.assertEqual(upgraded["observations"], "Existing collection observation")
        self.assertEqual(await engine.db("count", "image"), 1)

    async def test_evicted_work_stays_evicted_when_a_clearer_variant_is_only_previewed(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        page = engine.reader.add_page(0, 2)
        await self.source(engine, page)
        url = engine.reader.pages[page]["images"][0]["url"]
        existing = await asyncio.to_thread(engine.store.save_image, picture(1, 160, 240)["data"], "image/png", {})
        await asyncio.to_thread(engine.store.cleanup, 0, 1024 * 1024, 4)
        engine.reader.images[url] = picture(1, 640, 960)
        await engine._explore(job, None)
        record = await engine.db("get", "image", existing["id"])
        self.assertEqual(record["local_status"], "evicted")
        self.assertEqual(list(engine.store.images_dir.iterdir()), [])
        self.assertEqual(job["image_ids"], [])

    async def test_resized_duplicate_pending_batch_resumes_without_recounting_or_second_vlm_call(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        page = engine.reader.add_page(0, 3)
        await self.source(engine, page)
        urls = [item["url"] for item in engine.reader.pages[page]["images"]]
        engine.reader.images[urls[0]] = picture(1, 160, 240)
        engine.reader.images[urls[1]] = picture(1, 640, 960)
        engine.interrupt_preview_commit = True
        with self.assertRaises(asyncio.CancelledError):
            await engine._explore(job, None)
        saved = await engine.db("get", "job", job["id"])
        self.assertEqual(len(saved["state"]["pending_preview_results"]), 1)
        resumed = FakeEngine(self.directory / "data", preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        resumed.reader = engine.reader
        await resumed._explore(saved, None)
        self.assertEqual(saved["preview_count"], 2)
        self.assertEqual(await resumed.db("count", "preview"), 2)
        self.assertEqual(sum(len(call["images"]) for call in resumed.model_inputs if call["purpose"] == "preview"), 1)
        self.assertEqual(sum(len(call["images"]) for call in engine.model_inputs if call["purpose"] == "preview"), 1)

    async def test_resized_cross_site_variants_count_once_and_prefer_large_preview(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        page = engine.reader.add_page(0, 3)
        await self.source(engine, page)
        original = Image.new("RGB", (720, 900), "#ded4bd")
        draw = ImageDraw.Draw(original)
        draw.rectangle((120, 180, 420, 700), fill="#583a8b")
        draw.ellipse((430, 40, 650, 250), fill="#2d6658")
        small, large = engine.reader.pages[page]["images"][:2]
        large["url"] = "https://another-gallery.example/fullsize.jpg"
        for info, size in ((small, (288, 360)), (large, (720, 900))):
            content = BytesIO()
            original.resize(size).save(content, "JPEG", quality=91)
            engine.reader.images[info["url"]] = {"data": content.getvalue(), "mime_type": "image/jpeg", "width": size[0], "height": size[1]}
        await engine._explore(job, None)
        previews = await engine.db("list", "preview", limit=10)
        self.assertEqual(job["preview_count"], 2)
        self.assertEqual(len(previews), 2)
        target = next(item for item in previews if item["width"] == 720)
        self.assertEqual(target["image_url"], large["url"])
        self.assertEqual(sum(len(call["images"]) for call in engine.model_inputs if call["purpose"] == "preview"), 2)
        self.assertEqual(await engine.db("count", "image"), 0)

    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    async def engine_job(self, **targets: int):
        engine = FakeEngine(self.directory / "data", **targets)
        job = {"id": "test-job", "kind": "manual", "intent": "Look at many real photos, retain only what is interesting",
               "model_calls": 0, "errors": [], "image_ids": [], "observation_ids": [], "status": "running"}
        await engine.db("put", "job", job)
        return engine, job

    async def source(self, engine: FakeEngine, url: str) -> None:
        await engine.db("put", "source", {"id": url, "url": url, "name": "Photography source"})

    async def test_thirty_real_previews_then_ten_details_without_forced_collection(self) -> None:
        engine, job = await self.engine_job()
        await self.source(engine, engine.reader.add_page(0, 30))
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual((result["preview_count"], result["detail_count"]), (30, 10))
        self.assertTrue(result["target_met"])
        self.assertEqual(job["image_ids"], [])
        batches = [call for call in engine.model_inputs if call["purpose"] == "preview"]
        self.assertEqual(len(batches), 5)
        self.assertTrue(all(len(call["images"]) == 6 for call in batches))
        self.assertEqual(await engine.db("count", "observation"), 10)
        self.assertEqual(len(list(engine.store.images_dir.iterdir())), 0)
        records = await engine.db("list", "preview", limit=100)
        self.assertTrue(all("data" not in item and "data:image" not in json.dumps(item) for item in records))
        detailed = await engine.db("list", "observation", limit=100)
        self.assertTrue(all(item["preview_id"] in job["preview_ids"] for item in detailed))

    async def test_navigation_is_not_capped_at_twelve_steps(self) -> None:
        engine, job = await self.engine_job(preview_min=15, preview_max=15, detail_min=4, detail_max=4)
        for index in range(15):
            engine.reader.add_page(index, 1, next_page=f"https://photos.example/page/{index + 1}" if index < 14 else "")
        await self.source(engine, "https://photos.example/page/0")
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual(result["navigation_actions"], 15)
        self.assertEqual(result["preview_count"], 15)
        self.assertTrue(result["target_met"])

    async def test_same_pixels_at_multiple_urls_count_once(self) -> None:
        engine, job = await self.engine_job(preview_min=6, preview_max=6, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 6, duplicate=True))
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual(result["preview_count"], 1)
        self.assertEqual(result["detail_count"], 1)
        self.assertFalse(result["target_met"])
        self.assertEqual(await engine.db("count", "preview"), 1)

    async def test_new_links_are_progress_before_reaching_actual_gallery(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        for index in range(5):
            engine.reader.add_page(index, 2 if index == 4 else 0,
                next_page=f"https://photos.example/page/{index + 1}" if index < 4 else "")
        await self.source(engine, "https://photos.example/page/0")
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual(result["navigation_actions"], 5)
        self.assertEqual(result["preview_count"], 2)
        self.assertTrue(result["target_met"])

    async def test_new_search_result_is_visible_beyond_twenty_four_old_sources(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        for index in range(30):
            await self.source(engine, f"https://sources.example/site/{index}")
        found_url = engine.reader.add_page(0, 2)
        engine.decisions = [{"action": "search", "query": "specific photobook photographs",
                             "reason": "Find the requested work"}]
        await engine._explore(job, None)
        navigation = [call["data"] for call in engine.model_inputs if call["purpose"] == "decision"]
        self.assertEqual(len(navigation[0]["candidate_links"]), 24)
        self.assertNotIn(found_url, [item["url"] for item in navigation[0]["candidate_links"]])
        self.assertEqual(navigation[1]["candidate_links"][0]["url"], found_url)
        self.assertTrue(job["browse_results"]["daily"]["target_met"])
        self.assertIn(("open", found_url), engine.reader.calls)

    async def test_logos_do_not_satisfy_preview_target_and_stop_on_no_progress(self) -> None:
        engine, job = await self.engine_job()
        await self.source(engine, engine.reader.add_page(0, 30, logos=True))
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual(result["preview_count"], 0)
        self.assertEqual(result["detail_count"], 0)
        self.assertEqual(result["inspected_image_count"], 30)
        self.assertEqual(result["stop_reason"], "no_progress")
        self.assertEqual(await engine.db("count", "preview"), 30)

    async def test_icons_covers_and_failed_images_do_not_hide_later_photographs(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        url = engine.reader.add_page(0, 30)
        images = engine.reader.pages[url]["images"]
        for index, item in enumerate(images[:4]):
            item["url"] = f"https://photos.example/icon{index}.{'SVG' if index % 2 else 'ico'}?size=large"
        for item in images[4:22]:
            original = item["url"]
            item["url"] = original.replace("/work/", "/logo/")
            engine.reader.images[item["url"]] = engine.reader.images.pop(original)
        for item in images[22:28]:
            engine.reader.images[item["url"]] = reader_module.ReaderError("Unsupported image content")
        await self.source(engine, url)
        await engine._explore(job, None)
        result = job["browse_results"]["daily"]
        self.assertEqual(result["preview_count"], 2)
        self.assertEqual(result["inspected_image_count"], 20)
        self.assertTrue(result["target_met"])
        self.assertEqual(len(job["errors"]), 6)
        fetched = [value for action, value in engine.reader.calls if action == "image"]
        self.assertTrue(all("/icon" not in value for value in fetched))
        self.assertFalse(job["state"]["preview_queue"])

    async def test_premature_finish_is_rejected_while_real_candidates_remain(self) -> None:
        engine, job = await self.engine_job(preview_min=4, preview_max=4, detail_min=1, detail_max=1)
        first = engine.reader.add_page(0, 2, next_page="https://photos.example/page/1")
        engine.reader.add_page(1, 2)
        await self.source(engine, first)
        engine.decisions = [{"action": "finish", "reason": "I think enough"}] * 3
        await engine._explore(job, None)
        self.assertEqual(job["preview_count"], 4)
        self.assertTrue(job["browse_results"]["daily"]["target_met"])
        self.assertTrue(any("not_accepted" in entry for entry in job["state"]["history"]))

    async def test_malformed_batch_model_response_is_visible_and_not_counted(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.bad_preview = True
        with self.assertRaises(ModelFailure):
            await engine._explore(job, None)
        self.assertEqual(await engine.db("count", "preview"), 0)
        self.assertEqual(job["preview_ids"], [])

    async def test_failed_downloads_do_not_create_preview_records(self) -> None:
        engine, job = await self.engine_job(preview_min=4, preview_max=6, detail_min=2, detail_max=2)
        url = engine.reader.add_page(0, 6)
        for candidate in engine.reader.pages[url]["images"][:2]:
            engine.reader.images[candidate["url"]] = reader_module.ReaderError("HTTP 404 image unavailable")
        await self.source(engine, url)
        await engine._explore(job, None)
        self.assertEqual(job["preview_count"], 4)
        self.assertEqual(len(job["errors"]), 2)
        self.assertEqual(len(next(call["images"] for call in engine.model_inputs if call["purpose"] == "preview")), 4)

    async def test_cancelled_partial_batch_commit_resumes_without_second_preview_call(self) -> None:
        engine, job = await self.engine_job(preview_min=6, preview_max=6, detail_min=2, detail_max=2)
        await self.source(engine, engine.reader.add_page(0, 6))
        engine.interrupt_preview_commit = True
        with self.assertRaises(asyncio.CancelledError):
            await engine._explore(job, None)
        saved_job = await engine.db("get", "job", job["id"])
        self.assertEqual(len(saved_job["state"]["pending_preview_results"]), 6)
        self.assertEqual(await engine.db("count", "preview"), 1)
        resumed = FakeEngine(self.directory / "data", preview_min=6, preview_max=6, detail_min=2, detail_max=2)
        resumed.reader = engine.reader
        await resumed._explore(saved_job, None)
        self.assertEqual(saved_job["preview_count"], 6)
        self.assertEqual(await resumed.db("count", "preview"), 6)
        self.assertEqual(sum(call["purpose"] == "preview" for call in resumed.model_inputs), 0)
        self.assertEqual(len(saved_job["observation_ids"]), 2)

    async def test_small_photo_can_be_previewed_detailed_and_collected(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        await engine._explore(job, None)
        self.assertEqual(job["preview_count"], 2)
        self.assertEqual(len(job["observation_ids"]), 1)
        self.assertEqual(len(job["image_ids"]), 1, job["errors"])

    async def test_changed_image_at_detail_time_is_not_a_second_preview(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.reader.changed_on_detail = True
        await engine._explore(job, None)
        self.assertEqual(job["preview_count"], 2)
        self.assertEqual(len(job["observation_ids"]), 0)
        self.assertFalse(job["browse_results"]["daily"]["target_met"])
        self.assertTrue(all("已变化" in entry["error"] for entry in job["errors"]))

    async def test_unknown_title_does_not_block_real_observation_or_collection(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        engine.nullable_title = True
        await engine._explore(job, None)
        observation = (await engine.db("list", "observation"))[0]
        self.assertEqual(observation["title"], "")
        self.assertTrue(observation["observations"])
        self.assertEqual(len(job["image_ids"]), 1)
        self.assertTrue(job["browse_results"]["daily"]["target_met"])

    async def test_nullable_metadata_keeps_required_text_and_wrong_types_strict(self) -> None:
        validate = importlib.import_module(PACKAGE + ".browsing")._text
        for field in ("author", "work", "title", "identity_evidence"):
            self.assertEqual(validate(None, field, 100), "")
            for invalid in (0, {}, []):
                with self.assertRaises(ModelFailure):
                    validate(invalid, field, 100)
        for field in ("visual_note", "observations"):
            for invalid in (None, "", "   "):
                with self.assertRaises(ModelFailure):
                    validate(invalid, field, 100, required=True)

    async def test_unattributed_photo_can_be_collected_and_meet_viewing_targets(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        engine.known_author = False
        await engine._explore(job, None)
        observation = (await engine.db("list", "observation"))[0]
        self.assertFalse(observation["identity_verified"])
        self.assertEqual(observation["author"], "")
        self.assertEqual(len(job["image_ids"]), 1)
        self.assertTrue(job["browse_results"]["daily"]["target_met"])
        self.assertEqual(job["browse_results"]["daily"]["authors"], [])
        self.assertFalse(any(action == "search" for action, _ in engine.reader.calls))

    async def test_unattributed_collection_resumes_without_another_visual_model_call(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        engine.known_author = False
        engine.interrupt_observation_commit = True
        with self.assertRaises(asyncio.CancelledError):
            await engine._explore(job, None)
        saved_job = await engine.db("get", "job", job["id"])
        resumed = FakeEngine(self.directory / "data", preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        resumed.reader = engine.reader
        await resumed._explore(saved_job, None)
        self.assertEqual(len(saved_job["image_ids"]), 1)
        self.assertTrue(saved_job["browse_results"]["daily"]["target_met"])
        self.assertEqual(sum(call["purpose"] == "vision" for call in resumed.model_inputs), 0)

    async def test_cancelled_saved_raw_detail_resumes_without_another_visual_call(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        engine.vision_override = {"prompt": "not-for-storage", "character_context": {"private": "not-for-storage"},
                                  "chat": "not-for-storage"}
        engine.interrupt_pending_detail = True
        with self.assertRaises(asyncio.CancelledError):
            await engine._explore(job, None)
        saved = await engine.db("get", "job", job["id"])
        pending = saved["state"]["pending_detail_result"]
        self.assertEqual(pending["result"]["model"], "fake-vlm")
        self.assertIsInstance(pending["result"]["observations"], str)
        self.assertNotIn("not-for-storage", json.dumps(pending))
        self.assertNotIn("data", pending)
        self.assertEqual(await engine.db("count", "observation"), 0)
        resumed = FakeEngine(self.directory / "data", preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        resumed.reader = engine.reader
        await resumed._explore(saved, None)
        self.assertEqual(sum(call["purpose"] == "vision" for call in resumed.model_inputs), 0)
        self.assertEqual(len(saved["observation_ids"]), 1)
        self.assertEqual(len(saved["image_ids"]), 1)
        self.assertIsNone((await resumed.db("get", "job", job["id"]))["state"]["pending_detail_result"])

    async def test_observation_contract_errors_are_precise_and_retry_requests_only_unfinished_image(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        invalid_cases = [(None, True, "missing"), (None, False, "null"), ("", False, "empty_string"),
                         ("   ", False, "blank_string"), (["visible detail"], False, "array/list"),
                         ({"detail": "visible detail"}, False, "object/dict"), (42, False, "number")]
        for invalid, missing, expected in invalid_cases:
            engine.omit_observations = missing
            engine.vision_override = {"observations": invalid, "prompt": "do-not-save-me"}
            before = len([call for call in engine.model_inputs if call["purpose"] == "vision"])
            with self.subTest(expected=expected), self.assertRaisesRegex(ModelFailure, expected):
                await engine._explore(job, None)
            saved = await engine.db("get", "job", job["id"])
            error = saved["last_model_error"]
            self.assertEqual(error["field_types"]["observations"]["type"], expected)
            self.assertEqual(error["model"], "fake-vlm")
            self.assertTrue(error["preview_id"])
            self.assertTrue(error["image_url"].startswith("https://photos.example/"))
            self.assertIsNone(saved["state"]["pending_detail_result"])
            self.assertNotIn("do-not-save-me", json.dumps(error))
            self.assertLess(len(json.dumps(error)), 5000)
            self.assertEqual(await engine.db("count", "observation"), 0)
            self.assertEqual(len(job["state"]["detail_attempted"]), 0)
            self.assertEqual(sum(call["purpose"] == "vision" for call in engine.model_inputs), before + 1)
        previews_before = sum(call["purpose"] == "preview" for call in engine.model_inputs)
        engine.omit_observations = False
        engine.vision_override = {}
        engine.collect = True
        await engine._explore(job, None)
        self.assertTrue(job["browse_results"]["daily"]["target_met"])
        self.assertEqual(sum(call["purpose"] == "preview" for call in engine.model_inputs), previews_before)
        self.assertEqual(await engine.db("count", "observation"), 1)
        self.assertEqual(len(job["image_ids"]), 1)

    async def test_failed_second_detail_resumes_without_repeating_first_collection_or_previews(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=2, detail_max=2)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.collect = True
        engine.vision_results = [{}, {"observations": ["valid-looking content with the wrong JSON type"]}]
        with self.assertRaisesRegex(ModelFailure, "array/list"):
            await engine._explore(job, None)
        saved = await engine.db("get", "job", job["id"])
        first_observation = saved["observation_ids"][0]
        first_image = saved["image_ids"][0]
        self.assertEqual(len(saved["state"]["detail_attempted"]), 1)
        resumed = FakeEngine(self.directory / "data", preview_min=2, preview_max=2, detail_min=2, detail_max=2)
        resumed.reader = engine.reader
        resumed.collect = True
        await resumed._explore(saved, None)
        self.assertEqual([call["purpose"] for call in resumed.model_inputs], ["vision"])
        self.assertEqual(len(saved["observation_ids"]), 2)
        self.assertEqual(len(saved["image_ids"]), 2)
        self.assertIn(first_observation, saved["observation_ids"])
        self.assertIn(first_image, saved["image_ids"])
        self.assertTrue(saved["browse_results"]["daily"]["target_met"])

    async def test_pending_detail_download_failure_does_not_block_next_image(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.interrupt_pending_detail = True
        with self.assertRaises(asyncio.CancelledError):
            await engine._explore(job, None)
        saved = await engine.db("get", "job", job["id"])
        pending = saved["state"]["pending_detail_result"]
        resumed = FakeEngine(self.directory / "data", preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        resumed.reader = engine.reader
        resumed.reader.images[pending["image_url"]] = reader_module.ReaderError("HTTP 404 image no longer available")
        await resumed._explore(saved, None)
        self.assertTrue(saved["browse_results"]["daily"]["target_met"])
        self.assertIsNone(saved["state"]["pending_detail_result"])
        self.assertEqual(saved["errors"][0]["action"], "detail_download")
        self.assertEqual([call["purpose"] for call in resumed.model_inputs], ["vision"])
        observation = (await resumed.db("list", "observation"))[0]
        self.assertNotEqual(observation["preview_id"], pending["preview_id"])

    async def test_seed_work_id_maps_to_original_label_despite_display_translation(self) -> None:
        engine, job = await self.engine_job()
        seed = reader_module.parse_seeds("彭可：盐水池塘")[0]
        await engine.db("put", "seed", seed)
        await self.source(engine, engine.reader.add_page(0, 2))
        await engine._explore(job, seed)
        observation = (await engine.db("list", "observation"))[0]
        self.assertEqual(observation["work"], "Salt Ponds")
        self.assertEqual(observation["matched_works"], ["盐水池塘"])
        self.assertEqual(observation["matched_work_ids"], [seed["work_items"][0]["id"]])
        self.assertEqual(job["browse_results"][seed["id"]]["preview_count"], 2)

    async def test_unknown_work_ids_raise_instead_of_confirming_wrong_book(self) -> None:
        engine, job = await self.engine_job()
        seed = reader_module.parse_seeds("彭可：盐水池塘")[0]
        await self.source(engine, engine.reader.add_page(0, 2))
        engine.bad_work_id = True
        with self.assertRaises(ModelFailure):
            await engine._explore(job, seed)
        self.assertEqual(await engine.db("count", "observation"), 0)

    async def test_daily_context_contains_compact_seed_feedback_and_resolved_facts(self) -> None:
        engine, job = await self.engine_job(preview_min=2, preview_max=2, detail_min=1, detail_max=1)
        seed = reader_module.parse_seeds("彭可：盐水池塘")[0]
        await engine.db("put", "seed", seed)
        await engine.db("put", "feedback", {"id": "f", "status": "evaluated", "is_evaluation": True,
                                             "insight": "比较空间距离", "human_interest": 0.8})
        await engine.db("put", "fact_check", {"id": "fact", "status": "resolved", "question": "作品译名", "conclusion": "保留已核对的译名关系"})
        await self.source(engine, engine.reader.add_page(0, 2))
        await engine._explore(job, None)
        request = engine.model_inputs[0]["data"]
        self.assertEqual(request["seed_preferences"][0]["work_items"][0]["title"], "盐水池塘")
        self.assertEqual(request["recent_feedback"][0]["insight"], "比较空间距离")
        self.assertEqual(request["source_notes"], ["保留已核对的译名关系"])
        self.assertNotIn("confirmed_facts", request)
        self.assertNotIn("raw_line", request["seed_preferences"][0])
