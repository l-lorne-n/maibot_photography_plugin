"""Offline group-discussion capture, scoped interpretation, and replay behavior."""

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase

from PIL import Image

import asyncio
import importlib
import sys


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_feedback_runtime_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
PhotographyEngine = importlib.import_module(PACKAGE + ".workflow").PhotographyEngine
Settings = importlib.import_module(PACKAGE + ".config").Settings
create_plugin = importlib.import_module(PACKAGE + ".plugin").create_plugin


class FeedbackRuntimeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime.now(timezone.utc)
        self.messages: List[Dict[str, Any]] = []
        self.responses: List[Dict[str, Any]] = []
        self.calls: List[Dict[str, Any]] = []

        async def recent(chat_id: str, limit: int) -> List[Dict[str, Any]]:
            self.assertEqual(chat_id, "group")
            self.assertIn(limit, {24, 500})
            return self.messages

        self.ctx = SimpleNamespace(paths=SimpleNamespace(data_dir=Path(self.temp.name)),
                                   message=SimpleNamespace(get_recent=recent))
        self.engine = PhotographyEngine(self.ctx, Settings(daily_enabled=False, target_stream_id="group"))
        await asyncio.to_thread(self.engine.store.initialize)
        self.engine.character_context = {"bot": {"qq_account": "bot"}, "scope": {"kind": "group", "stream_id": "group"}}

        async def refresh() -> None:
            return None

        async def model(purpose: str, data: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
            self.assertEqual(purpose, "feedback")
            self.calls.append(data)
            return self.responses.pop(0) if self.responses else {
                "scope": "general", "is_evaluation": False, "human_interest": None,
                "reason": "本人发言", "insight": "朋友提到对黑白摄影构图的兴趣", "topics": ["构图"]}

        self.engine.refresh_character = refresh
        self.engine._model = model
        self.plugin = create_plugin()
        self.plugin._set_context(self.ctx)
        self.plugin.engine = self.engine

    def message(self, ident: str, text: str, seconds_ago: int = 60, speaker: str = "friend", stream: str = "group") -> Dict[str, Any]:
        return {"message_id": ident, "session_id": stream, "timestamp": str((self.now - timedelta(seconds=seconds_ago)).timestamp()),
                "message_info": {"user_info": {"user_id": speaker}}, "raw_message": [{"type": "text", "data": text}]}

    async def batch(self) -> List[Dict[str, Any]]:
        images = []
        for index, color in enumerate(("red", "blue")):
            buffer = BytesIO()
            Image.new("RGB", (640, 480), color).save(buffer, format="PNG")
            image = await asyncio.to_thread(self.engine.store.save_image, buffer.getvalue(), "image/png", {
                "author": "Alec Soth", "work": "Book", "observations": "A " + color + " dress near a window", "bot_interest": 0.8,
                "source_url": "https://photo.example/work", "directions": ["color"]})
            images.append(image)
            await self.engine.db("put", "share", {"id": "receipt-" + str(index), "message_id": "sent-" + str(index),
                "stream_id": "group", "image_id": image["id"], "job_id": "photo-batch",
                "created_at": (self.now - timedelta(minutes=10, seconds=-index)).isoformat()})
        return images

    async def test_natural_batch_discussion_merges_jobs_and_keeps_preferences_separate(self) -> None:
        images = await self.batch()
        one = await self.engine.capture_feedback(self.message("one", "感觉还是soth的更好看一点吧"))
        two = await self.engine.capture_feedback(self.message("two", "就是这些图片有点糊", 55))
        self.assertEqual(one["job_id"], two["job_id"])
        self.responses = [{"scope": "batch", "image_ids": [image["id"] for image in images], "is_evaluation": True,
                           "human_interest": 0.9, "insight": "朋友更喜欢这组Soth作品"},
                          {"scope": "batch", "is_evaluation": False, "human_interest": None, "insight": "讨论分享图片的清晰度"}]
        await self.engine._feedback(await self.engine.db("get", "job", one["job_id"]))
        for image in images:
            updated = await self.engine.db("get", "image", image["id"])
            self.assertIsNone(updated["human_interest"])
            self.assertEqual(updated["bot_interest"], 0.8)
        feedback = await self.engine.db("list", "feedback")
        self.assertEqual(len(feedback), 2)
        self.assertTrue(all(record["scope"] == "batch" for record in feedback))
        self.assertEqual(sum(job["kind"] == "review" for job in await self.engine.db("list", "job")), 1)

    async def test_actual_account_excludes_bot_with_empty_spare_configuration(self) -> None:
        self.engine.character_context = {"bot": {"qq_account": "", "platforms": []},
            "scope": {"kind": "group", "stream_id": "group", "platform": "qq", "account_id": "actual-bot"}}
        own = {**self.message("own", "想分享几张摄影作品来着", speaker="actual-bot"),
               "platform": "qq", "account_id": "actual-bot"}
        friend = {**self.message("friend", "我喜欢这些照片的颜色", seconds_ago=50),
                  "platform": "qq", "account_id": "actual-bot"}
        friend["message_info"]["user_info"]["user_nickname"] = "same bot nickname"
        result = await self.engine.backfill_feedback([own, friend])
        self.assertEqual(result["captured"], 1)
        record = (await self.engine.db("list", "feedback"))[0]
        self.assertEqual(record["speaker_id"], "friend")
        self.assertEqual(record["account_id"], "actual-bot")
        self.assertEqual(record["context_messages"][0]["is_self"], True)
        self.assertEqual(record["context_messages"][0]["speaker_id"], "actual-bot")

    async def test_worker_rechecks_legacy_pending_self_without_model_or_review(self) -> None:
        self.engine.character_context = {"bot": {},
            "scope": {"kind": "group", "stream_id": "group", "platform": "qq", "account_id": "actual-bot"}}
        record = {"id": "legacy-self", "speaker_id": "actual-bot", "stream_id": "group", "message_id": "old-own",
                  "text": "我喜欢摄影", "status": "pending", "created_at": self.now.isoformat()}
        await self.engine.db("put", "feedback", record)
        job = await self.engine.enqueue("feedback", "整理讨论", seed_ids=[record["id"]])
        job["feedback_has_insight"] = True
        await self.engine._feedback(job)
        ignored = await self.engine.db("get", "feedback", record["id"])
        self.assertEqual(ignored["status"], "ignored")
        self.assertEqual(ignored["ignore_reason"], "self_message")
        self.assertEqual(ignored["self_message_trace"]["status"], "pending")
        self.assertFalse(self.calls)
        self.assertFalse(any(item["kind"] == "review" for item in await self.engine.db("list", "job")))

    async def test_worker_relabels_legacy_context_from_actual_account_before_model(self) -> None:
        self.engine.character_context = {"bot": {},
            "scope": {"kind": "group", "stream_id": "group", "platform": "qq", "account_id": "actual-bot"}}
        record = {"id": "legacy-friend", "speaker_id": "friend", "stream_id": "group", "message_id": "friend-comment",
                  "text": "这些照片的光线我喜欢", "status": "pending", "created_at": self.now.isoformat(),
                  "context_messages": [{"message_id": "own", "speaker_id": "actual-bot", "text": "想分享摄影作品", "is_self": False},
                                       {"message_id": "human", "speaker_id": "friend", "text": "我喜欢窗边那张", "is_self": True}]}
        await self.engine.db("put", "feedback", record)
        job = await self.engine.enqueue("feedback", "整理讨论", seed_ids=[record["id"]])
        await self.engine._feedback(job)
        expected = [True, False]
        self.assertEqual([item["is_self"] for item in self.calls[0]["message"]["context_messages"]], expected)
        self.assertEqual([item["is_self"] for item in self.calls[0]["discussion_context"]], expected)
        saved = await self.engine.db("get", "feedback", record["id"])
        self.assertEqual([item["is_self"] for item in saved["context_messages"]], expected)

    async def test_realtime_followup_preserves_original_topic_and_excludes_other_times_and_groups(self) -> None:
        current = self.message("followup-now", "比刚才好", 30)
        self.messages = [self.message("topic", "这位新摄影师的构图值得多看看", 60, "bot"),
                         current, self.message("future", "未来照片不要读", 10),
                         self.message("foreign", "别群的摄影不要读", 50, stream="other")]
        captured = await self.engine.capture_feedback(current)
        self.assertTrue(captured["captured"])
        saved = await self.engine.db("get", "feedback", captured["id"])
        self.assertEqual(saved["association"], "conversation_followup")
        self.assertEqual([item["message_id"] for item in saved["context_messages"]], ["topic"])
        self.assertTrue(saved["context_messages"][0]["is_self"])
        self.messages = []
        await self.engine._feedback(await self.engine.db("get", "job", captured["job_id"]))
        self.assertEqual(self.calls[0]["discussion_context"][0]["message_id"], "topic")
        self.assertEqual(self.calls[0]["candidate_images"], [])

    async def test_realtime_followups_do_not_extend_topic_window_indefinitely(self) -> None:
        topic = self.message("topic", "这张摄影的光线好", 300, "bot")
        first = self.message("first-followup", "我也是", 190)
        self.messages = [topic]
        self.assertTrue((await self.engine.capture_feedback(first))["captured"])
        self.messages = [topic, first]
        self.assertFalse((await self.engine.capture_feedback(self.message("late-followup", "我也是", 179)))["captured"])
        self.assertFalse((await self.engine.capture_feedback(self.message("still-later", "我也是", 170)))["captured"])

    async def test_context_io_is_outside_capture_lock_and_errors_are_visible(self) -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def recent(chat_id: str, limit: int) -> List[Dict[str, Any]]:
            self.assertFalse(self.engine.feedback_capture_lock.locked())
            self.assertEqual(limit, 24)
            entered.set()
            await release.wait()
            raise RuntimeError("Host context read failed")

        self.ctx.message.get_recent = recent
        task = asyncio.create_task(self.engine.capture_feedback(self.message("request", "摄影构图挺有意思")))
        await entered.wait()
        async with self.engine.feedback_capture_lock:
            self.assertFalse(task.done())
        release.set()
        with self.assertRaisesRegex(RuntimeError, "Host context read failed"):
            await task
        self.assertEqual(await self.engine.db("count", "feedback"), 0)

    async def test_external_photo_discussion_keeps_host_description_without_claiming_viewing(self) -> None:
        external = self.message("external-image", "", 65)
        external["raw_message"] = [{"type": "image", "hash": "external-hash", "data": "Host描述：海边的人与大片留白",
                                    "binary_data_base64": "MUST NOT PERSIST"}]
        self.messages = [external]
        captured = await self.engine.capture_feedback(self.message("external-comment", "这张照片的留白很舒服", 60))
        record = await self.engine.db("get", "feedback", captured["id"])
        self.assertEqual(record["candidate_image_ids"], [])
        note = record["context_messages"][0]["host_images"][0]
        self.assertEqual(note["hash"], "external-hash")
        self.assertFalse(note["observed_by_plugin"])
        self.assertNotIn("binary_data_base64", note)
        await self.engine._feedback(await self.engine.db("get", "job", captured["job_id"]))
        self.assertEqual((await self.engine.db("get", "feedback", captured["id"]))["scope"], "general")
        self.assertEqual(await self.engine.db("count", "image"), 0)

    async def test_visual_description_can_resolve_one_candidate_only(self) -> None:
        images = await self.batch()
        result = await self.engine.capture_feedback(self.message("visual", "红裙子那张我更喜欢"))
        self.responses = [{"scope": "image", "image_id": images[0]["id"], "is_evaluation": True,
                           "human_interest": 1.0, "insight": "朋友喜欢红色裙子与空间的关系", "bot_interest": 0.0}]
        await self.engine._feedback(await self.engine.db("get", "job", result["job_id"]))
        first, second = [await self.engine.db("get", "image", image["id"]) for image in images]
        self.assertAlmostEqual(first["human_interest"], 0.6)
        self.assertIsNone(second["human_interest"])
        self.assertEqual(first["bot_interest"], second["bot_interest"])

    async def test_general_photography_without_any_collection_still_updates_discussion(self) -> None:
        result = await self.engine.capture_feedback(self.message("external", "想看看新摄影师的黑白街头作品"))
        self.assertTrue(result["captured"])
        self.responses = [{"scope": "general", "is_evaluation": False, "human_interest": None,
                           "insight": "朋友提出探索黑白街头摄影", "topics": ["街头摄影"], "photographer": "新作者", "work": ""}]
        await self.engine._feedback(await self.engine.db("get", "job", result["job_id"]))
        saved = await self.engine.db("get", "feedback", result["id"])
        self.assertEqual(saved["scope"], "general")
        self.assertTrue(saved["is_photography_related"])
        self.assertFalse(saved["is_evaluation"])
        self.assertEqual(await self.engine.db("count", "image"), 0)
        self.assertEqual(self.calls[0]["candidate_images"], [])

    async def test_backfill_covers_last_day_without_share_filters_and_is_idempotent(self) -> None:
        self.messages = [self.message("old", "昨天黑白摄影挺有意思", 86401), self.message("photo", "光线和构图值得多看看", 600),
                         self.message("bot", "摄影我也喜欢", 590, "bot"), self.message("follow", "这个我也一样", 580),
                         self.message("wrong", "喜欢这些摄影", 570, stream="private")]
        first = await self.plugin.webui_backfill_feedback()
        again = await self.plugin.webui_backfill_feedback()
        self.assertEqual(first["captured"], 2)
        self.assertEqual(again["captured"], 0)
        self.assertEqual(first["job_id"], again["job_id"])
        await self.engine._feedback(await self.engine.db("get", "job", first["job_id"]))
        final = await self.plugin.webui_backfill_feedback()
        self.assertEqual(final["pending"], 0)
        self.assertEqual(final["job_id"], "")
        self.assertEqual(await self.engine.db("count", "feedback"), 2)

    async def test_unrelated_neighbor_is_recorded_for_dedup_but_hidden_from_ui(self) -> None:
        result = await self.engine.capture_feedback(self.message("photo", "摄影构图有趣"))
        other = await self.engine.capture_feedback(self.message("unrelated", "我去吃晚饭了", 55))
        self.assertTrue(other["captured"])
        self.responses = [{"scope": "general", "is_evaluation": False, "insight": "讨论构图"},
                          {"scope": "unrelated", "is_evaluation": False, "insight": ""}]
        await self.engine._feedback(await self.engine.db("get", "job", result["job_id"]))
        self.assertEqual(await self.engine.db("count", "feedback"), 2)
        displayed = await self.plugin.webui_records("feedback")
        self.assertEqual(displayed["total"], 1)
        self.assertEqual(displayed["items"][0]["message_id"], "photo")
        self.assertFalse((await self.engine.capture_feedback(self.message("unrelated", "我去吃晚饭了", 55)))["captured"])

    async def test_invalid_model_scope_keeps_failure_visible_without_image_updates(self) -> None:
        images = await self.batch()
        result = await self.engine.capture_feedback(self.message("invalid", "这张颜色不错"))
        self.responses = [{"scope": "image", "image_id": "not-a-candidate", "is_evaluation": True, "human_interest": 1.0}]
        with self.assertRaisesRegex(ValueError, "未提供的图片"):
            await self.engine._feedback(await self.engine.db("get", "job", result["job_id"]))
        self.assertEqual((await self.engine.db("get", "feedback", result["id"]))["status"], "failed")
        for image in images:
            self.assertIsNone((await self.engine.db("get", "image", image["id"]))["human_interest"])

    async def test_legacy_share_receipt_and_hash_follow_canonical_work_id(self) -> None:
        images = await self.batch()
        old_id = "abcdef01" + "a" * 56
        await self.engine.db("put", "share", {"id": "old-receipt", "message_id": "old-send", "stream_id": "group",
            "image_id": old_id, "canonical_image_id": images[0]["id"], "job_id": "photo-batch",
            "created_at": (self.now - timedelta(minutes=3)).isoformat()})
        message = self.message("quote-old", "我喜欢这张")
        message["reply_to"] = "old-send"
        captured = await self.engine.capture_feedback(message)
        self.assertEqual((await self.engine.db("get", "feedback", captured["id"]))["image_id"], images[0]["id"])
        message = self.message("old-hash", "这张颜色好看")
        message["raw_message"].append({"type": "image", "data": "", "hash": old_id})
        captured = await self.engine.capture_feedback(message)
        self.assertEqual((await self.engine.db("get", "feedback", captured["id"]))["image_id"], images[0]["id"])
