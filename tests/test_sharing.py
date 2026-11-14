"""Sharing uses real local persistence and Host context injection, with mocked sends."""

from contextlib import closing
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List
from unittest import IsolatedAsyncioTestCase

from PIL import Image

import __future__
import ast
import asyncio
import importlib
import json
import sqlite3
import sys


ROOT = Path(__file__).parents[1]
PACKAGE_NAME = "photography_sharing_tests"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE_NAME] = package
plugin_module = importlib.import_module(PACKAGE_NAME + ".plugin")
Settings = importlib.import_module(PACKAGE_NAME + ".config").Settings
PhotographyEngine = importlib.import_module(PACKAGE_NAME + ".workflow").PhotographyEngine
sharing = importlib.import_module(PACKAGE_NAME + ".sharing")


def host_context_builder() -> Any:
    """Load the Host's two pure context helpers without starting any Host services."""
    source = ROOT.parents[2] / "src" / "plugin_runtime" / "component_query.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ComponentQueryService")
    names = {"_build_tool_context_payload", "_build_tool_invocation_payload"}
    owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    owner.bases = []
    owner.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[owner], type_ignores=[]))
    namespace: Dict[str, Any] = {}
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["ComponentQueryService"]


class SharingTests(IsolatedAsyncioTestCase):
    async def test_legacy_alias_receipt_and_observation_resolve_without_resending(self) -> None:
        canonical = self.images[0]["id"]
        alias = "a" * 64
        def install_alias() -> None:
            with closing(sqlite3.connect(self.engine.store.database_path)) as connection, connection:
                connection.execute("INSERT INTO image_alias(alias_id, work_id) VALUES (?,?)", (alias, canonical))
        await asyncio.to_thread(install_alias)
        job = await self.engine.enqueue("share", "Share the observed photograph", initial_data={
            "image_ids": [alias, canonical], "share_target_stream_id": "real-group", "share_status": "partial"})
        job.update(share_planned_image_ids=[alias, canonical], observation_ids=["this-observation"])
        await self.engine.db("put", "job", job)
        await self.engine.db("put", "observation", {"id": "this-observation", "image_id": alias,
            "observations": "Current observation linked through the old image ID"})
        await self.engine.db("put", "share", {"id": "old-receipt", "image_id": alias, "job_id": job["id"],
            "stream_id": "real-group", "message_id": "actual-image-receipt", "sent": True})
        await self.engine.refresh_character()
        reference = json.loads(await self.plugin._commentary_reference(job))
        self.assertEqual(len(reference["photographs"]), 1)
        self.assertEqual(reference["photographs"][0]["image_id"], canonical)
        self.assertEqual(reference["photographs"][0]["observations"], "Current observation linked through the old image ID")
        await self.plugin._finish_share_job(job, "Cancelled after picture delivery", "cancelled")
        self.assertEqual(job["share_status"], "sent")
        await self.engine.db("put", "job", {"id": job["id"], "status": "running"})
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(self.sent, [])
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(self.text_sent), 1)
        self.assertEqual((await self.engine.db("get", "share", "old-receipt"))["image_id"], alias)

    async def asyncSetUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.sent: List[Dict[str, Any]] = []
        self.send_results: List[Any] = []
        self.generated: List[Dict[str, Any]] = []
        self.generate_results: List[Any] = []
        self.text_sent: List[Dict[str, Any]] = []
        self.text_results: List[Any] = []
        self.character_refreshes = 0

        async def send_image(data: str, stream_id: str, return_details: bool) -> Dict[str, Any]:
            self.sent.append({"data": data, "stream_id": stream_id, "return_details": return_details})
            result = self.send_results.pop(0) if self.send_results else {"sent": True, "message_id": "sent-" + str(len(self.sent))}
            if isinstance(result, Exception):
                raise result
            return result

        async def get_streams(**kwargs: Any) -> List[Dict[str, Any]]:
            return [{"session_id": "real-group", "group_name": "摄影群", "platform": "qq", "is_group_session": True}]

        async def generate(stream_id: str, reference: str, **kwargs: Any) -> Dict[str, Any]:
            self.generated.append({"stream_id": stream_id, "reference": reference, **kwargs})
            result = self.generate_results.pop(0) if self.generate_results else {"success": True, "text": "我挺喜欢这两张里安静的留白。"}
            if isinstance(result, BaseException):
                raise result
            return result

        async def send_text(text: str, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
            self.text_sent.append({"text": text, "stream_id": stream_id, **kwargs})
            result = self.text_results.pop(0) if self.text_results else {"sent": True, "message_id": "text-" + str(len(self.text_sent))}
            if isinstance(result, BaseException):
                raise result
            return result

        async def refresh_character() -> None:
            self.character_refreshes += 1
            self.engine.character_context = {"personality": {"personality": "PERSONALITY_NOT_IN_REFERENCE"},
                "scope": {"stream_id": "real-group", "kind": "group", "name": "摄影群"},
                "recent_conversation": [{"stream_id": "real-group", "text": "RECENT_SCOPED_CHAT", "message_id": "group-message"},
                                        {"stream_id": "other-group", "text": "OTHER_GROUP_CHAT"}],
                "friends": [{"stream_id": "real-group", "name": "SCOPED_FRIEND"},
                            {"stream_id": "other-group", "name": "OTHER_GROUP_FRIEND"}]}

        self.ctx = SimpleNamespace(paths=SimpleNamespace(data_dir=Path(self.temporary.name)),
                                   send=SimpleNamespace(image=send_image, text=send_text),
                                   maisaka=SimpleNamespace(reply=SimpleNamespace(generate=generate)),
                                   chat=SimpleNamespace(get_all_streams=get_streams))
        self.engine = PhotographyEngine(self.ctx, Settings(daily_enabled=False, share_enabled=True,
                                                         target_stream_id="real-group", share_count=5))
        self.engine.character_status = {}
        self.engine.refresh_character = refresh_character
        self.engine.prompts["share_commentary"] = "用当前人格自然说一两句真实感想。"
        await asyncio.to_thread(self.engine.store.initialize)
        self.plugin = plugin_module.create_plugin()
        self.plugin._set_context(self.ctx)
        self.plugin.engine = self.engine
        self.plugin._config_values = Settings(daily_enabled=False).model_dump()
        await self.engine.db("put", "job", {"id": "daily-test", "kind": "daily", "status": "done", "share_status": "queued"})
        self.images = []
        for color in ("red", "blue"):
            buffer = BytesIO()
            Image.new("RGB", (640, 480), color).save(buffer, format="PNG")
            self.images.append(await asyncio.to_thread(self.engine.store.save_image, buffer.getvalue(), "image/png", {
                "job_id": "daily-test", "author": "Test Photographer", "work": "Test Book", "bot_interest": 0.8,
                "observations": "A quiet patch of " + color, "feeling": "I like the open space", "identity_verified": False,
                "source_url": "https://photos.example/series", "image_url": "https://photos.example/" + color + ".png",
            }))
        self.host_builder = host_context_builder()

    def injected(self, arguments: Dict[str, Any], session_id: str = "real-group") -> Dict[str, Any]:
        context = SimpleNamespace(stream_id=session_id, session_id=session_id, group_id="platform-group", user_id="",
                                  platform="qq", metadata={})
        invocation = SimpleNamespace(arguments=arguments, stream_id=session_id, reasoning="")
        return self.host_builder._build_tool_invocation_payload(invocation, context)

    async def test_model_schema_omits_routing_and_host_context_sends_real_receipts(self) -> None:
        component = next(item for item in self.plugin.get_components() if item["name"] == "photography_share")
        encoded = json.dumps(component, default=str)
        self.assertNotIn('"name": "stream_id"', encoded)
        self.assertNotIn('"name": "chat_id"', encoded)
        payload = self.injected({"image_ids": self.images[0]["id"]})
        self.assertEqual(payload["stream_id"], "real-group")
        result = await self.plugin.photography_share(**payload)
        self.assertTrue(result["success"])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["stream_id"], "real-group")
        receipts = await self.engine.db("list", "share")
        self.assertEqual(receipts[0]["job_id"], "daily-test")
        self.assertEqual(receipts[0]["message_id"], "sent-1")
        job = await self.engine.db("get", "job", "daily-test")
        self.assertEqual(job["share_status"], "sent")
        self.assertEqual(job["share_delivered_count"], 1)
        self.assertEqual(self.generated, [])
        self.assertEqual(self.text_sent, [])

    async def test_old_message_id_reports_context_mismatch_without_sending(self) -> None:
        payload = self.injected({"stream_id": "ROBOT1.0_wrong-message-id", "image_ids": self.images[0]["id"]})
        self.assertEqual(payload["chat_id"], "real-group")
        result = await self.plugin.photography_share(**payload)
        self.assertEqual(result["error_code"], "context_mismatch")
        self.assertNotIn("请先", result["error"])
        self.assertEqual(self.sent, [])
        job = await self.engine.db("get", "job", "daily-test")
        self.assertEqual(job["share_status"], "rejected")
        self.assertEqual(job["share_error"], result["error"])

    async def test_wrong_group_and_forged_stream_never_send(self) -> None:
        result = await self.plugin.photography_share(**self.injected({}, "other-group"))
        self.assertEqual(result["error_code"], "wrong_chat")
        result = await self.plugin.photography_share(**self.injected({"stream_id": "real-group"}, "other-group"))
        self.assertEqual(result["error_code"], "context_mismatch")
        self.assertEqual(self.sent, [])

    async def test_configuration_and_missing_context_errors_are_distinct(self) -> None:
        result = await self.plugin.photography_share()
        self.assertEqual(result["error_code"], "missing_context")
        self.engine.settings.share_enabled = False
        result = await self.plugin.photography_share(**self.injected({}))
        self.assertEqual(result["error_code"], "sharing_disabled")
        self.engine.settings.share_enabled = True
        self.engine.settings.target_stream_id = ""
        result = await self.plugin.photography_share(**self.injected({}))
        self.assertEqual(result["error_code"], "target_unset")
        self.assertEqual(self.sent, [])

    async def test_partial_send_preserves_only_successful_receipt(self) -> None:
        self.send_results = [{"sent": True, "message_id": "real-receipt"}, {"sent": False}]
        result = await self.plugin.photography_share(**self.injected({}))
        self.assertEqual(result["error_code"], "send_failed")
        self.assertEqual(len(result["delivered"]), 1)
        self.assertEqual(await self.engine.db("count", "share"), 1)
        job = await self.engine.db("get", "job", "daily-test")
        self.assertEqual(job["share_status"], "partial")
        self.assertEqual(job["share_delivered_count"], 1)

    async def test_rpc_error_remains_visible_and_records_job_failure(self) -> None:
        self.send_results = [RuntimeError("adapter unavailable")]
        with self.assertRaisesRegex(RuntimeError, "adapter unavailable"):
            await self.plugin.photography_share(**self.injected({}))
        self.assertEqual(await self.engine.db("count", "share"), 0)
        job = await self.engine.db("get", "job", "daily-test")
        self.assertEqual(job["share_status"], "failed")
        self.assertIn("adapter unavailable", job["share_error"])

    async def test_snapshot_exposes_summaries_and_extended_job_status(self) -> None:
        self.engine.character_status = {"status": "ready", "summary": "已接入角色摘要", "raw_personality": "private text"}
        await self.engine.db("put", "fact_check", {"id": "fact-test", "question": "Book attribution?", "status": "resolved",
            "conclusion": "Verified", "sources": ["https://photo.example/book"], "search_query": "internal query", "raw_prompt": "private prompt"})
        await self.engine.db("put", "preview", {"id": "preview-test", "job_id": "daily-test", "title": "Preview"})
        await self.engine.db("put", "job", {"id": "daily-test", "preview_count": 8, "detail_count": 2,
                                            "stop_reason": "author_target", "share_status": "queued"})
        snapshot = await self.plugin.webui_snapshot()
        self.assertEqual(snapshot["persona_status"], {"status": "ready", "summary": "已接入角色摘要"})
        self.assertNotIn("search_query", snapshot["fact_checks"][0])
        self.assertNotIn("raw_prompt", snapshot["fact_checks"][0])
        self.assertEqual(snapshot["daily"]["preview_count"], 8)
        self.assertEqual(snapshot["daily"]["share_status"], "queued")
        records = await self.plugin.webui_records("fact_check")
        self.assertEqual(records["items"], snapshot["fact_checks"])
        self.assertEqual((await self.plugin.webui_records("preview"))["total"], 1)

    async def test_review_api_only_queues_an_existing_feelings_review(self) -> None:
        components = self.plugin.get_components()
        self.assertTrue(any(component["name"] == "webui_review" and component["type"] == "API" for component in components))
        await self.engine.db("put", "diary", {"id": "existing-diary", "content": "Original observations"})
        before_images = await self.engine.db("count", "image")
        result = await self.plugin.webui_review()
        self.assertTrue(result["ok"])
        job = await self.engine.db("get", "job", result["job_id"])
        self.assertEqual(job["kind"], "review")
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["intent"], "用现在的人格整理已有摄影感受")
        self.assertEqual(job["model_calls"], 0)
        self.assertEqual(await self.engine.db("count", "image"), before_images)
        self.assertEqual((await self.engine.db("get", "diary", "existing-diary"))["content"], "Original observations")
        self.assertEqual(self.sent, [])

    async def test_config_version_update_keeps_dashboard_settings(self) -> None:
        await self.plugin.webui_settings(json.dumps({"share_count": 11, "preview_min": 32, "preview_max": 64}))
        expected = self.engine.settings.model_dump()
        incoming = {"plugin": {"enabled": True, "config_version": "0.2.1"},
                    "photography": {**self.plugin._config_values, "max_steps": 12}}
        await self.plugin.on_config_update("self", incoming, "0.2.1")
        self.assertEqual(self.engine.settings.model_dump(), expected)
        self.assertEqual((await self.engine.db("get", "settings", "current"))["values"], expected)

    async def test_real_config_change_merges_with_concurrent_dashboard_change(self) -> None:
        incoming = {"photography": {**self.plugin._config_values, "max_images": 701}}
        await asyncio.gather(self.plugin.on_config_update("self", incoming, "0.2.0"),
                             self.plugin.webui_settings(json.dumps({"share_count": 12})))
        self.assertEqual(self.engine.settings.max_images, 701)
        self.assertEqual(self.engine.settings.share_count, 12)
        self.assertTrue(self.engine.settings.share_enabled)
        self.assertEqual(self.engine.settings.target_stream_id, "real-group")
        saved = (await self.engine.db("get", "settings", "current"))["values"]
        self.assertEqual(saved, self.engine.settings.model_dump())
        self.assertEqual(self.plugin._config_values["max_images"], 701)

    async def test_invalid_config_change_does_not_advance_snapshot(self) -> None:
        previous = dict(self.plugin._config_values)
        with self.assertRaises(ValueError):
            await self.plugin.on_config_update("self", {"photography": {**previous, "share_count": 0}}, "bad")
        self.assertEqual(self.plugin._config_values, previous)
        self.assertEqual(self.engine.settings.share_count, 5)

    async def test_repeat_collection_shares_are_associated_with_current_job(self) -> None:
        image_id = self.images[0]["id"]
        await self.engine.db("put", "job", {"id": "daily-test", "share_status": "not_requested"})
        await self.engine.db("put", "job", {"id": "daily-repeat", "kind": "daily", "status": "done",
                                            "share_status": "queued", "image_ids": [image_id]})
        await self.engine.db("put", "job", {"id": "unrelated-newer", "kind": "manual", "status": "done",
                                            "share_status": "queued", "image_ids": [self.images[1]["id"]]})
        result = await self.plugin.photography_share(**self.injected({"image_ids": image_id}))
        self.assertTrue(result["success"])
        receipt = (await self.engine.db("list", "share"))[0]
        self.assertEqual(receipt["job_id"], "daily-repeat")
        self.assertEqual(receipt["source_job_id"], "daily-repeat")
        self.assertEqual((await self.engine.db("get", "job", "daily-repeat"))["share_status"], "sent")
        self.assertEqual((await self.engine.db("get", "job", "daily-test"))["share_status"], "not_requested")
        self.assertEqual((await self.engine.db("get", "job", "unrelated-newer"))["share_status"], "queued")
        self.assertEqual((await self.engine.db("get", "image", image_id))["job_id"], "daily-test")

    async def test_share_now_queues_one_atomic_request_for_concurrent_clicks(self) -> None:
        first, second = await asyncio.gather(self.plugin.webui_share_now(), self.plugin.webui_share_now())
        self.assertEqual(first["mode"], "share_existing")
        self.assertEqual(second["mode"], "in_progress")
        self.assertEqual(first["job_id"], second["job_id"])
        job = await self.engine.db("get", "job", first["job_id"])
        self.assertEqual(job["kind"], "share")
        self.assertEqual(job["share_target_stream_id"], "real-group")
        self.assertEqual(set(job["image_ids"]), {image["id"] for image in self.images})
        self.assertEqual(job["share_status"], "queued")
        self.assertEqual(job["model_calls"], 0)
        self.assertEqual(await self.engine.db("count", "job"), 2)
        self.assertEqual(self.sent, [])
        await self.engine.db("put", "job", {"id": job["id"], "status": "running"})
        again = await self.plugin.webui_share_now()
        self.assertEqual(again["job_id"], job["id"])

    async def test_share_now_uses_actual_today_receipt_not_other_group_or_failed_job(self) -> None:
        await self.engine.db("put", "job", {"id": "daily-test", "share_status": "sent"})
        await self.engine.db("put", "share", {"id": "other", "stream_id": "other-group", "message_id": "actual",
            "image_id": self.images[0]["id"], "job_id": "daily-test"})
        result = await self.plugin.webui_share_now()
        self.assertEqual(result["mode"], "share_existing")
        await self.engine.db("put", "job", {"id": result["job_id"], "status": "failed"})
        await self.engine.db("put", "share", {"id": "today", "stream_id": "real-group", "message_id": "actual",
            "image_id": self.images[0]["id"], "job_id": "daily-test"})
        result = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", result["job_id"])
        self.assertEqual(result["mode"], "explore_then_share")
        self.assertEqual(job["kind"], "share_cycle")
        self.assertEqual(job["image_ids"], [])
        self.assertEqual(self.sent, [])

    def test_today_receipts_follow_singapore_midnight_and_require_success(self) -> None:
        now = datetime(2026, 11, 13, 16, 0, tzinfo=timezone.utc)
        receipt = {"stream_id": "real-group", "message_id": "actual", "created_at": "2026-11-13T15:59:59+00:00"}
        self.assertFalse(sharing.has_receipt_today([receipt], "real-group", now))
        receipt["created_at"] = "2026-11-13T16:00:00+00:00"
        self.assertTrue(sharing.has_receipt_today([receipt], "real-group", now))
        receipt["sent"] = False
        self.assertFalse(sharing.has_receipt_today([receipt], "real-group", now))
        receipt.pop("sent")
        receipt["message_id"] = ""
        self.assertFalse(sharing.has_receipt_today([receipt], "real-group", now))

    async def test_previous_days_images_can_be_resent_but_unshared_are_preferred(self) -> None:
        self.engine.settings.share_count = 1
        yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        await self.engine.db("put", "share", {"id": "earlier", "stream_id": "real-group", "message_id": "old",
            "image_id": self.images[0]["id"], "job_id": "daily-test", "created_at": yesterday})
        result = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", result["job_id"])
        self.assertEqual(job["image_ids"], [self.images[1]["id"]])
        await self.engine.db("put", "job", {"id": job["id"], "status": "cancelled"})
        await self.engine.db("put", "share", {"id": "also-earlier", "stream_id": "real-group", "message_id": "old2",
            "image_id": self.images[1]["id"], "job_id": "daily-test", "created_at": yesterday})
        result = await self.plugin.webui_share_now()
        self.assertEqual(result["mode"], "share_existing")
        self.assertEqual(len((await self.engine.db("get", "job", result["job_id"]))["image_ids"]), 1)

    async def test_no_available_images_queues_browsing_before_share(self) -> None:
        await asyncio.to_thread(self.engine.store.cleanup, 0, 1024 * 1024, 4)
        result = await self.plugin.webui_share_now()
        self.assertEqual(result["mode"], "explore_then_share")
        self.assertEqual(self.sent, [])

    async def test_dispatch_receipts_belong_to_click_and_repeat_does_not_resend(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(result["delivered_count"], 2)
        receipts = await self.engine.db("list", "share")
        self.assertEqual({receipt["job_id"] for receipt in receipts}, {job["id"]})
        self.assertTrue(all(receipt["sent"] for receipt in receipts))
        self.assertEqual((await self.engine.db("get", "job", "daily-test"))["share_status"], "queued")
        self.assertEqual((await self.engine.db("get", "image", self.images[0]["id"]))["job_id"], "daily-test")
        again = await self.plugin._dispatch_share_job(job)
        self.assertTrue(again["success"])
        self.assertEqual(len(self.sent), 2)
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["status"], "done")
        self.assertEqual(saved["share_status"], "sent")
        self.assertEqual(saved["share_delivered_count"], 2)
        self.assertEqual(saved["share_commentary_status"], "sent")
        self.assertEqual(saved["share_commentary_message_id"], "text-1")
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(self.text_sent), 1)
        self.assertEqual((await self.plugin.webui_share_now())["mode"], "explore_then_share")

    async def test_partial_click_retry_only_sends_unconfirmed_images(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        self.send_results = [{"sent": True, "message_id": "first"}, {"sent": False}]
        result = await self.plugin._dispatch_share_job(job)
        self.assertFalse(result["success"])
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["status"], "partial")
        self.assertEqual(saved["share_status"], "partial")
        self.assertEqual(saved["share_delivered_count"], 1)
        self.assertEqual(self.generated, [])
        self.assertEqual(self.text_sent, [])
        confirmed_id = (await self.engine.db("list", "share"))[0]["image_id"]
        await self.engine.retry(job["id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        receipts = await self.engine.db("list", "share")
        self.assertEqual(len(receipts), 2)
        self.assertEqual(sum(receipt["image_id"] == confirmed_id for receipt in receipts), 1)
        self.assertEqual(len(self.sent), 3)

    async def test_share_dispatch_rejects_changed_or_unavailable_group(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        self.engine.settings.target_stream_id = "other-group"
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "target_changed")
        self.engine.settings.target_stream_id = "real-group"

        async def private_stream(**kwargs: Any) -> List[Dict[str, Any]]:
            return [{"session_id": "real-group", "is_group_session": False}]

        self.ctx.chat.get_all_streams = private_stream
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "target_unavailable")
        self.assertEqual(self.sent, [])

    async def test_target_change_during_batch_stops_remaining_images(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        original_send = self.ctx.send.image

        async def change_target_after_send(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = await original_send(*args, **kwargs)
            self.engine.settings.target_stream_id = "other-group"
            return result

        self.ctx.send.image = change_target_after_send
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "target_changed")
        self.assertEqual(len(self.sent), 1)
        self.assertEqual((await self.engine.db("get", "job", job["id"]))["status"], "partial")

    async def test_cycle_only_shares_its_observed_images_and_does_not_substitute_old_collection(self) -> None:
        job = await self.engine.enqueue("share_cycle", initial_data={"share_target_stream_id": "real-group",
            "share_mode": "explore_then_share", "share_status": "queued"})
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "no_local_images")
        self.assertEqual(self.sent, [])
        await self.engine.db("put", "job", {"id": job["id"], "image_ids": [self.images[1]["id"]], "status": "running"})
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual((await self.engine.db("list", "share"))[0]["image_id"], self.images[1]["id"])

    async def test_missing_receipt_and_rpc_exception_stay_visible_on_click_job(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        self.send_results = [{"sent": True, "message_id": ""}]
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "missing_receipt")
        self.assertEqual(await self.engine.db("count", "share"), 0)
        self.send_results = [RuntimeError("adapter unavailable")]
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "send_error")
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["status"], "failed")
        self.assertIn("adapter unavailable", saved["share_error"])

    async def test_rewrite_diaries_api_queues_without_changing_content(self) -> None:
        await self.engine.db("put", "diary", {"id": "old", "content": "original"})
        first = await self.plugin.webui_rewrite_diaries()
        second = await self.plugin.webui_rewrite_diaries()
        self.assertEqual(first["job_id"], second["job_id"])
        job = await self.engine.db("get", "job", first["job_id"])
        self.assertEqual(job["kind"], "rewrite_diaries")
        self.assertEqual(job["model_calls"], 0)
        self.assertEqual((await self.engine.db("get", "diary", "old"))["content"], "original")
        self.assertEqual(self.sent, [])

    async def test_seed_viewed_count_uses_seed_before_exact_author(self) -> None:
        await self.engine.db("put", "seed", {"id": "seed-a", "author": "Artist A"})
        await self.engine.db("put", "seed", {"id": "seed-b", "author": "Artist B"})
        records = [{"id": "one", "seed_id": "seed-a", "author": "Artist B"},
                   {"id": "two", "seed_id": "", "author": "Artist A"},
                   {"id": "three", "author": "Artist B"},
                   {"id": "four", "author": "Artist A Jr"},
                   {"id": "five", "seed_id": "seed-a", "author": "Artist A"}]
        for record in records:
            await self.engine.db("put", "observation", record)
        seeds = {seed["id"]: seed for seed in (await self.plugin.webui_snapshot())["seeds"]}
        self.assertEqual(seeds["seed-a"]["viewed_count"], 3)
        self.assertEqual(seeds["seed-b"]["viewed_count"], 1)

    async def test_commentary_uses_delivered_observations_and_fresh_scoped_context_only(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        await self.engine.db("put", "observation", {"id": "current-observation", "image_id": self.images[0]["id"],
            "observations": "CURRENT_VISUAL_NOTE", "feeling": "CURRENT_FEELING", "identity_verified": False})
        await self.engine.db("put", "job", {"id": job["id"], "observation_ids": ["current-observation"]})
        original_send = self.ctx.send.text

        async def assert_text_saved_before_send(text: str, stream_id: str, **kwargs: Any) -> Dict[str, Any]:
            stored = await self.engine.db("get", "job", job["id"])
            self.assertEqual(stored["share_commentary_text"], text)
            self.assertEqual(stored["share_commentary_status"], "sending")
            return await original_send(text, stream_id, **kwargs)

        self.ctx.send.text = assert_text_saved_before_send
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(self.character_refreshes, 1)
        self.assertEqual(len(self.generated), 1)
        call = self.generated[0]
        self.assertEqual(call["format_instruction"], self.engine.prompts["share_commentary"])
        reference = json.loads(call["reference"])
        self.assertEqual(reference["kind"], "share")
        self.assertEqual(len(reference["photographs"]), 2)
        self.assertTrue(all(item["message_id"] for item in reference["photographs"]))
        self.assertTrue(any(item["observations"] == "CURRENT_VISUAL_NOTE" for item in reference["photographs"]))
        self.assertIn("RECENT_SCOPED_CHAT", call["reference"])
        self.assertIn("SCOPED_FRIEND", call["reference"])
        self.assertNotIn("OTHER_GROUP", call["reference"])
        self.assertNotIn("PERSONALITY_NOT_IN_REFERENCE", call["reference"])
        saved = await self.engine.db("get", "job", job["id"])
        self.assertNotIn("RECENT_SCOPED_CHAT", json.dumps(saved))
        self.assertNotIn("SCOPED_FRIEND", json.dumps(saved))
        self.assertIn("并发出了感想", saved["message"])
        self.assertTrue(self.text_sent[0]["return_details"])
        self.assertTrue(self.text_sent[0]["sync_to_maisaka_history"])

    async def test_commentary_generation_failure_retry_keeps_image_receipts(self) -> None:
        self.generate_results = [{"success": False, "error": "Replyer unavailable"}]
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "commentary_failed")
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["status"], "partial")
        self.assertEqual(saved["share_status"], "sent")
        self.assertEqual(saved["share_commentary_status"], "failed")
        self.assertEqual(self.text_sent, [])
        await self.engine.retry(job["id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(len(self.text_sent), 1)

    async def test_explicit_text_refusal_retries_saved_text_without_regenerating(self) -> None:
        self.text_results = [{"sent": False, "error": "adapter refused"}]
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "commentary_failed")
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["share_status"], "sent")
        self.assertTrue(saved["share_commentary_text"])
        await self.engine.retry(job["id"])
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(self.text_sent), 2)
        self.assertEqual(self.text_sent[0]["text"], self.text_sent[1]["text"])

    async def test_text_unknown_receipt_or_rpc_error_never_automatically_resends(self) -> None:
        for result in ({"sent": True, "message_id": ""}, RuntimeError("connection interrupted")):
            with self.subTest(result=str(result)):
                job = await self.engine.enqueue("share", initial_data={"share_target_stream_id": "real-group",
                    "image_ids": [self.images[0]["id"]], "share_status": "queued"})
                self.text_results = [result]
                delivered = await self.plugin._dispatch_share_job(job)
                self.assertEqual(delivered["error_code"], "commentary_uncertain")
                before_images, before_text = len(self.sent), len(self.text_sent)
                await self.engine.retry(job["id"])
                delivered = await self.plugin._dispatch_share_job(job)
                self.assertEqual(delivered["error_code"], "commentary_uncertain")
                self.assertEqual(len(self.sent), before_images)
                self.assertEqual(len(self.text_sent), before_text)
                self.assertEqual((await self.engine.db("get", "job", job["id"]))["share_status"], "sent")

    async def test_target_change_or_cancel_after_generation_prevents_text_send(self) -> None:
        original_generate = self.ctx.maisaka.reply.generate
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])

        async def change_target(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = await original_generate(*args, **kwargs)
            self.engine.settings.target_stream_id = "other-group"
            return result

        self.ctx.maisaka.reply.generate = change_target
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "commentary_failed")
        self.assertEqual(self.text_sent, [])
        self.assertEqual((await self.engine.db("get", "job", job["id"]))["share_status"], "sent")
        self.engine.settings.target_stream_id = "real-group"
        other = await self.engine.enqueue("share", initial_data={"share_target_stream_id": "real-group",
            "image_ids": [self.images[0]["id"]], "share_status": "queued"})

        async def cancel_after_generate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            result = await original_generate(*args, **kwargs)
            self.engine.cancelled.add(other["id"])
            return result

        self.ctx.maisaka.reply.generate = cancel_after_generate
        result = await self.plugin._dispatch_share_job(other)
        self.assertEqual(result["error_code"], "commentary_failed")
        self.assertEqual(self.text_sent, [])

    async def test_cancelled_text_rpc_is_uncertain_and_restart_does_not_send_again(self) -> None:
        self.text_results = [asyncio.CancelledError()]
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        with self.assertRaises(asyncio.CancelledError):
            await self.plugin._dispatch_share_job(job)
        saved = await self.engine.db("get", "job", job["id"])
        self.assertEqual(saved["share_commentary_status"], "uncertain")
        self.assertEqual(saved["share_status"], "sent")
        result = await self.plugin._dispatch_share_job(job)
        self.assertEqual(result["error_code"], "commentary_uncertain")
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.text_sent), 1)

    async def test_old_completed_picture_only_job_is_not_backfilled(self) -> None:
        job = await self.engine.enqueue("share", initial_data={"share_target_stream_id": "real-group",
            "image_ids": [self.images[0]["id"]], "share_status": "queued"})
        await self.engine.db("put", "job", {"id": job["id"], "status": "done", "share_status": "sent", "share_delivered_count": 1})
        result = await self.plugin._dispatch_share_job(job)
        self.assertTrue(result["success"])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.text_sent, [])
        self.assertEqual(self.generated, [])

    async def test_completed_commentary_receipt_is_reused_even_after_target_changes(self) -> None:
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        self.assertTrue((await self.plugin._dispatch_share_job(job))["success"])
        self.engine.settings.target_stream_id = "other-group"
        self.engine.settings.share_enabled = False
        self.assertTrue((await self.plugin._dispatch_share_job(job))["success"])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(len(self.text_sent), 1)
        self.assertEqual((await self.engine.db("get", "job", job["id"]))["status"], "done")

    async def test_cancelled_generation_can_retry_because_text_was_not_sent(self) -> None:
        self.generate_results = [asyncio.CancelledError()]
        response = await self.plugin.webui_share_now()
        job = await self.engine.db("get", "job", response["job_id"])
        with self.assertRaises(asyncio.CancelledError):
            await self.plugin._dispatch_share_job(job)
        self.assertEqual(self.text_sent, [])
        self.assertEqual((await self.engine.db("get", "job", job["id"]))["share_commentary_status"], "failed")
        self.assertTrue((await self.plugin._dispatch_share_job(job))["success"])
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(len(self.text_sent), 1)
