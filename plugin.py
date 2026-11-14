"""MaiBot SDK entry point: dashboard, tools, and an observation-only chat hook."""

from base64 import b64encode
from maibot_sdk import API, Command, HookHandler, MaiBotPlugin, Tool
from maibot_sdk.types import HookMode, ToolParameterInfo, ToolParamType
from typing import Any, Dict, List
from uuid import uuid4

import asyncio
import json

from .config import PhotographyConfig, Settings
from .sharing import ShareJobsMixin
from .workflow import PhotographyEngine, stamp


def string_parameter(name: str, description: str, required: bool = False) -> ToolParameterInfo:
    return ToolParameterInfo(name=name, param_type=ToolParamType.STRING, description=description, required=required)


def compact_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Bound WebUI/Tool payloads; full browse traces remain in local storage."""
    keys = ("id", "kind", "intent", "status", "created_at", "updated_at", "finished_at", "progress", "message", "error", "model_calls", "unresolved_authors",
            "preview_count", "detail_count", "stop_reason", "share_status", "share_error", "share_delivered_count", "share_attempted_at", "share_mode",
            "share_commentary_status", "share_commentary_text", "share_commentary_message_id", "share_commentary_error", "share_commentary_attempted_at")
    return {**{key: job[key] for key in keys if key in job}, "errors": job.get("errors", [])[-3:],
            "error_count": len(job.get("errors", [])), "image_count": len(job.get("image_ids", [])),
            "observation_count": len(job.get("observation_ids", []))}


def compact_fact_check(record: Dict[str, Any]) -> Dict[str, Any]:
    """Expose fact-check conclusions and sources without internal model inputs."""
    keys = ("id", "question", "status", "conclusion", "sources", "checked_at", "error")
    return {key: record[key] for key in keys if key in record}


class PhotographyPlugin(ShareJobsMixin, MaiBotPlugin):
    config_model = PhotographyConfig

    def __init__(self) -> None:
        super().__init__()
        self._config_values: Dict[str, Any] = {}
        self._settings_lock = asyncio.Lock()
        self._share_request_lock = asyncio.Lock()

    async def on_load(self) -> None:
        config = self.get_plugin_config_data().get("photography", {})
        configured = Settings.model_validate(config)
        self._config_values = configured.model_dump()
        self.engine = PhotographyEngine(self.ctx, configured)
        self.engine.share_dispatch = self._dispatch_share_job
        await self.engine.start()
        self.ctx.logger.info("摄影漫游已加载；任务与图片保存在插件专属数据目录")

    async def on_unload(self) -> None:
        await self.engine.stop()

    async def on_config_update(self, scope: str, config_data: Dict[str, Any], version: str) -> None:
        if scope == "self":
            incoming = Settings.model_validate(config_data.get("photography", {})).model_dump()
            async with self._settings_lock:
                changes = {key: value for key, value in incoming.items() if self._config_values.get(key) != value}
                settings = Settings.model_validate({**self.engine.settings.model_dump(), **changes})
                await self.engine.db("put", "settings", {"id": "current", "values": settings.model_dump()})
                self.engine.settings = settings
                self._config_values = incoming

    async def _groups(self) -> list:
        streams = await self.ctx.chat.get_all_streams(platform="all_platforms")
        if not isinstance(streams, list):
            raise RuntimeError("Host 未返回聊天流列表")
        return [{"id": s["session_id"], "name": s["group_name"] or f"{s['platform']} 群（尚未命名）"}
                for s in streams if s["is_group_session"]]

    @API("webui_snapshot", version="1")
    async def webui_snapshot(self, **kwargs: Any) -> Dict[str, Any]:
        e = self.engine
        stats, profile, seeds, jobs, sources, groups, fact_checks, observations = await asyncio.gather(
            e.db("stats"), e.db("get", "profile", "current"), e.db("list", "seed", limit=100),
            e.db("list", "job", limit=15), e.db("list", "source", limit=200), self._groups(),
            e.db("list", "fact_check", limit=8), e.db("list", "observation", limit=10000))
        for seed in seeds:
            seed["viewed_count"] = sum(
                observation.get("seed_id") == seed["id"] or
                (not observation.get("seed_id") and observation.get("author") == seed["author"])
                for observation in observations)
        # Keep RPC snapshots small; detailed task traces stay in the records API.
        jobs = [compact_job(job) for job in jobs]
        for source in sources:
            source.update(bot_interest=source.get("bot_score", 0.5), human_interest=source.get("human_score"),
                          samples=source.get("observations", 0), weight=e.source_weight(source))
            source.pop("evidence", None)
        deployment = await e.db("get", "system", "deployment")
        embeddings = await e.db("list", "visual_embedding", limit=1)
        return {"stats": stats, "settings": e.settings.model_dump(), "profile": profile, "seeds": seeds,
                "jobs": jobs, "sources": sources, "groups": groups,
                "persona_status": {key: e.character_status[key] for key in ("status", "source", "summary", "updated_at", "error")
                                   if key in e.character_status},
                "fact_checks": [compact_fact_check(record) for record in fact_checks],
                "daily": next((j for j in jobs if j["kind"] == "daily"), None),
                "visual_memory": {"vectors": await e.db("count", "visual_embedding"),
                                  "model": embeddings[0]["model_name"] if embeddings else "",
                                  "dimension": embeddings[0]["dimension"] if embeddings else None},
                "jina_configured": bool(deployment and deployment.get("jina_configured"))}

    @API("webui_records", version="1")
    async def webui_records(self, kind: str, offset: int = 0, limit: int = 20, **kwargs: Any) -> Dict[str, Any]:
        if kind not in {"diary", "image", "feedback", "job", "fact_check", "preview"} or offset < 0 or not 1 <= limit <= 30:
            raise ValueError("记录类型或分页参数无效")
        if kind == "feedback":
            records = await self.engine.db("list", "feedback", limit=10000)
            visible = [record for record in records if record.get("status") in {"pending", "failed"}
                       or record.get("status") == "evaluated" and record.get("is_photography_related") is True]
            return {"items": visible[offset:offset + limit], "total": len(visible)}
        items, total = await asyncio.gather(self.engine.db("list", kind, limit=limit, offset=offset), self.engine.db("count", kind))
        if kind == "job":
            items = [compact_job(item) for item in items]
        elif kind == "fact_check":
            items = [compact_fact_check(item) for item in items]
        return {"items": items, "total": total}

    @API("webui_thumbnail", version="1")
    async def webui_thumbnail(self, image_id: str, **kwargs: Any) -> Dict[str, Any]:
        return {"data_url": await asyncio.to_thread(self.engine.store.thumbnail, image_id)}

    @API("webui_import_seeds", version="1")
    async def webui_import_seeds(self, content: str, filename: str = "摄影师与作品.txt", **kwargs: Any) -> Dict[str, Any]:
        return await self.engine.import_seeds(content, filename)

    @API("webui_import_sites", version="1")
    async def webui_import_sites(self, content: str, **kwargs: Any) -> Dict[str, Any]:
        return await self.engine.import_sites(content)

    @API("webui_explore", version="1")
    async def webui_explore(self, intent: str = "", **kwargs: Any) -> Dict[str, Any]:
        job = await self.engine.enqueue("manual", intent or "根据当前审美继续看看有意思的摄影作品")
        return {"ok": True, "job_id": job["id"], "message": "已排队去看摄影作品；看过后会整理感受、更新审美，并写下摄影日记。"}

    @API("webui_review", version="1")
    async def webui_review(self, **kwargs: Any) -> Dict[str, Any]:
        job = await self.engine.enqueue("review", "用现在的人格整理已有摄影感受")
        return {"ok": True, "job_id": job["id"]}

    @API("webui_visual_index", version="1")
    async def webui_visual_index(self, **kwargs: Any) -> Dict[str, Any]:
        job = await self.engine.enqueue("visual_index", "为已有摄影收藏整理视觉记忆")
        return {"ok": True, "job_id": job["id"]}

    @API("webui_rewrite_diaries", version="1")
    async def webui_rewrite_diaries(self, **kwargs: Any) -> Dict[str, Any]:
        job = await self.engine.enqueue("rewrite_diaries", "用她现在的人格重新整理已有摄影日记")
        return {"ok": True, "job_id": job["id"]}

    @API("webui_share_now", version="1")
    async def webui_share_now(self, **kwargs: Any) -> Dict[str, Any]:
        return await self._enqueue_share_now()

    @API("webui_cancel", version="1")
    async def webui_cancel(self, job_id: str, **kwargs: Any) -> Dict[str, Any]:
        return await self.engine.cancel(job_id)

    @API("webui_retry", version="1")
    async def webui_retry(self, job_id: str, **kwargs: Any) -> Dict[str, Any]:
        return await self.engine.retry(job_id)

    @API("webui_pin", version="1")
    async def webui_pin(self, image_id: str, pinned: bool, **kwargs: Any) -> Dict[str, Any]:
        async with self.engine.collection_lock:
            image = await self.engine.db("get", "image", image_id)
            if image is None:
                raise ValueError("图片不存在")
            await self.engine.db("put", "image", {"id": image_id, "pinned": pinned})
        return {"ok": True}

    @API("webui_settings", version="1")
    async def webui_settings(self, settings_json: str, **kwargs: Any) -> Dict[str, Any]:
        updates = json.loads(settings_json)
        if not isinstance(updates, dict) or set(updates) - set(Settings.model_fields):
            raise ValueError("设置包含未知字段")
        async with self._settings_lock:
            settings = Settings.model_validate({**self.engine.settings.model_dump(), **updates})
            if settings.target_stream_id and settings.target_stream_id not in {g["id"] for g in await self._groups()}:
                raise ValueError("请选择 Host 中已经存在的真实群聊")
            if settings.share_enabled and not settings.target_stream_id:
                raise ValueError("启用分享前需要选择目标群")
            await self.engine.db("put", "settings", {"id": "current", "values": settings.model_dump()})
            self.engine.settings = settings
        return {"ok": True, "settings": settings.model_dump()}

    @API("webui_cleanup", version="1")
    async def webui_cleanup(self, **kwargs: Any) -> Dict[str, Any]:
        # Disk work happens in a background job, within the WebUI's 10s RPC budget.
        job = await self.engine.enqueue("cleanup")
        return {"ok": True, "job_id": job["id"]}

    @Tool("photography_explore", description="当用户说再去看看摄影、找更多作品、探索某位摄影师时，安排一次摄影漫游：看图、整理感受、更新审美，并写下摄影日记。返回排队进度。",
          parameters=[string_parameter("intent", "希望寻找的作者、影集、风格或问题")])
    async def photography_explore(self, intent: str = "", **kwargs: Any) -> Dict[str, Any]:
        return await self.webui_explore(intent)

    @Tool("photography_memory", description="查询真实看过的摄影作品、图片编号、摄影日记、审美依据或浏览任务进度。记不清先查，不编造看图经历。",
          parameters=[string_parameter("query", "作者、作品名、图片编号或任务编号；空值查看最近记录")])
    async def photography_memory(self, query: str = "", **kwargs: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for kind in ("image", "diary", "job", "feedback"):
            records = await self.engine.db("list", kind, limit=300)
            selected = [r for r in records if not query or query.casefold() in json.dumps(r, ensure_ascii=False).casefold()][:6]
            if kind == "job":
                selected = [compact_job(record) for record in selected]
            result[kind] = selected
        result["profile"] = await self.engine.db("get", "profile", "current")
        result["note"] = "local_status=evicted 表示只剩当时的记录，不能声称刚刚又看到了原图"
        return result

    async def _sharing_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve each image's current sharing job without changing its collection history."""
        jobs = await self.engine.db("list", "job", limit=10000)
        pending = sorted((job for job in jobs if job["kind"] in {"daily", "manual"}
                          and job.get("share_status") in {"queued", "sending", "failed"}),
                         key=lambda job: job["created_at"], reverse=True)
        return [{**image, "share_job_id": next((job["id"] for job in pending if image["id"] in job.get("image_ids", [])),
                                               image.get("job_id", ""))} for image in images]

    async def _record_share_status(self, images: List[Dict[str, Any]], status: str, error: str = "") -> None:
        """Associate sharing attempts with their exploration jobs and real receipts."""
        job_ids = {image["share_job_id"] for image in images if image["share_job_id"]}
        if not job_ids:
            return
        receipts = await self.engine.db("list", "share", limit=10000)
        for job_id in job_ids:
            if await self.engine.db("get", "job", job_id) is not None:
                await self.engine.db("put", "job", {
                    "id": job_id, "share_status": status, "share_error": error, "share_attempted_at": stamp(),
                    "share_delivered_count": sum(receipt.get("job_id") == job_id for receipt in receipts),
                })

    async def _send_selected(self, selected: List[Dict[str, Any]], stream_id: str,
                             request_job_id: str = "") -> Dict[str, Any]:
        """Send under sharing_lock; receipts belong to the caller's selected share job."""
        e = self.engine
        await self._record_share_status(selected, "sending")
        delivered: List[Dict[str, Any]] = []
        try:
            for item in selected:
                error_code, error = "", ""
                if request_job_id and request_job_id in e.cancelled:
                    error_code, error = "cancelled", "分享任务已取消"
                elif not e.settings.share_enabled:
                    error_code, error = "sharing_disabled", "摄影分享已关闭"
                elif stream_id != e.settings.target_stream_id:
                    error_code, error = "target_changed", "固定群设置已改变，请重新开始分享"
                if error:
                    await self._record_share_status(selected, "partial" if delivered else "failed", error)
                    return {"success": False, "delivered": delivered, "error_code": error_code, "error": error}
                # Protect image bytes while reading, then release the disk lock before RPC.
                async with e.collection_lock:
                    data, _ = await asyncio.to_thread(e.store.image_bytes, item["id"])
                result = await self.ctx.send.image(b64encode(data).decode(), stream_id, return_details=True)
                if not isinstance(result, dict) or not result.get("sent"):
                    error_code, error = "send_failed", "图片发送失败，未将失败项计入分享记录"
                elif not isinstance(result.get("message_id"), str) or not result["message_id"].strip():
                    error_code, error = "missing_receipt", "已请求发送，但宿主未返回有效消息回执，未计为成功分享"
                if error:
                    await self._record_share_status(selected, "partial" if delivered else "failed", error)
                    return {"success": False, "delivered": delivered, "error_code": error_code, "error": error}
                receipt = {"id": uuid4().hex, "stream_id": stream_id, "message_id": result["message_id"],
                           "image_id": item["id"], "job_id": item["share_job_id"], "source_job_id": item["share_job_id"],
                           "sent": True, "created_at": stamp()}
                await e.db("put", "share", receipt)
                delivered.append({"image_id": item["id"], "author": item.get("author"), "work": item.get("work"),
                                  "observations": item.get("observations"), "feeling": item.get("feeling"),
                                  "source_url": item.get("source_url"), "message_id": result["message_id"]})
        except Exception as exc:
            await self._record_share_status(selected, "partial" if delivered else "failed", f"{type(exc).__name__}: {exc}")
            raise
        await self._record_share_status(selected, "sent")
        # Main conversation uses these grounded notes to phrase the accompanying reply.
        return {"success": True, "delivered": delivered,
                "instruction": "图片已发出，请结合现场自然聊一两句具体感受，并交代作者/来源；不整篇转发日记。"}

    @Tool("photography_share", description="当指定群里用户说再多发点、分享刚看的摄影时，从本插件收藏中选少量未分享作品，发真实图片并保留回执。当前聊天由宿主自动提供，只发到后台设置的固定群。",
          parameters=[string_parameter("image_ids", "可选，photography_memory 返回的精确图片编号，逗号分隔；留空自动选未发过的作品")])
    async def photography_share(self, stream_id: str = "", image_ids: str = "", **kwargs: Any) -> Dict[str, Any]:
        e = self.engine
        async with e.sharing_lock:
            requested = [part.strip() for part in image_ids.split(",") if part.strip()]
            requested = await asyncio.to_thread(e.store.resolve_image_ids, requested)
            images = await self._sharing_images(await e.db("list", "image", limit=5000))
            requested_images = [image for image in images if image["id"] in requested]
            # Host injects chat_id and stream_id; neither is a model-facing parameter.
            chat_id = kwargs.get("chat_id", "")
            error_code, error = "", ""
            if not isinstance(chat_id, str) or not isinstance(stream_id, str):
                error_code, error = "invalid_context", "宿主提供的聊天上下文格式无效，未发送图片"
            elif chat_id and stream_id and chat_id != stream_id:
                error_code, error = "context_mismatch", "聊天上下文不一致：消息编号不能作为聊天流编号；请使用宿主注入的当前聊天上下文，未发送图片"
            else:
                stream_id = chat_id or stream_id
                if not stream_id:
                    error_code, error = "missing_context", "缺少宿主注入的当前聊天上下文，未发送图片"
                elif not e.settings.share_enabled:
                    error_code, error = "sharing_disabled", "摄影分享尚未启用，请在摄影漫游后台开启分享"
                elif not e.settings.target_stream_id:
                    error_code, error = "target_unset", "尚未选择摄影分享的固定群，请在摄影漫游后台选择"
                elif stream_id != e.settings.target_stream_id:
                    error_code, error = "wrong_chat", "当前聊天不是摄影分享设置中的固定群，未发送图片"
            if error:
                await self._record_share_status(requested_images, "rejected", error)
                return {"success": False, "error_code": error_code, "error": error}
            shares = await self._canonical_shares()
            sent_ids = {s["image_id"] for s in shares if s["stream_id"] == stream_id}
            candidates = [im for im in images if im["local_status"] == "available" and
                          (im["id"] in requested if requested else im["id"] not in sent_ids)]
            candidates.sort(key=lambda im: im.get("bot_interest", 0.5), reverse=True)
            selected = candidates[:e.settings.share_count]
            if not selected:
                error = "没有符合条件的本地作品。可以调用 photography_explore 再去看看"
                await self._record_share_status(requested_images, "failed", error)
                return {"success": False, "error_code": "no_local_images", "error": error}
            return await self._send_selected(selected, stream_id)

    @Command("photography", description="查看摄影漫游状态或开始一次漫游", pattern=r"^/摄影(?:\s+(?P<intent>.*))?$")
    async def photography_command(self, **kwargs: Any) -> Any:
        groups = kwargs.get("matched_groups", {})
        intent = groups.get("intent", "")
        if intent:
            result = await self.webui_explore(intent)
            text = "已经开始摄影漫游，任务：" + result["job_id"]
        else:
            stats = await self.engine.db("stats")
            text = f"摄影收藏 {stats['image_count']} 张；用 /摄影 想看的作者或方向 启动浏览。日记和进度在 WebUI 的摄影漫游页面。"
        await self.ctx.send.text(text, kwargs["stream_id"])
        return True, text, 2

    @API("webui_backfill_feedback", version="1")
    async def webui_backfill_feedback(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.engine.backfill_recent_feedback()

    @HookHandler("chat.receive.after_process", name="photography_feedback", mode=HookMode.OBSERVE,
                 description="记录固定群对已分享作品的明确引用与近期摄影讨论，后台分别整理单图、整批及一般意见")
    async def photography_feedback(self, message: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        await self.engine.capture_feedback(message)
        return {"action": "continue"}


def create_plugin() -> PhotographyPlugin:
    return PhotographyPlugin()
