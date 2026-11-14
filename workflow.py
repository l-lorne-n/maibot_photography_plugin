"""Persistent, bounded photography exploration, independent of the Host process."""

from base64 import b64encode
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlsplit
from uuid import uuid4

import asyncio
import json
import time

from .browsing import BrowsingMixin
from .character import load_character_context
from .config import Settings
from .diary import DiaryMixin
from .errors import ModelFailure
from .facts import FactCheckingMixin
from .feedback_runtime import FeedbackRuntimeMixin
from .reader import Reader, parse_seeds, parse_sites, prepare_vision_image, seed_work_items
from .store import Store
from .visual_context import VisualContextMixin


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出必须为 JSON 对象")
    return value


class PhotographyEngine(BrowsingMixin, DiaryMixin, FactCheckingMixin, FeedbackRuntimeMixin, VisualContextMixin):
    def __init__(self, ctx: Any, settings: Settings) -> None:
        self.ctx = ctx
        self.settings = settings
        self.store = Store(Path(ctx.paths.data_dir))
        self.reader = Reader(ctx)
        self.prompts: Dict[str, str] = {}
        self.character_context: Dict[str, Any] = {}
        self.character_status: Dict[str, Any] = {"status": "pending", "summary": "正在读取日常人格与本群朋友资料"}
        self.worker: Optional[asyncio.Task] = None
        self.scheduler: Optional[asyncio.Task] = None
        self.active_id = ""
        self.cancelled: set = set()
        self.wake = asyncio.Event()
        self.queue_lock = asyncio.Lock()
        self.collection_lock = asyncio.Lock()
        self.sharing_lock = asyncio.Lock()
        self.feedback_capture_lock = asyncio.Lock()
        self.share_dispatch: Optional[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None
        self.stopping = False

    async def db(self, method: str, *args: Any, **kwargs: Any) -> Any:
        # This dispatch is deliberately restricted to the Store's synchronous API.
        operations = {"get": self.store.get, "put": self.store.put, "list": self.store.list,
                      "count": self.store.count, "stats": self.store.stats}
        return await asyncio.to_thread(operations[method], *args, **kwargs)

    async def save_job_progress(self, job: Dict[str, Any]) -> None:
        # Sharing can finish concurrently in the conversation runner. Progress
        # writes own only exploration fields, so they cannot overwrite receipts.
        await self.db("put", "job", {key: value for key, value in job.items()
                                     if not key.startswith("share_") and key != "proactive_receipt"})

    async def start(self) -> None:
        await asyncio.to_thread(self.store.initialize)
        raw = await asyncio.to_thread((Path(__file__).parent / "prompts.json").read_text, encoding="utf-8")
        self.prompts = json.loads(raw)["zh-CN"]
        saved = await self.db("get", "settings", "current")
        if saved:
            self.settings = Settings.model_validate(saved["values"])
        await self.refresh_character()
        await self.refresh_seed_experience()
        # A runner restart resumes persisted work instead of silently losing it.
        for job in await self.db("list", "job", limit=10000):
            if job["kind"] == "fact_check" and job["status"] in {"queued", "running"}:
                await self.db("put", "job", {"id": job["id"], "status": "cancelled", "message": "自动资料查证已关闭"})
                continue
            if job["status"] == "running":
                await self.db("put", "job", {"id": job["id"], "status": "queued", "message": "恢复中断的任务"})
        self.worker = asyncio.create_task(self._work(), name="photography-worker")
        self.scheduler = asyncio.create_task(self._schedule(), name="photography-scheduler")
        self.wake.set()

    async def refresh_character(self) -> None:
        self.character_context = await load_character_context(self.ctx, self.settings.target_stream_id)
        self.character_status = self.character_context["availability"]

    async def stop(self) -> None:
        self.stopping = True
        tasks = [task for task in (self.scheduler, self.worker) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def enqueue(self, kind: str, intent: str = "", seed_ids: Optional[List[str]] = None,
                      job_id: str = "", initial_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        async with self.queue_lock:
            if initial_data is not None and set(initial_data) - {"image_ids", "share_target_stream_id", "share_mode", "share_status"}:
                raise ValueError("任务初始化字段无效")
            existing = await self.db("get", "job", job_id) if job_id else None
            if existing:
                return existing
            queued = [j for j in await self.db("list", "job", limit=1000) if j["status"] in {"running", "queued"}]
            if kind in {"review", "rewrite_diaries", "visual_index"}:
                active = next((job for job in queued if job["kind"] == kind), None)
                if active:
                    return active
            if len(queued) >= 10:
                raise ValueError("已有 10 个待处理任务，请先完成或取消")
            job = {"id": job_id or uuid4().hex, "kind": kind, "intent": intent[:2000],
                   "seed_ids": seed_ids or [], "status": "queued", "created_at": stamp(),
                   "progress": 0, "cursor": 0, "message": "等待浏览", "errors": [],
                   "image_ids": [], "observation_ids": [], "preview_ids": [], "model_calls": 0,
                   "preview_count": 0, "detail_count": 0, "share_status": "not_requested"}
            job.update(initial_data or {})
            await self.db("put", "job", job)
            self.wake.set()
            return job

    async def import_seeds(self, content: str, filename: str) -> Dict[str, Any]:
        if len(content.encode("utf-8")) > 48000:
            raise ValueError("参考名单请保持在 48 KB 以内")
        seeds = parse_seeds(content)
        if not seeds or len(seeds) > 100:
            raise ValueError("请导入包含 1–100 位摄影师的文字名单")
        batch_id = sha256(content.encode()).hexdigest()
        await self.db("put", "seed_import", {"id": batch_id, "filename": Path(filename).name,
                                             "content": content, "created_at": stamp()})
        for seed in seeds:
            old = await self.db("get", "seed", seed["id"])
            if old is None:
                await self.db("put", "seed", {**seed, "import_id": batch_id, "created_at": stamp()})
            else:
                works = list(dict.fromkeys(old["works"] + seed["works"]))
                await self.db("put", "seed", {"id": seed["id"], "works": works,
                    "raw_line": old["raw_line"] if old["raw_line"] == seed["raw_line"] else old["raw_line"] + "\n" + seed["raw_line"],
                    "priority": "named_works" if works else "author_style", "import_id": batch_id})
        job = await self.enqueue("seed", "从朋友分享的摄影师与影集出发，看看哪些画面真正吸引我",
                                 [s["id"] for s in seeds], "seed-" + batch_id[:20])
        return {"ok": True, "count": len(seeds), "job_id": job["id"], "status": job["status"]}

    async def import_sites(self, content: str) -> Dict[str, Any]:
        if len(content.encode("utf-8")) > 48000:
            raise ValueError("网站列表过大")
        sources = parse_sites(content)
        if not sources or len(sources) > 200:
            raise ValueError("请输入 1–200 个有效的公开网站地址")
        for source in sources:
            if await self.db("get", "source", source["id"]) is None:
                await self.db("put", "source", {**source, "origin": "user", "created_at": stamp()})
        return {"ok": True, "count": len(sources)}

    async def cancel(self, job_id: str) -> Dict[str, Any]:
        job = await self.db("get", "job", job_id)
        if not job:
            raise ValueError("找不到该任务")
        if job["status"] in {"queued", "running"}:
            self.cancelled.add(job_id)
            await self.db("put", "job", {"id": job_id, "status": "cancelled", "message": "已取消；当前请求结束后停止"})
        return {"ok": True}

    async def retry(self, job_id: str) -> Dict[str, Any]:
        job = await self.db("get", "job", job_id)
        if not job or job["status"] not in {"failed", "cancelled", "partial"}:
            raise ValueError("只能继续失败、取消或未完成的任务")
        self.cancelled.discard(job_id)
        if job["kind"] == "fact_check":
            await self._prepare_fact_retry(job)
        # A failed detail observation still owns its full preview checkpoint.
        # Start another browse only after a completed round found no sendable image.
        if (job["kind"] == "share_cycle" and not job.get("share_planned_image_ids")
                and job.get("state") is None and job.get("share_error_code") == "no_local_images"):
            job["cursor"] = 0
            job["state"] = None
            job["share_planned_image_ids"] = None
            await self.db("put", "job", {"id": job["id"], "share_planned_image_ids": None})
        if job["status"] == "partial" and job["kind"] != "share_cycle":
            if job["kind"] == "seed":
                seeds = [await self.db("get", "seed", ident) for ident in job["seed_ids"]]
                job["seed_ids"] = [s["id"] for s in seeds if s["status"] != "observed"]
                if not job["seed_ids"]:
                    raise ValueError("所有作者已有样本；继续探索请新建漫游任务")
            job["cursor"] = 0
            job["state"] = None
        job["previous_errors"] = job["errors"]
        job["errors"] = []
        job.update(status="queued", message="从已保存进度继续", error="")
        await self.save_job_progress(job)
        if job["kind"] == "share_cycle":
            async with self.sharing_lock:
                current = await self.db("get", "job", job_id)
                if current["share_status"] == "failed":
                    await self.db("put", "job", {"id": job_id, "share_status": "queued",
                                                 "share_error": "", "share_error_code": ""})
        self.wake.set()
        return {"ok": True, "job_id": job_id}

    async def _schedule(self) -> None:
        await asyncio.sleep(30)
        while True:
            try:
                now = datetime.now(timezone(timedelta(hours=8)))
                seeds = await self.db("count", "seed")
                imports = [j for j in await self.db("list", "job", limit=1000)
                           if j["kind"] == "seed" and j["status"] in {"queued", "running"}]
                if self.settings.daily_enabled and seeds and not imports and now.hour >= self.settings.daily_hour:
                    await self.enqueue("daily", "按照当前审美寻找值得深入的新作品，也给陌生方向留下机会",
                                       job_id="daily-" + now.date().isoformat())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.ctx.logger.exception("摄影日程执行失败")
                await self.db("put", "system", {"id": "scheduler_error", "error": str(exc)[:2000]})
            await asyncio.sleep(60)

    async def _work(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            queued = sorted([j for j in await self.db("list", "job", limit=10000)
                             if j["status"] == "queued"], key=lambda j: j["created_at"])
            for job in queued:
                if job["id"] in self.cancelled:
                    continue
                self.active_id = job["id"]
                job.update(status="running", message="开始浏览")
                await self.save_job_progress(job)
                try:
                    async with asyncio.timeout(7200 if job["kind"] == "seed" else 1800):
                        await self._run(job)
                except asyncio.CancelledError:
                    if job["id"] in self.cancelled:
                        job.update(status="cancelled", message="任务已取消")
                    else:
                        job.update(status="queued", message="插件退出，已保存进度，加载后继续")
                    await self.save_job_progress(job)
                    raise
                except Exception as exc:
                    job.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:2000], message="任务失败，可查看原因后继续")
                    await self.save_job_progress(job)
                    if job["kind"] == "share_cycle":
                        async with self.sharing_lock:
                            current = await self.db("get", "job", job["id"])
                            if not current.get("share_planned_image_ids") and current.get("share_status") == "queued":
                                await self.db("put", "job", {"id": job["id"], "share_status": "failed",
                                    "share_error": "浏览或整理尚未完成，未开始分享：" + job["error"],
                                    "share_error_code": "exploration_failed"})
                    self.ctx.logger.exception("摄影任务失败: %s", job["id"])
                finally:
                    self.active_id = ""

    async def _model(self, purpose: str, data: Dict[str, Any], job: Dict[str, Any],
                     image: Optional[Dict[str, Any]] = None,
                     images: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if image is not None and images is not None:
            raise ValueError("单图与批量预览输入不能同时传入")
        material = {**data, "character_context": self.character_context}
        text = self.prompts["guide"] + "\n" + self.prompts[purpose] + "\n资料 JSON：\n" + json.dumps(material, ensure_ascii=False)
        prompt: Any = text
        visual = image is not None or images is not None
        if visual:
            batch = [image] if image is not None else images
            if not batch or len(batch) > 6:
                raise ValueError("每次视觉请求需要1至6张真实图片")
            prepared = await asyncio.gather(*[
                asyncio.to_thread(prepare_vision_image, item, max_side=768 if purpose == "preview" else 1280,
                                  max_bytes=250000 if purpose == "preview" else 750000) for item in batch])
            content = [{"type": "text", "text": text}]
            for index, item in enumerate(prepared):
                if images is not None:
                    content.append({"type": "text", "text": f"图片顺序 {index + 1}，编号 {batch[index].get('id', index + 1)}"})
                content.append({"type": "image_url", "image_url": {
                    "url": "data:" + item["mime_type"] + ";base64," + b64encode(item["data"]).decode()}})
            prompt = [{"role": "user", "content": content}]
        job["model_calls"] += 1
        await self.save_job_progress(job)
        result = await self.ctx.llm.generate(prompt=prompt, task_name=self.settings.vision_task if visual else self.settings.text_task,
                                             max_tokens=3000 if images else 2200, timeout_ms=180000)
        if not result.get("success"):
            raise ModelFailure("模型调用失败：" + str(result.get("error", "未返回成功结果"))[:1000])
        try:
            value = json_object(result["response"])
        except (ValueError, KeyError) as exc:
            raise ModelFailure("模型未返回有效 JSON，请检查模型或重试任务") from exc
        value["model"] = result.get("model", "")
        return value

    async def _run(self, job: Dict[str, Any]) -> None:
        if job["kind"] == "visual_index":
            await self._index_visual_memory(job)
            return
        if job["kind"] == "cleanup":
            result = await self.cleanup()
            job.update(status="done", progress=100, message=f"已清理 {result['evicted_count']} 张低关注图片", result=result)
            await self.save_job_progress(job)
            return
        if job["kind"] == "share" or (job["kind"] == "share_cycle" and job.get("share_planned_image_ids")):
            if self.share_dispatch is None:
                raise RuntimeError("分享处理器尚未连接")
            await self.share_dispatch(job)
            return
        await self.refresh_character()
        if job["kind"] == "fact_check":
            job.update(status="cancelled", message="自动资料查证已关闭")
            await self.save_job_progress(job)
            return
        if job["kind"] == "rewrite_diaries":
            await self._rewrite_diaries(job)
            return
        if job["kind"] == "feedback":
            await self._feedback(job)
            return
        if job["kind"] == "review":
            previous = next((item for item in await self.db("list", "job", limit=100)
                             if item["kind"] in {"seed", "daily", "manual", "share_cycle"} and item.get("observation_ids")), None)
            if previous is None:
                raise ValueError("还没有实际看图的记录，请先完成一次摄影探索")
            job.update(observation_ids=previous["observation_ids"], reviewed_job_id=previous["id"])
            await self._reflect(job, publish_diary=False)
            job.update(status="done", progress=100, finished_at=stamp(),
                       message="已整理最近的摄影感受")
            await self.save_job_progress(job)
            return
        await self._begin_visual_memory()
        units = job["seed_ids"] if job["kind"] == "seed" else [""]
        for index in range(job["cursor"], len(units)):
            if job["id"] in self.cancelled:
                job.update(status="cancelled", message="任务已停止")
                await self.save_job_progress(job)
                return
            seed = await self.db("get", "seed", units[index]) if units[index] else None
            await self._explore(job, seed)
            if job["id"] in self.cancelled:
                job.update(status="cancelled", message="任务已停止")
                await self.save_job_progress(job)
                return
            if seed:
                evidence = [await self.db("get", "observation", ident) for ident in job["observation_ids"]]
                viewed = [o for o in evidence if o.get("seed_id") == seed["id"]]
                matched = [o for o in viewed if o.get("matches_seed") is True]
                work_items = seed_work_items(seed)
                matched_ids = {ident for observation in matched for ident in observation.get("matched_work_ids", [])}
                seen_works = list(dict.fromkeys(seed.get("observed_works", []) +
                                    [item["title"] for item in work_items if item["id"] in matched_ids] +
                                    [o["matched_work"] for o in matched if o.get("matched_work") in seed["works"]]))
                await self.db("put", "seed", {"id": seed["id"], "status": "observed" if viewed else "pending",
                                               "observed_works": seen_works, "last_job_id": job["id"],
                                               "viewed_count": len(viewed), "note": ""})
            job["cursor"] = index + 1
            job["state"] = None
            job["progress"] = int(90 * (index + 1) / len(units))
            await self.save_job_progress(job)
        await self._reflect(job)
        seeds = [await self.db("get", "seed", ident) for ident in job["seed_ids"]]
        unresolved = [s["author"] for s in seeds if s["status"] != "observed"]
        results = list(job.get("browse_results", {}).values())
        partial = not job["observation_ids"] or bool(unresolved) or any(not result["target_met"] for result in results)
        job["preview_count"] = len(job.get("preview_ids", []))
        job["detail_count"] = len(job["observation_ids"])
        job["stop_reason"] = "；".join(dict.fromkeys(result["stop_explanation"] for result in results))
        job["unresolved_authors"] = unresolved
        exploration_status = "partial" if partial else "done"
        job.update(status="running" if job["kind"] == "share_cycle" else exploration_status,
                   exploration_status=exploration_status, progress=95 if job["kind"] == "share_cycle" else 100,
                   message=f"本轮结束：真实预览 {job['preview_count']} 张，精看 {job['detail_count']} 张，收藏 {len(job['image_ids'])} 张")
        if job["kind"] != "share_cycle":
            job["finished_at"] = stamp()
        await self.save_job_progress(job)
        await self.cleanup()
        await self.refresh_seed_experience()
        if job["kind"] == "share_cycle":
            if self.share_dispatch is None:
                raise RuntimeError("分享处理器尚未连接")
            await self.share_dispatch(job)
            return
        if job["kind"] != "seed" and job["image_ids"] and self.settings.share_enabled and self.settings.target_stream_id:
            await self.db("put", "job", {"id": job["id"], "share_status": "queued", "share_error": ""})
            try:
                receipt = await self.ctx.maisaka.proactive.trigger(
                    self.settings.target_stream_id,
                    intent="我刚看了一些摄影作品，想和朋友分享有意思的图。请调用 photography_share，用这次收藏的图片编号选图发送，再自然聊几句感受；不要整篇转发日记。此前的观看意图：" + job["intent"],
                    reason="摄影漫游完成", metadata={"photography_job_id": job["id"], "image_ids": job["image_ids"]})
            except Exception as exc:
                await self.db("put", "job", {"id": job["id"], "share_status": "failed", "share_error": str(exc)[:1500]})
                raise
            result = {"id": job["id"], "proactive_receipt": receipt}
            if not isinstance(receipt, dict) or not receipt.get("success"):
                result.update(share_status="failed", share_error="主体未接受摄影分享请求：" + str(receipt)[:1000])
            await self.db("put", "job", result)

    @staticmethod
    def source_weight(source: Dict[str, Any]) -> float:
        n = min(int(source.get("observations", 0)), 30)
        bot = float(source.get("bot_score", 0.5))
        human = source.get("human_score")
        interest = max(bot, float(human)) if human is not None else bot
        return round((3 * 0.5 + n * interest) / (3 + n), 3)

    async def _source_observed(self, url: str, interest: float, observation_id: str,
                               directions: Optional[List[str]] = None) -> None:
        parsed = urlsplit(url)
        origin = f"{parsed.scheme}://{parsed.netloc}/"
        ident = sha256(origin.encode()).hexdigest()[:20]
        sources = await self.db("list", "source", limit=1000)
        same = next((s for s in sources if urlsplit(s["url"]).netloc == parsed.netloc), None)
        source = same or {"id": ident, "url": origin, "name": parsed.netloc, "origin": "discovered", "observations": 0, "bot_score": 0.5}
        evidence = source.get("evidence", [])
        if observation_id in evidence:
            return
        n = int(source.get("observations", 0))
        source.update(observations=n + 1, bot_score=(source.get("bot_score", 0.5) * n + interest) / (n + 1),
                      evidence=(evidence + [observation_id])[-500:], last_viewed_at=stamp())
        await self.db("put", "source", source)
        for direction in (directions or [])[:8]:
            if not isinstance(direction, str) or not direction.strip():
                continue
            direction = direction.strip()[:100]
            direction_id = sha256((origin + "\0" + direction).encode()).hexdigest()[:24]
            record = await self.db("get", "source_direction", direction_id)
            if record is None:
                record = {"id": direction_id, "source_id": source["id"], "url": source["url"],
                          "direction": direction, "observations": 0, "bot_score": 0.5, "evidence": []}
            if observation_id not in record["evidence"]:
                n = record["observations"]
                record.update(observations=n + 1, bot_score=(record["bot_score"] * n + interest) / (n + 1),
                              evidence=(record["evidence"] + [observation_id])[-500:])
                await self.db("put", "source_direction", record)

    async def _reflect(self, job: Dict[str, Any], publish_diary: bool = True) -> None:
        observations = [await self.db("get", "observation", ident) for ident in job["observation_ids"][-30:]]
        current = await self.db("get", "profile", "current")
        recent_diaries = await self.db("list", "diary", limit=3)
        reflection = await self._model("reflection", {"observations": [
            {key: observation.get(key) for key in ("author", "work", "observations", "feeling", "reason", "directions", "source_url", "visual_comparisons")}
            for observation in observations], "previous_profile": current,
                                                     "review_mode": not publish_diary,
                                                     "actually_viewed_images": len(observations) if publish_diary else 0,
                                                     "past_observation_count": len(observations) if not publish_diary else 0,
                                                     "actually_previewed_images": len(job.get("preview_ids", [])),
                                                     "recent_diaries": [{"title": diary["title"], "content": diary["content"][:2000]} for diary in recent_diaries],
                                                     "seed_preferences": await self.db("list", "seed", limit=100),
                                                     "recent_feedback": [item for item in await self.db("list", "feedback", limit=50)
                                                        if item.get("status") == "evaluated" and item.get("scope") != "unrelated"][:15]}, job)
        if not isinstance(reflection.get("content"), str) or not isinstance(reflection.get("summary"), str):
            raise ValueError("日记输出缺少正文或审美总结")
        reflection.pop("fact_checks", None)
        if publish_diary:
            await self.db("put", "diary", {**reflection, "id": job["id"], "created_at": stamp(), "job_id": job["id"],
                                            "preview_count": len(job.get("preview_ids", [])), "detail_count": len(observations),
                                            "image_ids": job["image_ids"], "observation_ids": job["observation_ids"]})
        if observations:
            profile = {"id": "current", "summary": reflection["summary"], "directions": reflection.get("directions", []),
                       "questions": reflection.get("questions", []), "evidence": job["observation_ids"],
                       "version": int(current["version"]) + 1 if current else 1, "created_at": stamp(),
                       "note": "从真实观看和相处中慢慢形成；朋友分享的起点单独保留"}
            await self.db("put", "profile_history", {**profile, "id": str(profile["version"])})
            await self.db("put", "profile", profile)

    async def cleanup(self) -> Dict[str, Any]:
        async with self.collection_lock:
            result = await asyncio.to_thread(self.store.cleanup, self.settings.max_images,
                                             self.settings.max_mb * 1024 ** 2, self.settings.idle_days)
        return {"ok": True, **result}
