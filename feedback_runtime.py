"""Capture and replay scoped discussion, then evaluate it in the plugin's worker."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlsplit

import asyncio

from .account_identity import is_self_message, self_user_ids
from .feedback import _date, _own_text, apply_feedback, associate_message, has_photography_topic, interpret_feedback
from .reader import sanitized_failure


def host_image_descriptions(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep public Host descriptions as reported evidence, without image bytes."""
    return [{"hash": component.get("hash", ""),
             "description": component["data"][:700] if isinstance(component.get("data"), str) else "",
             "source": "host_message_image_description", "observed_by_plugin": False}
            for component in message.get("raw_message", []) if component.get("type") == "image"][:4]


class FeedbackRuntimeMixin:
    def _feedback_self_ids(self) -> Set[str]:
        return self_user_ids(self.character_context.get("bot", {}), self.character_context.get("scope", {}))

    def _feedback_is_self(self, record: Dict[str, Any]) -> bool:
        message = {"platform": record.get("platform"), "account_id": record.get("account_id"),
                   "message_info": {"user_info": {"user_id": record["speaker_id"]}}}
        return is_self_message(message, self.character_context.get("bot", {}), self.character_context.get("scope", {}))

    async def _feedback_material(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        shares, images, jobs = await asyncio.gather(self.db("list", "share", limit=10000), self.db("list", "image", limit=10000),
                                                  self.db("list", "job", limit=10000))
        aliases: Dict[str, List[str]] = {}
        for share in shares:
            aliases.setdefault(share.get("canonical_image_id", share["image_id"]), []).append(share["image_id"])
        images = [{**image, "reference_ids": list(dict.fromkeys([image["id"], *aliases.get(image["id"], [])]))} for image in images]
        shares = [{**share, "image_id": share.get("canonical_image_id", share["image_id"])} for share in shares]
        return shares, images, jobs

    async def _capture_feedback_record(self, message: Dict[str, Any],
                                       material: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]],
                                       context: List[Dict[str, Any]]) -> Dict[str, Any]:
        shares, images, jobs = material
        target = self.settings.target_stream_id
        previous_records = await self.db("list", "feedback", limit=40)
        recent = any(record.get("stream_id") == target and record.get("status") != "ignored"
                     and record.get("association") != "conversation_followup"
                     and 0 < float(message["timestamp"]) - _date(record["created_at"]).timestamp() <= 120
                     for record in previous_records)
        if not recent:
            known = {record["message_id"]: record for record in previous_records}
            for previous in context:
                if (previous.get("session_id") != target or previous.get("is_command") or previous.get("is_notify")
                        or previous["message_id"] == message["message_id"]
                        or not 0 <= float(message["timestamp"]) - float(previous["timestamp"]) <= 120):
                    continue
                recorded = known.get(previous["message_id"])
                if recorded is not None and recorded.get("status") == "ignored" and recorded.get("ignore_reason") != "self_message":
                    continue
                if has_photography_topic(previous, images) or associate_message(previous, shares, images, target, jobs):
                    recent = True
                    break
        linked = associate_message(message, shares, images, target, jobs, self._feedback_self_ids(), recent)
        if linked is None:
            return {"captured": False}
        existing = await self.db("get", "feedback", linked["id"])
        if existing is not None:
            return {"captured": False, "id": existing["id"], "pending": existing["status"] not in {"evaluated", "ignored"}}
        relevant_context = []
        for previous in context[-16:]:
            if previous.get("session_id") != target or previous.get("is_command") or previous.get("is_notify"):
                continue
            image_notes = host_image_descriptions(previous)
            # Bot remarks can clarify a visual reference, but never become a friend's preference.
            if is_self_message(previous, self.character_context.get("bot", {}), self.character_context.get("scope", {})):
                text = _own_text(previous)
                if text or image_notes:
                    relevant_context.append({"message_id": previous["message_id"], "is_self": True,
                        "speaker_id": previous["message_info"]["user_info"]["user_id"], "text": text[:2000],
                        "host_images": image_notes,
                        "created_at": datetime.fromtimestamp(float(previous["timestamp"]), timezone.utc).isoformat()})
                continue
            candidate = associate_message(previous, shares, images, target, jobs)
            if candidate is not None:
                relevant_context.append({**{key: candidate[key] for key in ("message_id", "speaker_id", "text", "created_at")},
                                         "is_self": False, "host_images": image_notes})
            elif image_notes:
                relevant_context.append({"message_id": previous["message_id"], "is_self": False,
                    "speaker_id": previous["message_info"]["user_info"]["user_id"], "text": _own_text(previous)[:2000],
                    "host_images": image_notes,
                    "created_at": datetime.fromtimestamp(float(previous["timestamp"]), timezone.utc).isoformat()})
        await self.db("put", "feedback", {**linked, "status": "pending", "host_images": host_image_descriptions(message),
                                          "context_messages": relevant_context[-8:]})
        return {"captured": True, "id": linked["id"], "pending": True}

    async def capture_feedback(self, message: Dict[str, Any]) -> Dict[str, Any]:
        target = self.settings.target_stream_id
        if message.get("session_id") != target:
            return {"captured": False}
        if is_self_message(message, self.character_context.get("bot", {}), self.character_context.get("scope", {})):
            return {"captured": False}
        messages = await self.ctx.message.get_recent(target, limit=24)
        if not isinstance(messages, list):
            raise ValueError("Host 未返回本群最近消息列表，摄影讨论上下文未保存")
        timestamp = float(message["timestamp"])
        context = sorted([previous for previous in messages if previous.get("session_id") == target
                          and previous["message_id"] != message["message_id"]
                          and float(previous["timestamp"]) <= timestamp], key=lambda previous: float(previous["timestamp"]))[-24:]
        async with self.feedback_capture_lock:
            if self.settings.target_stream_id != target:
                raise ValueError("固定群已改变，本条摄影讨论尚未保存")
            result = await self._capture_feedback_record(message, await self._feedback_material(), context)
            if result.get("pending"):
                result["job_id"] = (await self._queue_feedback([result["id"]]))["id"]
            return result

    async def _queue_feedback(self, ids: List[str]) -> Dict[str, Any]:
        for job in await self.db("list", "job", limit=10000):
            if job["kind"] == "feedback" and job["status"] == "queued" and job["seed_ids"]:
                first = await self.db("get", "feedback", job["seed_ids"][0])
                if first is not None and first.get("stream_id") == self.settings.target_stream_id:
                    return job
        return await self.enqueue("feedback", "整理本群朋友的摄影讨论", seed_ids=ids)

    async def backfill_feedback(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        async with self.feedback_capture_lock:
            return await self._backfill_feedback_locked(messages)

    async def _backfill_feedback_locked(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        material = await self._feedback_material()
        ordered = sorted([message for message in messages if message.get("session_id") == self.settings.target_stream_id],
                         key=lambda message: float(message["timestamp"]))
        captured, ids = 0, []
        for index, message in enumerate(ordered):
            result = await self._capture_feedback_record(message, material, ordered[max(0, index - 16):index])
            captured += int(result["captured"])
            if result.get("pending"):
                ids.append(result["id"])
        ids = list(dict.fromkeys(ids))
        job_id = ""
        if ids:
            job = await self._queue_feedback(ids)
            job_id = job["id"]
        return {"ok": True, "captured": captured, "pending": len(ids), "job_id": job_id, "scanned": len(ordered)}

    async def backfill_recent_feedback(self) -> Dict[str, Any]:
        target = self.settings.target_stream_id
        if not target:
            raise ValueError("请先选择摄影分享的固定群")
        messages = await self.ctx.message.get_recent(target, limit=500)
        if not isinstance(messages, list):
            raise ValueError("Host 未返回本群最近消息列表")
        now = datetime.now(timezone.utc).timestamp()
        since = now - 86400
        scoped = [message for message in messages if message.get("session_id") == target
                  and since <= float(message["timestamp"]) <= now]
        await self.refresh_character()
        if self.settings.target_stream_id != target:
            raise ValueError("固定群已改变，请重新整理讨论")
        result = await self.backfill_feedback(scoped)
        oldest = min((float(message["timestamp"]) for message in messages), default=now)
        result["limited_history"] = len(messages) >= 500 and oldest > since
        result["message"] = "已将有关摄影的讨论排队整理" if result["pending"] else "本次范围内没有尚待整理的摄影讨论"
        if result["limited_history"]:
            result["message"] += "；公开接口只提供最近24小时最多500条，本次未覆盖更早记录"
        return result

    async def _feedback(self, job: Dict[str, Any]) -> None:
        async with self.feedback_capture_lock:
            first = await self.db("get", "feedback", job["seed_ids"][0])
            pending = [record for record in await self.db("list", "feedback", limit=10000)
                       if first.get("stream_id") and record.get("stream_id") == first["stream_id"]
                       and record["status"] not in {"evaluated", "ignored"}]
            pending.sort(key=lambda record: _date(record["created_at"]))
            ids = list(dict.fromkeys(job["seed_ids"] + [record["id"] for record in pending]))
        existing = [await self.db("get", "feedback", record_id) for record_id in ids]
        job["feedback_has_insight"] = any(record["status"] == "evaluated" and record.get("is_photography_related")
            and record.get("insight") and not self._feedback_is_self(record) for record in existing)
        has_insight = False
        for index, record_id in enumerate(ids):
            if job["id"] in self.cancelled:
                job.update(status="cancelled", message="讨论整理已停止")
                await self.save_job_progress(job)
                return
            record = await self.db("get", "feedback", record_id)
            if self._feedback_is_self(record):
                if record.get("ignore_reason") != "self_message":
                    await self.db("put", "feedback", {"id": record_id, "status": "ignored", "scope": "unrelated",
                        "ignore_reason": "self_message", "self_message_trace": record,
                        "is_photography_related": False, "is_evaluation": False, "human_interest": None, "insight": ""})
                continue
            if record["status"] in {"evaluated", "ignored"}:
                continue
            record["context_messages"] = [{**item, "is_self": self._feedback_is_self(item)}
                                           for item in record.get("context_messages", [])]
            candidates = record.get("candidate_image_ids", [record["image_id"]] if record.get("image_id") else [])
            images = [await self.db("get", "image", image_id) for image_id in candidates]
            notes = [{key: image.get(key) for key in ("id", "author", "work", "title", "observations", "feeling", "directions", "source_url")}
                     for image in images if image is not None]
            try:
                judgement = await self._model("feedback", {"message": record, "candidate_images": notes,
                                              "discussion_context": record.get("context_messages", [])}, job)
                evaluated = interpret_feedback(record, judgement)
                evaluated["error"] = ""
            except Exception as exc:
                await self.db("put", "feedback", {"id": record_id, "status": "failed", "error": sanitized_failure(exc)})
                raise
            if evaluated["scope"] == "image" and evaluated["is_evaluation"] and evaluated["human_interest"] is not None:
                # Re-read after the model call so a concurrent WebUI pin is never overwritten.
                async with self.collection_lock:
                    image = await self.db("get", "image", evaluated["image_id"])
                    if image is None:
                        raise ValueError("反馈指向的图片记录已不存在")
                    updated = apply_feedback(image, evaluated)
                    await self.db("put", "image", updated)
                await self._feedback_source_preferences(updated, evaluated)
            await self.db("put", "feedback", evaluated)
            has_insight = has_insight or evaluated["is_photography_related"] and bool(evaluated["insight"])
            job["feedback_has_insight"] = job.get("feedback_has_insight", False) or has_insight
            job.update(progress=int(95 * (index + 1) / len(ids)), message="正在整理朋友对摄影的讨论")
            await self.save_job_progress(job)
        if job.get("feedback_has_insight"):
            review = await self.enqueue("review", "结合朋友最新的摄影讨论整理自己的感受，分别保留朋友意见与自己的判断")
            job["review_job_id"] = review["id"]
        job.update(status="done", progress=100, message="已分别保留朋友意见与作品/整批/一般讨论的依据")
        await self.save_job_progress(job)

    async def _feedback_source_preferences(self, image: Dict[str, Any], record: Dict[str, Any]) -> None:
        domain = urlsplit(image.get("source_url", "")).netloc
        for kind in ("source", "source_direction"):
            for source in await self.db("list", kind, limit=1000):
                if urlsplit(source["url"]).netloc != domain:
                    continue
                if kind == "source_direction" and source["direction"] not in image.get("directions", []):
                    continue
                evidence = source.get("feedback_ids", [])
                if record["id"] in evidence:
                    continue
                prior = source.get("human_score")
                score = 0.8 * (0.5 if prior is None else prior) + 0.2 * image["human_interest"]
                await self.db("put", kind, {"id": source["id"], "human_score": score,
                                          "feedback_ids": (evidence + [record["id"]])[-1000:]})
