"""Host-personality commentary for completed one-click picture deliveries."""

from typing import Any, Dict, List

import asyncio
import json

from .reader import sanitized_failure
from .workflow import stamp


class ShareCommentaryMixin:
    async def _save_commentary(self, job: Dict[str, Any], status: str, error: str = "", **fields: Any) -> None:
        patch = {"id": job["id"], "share_commentary_status": status, "share_commentary_error": error,
                 "share_commentary_attempted_at": stamp(), **fields}
        await self.engine.db("put", "job", patch)
        job.update(patch)

    async def _check_commentary_target(self, job: Dict[str, Any]) -> None:
        target = job["share_target_stream_id"]
        if job["id"] in self.engine.cancelled:
            raise ValueError("任务已取消，尚未发送摄影感想")
        if not self.engine.settings.share_enabled:
            raise ValueError("摄影分享已关闭，尚未发送摄影感想")
        if target != self.engine.settings.target_stream_id:
            raise ValueError("固定群设置已改变，尚未发送摄影感想")
        if target not in {group["id"] for group in await self._groups()}:
            raise ValueError("原固定群不再是 Host 中可用的真实群聊")

    async def _commentary_reference(self, job: Dict[str, Any]) -> str:
        """Use receipt-linked observations and freshly scoped chat, never persist this prompt."""
        e = self.engine
        target = job["share_target_stream_id"]
        context = e.character_context
        scope = context["scope"]
        if scope.get("kind") != "group" or scope.get("stream_id") != target:
            raise ValueError("未能读取原固定群的会话范围，尚未生成摄影感想")
        receipts = [receipt for receipt in await self._canonical_shares()
                    if receipt.get("job_id") == job["id"] and receipt.get("stream_id") == target
                    and receipt.get("message_id") and receipt.get("sent") is not False]
        receipts_by_image = {receipt["image_id"]: receipt for receipt in receipts}
        photographs: List[Dict[str, Any]] = []
        current_observations = [await e.db("get", "observation", ident) for ident in job["observation_ids"]]
        def resolve_references() -> Dict[str, Any]:
            return {"planned": e.store.resolve_image_ids(job["share_planned_image_ids"]),
                    "observations": {e.store.resolve_image_id(item["image_id"]): item
                                     for item in current_observations if item is not None and item.get("image_id")}}
        resolved = await asyncio.to_thread(resolve_references)
        fields = ("author", "work", "title", "observations", "feeling", "reason", "directions",
                  "source_url", "image_url", "identity_verified", "identity_evidence")
        for image_id in resolved["planned"]:
            receipt = receipts_by_image.get(image_id)
            if receipt is None:
                raise ValueError("本次图片尚未全部取得成功回执，不能生成已分享感想")
            image = await e.db("get", "image", image_id)
            if image is None:
                raise ValueError("已分享图片缺少本地观察记录")
            observation = resolved["observations"].get(image_id)
            if observation is None and image.get("observation_id"):
                observation = await e.db("get", "observation", image["observation_id"])
            material = {**image, **(observation or {})}
            photographs.append({"image_id": image_id, "message_id": receipt["message_id"],
                                "sent_at": receipt["created_at"], **{key: material[key] for key in fields if key in material}})
        if not photographs:
            raise ValueError("没有本任务实际成功发出的摄影图片")
        reference = {"job_id": job["id"], "kind": job["kind"], "intent": job["intent"],
                     "photographs": photographs, "scope": scope,
                     "recent_conversation": [item for item in context["recent_conversation"] if item.get("stream_id") == target],
                     "friends": [item for item in context["friends"] if item.get("stream_id") == target]}
        return json.dumps(reference, ensure_ascii=False)

    async def _share_commentary(self, job: Dict[str, Any]) -> None:
        """Generate once, then send with a real receipt while the caller holds sharing_lock."""
        status = job.get("share_commentary_status", "")
        if status == "sent":
            return
        if status in {"sending", "uncertain"}:
            await self._save_commentary(job, "uncertain", job.get("share_commentary_error") or
                                        "上次感想发送没有确定回执；为避免重复发言，未自动重送")
            return
        e = self.engine
        target = job["share_target_stream_id"]
        text = job.get("share_commentary_text", "")
        if not text:
            await self._save_commentary(job, "generating", share_commentary_message_id="")
            try:
                await self._check_commentary_target(job)
                await e.refresh_character()
                await self._check_commentary_target(job)
                reference = await self._commentary_reference(job)
                result = await self.ctx.maisaka.reply.generate(
                    target, reference, format_instruction=e.prompts["share_commentary"], timeout_ms=180000)
                if not isinstance(result, dict) or result.get("success") is not True:
                    raise RuntimeError("主体生成摄影感想失败：" + sanitized_failure(result))
                text = result.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("主体返回了空白或无效的摄影感想")
                text = text.strip()
                await self._save_commentary(job, "ready", share_commentary_text=text)
            except asyncio.CancelledError:
                await self._save_commentary(job, "failed", "感想生成被中断，尚未发送文字")
                raise
            except Exception as exc:
                await self._save_commentary(job, "failed", sanitized_failure(exc))
                return
        try:
            await self._check_commentary_target(job)
        except Exception as exc:
            await self._save_commentary(job, "failed", sanitized_failure(exc))
            return
        await self._save_commentary(job, "sending")
        try:
            result = await self.ctx.send.text(text, target, return_details=True, sync_to_maisaka_history=True,
                                              maisaka_source_kind="photography_share_commentary")
        except asyncio.CancelledError:
            await self._save_commentary(job, "uncertain", "感想发送期间被中断，是否发出尚不确定；未自动重送")
            raise
        except Exception as exc:
            await self._save_commentary(job, "uncertain", "感想发送结果不确定，未自动重送：" + sanitized_failure(exc))
            return
        if isinstance(result, dict) and result.get("sent") is True and isinstance(result.get("message_id"), str) and result["message_id"].strip():
            await self._save_commentary(job, "sent", share_commentary_message_id=result["message_id"].strip())
        elif isinstance(result, dict) and result.get("sent") is False:
            await self._save_commentary(job, "failed", "Host 确认感想文字未发送：" + sanitized_failure(result))
        else:
            await self._save_commentary(job, "uncertain", "感想发送未取得有效消息回执，是否发出尚不确定；未自动重送")
