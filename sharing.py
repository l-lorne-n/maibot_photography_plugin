"""Explicit dashboard sharing jobs, selected from actual observations and receipts."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncio

from .companion import ShareCommentaryMixin
from .workflow import stamp


LOCAL_TIMEZONE = timezone(timedelta(hours=8))


def has_receipt_today(receipts: List[Dict[str, Any]], target: str,
                      now: Optional[datetime] = None) -> bool:
    """A queued intent or a failed attempt never counts as a successful share."""
    today = (now or datetime.now(timezone.utc)).astimezone(LOCAL_TIMEZONE).date()
    for receipt in receipts:
        if receipt.get("stream_id") != target or not receipt.get("message_id") or receipt.get("sent") is False:
            continue
        created = datetime.fromisoformat(receipt["created_at"].replace("Z", "+00:00"))
        if created.tzinfo is None:
            raise ValueError("分享回执时间缺少时区")
        if created.astimezone(LOCAL_TIMEZONE).date() == today:
            return True
    return False


def rank_images(images: List[Dict[str, Any]], receipts: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    sent_ids = {receipt["image_id"] for receipt in receipts if receipt.get("stream_id") == target
                and receipt.get("message_id") and receipt.get("sent") is not False}
    return sorted(images, key=lambda item: (item["id"] not in sent_ids, item.get("bot_interest", 0.5)), reverse=True)


class ShareJobsMixin(ShareCommentaryMixin):
    async def _canonical_shares(self) -> List[Dict[str, Any]]:
        receipts = await self.engine.db("list", "share", limit=10000)
        def resolve() -> List[Dict[str, Any]]:
            return [{**item, "image_id": self.engine.store.resolve_image_id(item["image_id"])} for item in receipts]
        return await asyncio.to_thread(resolve)

    async def _enqueue_share_now(self) -> Dict[str, Any]:
        e = self.engine
        async with self._share_request_lock:
            if not e.settings.share_enabled:
                raise ValueError("请先开启摄影分享，并选择固定群")
            target = e.settings.target_stream_id
            groups = await self._groups()
            if not target or target not in {group["id"] for group in groups}:
                raise ValueError("请选择 Host 中已经存在的真实群聊")
            jobs = await e.db("list", "job", limit=10000)
            active = next((job for job in jobs if job["kind"] in {"share", "share_cycle"}
                           and job["status"] in {"queued", "running"}
                           and job.get("share_target_stream_id") == target), None)
            if active:
                return {"ok": True, "job_id": active["id"], "mode": "in_progress", "message": "这次分享正在进行，可以在任务进度里查看。"}
            receipts = await self._canonical_shares()
            images = [image for image in await e.db("list", "image", limit=10000)
                      if image["local_status"] == "available"]
            explore = has_receipt_today(receipts, target) or not images
            mode = "explore_then_share" if explore else "share_existing"
            ids = [] if explore else [image["id"] for image in rank_images(images, receipts, target)[:e.settings.share_count]]
            job = await e.enqueue("share_cycle" if explore else "share",
                                  "再看一些摄影作品，整理感受后分享给朋友" if explore else "挑选已有摄影作品分享给朋友",
                                  initial_data={"share_target_stream_id": target, "image_ids": ids,
                                                "share_mode": mode, "share_status": "queued"})
            return {"ok": True, "job_id": job["id"], "mode": mode,
                    "message": "已排队：看新作品、整理感受，再分享到固定群。" if explore else "已排队：从已有收藏里挑选作品，分享到固定群。"}

    async def _finish_share_job(self, job: Dict[str, Any], error: str = "", error_code: str = "") -> Dict[str, Any]:
        receipts = await self._canonical_shares()
        delivered_ids = {receipt["image_id"] for receipt in receipts if receipt.get("job_id") == job["id"]
                         and receipt.get("stream_id") == job["share_target_stream_id"]
                         and receipt.get("message_id") and receipt.get("sent") is not False}
        delivered = len(delivered_ids)
        planned_ids = await asyncio.to_thread(self.engine.store.resolve_image_ids, job.get("share_planned_image_ids", []))
        all_images_sent = bool(planned_ids) and set(planned_ids) <= delivered_ids
        status = "partial" if error and delivered else "failed" if error else "done"
        if error_code == "cancelled":
            status = "cancelled"
        share_status = "sent" if all_images_sent else "partial" if error and delivered else "failed" if error else "sent"
        message = f"已分享 {delivered} 张摄影作品" if not error else f"已分享 {delivered} 张；{error}"
        commentary_status = job.get("share_commentary_status", "")
        final_error, final_error_code = error, error_code
        if not error and commentary_status == "sent":
            message = f"已分享 {delivered} 张摄影作品，并发出了感想"
        elif not error and commentary_status:
            status = "partial"
            final_error = job.get("share_commentary_error") or "摄影感想尚未完成"
            final_error_code = "commentary_" + commentary_status
            message = f"已分享 {delivered} 张图片；感想文字未完成：{final_error}"
        patch = {"id": job["id"], "status": status, "share_status": share_status, "share_error": "" if all_images_sent else error,
                 "share_error_code": "" if all_images_sent else error_code, "share_delivered_count": delivered, "error": final_error,
                 "message": message, "progress": 100, "finished_at": stamp(), "share_attempted_at": stamp()}
        await self.engine.db("put", "job", patch)
        job.update(patch)
        return {"success": not final_error, "job_id": job["id"], "delivered_count": delivered,
                "error": final_error, "error_code": final_error_code}

    async def _dispatch_share_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Run the queued click once; retry only images without this job's receipts."""
        e = self.engine
        async with e.sharing_lock:
            persisted = await e.db("get", "job", job["id"])
            if persisted is None or persisted["kind"] not in {"share", "share_cycle"}:
                raise ValueError("只能派发已保存的一键分享任务")
            job.update(persisted)
            # Completed deliveries need no further sends; older picture-only jobs also stay unchanged.
            if job["status"] == "done" and ("share_commentary_status" not in job or job["share_commentary_status"] == "sent"):
                return {"success": True, "job_id": job["id"], "delivered_count": job.get("share_delivered_count", 0),
                        "error": "", "error_code": ""}
            target = job["share_target_stream_id"]
            if job["id"] in e.cancelled:
                return await self._finish_share_job(job, "分享任务已取消", "cancelled")
            if not e.settings.share_enabled:
                return await self._finish_share_job(job, "摄影分享已关闭", "sharing_disabled")
            if not target or target != e.settings.target_stream_id:
                return await self._finish_share_job(job, "固定群设置已改变，请重新点击分享", "target_changed")
            if target not in {group["id"] for group in await self._groups()}:
                return await self._finish_share_job(job, "固定群不再是 Host 中可用的真实群聊", "target_unavailable")
            receipts = await self._canonical_shares()
            confirmed = {receipt["image_id"] for receipt in receipts if receipt.get("job_id") == job["id"]
                         and receipt.get("stream_id") == target and receipt.get("message_id")
                         and receipt.get("sent") is not False}
            images = {image["id"]: image for image in await e.db("list", "image", limit=10000)}
            planned_ids = await asyncio.to_thread(e.store.resolve_image_ids, job.get("share_planned_image_ids") or [])
            if not planned_ids:
                resolved = await asyncio.to_thread(e.store.resolve_image_ids, job["image_ids"])
                candidates = [images[image_id] for image_id in resolved
                              if image_id in images and images[image_id]["local_status"] == "available"]
                planned_ids = [image["id"] for image in rank_images(candidates, receipts, target)[:e.settings.share_count]]
                job["share_planned_image_ids"] = planned_ids
                await e.db("put", "job", {"id": job["id"], "share_planned_image_ids": planned_ids})
            elif planned_ids != job.get("share_planned_image_ids"):
                job["share_planned_image_ids"] = planned_ids
                await e.db("put", "job", {"id": job["id"], "share_planned_image_ids": planned_ids})
            if not planned_ids:
                return await self._finish_share_job(job, "这次没有可发送的本地作品；已保留浏览和整理记录", "no_local_images")
            pending_ids = [image_id for image_id in planned_ids if image_id not in confirmed]
            if any(image_id not in images or images[image_id]["local_status"] != "available" for image_id in pending_ids):
                return await self._finish_share_job(job, "本次选中的部分图片已不在本地，请重新开始分享", "no_local_images")
            if not pending_ids:
                await self._share_commentary(job)
                return await self._finish_share_job(job)
            selected = [{**images[image_id], "share_job_id": job["id"]} for image_id in pending_ids]
            try:
                result = await self._send_selected(selected, target, request_job_id=job["id"])
            except Exception as exc:
                return await self._finish_share_job(job, f"{type(exc).__name__}: {exc}", "send_error")
            if result.get("success"):
                await self._share_commentary(job)
            return await self._finish_share_job(job, result.get("error", ""), result.get("error_code", ""))
