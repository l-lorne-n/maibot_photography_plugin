"""Personal notebook revisions and viewing-based seed progress."""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List

import json


class DiaryMixin:
    async def refresh_seed_experience(self) -> None:
        observations = await self.db("list", "observation", limit=10000)
        for seed in await self.db("list", "seed", limit=100):
            name = seed["author"].strip().casefold()
            viewed = [item for item in observations if item.get("seed_id") == seed["id"]
                      or (isinstance(item.get("author"), str) and item["author"].strip().casefold() == name)]
            changes = {"viewed_count": len(viewed), "status": "observed" if viewed else "pending", "note": ""}
            if any(seed.get(key) != value for key, value in changes.items()):
                await self.db("put", "seed", {"id": seed["id"], **changes})

    async def _rewrite_diaries(self, job: Dict[str, Any]) -> None:
        """Revise prose in place while retaining each entry's date and observations."""
        if "diary_ids" not in job:
            diaries = sorted(await self.db("list", "diary", limit=10000), key=lambda diary: diary["created_at"])
            job["diary_ids"] = [diary["id"] for diary in diaries]
            job["rewritten_diary_ids"] = []
            await self.save_job_progress(job)
        current_profile = await self.db("get", "profile", "current")
        for diary_id in job["diary_ids"]:
            if job["id"] in self.cancelled:
                job.update(status="cancelled", message="日记整理已停止")
                await self.save_job_progress(job)
                return
            if diary_id in job["rewritten_diary_ids"]:
                continue
            diary = await self.db("get", "diary", diary_id)
            if diary is None:
                raise ValueError("找不到要整理的日记：" + diary_id)
            if diary.get("rewrite_job_id") != job["id"]:
                observations: List[Dict[str, Any]] = []
                for ident in diary.get("observation_ids", [])[:60]:
                    record = await self.db("get", "observation", ident)
                    if record is not None:
                        observations.append({key: record.get(key) for key in (
                            "id", "author", "work", "title", "observations", "feeling", "reason", "source_url")})
                original = json.dumps(diary, ensure_ascii=False, sort_keys=True)
                revision_id = diary_id + "-" + sha256(original.encode()).hexdigest()[:16]
                if await self.db("get", "diary_revision", revision_id) is None:
                    await self.db("put", "diary_revision", {
                        "id": revision_id, "diary_id": diary_id, "original": diary, "rewrite_job_id": job["id"]})
                rewritten = await self._model("diary_rewrite", {
                    "diary": diary, "observations": observations, "current_profile": current_profile,
                }, job)
                if any(not isinstance(rewritten.get(key), str) or not rewritten[key].strip() for key in ("title", "body")):
                    raise ValueError("日记整理结果缺少标题或正文")
                questions = rewritten.get("questions", [])
                if not isinstance(questions, list) or any(not isinstance(item, str) for item in questions):
                    raise ValueError("日记的后续想法需要是文字列表")
                if "用户" in "\n".join([rewritten["title"], rewritten["body"], *questions]):
                    raise ValueError("日记仍使用了用户称呼，请重新整理")
                await self.db("put", "diary", {
                    "id": diary_id, "title": rewritten["title"].strip(), "content": rewritten["body"].strip(),
                    "questions": questions, "rewrite_job_id": job["id"],
                    "rewritten_at": datetime.now(timezone.utc).isoformat(),
                })
            job["rewritten_diary_ids"].append(diary_id)
            job.update(progress=int(100 * len(job["rewritten_diary_ids"]) / len(job["diary_ids"])),
                       message=f"已整理 {len(job['rewritten_diary_ids'])} 篇日记")
            await self.save_job_progress(job)
        job.update(status="done", progress=100, finished_at=datetime.now(timezone.utc).isoformat(),
                   message=f"已按现在的人格整理 {len(job['rewritten_diary_ids'])} 篇摄影日记")
        await self.save_job_progress(job)
