"""Connect historical image neighbours to actual visual observations and diaries."""

from typing import Any, Dict, List, Tuple

import asyncio

from .visual_memory import VisualMemory


class VisualContextMixin:
    async def _begin_visual_memory(self) -> None:
        self.visual_memory = VisualMemory(llm=self.ctx.llm, store=self.store,
                                          task_name=self.settings.image_embedding_task)
        await self.visual_memory.begin_run()
        existing = [image["id"] for image in await self.db("list", "image", limit=await self.db("count", "image"))
                    if image["local_status"] == "available"]
        self._visual_history_ids = await asyncio.to_thread(self.store.resolve_image_ids, existing)

    async def _historical_visual_context(self, actual: Dict[str, Any], preview: Dict[str, Any],
                                         job: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        identity = await asyncio.to_thread(self.store.identify_image, actual["data"], actual["mime_type"])
        vector = await self.visual_memory.embed_image(actual, work_id=identity["work_id"],
                                                     variant_id=identity["fingerprint"]["pixel_hash"], metadata=preview)
        historical_ids = set(await asyncio.to_thread(self.store.resolve_image_ids, self._visual_history_ids))
        available = [image["id"] for image in await self.db("list", "image", limit=await self.db("count", "image"))
                     if image["local_status"] == "available" and image["id"] in historical_ids]
        matches = await self.visual_memory.search(vector, limit=2, allowed_work_ids=available)
        diaries = await self.db("list", "diary", limit=await self.db("count", "diary"))
        diary_images = {diary["id"]: set(await asyncio.to_thread(self.store.resolve_image_ids, diary.get("image_ids", [])))
                        for diary in diaries}
        notes: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        for match in matches:
            previous = await self.db("get", "image", match["work_id"])
            if previous is None or previous["local_status"] != "available":
                continue
            data, mime_type = await asyncio.to_thread(self.store.image_bytes, previous["id"])
            related = [diary for diary in diaries if previous["id"] in diary_images[diary["id"]]]
            notes.append({"image_id": previous["id"], "similarity": match["similarity"],
                          "author": previous.get("author", ""), "work": previous.get("work", ""),
                          "observations": previous.get("observations", ""), "feeling": previous.get("feeling", ""),
                          "bot_interest": previous.get("bot_interest"), "human_interest": previous.get("human_interest"),
                          "diaries": [{"id": diary["id"], "title": diary["title"], "content": diary["content"][:1200]}
                                      for diary in related[:2]]})
            images.append({"data": data, "mime_type": mime_type, "width": previous["width"],
                           "height": previous["height"], "id": previous["id"]})
            if len(images) == 2:
                break
        await self.db("put", "visual_link", {"id": preview["id"], "job_id": job["id"],
            "work_id": identity["work_id"], "space_id": vector["space_id"],
            "neighbours": [{"work_id": item["image_id"], "similarity": item["similarity"]} for item in notes]})
        return notes, images

    async def _index_visual_memory(self, job: Dict[str, Any]) -> None:
        await self._begin_visual_memory()
        images = [image for image in await self.db("list", "image", limit=await self.db("count", "image"))
                  if image["local_status"] == "available"]
        processed = 0
        for image in images:
            if job["id"] in self.cancelled:
                job.update(status="cancelled", message="已停止整理视觉记忆")
                await self.save_job_progress(job)
                return
            data, mime_type = await asyncio.to_thread(self.store.image_bytes, image["id"])
            actual = {"data": data, "mime_type": mime_type, "width": image["width"], "height": image["height"]}
            identity = await asyncio.to_thread(self.store.identify_image, data, mime_type)
            await self.visual_memory.embed_image(actual, work_id=image["id"],
                                                 variant_id=identity["fingerprint"]["pixel_hash"], metadata=image)
            processed += 1
            job.update(progress=round(processed * 100 / len(images)), message=f"已整理 {processed}/{len(images)} 张作品的视觉记忆")
            await self.save_job_progress(job)
        job.update(status="done", progress=100, message=f"已为 {processed} 张收藏建立视觉记忆", indexed_count=processed)
        await self.save_job_progress(job)
