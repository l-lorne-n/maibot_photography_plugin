"""Real-image batch browsing with separate preview, detail, and collection stages."""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlsplit

import asyncio
import math

from .errors import ModelFailure
from .reader import ReaderError, image_content_hash, sanitized_failure, seed_work_items
from .similarity import is_clearer


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(*parts: str) -> str:
    return sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ModelFailure("视觉模型的兴趣评分必须为 0–1 的有限数字")
    return float(value)


def _text(value: Any, field: str, limit: int, required: bool = False) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        shape = _field_shape(value)
        raise ModelFailure(f"视觉模型的 {field} 必须为{'非空' if required else ''}字符串；实际为 {shape['type']}")
    return value.strip()[:limit]


_VISION_FIELDS = ("author", "work", "title", "identity_verified", "identity_evidence", "observations",
                  "feeling", "reason", "directions", "bot_interest", "collect", "matches_seed", "matched_work_ids", "visual_comparisons", "model")


def _field_shape(value: Any, present: bool = True) -> Dict[str, Any]:
    if not present:
        return {"type": "missing"}
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "empty_string" if not value else "blank_string" if not value.strip() else "string", "length": len(value)}
    if isinstance(value, list):
        return {"type": "array/list", "length": len(value),
                "item_types": list(dict.fromkeys(type(item).__name__ for item in value[:8]))}
    if isinstance(value, dict):
        return {"type": "object/dict", "length": len(value)}
    return {"type": "boolean" if isinstance(value, bool) else "number"}


def _limited_visual_value(value: Any, text_limit: int, items: int, depth: int = 2) -> Any:
    """Bound persisted model fields while retaining their JSON types for validation."""
    if isinstance(value, str):
        return value[:text_limit]
    if isinstance(value, list):
        return [_limited_visual_value(item, text_limit, items, depth - 1) for item in value[:items]] if depth else []
    if isinstance(value, dict):
        return {str(key)[:80]: _limited_visual_value(item, text_limit, items, depth - 1)
                for key, item in list(value.items())[:items]} if depth else {}
    return value


class BrowsingMixin:
    """Uses the engine's Reader, Store, `_model`, `db`, locks, and cancellation set.

    Only metadata and visual notes are persisted for previews. Images supplied
    to the model live in bounded batches in memory and are fetched again for
    detailed viewing. Detail observations and collections have independent IDs.
    """

    def _browse_targets(self, seed: Optional[Dict[str, Any]]) -> Dict[str, int]:
        if seed:
            books = max(1, len(seed_work_items(seed)))
            return {"preview_min": min(6, books * 2), "preview_max": 6,
                    "detail_min": min(3, books), "detail_max": 3}
        return {key: int(self.settings.model_dump()[key]) for key in
                ("preview_min", "preview_max", "detail_min", "detail_max")}

    async def _browse_persist(self, job: Dict[str, Any], state: Dict[str, Any]) -> None:
        job["preview_count"] = len(job["preview_ids"])
        job["inspected_image_count"] = len(job["viewed_preview_ids"])
        job["detail_count"] = len(job["observation_ids"])
        job["message"] = (f"{state['label']} · 实际预览 {len(state['preview_ids'])}/{state['targets']['preview_min']}，"
                          f"精看 {len(state['detail_ids'])}/{state['targets']['detail_min']}，收藏 {len(job['image_ids'])}")
        await self.save_job_progress(job)

    @staticmethod
    def _browse_link_candidates(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        attempted = set(state["attempted_pages"])
        return [link for link in state["links"] if link["url"] not in attempted]

    @staticmethod
    def _browse_add_links(state: Dict[str, Any], links: List[Dict[str, Any]]) -> int:
        known = {link["url"] for link in state["links"]} | set(state["attempted_pages"])
        added = [dict(link) for link in links if link["url"] not in known]
        # New search results and page links must be visible in the next compact
        # navigation input even when the saved source list fills that input.
        deduped: Dict[str, Dict[str, Any]] = {link["url"]: link for link in added + state["links"]}
        state["links"] = list(deduped.values())[:150]
        return len(added)

    def _browse_error(self, job: Dict[str, Any], state: Dict[str, Any], action: str,
                      error: Any, url: str = "") -> None:
        entry = {"at": _now(), "action": action, "error": sanitized_failure(error), "url": url}
        job["errors"].append(entry)
        job["errors"] = job["errors"][-100:]
        state["history"].append(entry)

    async def _explore(self, job: Dict[str, Any], seed: Optional[Dict[str, Any]]) -> None:
        job.setdefault("preview_ids", [])
        job.setdefault("viewed_preview_ids", [])
        job.setdefault("browse_results", {})
        sources = [source for source in await self.db("list", "source", limit=200) if source.get("enabled", True)]
        sources.sort(key=self.source_weight, reverse=True)
        unit = seed["id"] if seed else "daily"
        state = job.get("state")
        if state is not None and state.get("schema_version") != 2:
            raise ValueError("旧任务的导航状态不能直接续接批量预览，请新建浏览任务")
        if state is None:
            state = {"schema_version": 2, "unit": unit, "label": seed["author"] if seed else "自主漫游",
                     "targets": self._browse_targets(seed), "links": [], "attempted_pages": [], "queries": [],
                     "candidate_urls": [], "preview_queue": [], "pending_batch": [], "pending_preview_results": [],
                     "preview_ids": [], "viewed_ids": [],
                     "preview_hashes": {}, "detail_ids": [], "detail_attempted": [], "collected_authors": [],
                     "history": [], "navigation_due": True, "search_streak": 0, "no_progress": 0,
                     "page": None, "stop_reason": "", "stop_explanation": ""}
            self._browse_add_links(state, [{"url": item["url"], "title": item["name"]} for item in sources])
            job["state"] = state
        if state["unit"] != unit:
            raise ValueError("持久化浏览状态属于另一位作者，不能混用")
        state.setdefault("pending_detail_result", None)
        state.setdefault("preview_work_ids", {})
        job.setdefault("preview_work_ids", {})
        job.setdefault("viewed_work_ids", {})
        profile = await self.db("get", "profile", "current")
        seed_preferences = await self.db("list", "seed", limit=30)
        feedback = await self.db("list", "feedback", limit=30)
        facts = await self.db("list", "fact_check", limit=80)
        context = {
            "intent": job["intent"][:1200],
            "seed": {"id": seed["id"], "author": seed["author"], "work_items": seed_work_items(seed)} if seed else None,
            "profile": {"summary": str(profile.get("summary", ""))[:1200],
                        "directions": profile.get("directions", [])[:8], "questions": profile.get("questions", [])[:4]} if profile else None,
            "preferred_authors": self.settings.daily_authors,
            "exploration_ratio": self.settings.exploration_ratio,
            "sources": [{"url": source["url"], "weight": self.source_weight(source)} for source in sources[:12]],
            "seed_preferences": [{"author": item["author"], "work_items": seed_work_items(item)[:10]}
                                 for item in seed_preferences],
            "recent_feedback": [{"image_id": item.get("image_id", ""), "image_ids": item.get("image_ids", []),
                                 "scope": item.get("scope", "image"), "speaker_id": item.get("speaker_id", ""),
                                 "insight": str(item.get("insight", ""))[:300], "human_interest": item.get("human_interest")}
                                for item in feedback if item.get("status") == "evaluated" and item.get("scope") != "unrelated"][:8],
            "source_notes": [str(item.get("conclusion", ""))[:350]
                             for item in facts if item.get("status") == "resolved" and item.get("conclusion")][:8],
        }
        reason = ""
        explanation = ""
        while not reason:
            if job["id"] in self.cancelled:
                await self._browse_persist(job, state)
                return
            targets = state["targets"]
            preview_count = len(state["preview_ids"])
            detail_count = len(state["detail_ids"])
            # First compare a broad real-image sample. Detail is a subset of this
            # sample; sufficient browsing never implies mandatory collection.
            if preview_count >= targets["preview_min"]:
                await self._browse_details(job, state, seed, context)
                if job["id"] in self.cancelled:
                    await self._browse_persist(job, state)
                    return
                detail_count = len(state["detail_ids"])
                if detail_count >= targets["detail_min"]:
                    reason, explanation = "targets_met", "已完成真实预览和兴趣优先的精看，收藏数量独立决定"
                    break
            if preview_count >= targets["preview_max"]:
                reason, explanation = "preview_limit", "已达到真实预览上限，符合精看条件的样本不足"
                break
            if state["no_progress"] >= 3:
                if state["pending_preview_results"] or state["pending_batch"] or state["preview_queue"]:
                    # Navigation cannot declare a page exhausted while its actual
                    # discovered images remain untried behind icons or book covers.
                    state["navigation_due"] = False
                else:
                    reason, explanation = "no_progress", "图像候选已处理，连续三轮没有取得新的摄影作品或页面线索（重复内容、非作品图片、不可达或空结果）"
                    break
            if state["pending_preview_results"] or state["pending_batch"] or (state["preview_queue"] and not state["navigation_due"]):
                await self._browse_preview(job, state, seed, context)
                continue
            navigation = await self._browse_navigate(job, state, context)
            if navigation:
                reason, explanation = navigation
            await self._browse_persist(job, state)
        # Exhausted navigation may still leave a small, worthwhile detail sample.
        if reason != "targets_met" and job["id"] not in self.cancelled:
            await self._browse_details(job, state, seed, context)
        if job["id"] in self.cancelled:
            await self._browse_persist(job, state)
            return
        state.update(stop_reason=reason, stop_explanation=explanation)
        targets = state["targets"]
        result = {**targets, "preview_count": len(state["preview_ids"]), "detail_count": len(state["detail_ids"]),
                  "inspected_image_count": len(state["viewed_ids"]), "non_photo_count": len(state["viewed_ids"]) - len(state["preview_ids"]),
                  "target_met": len(state["preview_ids"]) >= targets["preview_min"] and len(state["detail_ids"]) >= targets["detail_min"],
                  "stop_reason": reason, "stop_explanation": explanation, "authors": state["collected_authors"],
                  "navigation_actions": sum(item.get("action") in {"search", "open"} for item in state["history"]),
                  "error_count": sum("error" in item for item in state["history"])}
        job["browse_results"][unit] = result
        await self._browse_persist(job, state)
        await self.db("put", "exploration", {"id": job["id"] + ":" + unit, "job_id": job["id"],
            "seed_id": seed["id"] if seed else "", "visited": state["attempted_pages"],
            "seen": list(state["preview_hashes"]), "preview_ids": state["preview_ids"],
            "observation_ids": state["detail_ids"], "history": state["history"], **result})

    async def _browse_navigate(self, job: Dict[str, Any], state: Dict[str, Any],
                               context: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        candidates = self._browse_link_candidates(state)
        recent = []
        for entry in state["history"][-5:]:
            recent.append({key: entry[key] for key in ("action", "url", "count", "photo_count", "new_images", "new_links") if key in entry} | {
                "reason": str(entry.get("reason", ""))[:240], "error": str(entry.get("error", ""))[:300],
                "notes": entry.get("notes", [])[:3]})
        decision = await self._model("decision", {**context, "targets": state["targets"],
            "preview_count": len(state["preview_ids"]), "detail_count": len(state["detail_ids"]),
            "pending_images": len(state["preview_queue"]), "candidate_links": candidates[:24],
            "current_page": state["page"], "recent_actions": recent}, job)
        if job["id"] in self.cancelled:
            return None
        action = decision.get("action")
        if action not in {"search", "open", "preview", "finish"}:
            raise ModelFailure("导航模型必须选择 search/open/preview/finish")
        reason = _text(decision.get("reason", ""), "reason", 600)
        outcome: Dict[str, Any] = {"action": action, "reason": reason, "at": _now()}
        if action == "finish":
            if state["preview_queue"]:
                outcome["not_accepted"] = "尚未达到浏览目标，继续已发现的真实图像候选"
                state["history"].append(outcome)
                state["navigation_due"] = False
                return None
            if candidates:
                outcome["not_accepted"] = "尚有未尝试页面且最低浏览量未满足，继续实际候选"
                state["history"].append(outcome)
                return await self._browse_open(job, state, candidates[0]["url"], "继续尚未尝试的页面候选")
            state["history"].append(outcome)
            return "candidates_exhausted", reason or "模型结束检索，已发现的页面与图像候选均已处理"
        if action == "preview":
            state["history"].append(outcome)
            if state["preview_queue"]:
                state["navigation_due"] = False
            else:
                self._browse_error(job, state, action, "没有已发现且未处理的图像候选")
                state["no_progress"] += 1
            return None
        if action == "open":
            url = decision.get("url")
            if not isinstance(url, str) or url not in {link["url"] for link in candidates}:
                self._browse_error(job, state, action, "打开的网址不在未尝试的实际候选中", str(url)[:8192])
                state["no_progress"] += 1
                return None
            return await self._browse_open(job, state, url, reason)
        query = decision.get("query")
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 128:
            raise ModelFailure("导航模型的搜索词必须为 1–128 字符")
        normalized_query = " ".join(query.split()).casefold()
        if normalized_query in state["queries"]:
            self._browse_error(job, state, "search", "本轮已尝试相同搜索词")
            state["no_progress"] += 1
            return None
        # New query spelling alone is not exploration progress. After repeated
        # search-only choices, use the actual discovered frontier before searching again.
        if state["search_streak"] >= 3:
            if state["preview_queue"]:
                state["navigation_due"] = False
                return None
            if candidates:
                return await self._browse_open(job, state, candidates[0]["url"], "先查看已经找到的页面")
        state["queries"].append(normalized_query)
        state["search_streak"] += 1
        await self._browse_persist(job, state)
        try:
            found = await self.reader.search(query.strip(), limit=5)
        except asyncio.CancelledError:
            state["queries"].remove(normalized_query)
            state["search_streak"] -= 1
            raise
        except (ReaderError, OSError, TimeoutError) as exc:
            self._browse_error(job, state, "search", exc)
            state["no_progress"] += 1
            return None
        added = self._browse_add_links(state, found)
        state["history"].append({**outcome, "query": query, "new_links": added, "results": found})
        state["no_progress"] = 0 if added else state["no_progress"] + 1
        return None

    async def _browse_open(self, job: Dict[str, Any], state: Dict[str, Any], url: str,
                           reason: str) -> Optional[Tuple[str, str]]:
        state["attempted_pages"].append(url)
        state["search_streak"] = 0
        await self._browse_persist(job, state)
        try:
            page = await self.reader.page(url)
        except asyncio.CancelledError:
            state["attempted_pages"].remove(url)
            raise
        except (ReaderError, OSError, TimeoutError) as exc:
            self._browse_error(job, state, "open", exc, url)
            state["no_progress"] += 1
            return None
        if job["id"] in self.cancelled:
            # Retry this unfinished open on resume; no successful page was recorded.
            state["attempted_pages"].remove(url)
            return None
        page_id = _identifier(job["id"], state["unit"], page["url"])
        await self.db("put", "browse_page", {"id": page_id, "job_id": job["id"], "url": page["url"],
                                              "title": page["title"], "text": page["text"][:12000]})
        new_links = self._browse_add_links(state, page["links"])
        known = set(state["candidate_urls"])
        added = 0
        for image in page["images"]:
            if unquote(urlsplit(image["url"]).path).lower().endswith((".svg", ".ico")):
                state["history"].append({"action": "unsupported_image", "url": image["url"],
                                         "reason": "SVG/ICO 网站图标不进入摄影图像预览"})
                continue
            if image["url"] in known:
                continue
            known.add(image["url"])
            state["candidate_urls"].append(image["url"])
            state["preview_queue"].append({"image_url": image["url"], "source_url": page["url"],
                                           "page_id": page_id, "alt": image.get("alt", "")[:300]})
            added += 1
        state["page"] = {"url": page["url"], "title": page["title"], "images": len(page["images"])}
        state["navigation_due"] = not bool(state["preview_queue"])
        state["history"].append({"action": "open", "url": url, "reason": reason,
                                 "new_images": added, "new_links": new_links})
        state["no_progress"] = 0 if added or new_links else state["no_progress"] + 1
        return None

    async def _browse_preview(self, job: Dict[str, Any], state: Dict[str, Any], seed: Optional[Dict[str, Any]],
                              context: Dict[str, Any]) -> None:
        if state["pending_preview_results"]:
            await self._browse_commit_previews(job, state)
            return
        before = len(state["preview_ids"])
        remaining = state["targets"]["preview_max"] - len(state["preview_ids"])
        if not state["pending_batch"]:
            size = min(2 if seed else 6, remaining, len(state["preview_queue"]))
            state["pending_batch"] = state["preview_queue"][:size]
            del state["preview_queue"][:size]
            await self._browse_persist(job, state)
        images = []
        metadata = []
        batch_works: Dict[str, int] = {}
        for candidate in state["pending_batch"]:
            if job["id"] in self.cancelled:
                return
            try:
                actual = await self.reader.image(candidate["image_url"])
                fingerprint = await asyncio.to_thread(image_content_hash, actual)
                identity = await asyncio.to_thread(self.store.identify_image, actual["data"], actual["mime_type"])
            except (ReaderError, OSError, TimeoutError) as exc:
                self._browse_error(job, state, "preview_download", exc, candidate["image_url"])
                continue
            work_id = identity["work_id"]
            async with self.collection_lock:
                await asyncio.to_thread(self.store.upgrade_image, actual["data"], actual["mime_type"],
                    {"source_url": candidate["source_url"], "image_url": candidate["image_url"]},
                    self.settings.max_images, self.settings.max_mb * 1024 ** 2, self.settings.idle_days)
            previous_id = state["preview_work_ids"].get(work_id) or state["preview_hashes"].get(fingerprint)
            if previous_id:
                previous = await self.db("get", "preview", previous_id)
                pending = state.get("pending_detail_result")
                if (previous is not None and not previous.get("detail_observation_id")
                        and not (pending and pending["preview_id"] == previous_id)
                        and previous.get("identity_signature")
                        and is_clearer(identity["fingerprint"], previous["identity_signature"])):
                    await self.db("put", "preview", {**candidate, "id": previous_id, "image_hash": fingerprint,
                        "width": actual["width"], "height": actual["height"], "identity_signature": identity["fingerprint"],
                        "work_id": work_id, "phash": identity["phash"], "low_resolution": min(actual["width"], actual["height"]) < 300})
                state["history"].append({"action": "duplicate_image", "url": candidate["image_url"],
                    "image_hash": fingerprint, "work_id": work_id, "distance": identity["distance"]})
                continue
            preview_id = _identifier("preview", job["id"], state["unit"], work_id)
            existing = await self.db("get", "preview", preview_id)
            if existing:
                self._browse_register_preview(job, state, existing)
                continue
            page = await self.db("get", "browse_page", candidate["page_id"])
            info = {**candidate, "id": preview_id, "image_hash": fingerprint,
                             "work_id": work_id, "phash": identity["phash"], "identity_signature": identity["fingerprint"],
                             "width": actual["width"], "height": actual["height"], "page_title": page["title"],
                             "source_excerpt": page["text"][:1200], "low_resolution": min(actual["width"], actual["height"]) < 300}
            if work_id in batch_works:
                position = batch_works[work_id]
                if is_clearer(identity["fingerprint"], metadata[position]["identity_signature"]):
                    metadata[position] = info
                    images[position] = {**actual, "id": preview_id}
                continue
            batch_works[work_id] = len(metadata)
            metadata.append(info)
            images.append({**actual, "id": preview_id})
        if job["id"] in self.cancelled:
            return
        if images:
            result = await self._model("preview", {**context, "images": metadata,
                "instruction": "编号与真实图像顺序对应。只评价画面中可见内容，小图不可断言细节；先看再筛，预览不等于收藏。"}, job, images=images)
            if job["id"] in self.cancelled:
                return
            items = result.get("items")
            if not isinstance(items, list) or len(items) != len(metadata):
                raise ModelFailure("批量预览必须为每张实际输入图片返回一个带原编号的记录")
            by_id = {item.get("id"): item for item in items if isinstance(item, dict)}
            if set(by_id) != {item["id"] for item in metadata} or len(by_id) != len(items):
                raise ModelFailure("批量预览结果的图片编号缺失、重复或不属于本批输入")
            records = []
            for info in metadata:
                item = by_id[info["id"]]
                if not isinstance(item.get("is_photo"), bool) or not isinstance(item.get("detail"), bool):
                    raise ModelFailure("批量预览缺少明确的摄影作品/精看选择标记")
                records.append({key: value for key, value in info.items() if key != "source_excerpt"} | {
                    "job_id": job["id"], "seed_id": seed["id"] if seed else "", "stage": "preview",
                    "visual_note": _text(item.get("visual_note"), "visual_note", 600, True),
                    "interest": _score(item.get("interest")), "is_photo": item["is_photo"], "detail": item["detail"],
                    "reason": _text(item.get("reason", ""), "reason", 500), "model": result.get("model", ""),
                    "created_at": _now(), "actually_viewed": True})
            state["pending_preview_results"] = records
            # Checkpoint the whole validated visual response before individual
            # record writes, so a partial commit resumes without another VLM call.
            await self._browse_persist(job, state)
            await self._browse_commit_previews(job, state)
            return
        state["no_progress"] = 0 if len(state["preview_ids"]) > before else state["no_progress"] + 1
        state["pending_batch"] = []
        state["navigation_due"] = True
        state["search_streak"] = 0
        await self._browse_persist(job, state)

    async def _browse_commit_previews(self, job: Dict[str, Any], state: Dict[str, Any]) -> None:
        records = state["pending_preview_results"]
        for record in records:
            await self.db("put", "preview", record)
            self._browse_register_preview(job, state, record)
        state["history"].append({"action": "preview_batch", "count": len(records),
                                 "photo_count": sum(record["is_photo"] for record in records),
                                 "notes": [{"id": record["id"], "note": record["visual_note"][:160],
                                            "interest": record["interest"]} for record in records]})
        state["no_progress"] = 0 if any(record["is_photo"] for record in records) else state["no_progress"] + 1
        state["pending_preview_results"] = []
        state["pending_batch"] = []
        state["navigation_due"] = True
        state["search_streak"] = 0
        await self._browse_persist(job, state)

    @staticmethod
    def _browse_register_preview(job: Dict[str, Any], state: Dict[str, Any], record: Dict[str, Any]) -> None:
        work_id = record.get("work_id", record["image_hash"])
        state.setdefault("preview_work_ids", {})[work_id] = record["id"]
        viewed = job.setdefault("viewed_work_ids", {})
        photos = job.setdefault("preview_work_ids", {})
        if record["id"] not in state["viewed_ids"]:
            state["viewed_ids"].append(record["id"])
        if work_id not in viewed:
            viewed[work_id] = record["id"]
            job["viewed_preview_ids"].append(record["id"])
        if record["is_photo"] and record["id"] not in state["preview_ids"]:
            state["preview_ids"].append(record["id"])
        if record["is_photo"] and work_id not in photos:
            photos[work_id] = record["id"]
            job["preview_ids"].append(record["id"])
        state["preview_hashes"][record["image_hash"]] = record["id"]

    async def _browse_details(self, job: Dict[str, Any], state: Dict[str, Any], seed: Optional[Dict[str, Any]],
                              context: Dict[str, Any]) -> None:
        previews = [await self.db("get", "preview", identifier) for identifier in state["preview_ids"]]
        photos = [preview for preview in previews if preview["is_photo"]]
        wanted = min(state["targets"]["detail_max"], max(state["targets"]["detail_min"],
                     sum(preview["detail"] or preview["interest"] >= 0.65 for preview in photos)))
        candidates = [preview for preview in photos if preview["id"] not in state["detail_attempted"]]
        candidates.sort(key=lambda record: (record["interest"], record["detail"]), reverse=True)
        for preview in candidates:
            if job["id"] in self.cancelled or len(state["detail_ids"]) >= wanted:
                return
            completed = await self._browse_detail(job, state, seed, context, preview)
            if completed:
                state["detail_attempted"].append(preview["id"])
            await self._browse_persist(job, state)

    async def _browse_detail(self, job: Dict[str, Any], state: Dict[str, Any], seed: Optional[Dict[str, Any]],
                             context: Dict[str, Any], preview: Dict[str, Any]) -> bool:
        observation_id = _identifier("detail", job["id"], preview.get("work_id", preview["image_hash"]))
        observation = await self.db("get", "observation", observation_id)
        pending = state.get("pending_detail_result")
        if pending and (pending["preview_id"] != preview["id"] or pending["image_hash"] != preview["image_hash"]):
            raise ValueError("待提交精看结果与当前图片不一致，不能混用视觉记录")
        actual = None
        if observation is None or (not observation.get("detail_complete") and observation["collect"]):
            try:
                actual = await self.reader.image(preview["image_url"])
                fingerprint = await asyncio.to_thread(image_content_hash, actual)
            except (ReaderError, OSError, TimeoutError) as exc:
                self._browse_error(job, state, "detail_download", exc, preview["image_url"])
                state["pending_detail_result"] = None
                return True
            if fingerprint != preview["image_hash"]:
                self._browse_error(job, state, "detail_download", "原网址的图像内容已变化，不能当作此前预览图精看", preview["image_url"])
                state["pending_detail_result"] = None
                return True
        if job["id"] in self.cancelled:
            return False
        if observation is None:
            if pending is None:
                page = await self.db("get", "browse_page", preview["page_id"])
                historical, historical_images = await self._historical_visual_context(actual, preview, job)
                material = {**context, "preview": preview, "historical_neighbours": historical,
                            "current_image_id": preview["id"], "seed": context["seed"],
                            "page_url": preview["source_url"], "page_text": page["text"][:9000]}
                if historical_images:
                    result = await self._model("vision", material, job,
                        images=[{**actual, "id": preview["id"]}] + historical_images)
                else:
                    result = await self._model("vision", material, job, image=actual)
                pending = {"preview_id": preview["id"], "image_hash": preview["image_hash"],
                    "historical_image_ids": [item["image_id"] for item in historical],
                    "image_url": preview["image_url"], "source_url": preview["source_url"],
                    "result": {key: _limited_visual_value(result[key], 12000, 24) for key in _VISION_FIELDS if key in result},
                    "field_types": {key: _field_shape(result.get(key), key in result) for key in _VISION_FIELDS},
                    "received_at": _now()}
                pending["truncated_fields"] = [key for key, value in pending["result"].items() if value != result[key]]
                state["pending_detail_result"] = pending
                # Save only this image's whitelisted visual output, never its
                # request prompt, character context, conversations, or image bytes.
                await self._browse_persist(job, state)
            if job["id"] in self.cancelled:
                return False
            result = pending["result"]
            try:
                if pending["truncated_fields"]:
                    raise ModelFailure("精看模型字段超出可保存范围：" + ", ".join(pending["truncated_fields"]))
                observation = self._browse_validate_detail(result, job, seed, preview, actual, observation_id)
                comparisons = result.get("visual_comparisons", [])
                known = set(pending.get("historical_image_ids", []))
                if not isinstance(comparisons, list) or len(comparisons) > len(known):
                    raise ModelFailure("历史作品比较必须为本次提供的历史图片记录数组")
                if known and not comparisons:
                    raise ModelFailure("已提供历史图片，精看模型未返回构图与色彩比较")
                checked = []
                used = set()
                for comparison in comparisons:
                    if not isinstance(comparison, dict) or comparison.get("image_id") not in known or comparison["image_id"] in used:
                        raise ModelFailure("历史作品比较包含未知或重复图片编号")
                    used.add(comparison["image_id"])
                    checked.append({"image_id": comparison["image_id"], **{
                        key: _text(comparison.get(key), key, 800, True)
                        for key in ("composition", "color", "visual_style", "preference_insight")}})
                observation["visual_comparisons"] = checked
            except ModelFailure as exc:
                job["last_model_error"] = {"at": _now(), "purpose": "vision", "preview_id": preview["id"],
                    "image_url": preview["image_url"], "source_url": preview["source_url"],
                    "model": _limited_visual_value(result.get("model", ""), 120, 1), "error": sanitized_failure(exc),
                    "field_types": pending["field_types"],
                    "result": {key: _limited_visual_value(value, 180, 4, 1) for key, value in result.items()}}
                # A failed response is diagnostic evidence, not a resumable result.
                # Explicit retry requests this same unfinished image again.
                state["pending_detail_result"] = None
                await self._browse_persist(job, state)
                raise
            # Persist the actual visual result before collection work. A cancelled
            # save can resume without paying for or inventing another observation.
            await self.db("put", "observation", observation)
        if pending is not None:
            state["pending_detail_result"] = None
            await self._browse_persist(job, state)
        if not observation.get("detail_complete"):
            saved = None
            if observation["collect"] and observation["reason"]:
                try:
                    async with self.collection_lock:
                        saved = await asyncio.to_thread(self.store.save_image, actual["data"], actual["mime_type"],
                            {**observation, "observation_id": observation_id}, self.settings.max_images,
                            self.settings.max_mb * 1024 ** 2, self.settings.idle_days)
                except (ValueError, OSError) as exc:
                    observation["collection_error"] = sanitized_failure(exc)
                    self._browse_error(job, state, "collect", exc, preview["image_url"])
            if saved:
                observation["image_id"] = saved["id"]
                if observation.get("visual_comparisons"):
                    await self.db("put", "image", {"id": saved["id"],
                        "visual_comparisons": observation["visual_comparisons"],
                        "visual_observation_id": observation_id})
            observation["detail_complete"] = True
            await self.db("put", "observation", observation)
            await self._source_observed(preview["source_url"], observation["bot_interest"],
                                        preview["image_hash"], observation["directions"])
        if observation_id not in state["detail_ids"]:
            state["detail_ids"].append(observation_id)
        if observation_id not in job["observation_ids"]:
            job["observation_ids"].append(observation_id)
        if observation.get("image_id") and observation["image_id"] not in job["image_ids"]:
            job["image_ids"].append(observation["image_id"])
        if observation.get("image_id") and observation["author"] and observation["author"] not in state["collected_authors"]:
            state["collected_authors"].append(observation["author"])
        await self.db("put", "preview", {"id": preview["id"], "detail_observation_id": observation_id})
        state["history"].append({"action": "detail", "preview_id": preview["id"], "observation_id": observation_id,
                                 "url": preview["image_url"], "collected": bool(observation.get("image_id")),
                                 "author": observation["author"], "observations": observation["observations"][:600]})
        return True

    @staticmethod
    def _browse_validate_detail(result: Dict[str, Any], job: Dict[str, Any],
                                seed: Optional[Dict[str, Any]], preview: Dict[str, Any],
                                actual: Dict[str, Any], observation_id: str) -> Dict[str, Any]:
        if not isinstance(result.get("collect"), bool) or not isinstance(result.get("identity_verified"), bool):
            raise ModelFailure("精看模型的 collect 和 identity_verified 必须为 boolean")
        matches_seed = result.get("matches_seed", False)
        work_ids = result.get("matched_work_ids", [])
        if not isinstance(matches_seed, bool) or not isinstance(work_ids, list) or any(not isinstance(item, str) for item in work_ids):
            raise ModelFailure("精看模型的 matches_seed 必须为 boolean，matched_work_ids 必须为字符串数组")
        known = {item["id"]: item["title"] for item in seed_work_items(seed)} if seed else {}
        if any(identifier not in known for identifier in work_ids):
            raise ModelFailure("精看模型返回了未提供的作品编号")
        directions = result.get("directions", [])
        if not isinstance(directions, list) or any(not isinstance(item, str) for item in directions):
            raise ModelFailure("精看模型的 directions 必须为字符串数组")
        observation = {"id": observation_id, "job_id": job["id"], "preview_id": preview["id"],
            "seed_id": seed["id"] if seed else "", "image_hash": preview["image_hash"], "stage": "detail",
            "work_id": preview.get("work_id", ""), "phash": preview.get("phash", ""),
            "source_url": preview["source_url"], "image_url": preview["image_url"], "created_at": _now(),
            "width": actual["width"], "height": actual["height"], "human_interest": None,
            "bot_interest": _score(result.get("bot_interest")), "collect": result["collect"],
            "identity_verified": result["identity_verified"], "matches_seed": matches_seed,
            "matched_work_ids": list(dict.fromkeys(work_ids)), "matched_works": [known[item] for item in work_ids],
            "directions": [item[:120] for item in directions[:12]], "model": result.get("model", ""),
            "detail_complete": False}
        for key, maximum, required in (("author", 200, False), ("work", 500, False), ("title", 300, False),
                ("identity_evidence", 2000, False), ("observations", 3000, True), ("feeling", 1200, False), ("reason", 1600, False)):
            if required and key not in result:
                raise ModelFailure(f"视觉模型的 {key} 必须为非空字符串；实际为 missing")
            observation[key] = _text(result.get(key), key, maximum, required)
        return observation
