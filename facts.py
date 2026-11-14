"""Source-backed attribution work, kept separate from aesthetic reflections."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Dict, List

import asyncio

from .reader import ReaderError


class FactCheckingMixin:
    async def _prepare_fact_retry(self, job: Dict[str, Any]) -> None:
        """Explicit retries may revisit this task's unsuccessful checks immediately."""
        attempted = set(job.get("attempted_fact_ids", []))
        source_job_id = job["id"][len("facts-"):] if job["id"].startswith("facts-") else ""
        retry_ids = []
        for fact in await self.db("list", "fact_check", limit=500):
            if fact["status"] not in {"failed", "unresolved", "checking", "pending"}:
                continue
            checked_job_id = fact.get("checked_job_id", "")
            owned = checked_job_id == job["id"] or fact["id"] in attempted
            legacy_owned = (not checked_job_id and source_job_id and fact.get("job_id") == source_job_id
                            and fact["status"] in {"failed", "unresolved", "checking"})
            if owned or legacy_owned:
                retry_ids.append(fact["id"])
        job["retry_fact_ids"] = retry_ids

    async def _fact_observation_context(self) -> List[Dict[str, Any]]:
        """Supply recorded provenance and visual notes, not new images or inferred authors."""
        observations = await self.db("list", "observation", limit=36)
        limits = {"id": 100, "job_id": 100, "author": 200, "work": 500, "title": 300,
                  "image_url": 2048, "identity_evidence": 700, "observations": 400}
        return [{**{key: observation[key][:maximum] for key, maximum in limits.items()
                    if isinstance(observation.get(key), str)},
                 "page_url": observation.get("source_url", "")[:2048],
                 "identity_verified": observation.get("identity_verified") is True}
                for observation in observations]

    async def _queue_fact_checks(self, items: List[Any], job: Dict[str, Any]) -> None:
        if not isinstance(items, list):
            raise ValueError("资料待查清单必须为数组")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("question"), str):
                raise ValueError("资料待查项缺少问题")
            question = item["question"].strip()[:1000]
            query = item.get("search_query", question).strip()[:128]
            if not question or not query:
                raise ValueError("资料待查问题或搜索词为空")
            ident = sha256(question.casefold().encode()).hexdigest()[:24]
            if await self.db("get", "fact_check", ident) is None:
                await self.db("put", "fact_check", {
                    "id": ident, "question": question, "search_query": query,
                    "status": "pending", "job_id": job["id"], "attempts": 0,
                    "conclusion": "", "sources": [], "error": "",
                })

    async def _check_facts(self, job: Dict[str, Any]) -> None:
        """Check a small separate queue; unresolved facts never block viewing."""
        now = datetime.now(timezone.utc)
        retry_ids = set(job.get("retry_fact_ids", []))
        pending = []
        for fact in await self.db("list", "fact_check", limit=500):
            if fact["status"] == "resolved":
                continue
            checked = fact.get("checked_at")
            if fact["id"] not in retry_ids and checked and now - datetime.fromisoformat(checked) < timedelta(days=1):
                continue
            pending.append(fact)
        pending.sort(key=lambda fact: (fact["id"] not in retry_ids, fact["created_at"]))
        observations = await self._fact_observation_context() if pending else []
        completed = 0
        for fact in pending[:2]:
            if job["id"] in self.cancelled:
                return
            attempted = list(job.get("attempted_fact_ids", []))
            if fact["id"] not in attempted:
                attempted.append(fact["id"])
            job["attempted_fact_ids"] = attempted
            job["retry_fact_ids"] = [ident for ident in job.get("retry_fact_ids", []) if ident != fact["id"]]
            await self.db("put", "job", job)
            fact.update(status="checking", attempts=fact.get("attempts", 0) + 1,
                        checked_job_id=job["id"], error="")
            await self.db("put", "fact_check", fact)
            try:
                results = await self.reader.search(fact["search_query"], limit=5)
                pages = []
                failures = []
                for result in results[:2]:
                    if job["id"] in self.cancelled:
                        return
                    try:
                        page = await self.reader.page(result["url"])
                        source_id = "source-" + sha256(page["url"].encode()).hexdigest()[:16]
                        if not any(opened["source_id"] == source_id for opened in pages):
                            pages.append({"source_id": source_id, "url": page["url"],
                                          "title": page["title"], "text": page["text"][:5000]})
                    except ReaderError as exc:
                        failures.append(str(exc))
                if not pages:
                    raise ReaderError("资料核查没有取得可读来源：" + "；".join(failures))
                judgement = await self._model("fact_check", {
                    "question": fact["question"], "pages": pages, "search_results": results,
                    "recent_observations": observations,
                }, job)
                if not isinstance(judgement.get("resolved"), bool) or not isinstance(judgement.get("conclusion"), str):
                    raise ValueError("资料核查结果缺少明确的结论状态")
                source_ids = judgement.get("source_ids")
                allowed = {page["source_id"]: page["url"] for page in pages}
                if not isinstance(source_ids, list) or any(not isinstance(ident, str) or ident not in allowed for ident in source_ids):
                    raise ValueError("资料核查 source_ids 必须引用本次实际打开页面的有效 source_id，不能填写URL或未知编号")
                source_ids = list(dict.fromkeys(source_ids))
                sources = [allowed[ident] for ident in source_ids]
                if judgement["resolved"] and (not sources or not judgement["conclusion"].strip()):
                    raise ValueError("已查明的资料必须有实际来源和结论")
                fact.update(status="resolved" if judgement["resolved"] else "unresolved",
                            conclusion=judgement["conclusion"], sources=sources, source_ids=source_ids,
                            checked_at=datetime.now(timezone.utc).isoformat(), error="；".join(failures))
            except asyncio.CancelledError:
                fact["status"] = "pending"
                await self.db("put", "fact_check", fact)
                raise
            except Exception as exc:
                fact.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:1500],
                            checked_at=datetime.now(timezone.utc).isoformat())
                await self.db("put", "fact_check", fact)
                raise
            await self.db("put", "fact_check", fact)
            completed += 1
        job.update(status="done", progress=100, message=f"已单独核查 {completed} 项来源资料")
        await self.db("put", "job", job)
