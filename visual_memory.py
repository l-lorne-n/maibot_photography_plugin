"""Plugin-owned image vectors, keyed by observed model space and prepared pixels."""

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Optional

from maibot_sdk.capabilities.llm import LLMCapability

import asyncio
import base64
import json
import math

from .errors import ModelFailure
from .reader import image_content_hash, prepare_vision_image
from .store import Store


PREPROCESS_VERSION = "photography-embedding-jpeg1024-v1"
RECORD_KIND = "visual_embedding"
_METADATA_FIELDS = ("author", "work", "title", "source_url", "image_url", "observations", "feeling")


def _digest(value: Dict[str, Any]) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalized_vector(values: Any) -> List[float]:
    """Validate a dense model vector and return unit length values."""
    if not isinstance(values, list) or not 1 <= len(values) <= 65536:
        raise ModelFailure("图片嵌入必须返回非空、长度受限的向量数组")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in values):
        raise ModelFailure("图片嵌入向量包含无效数字")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ModelFailure("图片嵌入向量的长度无效")
    return [float(value / norm) for value in values]


def _space(result: Dict[str, Any], dimension: int) -> Dict[str, Any]:
    fields = ("model_name", "model_identifier", "api_provider", "preprocess_version", "request_protocol_hash")
    if any(not isinstance(result.get(key), str) or not result[key].strip() for key in fields):
        raise ModelFailure("宿主图片嵌入结果缺少模型或预处理协议身份")
    fingerprint = result.get("embedding_fingerprint")
    if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get("hash"), str) or not fingerprint["hash"]:
        raise ModelFailure("宿主图片嵌入结果缺少向量空间指纹")
    if fingerprint.get("dimension") != dimension:
        raise ModelFailure("宿主图片嵌入指纹维度与实际向量不一致")
    if (fingerprint.get("model") != result["model_name"] or fingerprint.get("provider") != result["api_provider"]
            or fingerprint.get("model_identifier") != result["model_identifier"]):
        raise ModelFailure("宿主图片嵌入模型信息与指纹不一致")
    identity = {key: result[key] for key in fields}
    identity.update(dimension=dimension, embedding_fingerprint=dict(fingerprint))
    identity["space_id"] = _digest({**{key: identity[key] for key in fields}, "dimension": dimension,
                                    "fingerprint": fingerprint["hash"]})
    return identity


class VisualMemory:
    """One active browsing run, with persistent vectors across runs.

    Call begin_run before each browsing/indexing run. Its first actual request
    observes the Host's current model space; only then can cached vectors be
    reused. A different space during the same run stops the operation.
    """

    def __init__(self, llm: LLMCapability, store: Store, *, task_name: str = "image_embedding",
                 expected_model_name: str = "qwen3-vl-embedding") -> None:
        self.llm = llm
        self.store = store
        self.task_name = task_name
        self.expected_model_name = expected_model_name
        self._identity: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()

    async def begin_run(self) -> None:
        """Require a fresh model-space observation for the next run."""
        async with self._lock:
            self._identity = None

    async def embed_image(self, image: Dict[str, Any], *, work_id: str,
                          variant_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Embed downloaded pixels once per input/space and link them to a work."""
        if not isinstance(work_id, str) or not work_id:
            raise ValueError("图片嵌入需要作品编号")
        prepared = await asyncio.to_thread(prepare_vision_image, image, max_side=1024, max_bytes=700000)
        if not variant_id:
            variant_id = await asyncio.to_thread(image_content_hash, image)
        input_hash = sha256(prepared["data"]).hexdigest()
        source_hash = sha256(image["data"]).hexdigest()
        supplied_metadata = {**image, **(metadata if metadata is not None else {})}
        image_metadata = {key: supplied_metadata[key] for key in _METADATA_FIELDS
                          if isinstance(supplied_metadata.get(key), str)}
        async with self._lock:
            work_id = await asyncio.to_thread(self.store.resolve_image_id, work_id)
            if self._identity is not None:
                cache_id = _digest({"space_id": self._identity["space_id"], "input_hash": input_hash})
                cached = await asyncio.to_thread(self.store.get, RECORD_KIND, cache_id)
                if cached is not None:
                    identity = _space(cached, len(normalized_vector(cached["embedding"])))
                    if identity["space_id"] != self._identity["space_id"] or cached["space_id"] != identity["space_id"]:
                        raise ModelFailure("图片嵌入缓存的空间身份不一致")
                    return await self._associate(cached, work_id, variant_id, source_hash, image_metadata, cache_hit=True)
            result = await self.llm.embed(
                task_name=self.task_name, image_base64=base64.b64encode(prepared["data"]).decode("ascii"),
                mime_type=prepared["mime_type"], preprocess_version=PREPROCESS_VERSION, timeout_ms=180000,
            )
            if not isinstance(result, dict) or result.get("success") is not True:
                error = result.get("error", "无成功响应") if isinstance(result, dict) else "响应格式无效"
                raise ModelFailure("图片嵌入调用失败：" + str(error)[:1000])
            vector = normalized_vector(result.get("embedding"))
            identity = _space(result, len(vector))
            if identity["model_name"] != self.expected_model_name:
                raise ModelFailure("图片嵌入任务实际使用的模型与摄影插件指定模型不一致：" + identity["model_name"])
            if identity["preprocess_version"] != PREPROCESS_VERSION:
                raise ModelFailure("宿主返回的图片嵌入预处理版本不一致")
            if self._identity is not None and identity["space_id"] != self._identity["space_id"]:
                raise ModelFailure("本轮图片嵌入模型空间已改变，请重新开始这一轮")
            self._identity = identity
            cache_id = _digest({"space_id": identity["space_id"], "input_hash": input_hash})
            existing = await asyncio.to_thread(self.store.get, RECORD_KIND, cache_id)
            record = {**identity, "id": cache_id, "embedding": vector, "input_hash": input_hash,
                      "work_ids": existing["work_ids"] if existing is not None else [],
                      "variant_ids": existing["variant_ids"] if existing is not None else [],
                      "metadata_by_work": existing["metadata_by_work"] if existing is not None else {},
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "width": prepared["width"], "height": prepared["height"]}
            return await self._associate(record, work_id, variant_id, source_hash, image_metadata, cache_hit=False)

    async def _associate(self, record: Dict[str, Any], work_id: str, variant_id: str,
                         source_hash: str, metadata: Dict[str, str], *, cache_hit: bool) -> Dict[str, Any]:
        previous_metadata = record["metadata_by_work"].get(work_id, {})
        record = {**record, "work_id": work_id, "variant_id": variant_id, "source_hash": source_hash,
                  "work_ids": list(dict.fromkeys([*record["work_ids"], work_id])),
                  "variant_ids": list(dict.fromkeys([*record["variant_ids"], variant_id])),
                  "metadata_by_work": {**record["metadata_by_work"], work_id: {**previous_metadata, **metadata}}}
        saved = await asyncio.to_thread(self.store.put, RECORD_KIND, record)
        return {**saved, "cache_hit": cache_hit}

    async def search(self, query: Dict[str, Any], *, limit: int = 8,
                     exclude_work_ids: Iterable[str] = (), allowed_work_ids: Optional[Iterable[str]] = None,
                     min_similarity: float = -1.0) -> List[Dict[str, Any]]:
        """Search every vector in the same space, returning one match per work.

        Scores rank visual candidates. This method never merges work identities
        or changes aesthetic preferences based on vector similarity.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("相似作品数量必须为正整数")
        if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)) or not -1 <= min_similarity <= 1:
            raise ValueError("相似度阈值必须在 -1 至 1 之间")
        return await asyncio.to_thread(self._search, query, limit, list(exclude_work_ids),
                                       None if allowed_work_ids is None else list(allowed_work_ids), float(min_similarity))

    def _search(self, query: Dict[str, Any], limit: int, exclude_work_ids: List[str],
                allowed_work_ids: Optional[List[str]], min_similarity: float) -> List[Dict[str, Any]]:
        vector = normalized_vector(query["embedding"])
        identity = _space(query, len(vector))
        if query["space_id"] != identity["space_id"]:
            raise ModelFailure("查询向量的空间身份不一致")
        excluded = {self.store.resolve_image_id(work_id) for work_id in [query["work_id"], *exclude_work_ids]}
        allowed = None if allowed_work_ids is None else {self.store.resolve_image_id(work_id) for work_id in allowed_work_ids}
        records = self.store.list(RECORD_KIND, limit=self.store.count(RECORD_KIND))
        results: Dict[str, Dict[str, Any]] = {}
        resolved: Dict[str, str] = {}
        for record in records:
            if record["space_id"] != identity["space_id"]:
                continue
            candidate = normalized_vector(record["embedding"])
            if len(candidate) != len(vector) or _space(record, len(candidate))["space_id"] != identity["space_id"]:
                raise ModelFailure("同空间的图片嵌入缓存维度或身份不一致")
            similarity = max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(vector, candidate))))
            if similarity < min_similarity:
                continue
            for work_id in record["work_ids"]:
                if work_id not in resolved:
                    resolved[work_id] = self.store.resolve_image_id(work_id)
                canonical = resolved[work_id]
                if (canonical in excluded or allowed is not None and canonical not in allowed
                        or canonical in results and results[canonical]["similarity"] >= similarity):
                    continue
                image = self.store.get("image", canonical)
                metadata = ({key: image[key] for key in _METADATA_FIELDS if isinstance(image.get(key), str)}
                            if image is not None else record["metadata_by_work"][work_id])
                results[canonical] = {"work_id": canonical, "similarity": similarity,
                    "variant_id": record["variant_id"], "embedding_id": record["id"],
                    "space_id": identity["space_id"], "metadata": metadata}
        return sorted(results.values(), key=lambda item: (-item["similarity"], item["work_id"]))[:limit]
