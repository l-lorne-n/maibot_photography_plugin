"""Associate photography discussion with explicit evidence or a recent shared batch."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import hashlib
import math
import re
import unicodedata

from .account_identity import is_self_message

_IMAGE_REFERENCE = re.compile(r"(?<![a-fA-F0-9])([a-fA-F0-9]{8,64})(?![a-fA-F0-9])")
_PHOTO_DISCUSSION = re.compile(r"摄影|照片|图片|影集|画面|构图|色彩|颜色|光线|光影|黑白|彩色|作品|相机|镜头|快门|景深|曝光|胶片|取景|分辨率|清晰|模糊|糊|质感|风格|欧美|好看|喜欢|更好|氛围|舒服|安静|拍的|谁拍|这张|那张|几张|这些|这组|photograph|photo|picture|lighting|composition|colou?r", re.I)
_ORDINAL = re.compile(r"第([一二三四五六七八九十]|\d{1,2})张")
_RECENT_SECONDS = 45 * 60


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _mentions(text: str, label: str) -> bool:
    label = _normalize_text(label)
    if not label:
        return False
    start = r"(?<![a-z0-9])" if label[0].isascii() and label[0].isalnum() else ""
    end = r"(?![a-z0-9])" if label[-1].isascii() and label[-1].isalnum() else ""
    return re.search(start + re.escape(label) + end, text) is not None


def _own_text(message: Dict[str, Any]) -> str:
    # Host's processed_plain_text also contains rendered quotes and forwarded
    # messages. Only top-level text components belong to this speaker.
    if "raw_message" in message:
        return " ".join(
            component["data"]
            for component in message["raw_message"]
            if component["type"] == "text"
        ).strip()
    return message.get("processed_plain_text", "").strip()


def has_photography_topic(message: Dict[str, Any], images: List[Dict[str, Any]]) -> bool:
    """Identify direct topic anchors, without turning a follow-up into a new anchor."""
    text = _own_text(message)
    normalized = _normalize_text(text)
    names = [str(record.get("author", "")) for record in images]
    names += [name.split()[-1] for name in names if len(name.split()) > 1 and len(name.split()[-1]) >= 4]
    return bool(_PHOTO_DISCUSSION.search(text) or any(_mentions(normalized, name) for name in names))


def associate_message(
    message: Dict[str, Any],
    shares: List[Dict[str, Any]],
    images: List[Dict[str, Any]],
    target_stream_id: str,
    share_jobs: Optional[List[Dict[str, Any]]] = None,
    self_user_ids: Optional[Set[str]] = None,
    recent_discussion: bool = False,
) -> Optional[Dict[str, Any]]:
    """Keep strong references exact; loose discussion gets candidates, never guessed scores."""
    if (not target_stream_id or message.get("session_id") != target_stream_id
            or message.get("is_command") or message.get("is_notify")):
        return None
    text = _own_text(message)
    if not text:
        return None
    message_id = message["message_id"]
    speaker_id = message["message_info"]["user_info"]["user_id"]
    if not isinstance(message_id, str) or not message_id or not isinstance(speaker_id, str) or not speaker_id:
        raise ValueError("Feedback requires the Host's message id and speaker user id")
    if is_self_message(message) or speaker_id in (self_user_ids or set()):
        return None
    timestamp = float(message["timestamp"])
    if not math.isfinite(timestamp):
        raise ValueError("Host message timestamp must be finite Unix seconds")

    known_images = {record["id"]: record for record in images}
    reply_ids: Set[str] = set()
    if message.get("reply_to"):
        reply_ids.add(message["reply_to"])
    # The current inbound constructor does not populate top-level reply_to;
    # PluginMessageUtils serializes actual ReplyComponents in this form.
    for component in message.get("raw_message", []):
        if component["type"] == "reply":
            data = component["data"]
            reply_ids.add(data["target_message_id"] if isinstance(data, dict) else str(data))
    eligible = [share for share in shares if share["stream_id"] == target_stream_id
                and share.get("message_id") and share.get("sent") is not False and share["image_id"] in known_images]
    quoted = {
        share["image_id"] for share in eligible if share["message_id"] in reply_ids
    }
    referenced: Set[str] = set()
    for reference in _IMAGE_REFERENCE.findall(text):
        reference = reference.lower()
        referenced.update(image_id for image_id, record in known_images.items()
                          if any(identifier.startswith(reference) for identifier in record.get("reference_ids", [image_id])))
    visual = {image_id for component in message.get("raw_message", []) if component["type"] == "image"
              for image_id, record in known_images.items()
              if component.get("hash") and component["hash"] in {image_id, record.get("source_sha256"), *record.get("reference_ids", [])}}
    if quoted and referenced and quoted != referenced:
        return None
    explicit = quoted | referenced | visual
    dated = sorted([share for share in eligible if share.get("created_at")
                    and 0 <= timestamp - _date(share["created_at"]).timestamp() <= _RECENT_SECONDS],
                   key=lambda share: _date(share["created_at"]))
    share_job_id = dated[-1].get("job_id", "") if dated else ""
    batch = [share for share in dated if share.get("job_id", "") == share_job_id]
    batch_ids = list(dict.fromkeys(share["image_id"] for share in batch))
    candidates: List[str] = []
    image_id = ""
    scope = "unresolved"
    if explicit:
        candidates = sorted(explicit)
        image_id = candidates[0] if len(candidates) == 1 else ""
        scope = "image" if image_id else "batch"
        association = "reply_to" if quoted else "image_id" if referenced else "image_hash"
        associated = [share for share in eligible if share["image_id"] in explicit]
        if associated:
            share_job_id = associated[0].get("job_id", "")
    else:
        normalized = _normalize_text(text)
        named = {
            image_id for image_id, record in known_images.items()
            if record.get("author") and record.get("work")
            and _mentions(normalized, record["author"])
            and _mentions(normalized, record["work"])
        }
        if len(named) == 1:
            candidates, association = [next(iter(named))], "author_work"
        else:
            reply_job = next((job for job in share_jobs or [] if job.get("share_target_stream_id") == target_stream_id
                              and job.get("share_commentary_message_id") in reply_ids), None)
            if reply_job:
                share_job_id = reply_job["id"]
                candidates = list(dict.fromkeys(share["image_id"] for share in eligible if share.get("job_id") == share_job_id))
                association = "batch_reply"
            elif batch_ids:
                ordinals = _ORDINAL.findall(text)
                numbers = [int(value) if value.isdecimal() else "一二三四五六七八九十".index(value) + 1 for value in ordinals]
                if numbers and all(1 <= number <= len(batch_ids) for number in numbers):
                    candidates = list(dict.fromkeys(batch_ids[number - 1] for number in numbers))
                    image_id = candidates[0] if len(candidates) == 1 else ""
                    scope, association = ("image" if image_id else "batch"), "ordinal"
                else:
                    author_terms = [str(known_images[ident].get("author", "")) for ident in batch_ids]
                    author_terms += [term.split()[-1] for term in author_terms if len(term.split()) > 1 and len(term.split()[-1]) >= 4]
                    if not _PHOTO_DISCUSSION.search(text) and not any(_mentions(normalized, term) for term in author_terms):
                        if not recent_discussion:
                            return None
                        candidates, association = batch_ids, "conversation_followup"
                    else:
                        candidates, association = batch_ids, "recent_batch"
            else:
                author_terms = [str(record.get("author", "")) for record in images]
                author_terms += [term.split()[-1] for term in author_terms if len(term.split()) > 1 and len(term.split()[-1]) >= 4]
                if _PHOTO_DISCUSSION.search(text) or any(_mentions(normalized, term) for term in author_terms):
                    association = "general_topic"
                elif recent_discussion:
                    association = "conversation_followup"
                else:
                    return None
    if not candidates and association not in {"general_topic", "conversation_followup"}:
        return None
    identity = "\0".join((target_stream_id, message_id))
    return {
        "id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "image_id": image_id,
        "image_ids": [image_id] if image_id else [],
        "candidate_image_ids": candidates,
        "scope": scope,
        "share_job_id": share_job_id,
        "stream_id": target_stream_id,
        "message_id": message_id,
        "speaker_id": speaker_id,
        "platform": message.get("platform", ""),
        "account_id": message.get("account_id", ""),
        "user_name": (message["message_info"]["user_info"].get("user_cardname")
                      or message["message_info"]["user_info"].get("user_nickname") or "朋友"),
        "text": text,
        "association": association,
        "created_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
    }


def _score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Feedback preference must be a number between 0 and 1")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Feedback preference must be a number between 0 and 1")
    return score


def _date(value: str) -> datetime:
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Feedback timestamps require a timezone")
    return date.astimezone(timezone.utc)


def interpret_feedback(record: Dict[str, Any], judgement: Dict[str, Any]) -> Dict[str, Any]:
    """Allow a model to resolve candidate scope without rewriting speaker evidence."""
    scope = judgement.get("scope")
    if scope not in {"image", "batch", "general", "unrelated"}:
        raise ValueError("反馈模型必须明确区分单图、整批、泛泛讨论或无关消息")
    if not isinstance(judgement.get("is_evaluation"), bool):
        raise ValueError("反馈模型缺少明确的评价判断")
    candidates = record.get("candidate_image_ids", [record["image_id"]] if record.get("image_id") else [])
    anchored = record.get("image_id", "") if record.get("association") != "recent_batch" else ""
    image_id = ""
    image_ids: List[str] = []
    if scope == "image":
        image_id = anchored or judgement.get("image_id", "")
        if not image_id or image_id not in candidates:
            raise ValueError("反馈模型选择了未提供的图片，不能登记单图偏好")
        image_ids = [image_id]
    elif scope == "batch":
        image_ids = judgement.get("image_ids", candidates)
        if not isinstance(image_ids, list) or not image_ids or any(ident not in candidates for ident in image_ids):
            raise ValueError("整批讨论的图片范围必须来自提供的候选")
        image_ids = list(dict.fromkeys(image_ids))
    interest = judgement.get("human_interest")
    if interest is not None:
        interest = _score(interest)
    is_evaluation = judgement["is_evaluation"] and scope != "unrelated"
    if not is_evaluation or scope in {"general", "unrelated"}:
        interest = None
    notes = {}
    for key in ("reason", "insight"):
        value = judgement.get(key, "")
        if not isinstance(value, str):
            raise ValueError("反馈模型的讨论依据必须是文字")
        notes[key] = value[:2000]
    topics = judgement.get("topics", [])
    if not isinstance(topics, list) or any(not isinstance(topic, str) for topic in topics):
        raise ValueError("反馈模型的摄影话题必须为文字数组")
    names = {}
    for key in ("photographer", "work"):
        value = judgement.get(key, "")
        if not isinstance(value, str):
            raise ValueError("反馈模型的摄影师/作品线索必须为文字")
        names[key] = value[:300]
    return {**record, "scope": scope, "image_id": image_id or anchored, "image_ids": image_ids,
            "is_evaluation": is_evaluation, "human_interest": interest, **notes,
            "is_photography_related": scope != "unrelated", "topics": topics[:12], **names,
            "model": judgement.get("model", ""), "status": "ignored" if scope == "unrelated" else "evaluated"}


def apply_feedback(image: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one explicit evaluation, using a neutral prior and a small EWMA step."""
    result = deepcopy(image)
    if not isinstance(feedback["is_evaluation"], bool):
        raise ValueError("Feedback is_evaluation must be a boolean")
    if not feedback["is_evaluation"]:
        return result
    if "image_id" in feedback and feedback["image_id"] != image["id"]:
        raise ValueError("Feedback belongs to another image")
    feedback_id = feedback["id"]
    speaker_id = feedback["speaker_id"]
    if not isinstance(feedback_id, str) or not feedback_id or not isinstance(speaker_id, str) or not speaker_id:
        raise ValueError("Feedback requires a nonempty evidence id and speaker id")
    preferences = result.get("human_preferences", {})
    if any(feedback_id in preference["evidence_ids"] for preference in preferences.values()):
        return result
    observed = _score(feedback["human_interest"])
    occurred = _date(feedback["created_at"])
    previous = preferences.get(speaker_id, {"score": 0.5, "evidence_ids": [], "evidence": []})
    preference = deepcopy(previous)
    preference["score"] = round(0.8 * _score(previous["score"]) + 0.2 * observed, 8)
    preference["evidence_ids"].append(feedback_id)
    preference["evidence"].append({
        "id": feedback_id,
        "message_id": feedback.get("message_id"),
        "score": observed,
        "reason": feedback.get("reason", ""),
        "insight": feedback.get("insight", ""),
        "created_at": occurred.isoformat(),
    })
    preference["updated_at"] = occurred.isoformat()
    preferences[speaker_id] = preference
    result["human_preferences"] = preferences
    result["human_interest"] = max(_score(item["score"]) for item in preferences.values())
    latest = max(_date(result["last_interest_at"]), occurred)
    result["last_interest_at"] = latest.isoformat()
    return result
