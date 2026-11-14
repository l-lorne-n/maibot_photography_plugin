from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from unittest import TestCase

import importlib
import sys
import unittest


_PACKAGE = ModuleType("photography_feedback")
_PACKAGE.__path__ = [str(Path(__file__).parents[1])]
sys.modules[_PACKAGE.__name__] = _PACKAGE
_MODULE = importlib.import_module(_PACKAGE.__name__ + ".feedback")
associate_message = _MODULE.associate_message
apply_feedback = _MODULE.apply_feedback
interpret_feedback = _MODULE.interpret_feedback


class FeedbackTests(TestCase):
    def setUp(self) -> None:
        self.image = {
            "id": "abcdef12" + "a" * 56, "author": "川内伦子", "work": "Illuminance",
            "bot_interest": 0.8, "human_interest": 0.5, "last_interest_at": "2026-11-11T08:00:00+00:00",
        }
        self.other = {**self.image, "id": "1234abcd" + "b" * 56, "author": "Robert Adams", "work": "The New West"}
        self.images = [self.image, self.other]
        self.shares = [{"stream_id": "chosen-group", "message_id": "sent-1", "image_id": self.image["id"]}]

    def message(self, text: str, **updates: Any) -> Dict[str, Any]:
        return {
            "session_id": "chosen-group", "message_id": "incoming-1", "timestamp": "1794470400.0",
            "platform": "qq", "message_info": {"user_info": {"user_id": "person-1", "user_nickname": "Alice"}},
            "raw_message": [{"type": "text", "data": text}], "processed_plain_text": text, **updates,
        }

    def feedback(self, **updates: Any) -> Dict[str, Any]:
        return {
            "id": "feedback-1", "image_id": self.image["id"], "message_id": "incoming-1", "speaker_id": "alice",
            "is_evaluation": True, "human_interest": 1.0, "reason": "喜欢光线", "insight": "柔和的色彩",
            "created_at": "2026-11-12T08:00:00+00:00", **updates,
        }

    def test_stream_is_required_and_vague_praise_is_unassociated(self) -> None:
        self.assertEqual(associate_message(self.message("好看"), self.shares, self.images, "chosen-group")["candidate_image_ids"], [])
        self.assertIsNone(associate_message(self.message("abcdef12", session_id="another"), self.shares, self.images, "chosen-group"))
        self.assertIsNone(associate_message(self.message("abcdef12"), self.shares, self.images, ""))

    def test_actual_message_account_excludes_self_without_cached_character(self) -> None:
        own = self.message("喜欢这张照片", account_id="person-1")
        self.assertIsNone(associate_message(own, self.shares, self.images, "chosen-group"))
        friend = self.message("喜欢这张照片", account_id="actual-bot")
        self.assertEqual(associate_message(friend, self.shares, self.images, "chosen-group")["speaker_id"], "person-1")

    def test_reply_to_receipt_is_deterministic_and_uses_nested_host_speaker(self) -> None:
        message = self.message("好看", reply_to="sent-1")
        result = associate_message(message, self.shares, self.images, "chosen-group")
        self.assertEqual(result["image_id"], self.image["id"])
        self.assertEqual(result["speaker_id"], "person-1")
        self.assertEqual(result["association"], "reply_to")
        self.assertEqual(result, associate_message(message, self.shares, self.images, "chosen-group"))
        cross_stream = [{**self.shares[0], "stream_id": "another-group"}]
        self.assertEqual(associate_message(message, cross_stream, self.images, "chosen-group")["image_id"], "")

    def test_actual_raw_reply_component_excludes_quote_and_forward_content(self) -> None:
        message = self.message("我觉得一般")
        message["raw_message"].insert(0, {
            "type": "reply", "data": {"target_message_id": "sent-1", "target_message_content": "我特别喜欢"},
        })
        message["raw_message"].append({"type": "forward", "data": [{"text": "我特别喜欢"}]})
        message["processed_plain_text"] = "[回复了某人的消息：我特别喜欢] 我觉得一般 [转发：我特别喜欢]"
        result = associate_message(message, self.shares, self.images, "chosen-group")
        self.assertEqual(result["text"], "我觉得一般")
        self.assertEqual(result["image_id"], self.image["id"])
        message["raw_message"] = message["raw_message"][:1]
        self.assertIsNone(associate_message(message, self.shares, self.images, "chosen-group"))

    def test_id_full_or_unique_eight_character_prefix_and_conflicts(self) -> None:
        for reference in (self.image["id"], "ABCDEF12"):
            result = associate_message(self.message(f"照片 {reference} 喜欢"), self.shares, self.images, "chosen-group")
            self.assertEqual(result["association"], "image_id")
        self.assertEqual(associate_message(self.message("abcdef1 喜欢"), self.shares, self.images, "chosen-group")["image_id"], "")
        collision = {**self.other, "id": "abcdef12" + "c" * 56}
        self.assertEqual(associate_message(self.message("abcdef12 喜欢"), self.shares, [self.image, collision], "chosen-group")["scope"], "batch")
        self.assertIsNone(associate_message(self.message("1234abcd 喜欢", reply_to="sent-1"), self.shares, self.images, "chosen-group"))
        self.assertEqual(associate_message(self.message("abcdef12 和 1234abcd 喜欢"), self.shares, self.images, "chosen-group")["scope"], "batch")

    def test_author_and_work_required_and_unique(self) -> None:
        result = associate_message(self.message("川内伦子的 Illuminance 很温柔"), self.shares, self.images, "chosen-group")
        self.assertEqual(result["association"], "author_work")
        self.assertEqual(associate_message(self.message("川内伦子很好"), self.shares, self.images, "chosen-group")["image_id"], "")
        self.assertIsNone(associate_message(self.message("Illuminance 很好"), self.shares, self.images, "chosen-group"))
        same_book = {**self.image, "id": "fedcba98" + "d" * 56}
        self.assertEqual(associate_message(self.message("川内伦子 Illuminance 很好"), self.shares, [self.image, same_book], "chosen-group")["image_id"], "")

    def test_recent_batch_ordinals_visual_hash_and_staleness(self) -> None:
        shares = [{**self.shares[0], "created_at": "2026-11-12T07:50:00+00:00", "job_id": "batch"},
                  {"stream_id": "chosen-group", "message_id": "sent-2", "image_id": self.other["id"],
                   "created_at": "2026-11-12T07:50:03+00:00", "job_id": "batch"}]
        natural = associate_message(self.message("感觉这些更好看一点"), shares, self.images, "chosen-group")
        self.assertEqual(natural["association"], "recent_batch")
        self.assertEqual(natural["image_id"], "")
        self.assertEqual(natural["candidate_image_ids"], [self.image["id"], self.other["id"]])
        ordinal = associate_message(self.message("第二张颜色很好看"), shares, self.images, "chosen-group")
        self.assertEqual(ordinal["image_id"], self.other["id"])
        visual = self.message("这张好看")
        visual["raw_message"].append({"type": "image", "data": "", "hash": self.image["id"]})
        self.assertEqual(associate_message(visual, shares, self.images, "chosen-group")["association"], "image_hash")
        stale = associate_message(self.message("这组好看", timestamp=str(1794470400 + 3600)), shares, self.images, "chosen-group")
        self.assertEqual(stale["candidate_image_ids"], [])
        self.assertIsNone(associate_message(self.message("照片好看"), shares, self.images, "chosen-group", self_user_ids={"person-1"}))

    def test_general_photography_needs_no_shared_or_collected_image(self) -> None:
        record = associate_message(self.message("我最近想看布列松的街头摄影，黑白构图很有意思"), [], [], "chosen-group")
        self.assertEqual(record["candidate_image_ids"], [])
        result = interpret_feedback(record, {"scope": "general", "is_evaluation": False, "human_interest": None,
            "insight": "朋友想探索街头摄影的黑白构图", "topics": ["黑白", "街头摄影"], "photographer": "布列松"})
        self.assertTrue(result["is_photography_related"])
        self.assertEqual(result["image_ids"], [])
        self.assertEqual(result["photographer"], "布列松")
        with self.assertRaises(ValueError):
            interpret_feedback(record, {"scope": "image", "image_id": "invented", "is_evaluation": True, "human_interest": 1})

    def test_batch_human_preference_does_not_become_per_image_or_bot_score(self) -> None:
        record = {**self.feedback(), "image_id": "", "candidate_image_ids": [self.image["id"], self.other["id"]], "association": "recent_batch"}
        result = interpret_feedback(record, {"scope": "batch", "image_ids": [self.image["id"], self.other["id"]],
            "is_evaluation": True, "human_interest": 0.9, "insight": "朋友偏好这一组", "bot_interest": 0.1})
        self.assertEqual(result["image_id"], "")
        self.assertNotIn("bot_interest", result)

    def test_per_person_small_updates_preserve_disagreement_and_bot_interest(self) -> None:
        original = deepcopy(self.image)
        first = apply_feedback(self.image, self.feedback())
        self.assertAlmostEqual(first["human_interest"], 0.6)
        self.assertEqual(self.image, original)
        other = apply_feedback(first, self.feedback(id="feedback-2", speaker_id="bob", human_interest=0))
        self.assertAlmostEqual(other["human_preferences"]["bob"]["score"], 0.4)
        self.assertAlmostEqual(other["human_interest"], 0.6)
        self.assertEqual(other["bot_interest"], 0.8)
        repeated = apply_feedback(other, self.feedback(id="feedback-3", human_interest=1))
        self.assertAlmostEqual(repeated["human_preferences"]["alice"]["score"], 0.68)
        self.assertEqual(len(repeated["human_preferences"]["alice"]["evidence"]), 2)

    def test_duplicate_and_non_evaluations_do_not_renew_interest(self) -> None:
        first = apply_feedback(self.image, self.feedback())
        duplicate = apply_feedback(first, self.feedback(created_at="2026-11-13T08:00:00+00:00"))
        self.assertEqual(first, duplicate)
        non_evaluation = apply_feedback(first, self.feedback(id="question", is_evaluation=False, human_interest=0))
        self.assertEqual(first, non_evaluation)
        old = apply_feedback(first, self.feedback(id="older", speaker_id="bob", created_at="2026-11-10T08:00:00+00:00"))
        self.assertEqual(old["last_interest_at"], first["last_interest_at"])

    def test_invalid_model_scores_and_cross_image_feedback_fail(self) -> None:
        for value in (2, -0.1, float("nan"), "喜欢", True):
            with self.assertRaises(ValueError):
                apply_feedback(self.image, self.feedback(human_interest=value))
        with self.assertRaises(ValueError):
            apply_feedback(self.image, self.feedback(is_evaluation="true"))
        with self.assertRaises(ValueError):
            apply_feedback(self.image, self.feedback(image_id=self.other["id"]))


if __name__ == "__main__":
    unittest.main()
