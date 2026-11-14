"""Exercise real SDK normalization and group boundaries without a Host or models."""

from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest import IsolatedAsyncioTestCase, TestCase

from maibot_sdk.context import PluginContext

import importlib
import json
import sys
import time


ROOT = Path(__file__).parents[1]
PACKAGE = "photography_character_tests_module"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
character = importlib.import_module(PACKAGE + ".character")


def message(identifier, user="alice", text="我喜欢这张照片的光线", stream="chosen", card=""):
    return {"message_id": identifier, "session_id": stream, "timestamp": str(time.time()),
            "is_command": False, "platform": "qq",
            "message_info": {"user_info": {"user_id": user, "user_nickname": "本群昵称", "user_cardname": card}},
            "raw_message": [{"type": "text", "data": text}], "processed_plain_text": text}


class FakeHost:
    def __init__(self):
        self.calls = []
        self.denied = set()
        self.config = {
            "personality": {"selected_profile": "目前角色", "personality": "安静但有自己的判断", "behavior_style": "自行查资料",
                            "reply_style": "日常简短", "multiple_reply_style": ["随意"], "multiple_probability": 0.2,
                            "public_extension": {"interests": ["摄影"]},
                            "profiles": [{"name": "备用角色", "personality": "备用设定", "private_chat_prompts": "PRIVATE PROFILE RULE"}]},
            "bot": {"nickname": "Lorne", "alias_names": ["洛恩"], "platform": "qq", "qq_account": "bot",
                    "platforms": [], "unexpected_secret": "DO NOT EXPOSE"},
            "chat.reply_style.group_chat_prompt": "自然接话",
            "chat.reply_style.chat_prompts": [
                {"platform": "qq", "item_id": "group1", "rule_type": "group", "prompt": "本群额外要求"},
                {"platform": "qq", "item_id": "another", "rule_type": "group", "prompt": "OTHER GROUP RULE"},
                {"platform": "qq", "item_id": "group1", "rule_type": "private", "prompt": "PRIVATE RULE"},
                {"platform": "telegram", "item_id": "group1", "rule_type": "group", "prompt": "OTHER PLATFORM RULE"},
            ],
        }
        self.streams = [{"session_id": "chosen", "platform": "qq", "group_id": "group1", "group_name": "摄影讨论",
                         "is_group_session": True},
                        {"session_id": "private", "platform": "qq", "user_id": "alice", "is_group_session": False}]
        self.messages = [message("m1", card="群内称呼"), message("m2", user="bot", text="这张我也挺喜欢")]
        self.people = {"alice": {"person_id": "person-alice", "is_known": True, "person_name": "熟悉的称呼"}}

    async def rpc(self, method, plugin_id, payload, timeout_ms=None):
        assert method == "cap.call"
        capability, args = payload["capability"], payload["args"]
        self.calls.append((capability, deepcopy(args)))
        if capability in self.denied:
            raise PermissionError("Access denied")
        if capability == "config.get":
            return {"success": True, "value": deepcopy(self.config.get(args["key"]))}
        if capability == "chat.get_all_streams":
            return {"success": True, "streams": deepcopy(self.streams)}
        if capability == "message.get_recent":
            assert args == {"chat_id": "chosen", "limit": 24}
            return {"success": True, "messages": deepcopy(self.messages)}
        if capability == "person.get_id":
            return {"success": True, "person_id": self.people.get(args["user_id"], {}).get("person_id", "")}
        if capability == "person.get_value":
            person = next(person for person in self.people.values() if person["person_id"] == args["person_id"])
            return {"success": True, "value": person[args["field_name"]]}
        raise AssertionError("Unexpected capability: " + capability)

    def context(self):
        return PluginContext("lorne.photography", rpc_call=self.rpc)


class CharacterContextTests(IsolatedAsyncioTestCase):
    async def test_adapter_account_excludes_self_without_spare_bot_configuration(self):
        host = FakeHost()
        host.config["bot"].update(platform="", qq_account="", platforms=[])
        host.streams[0]["account_id"] = "actual-bot"
        host.messages = [message("friend", user="alice"), message("own", user="actual-bot", text="前两张摄影作品是我的分享")]
        for item in host.messages:
            item["account_id"] = "actual-bot"
        # The same nickname does not make a friend the bot.
        host.messages[0]["message_info"]["user_info"]["user_nickname"] = "Lorne"
        result = await character.load_character_context(host.context(), "chosen")
        self.assertEqual(result["scope"]["account_id"], "actual-bot")
        self.assertEqual([item["speaker_id"] for item in result["recent_conversation"] if item["is_self"]], ["actual-bot"])
        self.assertEqual([item["user_id"] for item in result["friends"]], ["alice"])
        self.assertFalse(any(capability == "person.get_id" and args["user_id"] == "actual-bot" for capability, args in host.calls))

    async def test_actual_message_route_precedes_stale_spare_account(self):
        host = FakeHost()
        host.streams[0]["account_id"] = "actual-bot"
        host.messages = [message("old-spare", user="bot"), message("own", user="actual-bot")]
        for item in host.messages:
            item["account_id"] = "actual-bot"
        result = await character.load_character_context(host.context(), "chosen")
        self.assertEqual([item["speaker_id"] for item in result["recent_conversation"] if item["is_self"]], ["actual-bot"])
        self.assertEqual([item["user_id"] for item in result["friends"]], ["bot"])

    async def test_preserves_public_personality_and_scoped_group_rules(self):
        host = FakeHost()
        result = await character.load_character_context(host.context(), "chosen")
        self.assertEqual(result["personality"]["public_extension"], {"interests": ["摄影"]})
        self.assertEqual(result["personality"]["multiple_reply_style"], ["随意"])
        self.assertEqual(result["personality"]["scoped_chat_prompts"], [host.config["chat.reply_style.chat_prompts"][0]])
        self.assertEqual(result["friends"][0]["name"], "群内称呼")
        self.assertEqual(result["friends"][0]["known_name"], "熟悉的称呼")
        self.assertEqual(result["scope"]["name"], "摄影讨论")
        self.assertEqual(result["availability"]["counts"], {"friends": 1, "messages": 2})
        encoded = json.dumps(result)
        for excluded in ("PRIVATE", "OTHER GROUP", "OTHER PLATFORM", "DO NOT EXPOSE"):
            self.assertNotIn(excluded, encoded)

    async def test_no_group_means_no_chat_or_person_requests(self):
        host = FakeHost()
        result = await character.load_character_context(host.context(), "")
        self.assertTrue(all(capability == "config.get" for capability, _ in host.calls))
        self.assertEqual(result["friends"], [])
        self.assertEqual(result["recent_conversation"], [])
        self.assertEqual(result["availability"]["components"]["group_scope"]["status"], "not_selected")

    async def test_private_or_missing_stream_is_not_read(self):
        for stream in ("private", "missing"):
            host = FakeHost()
            result = await character.load_character_context(host.context(), stream)
            self.assertNotIn("message.get_recent", [capability for capability, _ in host.calls])
            self.assertIn("selected_group_not_found", result["availability"]["error"])

    async def test_foreign_scope_quotes_and_forwarded_text_are_excluded(self):
        host = FakeHost()
        local = message("local", text="这张摄影的阴影很好")
        local["raw_message"].extend([
            {"type": "reply", "data": {"text": "PRIVATE QUOTATION"}},
            {"type": "forward", "data": [{"type": "text", "data": "PRIVATE FORWARD"}]},
        ])
        local["processed_plain_text"] += " PRIVATE RENDERED QUOTATION"
        host.messages = [message("foreign", stream="other", text="PRIVATE OTHER GROUP PHOTO"), local]
        result = await character.load_character_context(host.context(), "chosen")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual([item["message_id"] for item in result["recent_conversation"]], ["local"])

    async def test_relationship_quotes_are_evidence_without_inferred_label(self):
        host = FakeHost()
        host.messages = [message("relationship", text="朋友推荐的摄影集不错"), message("taste", text="这张的颜色我喜欢")]
        result = await character.load_character_context(host.context(), "chosen")
        friend = result["friends"][0]
        self.assertIsNone(friend["relationship"])
        self.assertEqual(friend["relationship_status"], "not_inferred")
        self.assertEqual(friend["relationship_evidence"][0]["message_id"], "relationship")
        self.assertEqual(result["availability"]["components"]["long_term_relationships"]["status"], "not_loaded")
        self.assertFalse(any(capability.startswith(("api.", "llm.", "tool.", "database.")) for capability, _ in host.calls))

    async def test_denied_configuration_is_explicit_not_fabricated(self):
        host = FakeHost()
        host.denied.add("config.get")
        result = await character.load_character_context(host.context(), "")
        self.assertEqual(result["personality"], {})
        self.assertEqual(result["bot"], {})
        self.assertEqual(result["availability"]["status"], "unavailable")
        self.assertIn("PermissionError", result["availability"]["error"])

    async def test_unknown_people_use_only_actual_group_name_and_limits(self):
        host = FakeHost()
        host.people.clear()
        host.messages = [message(f"m{index}", user=f"u{index}", text="摄影不错", card=f"名片{index}") for index in range(9)]
        result = await character.load_character_context(host.context(), "chosen")
        self.assertEqual(len(result["friends"]), 5)
        self.assertTrue(all(item["name"].startswith("名片") and item["known_name"] == "" for item in result["friends"]))
        self.assertNotIn("person.get_value", [capability for capability, _ in host.calls])


class PromptContractTests(TestCase):
    def test_all_languages_define_executor_purposes_and_stable_work_ids(self):
        prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
        purposes = {"guide", "decision", "preview", "vision", "reflection", "diary_rewrite", "fact_check", "feedback", "share_commentary"}
        self.assertEqual(set(prompts), {"zh-CN", "en", "ja"})
        for locale, values in prompts.items():
            self.assertEqual(set(values), purposes, locale)
            self.assertIn("search|open|preview|finish", values["decision"])
            self.assertIn("matched_work_ids", values["vision"])
            self.assertIn("character_context.personality", values["guide"])
            self.assertIn("source_notes", values["reflection"])
