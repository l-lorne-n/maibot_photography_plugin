"""Identity comes from scoped Host accounts, never from a displayed nickname."""

from pathlib import Path
from types import ModuleType
from unittest import TestCase

import importlib
import sys

PACKAGE = ModuleType("photography_identity_tests")
PACKAGE.__path__ = [str(Path(__file__).parents[1])]
sys.modules[PACKAGE.__name__] = PACKAGE
identity = importlib.import_module(PACKAGE.__name__ + ".account_identity")


class AccountIdentityTests(TestCase):
    def test_real_route_overrides_spare_accounts(self):
        bot = {"platform": "qq", "qq_account": "old", "platforms": ["qq:other", "telegram:tg"]}
        self.assertEqual(identity.self_user_ids(bot, {"platform": "qq", "account_id": "actual"}), {"actual"})

    def test_configured_accounts_are_platform_scoped_when_route_has_no_account(self):
        bot = {"platform": "qq", "qq_account": "qq-id", "platforms": ["telegram:tg-id"]}
        self.assertEqual(identity.self_user_ids(bot, {"platform": "telegram"}), {"tg-id"})
        self.assertEqual(identity.self_user_ids(bot, {"platform": "qq"}), {"qq-id"})

    def test_foreign_platform_never_reuses_group_account(self):
        message = {"platform": "telegram", "message_info": {"user_info": {"user_id": "same-number"}}}
        self.assertFalse(identity.is_self_message(message, {}, {"platform": "qq", "account_id": "same-number"}))

    def test_display_names_have_no_identity_effect(self):
        message = {"platform": "qq", "account_id": "actual",
                   "message_info": {"user_info": {"user_id": "friend", "user_nickname": "Lorne", "user_cardname": "Lorne"}}}
        self.assertFalse(identity.is_self_message(message, {"nickname": "Lorne"}))
