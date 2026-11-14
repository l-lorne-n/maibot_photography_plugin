"""Resolve the bot's own public account identity for one Host message route."""

from typing import Any, Dict, Optional, Set


def self_user_ids(bot: Dict[str, Any], route: Dict[str, Any]) -> Set[str]:
    """Adapter-provided account_id takes precedence over configured spare accounts."""
    account_id = route.get("account_id")
    if account_id:
        return {str(account_id).strip()}
    primary_platform = str(bot.get("platform") or "qq").strip().lower()
    platform = str(route.get("platform") or primary_platform).strip().lower()
    identities: Set[str] = set()
    configured = str(bot.get("qq_account") or "").strip()
    if platform == primary_platform and configured and configured != "0":
        identities.add(configured)
    for account in bot.get("platforms", []):
        account_platform, separator, identifier = account.partition(":")
        if separator and account_platform.strip().lower() == platform and identifier.strip():
            identities.add(identifier.strip())
    return identities


def is_self_message(message: Dict[str, Any], bot: Optional[Dict[str, Any]] = None,
                    stream: Optional[Dict[str, Any]] = None) -> bool:
    """Compare actual user/account IDs; display names never establish identity."""
    route = dict(stream or {})
    if message.get("platform"):
        if route.get("platform") and route["platform"].strip().lower() != message["platform"].strip().lower():
            route.pop("account_id", None)
        route["platform"] = message["platform"]
    if message.get("account_id"):
        route["account_id"] = message["account_id"]
    speaker_id = message.get("message_info", {}).get("user_info", {}).get("user_id")
    return bool(speaker_id and str(speaker_id).strip() in self_user_ids(bot or {}, route))
