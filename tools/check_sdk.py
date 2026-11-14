"""Exercise the installed SDK and Host validators without connecting to a live Host.

Run with the Host's Python environment. All persistence and Host import logs go
to a temporary instance directory; only the concise JSON report is retained.
"""

from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Dict, List, Optional, Set

from maibot_sdk.context import PluginContext, PluginPaths

import argparse
import ast
import asyncio
import faulthandler
import importlib
import json
import os
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_plugin_module() -> Any:
    """Use a package namespace so local imports behave as in the Runner."""
    name = "photography_sdk_validation"
    package = ModuleType(name)
    package.__path__ = [str(ROOT)]
    sys.modules[name] = package
    return importlib.import_module(name + ".plugin")


def referenced_capabilities() -> Set[str]:
    capabilities: Set[str] = set()
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            parts: List[str] = []
            value = node.func
            while isinstance(value, ast.Attribute):
                parts.insert(0, value.attr)
                value = value.value
            if "ctx" in parts:
                parts = parts[parts.index("ctx") + 1:]
                if len(parts) >= 2 and parts[0] != "logger":
                    capabilities.add(".".join(parts))
    return capabilities


def validate_host_declarations(host_root: Path) -> Dict[str, Any]:
    # Import only validators, after main() sets MAIBOT_INSTANCE_DIR to scratch.
    sys.path.insert(0, str(host_root))
    from src.plugin_runtime.runner.manifest_validator import ManifestValidator, PluginManifest
    from src.plugin_runtime.webui_schema import load_webui_extension

    raw = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    strict = PluginManifest.model_validate(raw)
    with (host_root / "pyproject.toml").open("rb") as stream:
        host_version = tomllib.load(stream)["project"]["version"]
    validator = ManifestValidator(
        host_version=host_version, sdk_version=version("maibot-plugin-sdk"),
        project_root=host_root, log_errors=False, log_compat_warnings=False,
        force_plugin_compatibility=False,
    )
    manifest = validator.load_from_plugin_path(ROOT)
    require(manifest is not None, "Host rejected manifest: " + "; ".join(validator.errors))
    extension = load_webui_extension(str(ROOT))
    require(extension is not None, "Host did not load webui.json")
    available = {}
    for path in (host_root / "plugins").glob("*/_manifest.json"):
        installed = json.loads(path.read_text(encoding="utf-8"))
        available[installed["id"]] = installed["version"]
    missing = validator.get_unsatisfied_plugin_dependencies(strict, available)
    source_calls = referenced_capabilities()
    require(source_calls.issubset(set(raw["capabilities"])),
            "Undeclared capability calls: " + ", ".join(sorted(source_calls - set(raw["capabilities"]))))
    return {
        "host_version": host_version, "sdk_version": version("maibot-plugin-sdk"),
        "strict_manifest": "passed", "runtime_manifest": "passed", "webui_declaration": "passed",
        "warnings": list(validator.warnings), "installed_plugin_dependencies_satisfied": not missing,
        "missing_installed_plugin_dependencies": missing,
        "referenced_capabilities": sorted(source_calls),
    }


class FakeHost:
    """Return actual Host capability envelopes; reject every non-test operation."""

    def __init__(self, manifest: Dict[str, Any]) -> None:
        self.manifest = manifest
        self.calls: List[str] = []
        self.model_entered = asyncio.Event()
        self.hold_model = asyncio.Event()

    async def rpc(
        self, method: str, plugin_id: str, payload: Dict[str, Any], timeout_ms: Optional[int] = None,
    ) -> Any:
        require(method == "cap.call", "Unexpected raw Host method: " + method)
        require(plugin_id == self.manifest["id"], "SDK used the wrong plugin identity")
        capability = payload["capability"]
        self.calls.append(capability)
        require(capability in self.manifest["capabilities"], "Undeclared capability: " + capability)
        if capability == "config.get":
            values = {"personality": {"personality": "独立而温和，喜欢摄影", "profiles": []},
                      "bot": {"nickname": "摄影测试", "qq_account": "test-bot"},
                      "chat.reply_style.group_chat_prompt": "自然地与朋友相处",
                      "chat.reply_style.chat_prompts": []}
            return {"success": True, "value": values[payload["args"]["key"]]}
        if capability == "chat.get_all_streams":
            require(payload["args"] == {"platform": "all_platforms"}, "Incorrect group lookup arguments")
            # src/plugin_runtime/capabilities/data.py returns this envelope;
            # the SDK extracts streams before PhotographyPlugin sees it.
            return {"success": True, "streams": [
                {"session_id": "real-group-stream", "platform": "qq", "is_group_session": True,
                 "group_id": "10001", "group_name": "摄影讨论群"},
                {"session_id": "real-private-stream", "platform": "qq", "is_group_session": False,
                 "group_id": None, "group_name": None},
            ]}
        if capability == "llm.generate":
            self.model_entered.set()
            await self.hold_model.wait()
            raise AssertionError("The test model is intentionally blocked until unload cancels it")
        raise AssertionError("Live action forbidden in SDK check: " + capability)


async def check_sdk(scratch: Path) -> Dict[str, Any]:
    module = load_plugin_module()
    plugin = module.create_plugin()
    manifest = json.loads((ROOT / "_manifest.json").read_text(encoding="utf-8"))
    fake = FakeHost(manifest)
    plugin._set_context(PluginContext(manifest["id"], rpc_call=fake.rpc, paths=PluginPaths(
        data_dir=scratch / "plugin-data", runtime_dir=scratch / "plugin-runtime",
    )))
    defaults = plugin.build_default_config()
    require(defaults["photography"]["max_images"] == 700, "Unexpected default image budget")
    require(defaults["photography"]["share_enabled"] is False, "Fresh installation must not send messages")
    defaults["photography"]["daily_enabled"] = False
    plugin.set_plugin_config(defaults)
    schema = plugin.build_config_schema(plugin_id=manifest["id"], plugin_name=manifest["name"])
    require(bool(schema), "Empty SDK config schema")
    json.dumps(schema, ensure_ascii=False)
    components = plugin.get_components()
    api_names = set()
    component_summary = []
    for component in components:
        handler = component["metadata"]["handler_name"]
        require(callable(getattr(plugin, handler)), "Registered handler does not exist: " + handler)
        component_summary.append({"type": component["type"], "name": component["name"], "handler": handler})
        if component["type"] == "API":
            api_names.add(component["name"])
    declaration = json.loads((ROOT / "webui.json").read_text(encoding="utf-8"))
    for page in declaration["pages"]:
        for operation in [*page.get("queries", {}).values(), *page.get("actions", {}).values()]:
            require(operation["api"] in api_names, "WebUI references an unregistered API: " + operation["api"])
    hook = next(component for component in components if component["name"] == "photography_feedback")
    require(hook["metadata"]["hook"] == "chat.receive.after_process", "Incorrect feedback hook")
    require(str(hook["metadata"]["mode"]).lower() == "observe", "Feedback hook must only observe")

    await plugin.on_load()
    try:
        require(not plugin.engine.settings.daily_enabled, "on_load ignored injected SDK config")
        require(plugin.engine.worker is not None and not plugin.engine.worker.done(), "Worker did not start")
        snapshot = await plugin.webui_snapshot()
        require(snapshot["groups"] == [{"id": "real-group-stream", "name": "摄影讨论群"}],
                "Snapshot did not consume the SDK-normalized Host group envelope")
        require(snapshot["stats"]["image_count"] == 0, "Check must start with an empty temporary collection")
        require(snapshot["jina_configured"] is False, "Check must not read a live Jina key")
        updated = deepcopy(defaults)
        updated["photography"]["max_images"] = 701
        plugin.set_plugin_config(updated)
        await plugin.on_config_update("self", updated, "sdk-check")
        require(plugin.engine.settings.max_images == 701, "Configuration update not applied")
        invalid = deepcopy(updated)
        invalid["photography"]["max_images"] = 0
        try:
            await plugin.on_config_update("self", invalid, "invalid-check")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid quota update was accepted")
        require(plugin.engine.settings.max_images == 701, "Failed config update mutated settings")
        for body in ({"target_stream_id": "invented-stream"}, {"share_enabled": True}, {"unknown_field": True}):
            try:
                await plugin.webui_settings(json.dumps(body))
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid dashboard settings were accepted: " + str(body))
        sites = await plugin.webui_import_sites("https://photo.example/\nhttps://second.example/")
        require(sites["count"] == 2, "Site import did not persist two sites")
        seeds = await plugin.webui_import_seeds("川内伦子：《Illuminance》\nRobert Adams", "参考.txt")
        require(seeds["count"] == 2 and seeds["job_id"], "Seed import did not enqueue work")
        await asyncio.wait_for(fake.model_entered.wait(), timeout=5)
        manual = await plugin.webui_explore("再看看不同的光线")
        await plugin.webui_cancel(manual["job_id"])
        cancelled = await plugin.engine.db("get", "job", manual["job_id"])
        require(cancelled["status"] == "cancelled", "Queued task cancellation failed")
        records = await plugin.webui_records("job")
        require(records["total"] == 2, "Dashboard task pagination is inconsistent")
        require(all("state" not in item for item in records["items"]), "RPC exposed a full task state")
        shared = await plugin.photography_share("real-group-stream")
        require(shared["success"] is False, "Sharing gate is open before user setup")
        ignored = await plugin.photography_feedback({"session_id": "unrelated-stream"})
        require(ignored == {"action": "continue"}, "Observation hook changed unrelated messages")
    finally:
        await plugin.on_unload()
    require(plugin.engine.worker.done() and plugin.engine.scheduler.done(), "Lifecycle left background tasks running")
    resumed = await plugin.engine.db("get", "job", seeds["job_id"])
    require(resumed["status"] == "queued", "Unload failed to preserve active work for restart")
    require(set(fake.calls) <= {"chat.get_all_streams", "llm.generate", "config.get"}, "Unexpected fake Host capability used")
    return {
        "factory": "passed", "sdk_config_schema": "passed", "components": component_summary,
        "webui_api_bindings": "passed", "lifecycle": "passed", "config_validation": "passed",
        "group_envelope": "passed", "seed_site_import_and_cancel": "passed", "unload_resume": "passed",
        "fake_capability_calls": fake.calls, "real_network_calls": 0, "real_model_calls": 0,
        "real_messages_sent": 0, "live_host_rpc_tested": False, "planner_selection_tested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-root", type=Path, default=ROOT.parents[2])
    arguments = parser.parse_args()
    output = ROOT / "test-results"
    output.mkdir(exist_ok=True)
    report: Dict[str, Any] = {"checked_at": datetime.now(timezone.utc).isoformat(), "success": False}
    old_instance = os.environ.get("MAIBOT_INSTANCE_DIR")
    faulthandler.dump_traceback_later(30, repeat=False)
    try:
        with TemporaryDirectory(prefix="sdk-check-", dir=output) as directory:
            scratch = Path(directory)
            os.environ["MAIBOT_INSTANCE_DIR"] = str(scratch / "host-instance")
            report.update(validate_host_declarations(arguments.host_root.resolve()))
            report.update(asyncio.run(check_sdk(scratch)))
            report["success"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if old_instance is None:
            os.environ.pop("MAIBOT_INSTANCE_DIR", None)
        else:
            os.environ["MAIBOT_INSTANCE_DIR"] = old_instance
        target = output / "sdk-check.json"
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Report: {target}")


if __name__ == "__main__":
    main()
