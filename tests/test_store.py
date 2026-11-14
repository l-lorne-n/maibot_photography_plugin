from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Dict
from unittest import TestCase

from PIL import Image

import base64
import importlib.util
import json
import sqlite3
import sys
import unittest


_PACKAGE = ModuleType("photography_store_tests")
_PACKAGE.__path__ = [str(Path(__file__).parents[1])]
sys.modules[_PACKAGE.__name__] = _PACKAGE
_SPEC = importlib.util.spec_from_file_location("photography_store_tests.store", Path(__file__).parents[1] / "store.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Store = _MODULE.Store


def picture(color: str, size=(640, 480)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


class StoreTests(TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = Store(self.root / "data")
        self.store.initialize()

    def save(self, color: str, **changes: Any) -> Dict[str, Any]:
        return self.store.save_image(picture(color), "image/png", {
            "author": "Original author", "work": "Original book", "source_url": "https://photo.example/work",
            "image_url": f"https://photo.example/{color}.png", "reason": "Subtle light", **changes,
        })

    def test_records_survive_reopen_and_concurrent_updates(self) -> None:
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(lambda number: self.store.put("diary", {"id": str(number), "text": str(number)}), range(20)))
        reopened = Store(self.root / "data")
        reopened.initialize()
        self.assertEqual(reopened.count("diary"), 20)
        reopened.put("diary", {"id": "0", "note": "kept"})
        self.assertEqual(reopened.get("diary", "0")["text"], "0")
        self.assertEqual(reopened.list("diary", limit=1)[0]["id"], "0")
        reopened.delete("diary", "0")
        self.assertIsNone(reopened.get("diary", "0"))

    def test_count_cap_preserves_strong_interest_and_metadata(self) -> None:
        weak = self.save("red", bot_interest=0.1, human_interest=0.2)
        strong = self.save("blue", bot_interest=0.9, human_interest=0.1)
        newest = self.store.save_image(picture("green"), "image/png", {}, max_images=2)
        self.assertEqual(self.store.stats()["image_count"], 2)
        self.assertEqual(self.store.get("image", weak["id"])["local_status"], "evicted")
        self.assertEqual(self.store.get("image", weak["id"])["work"], "Original book")
        self.assertEqual(self.store.get("image", weak["id"])["source_url"], "https://photo.example/work")
        self.assertEqual(self.store.get("image", strong["id"])["local_status"], "available")
        self.assertEqual(self.store.get("image", newest["id"])["local_status"], "available")
        with self.assertRaises(FileNotFoundError):
            self.store.image_bytes(weak["id"])

    def test_byte_cap_and_all_pinned_reject_without_deletion(self) -> None:
        first = self.save("red")
        self.store.put("image", {"id": first["id"], "pinned": True})
        with self.assertRaisesRegex(ValueError, "pinned or protected"):
            self.store.save_image(picture("blue"), "image/png", {}, max_images=1)
        self.assertEqual(self.store.stats()["image_count"], 1)
        self.assertEqual(self.store.image_bytes(first["id"])[1], "image/jpeg")
        self.store.put("image", {"id": first["id"], "pinned": False})
        second = self.store.save_image(picture("blue"), "image/png", {}, max_bytes=first["byte_size"] + 100)
        self.assertEqual(self.store.stats()["image_count"], 1)
        self.assertLessEqual(self.store.stats()["image_bytes"], first["byte_size"] + 100)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.store.save_image(picture("green"), "image/png", {}, max_bytes=1)
        self.assertEqual(self.store.get("image", second["id"])["local_status"], "available")

    def test_idle_both_low_interest_only_and_seed_never_touched(self) -> None:
        seed = self.root / "seed.jpg"
        seed.write_bytes(b"user-owned-original")
        old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        low = self.save("red", bot_interest=0.1, human_interest=0.1)
        bot_favorite = self.save("blue", bot_interest=0.9, human_interest=0.1)
        human_favorite = self.save("green", bot_interest=0.1, human_interest=0.9)
        protected = self.save("yellow", bot_interest=0.1, human_interest=0.1)
        unknown = self.save("purple")
        for record in (low, bot_favorite, human_favorite, protected, unknown):
            self.store.put("image", {"id": record["id"], "last_interest_at": old})
        result = self.store.cleanup(700, 1073741824, 4, protect_ids={protected["id"]})
        self.assertEqual(result["evicted_ids"], [low["id"]])
        self.assertEqual(seed.read_bytes(), b"user-owned-original")
        self.assertEqual(self.store.stats()["image_count"], 4)

    def test_dedup_preserves_original_evidence_and_does_not_renew_interest(self) -> None:
        first = self.save("red", bot_interest=0.5, human_interest=0.5)
        old = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        self.store.put("image", {"id": first["id"], "last_interest_at": old})
        second = self.save("red", author="Different credit", source_url="https://other.example/book")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(second["author"], "Original author")
        self.assertEqual(second["last_interest_at"], old)
        self.assertEqual(len(second["sources"]), 2)
        self.assertEqual(self.store.stats()["image_count"], 1)
        self.assertEqual(len(list(self.store.images_dir.iterdir())), 1)
        self.store.delete("image", first["id"])
        restored = self.save("red", source_url="https://third.example/book")
        self.assertEqual(restored["local_status"], "available")
        self.assertEqual(restored["author"], "Original author")
        self.assertEqual(len(restored["sources"]), 3)

    def test_reject_bad_images_and_invalid_interests(self) -> None:
        for data, mime in ((b"not a photo", "image/png"), (picture("red"), "text/html")):
            with self.assertRaises(ValueError):
                self.store.save_image(data, mime, {})
        with self.assertRaises(ValueError):
            self.save("red", bot_interest=float("nan"))
        self.assertEqual(self.store.count("image"), 0)
        self.assertEqual(list(self.store.images_dir.iterdir()), [])

    def test_unknown_human_interest_is_preserved_and_not_idle_evicted(self) -> None:
        unknown = self.save("red", id="observation-uuid", bot_interest=0.1, human_interest=None)
        self.assertEqual(len(unknown["id"]), 64)
        self.assertNotEqual(unknown["id"], "observation-uuid")
        self.assertIsNone(unknown["human_interest"])
        old = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        self.store.put("image", {"id": unknown["id"], "last_interest_at": old, "human_interest": None})
        self.assertEqual(self.store.cleanup(700, 1073741824, 4)["evicted_count"], 0)
        known = self.save("blue", bot_interest=0.6, human_interest=0.9)
        self.store.save_image(picture("green"), "image/png", {}, max_images=2)
        self.assertEqual(self.store.get("image", unknown["id"])["local_status"], "evicted")
        duplicate = self.save("blue", human_interest=None)
        self.assertEqual(duplicate["human_interest"], 0.9)
        self.assertEqual(self.store.get("image", known["id"])["local_status"], "available")

    def test_normalization_thumbnail_and_no_binary_database(self) -> None:
        record = self.store.save_image(picture("red", (3000, 2000)), "image/png", {})
        self.assertEqual(record["width"], 2048)
        data_url = self.store.thumbnail(record["id"])
        data = base64.b64decode(data_url.split(",", 1)[1])
        with Image.open(BytesIO(data)) as thumbnail:
            self.assertLessEqual(max(thumbnail.size), 600)
        self.assertNotIn("base64", json.dumps(self.store.get("image", record["id"])))

    def test_path_tampering_cannot_touch_seed(self) -> None:
        record = self.save("red")
        seed = self.root / "seed.jpg"
        seed.write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "cannot be changed"):
            self.store.put("image", {"id": record["id"], "relative_path": "../seed.jpg"})
        record["relative_path"] = "../seed.jpg"
        with closing(sqlite3.connect(self.store.database_path)) as connection:
            connection.execute("UPDATE records SET payload=? WHERE kind='image' AND id=?", (json.dumps(record), record["id"]))
            connection.commit()
        with self.assertRaisesRegex(ValueError, "Invalid managed image path"):
            self.store.cleanup(0, 0, 0)
        self.assertEqual(seed.read_bytes(), b"original")

    def test_restart_reconciles_missing_and_orphaned_cache_files(self) -> None:
        record = self.save("red")
        (self.store.data_dir / record["relative_path"]).unlink()
        orphan = self.store.images_dir / ("a" * 64 + ".jpg")
        orphan.write_bytes(b"interrupted download")
        self.store.initialize()
        self.assertFalse(orphan.exists())
        self.assertEqual(self.store.stats()["image_count"], 0)
        self.assertEqual(self.store.get("image", record["id"])["local_status"], "evicted")


if __name__ == "__main__":
    unittest.main()
