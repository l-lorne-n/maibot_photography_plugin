"""Offline perceptual identity, version upgrades, and legacy-reference migration."""

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Dict, Tuple
from unittest import TestCase

from PIL import Image, ImageDraw, ImageFilter

import importlib
import json
import sqlite3
import sys


ROOT = Path(__file__).parents[1]
PACKAGE = ModuleType("photography_identity_tests")
PACKAGE.__path__ = [str(ROOT)]
sys.modules[PACKAGE.__name__] = PACKAGE
Store = importlib.import_module(PACKAGE.__name__ + ".store").Store
similarity = importlib.import_module(PACKAGE.__name__ + ".similarity")


def photograph(size: Tuple[int, int] = (900, 600), *, variant: int = 0,
               quality: int = 90, blur: float = 0, encoding: str = "JPEG") -> bytes:
    image = Image.new("RGB", (900, 600), (174, 188, 202))
    draw = ImageDraw.Draw(image)
    if variant == 0:
        draw.rectangle((0, 270, 899, 599), fill=(47, 68, 43))
        draw.polygon(((0, 290), (220, 120), (440, 330)), fill=(113, 112, 102))
        draw.rectangle((490, 130, 740, 460), fill=(205, 150, 105))
        for x in range(505, 735, 28):
            draw.line((x, 155, x, 440), fill=(55, 44, 35), width=4)
        draw.ellipse((78, 75, 175, 172), fill=(243, 228, 178))
    else:
        draw.rectangle((0, 290, 899, 599), fill=(47, 68, 43))
        draw.ellipse((250, 50, 780, 530), fill=(205, 150, 105))
        draw.polygon(((20, 60), (420, 590), (30, 590)), fill=(35, 50, 112))
    image = image.resize(size, Image.Resampling.LANCZOS)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    output = BytesIO()
    image.save(output, format=encoding, quality=quality)
    return output.getvalue()


class IdentityTests(TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = Store(Path(self.temporary.name) / "data")
        self.store.initialize()

    def legacy(self, data: bytes, stamp: str, **metadata: Any) -> Dict[str, Any]:
        normalized, mime, width, height = self.store._normalize(data, "image/jpeg")
        identifier = sha256(normalized).hexdigest()
        record = {"id": identifier, "created_at": stamp, "updated_at": stamp, "last_interest_at": stamp,
                  "relative_path": f"images/{identifier}.jpg", "mime_type": mime, "width": width, "height": height,
                  "byte_size": len(normalized), "local_status": "available", "source_sha256": sha256(data).hexdigest(),
                  "author": "Legacy author", "work": "Book", "source_url": "https://photos.example/book",
                  "image_url": f"https://photos.example/{width}.jpg", "bot_interest": 0.7,
                  "human_interest": None, "pinned": False, **metadata}
        (self.store.data_dir / record["relative_path"]).write_bytes(normalized)
        with closing(sqlite3.connect(self.store.database_path)) as connection:
            connection.execute("INSERT INTO records(kind,id,created_at,updated_at,payload) VALUES('image',?,?,?,?)",
                               (identifier, stamp, stamp, json.dumps(record)))
            connection.commit()
        return record

    def test_resize_and_compression_share_work_but_different_scene_does_not(self) -> None:
        first = self.store.identify_image(photograph((300, 200), quality=55), "image/jpeg")
        second = self.store.identify_image(photograph((900, 600), quality=93), "image/jpeg")
        third = self.store.identify_image(photograph((900, 600), variant=1), "image/jpeg")
        self.assertEqual(first["work_id"], second["work_id"])
        self.assertNotEqual(first["fingerprint"]["pixel_hash"], second["fingerprint"]["pixel_hash"])
        self.assertIsNotNone(second["distance"])
        self.assertNotEqual(first["work_id"], third["work_id"])
        self.assertEqual(self.store.count("image"), 0)
        self.assertEqual(list(self.store.images_dir.iterdir()), [])
        with closing(sqlite3.connect(self.store.database_path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM image_identity").fetchone()[0], 2)
            self.assertTrue(all(len(row[0]) == 16 for row in connection.execute("SELECT phash FROM image_identity")))

    def test_flat_color_hash_collision_is_not_same_work(self) -> None:
        def flat(color: str) -> bytes:
            output = BytesIO()
            Image.new("RGB", (300, 200), color).save(output, format="PNG")
            return output.getvalue()
        red = self.store.identify_image(flat("red"), "image/png")
        blue = self.store.identify_image(flat("blue"), "image/png")
        self.assertEqual(red["phash"], blue["phash"])
        self.assertNotEqual(red["work_id"], blue["work_id"])

    def test_better_version_keeps_id_and_references_and_worse_never_downgrades(self) -> None:
        low = photograph((300, 200), quality=60)
        high = photograph((900, 600), quality=95)
        initial = self.store.save_image(low, "image/jpeg", {"observations": "My original reaction", "bot_interest": 0.8})
        self.store.put("image", {"id": initial["id"], "pinned": True, "human_interest": 0.91})
        self.store.put("diary", {"id": "day", "image_ids": [initial["id"]], "content": "A real past experience"})
        self.store.put("share", {"id": "sent", "image_id": initial["id"], "message_id": "actual-receipt"})
        upgraded = self.store.upgrade_image(high, "image/jpeg", {"source_url": "https://other.example/high"})
        self.assertEqual(upgraded["id"], initial["id"])
        self.assertEqual(upgraded["save_outcome"], "upgraded")
        self.assertEqual(upgraded["width"], 900)
        self.assertEqual(upgraded["observations"], "My original reaction")
        self.assertTrue(upgraded["pinned"])
        self.assertEqual(upgraded["human_interest"], 0.91)
        self.assertEqual(self.store.count("image"), 1)
        self.assertEqual(self.store.get("diary", "day")["image_ids"], [initial["id"]])
        self.assertEqual(self.store.get("share", "sent")["image_id"], initial["id"])
        retained = self.store.save_image(low, "image/jpeg", {})
        self.assertEqual((retained["save_outcome"], retained["width"]), ("retained", 900))
        self.assertEqual(len(list(self.store.images_dir.iterdir())), 1)

    def test_preview_upgrade_never_collects_unknown_or_evicted_work(self) -> None:
        self.assertIsNone(self.store.upgrade_image(photograph(), "image/jpeg", {}))
        item = self.store.save_image(photograph((300, 200)), "image/jpeg", {})
        self.store.delete("image", item["id"])
        self.assertIsNone(self.store.upgrade_image(photograph(), "image/jpeg", {}))
        self.assertEqual(self.store.get("image", item["id"])["local_status"], "evicted")

    def test_quality_comparison_rejects_large_blurry_version(self) -> None:
        clear = similarity.fingerprint(photograph((450, 300)))
        blurry = similarity.fingerprint(photograph((900, 600), blur=20))
        self.assertFalse(similarity.is_clearer(blurry, clear))

    def test_migration_dry_run_then_preserves_old_references_and_best_pixels(self) -> None:
        older = self.legacy(photograph((300, 200), quality=55), "2026-11-01T00:00:00+00:00", human_interest=0.8)
        newer = self.legacy(photograph((900, 600), quality=95), "2026-11-02T00:00:00+00:00", pinned=True)
        other = self.legacy(photograph(variant=1), "2026-11-03T00:00:00+00:00")
        self.store.put("diary", {"id": "j", "image_ids": [older["id"], newer["id"]], "content": "Keep this diary verbatim"})
        self.store.put("share", {"id": "s", "image_id": newer["id"], "message_id": "real-platform-id"})
        self.store.put("feedback", {"id": "f", "image_id": newer["id"], "insight": "A historical opinion",
                                    "candidate_image_ids": [newer["id"], older["id"]]})
        for number, image in enumerate((older, newer)):
            self.store.put("preview", {"id": f"p{number}", "job_id": "j", "is_photo": True})
            self.store.put("observation", {"id": f"o{number}", "preview_id": f"p{number}", "image_id": image["id"], "job_id": "j"})
        self.store.put("preview", {"id": "unknown", "job_id": "j", "is_photo": True})
        self.store.put("job", {"id": "j", "image_ids": [older["id"], newer["id"]],
            "share_planned_image_ids": [newer["id"]], "preview_ids": ["p0", "p1", "unknown"],
            "viewed_preview_ids": ["p0", "p1", "unknown"], "observation_ids": ["o0", "o1"],
            "preview_count": 3, "detail_count": 2})
        before_files = {p.name: p.read_bytes() for p in self.store.images_dir.iterdir()}
        before_rows = self.store.list("image", limit=100)
        plan = self.store.migrate_image_identities()
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["merged_count"], 1)
        self.assertEqual(self.store.list("image", limit=100), before_rows)
        self.assertEqual(before_files, {p.name: p.read_bytes() for p in self.store.images_dir.iterdir()})
        result = self.store.migrate_image_identities(dry_run=False)
        self.assertEqual(result["merged_count"], 1)
        self.assertEqual(self.store.resolve_image_id(newer["id"]), older["id"])
        self.assertEqual(self.store.resolve_image_ids([newer["id"], older["id"], other["id"]]), [older["id"], other["id"]])
        self.assertEqual(set(self.store.image_aliases(newer["id"])), {older["id"], newer["id"]})
        image = self.store.get("image", newer["id"])
        self.assertEqual(image["id"], older["id"])
        self.assertTrue(image["pinned"])
        self.assertEqual(image["human_interest"], 0.8)
        self.assertEqual(image["width"], 900)
        self.assertEqual(self.store.get_raw_image(newer["id"])["alias_of"], older["id"])
        self.assertEqual(self.store.count("image"), 2)
        self.assertEqual(self.store.stats()["image_count"], 2)
        self.assertEqual(self.store.image_bytes(newer["id"]), self.store.image_bytes(older["id"]))
        diary = self.store.get("diary", "j")
        self.assertEqual(diary["content"], "Keep this diary verbatim")
        self.assertEqual(diary["image_ids"], [older["id"]])
        self.assertEqual(diary["identity_original_references"]["image_ids"], [older["id"], newer["id"]])
        self.assertEqual(self.store.get("share", "s")["image_id"], newer["id"])
        self.assertEqual(self.store.get("share", "s")["canonical_image_id"], older["id"])
        self.assertEqual(self.store.get("feedback", "f")["image_id"], older["id"])
        self.assertEqual(self.store.get("feedback", "f")["candidate_image_ids"], [older["id"]])
        self.assertEqual(self.store.get("feedback", "f")["identity_original_references"]["candidate_image_ids"],
                         [newer["id"], older["id"]])
        job = self.store.get("job", "j")
        self.assertEqual((job["preview_count"], job["detail_count"]), (2, 1))
        self.assertEqual(job["identity_coverage"]["unknown_preview_count"], 1)
        self.assertFalse(job["identity_coverage"]["complete"])
        self.store.put("image", {"id": newer["id"], "pinned": False})
        self.assertFalse(self.store.get("image", older["id"])["pinned"])
        self.assertEqual(self.store.migrate_image_identities(dry_run=False)["merged_count"], 0)
        reopened = Store(self.store.data_dir)
        reopened.initialize()
        self.assertEqual(reopened.resolve_image_id(newer["id"]), older["id"])
        self.assertEqual(reopened.stats()["image_count"], 2)
