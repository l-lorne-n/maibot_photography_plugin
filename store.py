"""Synchronous, plugin-owned storage. Call from worker threads in async code."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import uuid

from .similarity import DEFAULT_PHASH_DISTANCE, fingerprint, is_clearer, match_distance

_LOCKS: Dict[str, Any] = {}
_LOCKS_LOCK = threading.Lock()
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_FILE_NAME = re.compile(r"^[0-9a-f]{64}\.(?:jpg|webp)$")
_IMAGE_FIELDS = {
    "id", "created_at", "relative_path", "mime_type", "byte_size", "width", "height",
    "local_status", "evicted_at", "eviction_reason", "source_sha256",
    "phash", "identity_signature", "alias_of", "version", "upgraded_at",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Storage timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _interest(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Image interest must be a number between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Image interest must be a number between 0 and 1")
    return number


def _human_interest(value: Any) -> Optional[float]:
    return None if value is None else _interest(value)


def _retention_interest(record: Dict[str, Any]) -> float:
    human = _human_interest(record["human_interest"])
    return max(_interest(record["bot_interest"]), 0.5 if human is None else human)


class Store:
    """SQLite records and a bounded image cache; original seed files are never owned here."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.images_dir = self.data_dir / "images"
        self.database_path = self.data_dir / "photography.sqlite3"
        with _LOCKS_LOCK:
            self._lock = _LOCKS.setdefault(os.path.normcase(str(self.data_dir)), threading.RLock())

    def initialize(self) -> None:
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._check_images_dir()
            self.images_dir.mkdir(exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS records ("
                    "kind TEXT NOT NULL, id TEXT NOT NULL, created_at TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(kind, id))"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS records_recent ON records(kind, updated_at DESC)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS image_identity ("
                    "work_id TEXT PRIMARY KEY, phash TEXT NOT NULL, signature TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS image_alias (alias_id TEXT PRIMARY KEY, work_id TEXT NOT NULL)"
                )
                connection.commit()
                connection.execute("BEGIN IMMEDIATE")
                available = self._available(connection)
                tracked = {self._image_path(record).name for record in available}
                # A process can stop after atomic file replacement and before SQLite commit.
                # Only our content-addressed cache files are eligible for orphan recovery.
                for path in self.images_dir.iterdir():
                    if _FILE_NAME.fullmatch(path.name) and path.name not in tracked:
                        self._assert_managed_path(path)
                        path.unlink()
                connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _check_images_dir(self) -> None:
        if self.images_dir.is_symlink() or not self.images_dir.resolve().is_relative_to(self.data_dir):
            raise ValueError("Managed image directory must stay inside plugin data directory")

    def _assert_managed_path(self, path: Path) -> None:
        self._check_images_dir()
        if path.is_symlink() or not path.resolve().is_relative_to(self.images_dir.resolve()):
            raise ValueError("Image path is outside the managed image directory")

    def _image_path(self, record: Dict[str, Any]) -> Path:
        image_id = record["id"]
        if not isinstance(image_id, str) or not _DIGEST.fullmatch(image_id):
            raise ValueError("Invalid managed image id")
        extensions = {"image/jpeg": "jpg", "image/webp": "webp"}
        if record["mime_type"] not in extensions:
            raise ValueError("Unsupported managed image format")
        name = f"{image_id}.{extensions[record['mime_type']]}"
        if record["relative_path"] != f"images/{name}":
            raise ValueError("Invalid managed image path")
        path = self.images_dir / name
        self._assert_managed_path(path)
        return path

    @staticmethod
    def _resolve_image_id(connection: sqlite3.Connection, image_id: str) -> str:
        seen: Set[str] = set()
        while image_id not in seen:
            seen.add(image_id)
            row = connection.execute("SELECT work_id FROM image_alias WHERE alias_id=?", (image_id,)).fetchone()
            if row is None:
                return image_id
            image_id = row["work_id"]
        raise ValueError("Cyclic image identity alias")

    def resolve_image_id(self, image_id: str) -> str:
        """Return the stable work ID; unknown IDs are unchanged, never fabricated."""
        with self._lock, self._connect() as connection:
            return self._resolve_image_id(connection, image_id)

    def resolve_image_ids(self, image_ids: List[str]) -> List[str]:
        """Resolve legacy references and remove duplicate work IDs, preserving order."""
        with self._lock, self._connect() as connection:
            return list(dict.fromkeys(self._resolve_image_id(connection, image_id) for image_id in image_ids))

    def image_aliases(self, image_id: str) -> List[str]:
        """Return the canonical ID and all retained legacy IDs for this work."""
        with self._lock, self._connect() as connection:
            work_id = self._resolve_image_id(connection, image_id)
            rows = connection.execute("SELECT alias_id FROM image_alias ORDER BY alias_id").fetchall()
            return [work_id] + [row["alias_id"] for row in rows
                                if self._resolve_image_id(connection, row["alias_id"]) == work_id]

    @staticmethod
    def _identity_rows(connection: sqlite3.Connection) -> List[Dict[str, Any]]:
        return [{"work_id": row["work_id"], "signature": json.loads(row["signature"]), "created_at": row["created_at"]}
                for row in connection.execute("SELECT * FROM image_identity ORDER BY created_at,work_id")]

    def _identify(self, connection: sqlite3.Connection, signature: Dict[str, Any],
                  max_distance: int = DEFAULT_PHASH_DISTANCE) -> Dict[str, Any]:
        # Deliberate full-table scan: the collection is small, and pHash is a
        # simple work-identity index, separate from semantic image embeddings.
        matches = []
        for item in self._identity_rows(connection):
            distance = match_distance(signature, item["signature"], max_distance)
            if distance is not None:
                matches.append((distance, item["created_at"], item["work_id"], item["signature"]))
        if matches:
            distance, _, work_id, known = min(matches, key=lambda item: item[:3])
            if is_clearer(signature, known):
                connection.execute("UPDATE image_identity SET phash=?,signature=? WHERE work_id=?",
                    (signature["phash"], json.dumps(signature, allow_nan=False), work_id))
            return {"work_id": self._resolve_image_id(connection, work_id), "distance": distance, "is_new": False,
                    "phash": signature["phash"], "fingerprint": signature}
        work_id = signature["pixel_hash"]
        connection.execute("INSERT INTO image_identity(work_id,phash,signature,created_at) VALUES(?,?,?,?)",
            (work_id, signature["phash"], json.dumps(signature, allow_nan=False), _now()))
        return {"work_id": work_id, "distance": None, "is_new": True,
                "phash": signature["phash"], "fingerprint": signature}

    def identify_image(self, data: bytes, mime_type: str = "",
                       max_distance: int = DEFAULT_PHASH_DISTANCE) -> Dict[str, Any]:
        """Register an actually decoded preview and return its stable work identity.

        This stores only a pHash signature in SQLite, never preview bytes. A job
        counts each returned work_id once; a match from an earlier job still may
        be viewed once in the current job. Use asyncio.to_thread at async callers.
        """
        if mime_type and not mime_type.lower().split(";", 1)[0].startswith("image/"):
            raise ValueError("Identity input must be an image")
        signature = fingerprint(data)
        match_distance(signature, signature, max_distance)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._identify(connection, signature, max_distance)
            image = self._read(connection, "image", result["work_id"])
            result.update(image_id=image["id"] if image else None, local_status=image["local_status"] if image else None)
            connection.commit()
            return result

    def get_raw_image(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Read historical image metadata without alias resolution, for audit/migration."""
        with self._lock, self._connect() as connection:
            return self._read(connection, "image", image_id)

    @staticmethod
    def _read(connection: sqlite3.Connection, kind: str, record_id: str) -> Optional[Dict[str, Any]]:
        row = connection.execute(
            "SELECT payload FROM records WHERE kind=? AND id=?", (kind, record_id)
        ).fetchone()
        return json.loads(row["payload"]) if row is not None else None

    @staticmethod
    def _write(connection: sqlite3.Connection, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(record)
        result["updated_at"] = _now()
        result.setdefault("created_at", result["updated_at"])
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        connection.execute(
            "INSERT INTO records(kind,id,created_at,updated_at,payload) VALUES(?,?,?,?,?) "
            "ON CONFLICT(kind,id) DO UPDATE SET updated_at=excluded.updated_at,payload=excluded.payload",
            (kind, result["id"], result["created_at"], result["updated_at"], payload),
        )
        return result

    def put(self, kind: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Atomically merge a record; image file ownership is changed only by cache methods."""
        if not kind or not isinstance(record.get("id"), str) or not record["id"]:
            raise ValueError("Records require nonempty kind and string id")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if kind == "image":
                record = {**record, "id": self._resolve_image_id(connection, record["id"])}
            existing = self._read(connection, kind, record["id"])
            if kind == "image":
                if existing is None:
                    raise ValueError("Create managed images with save_image")
                for key in _IMAGE_FIELDS:
                    if key in record and record[key] != existing.get(key):
                        raise ValueError(f"Managed image field cannot be changed with put: {key}")
            merged = {**(existing or {}), **record}
            if existing is not None:
                merged["created_at"] = existing["created_at"]
            if kind == "image":
                merged["bot_interest"] = _interest(merged["bot_interest"])
                merged["human_interest"] = _human_interest(merged["human_interest"])
                if not isinstance(merged.get("pinned", False), bool):
                    raise ValueError("Image pinned must be a boolean")
                _date(merged["last_interest_at"])
            result = self._write(connection, kind, merged)
            connection.commit()
            return result

    def get(self, kind: str, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            if kind == "image":
                record_id = self._resolve_image_id(connection, record_id)
            return self._read(connection, kind, record_id)

    def list(self, kind: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        if limit < 0 or offset < 0:
            raise ValueError("Pagination values must be nonnegative")
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM records WHERE kind=? "
                "AND (kind!='image' OR NOT EXISTS(SELECT 1 FROM image_alias WHERE alias_id=records.id)) "
                "ORDER BY updated_at DESC,id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
            return [json.loads(row["payload"]) for row in rows]

    def delete(self, kind: str, record_id: str) -> None:
        """Delete ordinary records; image deletion retains its source and diary evidence."""
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if kind == "image":
                record_id = self._resolve_image_id(connection, record_id)
                record = self._read(connection, kind, record_id)
                if record is not None and record["local_status"] == "available":
                    if record.get("pinned", False):
                        raise ValueError("Unpin the image before deleting its local copy")
                    self._evict(connection, record, "manual")
            else:
                connection.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, record_id))
            connection.commit()

    def count(self, kind: str) -> int:
        with self._lock, self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM records WHERE kind=? "
                "AND (kind!='image' OR NOT EXISTS(SELECT 1 FROM image_alias WHERE alias_id=records.id))", (kind,)).fetchone()[0])

    def _available(self, connection: sqlite3.Connection) -> List[Dict[str, Any]]:
        rows = connection.execute("SELECT payload FROM records WHERE kind='image'").fetchall()
        available: List[Dict[str, Any]] = []
        for row in rows:
            record = json.loads(row["payload"])
            if record["local_status"] != "available":
                continue
            path = self._image_path(record)
            if not path.is_file():
                record.update(local_status="evicted", evicted_at=_now(), eviction_reason="file_missing")
                self._write(connection, "image", record)
                continue
            record["byte_size"] = path.stat().st_size
            available.append(record)
        return available

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            available = self._available(connection)
            counts = {
                row["kind"]: row["total"]
                for row in connection.execute("SELECT kind,COUNT(*) AS total FROM records GROUP BY kind")
            }
            alias_count = int(connection.execute("SELECT COUNT(*) FROM image_alias").fetchone()[0])
            counts["image"] = int(connection.execute("SELECT COUNT(*) FROM records WHERE kind='image' "
                "AND NOT EXISTS(SELECT 1 FROM image_alias WHERE alias_id=records.id)").fetchone()[0])
            connection.commit()
            return {
                "counts": counts,
                "image_count": len(available),
                "image_bytes": sum(record["byte_size"] for record in available),
                "evicted_images": counts.get("image", 0) - len(available),
                "image_aliases": alias_count,
            }

    @staticmethod
    def _normalize(data: bytes, mime_type: str) -> Tuple[bytes, str, int, int]:
        if not mime_type.lower().split(";", 1)[0].startswith("image/"):
            raise ValueError("The downloaded response is not an image")
        try:
            with Image.open(BytesIO(data)) as opened:
                if opened.width * opened.height > 40_000_000:
                    raise ValueError("Image exceeds the collection pixel limit")
                opened.load()
                image = ImageOps.exif_transpose(opened)
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                transparent = image.mode in ("RGBA", "LA") or "transparency" in image.info
                output = BytesIO()
                if transparent:
                    image.convert("RGBA").save(output, format="WEBP", quality=88, method=4)
                    normalized_mime = "image/webp"
                else:
                    image.convert("RGB").save(output, format="JPEG", quality=88, optimize=True)
                    normalized_mime = "image/jpeg"
                return output.getvalue(), normalized_mime, image.width, image.height
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError(f"Downloaded image cannot be decoded: {error}") from error

    @staticmethod
    def _merge_sources(record: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        source = {
            key: metadata[key]
            for key in ("source_url", "image_url", "author", "work", "title", "reason")
            if metadata.get(key)
        }
        sources = list(record.get("sources", []))
        if source and source not in sources:
            sources.append(source)
        record["sources"] = sources

    def save_image(
        self,
        data: bytes,
        mime_type: str,
        metadata: Dict[str, Any],
        max_images: int = 700,
        max_bytes: int = 1073741824,
        idle_days: int = 4,
    ) -> Dict[str, Any]:
        # Validate serializability before planning any file eviction.
        json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        normalized, normalized_mime, width, height = self._normalize(data, mime_type)
        signature = fingerprint(data)
        cache_signature = {**signature, "width": width, "height": height}
        bot_interest = _interest(metadata.get("bot_interest", 0.5))
        human_interest = _human_interest(metadata.get("human_interest"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            identity = self._identify(connection, signature)
            image_id = identity["work_id"]
            existing = self._read(connection, "image", image_id)
            available = existing is not None and existing["local_status"] == "available"
            if available:
                available = self._image_path(existing).is_file()
            previous_signature = existing.get("identity_signature") if existing else None
            if available and previous_signature is None:
                previous_signature = fingerprint(self._image_path(existing).read_bytes())
            upgrade = available and is_clearer(cache_signature, previous_signature)
            write_file = not available or upgrade
            self._cleanup(
                connection, max_images, max_bytes, idle_days,
                needed_bytes=max(0, len(normalized) - existing["byte_size"]) if upgrade else 0 if available else len(normalized),
                needed_count=0 if available else 1,
                protect_ids={image_id},
            )
            if existing is None:
                record = {key: value for key, value in metadata.items() if key not in _IMAGE_FIELDS}
                record.update(
                    id=image_id, created_at=_now(), last_interest_at=_now(),
                    bot_interest=bot_interest, human_interest=human_interest,
                    pinned=False, source_sha256=hashlib.sha256(data).hexdigest(),
                    version=1,
                )
            else:
                record = dict(existing)
                bot_interest = _interest(metadata.get("bot_interest", record["bot_interest"]))
                human_interest = _human_interest(metadata.get("human_interest", record["human_interest"]))
                previous_human = _human_interest(record["human_interest"])
                # Seeing the same URL again is not fresh interest. Only a stronger
                # evaluation or an explicit discussion update renews retention.
                stronger_human = human_interest is not None and (
                    previous_human is None or human_interest > previous_human
                )
                if bot_interest > record["bot_interest"] or stronger_human:
                    record["last_interest_at"] = _now()
                record["bot_interest"] = max(record["bot_interest"], bot_interest)
                if stronger_human:
                    record["human_interest"] = human_interest
            self._merge_sources(record, metadata)
            previous_path = self._image_path(existing) if available else None
            if write_file:
                extension = "jpg" if normalized_mime == "image/jpeg" else "webp"
                record.update(
                    relative_path=f"images/{image_id}.{extension}", mime_type=normalized_mime,
                    byte_size=len(normalized), width=width, height=height, local_status="available",
                    phash=signature["phash"], identity_signature=cache_signature,
                    source_sha256=hashlib.sha256(data).hexdigest(),
                )
                if upgrade:
                    record.update(version=int(record.get("version", 1)) + 1, upgraded_at=_now())
            record.pop("evicted_at", None)
            record.pop("eviction_reason", None)
            if write_file:
                destination = self._image_path(record)
                self._write_image_bytes(destination, normalized)
            result = self._write(connection, "image", record)
            connection.commit()
            if previous_path and previous_path != self._image_path(record) and previous_path.exists():
                previous_path.unlink()
            return {**result, "save_outcome": "new" if existing is None else "upgraded" if upgrade else "retained" if available else "restored",
                    "match_distance": identity["distance"]}

    def _write_image_bytes(self, destination: Path, data: bytes) -> None:
        self._assert_managed_path(destination)
        temporary = self.images_dir / f".pending-{uuid.uuid4().hex}"
        self._assert_managed_path(temporary)
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def upgrade_image(self, data: bytes, mime_type: str, metadata: Dict[str, Any],
                      max_images: int = 700, max_bytes: int = 1073741824,
                      idle_days: int = 4) -> Optional[Dict[str, Any]]:
        """Improve an existing local collection item without collecting a new preview."""
        with self._lock:
            identity = self.identify_image(data, mime_type)
            existing = self.get("image", identity["work_id"])
            if existing is None or existing["local_status"] != "available":
                return None
            if not self._image_path(existing).is_file():
                raise FileNotFoundError("Collected image file is missing; cannot silently restore it during preview")
            return self.save_image(data, mime_type, metadata, max_images, max_bytes, idle_days)

    @staticmethod
    def _merge_collection_records(canonical: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(canonical)
        preferences: Dict[str, Dict[str, Any]] = {}
        for record in records:
            Store._merge_sources(merged, record)
            for source in record.get("sources", []):
                Store._merge_sources(merged, source)
            for speaker, preference in record.get("human_preferences", {}).items():
                known = preferences.setdefault(speaker, {"score": 0.5, "evidence_ids": [], "evidence": []})
                evidence = {item["id"]: item for item in known["evidence"]}
                evidence.update({item["id"]: item for item in preference.get("evidence", [])})
                known["evidence"] = sorted(evidence.values(), key=lambda item: (item["created_at"], item["id"]))
                known["evidence_ids"] = list(dict.fromkeys(known["evidence_ids"] + preference.get("evidence_ids", [])))
                if known["evidence"]:
                    score = 0.5
                    for item in known["evidence"]:
                        score = round(0.8 * score + 0.2 * _interest(item["score"]), 8)
                    known.update(score=score, updated_at=known["evidence"][-1]["created_at"])
                else:
                    known["score"] = max(known["score"], _interest(preference["score"]))
        merged["pinned"] = any(record.get("pinned", False) for record in records)
        merged["bot_interest"] = max(_interest(record["bot_interest"]) for record in records)
        humans = [_human_interest(record["human_interest"]) for record in records if record["human_interest"] is not None]
        merged["human_interest"] = max(humans) if humans else None
        if preferences:
            merged["human_preferences"] = preferences
            merged["human_interest"] = max(preference["score"] for preference in preferences.values())
        merged["last_interest_at"] = max(records, key=lambda item: _date(item["last_interest_at"]))["last_interest_at"]
        return merged

    def migrate_image_identities(self, *, dry_run: bool = True,
                                 max_distance: int = DEFAULT_PHASH_DISTANCE) -> Dict[str, Any]:
        """Explicit offline pHash backfill/merge. Back up the data directory before apply.

        initialize() never runs this migration. The oldest image ID in a matched
        group remains canonical; old image records, diaries, feedback, and share
        receipts keep their original IDs. Public image access resolves aliases.
        """
        if isinstance(max_distance, bool) or not isinstance(max_distance, int) or not 0 <= max_distance <= 64:
            raise ValueError("pHash distance must be an integer in 0..64")
        with self._lock, self._connect() as connection:
            if not dry_run:
                connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT payload FROM records WHERE kind='image' "
                "AND NOT EXISTS(SELECT 1 FROM image_alias WHERE alias_id=records.id) ORDER BY created_at,id").fetchall()
            groups: List[Dict[str, Any]] = []
            skipped = []
            for row in rows:
                record = json.loads(row["payload"])
                path = self._image_path(record)
                data = path.read_bytes() if record["local_status"] == "available" and path.is_file() else None
                signature = fingerprint(data) if data is not None else record.get("identity_signature")
                if signature is None:
                    skipped.append({"id": record["id"], "reason": "no_local_pixels_or_prior_signature"})
                    continue
                matches = [(distance, index) for index, group in enumerate(groups)
                           if (distance := match_distance(signature, group["signature"], max_distance)) is not None]
                if matches:
                    group = groups[min(matches)[1]]
                else:
                    group = {"canonical": record, "signature": signature, "members": [], "best": None}
                    groups.append(group)
                entry = {"record": record, "signature": signature, "data": data}
                group["members"].append(entry)
                if data is not None and (group["best"] is None or is_clearer(signature, group["best"]["signature"])):
                    group["best"] = entry
            plan = [{"work_id": group["canonical"]["id"],
                     "aliases": [item["record"]["id"] for item in group["members"][1:]],
                     "best_image_id": group["best"]["record"]["id"] if group["best"] else None}
                    for group in groups]
            report = {"dry_run": dry_run, "scanned": len(rows), "works": len(groups), "skipped": skipped,
                      "merged_count": sum(len(item["aliases"]) for item in plan), "plan": plan}
            if dry_run:
                proposed_aliases = {alias: item["work_id"] for item in plan for alias in item["aliases"]}
                report["references"] = self._migrate_identity_references(connection, proposed_aliases, write=False)
                return report
            obsolete: Set[Path] = set()
            for group in groups:
                canonical = group["canonical"]
                work_id = canonical["id"]
                records = [item["record"] for item in group["members"]]
                merged = self._merge_collection_records(canonical, records)
                best = group["best"]
                signature = best["signature"] if best else group["signature"]
                for record in records:
                    if record["local_status"] == "available":
                        obsolete.add(self._image_path(record))
                if best:
                    extension = "jpg" if best["record"]["mime_type"] == "image/jpeg" else "webp"
                    merged.update(mime_type=best["record"]["mime_type"], relative_path=f"images/{work_id}.{extension}",
                                  byte_size=len(best["data"]), width=signature["width"], height=signature["height"],
                                  local_status="available", source_sha256=best["record"].get("source_sha256", ""))
                    merged.pop("evicted_at", None)
                    merged.pop("eviction_reason", None)
                    destination = self._image_path(merged)
                    self._write_image_bytes(destination, best["data"])
                    obsolete.discard(destination)
                    if best["record"]["id"] != work_id:
                        merged.update(version=int(merged.get("version", 1)) + 1, upgraded_at=_now())
                merged.update(phash=signature["phash"], identity_signature=signature)
                merged.setdefault("version", 1)
                self._write(connection, "image", merged)
                for record in records[1:]:
                    alias = record["id"]
                    connection.execute("INSERT INTO image_alias(alias_id,work_id) VALUES(?,?) "
                                       "ON CONFLICT(alias_id) DO UPDATE SET work_id=excluded.work_id", (alias, work_id))
                    connection.execute("UPDATE image_alias SET work_id=? WHERE work_id=?", (work_id, alias))
                    self._write(connection, "image", {**record, "alias_of": work_id, "local_status": "merged"})
                    connection.execute("DELETE FROM image_identity WHERE work_id=?", (alias,))
                connection.execute("INSERT INTO image_identity(work_id,phash,signature,created_at) VALUES(?,?,?,?) "
                    "ON CONFLICT(work_id) DO UPDATE SET phash=excluded.phash,signature=excluded.signature",
                    (work_id, signature["phash"], json.dumps(signature, allow_nan=False), canonical["created_at"]))
            report["references"] = self._migrate_identity_references(connection)
            self._write(connection, "identity_migration", {**report, "id": uuid.uuid4().hex, "algorithm": "dct32-phash64-v1"})
            connection.commit()
            for path in obsolete:
                self._assert_managed_path(path)
                if path.exists():
                    path.unlink()
            return report

    def _migrate_identity_references(self, connection: sqlite3.Connection,
                                      proposed_aliases: Optional[Dict[str, str]] = None,
                                      write: bool = True) -> Dict[str, Any]:
        """Rewrite resolvable work references while preserving original IDs and prose."""
        changed: Dict[str, int] = {}

        def resolve(identifier: str) -> str:
            canonical = self._resolve_image_id(connection, identifier)
            return proposed_aliases.get(canonical, canonical) if proposed_aliases else canonical

        def preserve(record: Dict[str, Any], field: str, value: Any) -> None:
            if record.get(field) != value:
                record.setdefault("identity_original_references", {}).setdefault(field, record.get(field))
                record[field] = value

        def update_ids(record: Dict[str, Any], field: str) -> None:
            values = record.get(field)
            if isinstance(values, list):
                preserve(record, field, list(dict.fromkeys(resolve(value) for value in values)))
            elif isinstance(values, str) and values:
                preserve(record, field, resolve(values))

        records: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for kind in ("job", "diary", "observation", "feedback", "share", "preview", "exploration"):
            records[kind] = {row["id"]: json.loads(row["payload"]) for row in connection.execute(
                "SELECT id,payload FROM records WHERE kind=?", (kind,))}
        originals = {kind: {identifier: json.dumps(record, sort_keys=True) for identifier, record in rows.items()}
                     for kind, rows in records.items()}
        for kind, fields in (("job", ("image_ids", "share_planned_image_ids")), ("diary", ("image_ids",)),
                             ("observation", ("image_id",)), ("feedback", ("image_id", "image_ids", "candidate_image_ids"))):
            for record in records[kind].values():
                for field in fields:
                    update_ids(record, field)
        for share in records["share"].values():
            if share.get("image_id"):
                share["canonical_image_id"] = resolve(share["image_id"])
        for observation in records["observation"].values():
            work_id = observation.get("image_id") or observation.get("work_id")
            if work_id:
                observation["work_id"] = resolve(work_id)
                preview = records["preview"].get(observation.get("preview_id", ""))
                if preview is not None:
                    preview["work_id"] = observation["work_id"]
        for preview in records["preview"].values():
            if preview.get("work_id"):
                preview["work_id"] = resolve(preview["work_id"])

        def distinct_ids(ids: List[str], kind: str) -> Tuple[List[str], Dict[str, str], List[str]]:
            kept: List[str] = []
            work_ids: Dict[str, str] = {}
            unknown: List[str] = []
            for identifier in ids:
                work_id = records[kind].get(identifier, {}).get("work_id")
                if work_id:
                    if work_id in work_ids:
                        continue
                    work_ids[work_id] = identifier
                else:
                    unknown.append(identifier)
                kept.append(identifier)
            return kept, work_ids, unknown

        coverage = []
        for job in records["job"].values():
            if "preview_ids" not in job:
                continue
            old_preview_count = len(job.get("identity_original_references", {}).get("preview_ids", job["preview_ids"]))
            preview_ids, preview_works, unknown_previews = distinct_ids(job["preview_ids"], "preview")
            detail_ids, detail_works, unknown_details = distinct_ids(job.get("observation_ids", []), "observation")
            viewed_ids, viewed_works, _ = distinct_ids(job.get("viewed_preview_ids", []), "preview")
            preserve(job, "preview_ids", preview_ids)
            preserve(job, "observation_ids", detail_ids)
            if "viewed_preview_ids" in job:
                preserve(job, "viewed_preview_ids", viewed_ids)
                preserve(job, "inspected_image_count", len(viewed_ids))
            preserve(job, "preview_count", len(preview_ids))
            preserve(job, "detail_count", len(detail_ids))
            job.update(preview_work_ids=preview_works, viewed_work_ids=viewed_works)
            state = job.get("state")
            if isinstance(state, dict):
                for field, kind in (("preview_ids", "preview"), ("detail_ids", "observation"), ("viewed_ids", "preview")):
                    if field in state:
                        ids, _, _ = distinct_ids(state[field], kind)
                        preserve(state, field, ids)
                state["preview_work_ids"] = distinct_ids(state.get("viewed_ids", state.get("preview_ids", [])), "preview")[1]
            item = {"job_id": job["id"], "original_preview_count": old_preview_count,
                    "known_unique_previews": len(preview_works), "unknown_preview_count": len(unknown_previews),
                    "known_unique_details": len(detail_works), "unknown_detail_count": len(unknown_details),
                    "complete": not unknown_previews and not unknown_details}
            job["identity_coverage"] = item
            coverage.append(item)
            diary = records["diary"].get(job["id"])
            if diary is not None:
                preserve(diary, "preview_count", len(preview_ids))
                preserve(diary, "detail_count", len(detail_ids))
                diary["identity_coverage"] = item
        for exploration in records["exploration"].values():
            for field, kind in (("preview_ids", "preview"), ("observation_ids", "observation")):
                if field in exploration:
                    ids, _, _ = distinct_ids(exploration[field], kind)
                    preserve(exploration, field, ids)
        for kind, rows in records.items():
            for identifier, record in rows.items():
                if json.dumps(record, sort_keys=True) != originals[kind][identifier]:
                    if write:
                        self._write(connection, kind, record)
                    changed[kind] = changed.get(kind, 0) + 1
        return {"updated_records": changed, "job_coverage": coverage}

    def image_bytes(self, image_id: str) -> Tuple[bytes, str]:
        with self._lock, self._connect() as connection:
            image_id = self._resolve_image_id(connection, image_id)
            record = self._read(connection, "image", image_id)
            if record is None or record["local_status"] != "available":
                raise FileNotFoundError("Image is not in the local collection")
            return self._image_path(record).read_bytes(), record["mime_type"]

    def thumbnail(self, image_id: str) -> str:
        data, _ = self.image_bytes(image_id)
        with Image.open(BytesIO(data)) as image:
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=75, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    def _evict(self, connection: sqlite3.Connection, record: Dict[str, Any], reason: str) -> None:
        path = self._image_path(record)
        if path.exists():
            path.unlink()
        record.update(local_status="evicted", evicted_at=_now(), eviction_reason=reason)
        self._write(connection, "image", record)

    def cleanup(
        self,
        max_images: int,
        max_bytes: int,
        idle_days: int,
        needed_bytes: int = 0,
        needed_count: int = 0,
        protect_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = self._cleanup(
                connection, max_images, max_bytes, idle_days, needed_bytes, needed_count, protect_ids
            )
            connection.commit()
            return result

    def _cleanup(
        self,
        connection: sqlite3.Connection,
        max_images: int,
        max_bytes: int,
        idle_days: int,
        needed_bytes: int,
        needed_count: int,
        protect_ids: Optional[Set[str]],
    ) -> Dict[str, Any]:
        if min(max_images, max_bytes, idle_days, needed_bytes, needed_count) < 0:
            raise ValueError("Storage quotas and reservations must be nonnegative")
        if needed_count > max_images or needed_bytes > max_bytes:
            raise ValueError("Incoming image exceeds the configured collection capacity")
        available = self._available(connection)
        protected = {self._resolve_image_id(connection, image_id) for image_id in (protect_ids or set())}
        eligible = [record for record in available if not record.get("pinned", False) and record["id"] not in protected]
        eligible.sort(key=lambda record: (
            _retention_interest(record),
            _date(record["last_interest_at"]), record["id"],
        ))
        cutoff = datetime.now(timezone.utc) - timedelta(days=idle_days)
        selected: Dict[str, Tuple[Dict[str, Any], str]] = {
            record["id"]: (record, "idle")
            for record in eligible
            if record["human_interest"] is not None
            and _retention_interest(record) < 0.4
            and _date(record["last_interest_at"]) < cutoff
        }
        count = len(available) - len(selected)
        byte_size = sum(record["byte_size"] for record in available) - sum(
            record["byte_size"] for record, _ in selected.values()
        )
        for record in eligible:
            if count + needed_count <= max_images and byte_size + needed_bytes <= max_bytes:
                break
            if record["id"] not in selected:
                selected[record["id"]] = (record, "capacity")
                count -= 1
                byte_size -= record["byte_size"]
        if count + needed_count > max_images or byte_size + needed_bytes > max_bytes:
            # Check the complete plan first: a rejected admission never deletes a
            # useful collection without being able to make the requested space.
            raise ValueError("Collection capacity is occupied by pinned or protected images")
        for record, reason in selected.values():
            self._evict(connection, record, reason)
        return {
            "evicted_ids": list(selected), "evicted_count": len(selected),
            "image_count": count, "image_bytes": byte_size,
        }
