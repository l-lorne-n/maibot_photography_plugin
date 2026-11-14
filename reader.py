"""Read public photography sources through MaiBot's documented capabilities.

Search results and page text are evidence, never instructions. A page result is
not a visual observation; ``image`` returns successfully decoded pixels only.
"""

from __future__ import annotations

from base64 import b64decode
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import asyncio
import ipaddress
import re

from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from maibot_sdk import PluginContext


FETCH_API = "com.0-hz.fetch-url.fetch_url"
MAX_PAGE_CHARS = 12000
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMPORT_CHARS = 100_000
MAX_IMPORT_LINES = 500


class ReaderError(ValueError):
    """A readable source error, without upstream credential-bearing details."""


def sanitized_failure(value: Any) -> str:
    """Keep status/timeout evidence while excluding credentials from UI and logs."""
    if isinstance(value, dict):
        status = value.get("status_code", value.get("status"))
        details = next((value[key] for key in ("error", "message", "content", "detail") if value.get(key)), None)
        text = sanitized_failure(details) if details is not None else "上游未提供错误详情"
        if isinstance(status, int):
            text = f"HTTP {status}: {text}"
    else:
        text = str(value) if value is not None else "上游未提供错误详情"
        if isinstance(value, Exception) and not text.strip():
            text = type(value).__name__
    text = text[:20000]
    sensitive = re.compile(r"^(?:api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|auth(?:orization)?|password|secret|signature|sig)$", re.I)

    def redact_url(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            parsed = urlsplit(raw)
            netloc = "[REDACTED]@" + parsed.netloc.rsplit("@", 1)[-1] if "@" in parsed.netloc else parsed.netloc
            query = urlencode([(key, "REDACTED" if sensitive.fullmatch(key) else item)
                               for key, item in parse_qsl(parsed.query, keep_blank_values=True)])
            return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
        except ValueError:
            return "[无效网址已隐藏]"

    text = re.sub(r"https?://[^\s<>\"']+", redact_url, text, flags=re.I)
    text = re.sub(r"\bBearer\s+[^\s,;\"'<>]+", "Bearer [REDACTED]", text, flags=re.I)
    text = re.sub(r"\b(?:sk-|jina_)[A-Za-z0-9_-]+", "[REDACTED]", text, flags=re.I)
    credential = r"([\"']?\b(?:api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|authorization|proxy-authorization|password|secret|signature)\b[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
    text = re.sub(credential, lambda match: match.group(1) + "[REDACTED]", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()[:500]


def public_url(value: str, base_url: str = "") -> str:
    """Normalize a public HTTP(S) URL without resolving DNS in parsing code.

    Fetch URL additionally checks DNS results and every redirect immediately
    before requests. Its ``fetch.allow_private_networks`` must remain false.
    """
    value = unescape(value.strip())
    if not value or len(value) > 8192 or re.search(r"[\x00-\x20\x7f\\]", value):
        raise ReaderError("网址为空、过长或含有非法字符")
    try:
        parsed = urlsplit(urljoin(base_url, value) if base_url else value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ReaderError("网址格式无效") from exc
    if parsed.scheme.lower() not in {"https", "http"} or not host:
        raise ReaderError("仅接受公开的 HTTP(S) 网址")
    if parsed.username is not None or parsed.password is not None:
        raise ReaderError("摄影网址不能包含登录凭据")
    host = host.rstrip(".").lower()
    if "." not in host or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
        raise ReaderError("不能浏览本机或内网地址")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ReaderError("请使用公开网站的域名，不接受 IP 地址")
    if re.fullmatch(r"[0-9.]+", host) or host.startswith("0x"):
        raise ReaderError("不能使用数字形式的主机地址")
    if port not in {None, 80, 443}:
        raise ReaderError("摄影网站仅使用标准 HTTP(S) 端口")
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ReaderError("网站域名无效") from exc
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", ascii_host):
        raise ReaderError("网站域名无效")
    netloc = ascii_host
    if port is not None and not (parsed.scheme.lower() == "https" and port == 443) and not (
        parsed.scheme.lower() == "http" and port == 80
    ):
        netloc += f":{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _stable_id(prefix: str, text: str) -> str:
    return f"{prefix}_{sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _import_lines(content: str) -> List[str]:
    if not isinstance(content, str):
        raise ReaderError("导入内容必须是文本")
    if len(content) > MAX_IMPORT_CHARS:
        raise ReaderError("导入文本超过 100000 字符，请拆分后导入")
    lines = [line.strip() for line in content.lstrip("\ufeff").splitlines() if line.strip()]
    if len(lines) > MAX_IMPORT_LINES:
        raise ReaderError("单次最多导入 500 行")
    return lines


def parse_seeds(content: str) -> List[Dict[str, Any]]:
    """Preserve supplied names; listed photobooks receive explicit priority."""
    seeds: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for raw_line in _import_lines(content):
        line = re.sub(r"^\s*(?:[-*•]\s+|\d+[.、)]\s*)", "", raw_line)
        if line.startswith("#"):
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        author = parts[0].strip()
        if not author:
            raise ReaderError("摄影师姓名不能为空")
        works = [part.strip() for part in re.split(r"[,，、；;]|\s*&\s*", parts[1]) if part.strip()] if len(parts) == 2 else []
        if len(author) > 200 or any(len(work) > 500 for work in works):
            raise ReaderError("摄影师或作品名称过长，请每位摄影师单独一行")
        seed_id = _stable_id("seed", author)
        if seed_id in seen:
            prior = seen[seed_id]
            prior["works"] = list(dict.fromkeys([*prior["works"], *works]))
            prior["raw_line"] += "\n" + raw_line
            prior["priority"] = "named_works" if prior["works"] else "author_style"
            continue
        record = {
            "id": seed_id,
            "author": author,
            "works": list(dict.fromkeys(works)),
            "raw_line": raw_line,
            "status": "pending",
            "priority": "named_works" if works else "author_style",
            "identity_status": "unverified",
        }
        seeds.append(record)
        seen[seed_id] = record
    if not seeds:
        raise ReaderError("没有找到摄影师；请按“姓名：作品一，作品二”填写")
    for seed in seeds:
        seed["work_items"] = seed_work_items(seed)
    return seeds


def seed_work_items(seed: Dict[str, Any]) -> List[Dict[str, str]]:
    """Give original user book labels stable IDs independent of model translations."""
    return [{"id": _stable_id("work", seed["id"] + "\0" + title), "title": title}
            for title in seed["works"]]


def parse_sites(content: str) -> List[Dict[str, Any]]:
    """Accept the user's numbered site notes and one-URL-per-line imports."""
    sites: List[Dict[str, Any]] = []
    seen = set()
    pending_name = ""
    pattern = re.compile(r"(?:https?://|www\.)[^\s<>\"'，。；]+|(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s<>\"'，。；]*)?")
    for line in _import_lines(content):
        heading = re.match(r"^\s*\d+[.、)]\s*([^:：]+)[:：]", line)
        if heading:
            pending_name = heading.group(1).strip()
        matches = list(pattern.finditer(line))
        for match in matches:
            value = match.group(0).rstrip(".,;:!?)）】]")
            if not value.lower().startswith(("https://", "http://")):
                value = "https://" + value
            url = public_url(value)
            if url in seen:
                continue
            seen.add(url)
            sites.append({
                "id": _stable_id("source", url),
                "url": url,
                "name": pending_name if len(matches) == 1 and pending_name else urlsplit(url).hostname,
                "enabled": True,
                "bot_score": 0.5,
                "human_score": None,
                "observations": 0,
                "origin": "user",
            })
        if matches:
            pending_name = ""
    if not sites:
        raise ReaderError("没有找到网站地址；请输入域名或 HTTP(S) 网址")
    return sites


def _markdown_targets(text: str) -> Iterator[Tuple[bool, str, str]]:
    """Read inline/reference Markdown links, including balanced URL parentheses."""
    references: Dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*\[([^\]]+)\]:\s*<?([^\s>]+)>?", text):
        references[match.group(1).casefold()] = match.group(2)
    for match in re.finditer(r"!?\[", text):
        image = match.group(0).startswith("!")
        label_start = match.end()
        bracket_depth = 1
        cursor = label_start
        while cursor < len(text) and bracket_depth:
            char = text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
            cursor += 1
        if bracket_depth:
            continue
        label = text[label_start:cursor - 1]
        if cursor >= len(text):
            continue
        if text[cursor] == "[":
            ref_end = text.find("]", cursor + 1)
            if ref_end != -1:
                reference = (text[cursor + 1:ref_end] or label).casefold()
                if reference in references:
                    yield image, label, references[reference]
            continue
        if text[cursor] != "(":
            continue
        cursor += 1
        target_start = cursor
        depth = 1
        while cursor < len(text) and depth:
            char = text[cursor]
            if char == "\\":
                cursor += 2
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            cursor += 1
        if depth:
            continue
        destination = text[target_start:cursor - 1].strip()
        if destination.startswith("<") and ">" in destination:
            destination = destination[1:destination.index(">")]
        else:
            destination = re.split(r"\s+[\"']", destination, maxsplit=1)[0]
        yield image, label, re.sub(r"\\([()])", r"\1", destination)


def _srcset_best(value: str) -> Optional[str]:
    candidates: List[Tuple[float, str]] = []
    for candidate in value.split(","):
        parts = candidate.strip().split()
        if not parts:
            continue
        weight = 1.0
        if len(parts) > 1 and re.fullmatch(r"\d+(?:\.\d+)?[wx]", parts[-1]):
            weight = float(parts[-1][:-1])
        candidates.append((weight, parts[0]))
    return max(candidates, default=(0.0, ""))[1] or None


class _HTMLCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.images: List[Tuple[str, str]] = []
        self.links: List[Tuple[str, str]] = []
        self.title_parts: List[str] = []
        self.in_title = False
        self.anchor_url = ""
        self.anchor_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        values = {name: value for name, value in attrs if value is not None}
        if tag == "title":
            self.in_title = True
        if tag == "a" and "href" in values:
            self.anchor_url = values["href"]
            self.anchor_parts = []
        if tag in {"img", "source"}:
            candidates = [
                _srcset_best(values.get("data-srcset", "")),
                _srcset_best(values.get("srcset", "")),
                values.get("data-src"),
                values.get("data-lazy-src"),
                values.get("src"),
            ]
            for candidate in candidates:
                if candidate and not candidate.startswith("data:"):
                    self.images.append((candidate, values.get("alt", "")))
                    break

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if tag == "a" and self.anchor_url:
            self.links.append((self.anchor_url, "".join(self.anchor_parts).strip()))
            self.anchor_url = ""
            self.anchor_parts = []

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        if self.anchor_url:
            self.anchor_parts.append(data)


def extract_page(content: str, url: str, title: str = "") -> Dict[str, Any]:
    """Extract only URLs actually present in the fetched document."""
    content = content[:MAX_PAGE_CHARS]
    html = _HTMLCollector()
    html.feed(content)
    raw_links = html.links
    raw_images = html.images
    for is_image, label, target in _markdown_targets(content):
        (raw_images if is_image else raw_links).append((target, label))
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    seen_links = set()
    seen_images = set()
    for raw_values, records, seen, label_key, limit in (
        (raw_links, links, seen_links, "title", 40),
        (raw_images, images, seen_images, "alt", 30),
    ):
        for target, label in raw_values:
            if target.startswith("#"):
                continue
            try:
                normalized = public_url(target, url)
            except ReaderError:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            records.append({"url": normalized, label_key: unescape(label)[:300]})
            if len(records) == limit:
                break
    if not title:
        title = "".join(html.title_parts).strip()
    if not title:
        heading = re.search(r"(?m)^(?:Title:\s*|#\s+)(.+)$", content)
        title = heading.group(1).strip() if heading else urlsplit(url).hostname
    return {"url": url, "title": title[:300], "text": content, "links": links, "images": images}


def _decode_image(encoded: str, claimed_mime: str, url: str) -> Dict[str, Any]:
    if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ReaderError("图片超过 15 MiB，本轮不读取")
    try:
        data = b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ReaderError("Fetch URL 返回的图片编码无效") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ReaderError("图片为空或超过 15 MiB")
    try:
        with Image.open(BytesIO(data)) as picture:
            width, height = picture.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ReaderError("图片像素过多，本轮不读取")
            actual_mime = Image.MIME.get(picture.format)
            picture.verify()
        # verify checks the container; load verifies that complete pixels decode.
        with Image.open(BytesIO(data)) as picture:
            picture.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ReaderError("下载内容不能解码为完整图片") from exc
    if actual_mime is None:
        raise ReaderError("图片格式无法识别")
    normalized_claim = {"image/jpg": "image/jpeg"}.get(claimed_mime, claimed_mime)
    if normalized_claim != actual_mime:
        raise ReaderError("图片实际格式与 Fetch URL 返回的格式不一致")
    return {"data": data, "mime_type": actual_mime, "width": width, "height": height, "url": url}


def image_content_hash(image: Dict[str, Any]) -> str:
    """Hash decoded first-frame pixels, so identical PNG/JPEG-container aliases deduplicate."""
    with Image.open(BytesIO(image["data"])) as original:
        original.load()
        pixels = ImageOps.exif_transpose(original).convert("RGBA")
        digest = sha256(f"{pixels.width}x{pixels.height}:RGBA:".encode("ascii"))
        digest.update(pixels.tobytes())
    return digest.hexdigest()


def prepare_vision_image(image: Dict[str, Any], *, max_side: int = 1280,
                         max_bytes: int = 750000) -> Dict[str, Any]:
    """Prepare decoded JPEG pixels within the Host RPC image prompt budget.

    The source image remains available to collection storage. This derived
    in-memory view defaults to 1280 pixels on its longest edge and 750 KB,
    leaving room for base64 encoding and page evidence in the RPC frame.
    """
    source = image["data"]
    if not 1 <= max_side <= 4096 or not 1024 <= max_bytes <= MAX_IMAGE_BYTES:
        raise ReaderError("视觉输入压缩预算无效")
    if not isinstance(source, bytes) or not source or len(source) > MAX_IMAGE_BYTES:
        raise ReaderError("视觉输入必须是大小受限的已下载图片")
    try:
        with Image.open(BytesIO(source)) as original:
            if original.width * original.height > MAX_IMAGE_PIXELS:
                raise ReaderError("图片像素过多，无法生成视觉输入")
            original.load()
            picture = ImageOps.exif_transpose(original)
            original_width, original_height = picture.size
            if picture.mode in {"RGBA", "LA"} or "transparency" in picture.info:
                rgba = picture.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                picture = background.convert("RGB")
            else:
                picture = picture.convert("RGB")
            picture.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            while True:
                buffer = BytesIO()
                picture.save(buffer, format="JPEG", quality=80, optimize=True)
                data = buffer.getvalue()
                if len(data) <= max_bytes:
                    break
                size = (max(1, round(picture.width * 0.85)), max(1, round(picture.height * 0.85)))
                picture = picture.resize(size, Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ReaderError("无法生成有效的视觉输入图片") from exc
    return {
        **image,
        "data": data,
        "mime_type": "image/jpeg",
        "width": picture.width,
        "height": picture.height,
        "original_width": original_width,
        "original_height": original_height,
    }


class Reader:
    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        query = query.strip()
        if not query or len(query) > 128:
            raise ReaderError("搜索词须为 1–128 字符，以兼容当前 DeepSeek 搜索服务")
        if not 1 <= limit <= 10:
            raise ReaderError("单次搜索数量须为 1–10")
        try:
            response = await self.ctx.web.search(query, max_results=limit)
        except Exception as exc:
            raise ReaderError(f"Host 搜索调用失败（{type(exc).__name__}）：{sanitized_failure(exc)}") from None
        # PluginContext unwraps the Host's success/result envelope for web.search.
        if isinstance(response, dict) and (response.get("success") is False or "error" in response):
            raise ReaderError("Host 搜索失败：" + sanitized_failure(response))
        if not isinstance(response, dict) or not isinstance(response.get("results"), list):
            raise ReaderError("Host 搜索结果结构不符合当前 SDK 接口")
        hits: List[Dict[str, str]] = []
        for hit in response["results"]:
            if not isinstance(hit, dict) or not isinstance(hit.get("url"), str):
                raise ReaderError("Host 搜索结果缺少有效网址")
            try:
                normalized = public_url(hit["url"])
            except ReaderError:
                continue
            hits.append({
                "url": normalized,
                "title": str(hit.get("title", ""))[:300],
                "snippet": str(hit.get("snippet", ""))[:1500],
            })
            if len(hits) == limit:
                break
        return hits

    async def _fetch(self, url: str, return_image: bool) -> Dict[str, Any]:
        normalized = public_url(url)
        try:
            response = await self.ctx.api.call(
                FETCH_API,
                version="1",
                url=normalized,
                start_char=0,
                end_char=MAX_PAGE_CHARS,
                on_exceed="truncate",
                return_image=return_image,
            )
        except Exception as exc:
            raise ReaderError(f"Fetch URL 调用失败（{type(exc).__name__}）：{sanitized_failure(exc)}") from None
        if not isinstance(response, dict) or response.get("success") is not True:
            raise ReaderError("Fetch URL 未能读取该网址：" + sanitized_failure(response))
        return response

    async def page(self, url: str) -> Dict[str, Any]:
        response = await self._fetch(url, return_image=False)
        content = response.get("content")
        if not isinstance(content, str):
            raise ReaderError("Fetch URL 没有返回网页正文")
        if response.get("processed") == "summarized":
            raise ReaderError("Fetch URL 返回了摘要，无法据此提取实际页面链接")
        final_url = public_url(response.get("final_url", url))
        metadata = response.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ReaderError("Fetch URL 页面元信息无效")
        title = metadata.get("title", "")
        if not isinstance(title, str):
            raise ReaderError("Fetch URL 页面标题无效")
        result = await asyncio.to_thread(extract_page, content, final_url, title)
        result["provider"] = response.get("provider", "")
        result["truncated"] = response.get("processed") == "truncated" or len(content) > MAX_PAGE_CHARS
        return result

    async def image(self, url: str) -> Dict[str, Any]:
        response = await self._fetch(url, return_image=True)
        items = response.get("content_items")
        if not isinstance(items, list):
            raise ReaderError("该网址未返回图片字节，文字描述不计作看过图片")
        for item in items:
            if not isinstance(item, dict) or item.get("content_type") != "image":
                continue
            encoded = item.get("data")
            mime_type = item.get("mime_type")
            if not isinstance(encoded, str) or not isinstance(mime_type, str):
                raise ReaderError("Fetch URL 返回的图片数据不完整")
            final_url = public_url(response.get("final_url", url))
            return await asyncio.to_thread(_decode_image, encoded, mime_type, final_url)
        raise ReaderError("该网址未返回图片字节，文字描述不计作看过图片")
