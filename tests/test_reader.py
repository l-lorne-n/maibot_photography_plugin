from base64 import b64encode
from io import BytesIO
from types import SimpleNamespace
import asyncio

from PIL import Image
from maibot_sdk.context import PluginContext
import pytest

from reader import FETCH_API, Reader, ReaderError, extract_page, image_content_hash, parse_seeds, parse_sites, prepare_vision_image, public_url, sanitized_failure, seed_work_items


def test_seed_books_are_separate_priorities_without_correcting_supplied_names():
    content = "\ufeff陈川端：世界在巨蛇的腹中\n川内伦子：Utatane，Hanabi，Illuminance\n陈哲：蜜蜂 & 可承受的\n上田义彦\n川内伦子：Illuminance，新作\n"
    seeds = parse_seeds(content)
    assert len(seeds) == 4
    assert seeds[0]["author"] == "陈川端"
    assert seeds[0]["identity_status"] == "unverified"
    assert seeds[1]["works"] == ["Utatane", "Hanabi", "Illuminance", "新作"]
    assert seeds[2]["works"] == ["蜜蜂", "可承受的"]
    assert seeds[3]["priority"] == "author_style"
    assert parse_seeds("陈川端：别的作品")[0]["id"] == seeds[0]["id"]
    assert seeds[0]["raw_line"] == "陈川端：世界在巨蛇的腹中"


def test_numbered_site_notes_keep_human_names_and_paths():
    content = "1. Aint-Bad: 独立出版平台。\n网址：aint-bad.com\n2. Ignant Design: 设计。\n网址：ignant.com/category/design/\nhttps://www.broad.community/\naint-bad.com"
    sites = parse_sites(content)
    assert len(sites) == 3
    assert sites[0]["name"] == "Aint-Bad"
    assert sites[0]["url"] == "https://aint-bad.com/"
    assert sites[1]["name"] == "Ignant Design"
    assert sites[1]["url"] == "https://ignant.com/category/design/"
    assert all(site["human_score"] is None and site["observations"] == 0 for site in sites)


@pytest.mark.parametrize("url", [
    "file:///C:/secret", "http://127.0.0.1/x", "http://192.168.1.2/", "http://[::1]/",
    "https://localhost/", "http://machine.local/", "http://internal/", "http://2130706433/",
    "https://user:secret@example.com/", "https://example.com:8001/", "https://example.com\\@localhost/",
])
def test_public_urls_reject_nonpublic_or_credential_targets(url):
    with pytest.raises(ReaderError):
        public_url(url)


def test_url_normalization_preserves_query_and_relative_path():
    assert public_url("../work/photo.jpg?size=large&amp;quality=80#x", "https://Example.com/books/view") == "https://example.com/work/photo.jpg?size=large&quality=80"
    assert public_url("//cdn.example.com/image.jpg", "https://example.com/") == "https://cdn.example.com/image.jpg"


def test_markdown_and_html_extract_only_real_public_links_and_largest_srcset():
    content = '''# A real photographer
[Series](/series/book_(2020))
![Image 1: river](../images/river_(1).jpg "River")
[![Book](https://cdn.example.com/book.jpg)](/books/one)
![Archive image][archive]
[archive]: https://cdn.example.com/archive.jpg
<img src="tiny.jpg" srcset="small.jpg 320w, large.jpg 1400w" alt="portrait">
<img src="data:image/gif;base64,AAAA" data-src="lazy.jpg" alt="lake">
<a href="https://other.example.com/artist">Artist homepage</a>
[Local](http://127.0.0.1/secret)
![Credential](https://user:pass@example.com/image.jpg)
'''
    page = extract_page(content, "https://example.com/article/view/")
    image_urls = {item["url"] for item in page["images"]}
    links = {item["url"] for item in page["links"]}
    assert "https://example.com/article/images/river_(1).jpg" in image_urls
    assert "https://example.com/article/view/large.jpg" in image_urls
    assert "https://example.com/article/view/lazy.jpg" in image_urls
    assert "https://cdn.example.com/archive.jpg" in image_urls
    assert "https://example.com/series/book_(2020)" in links
    assert "https://example.com/books/one" in links
    assert all("127.0.0.1" not in value and "user:pass" not in value for value in [*image_urls, *links])
    assert page["title"] == "A real photographer"


def test_page_candidate_limits_and_deduplication():
    content = "\n".join(f"[Article {i}](https://example.com/{i}) ![Photo {i}](https://cdn.example.com/{i}.jpg)" for i in range(100))
    page = extract_page(content, "https://example.com/")
    assert len(page["links"]) == 40
    assert len(page["images"]) == 30
    assert len(page["text"]) <= 12000


def test_reader_search_unwraps_current_host_result():
    calls = []

    async def rpc(method, plugin_id, payload, **kwargs):
        assert method == "cap.call"
        assert payload["capability"] == "web.search"
        calls.append((payload["args"]["query"], payload["args"]["max_results"]))
        return {"success": True, "result": {"results": [{"title": "Book", "url": "https://example.com/book", "snippet": "A photobook"}]}}

    reader = Reader(PluginContext("lorne.photography.test", rpc_call=rpc))
    results = asyncio.run(reader.search("photographer book", 3))
    assert calls == [("photographer book", 3)]
    assert results[0]["url"] == "https://example.com/book"


def test_reader_fetch_declares_real_api_and_preserves_original_document():
    calls = []

    async def call(name, **kwargs):
        calls.append((name, kwargs))
        return {"success": True, "content": "# Work\n![Work](/photo.jpg)", "final_url": "https://example.com/work/", "provider": "jina", "metadata": {}, "processed": "full"}

    reader = Reader(SimpleNamespace(api=SimpleNamespace(call=call)))
    page = asyncio.run(reader.page("https://example.com/work/"))
    assert calls[0][0] == FETCH_API
    assert calls[0][1]["version"] == "1"
    assert calls[0][1]["on_exceed"] == "truncate"
    assert calls[0][1]["return_image"] is False
    assert page["images"] == [{"url": "https://example.com/photo.jpg", "alt": "Work"}]
    assert page["provider"] == "jina"


def test_reader_real_pixels_and_bad_or_small_images():
    def image_response(size=(640, 400), mime="image/png"):
        buffer = BytesIO()
        Image.new("RGB", size, "blue").save(buffer, "PNG")
        return {"success": True, "content_items": [{"content_type": "image", "data": b64encode(buffer.getvalue()).decode("ascii"), "mime_type": mime}]}

    responses = [image_response(), image_response((1000, 150)), image_response(mime="image/jpeg"), {"success": True, "content": "A lovely photograph"}]

    async def call(name, **kwargs):
        assert kwargs["return_image"] is True
        return responses.pop(0)

    reader = Reader(SimpleNamespace(api=SimpleNamespace(call=call)))
    result = asyncio.run(reader.image("https://example.com/photo.png"))
    assert (result["width"], result["height"], result["mime_type"]) == (640, 400, "image/png")
    assert result["data"].startswith(b"\x89PNG")
    small = asyncio.run(reader.image("https://example.com/banner.png"))
    assert (small["width"], small["height"]) == (1000, 150)
    assert prepare_vision_image(small)["original_height"] == 150
    with pytest.raises(ReaderError, match="不一致"):
        asyncio.run(reader.image("https://example.com/wrong.png"))
    with pytest.raises(ReaderError, match="文字描述不计作"):
        asyncio.run(reader.image("https://example.com/not-an-image"))


def test_upstream_errors_do_not_copy_credential_bearing_exception_text():
    async def call(name, **kwargs):
        raise RuntimeError("Authorization: Bearer VERY_SECRET_KEY")

    reader = Reader(SimpleNamespace(api=SimpleNamespace(call=call)))
    with pytest.raises(ReaderError) as error:
        asyncio.run(reader.page("https://example.com/"))
    assert "VERY_SECRET_KEY" not in str(error.value)
    assert "RuntimeError" in str(error.value)


def test_vision_input_compresses_real_large_pixels_and_keeps_original_metadata():
    output = BytesIO()
    Image.effect_noise((2400, 1800), 100).convert("RGB").save(output, format="PNG")
    original = output.getvalue()
    prepared = prepare_vision_image({"data": original, "mime_type": "image/png", "url": "https://example.com/photo.png"})
    assert len(prepared["data"]) <= 750_000
    assert max(prepared["width"], prepared["height"]) <= 1280
    assert (prepared["original_width"], prepared["original_height"]) == (2400, 1800)
    assert prepared["url"] == "https://example.com/photo.png"
    assert prepared["mime_type"] == "image/jpeg"
    with Image.open(BytesIO(prepared["data"])) as decoded:
        decoded.load()
        assert decoded.size == (prepared["width"], prepared["height"])


def test_vision_input_composites_transparency_on_white():
    output = BytesIO()
    Image.new("RGBA", (400, 400), (0, 0, 0, 0)).save(output, format="PNG")
    prepared = prepare_vision_image({"data": output.getvalue(), "mime_type": "image/png"})
    with Image.open(BytesIO(prepared["data"])) as decoded:
        assert min(decoded.convert("RGB").getpixel((200, 200))) > 250


def test_preview_compression_has_separate_budget_and_content_hash_ignores_png_container():
    pixels = Image.effect_noise((1800, 1200), 100).convert("RGB")
    first, second = BytesIO(), BytesIO()
    pixels.save(first, format="PNG", compress_level=1)
    pixels.save(second, format="PNG", compress_level=9)
    one = {"data": first.getvalue(), "mime_type": "image/png"}
    two = {"data": second.getvalue(), "mime_type": "image/png"}
    assert one["data"] != two["data"]
    assert image_content_hash(one) == image_content_hash(two)
    preview = prepare_vision_image(one, max_side=768, max_bytes=250000)
    assert len(preview["data"]) <= 250000
    assert max(preview["width"], preview["height"]) <= 768


def test_seed_work_ids_preserve_user_titles_and_are_stable_across_added_books():
    first = parse_seeds("彭可：盐水池塘")[0]
    second = parse_seeds("彭可：盐水池塘，其他作品")[0]
    assert first["work_items"][0] == second["work_items"][0]
    assert first["work_items"][0]["title"] == "盐水池塘"
    assert seed_work_items(first) == first["work_items"]


def test_sanitized_failure_keeps_status_timeout_and_redacts_credentials():
    message = ('HTTP 429 read timeout after 30s; Authorization: Bearer TEST_BEARER '
               'api_key="TEST_API_KEY"; token=TEST_TOKEN; sk-test-secret jina_test_secret '
               'https://testuser:testpass@example.com/page?api%5Fkey=TEST_QUERY&status=429&access_token=TEST_ACCESS')
    safe = sanitized_failure(message)
    assert "HTTP 429" in safe and "read timeout after 30s" in safe and "status=429" in safe
    assert "[REDACTED]" in safe
    for secret in ("TEST_BEARER", "TEST_API_KEY", "TEST_TOKEN", "sk-test-secret", "jina_test_secret", "TEST_QUERY", "TEST_ACCESS", "testuser", "testpass"):
        assert secret not in safe
    assert len(sanitized_failure("HTTP 503 " + "unavailable " * 100)) <= 500


def test_failure_response_preserves_upstream_status_and_safe_content():
    async def call(name, **kwargs):
        return {"success": False, "content": "Image CDN returned HTTP 403; token=FAKE_SECRET"}

    reader = Reader(SimpleNamespace(api=SimpleNamespace(call=call)))
    with pytest.raises(ReaderError) as error:
        asyncio.run(reader.page("https://example.com/work"))
    assert "HTTP 403" in str(error.value)
    assert "FAKE_SECRET" not in str(error.value)


def test_sdk_search_failure_envelope_keeps_status_without_key():
    async def rpc(method, plugin_id, payload, **kwargs):
        return {"success": False, "error": {"message": "HTTP 429 rate limit; api_key=FAKE_SEARCH_KEY"}}

    reader = Reader(PluginContext("lorne.photography.test", rpc_call=rpc))
    with pytest.raises(ReaderError) as error:
        asyncio.run(reader.search("Test Artist"))
    assert "HTTP 429" in str(error.value)
    assert "FAKE_SEARCH_KEY" not in str(error.value)
