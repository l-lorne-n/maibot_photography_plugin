"""Small, CPU-only photographic identity signatures; no model or vector service."""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from math import cos, pi
from statistics import median
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageOps, ImageStat


PHASH_ALGORITHM = "dct32-phash64-v1"
DEFAULT_PHASH_DISTANCE = 6


@lru_cache(maxsize=1)
def _cosines() -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(cos((2 * position + 1) * frequency * pi / 64)
                       for position in range(32)) for frequency in range(8))


def _phash(gray: Image.Image) -> str:
    pixels = gray.resize((32, 32), Image.Resampling.LANCZOS).tobytes()
    basis = _cosines()
    horizontal = [[sum(pixels[row * 32 + col] * basis[frequency][col] for col in range(32))
                   for frequency in range(8)] for row in range(32)]
    coefficients = [sum(horizontal[row][x] * basis[y][row] for row in range(32))
                    for y in range(8) for x in range(8)]
    coefficients = [0.0 if abs(value) < 1e-7 else value for value in coefficients]
    threshold = median(coefficients[1:])
    bits = 0
    for index, value in enumerate(coefficients):
        bits = (bits << 1) | int(index != 0 and value > threshold)
    return f"{bits:016x}"


def _detail_score(gray: Image.Image) -> float:
    """Compare edge detail at a common scale, so simple enlargement earns no bonus."""
    pixels = gray.resize((128, 128), Image.Resampling.LANCZOS).tobytes()
    total = 0
    for y in range(1, 127):
        for x in range(1, 127):
            pos = y * 128 + x
            total += abs(4 * pixels[pos] - pixels[pos - 1] - pixels[pos + 1]
                         - pixels[pos - 128] - pixels[pos + 128])
    return round(total / (126 * 126), 6)


def fingerprint(data: bytes) -> Dict[str, Any]:
    """Decode one frame and derive a 64-bit DCT pHash plus conservative match guards.

    pHash is intentionally a near-duplicate heuristic, not semantic similarity.
    Aspect ratio and coarse color reject common collisions such as solid-color
    placeholders. Call in a worker thread along with the Store operation.
    """
    with Image.open(BytesIO(data)) as opened:
        if opened.width * opened.height > 40_000_000:
            raise ValueError("Image exceeds the identity pixel limit")
        opened.load()
        quantization = opened.quantization if opened.format == "JPEG" else None
        quantization_mean = (sum(sum(table) for table in quantization.values()) /
                             sum(len(table) for table in quantization.values())) if quantization else None
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        digest = sha256(f"{image.width}x{image.height}:RGBA:".encode("ascii"))
        digest.update(image.tobytes())
        canvas = Image.new("RGBA", image.size, "white")
        canvas.alpha_composite(image)
        rgb = canvas.convert("RGB")
        gray = rgb.convert("L")
        reduced = rgb.resize((32, 32), Image.Resampling.LANCZOS)
        luma_stddev = ImageStat.Stat(reduced.convert("L")).stddev[0]
        return {
            "algorithm": PHASH_ALGORITHM, "phash": _phash(gray), "pixel_hash": digest.hexdigest(),
            "width": image.width, "height": image.height,
            "mean_rgb": [round(value, 3) for value in ImageStat.Stat(reduced).mean],
            "luma_stddev": round(luma_stddev, 6), "detail_score": _detail_score(gray),
            "quantization_mean": round(quantization_mean, 6) if quantization_mean is not None else None,
            "lossless_source": opened.format in {"PNG", "BMP"},
        }


def hamming_distance(first: str, second: str) -> int:
    if len(first) != 16 or len(second) != 16:
        raise ValueError("pHash must contain 16 hexadecimal characters")
    return (int(first, 16) ^ int(second, 16)).bit_count()


def match_distance(first: Dict[str, Any], second: Dict[str, Any],
                   max_distance: int = DEFAULT_PHASH_DISTANCE) -> Optional[int]:
    if isinstance(max_distance, bool) or not isinstance(max_distance, int) or not 0 <= max_distance <= 64:
        raise ValueError("pHash distance must be an integer in 0..64")
    if first["algorithm"] != PHASH_ALGORITHM or second["algorithm"] != PHASH_ALGORITHM:
        raise ValueError("Unsupported image identity algorithm")
    if first["pixel_hash"] == second["pixel_hash"]:
        return 0
    ratio_a = first["width"] / first["height"]
    ratio_b = second["width"] / second["height"]
    if abs(ratio_a / ratio_b - 1) > 0.035:
        return None
    color_difference = max(abs(a - b) for a, b in zip(first["mean_rgb"], second["mean_rgb"]))
    flat_a, flat_b = first["luma_stddev"] < 2, second["luma_stddev"] < 2
    if flat_a != flat_b or color_difference > (3 if flat_a else 28):
        return None
    distance = hamming_distance(first["phash"], second["phash"])
    return distance if distance <= max_distance else None


def is_clearer(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """Prefer more usable pixels without a clear loss of common-scale detail.

    At equal size, known lossless input or lower JPEG quantization can improve
    the stored version. Unknown codec quality is not guessed from file size.
    """
    if candidate["width"] < current["width"] or candidate["height"] < current["height"]:
        return False
    area = candidate["width"] * candidate["height"]
    previous_area = current["width"] * current["height"]
    if candidate["detail_score"] < current["detail_score"] * 0.85:
        return False
    if area >= previous_area * 1.15:
        return True
    if candidate["lossless_source"] and not current["lossless_source"]:
        return True
    old_quantization, new_quantization = current["quantization_mean"], candidate["quantization_mean"]
    return old_quantization is not None and new_quantization is not None and new_quantization < old_quantization * 0.85
