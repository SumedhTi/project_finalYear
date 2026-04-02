from pathlib import Path
import io

import file_handling
import huffman_coding


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "IO" / "Outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_project_path(user_path: str) -> Path:
    """Resolve a user path from either cwd or project root."""
    candidate = Path(user_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def read_target_percentage() -> float | None:
    raw = input(
        "Target compression percentage (optional, e.g. 10 or 30.5): "
    ).strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        print("Invalid percentage. Continuing without a target percentage.")
        return None
    if value < 0 or value > 100:
        print("Percentage must be between 0 and 100. Continuing without a target.")
        return None
    return value


def calculate_reduction_percentage(original_size: int, compressed_size: int) -> float:
    if original_size == 0:
        return 0.0
    return (1 - (compressed_size / original_size)) * 100


def compress_image_bytes(image_bytes: bytes) -> bytes:
    """Compress image bytes using Huffman coding."""
    # Convert bytes to bit string
    bit_string = ""
    for byte in image_bytes:
        bits = bin(byte)[2:].rjust(8, '0')
        bit_string += bits
    
    # Compress using Huffman
    compressed_bit_string = huffman_coding.compress(bit_string)
    
    # Convert back to bytes
    compressed_bytes = b""
    for i in range(0, len(compressed_bit_string), 8):
        byte_str = compressed_bit_string[i:i+8]
        if len(byte_str) < 8:
            byte_str = byte_str.ljust(8, '0')  # Pad if necessary
        compressed_bytes += bytes([int(byte_str, 2)])
    
    return compressed_bytes


def decompress_image_bytes(compressed_bytes: bytes, original_size: int) -> bytes:
    """Decompress image bytes using Huffman coding."""
    # This would need the Huffman tree, but for simplicity, assuming we have it
    # For now, just return as is, but ideally implement decompression
    return compressed_bytes  # Placeholder


def try_jpeg_target_compression(
    image_path: Path, target_percentage: float
) -> tuple[bool, str]:
    """Try lossy JPEG compression to meet a target size reduction percentage."""
    try:
        from PIL import Image
    except ImportError:
        return (
            False,
            "Pillow is not installed. Run: pip install pillow (or I can install it for you).",
        )

    original_size = image_path.stat().st_size
    target_size = int(original_size * (1 - (target_percentage / 100)))

    if target_size <= 0:
        return (
            False,
            "Target is too aggressive for file-based compression. Use a value less than 100%.",
        )

    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        best_meeting_data = None
        best_meeting_quality = None
        closest_data = None
        closest_quality = None
        closest_distance = float("inf")

        low, high = 1, 95
        while low <= high:
            quality = (low + high) // 2
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            size = len(data)
            distance = abs(size - target_size)

            if distance < closest_distance:
                closest_distance = distance
                closest_data = data
                closest_quality = quality

            if size <= target_size:
                best_meeting_data = data
                best_meeting_quality = quality
                low = quality + 1
            else:
                high = quality - 1

    output_path = OUTPUT_DIR / "target_compressed.jpg"
    chosen_data = best_meeting_data if best_meeting_data is not None else closest_data
    chosen_quality = (
        best_meeting_quality if best_meeting_quality is not None else closest_quality
    )

    if chosen_data is None or chosen_quality is None:
        return False, "Could not generate JPEG output for target compression."

    with open(output_path, "wb") as out_file:
        out_file.write(chosen_data)

    compressed_size = len(chosen_data)
    achieved = calculate_reduction_percentage(original_size, compressed_size)

    if best_meeting_data is not None:
        return (
            True,
            "Fallback JPEG used. "
            f"Target met at quality {chosen_quality}. "
            f"Achieved reduction: {round(achieved, 4)}%",
        )

    return (
        False,
        "Fallback JPEG used, but target still not met due to content limits. "
        f"Best quality {chosen_quality}, achieved reduction: {round(achieved, 4)}%",
    )
        from PIL import Image
    except ImportError:
        return (
            False,
            "Pillow is not installed. Run: pip install pillow (or I can install it for you).",
        )

    original_size = image_path.stat().st_size
    target_size = int(original_size * (1 - (target_percentage / 100)))

    if target_size <= 0:
        return (
            False,
            "Target is too aggressive for file-based compression. Use a value less than 100%.",
        )

    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        best_meeting_data = None
        best_meeting_quality = None
        closest_data = None
        closest_quality = None
        closest_distance = float("inf")

        low, high = 1, 95
        while low <= high:
            quality = (low + high) // 2
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            size = len(data)
            distance = abs(size - target_size)

            if distance < closest_distance:
                closest_distance = distance
                closest_data = data
                closest_quality = quality

            if size <= target_size:
                best_meeting_data = data
                best_meeting_quality = quality
                low = quality + 1
            else:
                high = quality - 1

    output_path = OUTPUT_DIR / "target_compressed.jpg"
    chosen_data = best_meeting_data if best_meeting_data is not None else closest_data
    chosen_quality = (
        best_meeting_quality if best_meeting_quality is not None else closest_quality
    )

    if chosen_data is None or chosen_quality is None:
        return False, "Could not generate JPEG output for target compression."

    with open(output_path, "wb") as out_file:
        out_file.write(chosen_data)

    compressed_size = len(chosen_data)
    achieved = calculate_reduction_percentage(original_size, compressed_size)

    if best_meeting_data is not None:
        return (
            True,
            "Fallback JPEG used. "
            f"Target met at quality {chosen_quality}. "
            f"Achieved reduction: {round(achieved, 4)}%",
        )

    return (
        False,
        "Fallback JPEG used, but target still not met due to content limits. "
        f"Best quality {chosen_quality}, achieved reduction: {round(achieved, 4)}%",
    )


image_path = resolve_project_path(input("Image Path: "))  # Example: IO/Inputs/Cat.jpg
target_percentage = read_target_percentage()

image_bit_string = file_handling.read_image_bit_string(str(image_path))
compressed_image_bit_string = huffman_coding.compress(image_bit_string)

file_handling.write_image(
    compressed_image_bit_string,
    str(OUTPUT_DIR / "compressed_image.bin"),
)

compression_ratio = len(image_bit_string) / len(compressed_image_bit_string)
original_size = image_path.stat().st_size
huffman_size = (OUTPUT_DIR / "compressed_image.bin").stat().st_size
size_reduction_percentage = calculate_reduction_percentage(original_size, huffman_size)

print("Compression Ratio (CR):", compression_ratio)
print("Achieved size reduction (%):", round(size_reduction_percentage, 4))

if target_percentage is not None:
    if size_reduction_percentage >= target_percentage:
        print(f"Target met: {target_percentage}%")
    else:
        print(
            f"Target not met: requested {target_percentage}%, achieved "
            f"{round(size_reduction_percentage, 4)}%"
        )
        print("Trying fallback algorithm (lossy JPEG) to meet your target...")
        fallback_met, fallback_message = try_jpeg_target_compression(
            image_path,
            target_percentage,
        )
        print(fallback_message)
        if not fallback_met:
            print(
                "Note: Huffman is lossless, and lossy JPEG also has practical limits "
                "based on image content."
            )

decompressed_image_bit_string = huffman_coding.decompress(
    compressed_image_bit_string
)
file_handling.write_image(
    decompressed_image_bit_string,
    str(OUTPUT_DIR / "decompressed_image.jpg"),
)