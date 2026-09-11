"""
Generates the sample video documents in data/videos/ and sample video
queries in data/video_queries/: short narrated slideshows (Pillow frames +
macOS `say` narration, muxed with the ffmpeg binary bundled in
imageio-ffmpeg). Swap them for real recordings — this just keeps the demo
self-contained.

    pip install pillow imageio-ffmpeg
    python make_sample_videos.py        # macOS (uses `say` for narration)
"""

import os
import shutil
import subprocess
import tempfile
import wave
import math

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 640, 360, 10


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def slide(title, lines, bg="#111827"):
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    d.text((32, 28), title, font=font(28, True), fill="white")
    y = 90
    for text in lines:
        d.text((32, y), text, font=font(20), fill="#e5e7eb")
        y += 36
    return img


def narrate(text, out_wav):
    aiff = out_wav + ".aiff"
    subprocess.run(["say", "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", aiff, out_wav], check=True)
    os.remove(aiff)


def make_video(out_path, slides, narration):
    """slides: list of (title, [lines]); narration: spoken text for the whole clip.
    Slides are spread evenly over the narration's duration."""
    tmp = tempfile.mkdtemp()
    try:
        wav = os.path.join(tmp, "narration.wav")
        narrate(narration, wav)
        with wave.open(wav) as w:
            duration = w.getnframes() / w.getframerate()
        seconds_per_slide = math.ceil(duration / len(slides))

        # Write frames: each slide repeated for seconds_per_slide * FPS frames.
        n = 0
        for title, lines in slides:
            img = slide(title, lines)
            for _ in range(seconds_per_slide * FPS):
                img.save(os.path.join(tmp, f"f{n:05d}.png"))
                n += 1
        subprocess.run(
            [
                FFMPEG, "-y", "-loglevel", "error",
                "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
                "-i", wav,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-shortest", out_path,
            ],
            check=True,
        )
    finally:
        shutil.rmtree(tmp)
    print("wrote", out_path)


if __name__ == "__main__":
    # --- video DOCUMENTS (ingested) ---
    make_video(
        os.path.join("data", "videos", "doc15_video.mp4"),
        slides=[
            ("Incident review: ERR-4521", ["Date: March 3rd", "Impact: checkout API down 18 minutes"]),
            ("Root cause", ["Connection pool exhausted (limit 50)", "Traffic spike from marketing campaign"]),
            ("Fix", ["Pool size raised from 50 to 200", "Added alert when pool usage > 80%"]),
        ],
        narration=(
            "Incident review for error ERR-4521. On March third the checkout API was down "
            "for eighteen minutes. Root cause: the database connection pool was exhausted "
            "at its limit of fifty connections during a marketing traffic spike. The fix "
            "was to raise the pool size from fifty to two hundred and add an alert when "
            "pool usage goes above eighty percent."
        ),
    )
    make_video(
        os.path.join("data", "videos", "doc16_video.mp4"),
        slides=[
            ("Reticulated python facts", ["Longest snake species in the world", "Native to Southeast Asia"]),
            ("Record holder", ["Medusa, a captive reticulated python", "Measured 25 feet 2 inches in 2011"]),
        ],
        narration=(
            "Reticulated python facts. It is the longest snake species in the world and is "
            "native to Southeast Asia. The record holder is Medusa, a captive reticulated "
            "python measured at twenty five feet two inches in twenty eleven."
        ),
    )

    # --- video QUERY (asked) ---
    make_video(
        os.path.join("data", "video_queries", "query_checkout_outage.mp4"),
        slides=[
            ("Checkout page", ["Loading..."]),
            ("Connection failed", ["Error code: ERR-4521", "Could not reach the database"]),
        ],
        narration="Hey, I keep seeing this on the checkout page. What caused it and how was it fixed?",
    )
