"""
Generates the sample image documents in data/images/ and sample image
queries in data/image_queries/ (mock screenshots rendered with Pillow).
Swap them for real photos/screenshots — this script just makes the demo
self-contained.

    pip install pillow
    python make_sample_images.py
"""

import os
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def error_dialog(path, title, lines, w=640, h=300):
    img = Image.new("RGB", (w, h), "#f3f4f6")
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, w - 20, h - 20], fill="white", outline="#9ca3af", width=2)
    d.rectangle([20, 20, w - 20, 64], fill="#dc2626")
    d.text((36, 30), title, font=font(22, True), fill="white")
    y = 90
    for text, bold in lines:
        d.text((40, y), text, font=font(18, bold), fill="#111827")
        y += 34
    d.rectangle([w - 140, h - 70, w - 40, h - 36], fill="#2563eb")
    d.text((w - 115, h - 63), "Close", font=font(16, True), fill="white")
    img.save(path)


def inventory_card(path):
    w, h = 640, 360
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 60], fill="#111827")
    d.text((24, 16), "WAREHOUSE INVENTORY REPORT", font=font(24, True), fill="white")
    rows = [
        ("SKU", "SKU-88213-XL"),
        ("Warehouse", "Rotterdam DC-3"),
        ("Expected quantity", "120"),
        ("Counted quantity", "97"),
        ("Discrepancy", "-23 units"),
        ("Status", "SHIPPING DELAYED"),
    ]
    y = 84
    for k, v in rows:
        d.text((24, y), k, font=font(18), fill="#6b7280")
        d.text((280, y), v, font=font(18, True), fill="#111827" if k != "Status" else "#dc2626")
        y += 42
    img.save(path)


if __name__ == "__main__":
    # --- image DOCUMENTS (ingested) ---
    error_dialog(
        os.path.join("data", "images", "doc13_image.png"),
        "Connection failed",
        [
            ("Error code: ERR-4521", True),
            ("Database connection timed out after 30 seconds.", False),
            ("Host: db-prod-01.internal   Port: 5432", False),
            ("Retry or contact the on-call DBA.", False),
        ],
    )
    inventory_card(os.path.join("data", "images", "doc14_image.png"))

    # --- image QUERIES (asked) ---
    error_dialog(
        os.path.join("data", "image_queries", "query_error_popup.png"),
        "Oops - something went wrong",
        [
            ("ERR-4521", True),
            ("The application could not reach the database.", False),
        ],
        w=520, h=240,
    )
    print("wrote data/images/*.png and data/image_queries/*.png")
