from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "assets"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    _write_dashboard(ASSET_DIR / "visual_dashboard.png")
    _write_form(ASSET_DIR / "visual_form.png")
    _write_invoice(ASSET_DIR / "visual_invoice.png")
    _write_slide(ASSET_DIR / "visual_slide.png")
    _write_parking(ASSET_DIR / "visual_parking.png")
    _write_map(ASSET_DIR / "visual_map.png")
    print("Wrote visual benchmark assets.")


def _canvas(size: tuple[int, int], color: str = "#F8FAFC") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", size, color)
    return image, ImageDraw.Draw(image)


def _write_dashboard(path: Path) -> None:
    image, draw = _canvas((1200, 720), "#F3F4F6")
    cards = [
        (90, 120, 350, 300, "#FECACA", "Revenue retention\nbelow target"),
        (420, 120, 680, 300, "#FECACA", "Support backlog\nbelow target"),
        (750, 120, 1010, 300, "#BBF7D0", "Pipeline coverage\non target"),
    ]
    for x1, y1, x2, y2, fill, label in cards:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=fill, outline="#111827", width=4)
        draw.multiline_text((x1 + 24, y1 + 40), label, fill="#111827", spacing=8)
    draw.text((90, 40), "Executive KPI Dashboard", fill="#111827")
    image.save(path)


def _write_form(path: Path) -> None:
    image, draw = _canvas((1080, 1350), "#FFFFFF")
    draw.text((80, 40), "Application Form", fill="#111827")
    fields = [
        ("Name", "Ada Lovelace", False),
        ("Work Authorization", "Required", True),
        ("Phone", "Format Invalid", True),
        ("LinkedIn", "https://example.com/in/ada", False),
    ]
    y = 140
    for label, value, error in fields:
        fill = "#FEE2E2" if error else "#E5E7EB"
        draw.rounded_rectangle((80, y, 1000, y + 170), radius=18, fill=fill, outline="#374151", width=3)
        draw.text((110, y + 22), label, fill="#111827")
        draw.text((110, y + 76), value, fill="#991B1B" if error else "#1F2937")
        y += 220
    image.save(path)


def _write_invoice(path: Path) -> None:
    image, draw = _canvas((1100, 820), "#FFFBEB")
    draw.text((80, 40), "Invoice", fill="#111827")
    draw.rounded_rectangle((680, 180, 1030, 520), radius=16, fill="#FFFFFF", outline="#111827", width=4)
    draw.text((720, 220), "Subtotal  $48,000", fill="#111827")
    draw.text((720, 300), "Tax       $3,840", fill="#111827")
    draw.text((720, 380), "Total     $51,840", fill="#111827")
    image.save(path)


def _write_slide(path: Path) -> None:
    image, draw = _canvas((1280, 720), "#E0F2FE")
    draw.text((80, 40), "Partner Update", fill="#111827")
    callouts = [
        (120, 170, 540, 350, "#FECACA", "Risk: invoice lock drift can reopen edits"),
        (700, 170, 1120, 350, "#BFDBFE", "Action: keep safe mode enabled"),
    ]
    for x1, y1, x2, y2, fill, label in callouts:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline="#1F2937", width=4)
        draw.multiline_text((x1 + 24, y1 + 40), label, fill="#111827", spacing=8)
    image.save(path)


def _write_parking(path: Path) -> None:
    image, draw = _canvas((1280, 920), "#D1D5DB")
    draw.rectangle((70, 70, 1210, 850), fill="#9CA3AF", outline="#111827", width=4)
    for row in range(6):
        y = 120 + row * 120
        draw.line((110, y, 1170, y), fill="#FFFFFF", width=4)
    for col in range(8):
        x = 140 + col * 130
        draw.line((x, 100, x, 820), fill="#FFFFFF", width=2)
    colors = ["#FFFFFF", "#34D399", "#60A5FA", "#F59E0B"]
    index = 0
    for row in range(8):
        for col in range(8):
            x1 = 110 + col * 130
            y1 = 110 + row * 90
            x2 = x1 + 80
            y2 = y1 + 40
            draw.rounded_rectangle((x1, y1, x2, y2), radius=10, fill=colors[index % len(colors)], outline="#111827", width=2)
            index += 1
    image.save(path)


def _write_map(path: Path) -> None:
    image, draw = _canvas((1080, 720), "#ECFCCB")
    draw.text((80, 40), "Facility Exit Map", fill="#111827")
    exits = [
        (160, 220, 380, 320, "#FCA5A5", "North Exit\nblocked"),
        (700, 220, 920, 320, "#86EFAC", "South Exit\nopen"),
    ]
    for x1, y1, x2, y2, fill, label in exits:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline="#111827", width=4)
        draw.multiline_text((x1 + 24, y1 + 28), label, fill="#111827", spacing=8)
    image.save(path)


if __name__ == "__main__":
    main()
