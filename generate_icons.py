"""
Genera íconos para la PWA.
Ejecutar: python generate_icons.py
"""
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Instala Pillow: pip install Pillow")
    exit(1)


def crear_icono(size: int, ruta: Path):
    """Crea un ícono con la letra M."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    color1 = (102, 126, 234)
    margin = int(size * 0.05)
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color1)

    font_size = int(size * 0.5)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    text = "M"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - bbox[1]
    draw.text((x, y), text, fill="white", font=font)

    img.save(ruta)
    print(f"✅ Ícono {size}x{size} guardado en: {ruta}")


if __name__ == "__main__":
    static_dir = Path("app/static")
    static_dir.mkdir(parents=True, exist_ok=True)

    crear_icono(192, static_dir / "icon-192.png")
    crear_icono(512, static_dir / "icon-512.png")
    print("\n✅ Íconos generados")