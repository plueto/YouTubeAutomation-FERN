from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageChops

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEARCH = ["cache/openverse", "cache/pexels", "assets", "fern_frames", "fern_reference", "cache"]

source = None
seen = set()

for folder in SEARCH:
    base = ROOT / folder
    if not base.exists():
        continue
    files = sorted(
        (p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in EXTS),
        key=lambda p: p.stat().st_size,
        reverse=True
    )
    for path in files:
        if path in seen or "node_modules" in path.parts:
            continue
        seen.add(path)
        try:
            with Image.open(path) as im:
                if im.width >= 640 and im.height >= 360:
                    source = path
                    break
        except Exception:
            pass
    if source:
        break

if source is None:
    raise SystemExit("No suitable cached image found. No project files changed.")

with Image.open(source) as im:
    im = im.convert("RGB")
    im.thumbnail((1100, 700), Image.Resampling.LANCZOS)

gray = ImageOps.grayscale(im).filter(ImageFilter.GaussianBlur(1.1))
edges = ImageOps.autocontrast(gray.filter(ImageFilter.FIND_EDGES))
mask = edges.point(lambda p: max(0, min(255, int((p - 28) * 2.2))))

# Slightly roughen the lines so they resemble chalk rather than clean vectors.
grain = Image.effect_noise(im.size, 22).convert("L")
grain = grain.point(lambda p: 110 + p // 2)
mask = ImageChops.multiply(mask, grain)

# Blackboard texture with pale chalk strokes and a subtle colored glow.
noise = Image.effect_noise(im.size, 12).convert("L")
board = ImageOps.colorize(noise, black="#101719", white="#252c2b").convert("RGBA")

glow_mask = mask.filter(ImageFilter.GaussianBlur(2.2)).point(lambda p: p // 4)
glow = Image.new("RGBA", im.size, (102, 190, 176, 0))
glow.putalpha(glow_mask)
board = Image.alpha_composite(board, glow)

chalk = Image.new("RGBA", im.size, (225, 238, 224, 0))
chalk.putalpha(mask)
result = Image.alpha_composite(board, chalk).convert("RGB")

out = ROOT / "output" / "chalk_style_test.png"
out.parent.mkdir(parents=True, exist_ok=True)
result.save(out, quality=95)

print(f"Source image: {source}")
print(f"Chalk test saved to: {out}")
