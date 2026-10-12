from pathlib import Path
import ast

path = Path("moduli/chalkboard.py")
source = path.read_text(encoding="utf-8")

start = source.index("def _draw_chalk_line(")
end = source.index("\ndef _draw_polyline_progress", start)

replacement = '''def _draw_chalk_line(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], progress: float,
                     color: str = CHALK, width: int = 3, seed: int = 0) -> None:
    if progress <= 0:
        return

    progress = min(1.0, progress)
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    full_length = math.hypot(dx, dy)
    if full_length < 1:
        return

    length = full_length * progress
    angle = math.atan2(dy, dx)
    nx, ny = -math.sin(angle), math.cos(angle)
    rng = random.Random(seed)

    # Keep support for named Pillow colors as well as hex colors.
    try:
        from PIL import ImageColor
        base = ImageColor.getrgb(color) if isinstance(color, str) else tuple(color[:3])
    except (ValueError, TypeError):
        base = (240, 238, 226)

    def shade(delta):
        return tuple(max(0, min(255, channel + delta)) for channel in base[:3])

    # Lay down short, irregular chalk fragments rather than one clean digital line.
    pos = 0.0
    while pos < length:
        chunk = min(length - pos, rng.uniform(14, 34))
        stop = pos + chunk
        points = []
        t = pos

        while t < stop:
            jitter = rng.uniform(-0.9, 0.9)
            points.append((
                int(x1 + math.cos(angle) * t + nx * jitter),
                int(y1 + math.sin(angle) * t + ny * jitter),
            ))
            t += rng.uniform(2.5, 5.0)

        points.append((
            int(x1 + math.cos(angle) * stop),
            int(y1 + math.sin(angle) * stop),
        ))
        if len(points) >= 2:
            draw.line(points, fill=shade(rng.randint(-10, 5)), width=max(1, width))

        # Fine parallel fragments mimic chalk catching the board grain.
        for _ in range(max(1, int(chunk / 13))):
            a = rng.uniform(pos, stop)
            b = min(stop, a + rng.uniform(3, 15))
            offset = rng.choice((-1, 1)) * rng.uniform(1, max(2, width + 1))
            draw.line((
                int(x1 + math.cos(angle) * a + nx * offset),
                int(y1 + math.sin(angle) * a + ny * offset),
                int(x1 + math.cos(angle) * b + nx * offset),
                int(y1 + math.sin(angle) * b + ny * offset),
            ), fill=shade(rng.randint(-18, 12)), width=1)

        # Small grains of chalk dust scattered around the mark.
        for _ in range(max(1, int(chunk / 7))):
            t = rng.uniform(pos, stop)
            offset = rng.gauss(0, max(1, width * 1.1))
            cx = int(x1 + math.cos(angle) * t + nx * offset)
            cy = int(y1 + math.sin(angle) * t + ny * offset)
            if rng.random() < 0.55:
                draw.point((cx, cy), fill=shade(rng.randint(-8, 18)))
            else:
                radius = 1
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=shade(rng.randint(-12, 15)),
                )

        # Slight gaps make the edge feel less mechanically perfect.
        pos = stop + rng.uniform(1, 2.5)
'''

candidate = source[:start] + replacement + source[end:]

# Validate the complete module before changing the original file.
ast.parse(candidate, filename=str(path))
path.write_text(candidate, encoding="utf-8")
print("SUCCESS: chalk stroke function updated; Python syntax validated.")
print("Backup remains at moduli/chalkboard.py.before_chalkstroke.bak")