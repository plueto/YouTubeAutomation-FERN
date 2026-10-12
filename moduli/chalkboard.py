"""Optional, local-only chalkboard visual renderer for FERN.

No AI image service is needed. It creates narrated explainer visuals from the
script using Pillow, then delegates audio/music muxing to FERN's existing
FFmpeg utilities. Enable with FERN_VISUAL_MODE=chalkboard.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps

# Colors used by the approved no-hand chalkboard prototype.
BG_DARK = "#11191b"
BG_LIGHT = "#293532"
CHALK = "#f0eee2"
CHALK_DIM = "#aab8b2"
TEAL = "#8bd6c5"
YELLOW = "#f2c76a"

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "if", "than", "then",
    "of", "to", "in", "on", "at", "by", "for", "from", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "there", "here", "you", "your",
    "we", "our", "they", "their", "he", "she", "his", "her", "them",
    "i", "me", "my", "not", "no", "do", "does", "did", "have", "has",
    "had", "will", "would", "could", "should", "can", "may", "might",
    "what", "when", "where", "why", "how", "who", "which", "because",
    "into", "about", "over", "under", "up", "down", "also", "just",
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if os.name == "nt":
        candidates.extend([
            # Segoe Print feels like deliberate handwriting on a board.
            r"C:\Windows\Fonts\segoeprb.ttf" if bold else r"C:\Windows\Fonts\segoepr.ttf",
            r"C:\Windows\Fonts\comicbd.ttf" if bold else r"C:\Windows\Fonts\comic.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = current + " " + word
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _ellipsize(text: str, max_words: int) -> str:
    words = re.sub(r"\s+", " ", text).strip().split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(" ,;:.!?—-") + "…"
    return " ".join(words)


def _sentences(script: str) -> list[str]:
    cleaned = re.sub(r"\[[^\]]{1,60}\]", " ", script or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ["The idea becomes clear when we connect the pieces."]
    pieces = re.split(r"(?<=[.!?])\s+|\s*\n+", cleaned)
    result: list[str] = []
    for piece in pieces:
        piece = piece.strip(" \t\r\n•-")
        if not piece:
            continue
        # Split excessively long sentences into readable visual beats.
        words = piece.split()
        while len(words) > 32:
            result.append(" ".join(words[:24]))
            words = words[24:]
        if words:
            result.append(" ".join(words))
    return result or [cleaned[:180]]


def _group_sentences(script: str, count: int) -> list[str]:
    """Split narration into unique, contiguous beats near the target word count.

    Keep sentence and clause boundaries whenever possible. The old allocator
    assigned whole sentences to buckets and copied neighboring text into empty
    buckets, which created duplicate boards. This version first makes compact
    spoken beats (usually 8–16 words), then groups adjacent beats without
    duplication. That gives the chalk reveal more frequent, narration-aligned
    visual changes without cutting every sentence at an arbitrary word.
    """
    cleaned = re.sub(r"\s+", " ", script or "").strip()
    if not cleaned:
        return ["The idea becomes clear when we connect the pieces."]

    # Split at strong sentence punctuation first, then at clause boundaries.
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", cleaned) if x.strip()]
    units: list[str] = []
    max_words = 15
    for sentence in sentences:
        clauses = [x.strip() for x in re.split(r"(?<=[,;:])\s+", sentence) if x.strip()]
        carry: list[str] = []
        for clause in clauses:
            words = clause.split()
            if len(words) > max_words:
                # Split an unusually long clause into contiguous chunks, ending
                # each piece cleanly without losing any spoken words.
                if carry:
                    units.append(" ".join(carry))
                    carry = []
                while len(words) > max_words:
                    cut = max_words
                    # Prefer a natural conjunction near the desired cut.
                    for j in range(max_words - 1, max(5, max_words - 5), -1):
                        if words[j - 1].lower().strip(",;:") in {"and", "but", "because", "while", "then", "so"}:
                            cut = j
                            break
                    units.append(" ".join(words[:cut]).rstrip(",;:"))
                    words = words[cut:]
                carry = words
            elif len(carry) + len(words) <= max_words:
                carry.extend(words)
            else:
                units.append(" ".join(carry).strip())
                carry = words
        if carry:
            units.append(" ".join(carry).strip())

    units = [u for u in units if u]
    if not units:
        units = [cleaned]

    words_total = len(cleaned.split())
    # A beat list can have more entries than requested; group neighbors. If it
    # has fewer, keep the natural beats intact rather than duplicating any.
    target_count = max(1, min(int(count), len(units), words_total))
    groups: list[str] = []
    for i in range(target_count):
        first = round(i * len(units) / target_count)
        last = round((i + 1) * len(units) / target_count)
        groups.append(" ".join(units[first:last]).strip())
    return [g for g in groups if g] or [cleaned]


def _phrases(text: str, count: int = 3) -> list[str]:
    """Produce distinct, readable chalk labels from the spoken excerpt."""
    count = max(1, int(count))
    raw = re.split(
        r",|;|:|\b(?:because|therefore|however|while|whereas|which means|as a result|then|but|and so)\b",
        text, flags=re.I,
    )
    candidates = [re.sub(r"\s+", " ", x).strip(" .,:;-") for x in raw]
    candidates = [x for x in candidates if x]

    def key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    def split_clause(clause: str) -> tuple[str, str] | None:
        words = clause.split()
        if len(words) < 4:
            return None
        center = len(words) // 2
        bad_end = {"a", "an", "the", "to", "of", "for", "at", "on", "in", "and", "but", "by", "with", "from"}
        bad_start = {"and", "but", "so", "then", "because", "while", "whereas"}
        choices = range(max(1, center - 2), min(len(words), center + 3))
        def score(cut: int) -> tuple[int, int]:
            penalty = int(words[cut - 1].lower().strip(",;:") in bad_end)
            penalty += int(words[cut].lower().strip(",;:") in bad_start)
            return penalty, abs(cut - center)
        cut = min(choices, key=score)
        left = " ".join(words[:cut]).strip(" ,;:")
        right = " ".join(words[cut:]).strip(" ,;:")
        return (left, right) if left and right else None

    # Keep distinct clauses in narration order. If there are fewer than three,
    # split the longest clause at a natural word boundary rather than duplicating
    # text to fill the diagram's three slots.
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        k = key(candidate)
        if k and k not in seen:
            unique.append(candidate)
            seen.add(k)

    while len(unique) < count:
        index = max(range(len(unique)), key=lambda i: len(unique[i].split()), default=-1)
        if index < 0:
            break
        split = split_clause(unique[index])
        if not split:
            break
        unique[index:index + 1] = list(split)
        seen = {key(item) for item in unique}

    result = [_ellipsize(value, 7).rstrip(" ,;:.!?—-") for value in unique[:count]]
    result = [value for value in result if value]
    # Very short extracts are rare, but the layout still needs three distinct
    # labels. Use neutral cues rather than repeat the same spoken words.
    neutral = ["Key point", "Evidence", "Consequence", "What changes"]
    for cue in neutral:
        if len(result) >= count:
            break
        if key(cue) not in {key(x) for x in result}:
            result.append(cue)
    return result[:count]


def _title_for(text: str) -> str:
    s = _sentences(text)[0]
    words = s.split()
    if len(words) > 8:
        s = " ".join(words[:8]).rstrip(" ,;:.!?—-") + "…"
    return s


def _scene_kind(text: str, index: int) -> str:
    lower = text.lower()
    if re.search(r"\b(first|next|before|after|finally|step|stage|phase|year|decade|century|timeline|over time)\b", lower):
        return "timeline"
    if re.search(r"\b(because|causes?|leads? to|results? in|therefore|as a result|consequently|effect|impact)\b", lower):
        return "flow"
    if re.search(r"\b(compare|compared|versus|vs\.?|whereas|while|rather than|unlike|difference|similar|both)\b", lower):
        return "compare"
    if re.search(r"\b(\d+(?:[.,]\d+)?%?|percent|percentage|million|billion|trillion|doubled|tripled|increased|decreased|grew by|fell by|rate|ratio|average)\b", lower):
        return "numbers"
    # Change the visual grammar as the narration advances without claiming a
    # causal or chronological relationship that the script did not state.
    return ("cards", "callout", "cards", "flow", "cards")[index % 5]


def _scene_heading(text: str, index: int) -> str:
    return {
        "cards": "KEY IDEAS",
        "flow": "HOW IT CONNECTS",
        "timeline": "HOW IT UNFOLDS",
        "compare": "THE CONTRAST",
        "numbers": "THE NUMBERS",
        "callout": "THE KEY POINT",
    }.get(_scene_kind(text, index), "KEY IDEAS")


def _board_background(width: int, height: int) -> Image.Image:
    noise = Image.effect_noise((width, height), 17).convert("L")
    board = ImageOps.colorize(noise, black=BG_DARK, white=BG_LIGHT).convert("RGB")
    # Give it a little depth while keeping the surface quiet enough for reading.
    shade = Image.new("RGB", (width, height), BG_DARK)
    board = Image.blend(board, shade, 0.37)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    rng = random.Random(1984)
    for _ in range(170):
        x = rng.randrange(20, width - 20)
        y = rng.randrange(20, height - 20)
        r = rng.choice([1, 1, 2])
        col = rng.choice([(220, 232, 222, 12), (210, 225, 218, 9), (0, 0, 0, 15)])
        d.ellipse((x-r, y-r, x+r, y+r), fill=col)
    # Faint chalk smudge bars near the edges.
    for _ in range(12):
        y = rng.choice([rng.randint(8, 30), rng.randint(height - 30, height - 8)])
        x = rng.randint(20, width - 170)
        d.line((x, y, x + rng.randint(30, 150), y + rng.randint(-3, 3)), fill=(220, 230, 220, 9), width=2)
    return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")


def _draw_chalk_line(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], progress: float,
                     color: str = CHALK, width: int = 3, seed: int = 0) -> None:
    if progress <= 0:
        return
    progress = min(1.0, progress)
    x1, y1 = p1
    x2, y2 = p2
    end = (x1 + (x2 - x1) * progress, y1 + (y2 - y1) * progress)
    rng = random.Random(seed)
    # Two imperfect passes make the lines look dusty and hand-drawn.
    draw.line((x1, y1, end[0], end[1]), fill=color, width=width)
    if width >= 2 and progress > 0.04:
        jitter = rng.choice([-2, -1, 1, 2])
        draw.line((x1 + jitter, y1 - 1, end[0] + jitter, end[1] - 1), fill=color, width=1)


def _draw_polyline_progress(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], progress: float,
                            color: str = CHALK, width: int = 3, seed: int = 0) -> None:
    if len(points) < 2 or progress <= 0:
        return
    lengths = [math.dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = max(1.0, sum(lengths))
    remaining = min(1.0, progress) * total
    for i, length in enumerate(lengths):
        if remaining <= 0:
            break
        fraction = min(1.0, remaining / max(1.0, length))
        _draw_chalk_line(draw, points[i], points[i + 1], fraction, color, width, seed + i)
        remaining -= length


def _draw_rect_progress(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], progress: float,
                        color: str = CHALK, width: int = 3, seed: int = 0) -> None:
    x1, y1, x2, y2 = box
    # Slightly uneven perimeter, with every edge drawn in sequence.
    pts = [(x1+4, y1+1), (x2-5, y1+2), (x2-1, y1+7), (x2-2, y2-5),
           (x2-7, y2-1), (x1+6, y2-2), (x1+1, y2-6), (x1+2, y1+7), (x1+4, y1+1)]
    _draw_polyline_progress(draw, pts, progress, color, width, seed)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], progress: float,
                color: str = TEAL, seed: int = 0) -> None:
    _draw_chalk_line(draw, start, end, progress, color, 3, seed)
    if progress >= 0.88:
        ang = math.atan2(end[1]-start[1], end[0]-start[0])
        size = 15
        left = (end[0] - size * math.cos(ang - 0.48), end[1] - size * math.sin(ang - 0.48))
        right = (end[0] - size * math.cos(ang + 0.48), end[1] - size * math.sin(ang + 0.48))
        _draw_chalk_line(draw, left, end, 1, color, 3, seed + 5)
        _draw_chalk_line(draw, end, right, 1, color, 3, seed + 9)


def _draw_text_reveal(image: Image.Image, text: str, xy: tuple[int, int], font: ImageFont.ImageFont,
                      color: str, progress: float, max_width: int, spacing: int = 7,
                      align: str = "left") -> None:
    """Reveal complete glyphs left-to-right, avoiding chopped half-letters.

    The old mask wipe sliced through characters and looked like a screen wipe.
    This writes the text in short, readable chalk-like character increments while
    reserving the final text width up front so centered/right-aligned labels don't drift.
    """
    if progress <= 0 or not text:
        return
    d = ImageDraw.Draw(image)
    lines = _wrap(text, font, max_width, d)
    layout = []
    total_chars = max(1, sum(len(line) for line in lines))
    x, y = xy
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        if align == "center":
            px = x - line_width // 2
        elif align == "right":
            px = x - line_width
        else:
            px = x
        layout.append((line, px, y))
        y += line_height + spacing
    chars_to_show = min(total_chars, max(0, int(math.ceil(total_chars * min(1.0, progress)))))
    remaining = chars_to_show
    for line, px, py in layout:
        if remaining <= 0:
            break
        prefix = line[:remaining]
        if prefix:
            d.text((px, py), prefix, font=font, fill=color, stroke_width=0)
        remaining -= len(line)

def _draw_card_scene(image: Image.Image, phrases: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    boxes = [(90, 260, 430, 535), (470, 260, 810, 535), (850, 260, 1190, 535)]
    small = _font(24)
    icon_font = _font(28, True)
    for i, box in enumerate(boxes):
        start = 0.07 + i * 0.18
        box_p = max(0, min(1, (p - start) / 0.17))
        _draw_rect_progress(d, box, box_p, TEAL if i == 1 else CHALK_DIM, 3, idx * 17 + i)
        cx = (box[0] + box[2]) // 2
        if box_p > 0.7:
            # Small hand-drawn marker rather than a generic icon pack.
            d.ellipse((cx-22, box[1]+24, cx+22, box[1]+68), outline=YELLOW if i == 2 else TEAL, width=3)
            mark = str(i + 1)
            bb = d.textbbox((0, 0), mark, font=icon_font)
            d.text((cx-(bb[2]-bb[0])//2, box[1]+29), mark, font=icon_font, fill=YELLOW if i == 2 else TEAL)
        text_p = max(0, min(1, (p - (start + 0.07)) / 0.19))
        _draw_text_reveal(image, phrases[i], (cx, box[1]+112), small, CHALK, text_p, box[2]-box[0]-44, 8, "center")


def _draw_flow_scene(image: Image.Image, phrases: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    boxes = [(70, 310, 390, 500), (480, 310, 800, 500), (890, 310, 1210, 500)]
    font = _font(25)
    for i, box in enumerate(boxes):
        start = 0.06 + i * 0.18
        bp = max(0, min(1, (p - start) / 0.17))
        _draw_rect_progress(d, box, bp, TEAL if i == 1 else CHALK, 3, idx*31+i)
        _draw_text_reveal(image, phrases[i], ((box[0]+box[2])//2, box[1]+65), font, CHALK,
                          max(0, min(1, (p - start - .06) / .17)), box[2]-box[0]-36, 6, "center")
    _draw_arrow(d, (395, 405), (472, 405), max(0, min(1, (p-.25)/.13)), YELLOW, idx+5)
    _draw_arrow(d, (805, 405), (882, 405), max(0, min(1, (p-.43)/.13)), YELLOW, idx+15)


def _draw_timeline_scene(image: Image.Image, phrases: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    labels=[x for x in phrases if x.strip()][:3]
    if len(labels)<2:
        return _draw_object_focus_scene(image, labels, p, "clock", idx)
    _draw_chalk_line(d, (120, 425), (1160, 425), max(0, min(1, (p-.025)/.40)), TEAL, 4, idx)
    xs = [260, 640, 1020] if len(labels)==3 else [390, 890]
    font = _font(27)
    for i, (x,label) in enumerate(zip(xs,labels)):
        start = .12 + i * (0.21 if len(labels)==3 else 0.28)
        q = max(0, min(1, (p-start)/.16))
        if q > 0:
            radius = 12
            d.arc((x-radius, 425-radius, x+radius, 425+radius), 0, int(360*q), fill=YELLOW if i == 1 else CHALK, width=3)
            _draw_chalk_line(d, (x, 410), (x, 360 if i % 2 == 0 else 490), q, YELLOW if i == 1 else CHALK, 3, idx+i)
        ty = 290 if i % 2 == 0 else 500
        _draw_text_reveal(image, label, (x, ty), font, CHALK, max(0, min(1, (p-start-.03)/.20)), 325, 6, "center")


def _draw_compare_scene(image: Image.Image, phrases: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    box = (170, 245, 1110, 555)
    _draw_rect_progress(d, box, max(0, min(1, (p-.04)/.24)), CHALK, 3, idx)
    split_x = 640
    _draw_chalk_line(d, (split_x, 246), (split_x, 555), max(0, min(1, (p-.20)/.17)), TEAL, 3, idx+3)
    _draw_chalk_line(d, (171, 340), (1109, 340), max(0, min(1, (p-.26)/.13)), CHALK_DIM, 2, idx+4)
    left = _font(26, True)
    body = _font(24)
    _draw_text_reveal(image, phrases[0], (405, 270), left, TEAL, max(0, min(1, (p-.16)/.16)), 390, 5, "center")
    _draw_text_reveal(image, phrases[1], (875, 270), left, YELLOW, max(0, min(1, (p-.28)/.16)), 390, 5, "center")
    _draw_text_reveal(image, phrases[2], (405, 390), body, CHALK, max(0, min(1, (p-.39)/.18)), 390, 7, "center")
    _draw_text_reveal(image, "What changes", (875, 390), body, CHALK, max(0, min(1, (p-.50)/.18)), 390, 7, "center")


def _draw_numbers_scene(image: Image.Image, phrases: list[str], text: str, p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|trillion)?\b", text, flags=re.I)
    numbers = [n.strip() for n in nums[:3]]
    if not numbers:
        numbers = ["KEY", "FIGURE", "RESULT"]
    while len(numbers) < 3:
        numbers.append(phrases[len(numbers)])
    xs = [250, 640, 1030]
    big = _font(43, True)
    body = _font(22)
    for i, x in enumerate(xs):
        start = .08 + i * .18
        q = max(0, min(1, (p-start)/.21))
        r = 80
        if q > 0:
            d.arc((x-r, 305-r, x+r, 305+r), start=210, end=210+int(290*q), fill=YELLOW if i == 1 else TEAL, width=4)
        _draw_text_reveal(image, numbers[i], (x, 292), big, YELLOW if i == 1 else CHALK, q,
                          310, 4, "center")
        _draw_text_reveal(image, phrases[i], (x, 425), body, CHALK_DIM, max(0, min(1, (p-start-.07)/.18)), 300, 6, "center")


def _draw_callout_scene(image: Image.Image, phrases: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    font = _font(37, True)
    body = _font(24)
    main = _ellipsize(phrases[0], 5)
    x = image.width // 2
    y = 325
    _draw_text_reveal(image, main, (x, y), font, YELLOW,
                      max(0, min(1, (p-.07)/.22)), 850, 5, "center")
    d = ImageDraw.Draw(image)
    # Draw an imperfect hand-sketched ellipse around the key message.
    q = max(0, min(1, (p-.32)/.20))
    if q > 0:
        d.arc((150, 275, 1130, 420), start=195, end=195+int(330*q), fill=TEAL, width=4)
    _draw_text_reveal(image, phrases[1], (x, 465), body, CHALK, max(0, min(1, (p-.41)/.17)), 850, 7, "center")
    _draw_text_reveal(image, phrases[2], (x, 520), body, CHALK_DIM, max(0, min(1, (p-.57)/.17)), 850, 7, "center")


def _draw_motif(draw: ImageDraw.ImageDraw, motif: str, cx: int, cy: int, size: int,
                progress: float, color: str = TEAL, seed: int = 0) -> None:
    """Draw a simple imperfect chalk symbol; no hand sprite or stock icon."""
    q = max(0.0, min(1.0, progress))
    if q <= 0:
        return
    r = max(12, size // 2)
    if motif == "coin":
        # Coin stack: draw the rims in sequence, then a small currency mark.
        for i in range(3):
            yy = cy + int((i - 1) * r * 0.55)
            part = max(0.0, min(1.0, (q - i * 0.22) / 0.45))
            if part > 0:
                draw.arc((cx-r, yy-r//3, cx+r, yy+r//3), 180, 180 + int(360 * part), fill=color, width=3)
                if part >= 0.85:
                    _draw_chalk_line(draw, (cx-r, yy), (cx-r, yy+r//2), 1, color, 3, seed+i)
                    _draw_chalk_line(draw, (cx+r, yy), (cx+r, yy+r//2), 1, color, 3, seed+i+5)
        if q > 0.72:
            _draw_text_reveal_on_draw(draw, "$", (cx, cy-r//3), _font(max(18, size//3), True), YELLOW, (q-.72)/.28, size)
    elif motif == "building":
        box = (cx-r, cy-r, cx+r, cy+r)
        _draw_rect_progress(draw, box, min(1, q*1.2), color, 3, seed)
        _draw_polyline_progress(draw, [(cx-r-8, cy-r), (cx, cy-r-20), (cx+r+8, cy-r)],
                                max(0, min(1, (q-.16)/.25)), YELLOW, 3, seed+3)
        for row in range(2):
            for col in range(3):
                start = .3 + row*.18 + col*.08
                if q > start:
                    xx = cx-r+int((col+1)*2*r/4)
                    yy = cy-r+int((row+1)*2*r/3)
                    ww = max(7, r//5)
                    draw.rectangle((xx-ww//2, yy-ww//2, xx+ww//2, yy+ww//2), outline=color, width=2)
    elif motif == "lock":
        _draw_rect_progress(draw, (cx-r//1.4, cy, cx+r//1.4, cy+r), max(0, min(1, (q-.25)/.5)), color, 3, seed)
        if q > .03:
            draw.arc((cx-r//2, cy-r, cx+r//2, cy+r//2), 180, 180+int(180*min(1,q*1.5)), fill=YELLOW, width=3)
        if q > .72:
            draw.ellipse((cx-4, cy+r//3, cx+4, cy+r//3+8), outline=color, width=2)
    elif motif == "eye":
        pts = [(cx-r, cy), (cx-r//2, cy-r//2), (cx, cy-r//2-4), (cx+r//2, cy-r//2), (cx+r, cy),
               (cx+r//2, cy+r//2), (cx, cy+r//2+4), (cx-r//2, cy+r//2), (cx-r, cy)]
        _draw_polyline_progress(draw, pts, q, color, 3, seed)
        if q > .42:
            qq = min(1, (q-.42)/.36)
            draw.arc((cx-r//3, cy-r//3, cx+r//3, cy+r//3), 0, int(360*qq), fill=YELLOW, width=3)
        if q > .78:
            draw.ellipse((cx-5, cy-5, cx+5, cy+5), fill=YELLOW)
    elif motif == "chip":
        _draw_rect_progress(draw, (cx-r//1.3, cy-r//1.3, cx+r//1.3, cy+r//1.3), q, color, 3, seed)
        for i in range(4):
            offset = -r//2 + i*r//3
            for edge in (-1, 1):
                part = max(0, min(1, (q-.25-i*.06)/.3))
                if edge < 0:
                    _draw_chalk_line(draw, (cx+offset, cy-r//1.3-12), (cx+offset, cy-r//1.3), part, YELLOW, 2, seed+i)
                    _draw_chalk_line(draw, (cx+offset, cy+r//1.3), (cx+offset, cy+r//1.3+12), part, YELLOW, 2, seed+i+7)
                else:
                    _draw_chalk_line(draw, (cx-r//1.3-12, cy+offset), (cx-r//1.3, cy+offset), part, YELLOW, 2, seed+i+13)
                    _draw_chalk_line(draw, (cx+r//1.3, cy+offset), (cx+r//1.3+12, cy+offset), part, YELLOW, 2, seed+i+19)
    elif motif == "document" or motif == "magnifier":
        if motif == "document":
            box = (cx-r*2//3, cy-r, cx+r*2//3, cy+r)
            _draw_rect_progress(draw, box, q, color, 3, seed)
            for i in range(3):
                start = .28 + i*.17
                _draw_chalk_line(draw, (box[0]+12, box[1]+30+i*20), (box[2]-12, box[1]+30+i*20),
                                 max(0,min(1,(q-start)/.2)), YELLOW if i == 1 else color, 2, seed+i)
        else:
            draw.arc((cx-r, cy-r, cx+r//2, cy+r//2), 0, int(360*min(1,q)), fill=color, width=4)
            if q > .45:
                _draw_chalk_line(draw, (cx+r//4, cy+r//4), (cx+r, cy+r), min(1,(q-.45)/.4), YELLOW, 4, seed+5)
    elif motif == "chart":
        _draw_chalk_line(draw, (cx-r, cy+r), (cx-r, cy-r), q, color, 3, seed)
        _draw_chalk_line(draw, (cx-r, cy+r), (cx+r, cy+r), max(0,min(1,(q-.12)/.22)), color, 3, seed+1)
        heights = [r//2, int(r*1.1), int(r*1.6)]
        for i, h in enumerate(heights):
            part = max(0,min(1,(q-.25-i*.15)/.3))
            bx = cx-r//2+i*r//2
            _draw_rect_progress(draw, (bx, cy+r-h, bx+r//4, cy+r), part, YELLOW if i==2 else color, 2, seed+i+2)
    elif motif == "warning":
        pts = [(cx, cy-r), (cx+r, cy+r), (cx-r, cy+r), (cx, cy-r)]
        _draw_polyline_progress(draw, pts, q, YELLOW, 3, seed)
        if q > .55:
            _draw_chalk_line(draw, (cx, cy-r//3), (cx, cy+r//3), min(1,(q-.55)/.25), color, 4, seed+5)
        if q > .8:
            draw.ellipse((cx-3, cy+r//2, cx+3, cy+r//2+3), fill=color)
    elif motif == "clock":
        draw.arc((cx-r, cy-r, cx+r, cy+r), 0, int(360*q), fill=color, width=3)
        _draw_chalk_line(draw, (cx, cy), (cx, cy-r//2), max(0,min(1,(q-.25)/.25)), YELLOW, 3, seed)
        _draw_chalk_line(draw, (cx, cy), (cx+r//2, cy+r//4), max(0,min(1,(q-.48)/.25)), color, 3, seed+1)
    elif motif == "box" or motif == "broken_link":
        _draw_rect_progress(draw, (cx-r, cy-r//2, cx+r, cy+r//2), q, color, 3, seed)
        _draw_polyline_progress(draw, [(cx-r,cy-r//2),(cx,cy-r),(cx+r,cy-r//2)], max(0,min(1,(q-.2)/.3)), YELLOW, 3, seed+2)
        if motif == "broken_link" and q > .6:
            _draw_chalk_line(draw, (cx-r//3, cy-r//2), (cx+r//3, cy+r//2), min(1,(q-.6)/.25), YELLOW, 4, seed+4)
    elif motif == "lightbulb":
        draw.arc((cx-r, cy-r, cx+r, cy+r), 190, 190+int(320*min(1,q)), fill=YELLOW, width=3)
        if q > .55:
            _draw_chalk_line(draw, (cx-r//3,cy+r//2),(cx+r//3,cy+r//2), min(1,(q-.55)/.15), color,3,seed)
            _draw_chalk_line(draw, (cx-r//4,cy+r//2+8),(cx+r//4,cy+r//2+8),min(1,(q-.7)/.2),color,3,seed+1)
    elif motif == "network":
        nodes = [(cx,cy),(cx-r,cy-r//2),(cx+r,cy-r//2),(cx-r//2,cy+r),(cx+r//2,cy+r)]
        for i in range(1,len(nodes)):
            _draw_chalk_line(draw,nodes[0],nodes[i],max(0,min(1,(q-(i-1)*.12)/.2)),color,2,seed+i)
        for i,(x,y) in enumerate(nodes):
            part=max(0,min(1,(q-.2-i*.1)/.22))
            if part > 0:
                draw.arc((x-10,y-10,x+10,y+10),0,int(360*part),fill=YELLOW if i==0 else color,width=3)
    else:
        # A neutral target/idea mark avoids arbitrary clip-art-like symbols.
        for i, rad in enumerate((r, int(r*.62), int(r*.28))):
            part = max(0, min(1, (q - i*.18) / .45))
            if part > 0:
                draw.arc((cx-rad,cy-rad,cx+rad,cy+rad), 0, int(360*part), fill=color if i else YELLOW, width=3)


def _draw_text_reveal_on_draw(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int],
                              font: ImageFont.ImageFont, color: str, progress: float,
                              max_width: int) -> None:
    """Reveal text inside an existing draw context using its own measured bounds."""
    if progress <= 0 or not text:
        return
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0])
    end = max(1, int(width * min(1.0, progress)))
    # Draw to a tiny local canvas, clip that label, then composite the visible prefix.
    temp = Image.new("RGBA", (min(max_width, width+4), max(1,bbox[3]-bbox[1]+4)), (0,0,0,0))
    td = ImageDraw.Draw(temp)
    td.text((1-bbox[0], 1-bbox[1]), text, font=font, fill=color)
    crop = temp.crop((0, 0, min(temp.width, end+2), temp.height))
    # ImageDraw has no paste, so use direct paste onto an internal image and draw it back.
    canvas = draw._image if hasattr(draw, "_image") else None
    if canvas is not None:
        canvas.paste(crop, (x, y), crop)
    else:
        draw.text((x,y), text[:max(1, round(len(text)*progress))], font=font, fill=color)


def _draw_causal_chain_scene(image: Image.Image, labels: list[str], p: float, idx: int) -> None:
    """Variable-node cause/effect diagram; never draws empty filler boxes."""
    d=ImageDraw.Draw(image)
    labels=[x for x in labels if x.strip()][:3]
    if len(labels)<2:
        return _draw_object_focus_scene(image, labels, p, "none", idx)
    if len(labels)==2:
        boxes=[(175,300,545,505),(735,300,1105,505)]
    else:
        boxes=[(55,315,365,505),(485,315,795,505),(915,315,1225,505)]
    for i,(box,label) in enumerate(zip(boxes,labels)):
        start=.05+i*.21
        q=max(0,min(1,(p-start)/.23))
        _draw_rect_progress(d,box,q,TEAL if i==1 else CHALK_DIM,3,idx*17+i)
        cx=(box[0]+box[2])//2
        _draw_motif(d,"none",cx,box[1]+52,48,max(0,min(1,(q-.13)/.35)),YELLOW if i==len(labels)-1 else TEAL,idx+i+40)
        _draw_text_reveal(image,label,(cx,box[1]+105),_font(27,True),CHALK,
                          max(0,min(1,(p-(start+.10))/.20)),box[2]-box[0]-28,5,"center")
    for i in range(len(boxes)-1):
        a=(boxes[i][2]+8,(boxes[i][1]+boxes[i][3])//2)
        b=(boxes[i+1][0]-8,(boxes[i+1][1]+boxes[i+1][3])//2)
        _draw_arrow(d,a,b,max(0,min(1,(p-(.27+i*.22))/.17)),YELLOW,idx+i+9)


def _draw_system_map_scene(image: Image.Image, labels: list[str], p: float, motif: str, idx: int) -> None:
    d = ImageDraw.Draw(image)
    cx, cy = 640, 405
    nodes = [(640, 230), (355, 475), (925, 475)]
    active_nodes = list(zip(nodes, labels[:3]))
    for i, ((x,y), label) in enumerate(active_nodes):
        _draw_arrow(d, (cx,cy), (x,y), max(0,min(1,(p-(.05+i*.16))/.18)), TEAL if i==1 else CHALK_DIM, idx+i)
    _draw_motif(d, motif, cx, cy, 118, max(0,min(1,(p-.04)/.42)), YELLOW, idx+12)
    for i, ((x,y), label) in enumerate(active_nodes):
        q = max(0,min(1,(p-(.22+i*.17))/.18))
        d.arc((x-14,y-14,x+14,y+14),0,int(360*q),fill=YELLOW if i==1 else CHALK,width=3)
        if i == 0:
            _draw_text_reveal(image,label,(x,y-62),_font(27,True),CHALK,q,390,4,"center")
        else:
            _draw_text_reveal(image,label,(x,y+36),_font(26),CHALK,q,300,5,"center")


def _draw_evidence_scene(image: Image.Image, labels: list[str], p: float, motif: str, idx: int) -> None:
    d = ImageDraw.Draw(image)
    doc_box = (175, 235, 540, 590)
    _draw_rect_progress(d, doc_box, max(0,min(1,p/.5)), CHALK_DIM, 3, idx)
    # Paper folds/lines reveal like underlined evidence, rather than putting a sentence in a box.
    for i in range(4):
        yy = 295 + i*48
        q = max(0,min(1,(p-(.16+i*.1))/.16))
        _draw_chalk_line(d,(215,yy),(500,yy),q,YELLOW if i==2 else CHALK_DIM,3,idx+i+3)
    if p > .34:
        _draw_rect_progress(d,(205,375,510,425),min(1,(p-.34)/.24),YELLOW,2,idx+10)
    _draw_arrow(d,(560,410),(690,410),max(0,min(1,(p-.4)/.16)),TEAL,idx+20)
    _draw_motif(d,motif,850,360,96,max(0,min(1,(p-.42)/.3)),YELLOW,idx+30)
    _draw_text_reveal(image,labels[0] if labels else "The evidence",(850,470),_font(30,True),CHALK,max(0,min(1,(p-.5)/.18)),470,5,"center")
    if len(labels)>1:
        _draw_text_reveal(image,labels[1],(850,525),_font(25),CHALK_DIM,max(0,min(1,(p-.64)/.17)),480,5,"center")
    if len(labels)>2:
        _draw_text_reveal(image,labels[2],(850,575),_font(25),YELLOW,max(0,min(1,(p-.76)/.16)),480,5,"center")


def _draw_feedback_scene(image: Image.Image, labels: list[str], p: float, idx: int) -> None:
    d = ImageDraw.Draw(image)
    all_nodes = [(640,245),(355,470),(925,470)]
    nodes = all_nodes[:max(2, min(3, len(labels)))]
    centers = nodes
    links = [(nodes[i], nodes[(i+1) % len(nodes)]) for i in range(len(nodes))]
    for i,(a,b) in enumerate(links):
        _draw_arrow(d,a,b,max(0,min(1,(p-(.05+i*.22))/.20)),YELLOW if i==len(links)-1 else TEAL,idx+i*3)
    font = _font(28,True)
    for i,(x,y) in enumerate(centers):
        q=max(0,min(1,(p-(.16+i*.2))/.19))
        r=94
        d.arc((x-r,y-r,x+r,y+r),0,int(360*q),fill=CHALK if i!=1 else TEAL,width=3)
        label=labels[i] if i<len(labels) else ""
        _draw_text_reveal(image,label,(x,y+112),font,CHALK,q,310,5,"center")
    if p > .75:
        _draw_text_reveal(image,"THE CYCLE",(640,385),_font(24,True),YELLOW,min(1,(p-.75)/.2),280,4,"center")


def _draw_decision_scene(image: Image.Image, labels: list[str], p: float, idx: int) -> None:
    d=ImageDraw.Draw(image)
    top=(480,225,800,335)
    left=(130,455,560,570)
    right=(720,455,1150,570)
    _draw_rect_progress(d,top,max(0,min(1,p/.22)),TEAL,3,idx)
    _draw_text_reveal(image,labels[0] if labels else "The choice",(640,263),_font(30,True),CHALK,max(0,min(1,(p-.08)/.2)),270,4,"center")
    _draw_arrow(d,(640,340),(345,438),max(0,min(1,(p-.27)/.18)),YELLOW,idx+2)
    _draw_arrow(d,(640,340),(935,438),max(0,min(1,(p-.44)/.18)),YELLOW,idx+8)
    _draw_rect_progress(d,left,max(0,min(1,(p-.4)/.23)),CHALK_DIM,3,idx+11)
    _draw_rect_progress(d,right,max(0,min(1,(p-.55)/.23)),CHALK,3,idx+12)
    if len(labels)>1:
        _draw_text_reveal(image,labels[1],(345,490),_font(28),CHALK,max(0,min(1,(p-.52)/.18)),370,5,"center")
    if len(labels)>2:
        _draw_text_reveal(image,labels[2],(935,490),_font(28),YELLOW,max(0,min(1,(p-.67)/.18)),370,5,"center")


def _draw_object_focus_scene(image: Image.Image, labels: list[str], p: float, motif: str, idx: int) -> None:
    d = ImageDraw.Draw(image)
    # A large focal symbol takes visual space; short labels act as annotations.
    _draw_motif(d, motif, 640, 365, 205, max(0,min(1,(p-.03)/.62)), YELLOW, idx+2)
    _draw_chalk_line(d,(250,360),(420,360),max(0,min(1,(p-.3)/.15)),TEAL,3,idx+7)
    _draw_chalk_line(d,(860,360),(1030,360),max(0,min(1,(p-.42)/.15)),TEAL,3,idx+8)
    if labels:
        _draw_text_reveal(image,labels[0],(335,310),_font(28,True),CHALK,max(0,min(1,(p-.25)/.18)),280,5,"center")
    if len(labels)>1:
        _draw_text_reveal(image,labels[1],(945,310),_font(28,True),CHALK,max(0,min(1,(p-.42)/.18)),300,5,"center")
    if len(labels)>2:
        _draw_text_reveal(image,labels[2],(640,535),_font(31,True),YELLOW,max(0,min(1,(p-.65)/.2)),760,5,"center")




def _draw_freeform_scene(base: Image.Image, title: str, actions: list[dict], progress: float,
                         idx: int) -> Image.Image:
    """Execute validated, AI-directed drawing primitives over the chalkboard.

    The Director chooses the actual geometry and order; this is not a fixed layout
    template. Every action is revealed over its own time interval in scene progress.
    """
    image = base.copy()
    w, h = image.size
    d = ImageDraw.Draw(image)
    palette = {"chalk": CHALK, "dim": CHALK_DIM, "teal": TEAL, "yellow": YELLOW}
    safe_title = re.sub(r"\s+", " ", str(title or "KEY IDEA")).strip()[:32]
    _draw_text_reveal(image, safe_title, (w // 2, int(h * .055)), _font(31, True), CHALK,
                      max(0.0, min(1.0, progress / .11)), int(w * .78), 4, "center")
    d = ImageDraw.Draw(image)
    _draw_chalk_line(d, (int(w*.37), int(h*.145)), (int(w*.63), int(h*.145)),
                     max(0.0, min(1.0, (progress-.045)/.11)), YELLOW, 3, idx+77)

    for action_idx, a in enumerate(actions):
        try:
            start = float(a.get("start", 0.0))
            end = float(a.get("end", 1.0))
            q = max(0.0, min(1.0, (progress-start) / max(.001, end-start)))
            if q <= 0:
                continue
            color = palette.get(str(a.get("color", "chalk")), CHALK)
            width = max(2, min(6, int(a.get("width", 3))))
            typ = a["type"]
            if typ in {"line", "arrow", "underline", "box", "scribble"}:
                x1, y1 = int(float(a["x1"])*w), int(float(a["y1"])*h)
                x2, y2 = int(float(a["x2"])*w), int(float(a["y2"])*h)
                if typ == "box":
                    _draw_rect_progress(d, (x1, y1, x2, y2), q, color, width, idx*31+action_idx)
                elif typ == "arrow":
                    _draw_arrow(d, (x1,y1), (x2,y2), q, color, idx*31+action_idx)
                elif typ == "scribble":
                    left,right=sorted((x1,x2)); top,bottom=sorted((y1,y2))
                    rows=max(2,min(6,round((bottom-top)/15)))
                    for ri in range(rows):
                        yy=top+(bottom-top)*(ri+.5)/rows
                        pts=[(left, int(yy)), (left+(right-left)*.25, int(yy-5)),
                             (left+(right-left)*.5, int(yy+4)), (left+(right-left)*.75, int(yy-3)), (right,int(yy))]
                        _draw_polyline_progress(d, [(int(x),int(y)) for x,y in pts],
                                                max(0,min(1,(q-ri/rows)*rows)), color, width, idx*31+action_idx+ri)
                else:
                    _draw_chalk_line(d, (x1,y1), (x2,y2), q, color, width, idx*31+action_idx)
            elif typ == "circle":
                cx, cy = int(float(a["x"])*w), int(float(a["y"])*h)
                radius = int(float(a.get("size", .08))*h)
                d.arc((cx-radius,cy-radius,cx+radius,cy+radius), 205, 205+int(345*q), fill=color, width=width)
            elif typ == "motif":
                cx, cy = int(float(a["x"])*w), int(float(a["y"])*h)
                size = int(float(a.get("size", .1))*h)
                _draw_motif(d, str(a.get("motif", "none")), cx, cy, size, q, color, idx*31+action_idx)
            elif typ == "text":
                size_map = {"small": 25, "medium": 34, "large": 47}
                font = _font(size_map.get(str(a.get("size", "medium")), 34), bold=(a.get("size") == "large"))
                x,y = int(float(a["x"])*w), int(float(a["y"])*h)
                align = str(a.get("align", "center"))
                max_width = int(float(a.get("max_width", .28))*w)
                _draw_text_reveal(image, str(a.get("text", "")), (x,y), font, color, q,
                                  max_width, 5, align)
            d = ImageDraw.Draw(image)
        except (ValueError, KeyError, TypeError, OverflowError):
            # Plans are validated before rendering; fail-safe skip guards against corrupt optional metadata.
            continue
    return image.convert("RGB")

def _draw_scene(base: Image.Image, text: str, index: int, progress: float,
                scene_plan: dict | None = None) -> Image.Image:
    """Render one directed beat.  The scene plan, not generic heuristics, owns layout."""
    image = base.copy()
    meta = scene_plan or {}
    custom_actions = meta.get("drawing_actions")
    if isinstance(custom_actions, list) and len(custom_actions) >= 4:
        return _draw_freeform_scene(base, str(meta.get("title") or "KEY IDEA"), custom_actions, progress, index)
    kind = str(meta.get("composition") or meta.get("visual_kind") or _scene_kind(text, index))
    kind = {
        "flow": "causal_chain", "compare": "comparison", "cards": "object_focus",
        "callout": "spotlight",
    }.get(kind, kind)
    labels = meta.get("labels") or meta.get("phrases") or _phrases(text, 3)
    labels = [str(x).strip() for x in labels if str(x).strip()]
    # Keep labels distinct even when a model proposes near-duplicates.
    unique = []
    used = set()
    for label in labels:
        norm = re.sub(r"[^a-z0-9]+", " ", label.casefold()).strip()
        if norm and norm not in used:
            unique.append(label)
            used.add(norm)
    labels = unique[:3]
    if not labels:
        labels = _phrases(text, 3)
    d = ImageDraw.Draw(image)
    # No repeated FIELD NOTES / HOW IT UNFOLDS header.  The plan supplies a
    # short, beat-specific title; index stays unobtrusive for QA traceability.
    title = str(meta.get("title") or {
        "causal_chain": "CAUSE → EFFECT", "timeline": "TIME & CONSEQUENCE",
        "comparison": "TWO SIDES", "numbers": "THE NUMBER",
        "system_map": "INSIDE THE SYSTEM", "evidence": "FOLLOW THE EVIDENCE",
        "spotlight": "THE REVEAL", "feedback_loop": "THE CYCLE",
        "decision_tree": "THE CHOICE", "object_focus": "THE SIGNAL",
    }.get(kind, "KEY IDEA"))
    title = re.sub(r"\s+", " ", title).strip()[:36]
    marker_font = _font(16, True)
    d.text((44, 35), f"{index+1:02d}", font=marker_font, fill=CHALK_DIM)
    _draw_text_reveal(image, title, (640, 48), _font(31, True), CHALK,
                      max(0, min(1, (progress-.015)/.16)), 980, 4, "center")
    _draw_chalk_line(ImageDraw.Draw(image), (470, 105), (810, 105),
                     max(0, min(1, (progress-.06)/.14)), YELLOW, 3, index+77)

    motif = str(meta.get("motif") or _local_motif_for_render(text, kind))
    if kind == "causal_chain":
        _draw_causal_chain_scene(image, labels, progress, index)
    elif kind == "timeline":
        _draw_timeline_scene(image, labels + [""] * (3-len(labels)), progress, index)
    elif kind == "comparison":
        if len(labels) >= 2:
            _draw_comparison_scene(image, labels, progress, index)
        else:
            _draw_spotlight_scene_v2(image, labels, progress, motif, index)
    elif kind == "numbers":
        _draw_numbers_scene_v2(image, labels, text, progress, index)
    elif kind == "system_map":
        _draw_system_map_scene(image, labels, progress, motif, index)
    elif kind == "evidence":
        _draw_evidence_scene(image, labels, progress, motif, index)
    elif kind == "feedback_loop":
        _draw_feedback_scene(image, labels, progress, index)
    elif kind == "decision_tree":
        if len(labels) >= 3:
            _draw_decision_scene(image, labels, progress, index)
        else:
            _draw_object_focus_scene(image, labels, progress, motif, index)
    elif kind == "spotlight":
        _draw_spotlight_scene_v2(image, labels, progress, motif, index)
    else:
        _draw_object_focus_scene(image, labels, progress, motif, index)

    return image.convert("RGB")


def _local_motif_for_render(text: str, kind: str) -> str:
    try:
        from moduli.chalkboard_director import _local_motif
        return _local_motif(text, kind)
    except Exception:
        return "none"


def _draw_comparison_scene(image: Image.Image, labels: list[str], p: float, idx: int) -> None:
    d=ImageDraw.Draw(image)
    box=(150,205,1130,570)
    _draw_rect_progress(d,box,max(0,min(1,(p-.03)/.24)),CHALK_DIM,3,idx)
    split_x=640
    _draw_chalk_line(d,(split_x,206),(split_x,570),max(0,min(1,(p-.18)/.2)),TEAL,3,idx+3)
    _draw_chalk_line(d,(151,340),(1129,340),max(0,min(1,(p-.25)/.14)),CHALK_DIM,2,idx+4)
    left=labels[0] if labels else "First case"
    right=labels[1] if len(labels)>1 else "Second case"
    _draw_text_reveal(image,left,(395,245),_font(31,True),TEAL,max(0,min(1,(p-.10)/.22)),400,5,"center")
    _draw_text_reveal(image,right,(885,245),_font(31,True),YELLOW,max(0,min(1,(p-.25)/.22)),400,5,"center")
    if len(labels)>2:
        # Use a lower takeaway strip only when the plan supplies one.
        _draw_chalk_line(d,(280,445),(1000,445),max(0,min(1,(p-.42)/.12)),CHALK_DIM,2,idx+9)
        _draw_text_reveal(image,labels[2],(640,485),_font(31,True),CHALK,max(0,min(1,(p-.48)/.23)),760,5,"center")
    else:
        _draw_text_reveal(image,"≠",(640,385),_font(58,True),YELLOW,max(0,min(1,(p-.42)/.14)),120,5,"center")


def _draw_numbers_scene_v2(image: Image.Image, labels: list[str], text: str, p: float, idx: int) -> None:
    d=ImageDraw.Draw(image)
    nums=[]
    for n in re.findall(r"(?:\$\s?\d[\d,.]*|\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|trillion|thousand|years?|months?)?)",text,flags=re.I):
        n=re.sub(r"\s+"," ",n).strip()
        if n and n.casefold() not in {x.casefold() for x in nums}:
            nums.append(n)
    nums=nums[:3]
    if not nums:
        # Don't fake metrics like KEY / FIGURE / RESULT if the narration gives no number.
        return _draw_object_focus_scene(image, labels, p, "chart", idx)
    if len(nums)==1:
        _draw_motif(d,"chart",440,405,145,max(0,min(1,(p-.1)/.55)),TEAL,idx)
        _draw_text_reveal(image,nums[0],(870,330),_font(64,True),YELLOW,max(0,min(1,(p-.10)/.24)),530,5,"center")
        if labels:
            _draw_text_reveal(image,labels[0],(870,420),_font(32,True),CHALK,max(0,min(1,(p-.35)/.22)),500,5,"center")
        if len(labels)>1:
            _draw_text_reveal(image,labels[1],(870,480),_font(25),CHALK_DIM,max(0,min(1,(p-.57)/.2)),500,5,"center")
    else:
        xs=[360,920] if len(nums)==2 else [260,640,1020]
        for i,num in enumerate(nums):
            q=max(0,min(1,(p-(.08+i*.2))/.22))
            x=xs[i]
            r=88
            d.arc((x-r,300-r,x+r,300+r),210,210+int(300*q),fill=YELLOW if i==len(nums)-1 else TEAL,width=4)
            _draw_text_reveal(image,num,(x,285),_font(39 if len(nums)==3 else 46,True),YELLOW if i==len(nums)-1 else CHALK,q,310,4,"center")
            if i<len(labels):
                _draw_text_reveal(image,labels[i],(x,425),_font(26),CHALK_DIM,max(0,min(1,(p-(.17+i*.2))/.21)),320,5,"center")


def _draw_spotlight_scene_v2(image: Image.Image, labels: list[str], p: float, motif: str, idx: int) -> None:
    d=ImageDraw.Draw(image)
    _draw_motif(d,motif,640,350,188,max(0,min(1,(p-.04)/.54)),YELLOW,idx+1)
    if labels:
        _draw_text_reveal(image,labels[0],(640,485),_font(37,True),CHALK,max(0,min(1,(p-.38)/.20)),780,5,"center")
    if len(labels)>1:
        _draw_text_reveal(image,labels[1],(640,540),_font(26),CHALK_DIM,max(0,min(1,(p-.56)/.20)),800,5,"center")
    if len(labels)>2:
        q=max(0,min(1,(p-.61)/.23))
        d.arc((240,450,1040,590),190,190+int(330*q),fill=TEAL,width=3)
        _draw_text_reveal(image,labels[2],(640,600),_font(25,True),YELLOW,q,760,4,"center")


def _find_audio_duration(audio_path: str) -> float:
    from moduli.montaggio import _media_duration
    return max(0.2, float(_media_duration(audio_path)))


def render_chalkboard_video(audio_path: str, script_text: str, output_path: str, *, mood=None,
                            on_progress=None, custom_music=None, use_ai_director: bool = False,
                            align_audio: bool = False, aligner_model: str = "base",
                            scene_plan: dict | None = None) -> None:
    """Create a local chalkboard animation from a validated, story-led scene plan.

    The new planner is separate from the drawing engine. Speech timing is
    explicitly marked as measured local word timestamps or a text-based estimate.
    No external request is made unless use_ai_director=True.
    """
    from moduli.montaggio import (
        _ffmpeg, _media_duration, _mux_audio_ffmpeg, _pick_music,
        OUTPUT_W, OUTPUT_H, OUTPUT_FPS,
    )
    from moduli.chalkboard_director import build_scene_plan

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Chalkboard mode: narration audio not found: {audio_path}")
    duration = _find_audio_duration(audio_path)
    width, height = int(OUTPUT_W), int(OUTPUT_H)
    fps = max(12, min(24, int(OUTPUT_FPS)))
    total_frames = max(1, int(math.ceil(duration * fps)))
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if scene_plan is None:
        plan = build_scene_plan(
            script_text, duration, audio_path=audio_path,
            use_ai=use_ai_director, align_audio=align_audio, aligner_model=aligner_model,
        )
    else:
        plan = scene_plan
        if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list):
            raise ValueError("Provided chalkboard scene plan is invalid")
    plan_rows = plan.get("scenes", [])
    if not plan_rows:
        raise ValueError("Director plan has no scenes; refusing to render an empty video")

    # Compute continuous frame intervals from the timeline in the JSON plan.
    # Derive frames by scene end to avoid rounding gaps; last frame count matches audio.
    scene_frames: list[int] = []
    frame_cursor = 0
    for i, row in enumerate(plan_rows):
        end_frame = total_frames if i == len(plan_rows) - 1 else int(round(float(row["end_seconds"]) * fps))
        end_frame = max(frame_cursor + 1, min(total_frames, end_frame))
        scene_frames.append(end_frame - frame_cursor)
        frame_cursor = end_frame
    # If short timing edges pushed a scene to the final frame early, fold any
    # residual frame count into the last scene rather than losing audio duration.
    if sum(scene_frames) != total_frames:
        scene_frames[-1] += total_frames - sum(scene_frames)
    if any(n <= 0 for n in scene_frames):
        raise ValueError("Director created a zero-length scene; refusing to render")

    for row in plan_rows:
        sd = max(0.01, float(row["end_seconds"]) - float(row["start_seconds"]))
        row["estimated_speaking_rate_wpm"] = round(int(row.get("word_count", len(str(row.get("narration_excerpt", "")).split()))) / sd * 60, 1)
    plan["duration_seconds"] = round(duration, 3)
    plan["scene_count"] = len(plan_rows)
    plan["render_dimensions"] = [width, height]
    plan["fps"] = fps
    plan["planner_version"] = "pass-4-coordinate-action-director"
    plan_path = target.with_suffix(".scene_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    scenes = [str(row.get("narration_excerpt", "")).strip() for row in plan_rows]
    if any(not s for s in scenes):
        raise ValueError("Director plan includes a blank narration beat; refusing to render")

    design_width, design_height = 1280, 720
    board = _board_background(design_width, design_height)
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="fern_chalkboard_", dir=str(target.parent)) as temp_dir:
        silent_path = str(Path(temp_dir) / "chalkboard_silent.mp4")
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-t", f"{duration:.3f}", silent_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        completed = 0
        last_report = -1
        try:
            assert proc.stdin is not None
            for i, row in enumerate(plan_rows):
                frames_this_scene = scene_frames[i]
                scene_text = scenes[i]
                for local_frame in range(frames_this_scene):
                    progress = local_frame / max(1, frames_this_scene - 1)
                    frame = _draw_scene(board, scene_text, i, progress, row)
                    if frame.size != (width, height):
                        frame = frame.resize((width, height), Image.Resampling.LANCZOS)
                    proc.stdin.write(frame.tobytes())
                    completed += 1
                    pct = int(completed / total_frames * 80)
                    if on_progress and pct >= last_report + 5:
                        last_report = pct
                        elapsed = time.time() - started
                        eta = elapsed / max(1, completed) * max(0, total_frames - completed)
                        on_progress(pct, f"chalkboard drawing; about {int(eta // 60)}m {int(eta % 60)}s left")
            proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr else b""
            return_code = proc.wait()
            if return_code:
                raise RuntimeError("Chalkboard FFmpeg render failed: " + stderr.decode("utf-8", "replace")[-3000:])
        except (BrokenPipeError, OSError) as exc:
            if proc.poll() is None:
                proc.kill()
            stderr = proc.stderr.read() if proc.stderr else b""
            raise RuntimeError("Chalkboard frame stream failed: " + stderr.decode("utf-8", "replace")[-2000:]) from exc

        if on_progress:
            on_progress(86, "mixing narration and music")
        music_path = custom_music if custom_music and os.path.isfile(custom_music) else _pick_music(mood)
        _mux_audio_ffmpeg(silent_path, audio_path, str(target), music_path, duration)
        if on_progress:
            on_progress(100, "chalkboard video ready")
    print(f"[chalkboard] Scene plan saved: {plan_path}", flush=True)
    print(f"[chalkboard] Director: {plan.get('director', 'unknown')} | timing: {plan.get('timing_source', 'unknown')}", flush=True)
    if plan.get("alignment_note"):
        print(f"[chalkboard] Alignment note: {plan['alignment_note']}", flush=True)
    if plan.get("ai_director_note"):
        print(f"[chalkboard] AI planning note: {plan['ai_director_note']}", flush=True)
    print(f"[chalkboard] Rendered {len(plan_rows)} directed chalk scenes: {target}", flush=True)

