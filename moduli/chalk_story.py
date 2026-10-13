"""Local story director for the chalk engine.

Given narrated beats (rows with timing and ``narration_excerpt``), decide the
visual story: which layout carries each beat, which illustrations are its
visual metaphors, which short chalk captions appear, and which *spoken word*
triggers each drawing.  It is rule-based and deterministic; an AI director may
override it (see ``sanitize_chalk``), and its output is always validated.

Design rules it follows:
* content is budgeted by beat length - a 3 s beat gets one drawing and a label,
  a 7 s beat can carry a diagram
* no layout is repeated back-to-back, hero illustrations are not reused
  while a fresh one fits, and hard topic breaks get an eraser wipe
* the *punchline* of a beat (number, negation, "secret", ...) is what gets the
  yellow emphasis
"""
from __future__ import annotations

import re
from typing import Any

from moduli.chalk_composer import LAYOUTS
from moduli.chalk_illustrations import KEYWORDS, LIBRARY

MIN_SECONDS = {
    "statement": 1.8, "hero": 2.6, "question": 3.0, "bignumber": 3.0, "duo": 3.6, "iceberg": 4.0,
    "versus": 4.6, "network": 6.5, "chain": 4.0, "funnel": 5.0, "timeline": 5.4, "evidence": 5.4,
}
_STOP = set("""a an the and or but so if than then of to in on at by for from with as is are was were be been being it its
this that these those there here you your we our they their he she his her them i me my not no do does did have has had
will would could should can may might what when where why how who which because into about over under up down also just
very really actually still now every all it's i'm you're don't doesn't isn't more most some any each""".split())
_NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
_UNITS = r"(minutes?|hours?|days?|weeks?|months?|years?|decades?|percent|%|million|billion|trillion|thousand|dollars?|users?|people|seconds?|times)"
_TOKEN = re.compile(r"[A-Za-z0-9$%][A-Za-z0-9$%'’\-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------ beat segmentation
_BREAK_WORDS = {"that", "which", "and", "but", "because", "while", "where", "so", "then", "when"}


def _split_long(words: list[str], max_w: int) -> list[list[str]]:
    if len(words) <= max_w:
        return [words]
    mid = len(words) / 2
    best, best_cost = None, 1e9
    for i in range(4, len(words) - 3):
        tok = words[i - 1]
        strong = tok.endswith((",", ";", ":", "—")) or words[i] == "—" or tok == "—"
        weak = words[i].casefold() in _BREAK_WORDS
        if not (strong or weak):
            continue
        cost = abs(i - mid) + (0 if strong else 3.0)
        if cost < best_cost:
            best, best_cost = i, cost
    if best is None:
        best = int(mid)
    return _split_long(words[:best], max_w) + _split_long(words[best:], max_w)


def segment_beats(script: str, target_words: int) -> list[str]:
    """Split a script into spoken beats that respect sentence structure.

    A beat is a run of whole sentences of roughly ``target_words`` words (long sentences are
    split at the strongest nearby comma/dash/conjunction).  A very short, dramatic sentence
    becomes its own beat - a 'statement card' - when it opens the piece or follows a
    substantial beat, which gives the video its rhythm: build, build, land the line.
    """
    text = re.sub(r"\s+", " ", script).strip()
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text) if x]
    hard_max = max(16, int(target_words * 1.5))
    pieces: list[list[str]] = []
    for sent in sents:
        pieces += _split_long(sent.split(), hard_max)
    beats: list[list[str]] = []
    cur: list[str] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            beats.append(cur)
            cur = []

    for idx, piece in enumerate(pieces):
        punchy = len(piece) <= 6 and piece[-1].endswith((".", "!", "?"))
        if punchy and (idx == 0 or len(cur) >= 12):
            flush()
            beats.append(list(piece))      # statement card
            continue
        if cur and len(cur) + len(piece) > hard_max:
            flush()
        cur += piece
        if len(cur) >= target_words:
            flush()
    flush()
    # a trailing sliver is merged back instead of becoming a stray scene
    if len(beats) >= 2 and len(beats[-1]) <= 3:
        beats[-2] += beats.pop()
    return [" ".join(b) for b in beats]


# --------------------------------------------------------------------- numbers
def find_number(text: str) -> tuple[str, str] | None:
    """(display, spoken_word_to_anchor) for the beat's most prominent figure."""
    m = re.search(r"(\$\s?\d[\d,\.]*\s?(?:million|billion|trillion|thousand|[kKmMbB](?![A-Za-z]))?|\d[\d,\.]*\s?(?:%|percent|million|billion|trillion|thousand|minutes?|hours?|days?|years?|months?|weeks?|times)?)", text)
    if m:
        disp = re.sub(r"\s+", " ", m.group(1)).strip().upper().replace("PERCENT", "%").replace(" %", "%")
        return disp, m.group(1).split()[0]
    m = re.search(r"\b(" + "|".join(_NUMWORDS) + r")\s+" + _UNITS + r"\b", text, re.I)
    if m:
        return f"{_NUMWORDS[m.group(1).lower()]} {m.group(2).upper()}".replace("PERCENT", "%"), m.group(1)
    m = re.search(r"\b(million|billion|trillion)s?\b", text, re.I)
    if m:
        return m.group(1).upper(), m.group(1)
    return None


# ---------------------------------------------------------------- enumerations
def list_items(text: str) -> list[str]:
    """Items of an enumeration like 'not by a teacher, not by a boss, not by a bureau'."""
    chunks = [c.strip(" .;:—-\"'") for c in re.split(r"[,;]|\band\b(?=\s+(?:not|by|a|an|the)\b)", text) if c.strip(" .;:—-\"'")]
    if len(chunks) < 3:
        return []
    cleaned = []
    for c in chunks:
        for _ in range(3):
            c = re.sub(r"^(?:and|or|not|nor|by|from|with|to|a|an|the|of|every|each|any|all|some|your|their|its)\s+", "", c, flags=re.I)
        words = [w for w in _tokens(c)]
        if 1 <= len(words) <= 4:
            cleaned.append(" ".join(words))
    # an enumeration needs *short parallel* items; skip ordinary prose
    if len(cleaned) >= 3 and len(cleaned) >= len(chunks) - 1:
        return cleaned[:4]
    return []


# --------------------------------------------------------------------- captions
def _content_words(text: str) -> list[str]:
    return [w for w in _tokens(text) if w.casefold().strip("'’") not in _STOP]


_WEAK = set("""much many even barely right now long about still ago opened help spend type types feeds feed decides goes comes
get got make makes made say says said go going come came one way thing things kind sort lot lots well back own same such
deployed assigns assign wants want need needs let lets start starts just only ever never always often really actually""".split())
_KEYWORD_SET = {w for kws in KEYWORDS.values() for w in kws if " " not in w}


def _strong(w: str, first: bool) -> bool:
    lw = w.casefold().strip("'’")
    return (lw in _KEYWORD_SET or any(c.isdigit() for c in w) or w.startswith("$")
            or (w[:1].isupper() and not first) or (len(lw) >= 8 and lw not in _WEAK))


def captions(text: str, k: int, avoid: set[str] | None = None) -> list[str]:
    """Short (1-3 word) chalk captions, in spoken order, from the highest-information phrases."""
    avoid = {a.casefold() for a in (avoid or set())}
    matches = list(_TOKEN.finditer(text))
    words = [m.group(0) for m in matches]
    scored: list[tuple[float, int, int, str]] = []
    for n in (2, 3, 1):
        for i in range(0, len(words) - n + 1):
            seg = words[i:i + n]
            low = [w.casefold().strip("'’") for w in seg]
            if len(set(low)) != len(low):
                continue  # avoid awkward captions such as "FEEDBACK FEEDBACK"
            # Do not form a phrase from the last word of one sentence and the
            # first word of the next sentence (tokens alone lose this boundary).
            if n > 1 and re.search(
                r"[.!?;:\n]",
                text[matches[i].end():matches[i + n - 1].start()],
            ):
                continue
            if low[0] in _STOP or low[-1] in _STOP or low[0] in _WEAK or low[-1] in _WEAK:
                continue
            if n == 3 and low[1] in _STOP and low[1] not in {"of", "for", "to", "from"}:
                continue
            strong = sum(1 for j, w in enumerate(seg) if _strong(w, i + j == 0))
            weak = sum(1 for w in low if w in _WEAK or w in _STOP)
            if strong == 0 or (n == 1 and strong < 1) or (n == 3 and strong < 2):
                continue
            score = strong * 1.6 + (n - strong - weak) * 0.35 - weak * 0.9
            if strong == n and n > 1:
                score += 0.9            # a run of strong words is a real noun phrase
            for w in seg:
                if any(c.isdigit() for c in w) or w.startswith("$"):
                    score += 1.6
                if w.casefold() in {"secret", "hidden", "silent", "score", "trust", "invisible", "ration", "dangerous", "graded"}:
                    score += 1.0
            score += {1: -0.5, 2: 0.6, 3: 0.1}[n]
            phrase = " ".join(seg).upper()
            if phrase.casefold() in avoid or len(phrase) > 24:
                continue
            scored.append((score, i, i + n, phrase))
    scored.sort(key=lambda x: -x[0])
    chosen: list[tuple[int, int, str]] = []
    for score, a, b, phrase in scored:
        if any(not (b <= ca or a >= cb) for ca, cb, _ in chosen):
            continue
        chosen.append((a, b, phrase))
        if len(chosen) >= k:
            break
    chosen.sort()
    return [c[2] for c in chosen]


def headline(text: str) -> str:
    """A 2-4 word chapter-ish title, never a sentence."""
    caps = captions(text, 1)
    return caps[0] if caps else ""


def shorten_statement(text: str, limit: int = 24) -> str:
    """Trim a punchy sentence to something that can be written in about a second and a half."""
    t = re.sub(r"[.!?\"]+$", "", text.strip()).upper()
    if len(t) <= limit:
        return t
    t2 = re.sub(r"\b(RIGHT NOW|EVEN|BARELY|JUST|REALLY|ACTUALLY|VERY)\b\s*", "", t).strip()
    t2 = re.sub(r"\b(THIS|THAT|IT|THEY|WE|YOU) (ISN'T|IS NOT|AREN'T|ARE NOT)\b", "NOT", t2).strip()
    t2 = re.sub(r"^(AND|BUT|SO|THEN|IT IS|THIS IS|THAT IS|YOU ARE|YOU'RE)\s+", "", t2).strip()
    if 4 <= len(t2) <= limit:
        return t2
    if len(t2) <= 28:
        return t2
    caps = captions(text, 1)
    return caps[0] if caps else t2[:limit].rsplit(" ", 1)[0]


def punchline(text: str, limit: int = 24) -> str:
    """Short declarative sentence usable as a big 'statement' card."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    best = ""
    for s in sents:
        w = _tokens(s)
        if 2 <= len(w) <= 7 and (not best or len(w) < len(_tokens(best))):
            best = s
    return shorten_statement(best, limit)


# ----------------------------------------------------------------- layout choice
def _layout_scores(text: str, dur: float, items: list[str], n_motifs: int, has_num: bool) -> dict[str, float]:
    s = text.casefold()
    sc = {k: 0.0 for k in LAYOUTS}
    sc["hero"] = 1.0
    sc["duo"] = 0.7 + (0.5 if n_motifs >= 2 else 0)
    words = len(_tokens(text))
    if has_num:
        sc["bignumber"] += 2.6
    if re.search(r"\b(hidden|never see|cannot appeal|can barely|beneath|underneath|invisible|behind the scenes|unseen|silent|secret(?:ly)?|quietly|prove it exists)\b", s):
        sc["iceberg"] += 2.6
    if re.search(r"\b(vs\.?|versus|compared|unlike|instead of|rather than|whereas|on the other hand|but the|but real|while the)\b", s):
        sc["versus"] += 2.4
    if len(items) >= 3:
        sc["chain"] += 3.2
    if re.search(r"\b(leads? to|led to|results? in|therefore|then|feeds|decides|so that|which means|as a result|turns? into|becomes?|because)\b", s):
        sc["chain"] += 1.6
    if re.search(r"\b(filter(?:ed|s)?|rank(?:ed|ing)?|compare[sd]?|select(?:ed)?|narrow|pipeline|passes? through|sorted|picked|chose|choose|gates?)\b", s):
        sc["funnel"] += 2.2
    if re.search(r"\b(in (?:19|20)\d\d|decade|century|first|next|finally|years? (?:ago|later)|before|after|timeline|history)\b", s):
        sc["timeline"] += 2.0
    if re.search(r"\b(document|memo|leak(?:ed)?|paper|report|policy|prompt|filing|records?|evidence|research papers?|privacy policy|directive)\b", s):
        sc["evidence"] += 2.3
    if re.search(r"\b(system|network|platform|connected|ecosystem|infrastructure|scaffold|architecture|mechanism)\b", s):
        sc["network"] += 1.9
    if "?" in text or re.match(r"^(how|why|what|who|is|are|can|could|would|do|does)\b", s):
        sc["question"] += 2.4
    for needs_cue in ("bignumber", "iceberg", "versus", "chain", "funnel", "timeline", "evidence", "network", "question"):
        if sc[needs_cue] == 0.0:
            sc[needs_cue] = -5.0        # a cue-driven layout without its cue would be decoration
    if words <= 9:
        sc["statement"] += 2.5
    if re.search(r"\b(this is not|it is the|that is the|the result|the truth|here is)\b", s):
        sc["statement"] += 1.4
    return sc


def choose_layout(text: str, dur: float, items: list[str], n_motifs: int, has_num: bool, recent: list[str]) -> str:
    counts = {k: recent.count(k) for k in set(recent)}
    sc = _layout_scores(text, dur, items, n_motifs, has_num)
    for lay in list(sc):
        if dur < MIN_SECONDS[lay]:
            sc[lay] -= 6.0
        sc[lay] -= 1.7 * counts.get(lay, 0)
        if recent and lay == recent[-1]:
            sc[lay] -= 4.0
        elif lay in recent[-3:]:
            sc[lay] -= 0.9
    best = max(sc.items(), key=lambda kv: kv[1])
    return best[0]


# ------------------------------------------------------------------ motif choice
def _matched_word(text: str, name: str) -> str | None:
    toks = {t.casefold(): t for t in _tokens(text)}
    low = " " + re.sub(r"[^a-z0-9$'’ -]+", " ", text.casefold()) + " "
    for kw in KEYWORDS.get(name, ()):
        if (" " in kw or "-" in kw) and f" {kw} " in low:
            return kw.split()[0]
        if kw in toks:
            return toks[kw]
    return None


def pick_motifs(text: str, k: int, used: dict[str, int], layout_hint: str = "") -> list[str]:
    s = " " + re.sub(r"[^a-z0-9$'’ -]+", " ", text.casefold()) + " "
    toks = set(s.split())
    scored: list[tuple[float, str]] = []
    for name, kws in KEYWORDS.items():
        if name not in LIBRARY:
            continue
        sc = 0.0
        for w in kws:
            if (" " in w or "-" in w) and f" {w} " in s:
                sc += 2.0
            elif w in toks:
                sc += 1.0
        if sc:
            sc -= 0.55 * used.get(name, 0)           # prefer fresh imagery
            scored.append((sc, name))
    scored.sort(key=lambda x: -x[0])
    out = [n for sc, n in scored if sc > -0.2][:k]
    fallback = [n for n in ("document", "person", "gear", "network", "eye", "magnifier", "lightbulb", "chip") if n not in out]
    fallback.sort(key=lambda n: used.get(n, 0))
    for n in fallback:
        if len(out) >= k:
            break
        out.append(n)
    out = out[:k]
    # draw in the order the ideas are spoken
    order = {t.casefold(): n for n, t in enumerate(_tokens(text))}
    def first_pos(name: str) -> int:
        hits = [order[w] for w in KEYWORDS.get(name, ()) if " " not in w and w in order]
        return min(hits) if hits else 10_000
    return sorted(out, key=first_pos) if len(out) > 1 else out


# ------------------------------------------------------------------- main entry
def plan_chalk_scenes(rows: list[dict[str, Any]]) -> None:
    """Attach ``row['chalk']`` (layout, motifs, captions, anchors, ...) to every scene row."""
    recent_layouts: list[str] = []
    used: dict[str, int] = {}
    for i, row in enumerate(rows):
        text = _clean(str(row.get("narration_excerpt", "")))
        dur = float(row["end_seconds"]) - float(row["start_seconds"])
        items = list_items(text)
        num = find_number(text)
        motifs = pick_motifs(text, 4, used)
        layout = choose_layout(text, dur, items, len(motifs), bool(num), recent_layouts)
        if layout != "iceberg":
            motifs = [m for m in motifs if m != "iceberg"] or pick_motifs(text, 3, {**used, "iceberg": 99})
        ch: dict[str, Any] = {"layout": layout}
        anchors: dict[str, str] = {}
        labels: list[str] = []
        title = ""

        if layout == "chain":
            if items:
                labels = [it.upper() for it in items[:4]]
                motifs = [(pick_motifs(it, 1, used) or ["document"])[0] for it in items[:4]]
                for j, it in enumerate(items[:4]):
                    anchors[f"motif{j}"] = it.split()[-1]
                    anchors[f"label{j}"] = it.split()[-1]
            else:
                labels = captions(text, 3)
                motifs = motifs[:3]
                for j, lab in enumerate(labels):
                    anchors[f"motif{j}"] = lab.split()[0]
                    anchors[f"label{j}"] = lab.split()[0]
            title = headline(text) if not items else ""
        elif layout == "bignumber":
            disp, spoken = num or ("?", "")
            ch["number"] = disp
            labels = captions(text, 2, avoid={disp})
            anchors["number"] = spoken or (labels[0].split()[0] if labels else "")
            motifs = [m for m in motifs if m not in {"coins"}] or motifs
            if labels:
                anchors["label0"] = labels[0].split()[0]
        elif layout == "statement":
            title = punchline(text, max(10, int((dur - 0.9) * 11))) or shorten_statement(text) or (captions(text, 1) or ["LOOK CLOSER"])[0]
            labels = [c for c in captions(text, 2, avoid={title}) if c != title][:1]
            anchors["title"] = title.split()[0]
            neg = re.search(r"\b(not|never|cannot|can't|nobody|no one|none)\b", text, re.I)
            anchors["emph"] = neg.group(1) if neg else title.split()[-1]
            motifs = motifs[:1]
        elif layout == "iceberg":
            labels = captions(text, 2)
            motifs = ["iceberg"] + [m for m in motifs if m != "iceberg"][:1]
            hid = re.search(r"\b(hidden|never|silent(?:ly)?|secret(?:ly)?|invisible|quietly|cannot|beneath|unseen)\b", text, re.I)
            anchors["motif0"] = _tokens(text)[0]
            if labels:
                anchors["label0"] = labels[0].split()[0]
            if hid:
                anchors["label1"] = hid.group(1)
            labels = ["WHAT YOU SEE"] + [labels[-1] if labels else "WHAT YOU DON'T"]
        elif layout in ("versus", "duo"):
            motifs = motifs[:2] if len(motifs) >= 2 else motifs + ["question"]
            labels = captions(text, 3)
            for j in range(2):
                m = _matched_word(text, motifs[j])
                if m:
                    anchors[f"motif{j}"] = m
                if j < len(labels):
                    anchors[f"label{j}"] = labels[j].split()[0]
            if layout == "duo":
                title = ""
                anchors["arrow"] = labels[0].split()[-1] if labels else ""
                if len(labels) > 2:
                    anchors["label2"] = labels[2].split()[0]
        elif layout == "question":
            q = re.search(r"([^.!?]*\?)", text)
            labels = [(captions(q.group(1), 1) if q else captions(text, 1)) or ["WHY?"]][0]
            labels = labels if isinstance(labels, list) else [labels]
            motifs = ["question", "person"]
            anchors["motif0"] = _tokens(text)[0]
            title = ""
        else:  # hero, network, timeline, evidence, funnel
            labels = captions(text, 3)
            for j, m in enumerate(motifs[:4]):
                w = _matched_word(text, m)
                if w:
                    anchors[f"motif{j}"] = w
            for j, lab in enumerate(labels):
                anchors[f"label{j}"] = lab.split()[0]
            if layout in ("hero", "network", "evidence", "timeline", "funnel"):
                title = headline(text) if layout == "hero" else ""
            if layout == "hero" and title:
                anchors["title"] = title.split()[0]
                labels = [l for l in labels if l != title][:2]
                if not labels:
                    labels = captions(text, 2, avoid={title})
            motifs = motifs[:4]

        ch["title"] = title
        ch["motifs"] = [m for m in motifs if m in LIBRARY]
        ch["labels"] = [l for l in labels if l]
        ch["anchors"] = {k: v for k, v in anchors.items() if v}
        ch["mirror"] = bool((i // 2) % 2)
        ch["camera"] = ("closeup" if layout in ("hero", "question", "iceberg", "evidence") and i % 3 == 1 else
                        "hold" if layout in ("statement", "bignumber") else "push")
        prev_erase = i > 0 and rows[i - 1].get("chalk", {}).get("transition") == "erase"
        wants_erase = layout == "statement" or (i > 0 and i % 4 == 3)
        ch["transition"] = "erase" if (wants_erase and not prev_erase and i > 0) else "pan"
        ch["detail"] = 3
        for m in ch["motifs"]:
            used[m] = used.get(m, 0) + 1
        recent_layouts.append(layout)
        row["chalk"] = ch


# ---------------------------------------------------------------- AI validation
def sanitize_chalk(raw: Any, text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Validate an AI-provided ``chalk`` object; anything unusable falls back to the local plan."""
    if not isinstance(raw, dict):
        return fallback
    out = dict(fallback)
    lay = str(raw.get("layout", "")).strip().lower()
    if lay in LAYOUTS:
        out["layout"] = lay
    motifs = [str(m).strip().lower() for m in (raw.get("motifs") or []) if str(m).strip().lower() in LIBRARY]
    if motifs:
        out["motifs"] = motifs[:4]
    labels = []
    for lab in raw.get("labels") or []:
        lab = _clean(str(lab)).upper()
        if lab and len(lab.split()) <= 4 and len(lab) <= 26 and re.search(r"[A-Z0-9]", lab):
            labels.append(lab)
    if labels:
        out["labels"] = labels[:4]
    title = _clean(str(raw.get("title", ""))).upper()
    if title and len(title.split()) <= 5 and len(title) <= 34:
        out["title"] = title
    anchors = {}
    low = text.casefold()
    for k, v in (raw.get("anchors") or {}).items():
        v = _clean(str(v))
        if v and v.casefold() in low and re.fullmatch(r"(motif[0-3]|label[0-3]|title|number|emph|arrow)", str(k)):
            anchors[str(k)] = v
    if anchors:
        out["anchors"] = {**fallback.get("anchors", {}), **anchors}
    num = _clean(str(raw.get("number", ""))).upper()
    if num and len(num) <= 14:
        out["number"] = num
    if raw.get("camera") in ("push", "closeup", "hold"):
        out["camera"] = raw["camera"]
    if raw.get("transition") in ("pan", "erase"):
        out["transition"] = raw["transition"]
    return out
