"""Validation and deterministic timing for documentary visual beat plans."""

import json
import os
import re
from difflib import SequenceMatcher


_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = set("""
about after again also because before being between could does during from have into just
more most other over same should some such than that their there these they this those through
under very what when where which while with would your and are but for not the then them was
were will you has had its our out who why how his her she him all any can did get got one two
too use used using via per con che del della delle degli dai dalla dalle dello dopo era essere
gli hanno come nel nella nelle nello non perche perché sono sua sue sul sulla sulle tra una uno
alla alle allo anche dove questa questo quella quello loro noi voi sia puo può piu più
""".split())
_FUNCTIONS = {
    "hook", "orientation", "evidence", "escalation", "contrast", "reveal",
    "consequence", "resolution", "context", "explanation",
}
_VISUAL_TYPES = {"video", "archival_image", "illustration", "map", "chart", "graphic", "still"}
_TRANSITIONS = {"cut", "dissolve", "match_cut", "audio_led"}
_SRT_TIME_RE = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$"
)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text or "")]


def parse_srt_cues(text: str) -> list[dict]:
    """Parse local SRT timing, ignoring malformed blocks safely."""
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        match = _SRT_TIME_RE.match(lines[1].strip())
        if not match:
            continue
        values = [int(value) for value in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        caption = " ".join(lines[2:]).strip()
        if caption and end > start:
            cues.append({"start": start, "end": end, "text": caption})
    return cues


def _sentences(script: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", script or "").strip()
    return [part.strip() for part in _SENTENCE_RE.split(cleaned) if part.strip()]


def _query_for(text: str) -> str:
    words = [word for word in _tokens(text) if len(word) > 2 and word not in _STOPWORDS]
    return " ".join(list(dict.fromkeys(words))[:5]) or "documentary subject detail"


def deterministic_visual_plan(script: str, words_per_beat: int = 26) -> list[dict]:
    """Build compact beats from script sentences without network or model calls."""
    groups: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _sentences(script):
        size = max(1, len(_tokens(sentence)))
        if current and count >= words_per_beat:
            groups.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += size
    if current:
        groups.append(" ".join(current)
        )

    beats = []
    for index, text in enumerate(groups):
        lowered = text.casefold()
        if index == 0:
            function = "hook"
        elif index == len(groups) - 1:
            function = "resolution"
        elif any(word in lowered for word in ("but", "however", "instead", "yet", "although")):
            function = "contrast"
        elif any(char.isdigit() for char in text) or any(word in lowered for word in ("evidence", "records", "study", "report")):
            function = "evidence"
        else:
            function = "escalation"
        beats.append({
            "id": f"beat-{index + 1:03d}",
            "narration_anchor": text,
            "dramatic_function": function,
            "visual_intent": f"Show concrete visual evidence for: {text[:140]}",
            "primary_query": _query_for(text),
            "alternate_queries": [],
            "visual_type": "video",
            "shot_type": "establishing wide" if index == 0 else "documentary detail",
            "composition": "Keep the subject clear; reserve safe negative space only for overlays.",
            "motion": ("slow_push" if index % 3 == 0 else
                       "lateral_pan" if index % 3 == 1 else "static_hold"),
            "duration_hint": 5.0,
            "transition": "cut",
            "overlay_text": "",
            "sound_cue": "",
            "music_intensity": 0.25 if index == 0 else 0.4,
            "hold_for_reveal": False,
        })
    return beats


def validate_visual_plan(script: str, candidate: object) -> list[dict]:
    """Normalize model output; use a deterministic script-derived fallback."""
    if not isinstance(candidate, list):
        return deterministic_visual_plan(script)

    beats = []
    for index, raw in enumerate(candidate):
        if not isinstance(raw, dict):
            continue
        anchor = str(raw.get("narration_anchor") or "").strip()
        query = " ".join(str(raw.get("primary_query") or "").split())
        if not anchor or not query:
            continue
        alternates = raw.get("alternate_queries")
        if not isinstance(alternates, list):
            alternates = []
        try:
            duration = float(raw.get("duration_hint", 5.0))
        except (TypeError, ValueError):
            duration = 5.0
        try:
            intensity = float(raw.get("music_intensity", 0.4))
        except (TypeError, ValueError):
            intensity = 0.4
        function = str(raw.get("dramatic_function") or "explanation").strip().casefold()
        visual_type = str(raw.get("visual_type") or "illustration").strip().casefold()
        transition = str(raw.get("transition") or "cut").strip().casefold()
        beats.append({
            "id": str(raw.get("id") or f"beat-{index + 1:03d}").strip(),
            "narration_anchor": anchor,
            "dramatic_function": function if function in _FUNCTIONS else "explanation",
            "visual_intent": str(raw.get("visual_intent") or anchor).strip(),
            "primary_query": query,
            "alternate_queries": list(dict.fromkeys(
                " ".join(str(query).split()) for query in alternates if str(query).strip()
            ))[:2],
            "visual_type": visual_type if visual_type in _VISUAL_TYPES else "illustration",
            "shot_type": str(raw.get("shot_type") or "documentary detail").strip(),
            "composition": str(raw.get("composition") or "Clear documentary framing.").strip(),
            "motion": str(raw.get("motion") or "static_hold").strip(),
            "duration_hint": min(12.0, max(2.0, duration)),
            "transition": transition if transition in _TRANSITIONS else "cut",
            "overlay_text": str(raw.get("overlay_text") or "").strip()[:80],
            "sound_cue": raw.get("sound_cue") if isinstance(raw.get("sound_cue"), (str, dict)) else "",
            "music_intensity": min(1.0, max(0.0, intensity)),
            "hold_for_reveal": bool(raw.get("hold_for_reveal", False)),
        })

    script_tokens = _tokens(script)
    matched = 0
    cursor = 0
    for beat in beats:
        start = _find_anchor_start(script_tokens, _tokens(beat["narration_anchor"]), cursor)
        if start is not None:
            matched += 1
            cursor = start + len(_tokens(beat["narration_anchor"]))

    if (not beats or len(beats) > max(1, len(script_tokens))
            or matched < max(1, (len(beats) + 1) // 2)):
        return deterministic_visual_plan(script)
    return beats


def save_visual_plan(script: str, candidate: object, path: str = "output/visual_plan.json") -> list[dict]:
    """Validate and atomically save the plan used by rendering and preview."""
    beats = validate_visual_plan(script, candidate)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(beats, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
    return beats


def _find_anchor_start(script_tokens: list[str], anchor_tokens: list[str], cursor: int) -> int | None:
    if not anchor_tokens:
        return None
    width = len(anchor_tokens)
    for start in range(cursor, len(script_tokens) - width + 1):
        if script_tokens[start:start + width] == anchor_tokens:
            return start
    return None


def _fallback_anchor_start(script: str, anchor_tokens: list[str], cursor: int) -> int | None:
    if not anchor_tokens:
        return None
    best: tuple[float, int] | None = None
    offset = 0
    for sentence in _sentences(script):
        sentence_tokens = _tokens(sentence)
        start = offset
        offset += len(sentence_tokens)
        if offset <= cursor:
            continue
        candidate = sentence_tokens[max(0, cursor - start):]
        score = SequenceMatcher(None, anchor_tokens, candidate, autojunk=False).ratio()
        if best is None or score > best[0]:
            best = (score, max(cursor, start))
    return best[1] if best and best[0] >= 0.28 else None


def _subtitle_anchor_times(beats: list[dict], subtitle_cues: list[dict]) -> list[float | None]:
    flattened = []
    for cue in subtitle_cues:
        if not isinstance(cue, dict):
            continue
        words = _tokens(str(cue.get("text") or ""))
        if not words:
            continue
        try:
            start, end = float(cue["start"]), float(cue["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        step = (end - start) / len(words)
        flattened.extend((word, start + i * step, start + (i + 1) * step)
                         for i, word in enumerate(words))

    result = []
    cursor = 0
    for beat in beats:
        anchor = _tokens(beat["narration_anchor"])
        found = None
        for start in range(cursor, len(flattened) - len(anchor) + 1):
            if [item[0] for item in flattened[start:start + len(anchor)]] == anchor:
                found = start
                break
        if found is None:
            result.append(None)
        else:
            result.append(flattened[found][1])
            cursor = found + len(anchor)
    return result


def build_shot_timeline(
    script: str,
    candidate: object,
    total_duration: float,
    max_duration: float | None = None,
    subtitle_cues: list[dict] | None = None,
) -> list[dict]:
    """Align ordered beat anchors to narration words and split into shot events."""
    duration = max(0.0, float(total_duration))
    if max_duration is not None:
        duration = min(duration, max(0.0, float(max_duration)))
    tokens = _tokens(script)
    if duration <= 0 or not tokens:
        return []

    beats = validate_visual_plan(script, candidate)
    if not beats:
        return []

    starts = []
    cursor = 0
    for index, beat in enumerate(beats):
        anchor_tokens = _tokens(beat["narration_anchor"])
        start = _find_anchor_start(tokens, anchor_tokens, cursor)
        if start is None:
            start = _fallback_anchor_start(script, anchor_tokens, cursor)
        if start is None:
            start = round(index * len(tokens) / len(beats))
        start = min(max(cursor, start), max(0, len(tokens) - (len(beats) - index)))
        starts.append(start)
        cursor = start + max(1, len(anchor_tokens))

    cue_times = _subtitle_anchor_times(beats, subtitle_cues or [])
    valid_cue_times = [time for time in cue_times if time is not None]
    use_cues = len(valid_cue_times) >= max(1, (len(beats) + 1) // 2)
    events = []
    for index, beat in enumerate(beats):
        start_time = 0.0 if index == 0 else duration * starts[index] / len(tokens)
        end_time = duration if index + 1 == len(starts) else duration * starts[index + 1] / len(tokens)
        if use_cues:
            if index > 0 and cue_times[index] is not None:
                start_time = cue_times[index]
            if index + 1 < len(cue_times) and cue_times[index + 1] is not None:
                end_time = cue_times[index + 1]
        start_time = max(0.0, min(duration, start_time))
        end_time = max(start_time, min(duration, end_time))
        if index == len(beats) - 1:
            end_time = duration
        beat_duration = max(0.0, end_time - start_time)
        if beat_duration <= 0:
            continue

        try:
            target = min(9.0, max(2.5, float(beat.get("duration_hint", 4.8))))
        except (TypeError, ValueError):
            target = 4.8
        if beat.get("hold_for_reveal") or beat["dramatic_function"] == "reveal":
            target = max(target, 6.0)
        elif beat["dramatic_function"] in {"hook", "evidence", "contrast"}:
            target = min(target, 4.5)
        shot_count = min(3, max(1, round(beat_duration / target + 0.35)))
        shot_duration = beat_duration / shot_count
        for shot_index in range(shot_count):
            shot_start = start_time + shot_index * shot_duration
            shot_end = end_time if shot_index == shot_count - 1 else shot_start + shot_duration
            events.append({
                "start": shot_start,
                "end": shot_end,
                "duration": shot_end - shot_start,
                "beat": beat,
                "shot_index": shot_index,
                "shot_count": shot_count,
            })
    return events