"""Structured scene planning for FERN's optional chalkboard renderer.

Passes 1-3: auditable inputs, explicit speech-timing provenance, and story-led
visual plans.  Uses only the standard library by default.  Optional local
word timestamps use an already-installed/cached recognizer; optional OpenRouter
planning is opt-in and never silently enabled.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPOSITIONS = {
    "causal_chain", "timeline", "comparison", "numbers", "system_map",
    "evidence", "spotlight", "feedback_loop", "decision_tree", "object_focus",
}
DRAW_ACTION_TYPES = {"line", "arrow", "box", "circle", "motif", "text", "underline", "scribble"}
DRAW_COLORS = {"chalk", "dim", "teal", "yellow"}
MOTIFS = {
    "coin", "building", "lock", "eye", "chip", "document", "chart",
    "magnifier", "warning", "network", "clock", "box", "lightbulb", "broken_link", "none",
}
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
    "very", "really", "quietly", "suddenly", "actually", "still", "now",
}


def _normal_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", value.casefold().replace("’", "'"))


def _normal_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def available_alignment_backends() -> dict[str, Any]:
    """Report locally installed timestamp engines without importing/downloading them."""
    packages = {}
    for name in ("faster_whisper", "whisper", "whisperx", "stable_whisper", "whisper_timestamped", "aeneas"):
        try:
            packages[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            packages[name] = False

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    faster_models = {}
    for model in ("tiny", "base", "small", "medium"):
        root = cache_root / f"models--Systran--faster-whisper-{model}" / "snapshots"
        paths = [p for p in root.glob("*") if p.is_dir()] if root.is_dir() else []
        faster_models[model] = str(paths[-1]) if paths else None
    whisper_cache = Path(os.environ.get("WHISPER_CACHE", Path.home() / ".cache" / "whisper"))
    whisper_models = {m: str(whisper_cache / f"{m}.pt") if (whisper_cache / f"{m}.pt").is_file() else None
                      for m in ("tiny", "base", "small", "medium")}
    return {
        "packages": packages,
        "cached_faster_whisper_models": faster_models,
        "cached_openai_whisper_models": whisper_models,
        "recommended_available": bool(
            (packages.get("faster_whisper") and any(faster_models.values()))
            or (packages.get("whisper") and any(whisper_models.values()))
        ),
    }


def _local_env() -> None:
    """Read .env without printing values; keep existing shell variables authoritative."""
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_file, override=False)
        return
    except Exception:
        pass
    try:
        for raw in env_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def _segment_script(script: str, target_words: int) -> list[str]:
    """Make contiguous spoken beats while preserving every script token in order."""
    cleaned = re.sub(r"\s+", " ", (script or "")).strip()
    if not cleaned:
        return []
    words = cleaned.split()
    target_words = max(7, min(18, int(target_words)))
    max_words = min(22, target_words + 5)
    min_words = max(5, int(target_words * 0.58))
    beats: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            beats.append(" ".join(current).strip())
            current = []

    for i, token in enumerate(words):
        current.append(token)
        count = len(current)
        ends_sentence = bool(re.search(r"[.!?][\"')\]]*$", token))
        ends_clause = bool(re.search(r"[,;:][\"')\]]*$", token))
        next_word = words[i + 1].casefold().strip("\"'()[]{}.,;:!?—-") if i + 1 < len(words) else ""
        natural_conjunction = next_word in {"but", "because", "however", "therefore", "while", "whereas", "which", "then", "yet"}
        if count >= max_words:
            flush()
        elif count >= min_words and (ends_sentence or (ends_clause and count >= target_words - 2) or (natural_conjunction and count >= target_words)):
            flush()
        elif count >= target_words + 2 and (ends_sentence or ends_clause):
            flush()
    flush()

    # Merge a tiny trailing fragment to its predecessor unless that would create
    # an unreadably long board. No duplication or fabricated filler is allowed.
    if len(beats) > 1 and len(beats[-1].split()) < 4 and len(beats[-2].split()) + len(beats[-1].split()) <= max_words + 2:
        beats[-2] += " " + beats[-1]
        beats.pop()
    return beats


def _candidate_labels(text: str, max_labels: int = 3) -> list[str]:
    """Extract concise, unique labels from a spoken beat; never invent full-sentence captions."""
    # First preserve semantic clause fragments; then shorten to high-information words.
    fragments = [x.strip(" ,;:.!?—-\"'") for x in re.split(
        r"[,;:.!?]+|\b(?:because|therefore|however|while|whereas|which means|as a result|then|but|and so|so that)\b",
        text, flags=re.I,
    ) if x.strip(" ,;:.!?—-\"'")]
    candidates: list[str] = []
    seen: set[str] = set()

    def add(fragment: str) -> None:
        tokens = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?%?", fragment)
        # Prefer content words while preserving number/named-entity terms.
        content = [t for t in tokens if _normal_word(t) not in STOPWORDS]
        if len(content) < 2:
            content = tokens
        if len(content) > 4:
            # Prefer the phrase's tail, where the semantic noun often appears.
            content = content[-4:]
        label = " ".join(content).strip(" ,;:.!?—-\"'")
        normalized = _normal_phrase(label)
        label_tokens = set(normalized.split())
        if not label or normalized in seen or len(label) > 30:
            return
        # Don't pad layouts with a shorter phrase that simply repeats most of
        # another label (e.g. "company made bold" after "company made bold promise").
        for old_label in candidates:
            old_tokens = set(_normal_phrase(old_label).split())
            overlap = len(label_tokens & old_tokens)
            if label_tokens and old_tokens and overlap / min(len(label_tokens), len(old_tokens)) >= 0.5:
                return
        seen.add(normalized)
        candidates.append(label[:30])

    for fragment in fragments:
        add(fragment)
        if len(candidates) >= max_labels:
            return candidates[:max_labels]

    # If clauses yielded fewer labels, create non-overlapping, balanced chunks of
    # content words. This is a compact plan label, not copied narration.
    content = [w for w in re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?%?", text)
               if _normal_word(w) not in STOPWORDS]
    if len(content) < 2:
        content = re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?%?", text)
    for width in (4, 3, 2):
        if len(candidates) >= max_labels:
            break
        step = max(2, width)
        for start in range(0, max(1, len(content) - 1), step):
            add(" ".join(content[start:start + width]))
            if len(candidates) >= max_labels:
                break
    return candidates[:max_labels]


def _local_composition(text: str, index: int, previous: str | None) -> str:
    s = text.casefold()
    if re.search(r"\b(vs\.?|versus|compared|comparison|whereas|unlike|on the other hand|instead of|rather than|difference between|both .* and|however|but the|but real|but actually)\b", s):
        choice = "comparison"
    elif re.search(r"\b(first|next|before|after|finally|year|decade|century|timeline|over time|in 19\d\d|in 20\d\d)\b", s):
        choice = "timeline"
    elif re.search(r"\b(\d+(?:[.,]\d+)?\s*%|percent|percentage|million|billion|trillion|doubled|tripled|grew by|fell by|increased by|decreased by|\$\s?\d|\d{4})\b", s):
        choice = "numbers"
    elif re.search(r"\b(trap|reveal|hidden truth|secret|exposed|turning point|the moment|what they hid|underneath the surface)\b", s):
        choice = "spotlight"
    elif re.search(r"\b(because|caused? by|leads? to|led to|results? in|therefore|as a result|consequently|which meant|this meant|so that|changed how|made .* possible|turned into)\b", s):
        choice = "causal_chain"
    elif re.search(r"\b(if .* then|choice|choose|decision|option|either|risk|trade.?off)\b", s):
        choice = "decision_tree"
    elif re.search(r"\b(feedback loop|cycle|repeats? itself|keeps feeding|self.?reinforcing|spiral)\b", s):
        choice = "feedback_loop"
    elif re.search(r"\b(memo|document|contract|record|report|email|letter|evidence|receipt|filing|paper trail|internal note)\b", s):
        choice = "evidence"
    elif re.search(r"\b(system|network|connected|interconnected|machine|incentives|pipeline|mechanism|platform|algorithm|score)\b", s):
        choice = "system_map"
    else:
        # Use story vocabulary and ordinal cycling to avoid a wall of identical cards.
        fallbacks = ["object_focus", "causal_chain", "evidence", "spotlight", "system_map", "comparison"]
        choice = fallbacks[index % len(fallbacks)]
    if choice == previous and choice not in {"timeline", "comparison", "numbers", "causal_chain"}:
        alternates = [x for x in ("object_focus", "evidence", "spotlight", "system_map", "feedback_loop") if x != previous]
        choice = alternates[index % len(alternates)]
    return choice


def _local_motif(text: str, composition: str) -> str:
    s = text.casefold()
    if re.search(r"\b(price|money|cost|revenue|profit|debt|payment|dollar|budget|financial|bill|economy|cash)\b", s):
        return "coin"
    if re.search(r"\b(artificial intelligence|\bai\b|algorithm|software|data|digital|computer|app|code|server|technology|tech)\b", s):
        return "chip"
    if re.search(r"\b(company|corporation|business|office|firm|headquarters|organization|factory)\b", s):
        return "building"
    if re.search(r"\b(secret|hidden|watch|monitor|judg|score|surveillance|private|quietly|invisible|tracking)\b", s):
        return "eye"
    if re.search(r"\b(memo|document|contract|record|report|email|letter|evidence|receipt|filing|proof)\b", s):
        return "document"
    if re.search(r"\b(risk|danger|warn|failure|collapse|crash|scandal|fraud|breach|threat)\b", s):
        return "warning"
    if re.search(r"\b(year|month|day|before|after|first|later|decade|century|time|clock)\b", s):
        return "clock"
    if composition in {"numbers", "timeline"}:
        return "chart"
    if composition in {"system_map", "feedback_loop"}:
        return "network"
    if composition == "evidence":
        return "magnifier"
    return "none"


def _local_title(text: str, labels: list[str], index: int, composition: str | None = None) -> str:
    # Short diagram-type headers, not an echo of the spoken sentence.  Variety
    # is driven by the scene planner, not a repeated chapter title.
    headers = {
        "causal_chain": "CAUSE → EFFECT", "timeline": "TIME & CONSEQUENCE",
        "comparison": "TWO SIDES", "numbers": "THE NUMBER",
        "system_map": "INSIDE THE SYSTEM", "evidence": "FOLLOW THE EVIDENCE",
        "spotlight": "THE REVEAL", "feedback_loop": "THE CYCLE",
        "decision_tree": "THE CHOICE", "object_focus": "THE SIGNAL",
    }
    return headers.get(composition or "", f"SCENE {index + 1}")


def _map_word_timestamps(script: str, recognized_words: list[dict[str, Any]]) -> tuple[list[tuple[float, float] | None], float]:
    script_words = script.split()
    norm_script = [_normal_word(x) for x in script_words]
    norm_rec = [_normal_word(str(x.get("word", ""))) for x in recognized_words]
    sm = difflib.SequenceMatcher(a=norm_script, b=norm_rec, autojunk=False)
    timing: list[tuple[float, float] | None] = [None] * len(script_words)
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for si, rj in zip(range(i1, i2), range(j1, j2)):
                try:
                    st = float(recognized_words[rj]["start"])
                    en = float(recognized_words[rj]["end"])
                    if en > st:
                        timing[si] = (st, en)
                        matched += 1
                except (KeyError, TypeError, ValueError):
                    continue
    # Fill a few omitted/unrecognized function words by interpolation between
    # trusted neighbors, but reject alignments with too little lexical coverage.
    coverage = matched / max(1, len(script_words))
    if coverage < 0.68:
        return timing, coverage
    known = [i for i, item in enumerate(timing) if item is not None]
    if not known:
        return timing, 0.0
    for i in range(len(timing)):
        if timing[i] is not None:
            continue
        left = max((j for j in known if j < i), default=None)
        right = min((j for j in known if j > i), default=None)
        if left is not None and right is not None:
            lo = timing[left][1]  # type: ignore[index]
            hi = timing[right][0]  # type: ignore[index]
            span = max(0.02, hi - lo)
            timing[i] = (lo + span * 0.33, lo + span * 0.67)
        elif left is not None:
            lo = timing[left][1]  # type: ignore[index]
            timing[i] = (lo + 0.015, lo + 0.12)
        elif right is not None:
            hi = timing[right][0]  # type: ignore[index]
            timing[i] = (max(0.0, hi - 0.12), max(0.015, hi - 0.015))
    return timing, coverage


def align_audio_words(audio_path: str, script: str, *, model: str = "base") -> dict[str, Any]:
    """Use an already-installed recognizer/model cache for word timestamps.

    This intentionally refuses to download models or packages. Run audit first;
    then pass --align-audio only when the selected local model is already cached.
    """
    audio = Path(audio_path).resolve()
    if not audio.is_file():
        return {"ok": False, "reason": f"Audio not found: {audio}"}
    inventory = available_alignment_backends()
    requested_model = model.strip().lower() or "base"
    recognized: list[dict[str, Any]] = []
    backend = None

    if inventory["packages"].get("faster_whisper") and inventory["cached_faster_whisper_models"].get(requested_model):
        try:
            from faster_whisper import WhisperModel  # type: ignore
            model_path = inventory["cached_faster_whisper_models"][requested_model]
            attempts = [("cuda", "float16"), ("cpu", "int8")]
            error = None
            engine = None
            for device, compute_type in attempts:
                try:
                    engine = WhisperModel(model_path, device=device, compute_type=compute_type)
                    break
                except Exception as exc:  # fall back to CPU only if local CUDA load fails
                    error = exc
            if engine is None:
                raise RuntimeError(str(error or "could not load local faster-whisper model"))
            segments, _info = engine.transcribe(str(audio), word_timestamps=True, vad_filter=True)
            for seg in segments:
                for w in (seg.words or []):
                    recognized.append({"word": w.word, "start": float(w.start), "end": float(w.end)})
            backend = f"faster-whisper:{requested_model}"
        except Exception as exc:
            return {"ok": False, "reason": f"Cached faster-whisper could not run: {exc}"}
    elif inventory["packages"].get("whisper") and inventory["cached_openai_whisper_models"].get(requested_model):
        try:
            import whisper  # type: ignore
            model_path = inventory["cached_openai_whisper_models"][requested_model]
            engine = whisper.load_model(model_path, device="cpu")
            result = engine.transcribe(str(audio), word_timestamps=True, fp16=False, verbose=False)
            for seg in result.get("segments", []):
                for w in (seg.get("words") or []):
                    recognized.append({"word": w.get("word", ""), "start": float(w["start"]), "end": float(w["end"])})
            backend = f"openai-whisper:{requested_model}"
        except Exception as exc:
            return {"ok": False, "reason": f"Cached Whisper could not run: {exc}"}
    else:
        return {
            "ok": False,
            "reason": "No supported ASR package with a cached local model was found. No download was attempted.",
            "inventory": inventory,
        }

    mapped, coverage = _map_word_timestamps(script, recognized)
    if coverage < 0.68:
        return {"ok": False, "reason": f"Transcript only matched {coverage:.0%} of script words; fallback to approximate timing.",
                "backend": backend, "coverage": round(coverage, 3), "recognized_word_count": len(recognized)}
    return {"ok": True, "backend": backend, "coverage": round(coverage, 3),
            "word_timings": mapped, "recognized_word_count": len(recognized),
            "timing_source": "local_asr_word_timestamps"}


def _detect_audio_silences(audio_path: str | None, duration: float) -> list[tuple[float, float]]:
    """Use local FFmpeg silencedetect to find short natural gaps; no ASR/download."""
    if not audio_path or not Path(audio_path).is_file():
        return []
    candidates = [ROOT / "ffmpeg.exe", ROOT / "ffmpeg"]
    path_on_path = shutil.which("ffmpeg")
    if path_on_path:
        candidates.append(Path(path_on_path))
    ffmpeg = next((str(x.resolve()) for x in candidates if x.is_file()), None)
    if not ffmpeg:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostats", "-i", str(Path(audio_path).resolve()),
             "-af", "silencedetect=noise=-32dB:d=0.12", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=max(30, min(300, int(duration * .5) + 30)),
            check=False,
        )
        log = (proc.stderr or "") + "\n" + (proc.stdout or "")
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log)]
        spans: list[tuple[float, float]] = []
        end_i = 0
        for start in starts:
            while end_i < len(ends) and ends[end_i] < start:
                end_i += 1
            if end_i < len(ends):
                end = ends[end_i]
                if end - start >= .12 and 0 <= start < end <= duration + .1:
                    spans.append((max(0.0, start), min(duration, end)))
                end_i += 1
        return spans
    except Exception:
        # This is a best-effort timing aid; the explicit fallback estimate remains valid.
        return []


def _nudge_boundaries_to_pauses(times: list[dict[str, Any]], audio_path: str | None,
                                duration: float) -> tuple[list[dict[str, Any]], int]:
    """Shift approximate scene boundaries to nearby measured silence intervals."""
    if len(times) < 2:
        return times, 0
    silences = _detect_audio_silences(audio_path, duration)
    if not silences:
        return times, 0
    boundaries = [float(times[0]["start_seconds"])] + [float(x["end_seconds"]) for x in times]
    adjustments = 0
    used: set[int] = set()
    for i in range(1, len(boundaries) - 1):
        target = boundaries[i]
        options = sorted(
            ((abs((start + end) / 2 - target), j, (start + end) / 2)
             for j, (start, end) in enumerate(silences) if j not in used),
            key=lambda x: x[0],
        )
        if not options:
            continue
        distance, j, midpoint = options[0]
        if distance > .85:
            continue
        # Don't make either visual beat too brief just to match a tiny pause.
        if midpoint - boundaries[i-1] < 1.35 or boundaries[i+1] - midpoint < 1.35:
            continue
        boundaries[i] = midpoint
        used.add(j)
        adjustments += 1
    if adjustments:
        for i, row in enumerate(times):
            row["start_seconds"] = round(boundaries[i], 3)
            row["end_seconds"] = round(boundaries[i + 1], 3)
            row["pause_adjusted"] = True
    return times, adjustments


def _timing_for_beats(beats: list[str], duration: float, alignment: dict[str, Any] | None) -> list[dict[str, Any]]:
    counts = [len(x.split()) for x in beats]
    total_words = max(1, sum(counts))
    if alignment and alignment.get("ok") and alignment.get("word_timings"):
        word_times = alignment["word_timings"]
        spans: list[tuple[float, float]] = []
        cursor = 0
        for beat in beats:
            n = len(beat.split())
            chunk = [x for x in word_times[cursor:cursor + n] if x is not None]
            cursor += n
            if chunk:
                spans.append((max(0.0, float(chunk[0][0])), min(duration, float(chunk[-1][1]))))
            else:
                spans.append((0.0, 0.0))
        # Use midpoints between adjacent spoken beats as scene boundaries. This
        # keeps scenes contiguous while preserving the true spoken-word timing.
        boundaries = [0.0]
        for i in range(1, len(spans)):
            left_end = spans[i - 1][1]
            right_start = spans[i][0]
            if right_start >= left_end:
                boundaries.append((left_end + right_start) / 2.0)
            else:
                boundaries.append(max(boundaries[-1] + 0.25, right_start))
        boundaries.append(duration)
        boundaries = [max(0.0, min(duration, x)) for x in boundaries]
        for i in range(1, len(boundaries)):
            boundaries[i] = max(boundaries[i], boundaries[i - 1] + 0.20)
        boundaries[-1] = duration
        result = []
        start_word = 0
        for i, beat in enumerate(beats):
            wc = len(beat.split())
            result.append({
                "start_seconds": round(boundaries[i], 3),
                "end_seconds": round(max(boundaries[i] + 0.2, boundaries[i + 1]), 3),
                "word_start_index": start_word,
                "word_end_index": start_word + wc,
                "timing_source": str(alignment.get("row_timing_source") or "local_asr_word_timestamps"),
                "timing_confidence": round(float(alignment.get("coverage", 0.85)), 2),
            })
            start_word += wc
        # Guarantee no end exceeds media duration and no final gap.
        result[-1]["end_seconds"] = round(duration, 3)
        for i in range(1, len(result)):
            result[i]["start_seconds"] = result[i - 1]["end_seconds"]
        return result

    # Explicitly approximate fallback. Word length slightly reflects syllable
    # density; punctuation adds a small pause weight. It is not advertised as
    # real word-level synchronization.
    weights = []
    for beat in beats:
        words = beat.split()
        syllables = 0
        for word in words:
            alpha = re.sub(r"[^a-z]", "", word.casefold())
            groups = re.findall(r"[aeiouy]+", alpha)
            syllables += max(1, len(groups) - int(alpha.endswith("e") and len(groups) > 1))
        pause = 0.18 * sum(1 for c in beat if c in ".!?;:") + 0.07 * sum(1 for c in beat if c == ",")
        weights.append(max(1.0, len(words) + 0.45 * syllables + pause))
    sum_weights = sum(weights) or 1.0
    result = []
    cursor = 0.0
    start_word = 0
    for i, (beat, weight) in enumerate(zip(beats, weights)):
        wc = len(beat.split())
        end = duration if i == len(beats) - 1 else duration * sum(weights[:i + 1]) / sum_weights
        result.append({
            "start_seconds": round(cursor, 3),
            "end_seconds": round(end, 3),
            "word_start_index": start_word,
            "word_end_index": start_word + wc,
            "timing_source": "syllable_weighted_duration_estimate",
            "timing_confidence": 0.42,
        })
        cursor = end
        start_word += wc
    return result


def _build_local_rows(beats: list[str], times: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_composition = None
    for i, (beat, timing) in enumerate(zip(beats, times)):
        composition = _local_composition(beat, i, previous_composition)
        candidates = _candidate_labels(beat, 3)
        if not candidates:
            raise ValueError(f"Scene {i + 1} contains no useful label candidates")
        composition = _resolve_composition_for_labels(composition, candidates)
        motif = _local_motif(beat, composition)
        # Category header is intentionally distinct from short evidence labels.
        title = _local_title(beat, candidates, i, composition)
        labels = candidates[:3]
        rows.append({
            "scene": i + 1,
            **timing,
            "word_count": len(beat.split()),
            "visual_kind": composition,
            "composition": composition,
            "narrative_purpose": _purpose(composition),
            "visual_metaphor": _metaphor(beat, composition, motif),
            "motif": motif,
            "title": title,
            "labels": labels,
            "phrases": labels,
            "drawing_sequence": _drawing_sequence(composition, labels, motif),
            "drawing_actions": _local_drawing_actions(composition, labels, motif, beat),
            "narration_excerpt": beat,
        })
        previous_composition = composition
    return rows


def _resolve_composition_for_labels(composition: str, labels: list[str]) -> str:
    """Do not claim to direct a multi-node diagram when the beat has no labels for it."""
    n = len(labels)
    if composition in {"timeline", "comparison", "causal_chain"} and n < 2:
        return "spotlight" if composition in {"timeline", "comparison"} else "object_focus"
    if composition == "decision_tree" and n < 3:
        return "object_focus"
    if composition == "feedback_loop" and n < 2:
        return "spotlight"
    return composition


def _purpose(composition: str) -> str:
    return {
        "causal_chain": "Make the cause-and-effect relationship visible",
        "timeline": "Clarify the order of events over time",
        "comparison": "Contrast two sides and reveal the consequence",
        "numbers": "Make the quantitative evidence visually legible",
        "system_map": "Show how the parts of a system connect",
        "evidence": "Treat a record or clue as evidence and show what it implies",
        "spotlight": "Pause on the key idea and give it visual weight",
        "feedback_loop": "Show a repeating or self-reinforcing cycle",
        "decision_tree": "Make a choice and its possible outcomes visible",
        "object_focus": "Use one concrete visual metaphor to anchor the idea",
    }.get(composition, "Clarify the current idea")


def _metaphor(text: str, composition: str, motif: str) -> str:
    short = " ".join(text.split()[:8]).rstrip(" ,;:.!?—-")
    descriptions = {
        "causal_chain": "Three connected stages; each arrow is a consequence",
        "timeline": "A drawn chronology with distinct moments and labeled evidence",
        "comparison": "A split board reveals a contrast, then the takeaway",
        "numbers": "A large numerical anchor, scale marks, and a contextual note",
        "system_map": "One focal symbol connected to three smaller system nodes",
        "evidence": "A marked-up record leads to a visible implication",
        "spotlight": "One prominent claim surrounded by a deliberate chalk emphasis",
        "feedback_loop": "Three stages loop back into the source of the cycle",
        "decision_tree": "One central decision branches into two outcomes",
        "object_focus": "A large symbolic drawing with two concise annotations",
    }.get(composition, "A focused chalkboard explanation")
    return f"{descriptions}; motif={motif}; narration context: {short}"[:360]


def _drawing_sequence(composition: str, labels: list[str], motif: str) -> list[dict[str, Any]]:
    actions = [
        {"at_fraction": 0.06, "action": "draw_title_anchor", "target": labels[0] if labels else "key idea"},
        {"at_fraction": 0.20, "action": "draw_focal_motif", "target": motif},
    ]
    action_map = {
        "causal_chain": ["draw_stage_one", "draw_arrow_one", "draw_stage_two", "draw_arrow_two", "draw_stage_three"],
        "timeline": ["draw_axis", "mark_first_event", "mark_next_event", "mark_final_event"],
        "comparison": ["draw_left_case", "draw_divider", "draw_right_case", "underline_takeaway"],
        "numbers": ["draw_scale", "write_main_number", "draw_context_marker"],
        "system_map": ["draw_central_node", "connect_first_node", "connect_second_node", "connect_third_node"],
        "evidence": ["draw_document", "mark_evidence", "draw_implication_arrow"],
        "spotlight": ["write_key_claim", "draw_emphasis_ring", "add_one_annotation"],
        "feedback_loop": ["draw_first_stage", "connect_cycle", "close_return_arrow"],
        "decision_tree": ["draw_decision_node", "draw_branches", "write_outcomes"],
        "object_focus": ["draw_symbol", "write_annotation_one", "write_annotation_two"],
    }.get(composition, ["draw_main_element", "add_annotation"])
    for i, action in enumerate(action_map):
        actions.append({"at_fraction": round(min(0.88, 0.30 + 0.18 * i), 2), "action": action,
                        "target": labels[min(i, len(labels) - 1)] if labels else motif})
    return actions



def _local_drawing_actions(composition: str, labels: list[str], motif: str,
                           excerpt: str = "") -> list[dict[str, Any]]:
    """Build a varied, progressive vector storyboard when AI planning is disabled.

    Coordinates are normalized to 0..1 so the renderer is resolution-independent.
    The scene uses a few deliberate strokes and labels rather than card templates.
    """
    labels = [re.sub(r"\s+", " ", str(x)).strip() for x in labels if str(x).strip()][:3]
    if not labels:
        labels = ["Key idea"]
    actions: list[dict[str, Any]] = []

    def box(x1, y1, x2, y2, at, end, color="dim"):
        actions.append({"type": "box", "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "start": at, "end": end, "color": color, "width": 3})

    def line(x1, y1, x2, y2, at, end, color="chalk", kind="line"):
        actions.append({"type": kind, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "start": at, "end": end, "color": color, "width": 3})

    def circle(x, y, radius, at, end, color="teal"):
        actions.append({"type": "circle", "x": x, "y": y, "size": radius,
                        "start": at, "end": end, "color": color, "width": 3})

    def motif_at(x, y, size, at, end, color="yellow"):
        if motif != "none":
            actions.append({"type": "motif", "x": x, "y": y, "size": size,
                            "motif": motif, "start": at, "end": end, "color": color})

    def label(text, x, y, at, end, size="medium", align="center", max_width=0.28, color="chalk"):
        actions.append({"type": "text", "text": text, "x": x, "y": y,
                        "start": at, "end": end, "size": size, "align": align,
                        "max_width": max_width, "color": color})

    c = composition
    if c == "causal_chain":
        centres = [0.18, 0.50, 0.82][:len(labels)]
        for i, (cx, word) in enumerate(zip(centres, labels)):
            at = 0.10 + i * 0.17
            box(cx - .125, .42, cx + .125, .64, at, at + .17, "teal" if i == len(labels)-1 else "dim")
            label(word, cx, .50, at + .07, min(.89, at + .28), "medium", "center", .22,
                  "yellow" if i == len(labels)-1 else "chalk")
            if i < len(centres) - 1:
                line(cx + .13, .53, centres[i+1] - .13, .53, at + .10, at + .28, "yellow", "arrow")
        motif_at(.50, .29, .075, .11, .32, "teal")
    elif c == "timeline":
        line(.12, .52, .88, .52, .10, .30, "teal")
        xs = [0.20, 0.50, 0.80][:len(labels)]
        for i, (x, word) in enumerate(zip(xs, labels)):
            at = .23 + i * .16
            circle(x, .52, .025, at, at + .13, "yellow" if i == len(xs)-1 else "chalk")
            label(word, x, .60, at + .07, min(.91, at + .25), "medium", "center", .24,
                  "yellow" if i == len(xs)-1 else "chalk")
        motif_at(.50, .32, .07, .12, .31)
    elif c == "comparison":
        line(.50, .29, .50, .68, .15, .38, "teal")
        line(.13, .36, .43, .36, .12, .26, "dim")
        line(.57, .36, .87, .36, .29, .43, "dim")
        if labels:
            label(labels[0], .28, .42, .20, .40, "large", "center", .30, "teal")
        if len(labels) > 1:
            label(labels[1], .72, .42, .35, .55, "large", "center", .30, "yellow")
        if len(labels) > 2:
            line(.24, .67, .76, .67, .57, .68, "yellow", "underline")
            label(labels[2], .50, .73, .65, .88, "medium", "center", .62, "chalk")
        else:
            circle(.50, .53, .045, .49, .65, "yellow")
    elif c == "numbers":
        nums = re.findall(r"(?:\$\s?\d[\d,.]*|\b\d+(?:[.,]\d+)?\s*(?:%|percent|million|billion|trillion|thousand|years?|months?)?)", excerpt, re.I)
        extracted_num = re.sub(r"\s+", " ", nums[0]).strip() if nums else labels[0]
        main_num = extracted_num if _normal_phrase(extracted_num) == _normal_phrase(labels[0]) else labels[0]
        line(.18, .69, .82, .69, .15, .36, "dim")
        for i, height in enumerate([.16, .28, .40]):
            x = .30 + i * .20
            box(x-.045, .68-height, x+.045, .69, .18 + i*.13, .36 + i*.13, "teal" if i < 2 else "yellow")
        circle(.72, .38, .12, .15, .38, "yellow")
        label(main_num, .72, .34, .24, .49, "large", "center", .25, "yellow")
        if len(labels) > 1:
            label(labels[1], .72, .49, .50, .70, "small", "center", .32, "chalk")
        if len(labels) > 2:
            label(labels[2], .50, .78, .71, .89, "medium", "center", .62, "teal")
    elif c == "evidence":
        box(.10, .27, .39, .73, .11, .34, "dim")
        for i in range(4):
            yy = .37 + i * .085
            line(.16, yy, .33 if i != 2 else .36, yy, .19 + i*.06, .32 + i*.06,
                 "yellow" if i == 2 else "dim", "underline" if i == 2 else "line")
        line(.41, .50, .58, .50, .42, .58, "teal", "arrow")
        motif_at(.72, .42, .13, .46, .72, "yellow")
        label(labels[0], .72, .60, .60, .76, "medium", "center", .34, "chalk")
        if len(labels) > 1:
            label(labels[1], .72, .68, .73, .89, "small", "center", .34, "teal")
        if len(labels) > 2:
            label(labels[2], .25, .79, .70, .88, "small", "center", .42, "yellow")
    elif c == "system_map":
        positions = [(0.50, .34), (0.25, .66), (0.75, .66)][:len(labels)]
        motif_at(.50, .50, .095, .14, .46, "yellow")
        for i, (x, y) in enumerate(positions):
            at = .21 + i*.16
            # Keep the hub label above its node and the lower labels below their nodes.
            line(.50, .50, x, y, at, at+.18, "teal", "arrow")
            circle(x, y, .055, at+.05, at+.20, "yellow" if i == len(positions)-1 else "dim")
            text_y = y-.105 if y < .5 else y+.075
            label(labels[i], x, text_y, at+.09, min(.92, at+.28),
                  "small", "center", .25, "yellow" if i == len(positions)-1 else "chalk")
    elif c == "feedback_loop":
        positions = [(0.50, .31), (0.25, .63), (0.75, .63)][:max(2, len(labels))]
        for i, (x,y) in enumerate(positions):
            nxt = positions[(i+1) % len(positions)]
            at = .12 + i*.18
            line(x, y, nxt[0], nxt[1], at, at+.18, "teal" if i < len(positions)-1 else "yellow", "arrow")
            circle(x, y, .075, at+.07, at+.23, "yellow" if i == len(positions)-1 else "dim")
            if i < len(labels):
                label(labels[i], x, y+.09, at+.11, min(.92, at+.29), "small", "center", .24,
                      "yellow" if i == len(labels)-1 else "chalk")
    elif c == "decision_tree":
        box(.34, .27, .66, .43, .12, .30, "teal")
        label(labels[0], .50, .32, .20, .39, "medium", "center", .29, "chalk")
        line(.50, .44, .27, .58, .29, .46, "yellow", "arrow")
        line(.50, .44, .73, .58, .39, .56, "yellow", "arrow")
        for i, (x, text) in enumerate(zip((.27,.73), labels[1:3])):
            box(x-.16, .58, x+.16, .72, .44+i*.14, .60+i*.14, "dim")
            label(text, x, .63, .51+i*.14, .70+i*.14, "small", "center", .27,
                  "yellow" if i == 1 else "chalk")
        motif_at(.50, .20, .055, .10, .23, "teal")
    else:  # spotlight / object_focus / a fallback for any valid kind
        motif_at(.50, .43, .19, .11, .60, "yellow")
        if labels:
            label(labels[0], .50, .66, .24, .46, "large", "center", .56, "chalk")
        if len(labels) > 1:
            line(.16, .48, .29, .48, .38, .50, "teal", "arrow")
            label(labels[1], .20, .39, .48, .65, "small", "center", .23, "teal")
        if len(labels) > 2:
            line(.71, .48, .84, .48, .54, .66, "yellow", "arrow")
            label(labels[2], .80, .39, .64, .81, "small", "center", .24, "yellow")
        line(.34, .78, .66, .78, .80, .90, "yellow", "underline")
    # Sparse single-label beats still need a visual build, not one icon and a caption.
    if len(actions) < 4:
        circle(.50, .43, .22, .18, .67, "teal")
    if len(actions) < 4:
        line(.37, .78, .63, .78, .73, .90, "yellow", "underline")
    if actions and max(float(a.get("end", 0)) for a in actions) < .78:
        line(.37, .79, .63, .79, .78, .91, "yellow", "underline")
    # Ordered start times make the action queue progressively reveal, not pop in.
    return sorted(actions, key=lambda a: (float(a.get("start", 0)), a.get("type", "")))


def _sanitize_ai_actions(raw: Any, title: str, labels: list[str]) -> list[dict[str, Any]] | None:
    """Accept only safe, bounded renderer primitives and planned narration labels."""
    if not isinstance(raw, list) or not (4 <= len(raw) <= 10):
        return None
    label_map = {_normal_phrase(x): x for x in [title, *labels]}
    seen_texts: set[str] = set()
    clean: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        typ = str(item.get("type", "")).strip().casefold()
        if typ not in DRAW_ACTION_TYPES:
            return None
        try:
            start, end = float(item.get("start")), float(item.get("end"))
        except (TypeError, ValueError):
            return None
        if not (0.0 <= start <= .90 and start + .035 <= end <= .96):
            return None
        color = str(item.get("color", "chalk")).strip().casefold()
        if color not in DRAW_COLORS:
            color = "chalk"
        a: dict[str, Any] = {"type": typ, "start": round(start, 3), "end": round(end, 3),
                             "color": color, "width": 3}
        try:
            if typ in {"line", "arrow", "underline", "box", "scribble"}:
                for name in ("x1", "y1", "x2", "y2"):
                    val = float(item[name])
                    lower = .17 if name in {"y1", "y2"} else .05
                    if not lower <= val <= .95:
                        return None
                    a[name] = val
                if typ == "box" and (a["x2"] - a["x1"] < .04 or a["y2"] - a["y1"] < .04):
                    return None
            elif typ in {"circle", "motif", "text"}:
                for name in ("x", "y"):
                    val = float(item[name])
                    lower, upper = (.20, .86) if name == "y" else (.07, .93)
                    if not lower <= val <= upper:
                        return None
                    a[name] = val
            if typ == "circle":
                size = float(item.get("size", .06))
                if not .025 <= size <= .16:
                    return None
                # Size is a fraction of canvas height, used as circle radius.
                if float(a["x"]) - size * 9/16 < .04 or float(a["x"]) + size * 9/16 > .96:
                    return None
                if float(a["y"]) - size < .18 or float(a["y"]) + size > .92:
                    return None
                a["size"] = size
            elif typ == "motif":
                motif = str(item.get("motif", "none")).strip().casefold()
                size = float(item.get("size", .10))
                if motif not in MOTIFS or motif == "none" or not .035 <= size <= .25:
                    return None
                # Motif drawing uses about half the requested size as its radius.
                if float(a["x"]) - size * 9/32 < .04 or float(a["x"]) + size * 9/32 > .96:
                    return None
                if float(a["y"]) - size/2 < .18 or float(a["y"]) + size/2 > .92:
                    return None
                a["motif"], a["size"] = motif, size
            elif typ == "text":
                phrase = re.sub(r"\s+", " ", str(item.get("text", ""))).strip(" .,:;!?—-")
                normalized = _normal_phrase(phrase)
                if normalized not in label_map or normalized in seen_texts:
                    return None
                if len(phrase) > 32 or len(phrase.split()) > 5:
                    return None
                seen_texts.add(normalized)
                a["text"] = label_map[normalized]
                a["size"] = str(item.get("size", "medium")).casefold()
                if a["size"] not in {"small", "medium", "large"}:
                    a["size"] = "medium"
                a["align"] = str(item.get("align", "center")).casefold()
                if a["align"] not in {"left", "center", "right"}:
                    a["align"] = "center"
                max_width = float(item.get("max_width", .28))
                if not .12 <= max_width <= .70:
                    return None
                x = float(a["x"])
                if a["align"] == "center" and (x - max_width/2 < .04 or x + max_width/2 > .96):
                    return None
                if a["align"] == "left" and x + max_width > .96:
                    return None
                if a["align"] == "right" and x - max_width < .04:
                    return None
                a["max_width"] = max_width
            clean.append(a)
        except (KeyError, TypeError, ValueError):
            return None
    required_texts = {_normal_phrase(x) for x in labels}
    if not required_texts.issubset(seen_texts):
        return None
    # All time intervals and positions are bounded; drawing is scheduled throughout
    # the scene, not front-loaded into the first half.
    starts = {round(float(x["start"]), 2) for x in clean}
    if max(float(x["end"]) for x in clean) < .78 or min(float(x["start"]) for x in clean) > .18 or len(starts) < 3:
        return None
    return sorted(clean, key=lambda x: (float(x["start"]), x["type"]))

def _validate_rows(rows: list[dict[str, Any]], original_script: str, duration: float) -> None:
    if not rows:
        raise ValueError("Director produced no scenes")
    excerpts = [" ".join(str(r.get("narration_excerpt", "")).split()) for r in rows]
    expected = " ".join(re.sub(r"\s+", " ", original_script).strip().split())
    if " ".join(excerpts).casefold() != expected.casefold():
        raise ValueError("Scene excerpts do not preserve the complete script in its original order")
    seen_excerpts: set[str] = set()
    last_end = 0.0
    for i, row in enumerate(rows):
        norm_excerpt = _normal_phrase(excerpts[i])
        if norm_excerpt in seen_excerpts:
            raise ValueError(f"Duplicate narration excerpt in scene {i + 1}")
        seen_excerpts.add(norm_excerpt)
        labels = [str(x).strip() for x in row.get("labels", [])]
        norms = [_normal_phrase(x) for x in labels]
        if not labels or any(not x for x in labels) or len(set(norms)) != len(norms):
            raise ValueError(f"Scene {i + 1} has empty or duplicated labels")
        if any(len(x.split()) > 5 for x in labels):
            raise ValueError(f"Scene {i + 1} label exceeds five words")
        if row.get("composition") not in COMPOSITIONS:
            raise ValueError(f"Scene {i + 1} has invalid composition: {row.get('composition')}")
        if row.get("motif") not in MOTIFS:
            raise ValueError(f"Scene {i + 1} has invalid motif: {row.get('motif')}")
        actions = row.get("drawing_actions")
        if not isinstance(actions, list) or len(actions) < 4:
            raise ValueError(f"Scene {i + 1} lacks an executable drawing-action plan")
        action_texts = []
        for action in actions:
            if not isinstance(action, dict) or action.get("type") not in DRAW_ACTION_TYPES:
                raise ValueError(f"Scene {i + 1} contains an invalid drawing action")
            ast, aen = float(action.get("start", -1)), float(action.get("end", -1))
            if not (0 <= ast < aen <= .96):
                raise ValueError(f"Scene {i + 1} contains invalid action timing")
            if action.get("type") == "text":
                action_texts.append(_normal_phrase(str(action.get("text", ""))))
        if len(action_texts) != len(set(action_texts)):
            raise ValueError(f"Scene {i + 1} repeats text in its drawing actions")
        if not {_normal_phrase(x) for x in labels}.issubset(set(action_texts)):
            raise ValueError(f"Scene {i + 1} drawing actions do not display every scene label")
        start = float(row.get("start_seconds", 0.0))
        end = float(row.get("end_seconds", 0.0))
        if abs(start - last_end) > 0.12 and i > 0:
            raise ValueError(f"Scene {i + 1} timeline gap/overlap: {start - last_end:.3f}s")
        if end <= start or start < 0 or end > duration + 0.15:
            raise ValueError(f"Scene {i + 1} has invalid timing {start:.3f}-{end:.3f}")
        last_end = end
    if abs(float(rows[0]["start_seconds"])) > 0.12 or abs(float(rows[-1]["end_seconds"]) - duration) > 0.15:
        raise ValueError("Scene plan does not span the full audio duration")


def _ai_direct(rows: list[dict[str, Any]], script: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Use AI for story choices; generate compact, validated chalk geometry locally."""
    _local_env()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY") or os.environ.get("OPENROUTER_TOKEN")
    if not key:
        return None, "AI Director requested but OPENROUTER_API_KEY/OPENROUTER_KEY was not found"
    model = (os.environ.get("FERN_CHALKBOARD_MODEL") or os.environ.get("OPENROUTER_MODEL") or
             "nvidia/nemotron-3-ultra-550b-a55b:free").strip()
    blueprint = [{
        "scene": r["scene"], "excerpt": r["narration_excerpt"],
        "current_composition": r["composition"], "word_count": r["word_count"],
    } for r in rows]
    try:
        from moduli.chalk_composer import LAYOUTS as _LAYOUTS
        from moduli.chalk_illustrations import LIBRARY as _LIB
        CHALK_LAYOUTS, CHALK_MOTIFS = ", ".join(_LAYOUTS), ", ".join(sorted(_LIB))
    except ImportError:
        CHALK_LAYOUTS, CHALK_MOTIFS = "hero", "document"
    system = (
        "You are a high-end visual director for a narrated chalkboard documentary. "
        "Choose visual explanations that match each spoken beat; avoid generic filler, repeated layouts, "
        "and copying whole narration sentences onto the board. Never invent facts. "
        "Return ONLY one compact, valid JSON object with a 'scenes' array containing exactly one object "
        "for every supplied scene, in the same order. Each object must contain ONLY: scene (same integer), "
        "purpose (under 8 words), composition (one of: " + ", ".join(sorted(COMPOSITIONS)) + "), "
        "motif (one of: " + ", ".join(sorted(MOTIFS)) + "), title (2-4 words, distinct from labels), "
        "labels (array of 2-3 concise, unique labels, each 2-4 words and at most 5 words), "
        "visual_metaphor (12 words maximum), and optionally chalk: an object with layout (one of: " + CHALK_LAYOUTS + "), "
        "motifs (1-4 of: " + CHALK_MOTIFS + "), labels (up to 3 captions of 1-3 words), title (2-4 words or empty), "
        "anchors (object mapping motif0..motif3, label0..label3, title or number to an EXACT word or phrase copied from "
        "that scene's excerpt - the spoken word that should trigger that drawing), camera (push, closeup or hold), "
        "transition (pan, or erase for a hard topic break) and number (only if the beat states a figure). Keep strings short. "
        "Do NOT output drawing_actions, coordinates, drawing_sequence, markdown, explanations, or extra keys: "
        "FERN generates the timed chalk strokes locally. Avoid the same composition as the immediately previous "
        "scene when another composition fits the meaning. Use timeline only for real chronology, numbers only "
        "when relevant numbers occur, causal_chain for actual cause/effect, and evidence for a clue or record. "
        "Make motifs specific to the beat. No stock footage, photo directions, UI mockups, or hand/marker sprites."
    )
    payload = {
        "model": model,
        "temperature": 0.25,
        # Keep the response within free-provider output limits. The model chooses
        # the story grammar; the local renderer supplies all detailed coordinates.
        "max_tokens": min(4500, 400 + 230 * len(rows)),
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"script_excerpt": script, "scene_blueprints": blueprint}, ensure_ascii=False)},
        ],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/", "X-Title": "FERN Chalkboard Director",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=50) as response:
            http_status = getattr(response, "status", 200)
            raw_response = response.read()
        try:
            data = json.loads(raw_response.decode("utf-8", "replace"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"OpenRouter returned non-JSON response (HTTP {http_status}; {type(exc).__name__})"
        if not isinstance(data, dict):
            return None, f"OpenRouter returned unexpected {type(data).__name__} response (HTTP {http_status})"
        if "choices" not in data:
            # OpenRouter can return an upstream/provider error payload even when the HTTP
            # status is 200. Report the safe diagnostic fields instead of hiding it behind
            # KeyError('choices'); never log request headers or the API key.
            api_error = data.get("error")
            if isinstance(api_error, dict):
                error_code = api_error.get("code", api_error.get("type", "unknown"))
                error_message = str(api_error.get("message", "no message provided"))[:500]
            elif api_error is not None:
                error_code = "unknown"
                error_message = str(api_error)[:500]
            else:
                error_code = "none"
                error_message = "no error field provided"
            provider = str(data.get("provider", "unknown"))[:100]
            returned_model = str(data.get("model", model))[:160]
            fields = ",".join(sorted(str(k) for k in data.keys())[:24])
            return None, (f"OpenRouter response had no 'choices' (HTTP {http_status}; fields={fields}; "
                          f"provider={provider}; model={returned_model}; error_code={error_code}; "
                          f"message={error_message})")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None, "OpenRouter returned an empty or malformed choices array"
        message = choices[0].get("message")
        if not isinstance(message, dict) or "content" not in message:
            finish = choices[0].get("finish_reason", "unknown")
            return None, f"OpenRouter returned no message content (finish_reason={finish})"
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
        if not isinstance(content, str) or not content.strip():
            return None, "OpenRouter returned empty/non-text content for the scene plan"
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
            decoder = json.JSONDecoder()
            for index, char in enumerate(content):
                if char != "{":
                    continue
                try:
                    candidate, _ = decoder.raw_decode(content[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict) and isinstance(candidate.get("scenes"), list):
                    parsed = candidate
                    break

            if parsed is None:
                finish_reason = str(choices[0].get("finish_reason", "unknown"))
                prefix = content[:180].replace("\r", " ").replace("\n", " ")
                return None, (
                    f"AI Director output was not valid JSON "
                    f"(finish_reason={finish_reason}; chars={len(content)}; "
                    f"prefix={prefix!r})"
                )
        planned = parsed.get("scenes") if isinstance(parsed, dict) else None
        if not isinstance(planned, list) or len(planned) != len(rows):
            got = len(planned) if isinstance(planned, list) else "missing/non-array"
            return None, f"AI Director returned {got} scenes; expected {len(rows)}"
        result = []
        global_previous = rows[0]
        for i, (base, item) in enumerate(zip(rows, planned)):
            if not isinstance(item, dict):
                return None, f"AI Director scene {i + 1} was not an object; no fallback used"
            composition = str(item.get("composition", base["composition"])).strip().casefold()
            motif = str(item.get("motif", base["motif"])).strip().casefold()
            if composition not in COMPOSITIONS:
                composition = base["composition"]
            if motif not in MOTIFS:
                motif = base["motif"]
            labels = item.get("labels")
            if not isinstance(labels, list):
                labels = base["labels"]
            labels = [re.sub(r"\s+", " ", str(x)).strip(" .,:;!?—-") for x in labels[:3]]
            labels = [x for x in labels if x and len(x.split()) <= 5 and len(x) <= 32]
            if len(labels) < 2 or len({_normal_phrase(x) for x in labels}) != len(labels):
                labels = base["labels"]
            composition = _resolve_composition_for_labels(composition, labels)
            title = re.sub(r"\s+", " ", str(item.get("title", base["title"]))).strip(" .,:;!?—-")[:32]
            if not title or _normal_phrase(title) in {_normal_phrase(x) for x in labels}:
                title = base["title"]
            seq = item.get("drawing_sequence")
            if not isinstance(seq, list):
                seq = base["drawing_sequence"]
            seq = [re.sub(r"\s+", " ", str(x)).strip()[:100] for x in seq[:6] if str(x).strip()]
            # The AI picks what the scene means; deterministic local geometry draws it.
            # Asking free models to emit every coordinate for every beat inflated the
            # response until OpenRouter truncated the JSON before rendering could begin.
            drawing_actions = _local_drawing_actions(
                composition, labels, motif, str(base.get("narration_excerpt") or "")
            )
            result.append({
                **base,
                "visual_kind": composition,
                "composition": composition,
                "motif": motif,
                "title": title,
                "labels": labels,
                "phrases": labels,
                "narrative_purpose": str(item.get("purpose", base["narrative_purpose"]))[:220],
                "visual_metaphor": str(item.get("visual_metaphor", base["visual_metaphor"]))[:360],
                "drawing_sequence": _drawing_sequence(composition, labels, motif),
                "drawing_sequence_text": seq or [x["action"] for x in _drawing_sequence(composition, labels, motif)],
                "drawing_actions": drawing_actions,
                "plan_source": "openrouter_ai_director",
                "chalk_raw": item.get("chalk") if isinstance(item.get("chalk"), dict) else None,
            })
        # Validate the actual chosen plan before rendering it.
        _validate_rows(result, script, float(rows[-1]["end_seconds"]))
        return result, None
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        return None, f"OpenRouter returned HTTP {exc.code}. {detail}"
    except Exception as exc:
        return None, f"AI Director failed ({type(exc).__name__}: {exc}); used local plans"


def _attach_word_times(rows: list[dict[str, Any]], alignment: dict[str, Any] | None) -> None:
    """Give every scene the spoken start/end of each of its words (absolute seconds)."""
    if not (alignment and alignment.get("ok") and alignment.get("word_timings")):
        return
    times = alignment["word_timings"]
    cursor = 0
    for row in rows:
        n = len(str(row.get("narration_excerpt", "")).split())
        chunk = times[cursor:cursor + n]
        cursor += n
        if len(chunk) == n and all(t is not None for t in chunk):
            row["word_times"] = [[round(float(a), 3), round(float(b), 3)] for a, b in chunk]
            row["word_times_source"] = str(alignment.get("timing_source", "measured"))


def _attach_chalk_plans(rows: list[dict[str, Any]]) -> None:
    """Attach the chalk-engine story plan (layout, illustrations, captions, anchors) to each scene.

    The local story planner always runs; an AI-supplied ``chalk_raw`` object may refine it but is
    validated field by field (see ``chalk_story.sanitize_chalk``) and can never remove the fallback.
    """
    try:
        from moduli.chalk_story import plan_chalk_scenes, sanitize_chalk
    except ImportError:  # OpenCV/numpy missing: the legacy renderer does not need a chalk plan
        for r in rows:
            r.pop("chalk_raw", None)
        return
    raw = [r.pop("chalk_raw", None) for r in rows]
    plan_chalk_scenes(rows)
    for r, rw in zip(rows, raw):
        if rw:
            r["chalk"] = sanitize_chalk(rw, str(r.get("narration_excerpt", "")), r["chalk"])


def build_scene_plan(script: str, duration: float, *, audio_path: str | None = None,
                     use_ai: bool = False, align_audio: bool = False, aligner_model: str = "base",
                     target_seconds: float | None = None, use_tts_timestamps: bool = True) -> dict[str, Any]:
    """Build a validated, timing-auditable scene plan for a single audio excerpt."""
    if not script or not script.strip():
        raise ValueError("Cannot direct an empty script")
    if duration <= 0:
        raise ValueError("Audio duration must be positive")
    cleaned = re.sub(r"\s+", " ", script).strip()
    total_words = len(cleaned.split())
    estimated_wpm = total_words * 60.0 / duration
    target_seconds_requested = bool(target_seconds)
    target_seconds = float(target_seconds) if target_seconds else 4.2
    target_words = max(7, min(18, round(estimated_wpm * target_seconds / 60.0)))
    if target_seconds_requested:
        from moduli.chalk_story import segment_beats
        beats = segment_beats(cleaned, target_words)
    else:
        beats = _segment_script(cleaned, target_words)
    if not beats:
        raise ValueError("Could not split script into spoken beats")
    alignment = None
    alignment_note = None
    if audio_path and use_tts_timestamps:
        # Word timestamps captured while the narration was synthesized beat any ASR guess.
        try:
            from moduli.chalk_voice import sidecar_alignment
            alignment = sidecar_alignment(audio_path, cleaned, duration)
        except ImportError:
            alignment = None
        if alignment:
            alignment["row_timing_source"] = alignment["timing_source"]
    if align_audio and not alignment:
        if not audio_path:
            alignment_note = "Audio alignment requested without an audio path; approximate timing used"
        else:
            alignment = align_audio_words(audio_path, cleaned, model=aligner_model)
            if not alignment.get("ok"):
                alignment_note = str(alignment.get("reason", "Local word alignment unavailable"))
    times = _timing_for_beats(beats, duration, alignment)
    pause_adjustments = 0
    if not (alignment and alignment.get("ok")):
        times, pause_adjustments = _nudge_boundaries_to_pauses(times, audio_path, duration)
        if pause_adjustments:
            for item in times:
                item["timing_source"] = "syllable_weighted_plus_audio_pauses"
                item["timing_confidence"] = 0.50
    rows = _build_local_rows(beats, times)
    source = "local_heuristic_director"
    ai_note = None
    if use_ai:
        planned, ai_note = _ai_direct(rows, cleaned)
        if planned:
            rows = planned
            source = "openrouter_ai_director"
        else:
            # The local renderer is still the requested chalkboard style. An AI
            # planning hiccup must not stop the preview or switch to stock footage.
            source = "local_heuristic_director"
            print(
                "[chalkboard] AI direction unavailable; using the local chalkboard director. "
                + str(ai_note or "Unknown AI planning error"),
                flush=True,
            )
    for row in rows:
        row["plan_source"] = source
    _attach_word_times(rows, alignment)
    _attach_chalk_plans(rows)
    _validate_rows(rows, cleaned, duration)
    return {
        "schema_version": "1.0",
        "renderer": "FERN local chalkboard backend",
        "director": source,
        "timing_source": alignment.get("timing_source") if alignment and alignment.get("ok") else ("syllable_weighted_plus_audio_pauses" if pause_adjustments else "syllable_weighted_duration_estimate"),
        "timing_confidence": 0.85 if alignment and alignment.get("ok") else (0.50 if pause_adjustments else 0.42),
        "pause_boundaries_adjusted": pause_adjustments,
        "alignment_backend": alignment.get("backend") if alignment and alignment.get("ok") else None,
        "alignment_coverage": alignment.get("coverage") if alignment and alignment.get("ok") else None,
        "alignment_note": alignment_note,
        "ai_director_note": ai_note,
        "duration_seconds": round(duration, 3),
        "source_script_characters": len(cleaned),
        "source_script_words": total_words,
        "scene_count": len(rows),
        "planning_target_seconds": target_seconds,
        "estimated_script_wpm": round(estimated_wpm, 1),
        "scenes": rows,
    }


