"""Render a short local cinematic preview without provider or upload access."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from moduli.visual_plan import deterministic_visual_plan, parse_srt_cues


def _word_matches(text: str) -> list[re.Match]:
    return list(re.finditer(r"[\w]+", text or "", flags=re.UNICODE))


def first_script_window(script: str, cues: list[dict], audio_duration: float,
                        seconds: float = 45.0) -> tuple[str, list[dict], str]:
    """Take the exact script prefix covered by local SRT cues when they match."""
    words = _word_matches(script)
    if not words or audio_duration < seconds:
        raise ValueError("Narration/script cannot supply the requested 45-second window")

    cursor = 0
    end_character = 0
    selected = []
    for cue in cues:
        if float(cue.get("start", seconds)) >= seconds or float(cue.get("end", seconds)) > seconds:
            break
        cue_words = re.findall(r"[\w]+", str(cue.get("text") or ""), flags=re.UNICODE)
        if not cue_words:
            continue
        anchor = [word.casefold() for word in cue_words]
        found = None
        for start in range(cursor, len(words) - len(anchor) + 1):
            if [match.group(0).casefold() for match in words[start:start + len(anchor)]] == anchor:
                found = start
                break
        if found is None:
            break
        cursor = found + len(anchor)
        end_character = words[cursor - 1].end()
        punctuation = re.match(r"[,.;:!?]+", script[end_character:end_character + 3])
        if punctuation:
            end_character += punctuation.end()
        selected.append(cue)
        if float(cue.get("end", 0.0)) >= seconds:
            break

    if end_character and selected and float(selected[-1].get("end", 0.0)) >= seconds - 5:
        return script[:end_character], selected, "srt"

    word_count = min(len(words), max(1, round(len(words) * seconds / audio_duration)))
    end_character = words[word_count - 1].end()
    return script[:end_character], [], "word-weighted"


def _ordered_anchor_matches(script_window: str, shots: list[dict]) -> bool:
    script_words = [match.group(0).casefold() for match in _word_matches(script_window)]
    cursor = 0
    for shot in shots:
        anchor_words = [match.group(0).casefold()
                        for match in _word_matches(str(shot.get("narration_anchor") or ""))]
        if not 4 <= len(anchor_words) <= 16:
            return False
        found = None
        for start in range(cursor, len(script_words) - len(anchor_words) + 1):
            if script_words[start:start + len(anchor_words)] == anchor_words:
                found = start
                break
        if found is None:
            return False
        cursor = found + len(anchor_words)
    return True


def _strict_live_shots(script_window: str, candidate: object) -> list[dict]:
    from moduli.visual_plan import validate_visual_plan

    if not isinstance(candidate, list) or not 6 <= len(candidate) <= 8:
        raise ValueError("Director plan must contain 6-8 shot cards")
    if not _ordered_anchor_matches(script_window, candidate):
        raise ValueError("Director plan anchors must occur verbatim and in order in the first-45s script")

    queries = [" ".join(str(shot.get("primary_query") or "").split()).casefold()
               for shot in candidate if isinstance(shot, dict)]
    if len(queries) != len(candidate) or any(len(query.split()) < 2 for query in queries):
        raise ValueError("Every shot needs a specific, multi-word Openverse query")
    if len(set(queries)) != len(queries):
        raise ValueError("Director plan repeats an identical visual search query")

    # Different wording can still describe the same shot. Reject repeated
    # visual concepts such as typing/keyboard scenes before they reach render.
    concept_families = {
        "typing": re.compile(r"\b(typing|typewriter|keyboard|keyboarding|data entry|hands typing)\b"),
        "meeting": re.compile(r"\b(meeting room|boardroom|conference room|business meeting)\b"),
        "server_room": re.compile(r"\b(server rack|server room|data center|server farm)\b"),
    }
    seen_families: dict[str, int] = {}
    for query in queries:
        for family, pattern in concept_families.items():
            if pattern.search(query):
                seen_families[family] = seen_families.get(family, 0) + 1
    repeated = [family for family, count in seen_families.items() if count > 1]
    if repeated:
        raise ValueError("Director plan repeats visual concept(s): " + ", ".join(repeated))

    normalized_input = []
    for shot in candidate:
        if not isinstance(shot, dict):
            raise ValueError("Director plan contains a non-object shot")
        try:
            duration = float(shot.get("duration_hint"))
        except (TypeError, ValueError) as error:
            raise ValueError("Every shot needs a numeric duration_hint") from error
        if not 2.5 <= duration <= 9.0:
            raise ValueError("Shot duration_hint must be between 2.5 and 9 seconds")
        for field in ("visual_intent", "shot_type", "composition"):
            if not str(shot.get(field) or "").strip():
                raise ValueError(f"Director plan is missing {field}")
        normalized_input.append({
            **shot,
            "visual_type": "illustration",
            "alternate_queries": [],
            "primary_query": " ".join(str(shot["primary_query"]).split()),
        })

    normalized = validate_visual_plan(script_window, normalized_input)
    if (len(normalized) != len(normalized_input)
            or any(normalized[i]["narration_anchor"] != normalized_input[i]["narration_anchor"]
                   for i in range(len(normalized)) )):
        raise ValueError("Visual-plan validation changed or replaced the returned shot plan")
    return normalized


def build_live_shot_plan(script: str, script_window: str,
                         plan_path: str = "output/cinematic_preview_live.plan.json",
                         request_fn=None) -> tuple[list[dict], int, bool]:
    """Reuse only a plan hash-matched to this script; otherwise issue one request."""
    script_hash = hashlib.sha256(script.encode("utf-8")).hexdigest()
    window_hash = hashlib.sha256(script_window.encode("utf-8")).hexdigest()
    try:
        cached = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if (isinstance(cached, dict) and cached.get("script_sha256") == script_hash
            and cached.get("window_sha256") == window_hash):
        try:
            return (_strict_live_shots(script_window, cached.get("shots")),
                    int(cached.get("model_requests_total", 0)), True)
        except (TypeError, ValueError):
            pass

    from dotenv import load_dotenv
    load_dotenv(override=False)
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is missing; no model request was made")

    prompt = (
        "Create exactly 6-8 documentary shot cards for ONLY the supplied first-45-second narration. "
        "Every narration_anchor must be an exact 4-16 word excerpt, in order, copied from the script. "
        "Each primary_query must be a distinct, concrete 2-8 word Openverse image search query. "
        "Every shot must depict a DIFFERENT visual idea, physical action, or setting from every other shot. "
        "Do not repeat typing, keyboards, hands on laptops, or people working at computers in multiple shots; "
        "use that visual concept at most once. Also avoid repeating meeting rooms or server rooms. "
        "Do not merely reword a query to disguise the same scene. Vary subjects and visual metaphors across the sequence. "
        "All retrieved visuals will be still images converted to short motion clips: set visual_type to illustration, "
        "never imply that an illustration is authentic footage or documentary evidence. Avoid generic AI brains, "
        "server racks, stock business people, and fabricated interfaces. Use one to three visual changes per idea, "
        "3-8 second duration_hint values, restrained motion, and cuts by default. No absolute timestamps. "
        "Sound cue must be empty unless a specific local sound could genuinely support the beat. Do not invent facts. "
        "Return JSON only in this schema: {\"shots\":[{\"id\":\"shot-01\","
        "\"narration_anchor\":\"exact excerpt\",\"dramatic_function\":\"hook|orientation|evidence|escalation|contrast|reveal|consequence|resolution\","
        "\"visual_intent\":\"what the viewer should understand\",\"primary_query\":\"specific image search\","
        "\"shot_type\":\"close-up, establishing wide, insert, etc.\",\"composition\":\"subject and framing\","
        "\"motion\":\"slow_push|lateral_pan|static_hold\",\"duration_hint\":5,\"transition\":\"cut|dissolve\","
        "\"overlay_text\":\"short or empty\",\"sound_cue\":\"empty unless justified\","
        "\"music_intensity\":0.3,\"hold_for_reveal\":false}]}.\n\n"
        "SCRIPT WINDOW:\n" + script_window
    )
    if request_fn is None:
        from moduli.ai_client import chat_openrouter
        request_fn = chat_openrouter
    print("[live-preview] OpenRouter request 1/1 for script-specific shot plan", flush=True)
    response = request_fn(
        prompt,
        system="You are a documentary picture editor. Be precise, evidence-conscious, and visually specific.",
        max_tokens=5000,
        json_mode=True,
        max_attempts=1,
    )
    try:
        parsed = json.loads(response)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"OpenRouter returned invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("OpenRouter plan response must be a JSON object")
    shots = _strict_live_shots(script_window, parsed.get("shots"))

    os.makedirs(os.path.dirname(plan_path) or ".", exist_ok=True)
    temporary = plan_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump({"script_sha256": script_hash, "window_sha256": window_hash,
               "model_requests_total": 1, "queries_attempted": [],
               "shots": shots}, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, plan_path)
    return shots, 1, False


def retrieve_openverse_beat_assets(shots: list[dict], cache_dir: str = "cache/openverse",
                                   max_queries: int = 8,
                                   plan_state_path: str | None = None
                                   ) -> tuple[dict[str, str], list[dict], int]:
    """Use only exact-query metadata cache entries or bounded Openverse search."""
    from moduli import asset

    unique_queries = list(dict.fromkeys(
        " ".join(str(shot.get("primary_query") or "").split()) for shot in shots
    ))
    if not unique_queries or len(unique_queries) > min(8, max_queries):
        raise ValueError("Live plan must have 1-8 distinct Openverse queries")

    asset_map: dict[str, str] = {}
    sources: list[dict] = []
    owners: dict[str, str] = {}
    cache_root = Path(cache_dir)
    resolved_by_query: dict[str, list[dict]] = {}
    state = {}
    if plan_state_path:
        try:
            state = json.loads(Path(plan_state_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
    attempted = list(dict.fromkeys(str(query) for query in state.get("queries_attempted", [])))
    if len(attempted) > min(8, max_queries):
        raise RuntimeError("The Openverse search budget for this script is already exhausted")

    def _record_attempt(query: str) -> None:
        if query in attempted:
            raise RuntimeError(f"Openverse query already attempted; refusing duplicate: {query}")
        if len(attempted) >= min(8, max_queries):
            raise RuntimeError("The Openverse search budget for this script is exhausted")
        attempted.append(query)
        if plan_state_path:
            state["queries_attempted"] = attempted
            temporary = plan_state_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, plan_state_path)

    for shot in shots:
        beat_id = str(shot["id"])
        query = " ".join(str(shot["primary_query"]).split())
        candidates = []

        if query.casefold() in resolved_by_query:
            candidates = resolved_by_query[query.casefold()][:2]
        else:
            for manifest_path in cache_root.glob("*.mp4.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (str(manifest.get("query") or "").casefold() != query.casefold()
                        or not manifest.get("asset_url") or not manifest.get("source_page")
                    or not manifest.get("license") or manifest.get("preview_excluded")):
                    continue
                media_path = Path(str(manifest_path)[:-5])
                if media_path.is_file():
                    candidates.append({**manifest, "_cached_path": str(media_path)})
                    if len(candidates) == 2:
                        break

        if not candidates:
            _record_attempt(query)
            print(f"[live-preview] Openverse search {len(attempted)}/{min(8, max_queries)}: {query}", flush=True)
            results = asset._ov_search(query, 2, candidate_limit=2)
            candidates = results[:2]
        if not candidates:
            raise RuntimeError(f"Openverse returned no candidates for beat {beat_id}: {query}")

        beat_paths = []
        for candidate_index, item in enumerate(candidates[:2]):
            if "_cached_path" not in item and not (
                (item.get("url") or item.get("thumbnail"))
                and (item.get("foreign_landing_url") or item.get("detail_url") or item.get("related_url"))
                and item.get("license") and item.get("license_url")
            ):
                continue
            if "_cached_path" in item:
                path = item["_cached_path"]
            else:
                path = asset._ov_download_and_render(
                    item, min(9.0, max(2.5, float(shot["duration_hint"])))
                )
            if not path:
                continue
            path = str(Path(path).resolve())
            beat_paths.append(path)
            resolved_by_query.setdefault(query.casefold(), []).append(
                {**item, "_cached_path": path}
            )
            title = str(item.get("title") or "Untitled Openverse image")
            key = f"{beat_id}|{query}|{title}#{candidate_index}"
            asset_map[key] = path
            sources.append({
                "beat_id": beat_id,
                "query": query,
                "asset_type": "Openverse still image converted locally to motion clip",
                "title": title,
                "creator": item.get("creator", ""),
                "asset_url": item.get("asset_url") or item.get("url") or item.get("thumbnail") or "",
                "source_page": item.get("source_page") or item.get("foreign_landing_url")
                               or item.get("detail_url") or item.get("related_url") or "",
                "license": item.get("license", ""),
                "license_url": item.get("license_url", ""),
                "local_path": path,
            })
        if not beat_paths:
            raise RuntimeError(f"No usable attributed Openverse image for beat {beat_id}: {query}")

    return asset_map, sources, len(attempted)


def render_live_preview(
    script_path: str = "director_input/script.txt",
    audio_path: str = "output/narration.mp3",
    subtitle_path: str = "output/sottotitoli.srt",
    plan_path: str = "output/cinematic_preview_live.plan.json",
    sources_path: str = "output/cinematic_preview_live.sources.json",
    output_path: str = "output/cinematic_preview_live.mp4",
) -> dict:
    """Generate one script-specific plan, retrieve bounded Openverse assets, render 45s."""
    required = (script_path, audio_path)
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing live-preview input(s): " + ", ".join(missing))
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing preview: {output_path}")

    script = Path(script_path).read_text(encoding="utf-8").strip()
    if not script:
        raise ValueError(f"Empty script: {script_path}")
    try:
        subtitle_text = Path(subtitle_path).read_text(encoding="utf-8")
        subtitle_cues = parse_srt_cues(subtitle_text)
    except OSError:
        subtitle_cues = []

    from moduli.montaggio import _media_duration, _select_timeline_clips, monta_video
    audio_duration = _media_duration(audio_path)
    if audio_duration < 45.0:
        raise ValueError(f"Narration is only {audio_duration:.2f}s; 45-second preview unavailable")
    script_window, window_cues, timing_source = first_script_window(
        script, subtitle_cues, audio_duration, 45.0
    )
    if len(_word_matches(script_window)) < 35:
        raise ValueError("The aligned first-45-second script window is unexpectedly short")

    shots, model_requests, reused_plan = build_live_shot_plan(script, script_window, plan_path)
    assets, sources, footage_searches = retrieve_openverse_beat_assets(
        shots, plan_state_path=plan_path
    )
    from moduli.visual_plan import build_shot_timeline
    timeline = build_shot_timeline(script, shots, audio_duration, 45.0,
                                   subtitle_cues=window_cues)
    if not timeline:
        raise RuntimeError("The script-specific plan produced no preview timeline")
    selected_paths = _select_timeline_clips(timeline, assets, list(assets.values()))
    assigned = []
    for event, selected in zip(timeline, selected_paths):
        beat_id = str(event["beat"].get("id") or "")
        if not any(beat_id in key and path == selected for key, path in assets.items()):
            raise RuntimeError(f"Asset assignment escaped its planned beat: {beat_id}")
        assigned.append({"beat_id": beat_id, "local_path": selected})

    audit = {"model_requests": model_requests, "footage_searches": footage_searches,
             "plan_reused": reused_plan, "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
             "timing_source": timing_source, "shots": shots,
             "asset_sources": sources, "assignments": assigned,
             "rendered_duration": None}
    os.makedirs(os.path.dirname(sources_path) or ".", exist_ok=True)
    with open(sources_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    os.replace(sources_path + ".tmp", sources_path)

    temporary_output = output_path + ".rendering.mp4"
    if os.path.exists(temporary_output):
        raise FileExistsError(f"Refusing to overwrite unfinished render: {temporary_output}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(
        f"[live-preview] {len(shots)} script-specific shots; {len(timeline)} timeline events; "
        f"{len(assets)} attributed Openverse assets; {footage_searches} searches; "
        f"SRT timing: {timing_source}. Plan verified for {len(assigned)} event assignments.",
        flush=True,
    )
    monta_video(
        audio_path,
        list(assets),
        assets,
        temporary_output,
        captions_text=script,
        visual_plan=shots,
        max_duration=45.0,
        subtitle_cues=window_cues or None,
    )
    rendered_duration = _media_duration(temporary_output)
    if rendered_duration > 45.1 or rendered_duration < 44.0:
        raise RuntimeError(f"Rendered preview duration {rendered_duration:.2f}s is outside the 45s limit")
    os.replace(temporary_output, output_path)
    audit["rendered_duration"] = rendered_duration
    with open(sources_path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    os.replace(sources_path + ".tmp", sources_path)
    return {"output_path": output_path, "shots": shots,
            "model_requests": model_requests, "footage_searches": footage_searches,
            "asset_sources": sources, "assignments": assigned,
            "rendered_duration": rendered_duration, "reused_plan": reused_plan}


def collect_local_assets(visual_plan: list[dict], cache_root: str = "cache",
                         limit: int = 36) -> dict[str, str]:
    """Build a relevance-labeled pool from local caches; never contacts providers."""
    root = Path(cache_root)
    candidates: dict[str, str] = {}
    index_path = root / "pexels" / "index.json"
    metadata_by_file = {}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index = {}
    if isinstance(index, dict):
        for metadata in index.values():
            if not isinstance(metadata, dict):
                continue
            filename = os.path.basename(str(metadata.get("file") or ""))
            description = " ".join(str(value) for value in (
                metadata.get("tags", []), metadata.get("url", ""),
                metadata.get("author", ""),
            ))
            if filename:
                metadata_by_file[filename] = description
                video_id = metadata.get("pexels_id")
                if video_id:
                    metadata_by_file[f"pex_{video_id}.mp4"] = description

    for folder in ("pexels", "openverse", "ai_video"):
        directory = root / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mp4")):
            if path.stem.endswith(("_720p", "_1080p", "_480p")):
                continue
            resolved = path.resolve()
            if str(resolved) in candidates.values():
                continue
            if folder == "pexels":
                label = metadata_by_file.get(path.name, "cached Pexels video")
                source_kind = "video"
            elif folder == "openverse":
                manifest_path = Path(str(path) + ".json")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
                label = " ".join(str(manifest.get(key) or "") for key in ("query", "title", "creator"))
                label = label.strip() or "unmapped Openverse illustrative still"
                source_kind = "still"
            else:
                manifest_path = Path(str(path) + ".json")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
                label = " ".join(str(manifest.get(key) or "") for key in ("query", "prompt"))
                label = label.strip() or "uncategorized cached AI video"
                source_kind = "still" if manifest.get("visual_type") == "illustration" else "video"
            key = f"{source_kind}|{label}#{len(candidates)}"
            candidates[key] = str(resolved)

    pixabay_manifest = root / "pixabay_preview" / "manifest.json"
    try:
        manifest_data = json.loads(pixabay_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_data = {}

    plan_shots_by_query: dict[str, set[str]] = {}
    for index, beat in enumerate(visual_plan, 1):
        if not isinstance(beat, dict):
            continue
        shot_id = str(beat.get("id") or f"shot-{index:02d}")
        queries = [beat.get("primary_query")]
        alternates = beat.get("alternate_queries")
        if isinstance(alternates, list):
            queries.extend(alternates)
        for query in queries:
            normalized_query = " ".join(str(query or "").split()).casefold()
            if normalized_query:
                plan_shots_by_query.setdefault(normalized_query, set()).add(shot_id)

    if isinstance(manifest_data, dict) and isinstance(manifest_data.get("clips"), list):
        pixabay_by_id: dict[str, dict] = {}
        path_to_id: dict[str, str] = {}
        for entry in manifest_data["clips"]:
            if not isinstance(entry, dict):
                continue
            video_id = str(entry.get("pixabay_video_id") or "").strip()
            if not video_id or not video_id.isdigit():
                continue
            raw_path = str(entry.get("download_path") or "").strip()
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            try:
                resolved = str(path.resolve(strict=True))
                if path.suffix.casefold() != ".mp4" or not Path(resolved).is_file():
                    continue
                if Path(resolved).stat().st_size < 1024:
                    continue
            except (OSError, RuntimeError):
                continue

            queries = entry.get("queries")
            if isinstance(queries, str):
                queries = [queries]
            if not isinstance(queries, list):
                queries = []
            queries = list(dict.fromkeys(
                " ".join(str(query).split()) for query in queries if str(query).strip()
            ))

            shot_ids = entry.get("shot_ids")
            if isinstance(shot_ids, str):
                shot_ids = [shot_ids]
            if not isinstance(shot_ids, list):
                shot_ids = []
            associated_shots = {str(shot_id) for shot_id in shot_ids if str(shot_id).strip()}

            matched_shots = entry.get("matched_shots")
            if isinstance(matched_shots, list):
                associated_shots.update(
                    str(shot.get("shot_id")) for shot in matched_shots
                    if isinstance(shot, dict) and str(shot.get("shot_id") or "").strip()
                )
            for query in queries:
                associated_shots.update(plan_shots_by_query.get(query.casefold(), set()))

            existing = pixabay_by_id.get(video_id)
            if existing is None:
                existing_id = path_to_id.get(resolved)
                if existing_id is not None:
                    existing = pixabay_by_id[existing_id]
                else:
                    existing = {
                        "entry": dict(entry),
                        "path": resolved,
                        "video_ids": [],
                        "queries": [],
                        "shot_ids": set(),
                    }
                    pixabay_by_id[video_id] = existing
                    path_to_id[resolved] = video_id
            if video_id not in existing["video_ids"]:
                existing["video_ids"].append(video_id)
            if video_id not in pixabay_by_id:
                pixabay_by_id[video_id] = existing
            for query in queries:
                if query.casefold() not in {value.casefold() for value in existing["queries"]}:
                    existing["queries"].append(query)
            existing["shot_ids"].update(associated_shots)

        existing_paths = {str(Path(path).resolve()) for path in candidates.values()}
        for item in pixabay_by_id.values():
            path = item["path"]
            if path in existing_paths:
                continue
            entry = item["entry"]
            metadata = {
                "source": "pixabay_video",
                "pixabay_video_ids": item["video_ids"],
                "queries": item["queries"],
                "shot_ids": sorted(item["shot_ids"]),
                "title": str(entry.get("title") or ""),
                "uploader": str(entry.get("uploader") or ""),
                "source_page_url": str(entry.get("source_page_url") or ""),
                "duration_seconds": entry.get("duration_seconds"),
                "resolution": entry.get("resolution"),
                "license": entry.get("license"),
                "license_url": entry.get("license_url"),
                "license_note": entry.get("license_note"),
                "provenance": entry.get("provenance"),
                "download_path": path,
            }
            key = "video|pixabay|" + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            candidates[key] = path
            existing_paths.add(path)

    if not candidates:
        return {}

    wanted = set()
    prefer_video = any(
        str(beat.get("visual_type") or "video").casefold() in {"video", "actual video"}
        for beat in visual_plan
    )
    for beat in visual_plan:
        wanted.update(re.findall(r"[a-z0-9]+", str(beat.get("primary_query", "")).casefold()))
        wanted.update(re.findall(r"[a-z0-9]+", str(beat.get("visual_intent", "")).casefold()))
    def rank(item: tuple[str, str]) -> tuple[int, str]:
        overlap = len(wanted & set(re.findall(r"[a-z0-9]+", item[0].casefold())))
        preference = -3 if prefer_video and item[0].startswith("still|") else 0
        pixabay_video_preference = 8 if item[0].startswith("video|pixabay|") else 0
        return (-(overlap + preference + pixabay_video_preference), item[0])

    ranked = sorted(candidates.items(), key=rank)
    return dict(ranked[:max(1, limit)])


def render_preview(
    script_path: str = "director_input/script.txt",
    audio_path: str = "output/narration.mp3",
    subtitle_path: str = "output/sottotitoli.srt",
    plan_path: str = "output/visual_plan.json",
    output_path: str = "output/cinematic_preview.mp4",
    duration: float = 45.0,
) -> str:
    """Render only local inputs and cached assets; has no publisher dependency."""
    if not 30.0 <= duration <= 60.0:
        raise ValueError("Preview duration must be between 30 and 60 seconds")

    missing = [path for path in (script_path, audio_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing local preview input(s): " + ", ".join(missing))
    if os.path.abspath(output_path) == os.path.abspath("output/output_finale.mp4"):
        raise ValueError("Preview output must not replace the production video")

    script = Path(script_path).read_text(encoding="utf-8").strip()
    if not script:
        raise ValueError(f"Empty local script: {script_path}")

    try:
        candidate = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        candidate = None
    if isinstance(candidate, dict) and isinstance(candidate.get("shots"), list):
        candidate = candidate["shots"]

    # Cached plans can be syntactically distinct while depicting the same idea
    # repeatedly. In offline/local preview mode, repair that by rebuilding a
    # script-derived plan rather than silently showing the same kind of shot.
    if isinstance(candidate, list):
        query_texts = [
            " ".join(str(shot.get("primary_query") or "").split()).casefold()
            for shot in candidate if isinstance(shot, dict)
        ]
        concept_patterns = {
            "typing": re.compile(r"\\b(typing|typewriter|keyboard|keyboarding|data entry|hands typing)\\b"),
            "meeting": re.compile(r"\\b(meeting room|boardroom|conference room|business meeting)\\b"),
            "server_room": re.compile(r"\\b(server rack|server room|data center|server farm)\\b"),
        }
        repeated_families = [
            family for family, pattern in concept_patterns.items()
            if sum(bool(pattern.search(query)) for query in query_texts) > 1
        ]
        if (len(query_texts) != len(candidate)
                or len(set(query_texts)) != len(query_texts)
                or repeated_families):
            print(
                "[preview] Repeated visual concept detected"
                + (": " + ", ".join(repeated_families) if repeated_families else "")
                + "; rebuilding a diverse local plan.",
                flush=True,
            )
            candidate = None

    plan = deterministic_visual_plan(script) if not isinstance(candidate, list) else candidate
    assets = collect_local_assets(plan)
    if not assets:
        raise FileNotFoundError(
            "No cached visual assets found under cache/pexels, cache/openverse, or cache/ai_video"
        )

    try:
        subtitle_text = Path(subtitle_path).read_text(encoding="utf-8")
        cues = parse_srt_cues(subtitle_text)
    except OSError:
        cues = []

    from moduli.montaggio import _media_duration, monta_video
    audio_duration = _media_duration(audio_path)
    preview_duration = min(float(duration), audio_duration)
    if preview_duration < 1.0:
        raise ValueError("Local narration is too short for a preview")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    monta_video(
        audio_path,
        list(assets),
        assets,
        output_path,
        captions_text=script,
        visual_plan=plan,
        max_duration=preview_duration,
        subtitle_cues=cues or None,
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a local preview; --live uses one script-specific plan request.")
    parser.add_argument("--live", action="store_true",
                        help="Generate/reuse an exact-script plan and retrieve only Openverse visuals.")
    parser.add_argument("--duration", type=float, default=45.0,
                        help="Preview length in seconds (30-60; default 45).")
    parser.add_argument("--script", default="director_input/script.txt")
    parser.add_argument("--audio", default="output/narration.mp3")
    parser.add_argument("--subtitles", default="output/sottotitoli.srt")
    parser.add_argument("--plan", default="output/visual_plan.json")
    parser.add_argument("--output", default="output/cinematic_preview.mp4")
    args = parser.parse_args()
    if args.live:
        try:
            report = render_live_preview(args.script, args.audio, args.subtitles)
        except Exception as error:
            print(f"Live preview stopped without fallback: {error}", file=sys.stderr)
            return 2
        print(f"Preview: {report['output_path']} ({report['rendered_duration']:.2f}s)")
        print(f"OpenRouter requests: {report['model_requests']}; Openverse searches: {report['footage_searches']}")
        for source in report["asset_sources"]:
            print(f"{source['beat_id']}: {source['title']} | {source['license']} | {source['source_page']}")
        return 0
    try:
        path = render_preview(args.script, args.audio, args.subtitles, args.plan,
                              args.output, args.duration)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Preview not rendered: {error}", file=sys.stderr)
        return 2
    print(f"Local preview saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())