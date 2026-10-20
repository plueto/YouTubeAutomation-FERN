"""Discover a small pool of openly licensed, directly downloadable archival videos.

This source adapter deliberately does not use YouTube scraping or generic stock
footage. It searches Internet Archive's public metadata API, accepts only
CC0/Public Domain Mark/CC BY 3.0 or 4.0 records, checks topic relevance, and
downloads bounded local MP4 files. Unclear licenses, weak topic matches, missing
media, or missing attribution metadata are rejected; the chalk engine remains
the visual fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

from moduli.credible_footage import (
    MANIFEST_SCHEMA,
    _relevance,
    _scene_query,
    _tokens,
    script_fingerprint,
)

_SEARCH_URL = "https://archive.org/advancedsearch.php"
_METADATA_URL = "https://archive.org/metadata"
_USER_AGENT = "FERN-Documentary-Footage/1.0"
_DEFAULT_MAX_SEARCHES = 10
_DEFAULT_MAX_DOWNLOADS = 8
_DEFAULT_MAX_FILE_MB = 100
_DEFAULT_RESULTS_PER_QUERY = 12
_FIELDS = ("identifier", "title", "description", "creator", "licenseurl", "subject", "format", "mediatype")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _as_text(value: object) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")


def license_from_url(value: object) -> str | None:
    """Map only clearly reusable licenses to a narrow canonical label."""
    url = str(value or "").strip().casefold().rstrip("/")
    if not url.startswith(("https://creativecommons.org/", "http://creativecommons.org/")):
        return None
    if "/publicdomain/zero/1.0" in url:
        return "CC0"
    if "/publicdomain/mark/1.0" in url:
        return "Public Domain Mark"
    match = re.search(r"/licenses/by/(3\.0|4\.0)(?:/|$)", url)
    if match and "/by-nc" not in url and "/by-nd" not in url and "/by-sa" not in url:
        return f"CC BY {match.group(1)}"
    return None


def _terms_for_scene(scene: dict) -> list[str]:
    """Take high-signal words in title/labels first, then the narration."""
    values: list[str] = []
    labels = scene.get("labels") or scene.get("phrases") or []
    if not isinstance(labels, list):
        labels = [labels]
    for value in (
        scene.get("title", ""),
        *labels,
        scene.get("narration_excerpt", ""),
        scene.get("narration_anchor", ""),
        scene.get("visual_intent", ""),
    ):
        for word in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
            if word in _tokens(word) and word not in values:
                values.append(word)
    return values[:5]


def _search_query(terms: list[str]) -> str:
    # Terms are restricted to lowercase alphanumerics before being inserted
    # into the Archive.org Lucene query.
    clauses = []
    for term in terms:
        safe = re.sub(r"[^a-z0-9]", "", term.casefold())
        if safe:
            clauses.extend((f'title:"{safe}"', f'description:"{safe}"'))
    return "mediatype:movies AND (" + " OR ".join(clauses) + ")" if clauses else ""


def _search_archive(query: str, rows: int) -> list[dict]:
    params = [
        ("q", query), ("rows", rows), ("page", 1), ("output", "json"),
        *[("fl[]", field) for field in _FIELDS],
    ]
    response = requests.get(
        _SEARCH_URL, params=params, timeout=(10, 25),
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    body = response.json()
    docs = body.get("response", {}).get("docs", [])
    return [doc for doc in docs if isinstance(doc, dict)]


def _media_files(metadata: dict, max_bytes: int) -> list[dict]:
    files = metadata.get("files")
    if not isinstance(files, list):
        return []
    usable = []
    for item in files:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or Path(name).name != name or name in {".", ".."}:
            continue
        if not name.casefold().endswith(".mp4"):
            continue
        if str(item.get("private") or "").casefold() in {"true", "1", "yes"}:
            continue
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        if size < 0 or (size and size > max_bytes):
            continue
        # A file named .mp4 is not enough if the catalog says it is not a video.
        fmt = _as_text(item.get("format")).casefold()
        filetype = str(item.get("filetype") or "").casefold()
        if fmt and not any(token in fmt for token in ("mpeg4", "mpeg-4", "mp4", "h.264", "h264")):
            continue
        if filetype and not ("video" in filetype or "mp4" in filetype):
            continue
        usable.append({**item, "name": name, "_size": size})
    # Prefer the highest-bitrate file within our bounded download size.
    usable.sort(key=lambda item: (item["_size"] or 0), reverse=True)
    return usable


def _probe_duration(path: Path) -> float | None:
    try:
        from moduli.ffmpeg_utils import ffprobe_path
        result = subprocess.run(
            [
                ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=20,
            stdin=subprocess.DEVNULL, check=True,
        )
        duration = float(result.stdout.strip())
        return duration if duration >= 4.0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _download_file(identifier: str, file_name: str, dest: Path, max_bytes: int) -> bool:
    url = f"https://archive.org/download/{quote(identifier, safe='')}/{quote(file_name, safe='')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1024:
        return _probe_duration(dest) is not None

    temp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(
            url, stream=True, timeout=(10, 60),
            headers={"User-Agent": _USER_AGENT},
        ) as response:
            response.raise_for_status()
            host = (urlparse(response.url).hostname or "").casefold()
            if host != "archive.org" and not host.endswith(".archive.org"):
                raise RuntimeError("download redirected outside archive.org")
            content_type = str(response.headers.get("content-type") or "").casefold()
            if content_type and "video" not in content_type and "octet-stream" not in content_type:
                raise RuntimeError(f"unexpected media content type: {content_type}")
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise RuntimeError("archive media exceeds configured download limit")
            downloaded = 0
            with temp.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 128):
                    if not block:
                        continue
                    downloaded += len(block)
                    if downloaded > max_bytes:
                        raise RuntimeError("archive media exceeded configured download limit")
                    handle.write(block)
        if temp.stat().st_size < 1024 or _probe_duration(temp) is None:
            raise RuntimeError("downloaded file is empty, invalid, or not a playable video")
        os.replace(temp, dest)
        return True
    except Exception as exc:
        print(f"[footage-discovery] skipped {identifier}/{file_name}: {type(exc).__name__}: {exc}", flush=True)
        temp.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        return False


def _scene_indices(scene_count: int, limit: int) -> list[int]:
    if scene_count <= 0:
        return []
    if scene_count <= limit:
        return list(range(scene_count))
    if limit == 1:
        return [0]
    return list(dict.fromkeys(round(i * (scene_count - 1) / (limit - 1)) for i in range(limit)))


def discover_archive_footage(
    scenes: list[dict],
    script: str,
    manifest_path: str | Path,
) -> dict:
    """Search a bounded set of scene queries and atomically write a script-bound manifest.

    No media is accepted based on search ranking alone. Results need an explicit
    approved reuse license from Archive.org metadata, a strong title/description
    match, a downloadable MP4, a playable local file, and creator attribution
    where the license requires it.
    """
    out = Path(manifest_path).expanduser().resolve()
    max_searches = _env_int("FERN_ARCHIVE_MAX_SEARCHES", _DEFAULT_MAX_SEARCHES, 1, 20)
    max_downloads = _env_int("FERN_ARCHIVE_MAX_DOWNLOADS", _DEFAULT_MAX_DOWNLOADS, 1, 16)
    max_bytes = _env_int("FERN_ARCHIVE_MAX_DOWNLOAD_MB", _DEFAULT_MAX_FILE_MB, 20, 250) * 1024 * 1024
    rows = _env_int("FERN_ARCHIVE_RESULTS_PER_QUERY", _DEFAULT_RESULTS_PER_QUERY, 4, 20)
    index_dir = out.parent.parent / "cache" / "fern_archive"
    indexed: dict[str, dict] = {}
    queries_run = 0
    errors: list[str] = []
    rejections = {"license": 0, "relevance": 0, "metadata": 0, "media_file": 0, "download": 0}

    queries: list[tuple[int, dict, list[str]]] = []
    seen_queries = set()
    for idx in _scene_indices(len(scenes), max_searches):
        scene = scenes[idx]
        if not isinstance(scene, dict):
            continue
        terms = _terms_for_scene(scene)
        query = _search_query(terms)
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        queries.append((idx, scene, terms))

    found: list[tuple[dict, dict, int]] = []
    seen_ids = set()
    for scene_index, scene, terms in queries:
        query = _search_query(terms)
        try:
            docs = _search_archive(query, rows)
            queries_run += 1
        except Exception as exc:
            errors.append(f"query scene {scene.get('id', scene_index)}: {type(exc).__name__}: {exc}")
            continue
        search_text = _scene_query(scene)
        for doc in docs:
            identifier = str(doc.get("identifier") or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", identifier):
                continue
            if identifier in seen_ids:
                continue
            title = _as_text(doc.get("title"))
            description = _as_text(doc.get("description"))
            license_url = _as_text(doc.get("licenseurl"))
            license_name = license_from_url(license_url)
            if not license_name:
                rejections["license"] += 1
                continue
            subject = _as_text(doc.get("subject"))
            candidate_for_score = {
                "title": title, "description": description, "tags": subject.split(),
            }
            _score, _required, _matched, reason = _relevance(search_text, candidate_for_score)
            if reason != "relevant":
                rejections["relevance"] += 1
                continue
            seen_ids.add(identifier)
            found.append((doc, scene, scene_index))
            if len(found) >= max_downloads * 3:
                break
        if len(found) >= max_downloads * 3:
            break

    candidates = []
    for doc, scene, scene_index in found:
        if len(candidates) >= max_downloads:
            break
        identifier = str(doc.get("identifier") or "").strip()
        try:
            response = requests.get(
                f"{_METADATA_URL}/{quote(identifier, safe='')}",
                timeout=(10, 25), headers={"User-Agent": _USER_AGENT},
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            rejections["metadata"] += 1
            errors.append(f"metadata {identifier}: {type(exc).__name__}: {exc}")
            continue
        metadata = body.get("metadata") if isinstance(body, dict) else None
        if not isinstance(metadata, dict):
            rejections["metadata"] += 1
            continue

        license_url = _as_text(metadata.get("licenseurl") or doc.get("licenseurl"))
        license_name = license_from_url(license_url)
        if not license_name:
            rejections["license"] += 1
            continue
        files = _media_files(body, max_bytes)
        if not files:
            rejections["media_file"] += 1
            continue
        file = files[0]
        file_name = str(file["name"])
        digest = hashlib.sha256(f"{identifier}/{file_name}".encode("utf-8")).hexdigest()[:20]
        local_path = index_dir / f"ia_{digest}.mp4"
        if not _download_file(identifier, file_name, local_path, max_bytes):
            rejections["download"] += 1
            continue
        duration = _probe_duration(local_path)
        try:
            min_scene_seconds = max(
                4.0,
                min(30.0, float(os.environ.get("FERN_CHALK_BEAT_SECONDS", "7.5"))),
            )
        except (TypeError, ValueError):
            min_scene_seconds = 7.5
        if duration is None or duration < min_scene_seconds:
            rejections["media_file"] += 1
            local_path.unlink(missing_ok=True)
            continue
        # Prefer the middle of the clip, not a likely intro/logo frame at 0s.
        # The scene renderer consumes frames continuously and falls back to chalk
        # if a source runs out; the minimum-duration check reduces mid-scene cuts.
        source_start_seconds = max(0.0, (duration - min_scene_seconds) / 2.0)

        title = _as_text(metadata.get("title") or doc.get("title"))
        description = _as_text(metadata.get("description") or doc.get("description"))
        creator = _as_text(metadata.get("creator") or metadata.get("uploader") or doc.get("creator"))
        if license_name.startswith("CC BY") and not creator.strip():
            rejections["metadata"] += 1
            local_path.unlink(missing_ok=True)
            continue
        subjects = _as_text(metadata.get("subject") or doc.get("subject"))
        candidates.append({
            "title": title,
            "description": description,
            "tags": subjects.split(),
            "channel_title": creator,
            "creator": creator,
            "attribution": creator or "Public-domain archival source",
            "source_url": f"https://archive.org/details/{identifier}",
            "source_type": "licensed_archive_video",
            "source_verified": True,
            "verification_method": "public_archive_catalog",
            "verification_evidence_url": f"{_METADATA_URL}/{identifier}",
            "license": license_name,
            "license_url": license_url,
            "rights_documentation": f"Archive.org item metadata: {_METADATA_URL}/{identifier}",
            "path": str(local_path.resolve()),
            "video_id": identifier,
            "review_status": "approved",
            "approval_policy": "archive-license-topic-gate-v1",
            "source_start_seconds": round(source_start_seconds, 3),
        })
        indexed[identifier] = {
            "title": title, "source_url": f"https://archive.org/details/{identifier}",
            "license": license_name, "local_path": str(local_path.resolve()),
            "scene_hint": str(scene.get("id") or scene_index),
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "script_sha256": script_fingerprint(script),
        "script_file": "",
        "candidate_export": "automatic Internet Archive search",
        "policy": {
            "provider": "Internet Archive metadata API",
            "allowed_licenses": ["CC0", "Public Domain Mark", "CC BY 3.0", "CC BY 4.0"],
            "require_title_relevance": True,
            "max_searches": max_searches,
            "max_downloads": max_downloads,
            "fallback": "chalk",
        },
        "candidates": candidates,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, out)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
    return {
        "status": "complete" if candidates else "no_suitable_results",
        "queries_run": queries_run,
        "scenes_considered": len(queries),
        "candidates_downloaded": len(candidates),
        "candidates": indexed,
        "rejections": rejections,
        "errors": errors[:20],
        "manifest": str(out),
    }
