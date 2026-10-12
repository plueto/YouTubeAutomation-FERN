"""Acquire licensed Pixabay video candidates for the existing FERN preview plan."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "output" / "cinematic_preview_live.plan.json"
OUTPUT_DIR = ROOT / "cache" / "pixabay_preview"
SEARCH_CACHE_PATH = OUTPUT_DIR / "search_cache.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
API_URL = "https://pixabay.com/api/videos/"
LICENSE_URL = "https://pixabay.com/service/license-summary/"
MAX_UNIQUE_QUERIES = 8
PIXABAY_MIN_PER_PAGE = 3
MAX_CANDIDATES_PER_QUERY = 2
SEARCH_CACHE_SECONDS = 24 * 60 * 60
MAX_DOWNLOAD_BYTES = 250 * 1024 * 1024
REQUEST_TIMEOUT = (10, 40)
DOWNLOAD_TIMEOUT = (10, 120)
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "before",
    "being", "but", "for", "from", "have", "into", "its", "not", "of",
    "on", "or", "our", "over", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "to",
    "under", "was", "were", "what", "when", "where", "which", "while",
    "with", "you", "your",
}


def _sanitize_error_body(body: str, api_key: str, limit: int = 320) -> str:
    text = str(body or "")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(
        r"(?i)([?&]key=)[^&\s\"']+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)([\"']?key[\"']?\s*[:=]\s*[\"']?)[^\s,}\"']+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def _http_error_detail(error: requests.RequestException, api_key: str) -> str:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if response is not None:
        try:
            body = response.text
        except Exception:
            body = ""
    else:
        body = ""
    detail = f"HTTP {status}" if status is not None else "HTTP status unavailable"
    sanitized = _sanitize_error_body(body, api_key)
    return f"{detail}: {sanitized}" if sanitized else detail


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(word) > 2 and word not in STOPWORDS
    }


def _read_shots(plan_path: Path) -> tuple[list[dict], list[dict]]:
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Shot plan not found: {plan_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Shot plan is not valid JSON: {plan_path}") from error

    shots = plan.get("shots") if isinstance(plan, dict) else None
    if not isinstance(shots, list) or not shots:
        raise ValueError("Shot plan must contain a non-empty 'shots' list")

    queries: list[dict] = []
    by_query: dict[str, dict] = {}
    shot_records = []

    def add_query(query: str, shot_record: dict) -> None:
        query = " ".join(str(query or "").split())
        if not query or len(queries) >= MAX_UNIQUE_QUERIES:
            return
        key = query.casefold()
        record = by_query.get(key)
        if record is None:
            record = {"query": query, "shots": []}
            by_query[key] = record
            queries.append(record)
        if all(existing["shot_id"] != shot_record["shot_id"] for existing in record["shots"]):
            record["shots"].append(shot_record)

    for index, shot in enumerate(shots, 1):
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("id") or f"shot-{index:02d}")
        shot_record = {
            "shot_id": shot_id,
            "narration_anchor": str(shot.get("narration_anchor") or ""),
            "visual_intent": str(shot.get("visual_intent") or ""),
            "shot_type": str(shot.get("shot_type") or ""),
            "primary_query": " ".join(str(shot.get("primary_query") or "").split()),
        }
        shot_records.append(shot_record)
        add_query(shot.get("primary_query"), shot_record)
        alternates = shot.get("alternate_queries")
        if isinstance(alternates, list):
            for alternate in alternates:
                add_query(str(alternate), shot_record)

    attempted = plan.get("queries_attempted", []) if isinstance(plan, dict) else []
    if isinstance(attempted, list) and len(queries) < MAX_UNIQUE_QUERIES:
        for historical_query in attempted:
            normalized = " ".join(str(historical_query or "").split())
            if not normalized or normalized.casefold() in by_query:
                continue
            query_terms = _tokens(normalized)
            fallback_shot = {
                "shot_id": f"plan-query-{len(queries) + 1:02d}",
                "narration_anchor": "",
                "visual_intent": "",
                "shot_type": "",
                "primary_query": "",
            }

            def association_score(shot_record: dict) -> int:
                context = " ".join((
                    shot_record["primary_query"],
                    shot_record["narration_anchor"],
                    shot_record["visual_intent"],
                    shot_record["shot_type"],
                ))
                return len(query_terms & _tokens(context))

            best_shot = max(shot_records, key=association_score, default=fallback_shot)
            add_query(normalized, best_shot)

    if not queries:
        raise ValueError("Shot plan contains no usable primary_query values")
    return shots, queries


def _load_search_cache(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _search_pixabay(session: requests.Session, api_key: str, query: str,
                    cache: dict, now: float) -> tuple[list[dict], bool]:
    cache_key = query.casefold()
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and now - float(cached.get("fetched_at", 0)) < SEARCH_CACHE_SECONDS:
        hits = cached.get("hits")
        if isinstance(hits, list):
            return hits[:MAX_CANDIDATES_PER_QUERY], True

    try:
        response = session.get(
            API_URL,
            params={
                "key": api_key,
                "q": query,
                "per_page": max(PIXABAY_MIN_PER_PAGE, MAX_CANDIDATES_PER_QUERY),
                "safesearch": "true",
                "video_type": "all",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as error:
        raise RuntimeError(f"Pixabay search timed out for query {query!r}") from error
    except requests.RequestException as error:
        detail = _http_error_detail(error, api_key)
        raise RuntimeError(f"Pixabay search failed for query {query!r}: {detail}") from error
    except ValueError as error:
        raise RuntimeError(f"Pixabay returned invalid JSON for query {query!r}") from error

    hits = payload.get("hits") if isinstance(payload, dict) else None
    if not isinstance(hits, list):
        raise RuntimeError(f"Pixabay response has no results list for query {query!r}")
    hits = hits[:MAX_CANDIDATES_PER_QUERY]
    cache[cache_key] = {"query": query, "fetched_at": now, "hits": hits}
    return hits, False


def _video_variant(hit: dict) -> dict | None:
    variants = hit.get("videos")
    if not isinstance(variants, dict):
        return None
    usable = []
    for name, variant in variants.items():
        if not isinstance(variant, dict):
            continue
        url = str(variant.get("url") or "")
        parsed_path = urlparse(url).path.casefold()
        width = int(variant.get("width") or 0)
        height = int(variant.get("height") or 0)
        if (not url.startswith("https://") or not parsed_path.endswith(".mp4")
                or width < 640 or height < 360 or width < height):
            continue
        usable.append({
            "variant": name,
            "url": url,
            "width": width,
            "height": height,
            "size": int(variant.get("size") or 0),
        })
    if not usable:
        return None
    usable.sort(key=lambda item: (
        item["width"] >= 1280,
        min(item["width"], 1920),
        -item["size"] if item["size"] else 0,
    ), reverse=True)
    return usable[0]


def _rank_hits(hits: list[dict], query: str, shots: list[dict]) -> list[dict]:
    query_terms = _tokens(query)
    ranked = []
    seen_ids = set()
    for hit in hits[:MAX_CANDIDATES_PER_QUERY]:
        if not isinstance(hit, dict) or not hit.get("id"):
            continue
        video_id = int(hit["id"])
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        variant = _video_variant(hit)
        if not variant:
            continue
        tags = str(hit.get("tags") or "")
        title = str(hit.get("title") or "")
        title_terms = _tokens(title)
        tag_terms = _tokens(tags)
        metadata_terms = title_terms | tag_terms
        query_score = (
            5 * len(query_terms & title_terms)
            + 2 * len(query_terms & tag_terms)
        )
        shot_scores = []
        for shot in shots:
            anchor_terms = _tokens(str(shot.get("narration_anchor") or ""))
            intent_terms = _tokens(str(shot.get("visual_intent") or ""))
            type_terms = _tokens(str(shot.get("shot_type") or ""))
            shot_query_terms = _tokens(str(shot.get("primary_query") or ""))
            shot_score = (
                3 * len(anchor_terms & metadata_terms)
                + 4 * len(intent_terms & metadata_terms)
                + len(type_terms & metadata_terms)
                + 2 * len(shot_query_terms & metadata_terms)
            )
            shot_scores.append((shot_score, shot))
        shot_scores.sort(key=lambda item: (-item[0], str(item[1].get("shot_id") or "")))
        ranked.append({
            "hit": hit,
            "variant": variant,
            "relevance_score": query_score + (shot_scores[0][0] if shot_scores else 0),
            "shot_scores": shot_scores,
        })
    ranked.sort(key=lambda item: (
        -item["relevance_score"],
        -item["variant"]["width"],
        int(item["hit"].get("id") or 0),
    ))

    assigned_shots = set()
    for candidate in ranked:
        available = [item for item in candidate["shot_scores"]
                     if str(item[1].get("shot_id") or "") not in assigned_shots]
        choices = available or candidate["shot_scores"]
        if choices:
            best_score = choices[0][0]
            best = next((item for item in choices if item[0] == best_score), choices[0])
            candidate["matched_shots"] = [best[1]]
            assigned_shots.add(str(best[1].get("shot_id") or ""))
        else:
            candidate["matched_shots"] = []
    return ranked


def _download_mp4(session: requests.Session, url: str, destination: Path) -> int:
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    try:
        with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
            if content_type and content_type not in {"video/mp4", "application/octet-stream"}:
                raise RuntimeError(f"Unexpected video content type: {content_type}")
            length = response.headers.get("content-length")
            if length and int(length) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Selected Pixabay MP4 exceeds the 250 MB download limit")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_BYTES:
                        raise RuntimeError("Selected Pixabay MP4 exceeded the 250 MB download limit")
                    handle.write(chunk)
        if downloaded < 1024:
            raise RuntimeError("Selected Pixabay MP4 download was unexpectedly small")
        os.replace(temporary, destination)
        return downloaded
    except requests.Timeout as error:
        raise RuntimeError("Pixabay video download timed out") from error
    except requests.RequestException as error:
        detail = _http_error_detail(error, "")
        raise RuntimeError(f"Pixabay video download failed: {detail}") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def acquire_pixabay_videos(
    plan_path: Path = PLAN_PATH,
    output_dir: Path = OUTPUT_DIR,
    max_queries: int = MAX_UNIQUE_QUERIES,
) -> dict:
    """Download relevant real MP4 videos for up to eight unique shot queries."""
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    api_key = os.environ.get("PIXABAY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(f"PIXABAY_API_KEY is missing from {ROOT / '.env'}")

    _, query_records = _read_shots(Path(plan_path))
    query_records = query_records[:max(0, min(MAX_UNIQUE_QUERIES, int(max_queries)))]
    if not query_records:
        raise ValueError("No unique visual queries are available in the shot plan")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _load_search_cache(output_dir / "search_cache.json")
    session = requests.Session()
    selected_by_id: dict[int, dict] = {}
    errors = []
    search_count = 0
    cache_hits = 0
    downloaded_files: set[str] = set()

    try:
        for record in query_records:
            query = record["query"]
            try:
                cached_record = cache.get(query.casefold())
                cached_fresh = (
                    isinstance(cached_record, dict)
                    and isinstance(cached_record.get("hits"), list)
                    and time.time() - float(cached_record.get("fetched_at", 0)) < SEARCH_CACHE_SECONDS
                )
                if not cached_fresh:
                    search_count += 1
                hits, was_cached = _search_pixabay(session, api_key, query, cache, time.time())
                if was_cached:
                    cache_hits += 1
                ranked = _rank_hits(hits, query, record["shots"])
                if not ranked:
                    errors.append({"query": query, "error": "No usable landscape MP4 among the two results"})
                    continue

                for candidate in ranked[:MAX_CANDIDATES_PER_QUERY]:
                    hit = candidate["hit"]
                    video_id = int(hit["id"])
                    matched_shots = candidate.get("matched_shots", [])
                    if video_id in selected_by_id:
                        existing = selected_by_id[video_id]
                        if query not in existing["queries"]:
                            existing["queries"].append(query)
                        seen_shots = {shot["shot_id"] for shot in existing["matched_shots"]}
                        for shot in matched_shots:
                            if shot["shot_id"] not in seen_shots:
                                existing["matched_shots"].append(shot)
                                seen_shots.add(shot["shot_id"])
                        existing["shot_ids"] = [shot["shot_id"] for shot in existing["matched_shots"]]
                        continue
                    variant = candidate["variant"]
                    destination = output_dir / f"pixabay_{video_id}.mp4"
                    try:
                        if destination.is_file() and destination.stat().st_size > 1024:
                            file_size = destination.stat().st_size
                        else:
                            file_size = _download_mp4(session, variant["url"], destination)
                    except (OSError, RuntimeError) as error:
                        errors.append({"query": query, "video_id": video_id,
                                       "error": str(error)})
                        continue

                    resolved_path = str(destination.resolve())
                    if resolved_path in downloaded_files:
                        continue
                    downloaded_files.add(resolved_path)
                    selected_by_id[video_id] = {
                        "shot_ids": [shot["shot_id"] for shot in matched_shots],
                        "matched_shots": list(matched_shots),
                        "queries": [query],
                        "pixabay_video_id": video_id,
                        "title": str(hit.get("tags") or hit.get("title") or "Untitled Pixabay video"),
                        "uploader": str(hit.get("user") or ""),
                        "source_page_url": str(hit.get("pageURL") or ""),
                        "download_path": resolved_path,
                        "duration_seconds": float(hit.get("duration") or 0),
                        "resolution": {"width": variant["width"], "height": variant["height"]},
                        "selected_variant": variant["variant"],
                        "relevance_score": candidate["relevance_score"],
                        "file_size_bytes": file_size,
                        "license": hit.get("license") or None,
                        "license_url": hit.get("licenseUrl") or LICENSE_URL,
                        "license_note": (
                            "Per-item license was not included in the API response; "
                            "verify the source page and Pixabay license before publication."
                            if not hit.get("license") else ""
                        ),
                        "provenance": "Pixabay official video API",
                        "source_video_url": variant["url"],
                    }
            except (RuntimeError, ValueError, requests.RequestException) as error:
                errors.append({"query": query, "error": str(error)})
            finally:
                _atomic_json(output_dir / "search_cache.json", cache)

    finally:
        session.close()

    manifest = {
        "plan_path": str(Path(plan_path).resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_count": len(query_records),
        "selected_queries": [record["query"] for record in query_records],
        "search_requests_made": search_count,
        "cached_searches_reused": cache_hits,
        "max_candidates_per_query": MAX_CANDIDATES_PER_QUERY,
        "clips": list(selected_by_id.values()),
        "errors": errors,
    }
    _atomic_json(output_dir / "manifest.json", manifest)

    count = len(manifest["clips"])
    if count < 6:
        print(f"Shortage: acquired {count} distinct usable Pixabay videos; 6 were requested.")
    print(f"Unique queries: {len(query_records)}; new Pixabay searches: {search_count}; "
          f"cached searches reused: {cache_hits}; distinct video files: {count}.")
    for error in errors:
        print(f"Query issue ({error['query']}): {error['error']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire up to two relevant Pixabay MP4s for each shot-plan query."
    )
    parser.add_argument("--plan", type=Path, default=PLAN_PATH,
                        help="Shot-plan JSON (default: output/cinematic_preview_live.plan.json)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Cache and manifest directory (default: cache/pixabay_preview)")
    args = parser.parse_args()
    try:
        acquire_pixabay_videos(plan_path=args.plan, output_dir=args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Pixabay acquisition stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())