"""Quality gate for mixing verified, licensed footage into FERN's chalk scene timeline.

The selector is deliberately conservative: an unsuitable or unlicensed result is
not substituted with a random clip. The corresponding scene stays chalk-only.
No network access or downloads happen in this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "fern-footage-candidates-v1"

_STOPWORDS = set("""
a an and are as at be been being but by can could did do does for from had has have
he her hers him his how i if in into is it its me more most my no not of on or our
ours she so than that the their theirs them then there these they this those through
to too under up us was we were what when where which while who why will with would you
your yours also about after before because over only some such very just now still
""".split())
_GENERIC_TERMS = {
    "video", "footage", "clip", "scene", "shot", "story", "explained", "explainer",
    "documentary", "technology", "tech", "future", "digital", "ai", "artificial",
    "intelligence", "news", "latest", "official", "real", "actual", "interview",
    "company", "people", "thing", "things", "world", "today",
}

# These describe the evidence being shown, not just a channel's editorial opinion.
_PRIMARY_SOURCE_TYPES = {
    "official_original", "official_interview", "interview", "press_conference",
    "product_demo", "event_recording", "court_hearing", "government_record",
    "public_domain_archive", "original_event", "licensed_primary_source",
}
_RELIABLE_REPORTING_TYPES = {
    "news_original_reporting", "university_primary", "licensed_stock",
}
_DISALLOWED_TYPES = {
    "ai_summary", "ai_storytelling", "storytelling", "commentary", "reaction",
    "recap", "compilation", "opinion", "unknown",
}


def script_fingerprint(script: str) -> str:
    """Stable SHA-256 for a whitespace-normalized narration script."""
    normalized = " ".join(str(script or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) >= 3 and token not in _STOPWORDS and token not in _GENERIC_TERMS
    }


def _source_type(candidate: dict[str, Any]) -> str:
    return _norm(candidate.get("source_type") or candidate.get("source_kind")).replace(" ", "_")


def _rights_status(candidate: dict[str, Any]) -> tuple[bool, str]:
    """Require a reusable license or explicitly documented permission.

    Being public on YouTube, being from a credible publisher, or appearing in
    search results does not by itself grant reuse rights.
    """
    license_text = _norm(candidate.get("license") or candidate.get("license_name"))
    license_compact = license_text.replace(" ", "")
    if candidate.get("rights_confirmed") is True:
        if candidate.get("rights_documentation"):
            return True, "documented permission/rights confirmation"
        return False, "rights_confirmed requires a rights_documentation record"
    if not license_text:
        return False, "missing license/permission metadata"
    if any(term in license_text for term in ("non commercial", "noncommercial", "no derivatives", "no derivative")):
        return False, f"license is not suitable for monetized/editing use: {license_text}"
    if license_compact in {"cc0", "publicdomain", "publicdomainmark", "creativecommonszero"}:
        return True, f"license: {license_text}"
    if license_compact in {
        "ccby", "ccby40", "ccby3", "ccby30", "creativecommonsattribution",
        "ccby-sa", "ccbysa", "ccbysa40", "ccbysa30",
        "creativecommonsattributionsharealike", "creativecommons", "creativecommon",
    } or license_text.startswith("creative commons attribution"):
        has_credit = bool(str(candidate.get("attribution") or candidate.get("credit") or candidate.get("channel_title") or "").strip())
        if "sharealike" in license_compact or "share alike" in license_text:
            if candidate.get("sharealike_compliance_confirmed") is not True:
                return False, "CC BY-SA share-alike obligations require explicit compliance confirmation"
        if license_compact.startswith("ccbysa") and not has_credit:
            return False, "CC BY-SA requires attribution metadata"
        if (license_compact.startswith("ccby") or license_text.startswith("creative commons attribution")) and not has_credit:
            return False, "CC BY requires attribution metadata"
        return True, f"license: {license_text}"
    if license_text in {"licensed", "permission granted", "used with permission", "owned by us"}:
        return (bool(candidate.get("rights_documentation")), "documented license" if candidate.get("rights_documentation") else "license label lacks a rights record")
    return False, f"unsupported or unclear license: {license_text}"


def _scene_query(scene: dict[str, Any]) -> str:
    explicit = str(scene.get("footage_query") or "").strip()
    if explicit:
        return explicit
    labels = scene.get("labels") or scene.get("phrases") or []
    if not isinstance(labels, list):
        labels = [labels]
    title = str(scene.get("title") or "")
    excerpt = str(scene.get("narration_excerpt") or scene.get("narration_anchor") or "")
    visual_intent = str(scene.get("visual_intent") or "")
    return " ".join([title, " ".join(map(str, labels)), visual_intent, excerpt])


def _relevance(query: str, candidate: dict[str, Any]) -> tuple[int, int, set[str], str]:
    query_terms = _tokens(query)
    title = str(candidate.get("title") or "")
    description = str(candidate.get("description") or "")
    tags = candidate.get("tags") or []
    searchable = " ".join([title, description, " ".join(map(str, tags))])
    matched = query_terms & _tokens(searchable)
    title_matched = query_terms & _tokens(title)
    required = 1 if len(query_terms) <= 2 else max(2, min(4, math.ceil(len(query_terms) * 0.22)))
    score = len(title_matched) * 4 + len(matched - title_matched)
    query_norm, title_norm, searchable_norm = _norm(query), _norm(title), _norm(searchable)
    if query_norm and (query_norm in title_norm or query_norm in searchable_norm):
        score += 5
    if len(query_terms) <= 1 and query_terms and not title_matched:
        return score, required, matched, "specific topic absent from title"
    if len(matched) < required:
        return score, required, matched, f"weak topic match ({len(matched)}/{required} key terms)"
    if len(query_terms) > 2 and not title_matched:
        return score, required, matched, "no key topic term in title"
    return score, required, matched, "relevant"


def _source_tier(candidate: dict[str, Any]) -> tuple[int, str]:
    kind = _source_type(candidate)
    if kind in _DISALLOWED_TYPES or any(x in kind for x in ("storytelling", "commentary", "reaction", "recap", "compilation", "opinion", "ai_summary")):
        return 0, f"rejected source type: {kind or 'unknown'}"
    verification_method = _norm(candidate.get("verification_method")).replace(" ", "_")
    approved_methods = {
        "official_channel", "official_domain", "verified_uploader",
        "publisher_catalog", "public_archive_catalog", "government_domain",
        "university_domain", "licensed_stock_catalog",
    }
    if candidate.get("source_verified") is not True or verification_method not in approved_methods:
        return 0, "source is not independently verified by an approved method"
    if kind in _PRIMARY_SOURCE_TYPES:
        return 3, f"verified primary/original footage: {kind} ({verification_method})"
    if kind in _RELIABLE_REPORTING_TYPES:
        return 2, f"verified licensed reporting/stock: {kind} ({verification_method})"
    return 0, f"unverified source type: {kind or 'unknown'}"


def _resolve_candidate_path(candidate: dict[str, Any], manifest_path: Path) -> Path | None:
    raw = str(candidate.get("path") or candidate.get("local_path") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        return resolved if resolved.is_file() else None
    options = [manifest_path.parent / path, Path.cwd() / path, manifest_path.parent.parent / path]
    for option in options:
        try:
            resolved = option.resolve()
            if resolved.is_file():
                return resolved
        except OSError:
            continue
    return None


def _candidate_identity(candidate: dict[str, Any], path: Path) -> str:
    return str(candidate.get("video_id") or candidate.get("id") or candidate.get("source_url") or candidate.get("url") or path).strip().casefold()


def select_footage_for_scenes(
    scenes: list[dict[str, Any]], candidates: list[dict[str, Any]], manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign unique, relevant, rights-cleared original clips; leave other beats chalk-only."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    usable: list[tuple[dict[str, Any], Path, int, str, str]] = []
    rejected: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            rejected.append({"candidate": str(index), "reason": "candidate is not an object"})
            continue
        rights_ok, rights_reason = _rights_status(candidate)
        tier, source_reason = _source_tier(candidate)
        path = _resolve_candidate_path(candidate, manifest_file)
        source_url = str(candidate.get("source_url") or candidate.get("url") or "").strip()
        review_status = str(candidate.get("review_status") or "").strip().casefold()
        if not rights_ok:
            reason = rights_reason
        elif not tier:
            reason = source_reason
        elif review_status != "approved":
            reason = "candidate has not passed explicit review_status=approved"
        elif not source_url:
            reason = "missing source URL/provenance"
        elif path is None:
            reason = "local clip path missing or not a file"
        else:
            usable.append((candidate, path, tier, rights_reason, _candidate_identity(candidate, path)))
            continue
        rejected.append({"candidate": str(candidate.get("title") or index), "reason": reason})

    used: set[str] = set()
    output: list[dict[str, Any]] = []
    selected_count = 0
    decisions: list[dict[str, Any]] = []
    for index, original in enumerate(scenes):
        row = dict(original)
        row.pop("footage", None)
        query = _scene_query(row)
        ranked = []
        scene_rejections = []
        for candidate, path, tier, rights_reason, identity in usable:
            if identity in used:
                continue
            score, required, matched, reason = _relevance(query, candidate)
            if reason != "relevant":
                scene_rejections.append({"title": str(candidate.get("title") or ""), "reason": reason})
                continue
            total = score + tier * 6
            ranked.append((total, len(matched), -len(str(candidate.get("title") or "")), candidate, path, tier, rights_reason, identity, matched))
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1], item[2], str(item[3].get("title") or "").casefold()), reverse=True)
            _total, _matched_count, _title_len, candidate, path, tier, rights_reason, identity, matched = ranked[0]
            used.add(identity)
            row["footage"] = {
                "path": str(path),
                "title": str(candidate.get("title") or ""),
                "channel": str(candidate.get("channel_title") or candidate.get("channel") or ""),
                "source_url": str(candidate.get("source_url") or candidate.get("url") or ""),
                "source_type": _source_type(candidate),
                "license": str(candidate.get("license") or candidate.get("license_name") or rights_reason),
                "attribution": str(candidate.get("attribution") or candidate.get("credit") or candidate.get("channel_title") or ""),
                "source_start_seconds": max(0.0, float(candidate.get("source_start_seconds") or 0.0)),
                "relevance_terms": sorted(matched),
                "source_tier": tier,
            }
            selected_count += 1
            decisions.append({"scene": str(row.get("id") or index), "decision": "footage", "title": row["footage"]["title"], "matched_terms": sorted(matched)})
        else:
            row["footage_decision"] = "chalk_fallback_no_suitable_verified_clip"
            decisions.append({
                "scene": str(row.get("id") or index), "decision": "chalk_fallback",
                "reason": "no candidate passed topic, source-type, local-file and rights checks",
                "candidate_rejections": scene_rejections[:8],
            })
        output.append(row)

    report = {
        "schema": MANIFEST_SCHEMA,
        "scenes_total": len(output),
        "scenes_with_footage": selected_count,
        "scenes_chalk_fallback": len(output) - selected_count,
        "candidates_seen": len(candidates),
        "candidates_usable_before_relevance": len(usable),
        "candidates_rejected": rejected[:100],
        "decisions": decisions,
    }
    return output, report


def apply_footage_manifest(plan: dict[str, Any], script: str, manifest_path: str | Path) -> dict[str, Any]:
    """Apply the manifest only when it was created for this exact script."""
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        plan["footage_report"] = {"status": "no_manifest", "message": "No footage candidate manifest; all scenes remain chalk-only."}
        return plan
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        plan["footage_report"] = {"status": "invalid_manifest", "message": str(exc)}
        return plan
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        plan["footage_report"] = {"status": "invalid_manifest", "message": f"Expected schema {MANIFEST_SCHEMA}."}
        return plan
    if manifest.get("script_sha256") != script_fingerprint(script):
        plan["footage_report"] = {"status": "stale_manifest", "message": "Footage manifest belongs to a different narration script; ignored."}
        return plan
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        plan["footage_report"] = {"status": "invalid_manifest", "message": "Manifest candidates must be a list."}
        return plan
    scenes = plan.get("scenes")
    if not isinstance(scenes, list):
        plan["footage_report"] = {"status": "invalid_plan", "message": "Scene plan has no scenes list."}
        return plan
    selected, report = select_footage_for_scenes(scenes, candidates, path)
    plan["scenes"] = selected
    report["status"] = "applied"
    report["manifest"] = str(path)
    plan["footage_report"] = report
    return plan
