"""Safely unpack the chalk-engine handoff ZIP into a reviewable source folder.

Skips common credential files and likely hard-coded secrets. It does not commit,
push, run code from the archive, or modify existing extracted files by default.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "chalk_engine_patch" / "files.zip"
DEFAULT_DEST = ROOT / "chalk_engine_patch" / "engine_source"
MAX_MEMBER_BYTES = 100 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 10 * 1024 * 1024

SECRET_NAME = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|credentials\.json|token\.json|client_secret[^/]*\.json|"
    r".*secret.*|id_rsa|id_ed25519|authorized_keys|[^/]*\.(?:pem|key|p12|pfx))$",
    re.IGNORECASE,
)
TEXT_EXTENSIONS = {
    ".py", ".patch", ".diff", ".json", ".toml", ".yaml", ".yml", ".md", ".txt",
    ".ini", ".cfg", ".xml", ".html", ".js", ".ts", ".css", ".csv", ".ps1",
    ".bat", ".cmd", ".sh", ".env", ".conf", ".properties",
}
SECRET_PATTERNS = [
    ("private-key material", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("GitHub access token", re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("literal credential assignment", re.compile(
        r"""(?im)\b(?:OPENROUTER_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|YOUTUBE_API_KEY|"""
        r"""TELEGRAM_BOT_TOKEN|AWS_SECRET_ACCESS_KEY|api[_-]?key|access[_-]?token|client_secret)"""
        r"""\s*[:=]\s*["'](?!your_|changeme|placeholder|example|<|\$\{)([A-Za-z0-9_./+=-]{20,})["']"""
    )),
]


def safe_member_path(name: str, dest: Path) -> Path | None:
    # Normalise both ZIP-style and Windows-style separators before validation.
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        return None
    if ":" in pure.parts[0]:
        return None
    target = (dest / Path(*pure.parts)).resolve()
    try:
        target.relative_to(dest.resolve())
    except ValueError:
        return None
    return target


def credential_reason(name: str, data: bytes) -> str | None:
    if SECRET_NAME.search(name.replace("\\", "/")):
        return "sensitive-looking filename"
    ext = Path(name).suffix.casefold()
    if ext not in TEXT_EXTENSIONS:
        return None
    if len(data) > MAX_TEXT_SCAN_BYTES:
        return "text file too large to scan safely"
    text = data.decode("utf-8", errors="ignore")
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return f"possible {label}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--destination", default=str(DEFAULT_DEST))
    parser.add_argument("--force", action="store_true", help="Overwrite files already present in the destination.")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    dest = Path(args.destination).resolve()
    if not archive.is_file():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)

    extracted: list[str] = []
    skipped: list[dict[str, str]] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            print(f"Archive: {archive}")
            print(f"Files in archive: {len(members)}")
            print(f"Extraction folder: {dest}")
            for info in members:
                raw_name = info.filename
                target = safe_member_path(raw_name, dest)
                if target is None:
                    skipped.append({"file": raw_name, "reason": "unsafe archive path"})
                    continue
                if info.file_size > MAX_MEMBER_BYTES:
                    skipped.append({"file": raw_name, "reason": "file exceeds 100 MiB safety limit"})
                    continue
                # Ignore standard ZIP metadata folders.
                if raw_name.replace("\\", "/").startswith("__MACOSX/"):
                    skipped.append({"file": raw_name, "reason": "macOS ZIP metadata"})
                    continue
                if target.exists() and not args.force:
                    skipped.append({"file": raw_name, "reason": "destination already exists (use --force to replace)"})
                    continue
                data = zf.read(info)
                reason = credential_reason(raw_name, data)
                if reason:
                    skipped.append({"file": raw_name, "reason": reason})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                extracted.append(raw_name)
                total_bytes += len(data)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"ERROR: extraction failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    manifest = {
        "source_archive": str(archive.relative_to(ROOT)) if archive.is_relative_to(ROOT) else archive.name,
        "destination": str(dest.relative_to(ROOT)) if dest.is_relative_to(ROOT) else dest.name,
        "extracted_count": len(extracted),
        "extracted_bytes": total_bytes,
        "extracted": extracted,
        "skipped": skipped,
        "note": "Review skipped entries before deciding whether any are legitimate; no skipped entry was copied.",
    }
    manifest_path = dest / "_extract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nExtracted: {len(extracted)} files ({total_bytes:,} bytes)")
    print(f"Manifest: {manifest_path}")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for item in skipped:
            print(f"  SKIP: {item['file']} — {item['reason']}")
    else:
        print("Skipped: 0")
    print("\nNothing was committed or pushed. Review the extracted source and manifest before committing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
