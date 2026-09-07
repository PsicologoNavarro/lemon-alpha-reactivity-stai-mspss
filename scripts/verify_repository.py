#!/usr/bin/env python3
"""Fail-closed privacy, size, syntax, metadata, and integrity audit."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


MAX_BYTES = 50 * 1024 * 1024
FORBIDDEN_SUFFIXES = (
    ".bdf", ".edf", ".eeg", ".vhdr", ".vmrk", ".fif", ".set",
    ".parquet", ".npz", ".csv.gz", ".zip",
)
FORBIDDEN_NAMES = {
    "LEMON_139_participantes_61_canales_88_intervalos_EO_EC_reactividad_STAI_MSPSS_JMP.csv",
    "Data_participant_level_complete_cases.csv",
    "Table_S4_leave_one_participant_out.csv",
}
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", ".venv-analysis", ".venv-preprocessing", "workdir",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
TEXT_SUFFIXES = {
    ".py", ".ps1", ".md", ".txt", ".json", ".csv", ".tsv", ".yml",
    ".yaml", ".toml", ".cff", ".gitignore", ".gitattributes",
}
ALLOWED_ID_FILES = {
    "scripts/02_audit_sources.py",
    "scripts/03_preprocess_eeg.py",
    "scripts/07_run_analysis.py",
    "docs/METHODS_PROVENANCE.md",
    "docs/DATA_ACCESS_AND_PRIVACY.md",
    "reference/known_source_variants.json",
    "reference/source_audit_summary_reference.json",
    "results/reports/UPSTREAM_METHOD_SPECIFICATIONS_ES_EN.txt",
}
SAFE_AGGREGATE_HEADER_EXCEPTIONS = {
    "results/tables/Table_S8_zero_order_correlations.csv": [
        "variable",
        "stai_trait_total",
        "mspss_total",
        "alpha_roi_reactivity_ec_minus_eo_db",
    ],
}
REQUIRED = {
    "README.md", "README_ES.md", "LICENSE", ".gitignore", ".gitattributes",
    "CITATION.cff", ".zenodo.json", "CITATION.cff.template",
    ".zenodo.json.template", "RELEASE_NOTES_v1.0.0.md",
    "requirements/preprocessing-lock.txt", "requirements/analysis-lock.txt",
    "scripts/01_download_lemon.py", "scripts/02_audit_sources.py",
    "scripts/03_preprocess_eeg.py", "scripts/04_compute_psd_qc.py",
    "scripts/05_freeze_qc.py", "scripts/06_build_analysis_inputs.py",
    "scripts/07_run_analysis.py", "scripts/run_pipeline.ps1",
    "docs/GITHUB_ZENODO_RELEASE.md", "docs/DATA_ACCESS_AND_PRIVACY.md",
    "docs/REPRODUCIBILITY.md", "docs/METHODS_PROVENANCE.md",
    "reference/expected_results.json", "reference/known_source_variants.json",
}
MANIFEST_NAME = "SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded(relative: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORIES or part.startswith(".venv-") for part in relative.parts)


def repository_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if is_excluded(relative):
            continue
        if path.is_symlink():
            files.append(path)
        elif path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix().lower())


def text_content(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if path.name in {".gitignore", ".gitattributes"} or suffix in TEXT_SUFFIXES or path.name.endswith(".template"):
        return path.read_text(encoding="utf-8-sig", errors="replace")
    return None


def git_tracked_files(root: Path) -> set[str] | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return {item.decode("utf-8").replace("\\", "/") for item in result.stdout.split(b"\0") if item}


def parse_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Invalid {MANIFEST_NAME} line {line_number}")
        result[match.group(2)] = match.group(1)
    return result


def cff_scalar(text: str, key: str) -> str | None:
    match = re.search(
        rf"(?m)^{re.escape(key)}:\s*(?:\"([^\"]*)\"|'([^']*)'|([^#\r\n]+))\s*$",
        text,
    )
    if not match:
        return None
    return next((value.strip() for value in match.groups() if value is not None), None)


def _cff_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        return value[1:-1]
    return value


def cff_authors(text: str) -> list[dict[str, str]]:
    """Parse the controlled top-level CFF authors block without a YAML dependency."""
    authors: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_authors = False
    for line in text.splitlines():
        if line == "authors:":
            in_authors = True
            continue
        if not in_authors:
            continue
        if line and not line[0].isspace():
            break
        start = re.fullmatch(r"\s{2}-\s+family-names:\s*(.+)", line)
        if start:
            if current is not None:
                authors.append(current)
            current = {"family-names": _cff_value(start.group(1))}
            continue
        field = re.fullmatch(r"\s{4}(given-names|affiliation|orcid):\s*(.+)", line)
        if field and current is not None:
            current[field.group(1)] = _cff_value(field.group(2))
    if current is not None:
        authors.append(current)
    return authors


def normalize_orcid(value: object) -> str:
    return str(value or "").strip().removeprefix("https://orcid.org/")


def valid_orcid(value: object) -> bool:
    compact = normalize_orcid(value).replace("-", "")
    if not re.fullmatch(r"\d{15}[\dX]", compact):
        return False
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    check_value = (12 - total % 11) % 11
    expected = "X" if check_value == 10 else str(check_value)
    return compact[-1] == expected


def write_manifest(root: Path, files: list[Path]) -> None:
    lines = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    (root / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def audit(root: Path, release: bool) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    files = repository_files(root)
    relative_paths = [path.relative_to(root).as_posix() for path in files]
    lower_paths = [item.lower() for item in relative_paths]
    if len(lower_paths) != len(set(lower_paths)):
        errors.append("case-insensitive duplicate paths detected")
    missing = sorted(REQUIRED - set(relative_paths))
    errors.extend(f"required file missing: {item}" for item in missing)

    windows_path = re.compile(r"(?i)\b[a-z]:[\\/]")
    unix_path = re.compile(r"(?i)(?:^|[\s\"'])(?:/users/|/home/)[^\s\"']+")
    email = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
    secret_patterns = {
        "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
        "AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "Bearer token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{20,}"),
    }

    for path, relative in zip(files, relative_paths):
        if path.is_symlink():
            errors.append(f"symlink/reparse-style link not allowed: {relative}")
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"file exceeds 50 MiB: {relative} ({path.stat().st_size} bytes)")
        lower = relative.lower()
        if any(lower.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"data/archive extension forbidden: {relative}")
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"participant-level or oversized artifact forbidden: {relative}")
        content = text_content(path)
        if content is None:
            continue
        if windows_path.search(content) or unix_path.search(content):
            errors.append(f"local absolute path detected: {relative}")
        local_username = "".join(chr(value) for value in (100, 97, 110, 97, 118))
        if re.search(rf"(?i)\b{re.escape(local_username)}\b", content):
            errors.append(f"local username detected: {relative}")
        if email.search(content) and relative not in {"CITATION.cff", ".zenodo.json"}:
            errors.append(f"email address outside live citation metadata: {relative}")
        for label, pattern in secret_patterns.items():
            if pattern.search(content):
                errors.append(f"{label} pattern detected: {relative}")
        if re.search(r"sub-\d{6}", content) and relative not in ALLOWED_ID_FILES:
            errors.append(f"LEMON dataset ID outside documented exception files: {relative}")
        if path.suffix.lower() == ".py":
            try:
                ast.parse(content, filename=relative)
            except SyntaxError as error:
                errors.append(f"Python syntax error in {relative}: {error}")
        if path.suffix.lower() == ".json" or path.name == ".zenodo.json":
            try:
                json.loads(content)
            except json.JSONDecodeError as error:
                errors.append(f"JSON parse error in {relative}: {error}")
        if relative.startswith("results/tables/") and path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            header = rows[0] if rows else []
            sensitive_columns = {
                "participant_id", "subject_id", "omitted_participant", "stai_trait_total",
                "mspss_total", "relationship_status_raw",
            }
            found = sorted(sensitive_columns & set(header))
            if found:
                expected_header = SAFE_AGGREGATE_HEADER_EXCEPTIONS.get(relative)
                safe_small_matrix = (
                    expected_header is not None
                    and header == expected_header
                    and len(rows) == 4
                )
                if not safe_small_matrix:
                    errors.append(f"participant-level columns in aggregate result {relative}: {found}")

    manifest_path = root / MANIFEST_NAME
    if manifest_path.exists():
        try:
            manifest = parse_manifest(manifest_path)
            expected_manifest_paths = {item for item in relative_paths if item != MANIFEST_NAME}
            if set(manifest) != expected_manifest_paths:
                missing_manifest = sorted(expected_manifest_paths - set(manifest))
                extra_manifest = sorted(set(manifest) - expected_manifest_paths)
                errors.append(
                    f"manifest path mismatch; missing={missing_manifest[:5]}, extra={extra_manifest[:5]}"
                )
            for relative, expected_hash in manifest.items():
                candidate = root / Path(relative)
                if candidate.is_file() and sha256_file(candidate) != expected_hash:
                    errors.append(f"manifest hash mismatch: {relative}")
        except ValueError as error:
            errors.append(str(error))
    else:
        warnings.append(f"{MANIFEST_NAME} not present; run --write-manifest before release")

    tracked = git_tracked_files(root)
    if tracked is not None:
        public_files = set(relative_paths)
        untracked = sorted(public_files - tracked - {MANIFEST_NAME})
        if release and untracked:
            errors.append(f"release audit found untracked public files: {untracked[:10]}")
        forbidden_tracked = sorted(
            item for item in tracked
            if Path(item).name in FORBIDDEN_NAMES
            or any(item.lower().endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
            or is_excluded(Path(item))
        )
        if forbidden_tracked:
            errors.append(f"forbidden tracked files: {forbidden_tracked[:10]}")
    elif release:
        errors.append("release audit requires a Git repository and git ls-files")

    if release:
        for live in ("CITATION.cff", ".zenodo.json"):
            if not (root / live).is_file():
                errors.append(f"release metadata missing: {live}; copy and complete its template")
        live_text = "\n".join(
            (root / name).read_text(encoding="utf-8", errors="replace")
            for name in ("CITATION.cff", ".zenodo.json")
            if (root / name).is_file()
        )
        if re.search(r"REPLACE_WITH|TEMPLATE_REQUIRED|YOUR_[A-Z_]", live_text):
            errors.append("unresolved release-metadata placeholder")
        zenodo: dict[str, object] | None = None
        zenodo_path = root / ".zenodo.json"
        if zenodo_path.is_file():
            try:
                zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
                if zenodo.get("upload_type") != "software":
                    errors.append(".zenodo.json upload_type must be software")
                if zenodo.get("access_right") != "open":
                    errors.append(".zenodo.json access_right must be open")
                if zenodo.get("license") != "MIT":
                    errors.append(".zenodo.json license must match the repository MIT license")
                if zenodo.get("version") != "1.0.0":
                    errors.append(".zenodo.json version must be 1.0.0 for tag v1.0.0")
                if not zenodo.get("creators"):
                    errors.append(".zenodo.json creators must not be empty")
            except json.JSONDecodeError:
                pass
        citation_path = root / "CITATION.cff"
        if citation_path.is_file():
            citation = citation_path.read_text(encoding="utf-8")
            for fragment in ('cff-version: "1.2.0"', 'version: "1.0.0"', 'license: MIT'):
                if fragment not in citation:
                    errors.append(f"CITATION.cff missing expected fragment: {fragment}")
            citation_authors = cff_authors(citation)
            if not citation_authors:
                errors.append("CITATION.cff authors must not be empty")
            for index, author in enumerate(citation_authors, start=1):
                if not valid_orcid(author.get("orcid")):
                    errors.append(f"CITATION.cff author {index} has an invalid ORCID")
            if zenodo is not None:
                for cff_key, zenodo_key in (("title", "title"), ("version", "version"), ("license", "license")):
                    cff_value = cff_scalar(citation, cff_key)
                    zenodo_value = str(zenodo.get(zenodo_key, "")).strip()
                    if cff_value != zenodo_value:
                        errors.append(
                            f"citation metadata mismatch for {cff_key}: "
                            f"CITATION.cff={cff_value!r}, .zenodo.json={zenodo_value!r}"
                        )
                creators = zenodo.get("creators")
                if not isinstance(creators, list):
                    errors.append(".zenodo.json creators must be a list")
                elif len(citation_authors) != len(creators):
                    errors.append(
                        "author-count mismatch: "
                        f"CITATION.cff={len(citation_authors)}, .zenodo.json={len(creators)}"
                    )
                else:
                    for index, (author, creator) in enumerate(zip(citation_authors, creators), start=1):
                        if not isinstance(creator, dict):
                            errors.append(f".zenodo.json creator {index} must be an object")
                            continue
                        expected_name = f"{author.get('family-names', '')}, {author.get('given-names', '')}"
                        if creator.get("name") != expected_name:
                            errors.append(f"author-order/name mismatch at position {index}")
                        if creator.get("affiliation") != author.get("affiliation"):
                            errors.append(f"author-affiliation mismatch at position {index}")
                        if normalize_orcid(creator.get("orcid")) != normalize_orcid(author.get("orcid")):
                            errors.append(f"author-ORCID mismatch at position {index}")
                        if not valid_orcid(creator.get("orcid")):
                            errors.append(f".zenodo.json creator {index} has an invalid ORCID")

    return {
        "status": "PASS" if not errors else "FAIL",
        "mode": "release" if release else "pre-release",
        "files_scanned": len(files),
        "bytes_scanned": sum(path.stat().st_size for path in files if path.is_file()),
        "errors": errors,
        "warnings": warnings,
        "git_index_checked": tracked is not None,
        "privacy_boundary": "No raw EEG, participant-level measurements, oversized derivatives, local paths, or secrets",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--release", action="store_true", help="Also require live, completed citation/Zenodo metadata and a Git index")
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository.expanduser().resolve()
    if args.write_manifest:
        write_manifest(root, repository_files(root))
    result = audit(root, args.release)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
