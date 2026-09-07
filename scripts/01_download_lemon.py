#!/usr/bin/env python3
"""Download the public LEMON inputs and reconstruct the study cohort.

The repository never redistributes LEMON data. This command downloads the
metadata and BrainVision files from the official GWDG mirror into a user-owned
work directory. Large downloads are resumable through ``.part`` files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


LEMON_BASE_URL = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/"
METADATA_FILES = (
    "Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON/MSPSS.csv",
    "Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON/MSPSS_info.txt",
    "Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON/STAI_G_X2.csv",
    "Behavioural_Data_MPILMBB_LEMON/Emotion_and_Personality_Test_Battery_LEMON/STAI-G-X2_info.txt",
    "Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv",
    "Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON_INFO",
    "Behavioural_Data_MPILMBB_LEMON/Data_Availability_Tables/Availability_LEMON_Day1_Data.csv",
    "Behavioural_Data_MPILMBB_LEMON/Data_Availability_Tables/Availability_LEMON_Day1_Data_Info",
    "EEG_MPILMBB_LEMON/EEG_Info",
)
REFERENCE_METADATA_SHA256 = {
    "MSPSS.csv": "81f074fe348729e0fdead7dc711ccf4e9dbe7d329d052b36cd9631656dbb0f7f",
    "MSPSS_info.txt": "fb643d49748132cfc7991d7317df0f29bd85123cce74710b822f804cb65ea260",
    "STAI_G_X2.csv": "025d917b961d8148212b334a696f402ebd99b55b40ec8f6f4d5845a2452b66a6",
    "STAI-G-X2_info.txt": "7143d6cdaeed7fb52ba8ec0073f7d0a4c4d3fa4e79a514c9afa81dfc3ababc6a",
    "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv": "31a85f8723f2d2be63303dbeabd35aef8e762c4f1155750fdd849030cd68a07c",
    "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON_INFO": "7b91fb82f73d362c9f378fe502d5747af03c7ea1a03622e5ff422f210120c0d0",
    "Availability_LEMON_Day1_Data.csv": "98a7947721738686456781d245653344c1e98afd3717aeda31a9af1d2838a826",
    "Availability_LEMON_Day1_Data_Info": "e0713c272aff56aefff4d538477141de64be6eefbd3e957f08f55f23efa2ecf0",
    "EEG_Info": "7c4c42e99046a253f59dfe0c97c6e512aac49c0c50e39273d60298a0c0984b02",
}
EXPECTED_ELIGIBLE = 147
EXPECTED_COMPLETE_TRIPLETS = 142
CHUNK_BYTES = 8 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def encoded_url(relative: str) -> str:
    return LEMON_BASE_URL + "/".join(quote(part) for part in relative.split("/"))


def download_file(url: str, target: Path, retries: int = 5) -> dict[str, object]:
    """Download one file, resuming a partial transfer when the server permits."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return {
            "method": "existing_file_hashed",
            "returncode": 0,
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "error": "",
        }
    partial = target.with_name(target.name + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "LEMON-alpha-reactivity-replication/1.0"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=120) as response:
                status = int(getattr(response, "status", 200))
                append = start > 0 and status == 206
                mode = "ab" if append else "wb"
                if start and not append:
                    start = 0
                with partial.open(mode) as stream:
                    shutil.copyfileobj(response, stream, length=CHUNK_BYTES)
            if partial.stat().st_size <= 0:
                raise OSError("download produced an empty file")
            os.replace(partial, target)
            return {
                "method": "urllib_resume" if append else "urllib",
                "returncode": 0,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "error": "",
            }
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(min(2**attempt, 30))
    return {
        "method": "urllib_failed",
        "returncode": 1,
        "size_bytes": partial.stat().st_size if partial.exists() else 0,
        "sha256": "",
        "error": last_error,
    }


def download_metadata(root: Path, allow_source_drift: bool) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for index, relative in enumerate(METADATA_FILES, start=1):
        target = root / "data" / "lemon" / "metadata" / Path(relative).name
        result = download_file(encoded_url(relative), target)
        observed = str(result["sha256"])
        expected = REFERENCE_METADATA_SHA256[target.name]
        hash_match = observed.lower() == expected.lower()
        if int(result["returncode"]) != 0 or (not hash_match and not allow_source_drift):
            failures.append(target.name)
        rows.append(
            {
                "relative_url": relative,
                "local_file": target.relative_to(root).as_posix(),
                **result,
                "reference_sha256": expected,
                "reference_hash_match": hash_match,
            }
        )
        print(f"METADATA {index:02d}/{len(METADATA_FILES)} {target.name} hash_match={hash_match}", flush=True)
    write_csv(
        root / "outputs" / "01_downloads" / "lemon_metadata_downloads.csv",
        rows,
        [
            "relative_url", "local_file", "method", "returncode", "size_bytes",
            "sha256", "reference_sha256", "reference_hash_match", "error",
        ],
    )
    if failures:
        raise RuntimeError(
            "Metadata download/hash validation failed for: " + ", ".join(failures)
        )
    return {"metadata_files": len(rows), "reference_hashes_matched": sum(bool(r["reference_hash_match"]) for r in rows)}


def index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["ID"].strip(): row for row in rows if row.get("ID", "").strip()}


def build_cohort(root: Path, allow_source_drift: bool) -> dict[str, object]:
    metadata = root / "data" / "lemon" / "metadata"
    by_mspss = index_by_id(read_csv(metadata / "MSPSS.csv"))
    by_stai = index_by_id(read_csv(metadata / "STAI_G_X2.csv"))
    by_meta = index_by_id(read_csv(metadata / "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"))
    available = {
        row.get("ID", "").strip()
        for row in read_csv(metadata / "Availability_LEMON_Day1_Data.csv")
        if row.get("ID", "").strip()
    }
    identifiers = sorted(set(by_mspss) | set(by_stai) | set(by_meta))
    eligible: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for identifier in identifiers:
        reasons: list[str] = []
        meta_row = by_meta.get(identifier, {})
        stai_row = by_stai.get(identifier, {})
        mspss_row = by_mspss.get(identifier, {})
        age_band = meta_row.get("Age", "").strip()
        if age_band not in {"20-25", "25-30", "30-35"}:
            reasons.append("age_not_20_35")
        if not stai_row.get("STAI_Trait_Anxiety", "").strip():
            reasons.append("missing_stai_trait")
        if not mspss_row.get("MSPSS_total", "").strip():
            reasons.append("missing_mspss_total")
        if identifier not in available:
            reasons.append("day1_availability_missing")
        if reasons:
            excluded.append({"ID": identifier, "reasons": ";".join(reasons)})
        else:
            eligible.append(
                {
                    "ID": identifier,
                    "Age": age_band,
                    "Gender_ 1=female_2=male": meta_row.get("Gender_ 1=female_2=male", ""),
                    "STAI_Trait_Anxiety": stai_row.get("STAI_Trait_Anxiety", ""),
                    "MSPSS_total": mspss_row.get("MSPSS_total", ""),
                }
            )
    out = root / "outputs" / "02_cohort"
    write_csv(
        out / "lemon_eligible_subjects.csv",
        eligible,
        ["ID", "Age", "Gender_ 1=female_2=male", "STAI_Trait_Anxiety", "MSPSS_total"],
    )
    write_csv(out / "lemon_excluded_subjects.csv", excluded, ["ID", "reasons"])
    counts: dict[str, int] = {}
    for row in excluded:
        for reason in str(row["reasons"]).split(";"):
            counts[reason] = counts.get(reason, 0) + 1
    write_csv(
        out / "lemon_exclusion_reasons.csv",
        [{"reason": key, "n": value} for key, value in sorted(counts.items())],
        ["reason", "n"],
    )
    flow = {
        "created_utc": utc_now(),
        "n_ids_seen": len(identifiers),
        "n_eligible": len(eligible),
        "n_excluded": len(excluded),
        "reasons": counts,
        "scores_used": ["STAI_Trait_Anxiety", "MSPSS_total"],
        "score_note": "Released totals are used; item scores are not reconstructed.",
    }
    write_json(out / "lemon_cohort_flow.json", flow)
    if len(eligible) != EXPECTED_ELIGIBLE and not allow_source_drift:
        raise RuntimeError(f"Expected {EXPECTED_ELIGIBLE} eligible participants, observed {len(eligible)}")
    print(json.dumps(flow, indent=2), flush=True)
    return flow


def triplet_relative(subject_id: str, extension: str) -> str:
    return f"EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/{subject_id}/RSEEG/{subject_id}.{extension}"


def eeg_target_paths(root: Path, subject_id: str) -> dict[str, Path]:
    folder = root / "data" / "lemon" / "raw_eeg" / subject_id / "RSEEG"
    return {extension: folder / f"{subject_id}.{extension}" for extension in ("vhdr", "vmrk", "eeg")}


def subject_requires_transfer(paths: dict[str, Path]) -> bool:
    return any(not path.is_file() or path.stat().st_size <= 0 for path in paths.values())


def download_eeg(root: Path, max_subjects: int, allow_source_drift: bool) -> dict[str, object]:
    eligible = read_csv(root / "outputs" / "02_cohort" / "lemon_eligible_subjects.csv")
    file_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    touched = 0
    for index, row in enumerate(eligible, start=1):
        subject_id = row["ID"]
        targets = eeg_target_paths(root, subject_id)
        needs_transfer = subject_requires_transfer(targets)
        if max_subjects and needs_transfer and touched >= max_subjects:
            subject_rows.append({"ID": subject_id, "status": "DEFERRED_BY_BATCH_LIMIT", "triplet_complete": False, "files_ok": "", "total_size_bytes": ""})
            continue
        records: list[dict[str, object]] = []
        for extension in ("vhdr", "vmrk", "eeg"):
            relative = triplet_relative(subject_id, extension)
            target = targets[extension]
            result = download_file(encoded_url(relative), target)
            record = {
                "ID": subject_id,
                "extension": extension,
                "relative_url": relative,
                "local_file": target.relative_to(root).as_posix(),
                **result,
            }
            records.append(record)
            file_rows.append(record)
        files_ok = sum(int(item["returncode"]) == 0 and int(item["size_bytes"]) > 0 for item in records)
        status = "PASS" if files_ok == 3 else "INCOMPLETE"
        subject_rows.append(
            {
                "ID": subject_id,
                "status": status,
                "triplet_complete": status == "PASS",
                "files_ok": files_ok,
                "total_size_bytes": sum(int(item["size_bytes"]) for item in records),
            }
        )
        if needs_transfer:
            touched += 1
        print(f"EEG {index:03d}/{len(eligible)} {subject_id} {status}", flush=True)
    out = root / "outputs" / "03_lemon_eeg_downloads"
    write_csv(
        out / "lemon_eeg_file_downloads.csv",
        file_rows,
        ["ID", "extension", "relative_url", "local_file", "method", "returncode", "size_bytes", "sha256", "error"],
    )
    write_csv(
        out / "lemon_eeg_subject_downloads.csv",
        subject_rows,
        ["ID", "status", "triplet_complete", "files_ok", "total_size_bytes"],
    )
    complete = sum(row["status"] == "PASS" for row in subject_rows)
    summary = {
        "created_utc": utc_now(),
        "eligible_subjects": len(eligible),
        "batch_limit": max_subjects or "unlimited",
        "subjects_touched": touched,
        "subjects_complete": complete,
        "subjects_incomplete": sum(row["status"] == "INCOMPLETE" for row in subject_rows),
        "subjects_deferred": sum(row["status"] == "DEFERRED_BY_BATCH_LIMIT" for row in subject_rows),
        "file_records": len(file_rows),
    }
    write_json(out / "lemon_eeg_download_summary.json", summary)
    if not max_subjects and complete != EXPECTED_COMPLETE_TRIPLETS and not allow_source_drift:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPLETE_TRIPLETS} complete BrainVision triplets, observed {complete}. "
            "Inspect the download manifests before using --allow-source-drift."
        )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=repository / "workdir")
    parser.add_argument("--stage", choices=("metadata", "cohort", "eeg", "all"), default="all")
    parser.add_argument("--max-subjects", type=int, default=0, help="Resumable EEG batch limit; 0 means all eligible participants")
    parser.add_argument("--accept-data-responsibility", action="store_true", help="Required before downloading EEG; confirms that the user will follow the LEMON terms and cite the source study")
    parser.add_argument("--allow-source-drift", action="store_true", help="Continue after changed metadata hashes/cohort counts; use only after documenting the change")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"metadata", "all"}:
        download_metadata(root, args.allow_source_drift)
    if args.stage in {"cohort", "all"}:
        build_cohort(root, args.allow_source_drift)
    if args.stage in {"eeg", "all"}:
        if not args.accept_data_responsibility:
            raise SystemExit(
                "EEG download not started. Read docs/DATA_ACCESS_AND_PRIVACY.md and rerun with "
                "--accept-data-responsibility."
            )
        download_eeg(root, args.max_subjects, args.allow_source_drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
