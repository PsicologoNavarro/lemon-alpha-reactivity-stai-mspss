#!/usr/bin/env python
"""Freeze and audit the locally downloaded full-density LEMON BrainVision sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPECTED_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "VEOG", "CP5", "CP1", "CP2", "CP6", "AFz",
    "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7",
    "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8",
    "C5", "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1",
    "P2", "P6", "PO7", "PO3", "POz", "PO4", "PO8",
]
RECOVERABLE_SOURCE_VARIANTS = {
    "sub-010020": "normalize historical DataFile/MarkerFile sidecar names",
    "sub-010126": "map marker S208 to eyes_closed and S200 to eyes_open",
    "sub-010193": "normalize historical DataFile/MarkerFile sidecar names",
}
EXPECTED_HARD_EXCLUSIONS = {
    "sub-010015": "truncated_binary_payload",
    "sub-010078": "overlapping_condition_blocks",
    "sub-010100": "truncated_binary_payload",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def header_details(path: Path) -> dict:
    text = read_text(path)
    nchan_match = re.search(r"^NumberOfChannels=(\d+)\s*$", text, re.MULTILINE)
    interval_match = re.search(r"^SamplingInterval=([0-9.]+)\s*$", text, re.MULTILINE)
    binary_match = re.search(r"^BinaryFormat=([^\r\n]+)", text, re.MULTILINE)
    data_match = re.search(r"^DataFile=([^\r\n]+)", text, re.MULTILINE)
    marker_match = re.search(r"^MarkerFile=([^\r\n]+)", text, re.MULTILINE)
    channels: list[tuple[int, str]] = []
    for match in re.finditer(r"^Ch(\d+)=([^,]+),", text, re.MULTILINE):
        channels.append((int(match.group(1)), match.group(2).replace(r"\1", ",").strip()))
    channels.sort()
    interval = float(interval_match.group(1)) if interval_match else float("nan")
    return {
        "n_channels": int(nchan_match.group(1)) if nchan_match else None,
        "sampling_interval_us": interval,
        "sfreq": 1_000_000.0 / interval if interval > 0 else None,
        "binary_format": binary_match.group(1).strip() if binary_match else None,
        "data_file_declared": data_match.group(1).strip() if data_match else None,
        "marker_file_declared": marker_match.group(1).strip() if marker_match else None,
        "channels": [name for _, name in channels],
    }


def marker_details(path: Path, subject_id: str, sfreq: float, duration_s: float) -> dict:
    marker_map = (
        {"S200": "eyes_open", "S208": "eyes_closed"}
        if subject_id == "sub-010126"
        else {"S200": "eyes_open", "S210": "eyes_closed"}
    )
    events: list[tuple[float, str]] = []
    for line in read_text(path).splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 3:
            continue
        description = fields[1].replace(" ", "").strip()
        if description not in marker_map:
            continue
        try:
            onset = (int(fields[2]) - 1) / sfreq
        except ValueError:
            continue
        events.append((onset, marker_map[description]))
    events.sort()
    collapsed: list[tuple[float, str]] = []
    last_onset = None
    last_condition = None
    for onset, condition in events:
        new_train = last_onset is None or condition != last_condition or onset - last_onset > 5.0
        if new_train:
            collapsed.append((onset, condition))
        last_onset, last_condition = onset, condition
    raw_counts = Counter(condition for _, condition in events)
    counts = Counter(condition for _, condition in collapsed)
    issues: list[str] = []
    for condition in ("eyes_open", "eyes_closed"):
        if counts[condition] != 8:
            issues.append(f"{condition}: expected 8 collapsed blocks, observed {counts[condition]}")
    for index, (onset, condition) in enumerate(collapsed):
        if onset + 60.0 > duration_s + 1.0 / sfreq:
            issues.append(f"{condition} block {index + 1} exceeds recording duration")
        if index and collapsed[index - 1][0] + 60.0 > onset + 1.0 / sfreq:
            issues.append(f"overlap before collapsed block {index + 1}")
    return {
        "raw_eo_markers": raw_counts["eyes_open"],
        "raw_ec_markers": raw_counts["eyes_closed"],
        "collapsed_eo_blocks": counts["eyes_open"],
        "collapsed_ec_blocks": counts["eyes_closed"],
        "collapsed_blocks": collapsed,
        "marker_ok": not issues,
        "marker_issues": "; ".join(issues),
        "marker_map": marker_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_project.resolve()
    output = args.output_root.resolve()
    tables = output / "audit" / "tables"
    metadata = output / "audit" / "metadata"
    logs = output / "audit" / "logs"
    for folder in (tables, metadata, logs):
        folder.mkdir(parents=True, exist_ok=True)

    download_csv = source / "outputs/03_lemon_eeg_downloads/lemon_eeg_subject_downloads.csv"
    eligible_csv = source / "outputs/02_cohort/lemon_eligible_subjects.csv"
    context_csv = source / "data/lemon/metadata/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    downloads = pd.read_csv(download_csv, dtype=str).fillna("")
    eligible = pd.read_csv(eligible_csv, dtype=str).fillna("")
    context = pd.read_csv(context_csv, dtype=str).fillna("")
    complete_ids = sorted(downloads.loc[downloads.status.eq("PASS"), "ID"].tolist())

    subject_rows = []
    file_rows = []
    channel_rows = []
    log_path = logs / "source_audit.log"
    with log_path.open("w", encoding="utf-8") as log:
        for index, subject_id in enumerate(complete_ids, start=1):
            folder = source / "data/lemon/raw_eeg" / subject_id / "RSEEG"
            paths = {ext: folder / f"{subject_id}.{ext}" for ext in ("vhdr", "vmrk", "eeg")}
            details = header_details(paths["vhdr"])
            bytes_per_value = 2 if details["binary_format"] == "INT_16" else None
            denominator = details["n_channels"] * bytes_per_value if details["n_channels"] and bytes_per_value else 0
            eeg_bytes = paths["eeg"].stat().st_size
            divisible = bool(denominator and eeg_bytes % denominator == 0)
            n_samples = eeg_bytes // denominator if divisible else 0
            duration_s = n_samples / details["sfreq"] if n_samples and details["sfreq"] else 0.0
            markers = marker_details(paths["vmrk"], subject_id, float(details["sfreq"]), duration_s)
            channel_ok = details["channels"] == EXPECTED_CHANNELS
            declared_ok = (
                details["data_file_declared"] == f"{subject_id}.eeg"
                and details["marker_file_declared"] == f"{subject_id}.vmrk"
            )
            strict_ok = channel_ok and declared_ok and divisible and markers["marker_ok"]
            processable_with_normalization = channel_ok and divisible and markers["marker_ok"]
            if subject_id in RECOVERABLE_SOURCE_VARIANTS and processable_with_normalization:
                status = "RECOVERABLE"
                disposition = "PROCESS_WITH_DOCUMENTED_NORMALIZATION"
                reason = RECOVERABLE_SOURCE_VARIANTS[subject_id]
            elif strict_ok:
                status = "PASS"
                disposition = "PROCESS"
                reason = ""
            elif subject_id in EXPECTED_HARD_EXCLUSIONS:
                status = "EXPECTED_EXCLUSION"
                disposition = "EXCLUDE"
                reason = EXPECTED_HARD_EXCLUSIONS[subject_id]
            else:
                status = "FAIL_UNEXPECTED"
                disposition = "STOP_AND_REVIEW"
                reason = "unexpected source-audit failure"
            subject_rows.append({
                "participant_id": subject_id,
                "status": status,
                "disposition": disposition,
                "reason": reason,
                "n_channels": details["n_channels"],
                "scalp_eeg_channels": details["n_channels"] - int("VEOG" in details["channels"]),
                "sfreq": details["sfreq"],
                "n_samples": n_samples,
                "duration_s": duration_s,
                "binary_format": details["binary_format"],
                "payload_divisible": divisible,
                "header_references_match": declared_ok,
                "channel_order_match": channel_ok,
                "raw_eo_markers": markers["raw_eo_markers"],
                "raw_ec_markers": markers["raw_ec_markers"],
                "collapsed_eo_blocks": markers["collapsed_eo_blocks"],
                "collapsed_ec_blocks": markers["collapsed_ec_blocks"],
                "marker_ok": markers["marker_ok"],
                "marker_issues": markers["marker_issues"],
                "vhdr": paths["vhdr"].relative_to(source).as_posix(),
                "vmrk": paths["vmrk"].relative_to(source).as_posix(),
                "eeg": paths["eeg"].relative_to(source).as_posix(),
            })
            for position, channel in enumerate(details["channels"], start=1):
                channel_rows.append({"participant_id": subject_id, "position": position, "channel": channel, "type": "eog" if channel == "VEOG" else "eeg"})
            for ext, path in paths.items():
                file_rows.append({
                    "participant_id": subject_id,
                    "extension": ext,
                    "relative_path": path.relative_to(source).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
            message = f"{index:03d}/{len(complete_ids)} {subject_id} {status} channels={details['n_channels']} blocks={markers['collapsed_eo_blocks']}+{markers['collapsed_ec_blocks']}"
            print(message, flush=True)
            log.write(message + "\n")

    subjects = pd.DataFrame(subject_rows)
    files = pd.DataFrame(file_rows)
    channels = pd.DataFrame(channel_rows)
    subjects.to_csv(tables / "source_subject_audit.csv", index=False)
    files.to_csv(metadata / "source_file_manifest_sha256.csv", index=False)
    channels.to_csv(tables / "source_channel_inventory.csv", index=False)

    frozen = eligible[eligible.ID.isin(complete_ids)].copy()
    frozen = frozen.merge(context[["ID", "Relationship_Status"]], on="ID", how="left", validate="one_to_one")
    frozen = frozen.merge(subjects[["participant_id", "status"]], left_on="ID", right_on="participant_id", how="left", validate="one_to_one")
    frozen = frozen.drop(columns="participant_id").sort_values("ID")
    frozen.to_csv(tables / "cohort_full62_frozen.csv", index=False)
    downloads.loc[~downloads.status.eq("PASS")].to_csv(tables / "excluded_incomplete_downloads.csv", index=False)

    age_distribution = frozen.groupby("Age", dropna=False).size().to_dict()
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": "<USER_SELECTED_LEMON_DATA_ROOT>",
        "download_eligible": int(len(downloads)),
        "triplet_complete": int(len(complete_ids)),
        "triplet_incomplete": int((~downloads.status.eq("PASS")).sum()),
        "source_audit_pass": int(subjects.status.eq("PASS").sum()),
        "source_audit_fail": int(subjects.status.ne("PASS").sum()),
        "source_audit_recoverable": int(subjects.status.eq("RECOVERABLE").sum()),
        "source_expected_exclusions": int(subjects.status.eq("EXPECTED_EXCLUSION").sum()),
        "source_unexpected_failures": int(subjects.status.eq("FAIL_UNEXPECTED").sum()),
        "processable_after_documented_normalization": int(subjects.disposition.str.startswith("PROCESS").sum()),
        "expected_channels_total": 62,
        "expected_scalp_eeg": 61,
        "auxiliary_eog": "VEOG",
        "age_distribution": age_distribution,
        "source_bytes_hashed": int(files.bytes.sum()),
        "source_gib_hashed": float(files.bytes.sum() / 1024**3),
        "files_hashed": int(len(files)),
        "hash_algorithm": "SHA-256",
        "inferential_unit": "participant",
    }
    (metadata / "source_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    expected_pattern = (
        summary["download_eligible"] == 147
        and summary["triplet_complete"] == 142
        and summary["source_audit_pass"] == 136
        and summary["source_audit_recoverable"] == 3
        and summary["source_expected_exclusions"] == 3
        and summary["source_unexpected_failures"] == 0
        and summary["processable_after_documented_normalization"] == 139
    )
    if not expected_pattern:
        raise RuntimeError(
            "Source audit does not match the frozen 147 -> 142 -> 139 pattern; "
            "inspect audit/tables/source_subject_audit.csv before proceeding"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
