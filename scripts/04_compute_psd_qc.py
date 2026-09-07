#!/usr/bin/env python
"""Compute auditable window/channel/band PSD features from clean full-density LEMON FIFs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import periodogram


BANDS = [
    ("delta", 1.0, 4.0, "primary"),
    ("theta", 4.0, 8.0, "primary"),
    ("alpha", 8.0, 13.0, "primary"),
    ("beta", 13.0, 30.0, "primary"),
    ("beta_low", 13.0, 20.0, "secondary"),
    ("beta_high", 20.0, 30.0, "secondary"),
]
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
BLOCK_SECONDS = 60.0
BLOCKS_PER_CONDITION = 8
MAX_ABS_UV = 250.0
MAX_PTP_UV = 500.0
MIN_ROBUST_STD_UV = 0.1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cohort_metadata(output_root: Path) -> pd.DataFrame:
    cohort = pd.read_csv(
        output_root / "audit" / "tables" / "cohort_full62_frozen.csv",
        dtype={"ID": str},
    )
    cohort = cohort.rename(columns={
        "ID": "participant_id",
        "Age": "age_bin",
        "Gender_ 1=female_2=male": "sex_code",
        "STAI_Trait_Anxiety": "stai_trait_total",
        "MSPSS_total": "mspss_total",
        "Relationship_Status": "relationship_status",
    })
    cohort["sex"] = cohort["sex_code"].map({1: "female", 2: "male", "1": "female", "2": "male"})
    return cohort


def preprocess_qc(output_root: Path, subject_id: str) -> dict:
    path = output_root / "preprocessing" / "qc" / "subjects" / f"{subject_id}_preprocessing_qc.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_clean_raw(raw, condition: str) -> None:
    eeg = raw.copy().pick("eeg")
    if len(raw.ch_names) != 62 or len(eeg.ch_names) != 61:
        raise RuntimeError(f"Expected 62 total/61 EEG channels; got {len(raw.ch_names)}/{len(eeg.ch_names)}")
    if not math.isclose(float(raw.info["sfreq"]), 250.0):
        raise RuntimeError(f"Expected 250 Hz, got {raw.info['sfreq']}")
    if raw.n_times != 120000 or not math.isclose(raw.n_times / raw.info["sfreq"], 480.0):
        raise RuntimeError(f"Expected 120000 samples/480 s, got {raw.n_times}")
    block_descriptions = [str(item) for item in raw.annotations.description if str(item).startswith("block/")]
    bad_boundaries = sum(str(item) == "BAD boundary" for item in raw.annotations.description)
    edge_boundaries = sum(str(item) == "EDGE boundary" for item in raw.annotations.description)
    expected = [f"block/{condition}/{index:02d}" for index in range(1, 9)]
    if block_descriptions != expected or bad_boundaries != 7 or edge_boundaries != 7:
        raise RuntimeError(
            f"Block annotation audit failed: blocks={block_descriptions}, BAD={bad_boundaries}, EDGE={edge_boundaries}"
        )


def process_subject(output_root: Path, subject_id: str, force: bool) -> dict:
    import mne

    started = time.perf_counter()
    window_dir = output_root / "features" / "window_channel_bandpower"
    subject_dir = output_root / "features" / "subject_summaries"
    qc_dir = output_root / "features" / "qc" / "subjects"
    for folder in (window_dir, subject_dir, qc_dir):
        folder.mkdir(parents=True, exist_ok=True)
    window_out = window_dir / f"{subject_id}_window_channel_bandpower.parquet"
    subject_out = subject_dir / f"{subject_id}_subject_condition_channel_band.csv"
    qc_out = qc_dir / f"{subject_id}_psd_qc.json"
    if not force and all(path.exists() for path in (window_out, subject_out, qc_out)):
        payload = json.loads(qc_out.read_text(encoding="utf-8"))
        payload["run_action"] = "SKIPPED_EXISTING"
        return payload

    meta = cohort_metadata(output_root).set_index("participant_id").loc[subject_id]
    prep = preprocess_qc(output_root, subject_id)
    all_frames = []
    window_qc_rows = []
    channel_names = None
    for condition in ("eyes_open", "eyes_closed"):
        fif = output_root / "preprocessing" / "clean_fif" / f"{subject_id}_{condition}_clean_raw.fif"
        raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
        validate_clean_raw(raw, condition)
        eeg = raw.copy().pick("eeg")
        channel_names = eeg.ch_names
        data_uv = eeg.get_data() * 1e6
        sfreq = float(eeg.info["sfreq"])
        window_n = int(round(WINDOW_SECONDS * sfreq))
        step_n = int(round(STEP_SECONDS * sfreq))
        block_n = int(round(BLOCK_SECONDS * sfreq))
        for block_index in range(BLOCKS_PER_CONDITION):
            block_start = block_index * block_n
            starts = range(0, block_n - window_n + 1, step_n)
            for window_index, relative_start in enumerate(starts, start=1):
                absolute_start = block_start + relative_start
                segment = data_uv[:, absolute_start:absolute_start + window_n]
                finite = np.isfinite(segment).all(axis=1)
                max_abs = np.max(np.abs(segment), axis=1)
                ptp = np.ptp(segment, axis=1)
                med = np.median(segment, axis=1, keepdims=True)
                robust_std = 1.4826 * np.median(np.abs(segment - med), axis=1)
                accepted = (
                    finite
                    & (max_abs <= MAX_ABS_UV)
                    & (ptp <= MAX_PTP_UV)
                    & (robust_std >= MIN_ROBUST_STD_UV)
                )
                global_accepted = bool(accepted.all())
                frequencies, pxx_uv2_hz = periodogram(
                    segment,
                    fs=sfreq,
                    window="hamming",
                    detrend="constant",
                    return_onesided=True,
                    scaling="density",
                    axis=-1,
                )
                for band, low, high, role in BANDS:
                    if band in {"beta", "beta_high"}:
                        mask = (frequencies >= low) & (frequencies <= high)
                        boundary_rule = "low<=f<=high"
                    else:
                        mask = (frequencies >= low) & (frequencies < high)
                        boundary_rule = "low<=f<high"
                    selected = pxx_uv2_hz[:, mask]
                    mean_density = np.mean(selected, axis=1)
                    integrated = np.trapezoid(selected, frequencies[mask], axis=1)
                    db_density = 10.0 * np.log10(np.maximum(mean_density, np.finfo(float).tiny))
                    frame = pd.DataFrame({
                        "participant_id": subject_id,
                        "condition": condition,
                        "block": block_index + 1,
                        "window": window_index,
                        "window_start_s_in_block": relative_start / sfreq,
                        "window_start_s_in_condition": absolute_start / sfreq,
                        "window_duration_s": WINDOW_SECONDS,
                        "channel": channel_names,
                        "band": band,
                        "band_role": role,
                        "f_low_hz": low,
                        "f_high_hz": high,
                        "frequency_boundary_rule": boundary_rule,
                        "frequency_resolution_hz": sfreq / window_n,
                        "n_frequency_bins": int(mask.sum()),
                        "mean_psd_uv2_per_hz": mean_density,
                        "mean_psd_db_uv2_per_hz": db_density,
                        "integrated_power_uv2": integrated,
                        "channel_window_accepted": accepted,
                        "global_window_accepted": global_accepted,
                        "channel_max_abs_uv": max_abs,
                        "channel_peak_to_peak_uv": ptp,
                        "channel_robust_std_uv": robust_std,
                    })
                    all_frames.append(frame)
                window_qc_rows.append({
                    "participant_id": subject_id,
                    "condition": condition,
                    "block": block_index + 1,
                    "window": window_index,
                    "window_start_s_in_condition": absolute_start / sfreq,
                    "channels_accepted": int(accepted.sum()),
                    "channels_total": int(len(accepted)),
                    "global_window_accepted": global_accepted,
                })

    windows = pd.concat(all_frames, ignore_index=True)
    windows["preprocessing_status"] = prep["status"]
    windows["preprocessing_manual_review"] = prep["status"] != "PASS"
    windows["age_bin"] = meta["age_bin"]
    windows["sex"] = meta["sex"]
    windows["relationship_status"] = meta["relationship_status"]
    windows["stai_trait_total"] = float(meta["stai_trait_total"])
    windows["mspss_total"] = float(meta["mspss_total"])
    windows.to_parquet(window_out, index=False, compression="zstd")

    accepted = windows.loc[windows["channel_window_accepted"]].copy()
    grouped = accepted.groupby(
        ["participant_id", "condition", "channel", "band", "band_role"],
        observed=True,
    )
    summary = grouped.agg(
        windows_accepted=("mean_psd_uv2_per_hz", "size"),
        mean_psd_uv2_per_hz=("mean_psd_uv2_per_hz", "mean"),
        sd_psd_uv2_per_hz=("mean_psd_uv2_per_hz", "std"),
        mean_window_psd_db_uv2_per_hz=("mean_psd_db_uv2_per_hz", "mean"),
        sd_window_psd_db_uv2_per_hz=("mean_psd_db_uv2_per_hz", "std"),
        mean_integrated_power_uv2=("integrated_power_uv2", "mean"),
    ).reset_index()
    summary["db_of_mean_psd_uv2_per_hz"] = 10.0 * np.log10(
        np.maximum(summary["mean_psd_uv2_per_hz"], np.finfo(float).tiny)
    )
    summary["windows_expected"] = BLOCKS_PER_CONDITION * 29
    summary["window_retention_fraction"] = summary["windows_accepted"] / summary["windows_expected"]
    summary["preprocessing_status"] = prep["status"]
    summary["age_bin"] = meta["age_bin"]
    summary["sex"] = meta["sex"]
    summary["relationship_status"] = meta["relationship_status"]
    summary["stai_trait_total"] = float(meta["stai_trait_total"])
    summary["mspss_total"] = float(meta["mspss_total"])
    summary.to_csv(subject_out, index=False)

    window_qc = pd.DataFrame(window_qc_rows)
    payload = {
        "participant_id": subject_id,
        "status": "PASS",
        "run_action": "PROCESSED",
        "created_utc": utc_now(),
        "preprocessing_status": prep["status"],
        "channels_eeg": 61,
        "conditions": 2,
        "blocks_per_condition": BLOCKS_PER_CONDITION,
        "windows_per_block": 29,
        "windows_per_condition": 232,
        "channel_windows_total": int(len(window_qc) * 61),
        "channel_windows_accepted": int(windows.loc[windows.band.eq("delta"), "channel_window_accepted"].sum()),
        "global_windows_total": int(len(window_qc)),
        "global_windows_accepted": int(window_qc.global_window_accepted.sum()),
        "global_window_retention_fraction": float(window_qc.global_window_accepted.mean()),
        "bands": [item[0] for item in BANDS],
        "primary_bands": [item[0] for item in BANDS if item[3] == "primary"],
        "psd_method": {
            "estimator": "single-segment periodogram",
            "window": "Hamming",
            "window_seconds": WINDOW_SECONDS,
            "overlap_fraction": 0.5,
            "detrend": "constant",
            "scaling": "density",
            "frequency_resolution_hz": 0.25,
            "linear_density_unit": "microvolt squared per Hz",
            "db_density_formula": "10*log10(mean linear density in band)",
        },
        "window_qc_rule": {
            "channel_max_abs_uv_lte": MAX_ABS_UV,
            "channel_peak_to_peak_uv_lte": MAX_PTP_UV,
            "channel_robust_std_uv_gte": MIN_ROBUST_STD_UV,
            "finite_required": True,
        },
        "outputs": {
            "window_parquet": window_out.relative_to(output_root).as_posix(),
            "window_parquet_sha256": sha256(window_out),
            "subject_summary_csv": subject_out.relative_to(output_root).as_posix(),
            "subject_summary_sha256": sha256(subject_out),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(qc_out, payload)
    return payload


def aggregate(output_root: Path, run_rows: list[dict], export_csv_gz: bool) -> dict:
    feature_root = output_root / "features"
    qc_root = feature_root / "qc"
    qc_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(qc_root / "psd_run_manifest.csv", index=False)
    summaries = [pd.read_csv(path) for path in sorted((feature_root / "subject_summaries").glob("*.csv"))]
    aggregate_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    aggregate_csv = feature_root / "subject_condition_channel_band.csv"
    aggregate_summary.to_csv(aggregate_csv, index=False)

    csv_gz_path = feature_root / "window_channel_bandpower_all_subjects.csv.gz"
    if export_csv_gz:
        parquet_paths = sorted((feature_root / "window_channel_bandpower").glob("*.parquet"))
        with gzip.open(csv_gz_path, "wt", encoding="utf-8", newline="") as stream:
            first = True
            for path in parquet_paths:
                frame = pd.read_parquet(path)
                frame.to_csv(stream, index=False, header=first)
                first = False

    all_qc = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((qc_root / "subjects").glob("*_psd_qc.json"))
    ]
    summary = {
        "created_utc": utc_now(),
        "subjects_with_psd": len(all_qc),
        "status_counts": dict(Counter(item.get("status", "UNKNOWN") for item in all_qc)),
        "processed_this_run": sum(item.get("run_action") == "PROCESSED" for item in run_rows),
        "skipped_existing_this_run": sum(item.get("run_action") == "SKIPPED_EXISTING" for item in run_rows),
        "failed_this_run": sum(item.get("status") == "FAIL" for item in run_rows),
        "subject_summary_rows": int(len(aggregate_summary)),
        "aggregate_subject_csv": aggregate_csv.relative_to(output_root).as_posix(),
        "aggregate_subject_csv_sha256": sha256(aggregate_csv),
        "combined_window_csv_gz": csv_gz_path.relative_to(output_root).as_posix() if export_csv_gz else None,
        "combined_window_csv_gz_sha256": sha256(csv_gz_path) if export_csv_gz else None,
        "inferential_unit": "participant; windows are repeated measurements/QC only",
    }
    write_json(qc_root / "psd_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", help="Comma-separated subject IDs; default all completed preprocessing QC")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-csv-gz", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if args.subjects:
        subjects = [item.strip() for item in args.subjects.split(",") if item.strip()]
    else:
        subjects = sorted(
            path.name.replace("_preprocessing_qc.json", "")
            for path in (output_root / "preprocessing" / "qc" / "subjects").glob("*_preprocessing_qc.json")
        )
    rows = []
    log_path = output_root / "features" / "qc" / "psd.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for index, subject_id in enumerate(subjects, start=1):
            print(f"{index:03d}/{len(subjects)} START {subject_id}", flush=True)
            try:
                payload = process_subject(output_root, subject_id, force=args.force)
                row = {
                    "participant_id": subject_id,
                    "status": payload["status"],
                    "run_action": payload["run_action"],
                    "preprocessing_status": payload.get("preprocessing_status"),
                    "global_window_retention_fraction": payload.get("global_window_retention_fraction"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                }
                message = (
                    f"{index:03d}/{len(subjects)} {payload['run_action']} {subject_id} "
                    f"retention={payload.get('global_window_retention_fraction', float('nan')):.3f} "
                    f"seconds={payload.get('elapsed_seconds', 0):.1f}"
                )
            except Exception as error:
                tb = traceback.format_exc()
                error_path = output_root / "features" / "qc" / "subjects" / f"{subject_id}_psd_error.txt"
                error_path.write_text(tb, encoding="utf-8")
                row = {
                    "participant_id": subject_id,
                    "status": "FAIL",
                    "run_action": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                message = f"{index:03d}/{len(subjects)} FAIL {subject_id}: {type(error).__name__}: {error}"
            rows.append(row)
            print(message, flush=True)
            log.write(f"{utc_now()} {message}\n")
    summary = aggregate(output_root, rows, export_csv_gz=args.export_csv_gz)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["failed_this_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
