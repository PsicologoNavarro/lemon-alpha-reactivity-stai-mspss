#!/usr/bin/env python
"""Auditable full-density preprocessing for the locally downloaded LEMON rsEEG.

The script never modifies source BrainVision files. It creates small normalized
sidecars plus a hard link to the original binary, extracts the sixteen 60-s
resting blocks, fits one rank-aware extended-Infomax ICA per participant on EO
and EC jointly, labels all components with ICLabel, applies a frozen rejection
rule, interpolates detected bad scalp channels, and saves EO/EC separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCALP_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "AFz", "P7",
    "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "AF7", "AF3",
    "AF4", "AF8", "F5", "F1", "F2", "F6", "FT7", "FC3", "FC4", "FT8", "C5",
    "C1", "C2", "C6", "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2",
    "P6", "PO7", "PO3", "POz", "PO4", "PO8",
]
EXPECTED_CHANNELS = SCALP_CHANNELS[:16] + ["VEOG"] + SCALP_CHANNELS[16:]
ICLABEL_CLASSES = [
    "brain", "muscle artifact", "eye blink", "heart beat",
    "line noise", "channel noise", "other",
]
ARTIFACT_CLASSES = {
    "muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"
}
ICLABEL_THRESHOLD = 0.80
BASE_ICA_SEED = 20260902
SPECIAL_MARKER_MAP = {"sub-010126": {"S208": "eyes_closed", "S200": "eyes_open"}}
HARD_EXCLUSIONS = {
    "sub-010015": "truncated_binary_payload",
    "sub-010078": "overlapping_condition_blocks",
    "sub-010100": "truncated_binary_payload",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "si", "sí"})


def preprocessing_output_paths(output_root: Path, subject_id: str) -> dict[str, Path]:
    return {
        "eyes_open_fif": output_root / "preprocessing" / "clean_fif" / f"{subject_id}_eyes_open_clean_raw.fif",
        "eyes_closed_fif": output_root / "preprocessing" / "clean_fif" / f"{subject_id}_eyes_closed_clean_raw.fif",
        "ica_fif": output_root / "preprocessing" / "ica" / f"{subject_id}-ica.fif",
        "qc_json": output_root / "preprocessing" / "qc" / "subjects" / f"{subject_id}_preprocessing_qc.json",
        "component_csv": output_root / "preprocessing" / "qc" / "components" / f"{subject_id}_ica_components.csv",
        "channel_metrics_csv": output_root / "preprocessing" / "qc" / "channels" / f"{subject_id}_channel_metrics.csv",
        "bad_channel_detection_csv": output_root / "preprocessing" / "qc" / "channels" / f"{subject_id}_bad_channel_detection.csv",
    }


def subject_preprocessing_complete(output_root: Path, subject_id: str) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in preprocessing_output_paths(output_root, subject_id).values())


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
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def patch_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}=.*$", flags=re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(f"Missing {key} assignment")
    return pattern.sub(f"{key}={value}", text, count=1)


def normalized_brainvision_sidecars(source_folder: Path, subject_id: str, work_root: Path) -> Path:
    """Create normalized text sidecars and a zero-copy hard link to the EEG binary."""
    stage = work_root / subject_id
    stage.mkdir(parents=True, exist_ok=True)
    source_vhdr = source_folder / f"{subject_id}.vhdr"
    source_vmrk = source_folder / f"{subject_id}.vmrk"
    source_eeg = source_folder / f"{subject_id}.eeg"
    stage_vhdr = stage / f"{subject_id}.vhdr"
    stage_vmrk = stage / f"{subject_id}.vmrk"
    stage_eeg = stage / f"{subject_id}.eeg"

    vhdr_text = patch_assignment(read_text(source_vhdr), "DataFile", stage_eeg.name)
    vhdr_text = patch_assignment(vhdr_text, "MarkerFile", stage_vmrk.name)
    vmrk_text = patch_assignment(read_text(source_vmrk), "DataFile", stage_eeg.name)
    stage_vhdr.write_text(vhdr_text, encoding="utf-8")
    stage_vmrk.write_text(vmrk_text, encoding="utf-8")
    transfer_method = "existing"
    if stage_eeg.exists():
        if stage_eeg.stat().st_size != source_eeg.stat().st_size:
            raise RuntimeError(f"Existing staged binary size mismatch: {stage_eeg}")
    else:
        try:
            os.link(source_eeg, stage_eeg)
            transfer_method = "hardlink"
        except OSError:
            # Hard links cannot cross filesystems/volumes. A byte copy is the
            # portable fallback; the source is never modified.
            shutil.copy2(source_eeg, stage_eeg)
            transfer_method = "copy2"
    if transfer_method == "copy2" and sha256(stage_eeg) != sha256(source_eeg):
        raise RuntimeError(f"Copied EEG binary hash mismatch: {stage_eeg}")
    return stage_vhdr


def parse_marker_blocks(vmrk: Path, subject_id: str, sfreq: float) -> tuple[list[dict], dict]:
    marker_map = SPECIAL_MARKER_MAP.get(
        subject_id, {"S200": "eyes_open", "S210": "eyes_closed"}
    )
    events: list[tuple[float, str, str]] = []
    for line in read_text(vmrk).splitlines():
        if not line.startswith("Mk") or "=" not in line:
            continue
        fields = line.split("=", 1)[1].split(",")
        if len(fields) < 3:
            continue
        description = fields[1].replace(" ", "").strip()
        condition = marker_map.get(description)
        if condition is None:
            continue
        onset = (int(fields[2]) - 1) / sfreq
        events.append((onset, condition, description))
    events.sort()
    trains: list[dict] = []
    for onset, condition, marker in events:
        new_train = (
            not trains
            or condition != trains[-1]["condition"]
            or onset - trains[-1]["last_marker_s"] > 5.0
        )
        if new_train:
            trains.append({
                "condition": condition,
                "marker": marker,
                "start_s": onset,
                "last_marker_s": onset,
                "n_markers": 1,
            })
        else:
            trains[-1]["last_marker_s"] = onset
            trains[-1]["n_markers"] += 1
    counts = Counter(item["condition"] for item in trains)
    problems = []
    if counts != Counter({"eyes_open": 8, "eyes_closed": 8}):
        problems.append(f"block counts={dict(counts)}")
    for previous, current in zip(trains, trains[1:]):
        if previous["start_s"] + 60.0 > current["start_s"] + 1.0 / sfreq:
            problems.append(
                f"overlap: {previous['start_s']:.3f}-{previous['start_s'] + 60:.3f} "
                f"before {current['start_s']:.3f}"
            )
    return trains, {
        "marker_map": marker_map,
        "n_raw_markers": len(events),
        "n_blocks": len(trains),
        "condition_counts": dict(counts),
        "problems": problems,
    }


def load_condition_blocks(raw_header, blocks: list[dict], condition: str, out_sfreq: float):
    import mne

    selected = [item for item in blocks if item["condition"] == condition]
    segments = []
    for item in selected:
        segment = raw_header.copy().crop(
            tmin=float(item["start_s"]),
            tmax=float(item["start_s"]) + 60.0,
            include_tmax=False,
        )
        segment.load_data(verbose="ERROR")
        segment.set_annotations(None)
        segment.resample(out_sfreq, npad="auto", verbose="ERROR")
        # Cropped BrainVision objects retain the absolute source first_samp.
        # Rebuild each extracted block on a local 0-s clock before concatenation
        # so boundaries and block annotations align at exact 60-s intervals.
        local_info = segment.info.copy()
        local_info.set_meas_date(None)
        segment = mne.io.RawArray(
            segment.get_data(), local_info, first_samp=0, verbose="ERROR"
        )
        segments.append(segment)
    if len(segments) != 8:
        raise RuntimeError(f"{condition}: expected 8 blocks, got {len(segments)}")
    combined = mne.concatenate_raws(segments, preload=True, verbose="ERROR")
    onsets = np.arange(8, dtype=float) * 60.0
    block_annotations = mne.Annotations(
        onset=onsets,
        duration=np.full(8, 60.0 - 1.0 / out_sfreq),
        description=[f"block/{condition}/{index:02d}" for index in range(1, 9)],
        orig_time=combined.annotations.orig_time,
    )
    # Preserve the BAD/EDGE boundary annotations inserted by MNE at every
    # concatenation point so FIR filtering and ICA rejection never bridge two
    # discontinuous source blocks.
    combined.set_annotations(combined.annotations + block_annotations)
    return combined


def channel_metrics(raw, stage: str) -> pd.DataFrame:
    from scipy.stats import kurtosis

    data_uv = raw.get_data(picks="eeg") * 1e6
    rows = []
    for index, channel in enumerate(raw.copy().pick("eeg").ch_names):
        x = data_uv[index]
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        rows.append({
            "stage": stage,
            "channel": channel,
            "median_uv": median,
            "robust_std_uv": 1.4826 * mad,
            "std_uv": float(np.std(x)),
            "peak_to_peak_uv": float(np.ptp(x)),
            "extreme_fraction_abs_gt_250uv": float(np.mean(np.abs(x) > 250.0)),
            "flat_diff_fraction_lt_0_01uv": float(np.mean(np.abs(np.diff(x)) < 0.01)),
            "kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        })
    return pd.DataFrame(rows)


def detect_bad_channels(qc_raw) -> tuple[list[str], pd.DataFrame]:
    from mne.preprocessing import find_bad_channels_lof

    eeg_names = qc_raw.copy().pick("eeg").ch_names
    data_uv = qc_raw.get_data(picks="eeg") * 1e6
    robust_std = 1.4826 * np.median(
        np.abs(data_uv - np.median(data_uv, axis=1, keepdims=True)), axis=1
    )
    log_std = np.log10(np.maximum(robust_std, np.finfo(float).tiny))
    med = np.median(log_std)
    mad = np.median(np.abs(log_std - med))
    robust_z = (log_std - med) / max(1.4826 * mad, np.finfo(float).eps)
    positions = np.array([
        qc_raw.info["chs"][qc_raw.ch_names.index(name)]["loc"][:3]
        for name in eeg_names
    ])
    correlation_data = data_uv[:, ::5]
    neighbor_correlations = []
    for index in range(len(eeg_names)):
        distances = np.linalg.norm(positions - positions[index], axis=1)
        nearest = np.argsort(distances)[1:7]
        neighbor_median = np.median(correlation_data[nearest], axis=0)
        correlation = np.corrcoef(correlation_data[index], neighbor_median)[0, 1]
        neighbor_correlations.append(float(correlation))
    neighbor_correlations = np.asarray(neighbor_correlations)
    corr_med = np.nanmedian(neighbor_correlations)
    corr_mad = np.nanmedian(np.abs(neighbor_correlations - corr_med))
    corr_z = (neighbor_correlations - corr_med) / max(1.4826 * corr_mad, np.finfo(float).eps)
    lof_bads, lof_scores = find_bad_channels_lof(
        qc_raw, picks="eeg", n_neighbors=20, threshold=2.0,
        return_scores=True, verbose="ERROR",
    )
    flat_bads = [name for name, value in zip(eeg_names, robust_std) if value < 0.1]
    scale_bads = [name for name, value in zip(eeg_names, robust_z) if abs(value) > 5.0]
    neighbor_bads = [
        name for name, correlation, z_value in zip(eeg_names, neighbor_correlations, corr_z)
        if correlation < 0.40 and z_value < -5.0
    ]
    # LOF is recorded as an advisory flag. On these data it can identify
    # physiologic frontal/ocular topographies as channel outliers, so it is not
    # sufficient by itself to trigger interpolation.
    bads = sorted(set(flat_bads) | set(scale_bads) | set(neighbor_bads), key=eeg_names.index)
    rows = []
    for name, rstd, rz, lof in zip(eeg_names, robust_std, robust_z, lof_scores):
        rows.append({
            "channel": name,
            "robust_std_uv": float(rstd),
            "robust_logstd_z": float(rz),
            "lof_score": float(lof),
            "lof_advisory_flag_threshold_2": name in lof_bads,
            "nearest6_median_correlation": float(neighbor_correlations[eeg_names.index(name)]),
            "neighbor_correlation_robust_z": float(corr_z[eeg_names.index(name)]),
            "neighbor_flag_corr_lt_0_4_and_z_lt_neg5": name in neighbor_bads,
            "flat_flag_lt_0_1uv": name in flat_bads,
            "scale_flag_abs_z_gt_5": name in scale_bads,
            "bad_final": name in bads,
        })
    return bads, pd.DataFrame(rows)


def prepare_analysis(raw, bads: list[str]):
    raw = raw.copy()
    raw.info["bads"] = bads
    raw.filter(1.0, 45.0, picks="eeg", fir_design="firwin", fir_window="hamming", verbose="ERROR")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return raw


def make_joint_fit(eo, ec, bads: list[str]):
    import mne

    joint = mne.concatenate_raws([eo.copy(), ec.copy()], preload=True, verbose="ERROR")
    joint.info["bads"] = bads
    joint.notch_filter([50.0, 100.0], picks="eeg", verbose="ERROR")
    joint.filter(1.0, 100.0, picks="eeg", fir_design="firwin", fir_window="hamming", verbose="ERROR")
    joint.set_eeg_reference("average", projection=False, verbose="ERROR")
    return joint


def save_component_figures(ica, figures_dir: Path, subject_id: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    result = ica.plot_components(show=False)
    figures = result if isinstance(result, list) else [result]
    paths = []
    for page, fig in enumerate(figures, start=1):
        path = figures_dir / f"{subject_id}_ica_topographies_page{page:02d}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def process_subject(source_project: Path, output_root: Path, subject_id: str, *,
                    force: bool, figures: bool, seed: int) -> dict:
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel.iclabel import iclabel_label_components

    start_clock = time.perf_counter()
    clean_dir = output_root / "preprocessing" / "clean_fif"
    ica_dir = output_root / "preprocessing" / "ica"
    subject_qc_dir = output_root / "preprocessing" / "qc" / "subjects"
    component_dir = output_root / "preprocessing" / "qc" / "components"
    channel_dir = output_root / "preprocessing" / "qc" / "channels"
    figure_dir = output_root / "preprocessing" / "qc" / "figures"
    work_dir = output_root / "work" / "brainvision_sidecars"
    for folder in (clean_dir, ica_dir, subject_qc_dir, component_dir, channel_dir, figure_dir, work_dir):
        folder.mkdir(parents=True, exist_ok=True)

    output_paths = preprocessing_output_paths(output_root, subject_id)
    eo_out = output_paths["eyes_open_fif"]
    ec_out = output_paths["eyes_closed_fif"]
    ica_out = output_paths["ica_fif"]
    qc_out = output_paths["qc_json"]
    if not force and subject_preprocessing_complete(output_root, subject_id):
        payload = json.loads(qc_out.read_text(encoding="utf-8"))
        payload["run_action"] = "SKIPPED_EXISTING"
        return payload

    source_folder = source_project / "data" / "lemon" / "raw_eeg" / subject_id / "RSEEG"
    stage_vhdr = normalized_brainvision_sidecars(source_folder, subject_id, work_dir)
    raw_header = mne.io.read_raw_brainvision(stage_vhdr, preload=False, verbose="ERROR")
    if raw_header.ch_names != EXPECTED_CHANNELS:
        raise RuntimeError(f"Unexpected channel order for {subject_id}")
    raw_header.set_channel_types({"VEOG": "eog"})
    raw_header.set_montage("standard_1005", match_case=False, on_missing="raise", verbose="ERROR")
    sfreq_source = float(raw_header.info["sfreq"])
    blocks, marker_audit = parse_marker_blocks(
        source_folder / f"{subject_id}.vmrk", subject_id, sfreq_source
    )
    if marker_audit["problems"]:
        raise RuntimeError("; ".join(marker_audit["problems"]))
    if max(item["start_s"] + 60.0 for item in blocks) > raw_header.times[-1] + 1.0 / sfreq_source:
        raise RuntimeError("A 60-s marker block exceeds the available binary payload")

    eo = load_condition_blocks(raw_header, blocks, "eyes_open", 250.0)
    ec = load_condition_blocks(raw_header, blocks, "eyes_closed", 250.0)
    joint_qc = mne.concatenate_raws([eo.copy(), ec.copy()], preload=True, verbose="ERROR")
    joint_qc.filter(1.0, 45.0, picks="eeg", fir_design="firwin", fir_window="hamming", verbose="ERROR")
    pre_metrics = channel_metrics(joint_qc, "pre_ica")
    bads, bad_detection = detect_bad_channels(joint_qc)
    if len(bads) > 10:
        raise RuntimeError(f"Too many automatically bad scalp channels ({len(bads)}): {bads}")

    fit_raw = make_joint_fit(eo, ec, bads)
    rank = int(mne.compute_rank(fit_raw, rank=None, verbose="ERROR")["eeg"])
    if rank < 40:
        raise RuntimeError(f"Abnormally low EEG rank: {rank}")
    ica = ICA(
        n_components=rank,
        method="infomax",
        fit_params={"extended": True},
        random_state=seed,
        max_iter=1024,
    )
    ica.fit(
        fit_raw, picks="eeg", decim=2,
        reject={"eeg": 500e-6}, tstep=2.0,
        reject_by_annotation=True, verbose="ERROR",
    )
    if int(ica.n_iter_) >= 1024:
        warnings.warn("ICA reached max_iter=1024; flagged for manual review")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = iclabel_label_components(fit_raw, ica, inplace=True, backend="onnx")
    iclabel_warnings = [str(item.message) for item in caught]
    labels = [ICLABEL_CLASSES[int(np.argmax(row))] for row in probabilities]
    max_probabilities = probabilities.max(axis=1)
    iclabel_excluded = [
        index for index, (label, probability) in enumerate(zip(labels, max_probabilities))
        if label in ARTIFACT_CLASSES and float(probability) >= ICLABEL_THRESHOLD
    ]
    try:
        eog_indices, eog_scores = ica.find_bads_eog(
            fit_raw, ch_name="VEOG", threshold=3.0, measure="zscore", verbose="ERROR"
        )
        eog_scores = np.asarray(eog_scores, dtype=float)
    except Exception as error:
        eog_indices, eog_scores = [], np.full(rank, np.nan)
        iclabel_warnings.append(f"EOG correlation failed: {type(error).__name__}: {error}")
    # EOG z-score detection is retained as audit evidence, but long recordings
    # can flag modest correlations even for high-confidence brain components.
    # It therefore adds an exclusion only with convergent ocular evidence.
    eog_assisted_excluded = [
        index for index in eog_indices
        if index < len(eog_scores)
        and abs(float(eog_scores[index])) >= 0.50
        and float(probabilities[index, ICLABEL_CLASSES.index("eye blink")]) >= 0.30
    ]
    excluded = sorted(set(iclabel_excluded) | set(eog_assisted_excluded))
    ica.exclude = excluded

    component_rows = []
    for index, (label, probability_row) in enumerate(zip(labels, probabilities)):
        row = {
            "participant_id": subject_id,
            "component": index,
            "argmax_label": label,
            "argmax_probability": float(np.max(probability_row)),
            "iclabel_threshold": ICLABEL_THRESHOLD,
            "iclabel_excluded": index in iclabel_excluded,
            "eog_zscore_flag": index in eog_indices,
            "eog_assisted_excluded": index in eog_assisted_excluded,
            "eog_score": float(eog_scores[index]) if index < len(eog_scores) else np.nan,
            "excluded_final": index in excluded,
        }
        row.update({
            "p_" + name.replace(" ", "_"): float(value)
            for name, value in zip(ICLABEL_CLASSES, probability_row)
        })
        component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(
        component_dir / f"{subject_id}_ica_components.csv", index=False
    )

    eo_clean = prepare_analysis(eo, bads)
    ec_clean = prepare_analysis(ec, bads)
    ica.apply(eo_clean, exclude=excluded, verbose="ERROR")
    ica.apply(ec_clean, exclude=excluded, verbose="ERROR")
    if bads:
        eo_clean.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
        ec_clean.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
    post_joint = mne.concatenate_raws([eo_clean.copy(), ec_clean.copy()], preload=True, verbose="ERROR")
    post_metrics = channel_metrics(post_joint, "post_ica_interpolated")
    pd.concat([pre_metrics, post_metrics], ignore_index=True).assign(
        participant_id=subject_id
    ).to_csv(channel_dir / f"{subject_id}_channel_metrics.csv", index=False)
    bad_detection.assign(participant_id=subject_id).to_csv(
        channel_dir / f"{subject_id}_bad_channel_detection.csv", index=False
    )

    eo_clean.save(eo_out, overwrite=True, fmt="single", verbose="ERROR")
    ec_clean.save(ec_out, overwrite=True, fmt="single", verbose="ERROR")
    ica.save(ica_out, overwrite=True, verbose="ERROR")
    figure_paths = save_component_figures(ica, figure_dir, subject_id) if figures else []

    label_counts = Counter(labels)
    excluded_label_counts = Counter(labels[index] for index in excluded)
    payload = {
        "participant_id": subject_id,
        "status": "PASS" if int(ica.n_iter_) < 1024 else "PASS_MANUAL_REVIEW_ICA_CONVERGENCE",
        "run_action": "PROCESSED",
        "created_utc": utc_now(),
        "source_vhdr": f"<LEMON_DATA_ROOT>/data/lemon/raw_eeg/{subject_id}/RSEEG/{subject_id}.vhdr",
        "normalized_sidecar_vhdr": stage_vhdr.relative_to(output_root).as_posix(),
        "source_sfreq_hz": sfreq_source,
        "output_sfreq_hz": 250.0,
        "filter_analysis_hz": [1.0, 45.0],
        "filter_ica_hz": [1.0, 100.0],
        "ica_notch_hz": [50.0, 100.0],
        "reference": "average excluding detected bad EEG channels",
        "channels_total": 62,
        "scalp_eeg_channels": 61,
        "auxiliary_eog": "VEOG",
        "condition_duration_s": {"eyes_open": 480.0, "eyes_closed": 480.0},
        "marker_audit": marker_audit,
        "bad_channel_method": {
            "LOF": {"n_neighbors": 20, "threshold": 2.0},
            "LOF_role": "advisory only; not sufficient alone for interpolation",
            "flat_robust_std_uv_lt": 0.1,
            "robust_logstd_abs_z_gt": 5.0,
            "neighbor_rule": {"nearest_channels": 6, "correlation_lt": 0.40, "robust_z_lt": -5.0},
        },
        "bad_channels": bads,
        "n_bad_channels": len(bads),
        "interpolated_channels": bads,
        "ica_method": "extended infomax",
        "ica_seed": seed,
        "ica_decim": 2,
        "ica_max_iter": 1024,
        "ica_reject_eeg_uv": 500.0,
        "ica_rank": rank,
        "ica_n_components": int(ica.n_components_),
        "ica_n_iter": int(ica.n_iter_),
        "ica_n_samples_fit": int(ica.n_samples_),
        "iclabel_backend": "onnx",
        "iclabel_threshold": ICLABEL_THRESHOLD,
        "eog_assisted_rule": {"abs_correlation_gte": 0.50, "p_eye_blink_gte": 0.30},
        "iclabel_label_counts": dict(label_counts),
        "eog_flagged_components": list(map(int, eog_indices)),
        "eog_assisted_excluded_components": list(map(int, eog_assisted_excluded)),
        "iclabel_excluded_components": list(map(int, iclabel_excluded)),
        "excluded_components_final": list(map(int, excluded)),
        "n_excluded_components_final": len(excluded),
        "excluded_label_counts": dict(excluded_label_counts),
        "iclabel_warnings": iclabel_warnings,
        "outputs": {
            "eyes_open_fif": eo_out.relative_to(output_root).as_posix(),
            "eyes_open_sha256": sha256(eo_out),
            "eyes_closed_fif": ec_out.relative_to(output_root).as_posix(),
            "eyes_closed_sha256": sha256(ec_out),
            "ica_fif": ica_out.relative_to(output_root).as_posix(),
            "ica_sha256": sha256(ica_out),
            "component_csv": (component_dir / f"{subject_id}_ica_components.csv").relative_to(output_root).as_posix(),
            "component_figures": [Path(path).relative_to(output_root).as_posix() for path in figure_paths],
        },
        "elapsed_seconds": time.perf_counter() - start_clock,
    }
    write_json(qc_out, payload)
    return payload


def aggregate_outputs(output_root: Path, run_rows: list[dict], exclusions: list[dict]) -> dict:
    aggregate_dir = output_root / "preprocessing" / "qc"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(aggregate_dir / "preprocessing_run_manifest.csv", index=False)
    pd.DataFrame(exclusions).to_csv(aggregate_dir / "preprocessing_exclusions.csv", index=False)
    all_qc = []
    for path in sorted((aggregate_dir / "subjects").glob("*_preprocessing_qc.json")):
        all_qc.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "created_utc": utc_now(),
        "subjects_with_qc": len(all_qc),
        "status_counts": dict(Counter(item.get("status", "UNKNOWN") for item in all_qc)),
        "processed_this_run": sum(item.get("run_action") == "PROCESSED" for item in run_rows),
        "skipped_existing_this_run": sum(item.get("run_action") == "SKIPPED_EXISTING" for item in run_rows),
        "failed_this_run": sum(item.get("status") == "FAIL" for item in run_rows),
        "hard_exclusions": HARD_EXCLUSIONS,
        "recoverable_source_variants": {
            "sub-010020": "normalized historical DataFile/MarkerFile names in sidecar only",
            "sub-010193": "normalized historical DataFile/MarkerFile names in sidecar only",
            "sub-010126": "mapped alternating S208 trains to eyes_closed; source retained unchanged",
        },
        "expected_potentially_processable": 139,
    }
    write_json(aggregate_dir / "preprocessing_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", help="Comma-separated participant IDs; default is all processable")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-subjects",
        help="Comma-separated participant IDs to regenerate while resuming all others",
    )
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--seed", type=int, default=BASE_ICA_SEED)
    args = parser.parse_args()
    source_project = args.source_project.resolve()
    output_root = args.output_root.resolve()
    audit_csv = output_root / "audit" / "tables" / "source_subject_audit.csv"
    audit = pd.read_csv(audit_csv)
    usable_mask = (
        audit["status"].isin({"PASS", "RECOVERABLE"})
        & audit["disposition"].astype(str).str.startswith("PROCESS")
        & as_bool(audit["payload_divisible"])
        & as_bool(audit["channel_order_match"])
        & as_bool(audit["marker_ok"])
        & ~audit["participant_id"].isin(HARD_EXCLUSIONS)
    )
    usable = sorted(audit.loc[usable_mask, "participant_id"].tolist())
    if not args.subjects and len(usable) != 139:
        raise SystemExit(
            f"Expected 139 processable participants from the audited disposition, observed {len(usable)}"
        )
    force_subjects = {
        item.strip() for item in (args.force_subjects or "").split(",") if item.strip()
    }
    if args.subjects:
        requested = [item.strip() for item in args.subjects.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(usable))
        if unknown:
            raise SystemExit(f"Requested subjects are not processable: {unknown}")
        usable = requested
    if args.max_subjects > 0:
        pending = [
            subject_id for subject_id in usable
            if args.force
            or subject_id in force_subjects
            or not subject_preprocessing_complete(output_root, subject_id)
        ]
        usable = pending[: args.max_subjects]

    exclusions = [
        {"participant_id": sid, "reason": reason, "stage": "source_audit"}
        for sid, reason in HARD_EXCLUSIONS.items()
    ]
    rows = []
    log_path = output_root / "preprocessing" / "qc" / "preprocessing.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for index, subject_id in enumerate(usable, start=1):
            print(f"{index:03d}/{len(usable)} START {subject_id}", flush=True)
            log.write(f"{utc_now()} {index:03d}/{len(usable)} START {subject_id}\n")
            try:
                payload = process_subject(
                    source_project, output_root, subject_id,
                    force=args.force or subject_id in force_subjects,
                    figures=args.figures,
                    seed=(args.seed + int(subject_id.rsplit("-", 1)[-1])) % (2**32 - 1),
                )
                row = {
                    "participant_id": subject_id,
                    "status": payload["status"],
                    "run_action": payload["run_action"],
                    "n_bad_channels": payload.get("n_bad_channels"),
                    "ica_rank": payload.get("ica_rank"),
                    "ica_n_components": payload.get("ica_n_components"),
                    "ica_n_iter": payload.get("ica_n_iter"),
                    "n_excluded_components": payload.get("n_excluded_components_final"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                }
                message = (
                    f"{index:03d}/{len(usable)} {payload['run_action']} {subject_id} "
                    f"status={payload['status']} bads={payload.get('n_bad_channels')} "
                    f"ICA={payload.get('ica_n_components')} exclude={payload.get('n_excluded_components_final')} "
                    f"seconds={payload.get('elapsed_seconds', 0):.1f}"
                )
            except Exception as error:
                tb = traceback.format_exc()
                row = {
                    "participant_id": subject_id,
                    "status": "FAIL",
                    "run_action": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                error_path = output_root / "preprocessing" / "qc" / "subjects" / f"{subject_id}_error.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(tb, encoding="utf-8")
                message = f"{index:03d}/{len(usable)} FAIL {subject_id}: {type(error).__name__}: {error}"
            rows.append(row)
            print(message, flush=True)
            log.write(f"{utc_now()} {message}\n")
    summary = aggregate_outputs(output_root, rows, exclusions)
    print(json.dumps(summary, indent=2), flush=True)
    return 1 if summary["failed_this_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
