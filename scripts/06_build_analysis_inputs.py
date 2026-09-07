#!/usr/bin/env python3
"""Build the exact 1-45 Hz and canonical-band inputs used by the analysis.

PSD is estimated from 4-s Hamming periodograms (50% overlap; 0.25-Hz FFT
spacing). Channel-window QC is applied before averaging in linear units. The
88 contiguous 0.5-Hz bins span 1-45 Hz. Canonical bands are additive
integrals of those bins and are converted to dB only after linear averaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram


QC_REL = Path("analysis/QC_FREEZE/tables/participant_qc_master.csv")
FULL_CACHE_REL = Path("features/full_spectrum_EO_EC_reactivity_1.00_45.00Hz_bin0.50Hz.npz")
CANONICAL_CACHE_REL = Path("features/canonical_band_psd_EO_EC_reactivity_delta_theta_alpha_beta.npz")
LONG_CSV_REL = Path(
    "LEMON_139_participantes_61_canales_88_intervalos_"
    "EO_EC_reactividad_STAI_MSPSS_JMP.csv"
)
BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
)
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
BLOCK_SECONDS = 60.0
BLOCKS_PER_CONDITION = 8
MAX_ABS_UV = 250.0
MAX_PTP_UV = 500.0
MIN_ROBUST_STD_UV = 0.1
EXPECTED_SFREQ = 250.0
EXPECTED_SAMPLES = 120_000
EXPECTED_CHANNELS = 61
SPECTRAL_LOW_HZ = 1.0
SPECTRAL_HIGH_HZ = 45.0
BIN_WIDTH_HZ = 0.5
NATIVE_FFT_SPACING_HZ = 1.0 / WINDOW_SECONDS
HAMMING_ENBW_BINS = 1.36
EFFECTIVE_HAMMING_RESOLUTION_HZ = HAMMING_ENBW_BINS * NATIVE_FFT_SPACING_HZ


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "si", "sí"})


def zscore_ddof1(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    deviation = values.std(ddof=1)
    if not np.isfinite(deviation) or deviation <= 0:
        raise ValueError("A sample SD > 0 is required for z-standardization")
    return (values - values.mean()) / deviation


def resolve_recorded_path(project: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def load_primary_qc(project: Path, max_participants: int) -> tuple[pd.DataFrame, Path]:
    qc_path = project / QC_REL
    if not qc_path.is_file():
        raise FileNotFoundError(f"Frozen QC table not found: {qc_path}")
    qc = pd.read_csv(qc_path)
    required = {
        "participant_id", "stai_trait_total", "mspss_total", "sex", "partnered",
        "context_complete", "main_all_usable_included", "eyes_open_fif", "eyes_closed_fif",
        "eyes_open_sha256_recorded", "eyes_closed_sha256_recorded", "preprocessing_status",
    }
    missing = sorted(required - set(qc.columns))
    if missing:
        raise ValueError(f"Frozen QC columns missing: {missing}")
    qc = qc.loc[as_bool(qc["main_all_usable_included"])].copy()
    qc = qc.sort_values("participant_id").drop_duplicates("participant_id", keep="first")
    if max_participants:
        qc = qc.head(max_participants).copy()
    elif len(qc) != 139:
        raise ValueError(f"Expected 139 primary participants, observed {len(qc)}")
    for column in ("stai_trait_total", "mspss_total"):
        qc[column] = pd.to_numeric(qc[column], errors="raise")
    for column in ("eyes_open_fif", "eyes_closed_fif"):
        paths = [resolve_recorded_path(project, value) for value in qc[column]]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing {column} file; first: {missing_paths[0]}")
        qc[column] = [str(path) for path in paths]
    return qc.reset_index(drop=True), qc_path


def source_signature(qc: pd.DataFrame) -> str:
    lines = [
        f"range={SPECTRAL_LOW_HZ:.8f}-{SPECTRAL_HIGH_HZ:.8f}",
        f"window={WINDOW_SECONDS:.8f}|step={STEP_SECONDS:.8f}|bin={BIN_WIDTH_HZ:.8f}",
        "estimator=periodogram|window=hamming|detrend=constant|scaling=density",
        f"qc=max_abs<={MAX_ABS_UV}|ptp<={MAX_PTP_UV}|robust_sd>={MIN_ROBUST_STD_UV}",
    ]
    for row in qc.itertuples(index=False):
        lines.append(
            f"{row.participant_id}|{row.eyes_open_sha256_recorded}|{row.eyes_closed_sha256_recorded}"
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def validate_raw(raw: Any, expected_condition: str) -> Any:
    eeg = raw.copy().pick("eeg")
    if len(raw.ch_names) != 62 or len(eeg.ch_names) != EXPECTED_CHANNELS:
        raise ValueError(
            f"{expected_condition}: expected 62 total/61 EEG channels; "
            f"observed {len(raw.ch_names)}/{len(eeg.ch_names)}"
        )
    if not math.isclose(float(raw.info["sfreq"]), EXPECTED_SFREQ):
        raise ValueError(f"{expected_condition}: expected 250 Hz; observed {raw.info['sfreq']}")
    if raw.n_times != EXPECTED_SAMPLES:
        raise ValueError(f"{expected_condition}: expected 120000 samples; observed {raw.n_times}")
    descriptions = [str(value) for value in raw.annotations.description]
    expected_blocks = [f"block/{expected_condition}/{index:02d}" for index in range(1, 9)]
    actual_blocks = [value for value in descriptions if value.startswith("block/")]
    if actual_blocks != expected_blocks:
        raise ValueError(f"{expected_condition}: block annotations are not the frozen 8-block sequence")
    if float(eeg.info["highpass"]) > SPECTRAL_LOW_HZ + 1e-9:
        raise ValueError(f"{expected_condition}: high-pass excludes requested 1 Hz edge")
    if float(eeg.info["lowpass"]) < SPECTRAL_HIGH_HZ - 1e-9:
        raise ValueError(f"{expected_condition}: low-pass excludes requested 45 Hz edge")
    return eeg


def interval_mean_density(
    frequencies: np.ndarray,
    density: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Trapezoidal integral divided by width; endpoints are both included."""
    selected = (frequencies >= low_hz - 1e-12) & (frequencies <= high_hz + 1e-12)
    selected_frequencies = frequencies[selected]
    if len(selected_frequencies) < 2:
        raise ValueError(f"Insufficient FFT points in {low_hz}-{high_hz} Hz")
    if not (
        math.isclose(float(selected_frequencies[0]), low_hz, abs_tol=1e-12)
        and math.isclose(float(selected_frequencies[-1]), high_hz, abs_tol=1e-12)
    ):
        raise ValueError(f"Interval edges {low_hz}-{high_hz} Hz are off the FFT grid")
    return np.trapezoid(density[..., selected], selected_frequencies, axis=-1) / (high_hz - low_hz)


def condition_spectrum(fif_path: Path, condition: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    import mne

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    eeg = validate_raw(raw, condition)
    data_uv = eeg.get_data() * 1e6
    sfreq = float(eeg.info["sfreq"])
    window_n = int(round(WINDOW_SECONDS * sfreq))
    step_n = int(round(STEP_SECONDS * sfreq))
    block_n = int(round(BLOCK_SECONDS * sfreq))
    expected_fft = np.fft.rfftfreq(window_n, d=1.0 / sfreq)
    bin_edges = np.arange(
        SPECTRAL_LOW_HZ,
        SPECTRAL_HIGH_HZ + BIN_WIDTH_HZ / 2.0,
        BIN_WIDTH_HZ,
    )
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    density_sum = np.zeros((EXPECTED_CHANNELS, len(centers)), dtype=np.float64)
    accepted_count = np.zeros(EXPECTED_CHANNELS, dtype=np.int32)
    for block_index in range(BLOCKS_PER_CONDITION):
        block_start = block_index * block_n
        windows = np.stack(
            [
                data_uv[:, block_start + start : block_start + start + window_n]
                for start in range(0, block_n - window_n + 1, step_n)
            ],
            axis=0,
        )
        if windows.shape[0] != 29:
            raise RuntimeError(f"Expected 29 windows per block; observed {windows.shape[0]}")
        finite = np.isfinite(windows).all(axis=2)
        max_abs = np.max(np.abs(windows), axis=2)
        peak_to_peak = np.ptp(windows, axis=2)
        median = np.median(windows, axis=2, keepdims=True)
        robust_std = 1.4826 * np.median(np.abs(windows - median), axis=2)
        accepted = (
            finite
            & (max_abs <= MAX_ABS_UV)
            & (peak_to_peak <= MAX_PTP_UV)
            & (robust_std >= MIN_ROBUST_STD_UV)
        )
        frequencies, density = periodogram(
            windows,
            fs=sfreq,
            window="hamming",
            detrend="constant",
            return_onesided=True,
            scaling="density",
            axis=-1,
        )
        if not np.allclose(frequencies, expected_fft, rtol=0.0, atol=1e-12):
            raise RuntimeError("Unexpected periodogram frequency grid")
        binned = np.empty((29, EXPECTED_CHANNELS, len(centers)), dtype=np.float64)
        for index, (low_hz, high_hz) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            binned[:, :, index] = interval_mean_density(frequencies, density, low_hz, high_hz)
        density_sum += np.sum(binned * accepted[:, :, None], axis=0)
        accepted_count += accepted.sum(axis=0).astype(np.int32)
    if np.any(accepted_count <= 0):
        missing = np.asarray(eeg.ch_names)[accepted_count <= 0].tolist()
        raise ValueError(f"{condition}: channels without accepted windows: {missing}")
    mean_linear_density = density_sum / accepted_count[:, None]
    mean_db_density = 10.0 * np.log10(np.maximum(mean_linear_density, np.finfo(float).tiny))
    if not np.isfinite(mean_db_density).all():
        raise ValueError(f"{condition}: non-finite PSD values")
    return centers, list(eeg.ch_names), mean_db_density, accepted_count


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def build_or_load_full_cache(
    project: Path,
    output_root: Path,
    qc: pd.DataFrame,
    force: bool,
) -> tuple[dict[str, np.ndarray], Path]:
    cache_path = output_root / FULL_CACHE_REL
    signature = source_signature(qc)
    rebuild = force or not cache_path.is_file()
    if not rebuild:
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                rebuild = not (
                    {"psd_open", "psd_closed", "reactivity"}.issubset(cached.files)
                    and str(cached["source_signature"].item()) == signature
                    and cached["participant_ids"].astype(str).tolist()
                    == qc["participant_id"].astype(str).tolist()
                )
        except Exception:
            rebuild = True
    if rebuild:
        participants = qc["participant_id"].astype(str).tolist()
        open_rows: list[np.ndarray] = []
        closed_rows: list[np.ndarray] = []
        open_counts: list[np.ndarray] = []
        closed_counts: list[np.ndarray] = []
        reference_channels: list[str] | None = None
        reference_frequencies: np.ndarray | None = None
        started = time.perf_counter()
        for index, row in enumerate(qc.itertuples(index=False), start=1):
            frequencies_open, channels_open, spectrum_open, count_open = condition_spectrum(
                Path(row.eyes_open_fif), "eyes_open"
            )
            frequencies_closed, channels_closed, spectrum_closed, count_closed = condition_spectrum(
                Path(row.eyes_closed_fif), "eyes_closed"
            )
            if channels_open != channels_closed or not np.array_equal(frequencies_open, frequencies_closed):
                raise ValueError(f"EO/EC structure mismatch for {row.participant_id}")
            if reference_channels is None:
                reference_channels = channels_open
                reference_frequencies = frequencies_open
            if channels_open != reference_channels or not np.array_equal(frequencies_open, reference_frequencies):
                raise ValueError(f"Cross-participant channel/frequency mismatch for {row.participant_id}")
            open_rows.append(spectrum_open.T)
            closed_rows.append(spectrum_closed.T)
            open_counts.append(count_open)
            closed_counts.append(count_closed)
            print(
                f"SPECTRUM {index:03d}/{len(qc)} {row.participant_id} elapsed_s={time.perf_counter()-started:.1f}",
                flush=True,
            )
        psd_open = np.stack(open_rows)
        psd_closed = np.stack(closed_rows)
        reactivity = psd_closed - psd_open
        save_npz_atomic(
            cache_path,
            participant_ids=np.asarray(participants, dtype="U32"),
            channels=np.asarray(reference_channels, dtype="U16"),
            frequencies=np.asarray(reference_frequencies, dtype=np.float64),
            psd_open=psd_open.astype(np.float64),
            psd_closed=psd_closed.astype(np.float64),
            reactivity=reactivity.astype(np.float64),
            counts_open=np.stack(open_counts).astype(np.int32),
            counts_closed=np.stack(closed_counts).astype(np.int32),
            source_signature=np.asarray(signature, dtype="U64"),
            fmin=np.asarray(SPECTRAL_LOW_HZ),
            fmax=np.asarray(SPECTRAL_HIGH_HZ),
            native_fft_spacing_hz=np.asarray(NATIVE_FFT_SPACING_HZ),
            effective_hamming_resolution_hz=np.asarray(EFFECTIVE_HAMMING_RESOLUTION_HZ),
            spectral_bin_width_hz=np.asarray(BIN_WIDTH_HZ),
        )
        metadata = {
            "created_utc": utc_now(),
            "cache": FULL_CACHE_REL.as_posix(),
            "participants": len(participants),
            "channels": len(reference_channels or []),
            "frequency_bins": len(reference_frequencies) if reference_frequencies is not None else 0,
            "frequency_range_hz": [SPECTRAL_LOW_HZ, SPECTRAL_HIGH_HZ],
            "native_fft_spacing_hz": NATIVE_FFT_SPACING_HZ,
            "hamming_enbw_bins": HAMMING_ENBW_BINS,
            "effective_hamming_resolution_hz": EFFECTIVE_HAMMING_RESOLUTION_HZ,
            "spectral_bin_width_hz": BIN_WIDTH_HZ,
            "window_seconds": WINDOW_SECONDS,
            "overlap_fraction": 0.5,
            "accepted_windows_expected_per_condition": 232,
            "psd_formula": "mean accepted-window linear density after trapezoidal integration within each 0.5-Hz interval, then 10*log10",
            "reactivity_formula": "EC_dB - EO_dB",
            "source_signature": signature,
        }
        cache_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    with np.load(cache_path, allow_pickle=False) as cached:
        arrays = {name: cached[name].copy() for name in cached.files}
    expected_shape = (len(qc), 88, EXPECTED_CHANNELS)
    for name in ("psd_open", "psd_closed", "reactivity"):
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{name} shape {arrays[name].shape}; expected {expected_shape}")
    identity_error = float(np.max(np.abs(arrays["reactivity"] - (arrays["psd_closed"] - arrays["psd_open"]))))
    if identity_error > 1e-12:
        raise ValueError(f"Full-frequency EC-EO identity error: {identity_error:.3e}")
    return arrays, cache_path


def build_canonical_cache(output_root: Path, full: dict[str, np.ndarray]) -> Path:
    frequencies = full["frequencies"].astype(float)
    lower = frequencies - BIN_WIDTH_HZ / 2.0
    upper = frequencies + BIN_WIDTH_HZ / 2.0
    open_linear = np.power(10.0, full["psd_open"].astype(float) / 10.0)
    closed_linear = np.power(10.0, full["psd_closed"].astype(float) / 10.0)
    open_bands: list[np.ndarray] = []
    closed_bands: list[np.ndarray] = []
    for _name, low_hz, high_hz in BANDS:
        selected = (lower >= low_hz - 1e-12) & (upper <= high_hz + 1e-12)
        expected_bins = int(round((high_hz - low_hz) / BIN_WIDTH_HZ))
        if int(selected.sum()) != expected_bins:
            raise ValueError(f"Unexpected number of 0.5-Hz intervals for {low_hz}-{high_hz} Hz")
        width = high_hz - low_hz
        open_band = np.sum(open_linear[:, selected, :] * BIN_WIDTH_HZ, axis=1) / width
        closed_band = np.sum(closed_linear[:, selected, :] * BIN_WIDTH_HZ, axis=1) / width
        open_bands.append(10.0 * np.log10(np.maximum(open_band, np.finfo(float).tiny)))
        closed_bands.append(10.0 * np.log10(np.maximum(closed_band, np.finfo(float).tiny)))
    psd_open = np.stack(open_bands, axis=1)
    psd_closed = np.stack(closed_bands, axis=1)
    reactivity = psd_closed - psd_open
    cache_path = output_root / CANONICAL_CACHE_REL
    save_npz_atomic(
        cache_path,
        participant_ids=full["participant_ids"],
        channels=full["channels"],
        bands=np.asarray([item[0] for item in BANDS], dtype="U16"),
        band_lows=np.asarray([item[1] for item in BANDS], dtype=np.float64),
        band_highs=np.asarray([item[2] for item in BANDS], dtype=np.float64),
        psd_open=psd_open.astype(np.float64),
        psd_closed=psd_closed.astype(np.float64),
        reactivity=reactivity.astype(np.float64),
        counts_open=full["counts_open"].astype(np.int32),
        counts_closed=full["counts_closed"].astype(np.int32),
        source_signature=full["source_signature"],
        native_fft_spacing_hz=np.asarray(NATIVE_FFT_SPACING_HZ),
        effective_hamming_resolution_hz=np.asarray(EFFECTIVE_HAMMING_RESOLUTION_HZ),
    )
    metadata = {
        "created_utc": utc_now(),
        "cache": CANONICAL_CACHE_REL.as_posix(),
        "participants": int(len(full["participant_ids"])),
        "channels": int(len(full["channels"])),
        "bands": len(BANDS),
        "band_definitions_hz": [
            {"band": name, "low_hz": low, "high_hz": high, "width_hz": high - low}
            for name, low, high in BANDS
        ],
        "native_fft_spacing_hz": NATIVE_FFT_SPACING_HZ,
        "hamming_enbw_bins": HAMMING_ENBW_BINS,
        "effective_hamming_resolution_hz": EFFECTIVE_HAMMING_RESOLUTION_HZ,
        "band_boundaries": "closed endpoints with additive trapezoidal intervals; shared endpoints carry zero area",
        "psd_formula": "sum 0.5-Hz linear-density interval integrals / band width, then 10*log10",
        "psd_units": "dB re 1 uV^2/Hz",
        "reactivity_formula": "PSD_EC_dB - PSD_EO_dB = 10*log10(PSD_EC_linear/PSD_EO_linear)",
        "source_signature": str(full["source_signature"].item()),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return cache_path


def prepare_psychometrics(qc: pd.DataFrame) -> pd.DataFrame:
    result = qc.copy()
    result["stai_z_all139"] = zscore_ddof1(result["stai_trait_total"])
    result["mspss_z_all139"] = zscore_ddof1(result["mspss_total"])
    context = as_bool(result["context_complete"])
    result["lmm_context_complete"] = context.astype(int)
    result["stai_z_lmm137"] = np.nan
    result["mspss_z_lmm137"] = np.nan
    result.loc[context, "stai_z_lmm137"] = zscore_ddof1(result.loc[context, "stai_trait_total"])
    result.loc[context, "mspss_z_lmm137"] = zscore_ddof1(result.loc[context, "mspss_total"])
    result["male"] = result["sex"].map({"female": 0.0, "male": 1.0})
    return result


def export_long_csv(output_root: Path, qc: pd.DataFrame, full: dict[str, np.ndarray]) -> tuple[Path, dict[str, object]]:
    qc = prepare_psychometrics(qc)
    frequencies = full["frequencies"].astype(float)
    channels = full["channels"].astype(str).tolist()
    psd_open = full["psd_open"].astype(float)
    psd_closed = full["psd_closed"].astype(float)
    reactivity = full["reactivity"].astype(float)
    counts_open = full["counts_open"].astype(int)
    counts_closed = full["counts_closed"].astype(int)
    output = output_root / LONG_CSV_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    columns = [
        "participant_id", "participant_index", "channel", "channel_index",
        "frequency_bin_index", "frequency_low_hz", "frequency_center_hz", "frequency_high_hz",
        "eyes_open_psd_db_uV2_per_Hz", "eyes_closed_psd_db_uV2_per_Hz",
        "reactivity_db_EC_minus_EO", "eyes_open_global_mean61_psd_db_uV2_per_Hz",
        "eyes_closed_global_mean61_psd_db_uV2_per_Hz",
        "reactivity_global_mean61_db_EC_minus_EO",
        "eyes_open_fullrange_power_db_uV2_1_45Hz",
        "eyes_closed_fullrange_power_db_uV2_1_45Hz",
        "reactivity_fullrange_db_EC_minus_EO",
        "eyes_open_accepted_windows_channel", "eyes_closed_accepted_windows_channel",
        "stai_trait_total", "mspss_total", "stai_z_all139", "mspss_z_all139",
        "sex", "male", "partnered", "lmm_context_complete", "stai_z_lmm137", "mspss_z_lmm137",
    ]
    lower = frequencies - BIN_WIDTH_HZ / 2.0
    upper = frequencies + BIN_WIDTH_HZ / 2.0
    rows_per_participant = len(frequencies) * len(channels)
    wrote_header = False
    started = time.perf_counter()
    for participant_index, row in enumerate(qc.itertuples(index=False), start=1):
        open_spectrum = psd_open[participant_index - 1].T
        closed_spectrum = psd_closed[participant_index - 1].T
        participant_reactivity = reactivity[participant_index - 1]
        open_linear = np.power(10.0, open_spectrum / 10.0)
        closed_linear = np.power(10.0, closed_spectrum / 10.0)
        open_global = open_spectrum.mean(axis=0)
        closed_global = closed_spectrum.mean(axis=0)
        reactivity_global = participant_reactivity.mean(axis=1)
        open_full = 10.0 * np.log10(np.sum(open_linear * BIN_WIDTH_HZ, axis=1))
        closed_full = 10.0 * np.log10(np.sum(closed_linear * BIN_WIDTH_HZ, axis=1))
        full_reactivity = closed_full - open_full
        frame = pd.DataFrame(
            {
                "participant_id": np.repeat(str(row.participant_id), rows_per_participant),
                "participant_index": np.repeat(participant_index, rows_per_participant),
                "channel": np.tile(np.asarray(channels, dtype=object), len(frequencies)),
                "channel_index": np.tile(np.arange(1, len(channels) + 1), len(frequencies)),
                "frequency_bin_index": np.repeat(np.arange(1, len(frequencies) + 1), len(channels)),
                "frequency_low_hz": np.repeat(lower, len(channels)),
                "frequency_center_hz": np.repeat(frequencies, len(channels)),
                "frequency_high_hz": np.repeat(upper, len(channels)),
                "eyes_open_psd_db_uV2_per_Hz": open_spectrum.T.reshape(-1),
                "eyes_closed_psd_db_uV2_per_Hz": closed_spectrum.T.reshape(-1),
                "reactivity_db_EC_minus_EO": participant_reactivity.reshape(-1),
                "eyes_open_global_mean61_psd_db_uV2_per_Hz": np.repeat(open_global, len(channels)),
                "eyes_closed_global_mean61_psd_db_uV2_per_Hz": np.repeat(closed_global, len(channels)),
                "reactivity_global_mean61_db_EC_minus_EO": np.repeat(reactivity_global, len(channels)),
                "eyes_open_fullrange_power_db_uV2_1_45Hz": np.tile(open_full, len(frequencies)),
                "eyes_closed_fullrange_power_db_uV2_1_45Hz": np.tile(closed_full, len(frequencies)),
                "reactivity_fullrange_db_EC_minus_EO": np.tile(full_reactivity, len(frequencies)),
                "eyes_open_accepted_windows_channel": np.tile(counts_open[participant_index - 1], len(frequencies)),
                "eyes_closed_accepted_windows_channel": np.tile(counts_closed[participant_index - 1], len(frequencies)),
                "stai_trait_total": np.repeat(float(row.stai_trait_total), rows_per_participant),
                "mspss_total": np.repeat(float(row.mspss_total), rows_per_participant),
                "stai_z_all139": np.repeat(float(row.stai_z_all139), rows_per_participant),
                "mspss_z_all139": np.repeat(float(row.mspss_z_all139), rows_per_participant),
                "sex": np.repeat(str(row.sex), rows_per_participant),
                "male": np.repeat(float(row.male), rows_per_participant),
                "partnered": np.repeat(row.partnered, rows_per_participant),
                "lmm_context_complete": np.repeat(int(row.lmm_context_complete), rows_per_participant),
                "stai_z_lmm137": np.repeat(row.stai_z_lmm137, rows_per_participant),
                "mspss_z_lmm137": np.repeat(row.mspss_z_lmm137, rows_per_participant),
            },
            columns=columns,
        )
        frame.to_csv(
            temporary,
            mode="a",
            header=not wrote_header,
            index=False,
            encoding="utf-8",
            float_format="%.12g",
            lineterminator="\n",
        )
        wrote_header = True
        print(
            f"EXPORT {participant_index:03d}/{len(qc)} {row.participant_id} "
            f"rows={participant_index*rows_per_participant:,} elapsed_s={time.perf_counter()-started:.1f}",
            flush=True,
        )
    os.replace(temporary, output)
    audit = validate_long_csv(output, len(qc))
    audit.update({"output": LONG_CSV_REL.as_posix(), "bytes": output.stat().st_size, "sha256": sha256_file(output)})
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return output, audit


def validate_long_csv(path: Path, participants_expected: int) -> dict[str, object]:
    total_rows = 0
    participants: set[str] = set()
    channels: set[str] = set()
    frequencies: set[float] = set()
    max_identity_error = 0.0
    nonfinite_measurements = 0
    measurement_columns = [
        "eyes_open_psd_db_uV2_per_Hz", "eyes_closed_psd_db_uV2_per_Hz",
        "reactivity_db_EC_minus_EO", "eyes_open_global_mean61_psd_db_uV2_per_Hz",
        "eyes_closed_global_mean61_psd_db_uV2_per_Hz",
        "reactivity_global_mean61_db_EC_minus_EO",
        "eyes_open_fullrange_power_db_uV2_1_45Hz",
        "eyes_closed_fullrange_power_db_uV2_1_45Hz",
        "reactivity_fullrange_db_EC_minus_EO", "stai_trait_total", "mspss_total",
    ]
    for chunk in pd.read_csv(path, chunksize=100_000):
        total_rows += len(chunk)
        participants.update(chunk["participant_id"].astype(str))
        channels.update(chunk["channel"].astype(str))
        frequencies.update(chunk["frequency_center_hz"].astype(float))
        error = np.abs(
            chunk["reactivity_db_EC_minus_EO"].to_numpy(float)
            - (
                chunk["eyes_closed_psd_db_uV2_per_Hz"].to_numpy(float)
                - chunk["eyes_open_psd_db_uV2_per_Hz"].to_numpy(float)
            )
        )
        max_identity_error = max(max_identity_error, float(error.max()))
        nonfinite_measurements += int((~np.isfinite(chunk[measurement_columns].to_numpy(float))).sum())
    expected_rows = participants_expected * EXPECTED_CHANNELS * 88
    if total_rows != expected_rows:
        raise ValueError(f"Long table rows {total_rows}; expected {expected_rows}")
    if len(participants) != participants_expected or len(channels) != 61 or len(frequencies) != 88:
        raise ValueError("Long table dimensions are inconsistent")
    if max_identity_error > 1e-8 or nonfinite_measurements:
        raise ValueError(
            f"Long table validation failed: EC-EO error={max_identity_error:.3e}, nonfinite={nonfinite_measurements}"
        )
    return {
        "rows": total_rows,
        "columns": 29,
        "participants": len(participants),
        "channels": len(channels),
        "frequency_bins": len(frequencies),
        "max_serialized_EC_minus_EO_error": max_identity_error,
        "nonfinite_measurements": nonfinite_measurements,
    }


def validate_against_references(
    canonical_path: Path,
    long_path: Path | None,
    reference_canonical: Path | None,
    reference_fullfreq: Path | None,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    if reference_canonical:
        with np.load(canonical_path, allow_pickle=False) as observed, np.load(reference_canonical, allow_pickle=False) as reference:
            observed_ids = observed["participant_ids"].astype(str).tolist()
            reference_ids = reference["participant_ids"].astype(str).tolist()
            indices = [reference_ids.index(identifier) for identifier in observed_ids]
            for name in ("psd_open", "psd_closed", "reactivity"):
                value = float(np.max(np.abs(observed[name].astype(float) - reference[name].astype(float)[indices])))
                errors[f"canonical_{name}"] = value
                if value > 1e-10:
                    raise ValueError(f"Reference mismatch for {name}: {value:.3e}")
    if reference_fullfreq:
        if long_path is None:
            raise ValueError("--reference-fullfreq requires long-CSV generation")
        participants = pd.read_csv(long_path, usecols=["participant_id"])["participant_id"].nunique()
        rows = participants * EXPECTED_CHANNELS * 88
        usecols = [
            "participant_id", "channel", "frequency_center_hz",
            "eyes_open_psd_db_uV2_per_Hz", "eyes_closed_psd_db_uV2_per_Hz",
            "reactivity_db_EC_minus_EO", "eyes_open_fullrange_power_db_uV2_1_45Hz",
            "eyes_closed_fullrange_power_db_uV2_1_45Hz", "reactivity_fullrange_db_EC_minus_EO",
            "eyes_open_accepted_windows_channel", "eyes_closed_accepted_windows_channel",
        ]
        observed = pd.read_csv(long_path, usecols=usecols, nrows=rows)
        reference = pd.read_csv(reference_fullfreq, usecols=usecols, nrows=rows)
        keys = ["participant_id", "channel", "frequency_center_hz"]
        if not observed[keys].equals(reference[keys]):
            raise ValueError("Reference full-frequency row keys do not match")
        for column in usecols:
            if column in keys:
                continue
            value = float(np.max(np.abs(observed[column].to_numpy(float) - reference[column].to_numpy(float))))
            errors[f"fullfreq_{column}"] = value
            tolerance = 0.0 if "accepted_windows" in column else 1e-8
            if value > tolerance:
                raise ValueError(f"Reference mismatch for {column}: {value:.3e}")
    return errors


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=repository / "workdir", help="Root containing frozen QC and cleaned FIF files")
    parser.add_argument("--output-root", type=Path, default=None, help="Where derived inputs are written; defaults to project root")
    parser.add_argument("--max-participants", type=int, default=0, help="Smoke-test prefix; 0 enforces all 139")
    parser.add_argument("--force-spectral-cache", action="store_true")
    parser.add_argument("--skip-long-csv", action="store_true")
    parser.add_argument("--reference-canonical", type=Path)
    parser.add_argument("--reference-fullfreq", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.expanduser().resolve()
    output_root = (args.output_root or project).expanduser().resolve()
    qc, qc_path = load_primary_qc(project, args.max_participants)
    full, full_cache_path = build_or_load_full_cache(
        project, output_root, qc, args.force_spectral_cache
    )
    canonical_path = build_canonical_cache(output_root, full)
    long_path: Path | None = None
    long_audit: dict[str, object] | None = None
    if not args.skip_long_csv:
        long_path, long_audit = export_long_csv(output_root, qc, full)
    reference_errors = validate_against_references(
        canonical_path,
        long_path,
        args.reference_canonical.resolve() if args.reference_canonical else None,
        args.reference_fullfreq.resolve() if args.reference_fullfreq else None,
    )
    summary = {
        "status": "PASS",
        "created_utc": utc_now(),
        "participants": len(qc),
        "qc_source": QC_REL.as_posix(),
        "full_cache": str(full_cache_path.relative_to(output_root).as_posix()),
        "full_cache_sha256": sha256_file(full_cache_path),
        "canonical_cache": str(canonical_path.relative_to(output_root).as_posix()),
        "canonical_cache_sha256": sha256_file(canonical_path),
        "long_csv": LONG_CSV_REL.as_posix() if long_path else None,
        "long_csv_audit": long_audit,
        "reference_max_errors": reference_errors,
        "portable_hash_policy": "structural and numeric validation; byte identity is not required across platforms",
    }
    summary_path = output_root / "features" / "analysis_input_build_audit.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
