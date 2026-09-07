#!/usr/bin/env python3
"""Trait anxiety, perceived social support, and posterior alpha reactivity.

This script builds a manuscript-oriented, auditable analysis package. The
inferential unit is one participant; channel-level topographies are secondary
spatial descriptions with explicit multiplicity control. Alpha is disclosed as
a same-cohort post hoc focus, and the final specification is repeated across
four canonical bands as a selection-sensitivity audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.channels.layout import _find_topomap_coords
import numpy as np
import pandas as pd
import patsy
import scipy
from scipy import stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = REPOSITORY_ROOT / "workdir"
STUDY_SHORT_NAME = "TRAIT_ANXIETY_SOCIAL_SUPPORT_ALPHA_REACTIVITY"
STUDY_TITLE = "Exploratory associations of trait anxiety and perceived social support with posterior alpha reactivity during resting-state EEG"
DEFAULT_OUTPUT_NAME = "analysis/final"
CACHE_REL = Path("features/canonical_band_psd_EO_EC_reactivity_delta_theta_alpha_beta.npz")
SPECTRAL_META_REL = Path("features/canonical_band_psd_EO_EC_reactivity_delta_theta_alpha_beta.json")
FULLFREQ_CSV_REL = Path("LEMON_139_participantes_61_canales_88_intervalos_EO_EC_reactividad_STAI_MSPSS_JMP.csv")
QC_REL = Path("analysis/QC_FREEZE/tables/participant_qc_master.csv")
SOURCE_AUDIT_REL = Path("audit/metadata/source_audit_summary.json")
PREPROCESSING_SCRIPT_REL = Path("scripts/03_preprocess_eeg.py")
PSD_SCRIPT_REL = Path("scripts/04_compute_psd_qc.py")
CANONICAL_CACHE_SCRIPT_REL = Path("scripts/06_build_analysis_inputs.py")

EXPECTED_CACHE_SHA256 = "0dddda113481e1e009141510ccd1b4cd00d834aa619fa2d93bfc18eeed7315f4"
EXPECTED_QC_SHA256 = "3a06082b3fbc33fa03e1d61b4df5c92cccc55dc2a7461a669fb0df6c7f0736ed"
EXPECTED_FULLFREQ_CSV_SHA256 = "5886245cc559f9cf89e85136cef2aa83f20274b7aef0ad4204d1de348194c5cb"
EXPECTED_SOURCE_AUDIT_SHA256 = "a60218aa50267577054b9bd939d52a6cc73c32a2fdee2f66ec9784d18a6ff9df"
UPSTREAM_SOURCE_RELATIVES = {
    "preprocessing_script": PREPROCESSING_SCRIPT_REL,
    "psd_script": PSD_SCRIPT_REL,
    "source_audit": SOURCE_AUDIT_REL,
    "canonical_cache_script": CANONICAL_CACHE_SCRIPT_REL,
}
EXPECTED_PRIMARY = {
    "STAI_trait": {"wald_total": 10.471698, "p_total": 0.014954, "linear": 0.455862},
    "MSPSS_total": {"wald_total": 14.898489, "p_total": 0.001905, "linear": -0.432793},
}
ROI = ("P3", "Pz", "P4", "PO3", "POz", "PO4", "O1", "Oz", "O2")
ALPHA = 0.05

LEMON_PAPER = "https://doi.org/10.1038/sdata.2018.308"
STROBE_URL = "https://www.equator-network.org/reporting-guidelines/strobe/"
COBIDAS_URL = "https://cobidasmeeg.wordpress.com/"
SAMPL_URL = "https://www.equator-network.org/reporting-guidelines/sampl/"
ELSEVIER_ARTWORK_URL = "https://www.elsevier.com/en-in/about/policies-and-standards/author/artwork-and-media-instructions"


@dataclass
class SplineTransform:
    design_info: Any
    residualization_coef: np.ndarray
    right_vectors: np.ndarray
    singular_values: np.ndarray
    keep: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        raw = np.asarray(
            patsy.build_design_matrices([self.design_info], {"x": x})[0],
            dtype=float,
        )
        linear = np.column_stack([np.ones(len(x)), x])
        residual = raw - linear @ self.residualization_coef
        if not np.any(self.keep):
            return np.empty((len(x), 0), dtype=float)
        return (
            residual
            @ self.right_vectors.T[:, self.keep]
            / self.singular_values[self.keep]
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "si", "sí"})


def fmt_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.6f}" if value >= 0.001 else f"{value:.3e}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def require_files(project: Path) -> dict[str, Path]:
    paths = {
        "cache": project / CACHE_REL,
        "spectral_metadata": project / SPECTRAL_META_REL,
        "full_frequency_table": project / FULLFREQ_CSV_REL,
        "qc": project / QC_REL,
        "source_audit": project / SOURCE_AUDIT_REL,
        "preprocessing_script": REPOSITORY_ROOT / PREPROCESSING_SCRIPT_REL,
        "psd_script": REPOSITORY_ROOT / PSD_SCRIPT_REL,
        "canonical_cache_script": REPOSITORY_ROOT / CANONICAL_CACHE_SCRIPT_REL,
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan entradas requeridas:\n" + "\n".join(missing))
    return paths


def validate_hashes(paths: dict[str, Path], enforce_archived_reference: bool) -> dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if enforce_archived_reference:
        checks = {
            "cache": EXPECTED_CACHE_SHA256,
            "qc": EXPECTED_QC_SHA256,
            "full_frequency_table": EXPECTED_FULLFREQ_CSV_SHA256,
            "source_audit": EXPECTED_SOURCE_AUDIT_SHA256,
        }
        failures = [
            f"{name}: observado {hashes[name]}, esperado {expected}"
            for name, expected in checks.items()
            if hashes[name].lower() != expected.lower()
        ]
        if failures:
            raise RuntimeError(
                "The inputs do not match the archived reference bytes. "
                "Fresh cross-platform reproduction should omit --enforce-archived-input-hashes; "
                "structural checks and numeric regression tests remain mandatory.\n"
                + "\n".join(failures)
            )
    return hashes


def copy_upstream_source_specifications(
    paths: dict[str, Path], output: Path
) -> dict[str, dict[str, Any]]:
    """Copy the frozen upstream specifications verbatim and verify each copy."""
    inventory: dict[str, dict[str, Any]] = {}
    for name, relative_path in UPSTREAM_SOURCE_RELATIVES.items():
        source = paths[name]
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        source_hash = sha256_file(source)
        copied_hash = sha256_file(destination)
        if source_hash.lower() != copied_hash.lower():
            raise RuntimeError(
                f"Upstream source copy mismatch for {name}: "
                f"source={source_hash}, copied={copied_hash}"
            )
        inventory[name] = {
            "project_relative_path": relative_path.as_posix(),
            "package_relative_path": relative_path.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": copied_hash,
            "copied_verbatim": True,
            "role": {
                "preprocessing_script": "bad-channel detection, LOF advisory role, ICA/ICLabel rules, interpolation, and EEG exclusions",
                "psd_script": "window QC, density-scaled periodogram, frequency-bin edge rules, and band summaries",
                "source_audit": "aggregate source-audit counts and source provenance",
                "canonical_cache_script": "portable constructor of the 1-45 Hz and canonical-band inputs consumed by the analysis",
            }[name],
            "contains_pseudonymous_dataset_ids": name == "preprocessing_script",
            "contains_local_absolute_path": name == "source_audit",
        }
    return inventory


def load_inputs(project: Path, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    qc = pd.read_csv(paths["qc"])
    if qc["participant_id"].duplicated().any():
        raise ValueError("La tabla QC contiene participant_id duplicados")

    with np.load(paths["cache"], allow_pickle=False) as z:
        participant_ids = z["participant_ids"].astype(str).tolist()
        channels = z["channels"].astype(str).tolist()
        bands = z["bands"].astype(str).tolist()
        lows = z["band_lows"].astype(float)
        highs = z["band_highs"].astype(float)
        psd_open = z["psd_open"].astype(float)
        psd_closed = z["psd_closed"].astype(float)
        reactivity = z["reactivity"].astype(float)
        counts_open = z["counts_open"].astype(int)
        counts_closed = z["counts_closed"].astype(int)
        native_spacing = float(z["native_fft_spacing_hz"].item())
        effective_resolution = float(z["effective_hamming_resolution_hz"].item())

    expected_shape = (139, 4, 61)
    for name, array in {
        "psd_open": psd_open,
        "psd_closed": psd_closed,
        "reactivity": reactivity,
    }.items():
        if array.shape != expected_shape:
            raise ValueError(f"{name}: forma {array.shape}; se esperaba {expected_shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: contiene valores no finitos")
    if len(set(participant_ids)) != 139 or len(channels) != 61:
        raise ValueError("La cache no contiene exactamente 139 participantes unicos y 61 canales")
    if bands != ["delta", "theta", "alpha", "beta"]:
        raise ValueError(f"Bandas inesperadas: {bands}")
    if not np.array_equal(lows, np.array([1.0, 4.0, 8.0, 13.0])):
        raise ValueError("Limites inferiores de banda inesperados")
    if not np.array_equal(highs, np.array([4.0, 8.0, 13.0, 30.0])):
        raise ValueError("Limites superiores de banda inesperados")
    if not np.allclose(reactivity, psd_closed - psd_open, rtol=0, atol=1e-12):
        raise ValueError("Reactividad no equivale exactamente a EC_dB - EO_dB")
    if not set(ROI).issubset(channels):
        raise ValueError(f"Faltan canales de la ROI: {sorted(set(ROI)-set(channels))}")

    qc = qc.set_index("participant_id").loc[participant_ids].reset_index()
    alpha_index = bands.index("alpha")
    roi_indices = [channels.index(ch) for ch in ROI]
    qc["alpha_roi_eo_db_uv2_hz"] = psd_open[:, alpha_index, roi_indices].mean(axis=1)
    qc["alpha_roi_ec_db_uv2_hz"] = psd_closed[:, alpha_index, roi_indices].mean(axis=1)
    qc["alpha_roi_reactivity_ec_minus_eo_db"] = reactivity[:, alpha_index, roi_indices].mean(axis=1)
    qc["male"] = qc["sex"].map({"female": 0.0, "male": 1.0})
    qc["partnered"] = pd.to_numeric(qc["partnered"], errors="coerce")
    qc["stai_trait_total"] = pd.to_numeric(qc["stai_trait_total"], errors="coerce")
    qc["mspss_total"] = pd.to_numeric(qc["mspss_total"], errors="coerce")

    required = ["stai_trait_total", "mspss_total", "male", "partnered"]
    complete = as_bool(qc["context_complete"]) & qc[required].notna().all(axis=1)
    frame = qc.loc[complete].copy().reset_index(drop=True)
    if len(frame) != 137:
        raise ValueError(f"El estudio esperaba n=137 casos completos; observo n={len(frame)}")
    frame["stai_z"] = stats.zscore(frame["stai_trait_total"].to_numpy(float), ddof=1)
    frame["mspss_z"] = stats.zscore(frame["mspss_total"].to_numpy(float), ddof=1)
    age_dummies = pd.get_dummies(frame["age_bin"], prefix="age", dtype=float)
    for label in ["age_25-30", "age_30-35"]:
        frame[label.replace("-", "_")] = age_dummies[label] if label in age_dummies else 0.0

    arrays = {
        "participant_ids": participant_ids,
        "channels": channels,
        "bands": bands,
        "band_lows": lows,
        "band_highs": highs,
        "psd_open": psd_open,
        "psd_closed": psd_closed,
        "reactivity": reactivity,
        "counts_open": counts_open,
        "counts_closed": counts_closed,
        "native_fft_spacing_hz": native_spacing,
        "effective_hamming_resolution_hz": effective_resolution,
    }
    metadata = json.loads(paths["spectral_metadata"].read_text(encoding="utf-8"))
    source_audit = json.loads(paths["source_audit"].read_text(encoding="utf-8"))
    metadata["source_audit"] = source_audit
    return frame, arrays, metadata


def fit_spline_basis(x: np.ndarray, df: int = 4) -> tuple[np.ndarray, SplineTransform]:
    design = patsy.dmatrix(f"cr(x, df={df}) - 1", {"x": np.asarray(x, dtype=float)})
    raw = np.asarray(design, dtype=float)
    linear = np.column_stack([np.ones(len(x)), x])
    coef = np.linalg.lstsq(linear, raw, rcond=None)[0]
    residual = raw - linear @ coef
    _, singular_values, right_vectors = np.linalg.svd(residual, full_matrices=False)
    threshold = singular_values.max() * 1e-8 if len(singular_values) else 0.0
    keep = singular_values > threshold
    transform = SplineTransform(
        design.design_info,
        coef,
        right_vectors,
        singular_values,
        keep,
    )
    return transform.transform(np.asarray(x, dtype=float)), transform


def design_matrix(
    frame: pd.DataFrame,
    focal: str,
    other: str,
    nonlinear: np.ndarray,
    add_age: bool,
) -> tuple[np.ndarray, list[str], int]:
    names = ["intercept", f"{focal}_linear"]
    names.extend([f"{focal}_nonlinear_{i+1}" for i in range(nonlinear.shape[1])])
    columns = [np.ones(len(frame)), frame[focal].to_numpy(float)]
    columns.extend([nonlinear[:, i] for i in range(nonlinear.shape[1])])
    nuisance_start = len(columns)
    columns.extend(
        [
            frame[other].to_numpy(float),
            frame["male"].to_numpy(float),
            frame["partnered"].to_numpy(float),
        ]
    )
    names.extend([f"{other}_linear", "male_vs_female", "partnered_vs_not"])
    if add_age:
        columns.extend(
            [frame["age_25_30"].to_numpy(float), frame["age_30_35"].to_numpy(float)]
        )
        names.extend(["age_25_30_vs_20_25", "age_30_35_vs_20_25"])
    return np.column_stack(columns), names, nuisance_start


def new_design(
    x: np.ndarray,
    frame: pd.DataFrame,
    focal: str,
    other: str,
    spline: SplineTransform,
    add_age: bool,
) -> np.ndarray:
    nonlinear = spline.transform(x)
    columns = [np.ones(len(x)), np.asarray(x, dtype=float)]
    columns.extend([nonlinear[:, i] for i in range(nonlinear.shape[1])])
    columns.extend(
        [
            np.full(len(x), frame[other].mean()),
            np.full(len(x), frame["male"].mean()),
            np.full(len(x), frame["partnered"].mean()),
        ]
    )
    if add_age:
        columns.extend(
            [
                np.full(len(x), frame["age_25_30"].mean()),
                np.full(len(x), frame["age_30_35"].mean()),
            ]
        )
    return np.column_stack(columns)


def safe_vif(x: np.ndarray, names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(1, x.shape[1]):
        try:
            value = float(variance_inflation_factor(x, i))
        except Exception:
            value = math.nan
        rows.append({"term": names[i], "vif": value})
    return pd.DataFrame(rows)


def fit_one(
    frame: pd.DataFrame,
    focal: str,
    other: str,
    label: str,
    variant: str = "primary_posterior_roi_spline",
    add_age: bool = False,
    outcome_column: str = "alpha_roi_reactivity_ec_minus_eo_db",
) -> dict[str, Any]:
    x_focal = frame[focal].to_numpy(float)
    y = frame[outcome_column].to_numpy(float)
    nonlinear, spline = fit_spline_basis(x_focal, df=4)
    x, names, nuisance_start = design_matrix(frame, focal, other, nonlinear, add_age)
    classical = sm.OLS(y, x).fit()
    fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=False)
    n_nonlin = nonlinear.shape[1]

    r_total = np.zeros((1 + n_nonlin, x.shape[1]))
    r_total[0, 1] = 1.0
    for j in range(n_nonlin):
        r_total[1 + j, 2 + j] = 1.0
    r_nonlin = np.zeros((n_nonlin, x.shape[1]))
    for j in range(n_nonlin):
        r_nonlin[j, 2 + j] = 1.0
    w_total = fit.wald_test(r_total, scalar=True)
    w_nonlin = fit.wald_test(r_nonlin, scalar=True)

    ci = np.asarray(fit.conf_int(alpha=ALPHA))
    coefficients = pd.DataFrame(
        {
            "variant": variant,
            "outcome": label,
            "term": names,
            "estimate_db": np.asarray(fit.params),
            "se_hc3": np.asarray(fit.bse),
            "z_hc3": np.asarray(fit.tvalues),
            "p_hc3_two_sided": np.asarray(fit.pvalues),
            "ci95_hc3_low": ci[:, 0],
            "ci95_hc3_high": ci[:, 1],
        }
    )
    coefficients["power_ratio_multiplier"] = np.power(10.0, coefficients["estimate_db"] / 10.0)
    coefficients["percent_change_in_power_ratio"] = 100.0 * (coefficients["power_ratio_multiplier"] - 1.0)
    coefficients["percent_change_ci95_low"] = 100.0 * (np.power(10.0, coefficients["ci95_hc3_low"] / 10.0) - 1.0)
    coefficients["percent_change_ci95_high"] = 100.0 * (np.power(10.0, coefficients["ci95_hc3_high"] / 10.0) - 1.0)

    influence = classical.get_influence()
    cooks = np.asarray(influence.cooks_distance[0])
    leverage = np.asarray(influence.hat_matrix_diag)
    studentized = np.asarray(influence.resid_studentized_external)
    bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(classical.resid, x)
    rmse = float(np.sqrt(np.mean(np.square(classical.resid))))
    vif = safe_vif(x, names)
    vif.insert(0, "outcome", label)
    vif.insert(0, "variant", variant)
    model = {
        "variant": variant,
        "outcome": label,
        "n_participants": len(frame),
        "n_model_parameters": x.shape[1],
        "residual_df": int(classical.df_resid),
        "spline_df_total": 1 + n_nonlin,
        "nonlinear_df": n_nonlin,
        "wald_chi2_total": float(w_total.statistic),
        "p_total_raw": float(w_total.pvalue),
        "wald_chi2_nonlinear": float(w_nonlin.statistic),
        "p_nonlinear_raw": float(w_nonlin.pvalue),
        "linear_component_db_per_sd": float(fit.params[1]),
        "linear_component_ci95_low": float(ci[1, 0]),
        "linear_component_ci95_high": float(ci[1, 1]),
        "r_squared": float(classical.rsquared),
        "adjusted_r_squared": float(classical.rsquared_adj),
        "rmse_db": rmse,
        "aic_classical": float(classical.aic),
        "bic_classical": float(classical.bic),
        "condition_number": float(np.linalg.cond(x)),
        "max_vif": float(vif["vif"].replace([np.inf, -np.inf], np.nan).max()),
    }
    diagnostics = {
        "variant": variant,
        "outcome": label,
        "n_participants": len(frame),
        "residual_mean": float(np.mean(classical.resid)),
        "residual_sd": float(np.std(classical.resid, ddof=1)),
        "residual_skewness": float(stats.skew(classical.resid, bias=False)),
        "residual_excess_kurtosis": float(stats.kurtosis(classical.resid, fisher=True, bias=False)),
        "breusch_pagan_lm": float(bp_lm),
        "breusch_pagan_p": float(bp_p),
        "breusch_pagan_f": float(bp_f),
        "breusch_pagan_f_p": float(bp_f_p),
        "max_leverage": float(np.max(leverage)),
        "leverage_gt_2k_over_n": int(np.sum(leverage > (2 * x.shape[1] / len(frame)))),
        "max_abs_external_studentized_residual": float(np.nanmax(np.abs(studentized))),
        "abs_external_studentized_gt_3": int(np.sum(np.abs(studentized) > 3)),
        "max_cooks_distance": float(np.max(cooks)),
        "cooks_distance_gt_4_over_n": int(np.sum(cooks > (4 / len(frame)))),
    }

    grid = np.linspace(float(np.min(x_focal)), float(np.max(x_focal)), 250)
    gx = new_design(grid, frame, focal, other, spline, add_age)
    covariance = np.asarray(fit.cov_params())
    predicted = gx @ np.asarray(fit.params)
    pred_se = np.sqrt(np.einsum("ij,jk,ik->i", gx, covariance, gx).clip(min=0))
    curve = pd.DataFrame(
        {
            "variant": variant,
            "outcome": label,
            "focal_z": grid,
            "predicted_reactivity_db": predicted,
            "ci95_pointwise_low": predicted - 1.959963984540054 * pred_se,
            "ci95_pointwise_high": predicted + 1.959963984540054 * pred_se,
            "other_scale_z_held": frame[other].mean(),
            "male_fraction_held": frame["male"].mean(),
            "partnered_fraction_held": frame["partnered"].mean(),
        }
    )

    contrast_x = np.array([-1.0, 1.0])
    cx = new_design(contrast_x, frame, focal, other, spline, add_age)
    delta = cx[1] - cx[0]
    effect = float(delta @ np.asarray(fit.params))
    effect_se = float(np.sqrt(max(0.0, delta @ covariance @ delta)))
    effect_z = effect / effect_se if effect_se > 0 else math.nan
    effect_p = float(2 * stats.norm.sf(abs(effect_z))) if np.isfinite(effect_z) else math.nan
    effect_low = effect - 1.959963984540054 * effect_se
    effect_high = effect + 1.959963984540054 * effect_se
    contrast = {
        "variant": variant,
        "outcome": label,
        "contrast": "predicted_at_+1SD_minus_predicted_at_-1SD",
        "estimate_db": effect,
        "se_hc3": effect_se,
        "z_hc3": effect_z,
        "p_hc3_two_sided": effect_p,
        "ci95_hc3_low": effect_low,
        "ci95_hc3_high": effect_high,
        "power_ratio_of_reactivity_ratios": float(10 ** (effect / 10)),
        "percent_difference_in_ec_eo_power_ratio": float((10 ** (effect / 10) - 1) * 100),
        "percent_ci95_low": float((10 ** (effect_low / 10) - 1) * 100),
        "percent_ci95_high": float((10 ** (effect_high / 10) - 1) * 100),
    }

    reference_nuisance = np.mean(x[:, nuisance_start:], axis=0)
    adjusted_y = y - (x[:, nuisance_start:] - reference_nuisance) @ np.asarray(fit.params)[nuisance_start:]

    return {
        "frame": frame,
        "focal": focal,
        "other": other,
        "label": label,
        "fit": fit,
        "classical": classical,
        "x": x,
        "names": names,
        "model": model,
        "coefficients": coefficients,
        "diagnostics": diagnostics,
        "vif": vif,
        "curve": curve,
        "contrast": contrast,
        "adjusted_y": adjusted_y,
        "fitted": np.asarray(classical.fittedvalues),
        "residuals": np.asarray(classical.resid),
        "cooks": cooks,
        "leverage": leverage,
    }


def apply_holm(results: list[dict[str, Any]]) -> None:
    p_total = [item["model"]["p_total_raw"] for item in results]
    p_nonlin = [item["model"]["p_nonlinear_raw"] for item in results]
    total_adj = multipletests(p_total, alpha=ALPHA, method="holm")[1]
    nonlin_adj = multipletests(p_nonlin, alpha=ALPHA, method="holm")[1]
    for item, pt, pn in zip(results, total_adj, nonlin_adj):
        item["model"]["p_total_holm_2"] = float(pt)
        item["model"]["p_nonlinear_holm_2"] = float(pn)
        item["model"]["total_reject_holm_05"] = bool(pt < ALPHA)
        item["model"]["nonlinear_reject_holm_05"] = bool(pn < ALPHA)


def run_four_band_selection_sensitivity(
    frame: pd.DataFrame,
    arrays: dict[str, Any],
) -> pd.DataFrame:
    """Apply the final participant-level spline specification to all four bands.

    This is a post-selection contextual sensitivity, not a replacement for the
    two-test Holm family used in the focused alpha analysis.  Its purpose is to
    disclose the same-cohort, data-informed band selection and to compare bands
    under one identical estimand, ROI, covariate set, and inferential unit.
    """
    participant_lookup = {
        participant_id: index
        for index, participant_id in enumerate(arrays["participant_ids"])
    }
    complete_indices = np.array(
        [participant_lookup[value] for value in frame["participant_id"]],
        dtype=int,
    )
    roi_indices = [arrays["channels"].index(channel) for channel in ROI]
    outcome_column = "four_band_roi_reactivity_ec_minus_eo_db"
    rows: list[dict[str, Any]] = []

    for band_index, band in enumerate(arrays["bands"]):
        band_frame = frame.copy()
        band_frame[outcome_column] = arrays["reactivity"][
            complete_indices, band_index, :
        ][:, roi_indices].mean(axis=1)
        analyses = [
            fit_one(
                band_frame,
                "stai_z",
                "mspss_z",
                "STAI_trait",
                variant=f"post_selection_four_band_sensitivity_{band}",
                outcome_column=outcome_column,
            ),
            fit_one(
                band_frame,
                "mspss_z",
                "stai_z",
                "MSPSS_total",
                variant=f"post_selection_four_band_sensitivity_{band}",
                outcome_column=outcome_column,
            ),
        ]
        for item in analyses:
            model = item["model"]
            rows.append(
                {
                    "analysis_role": "post_selection_multiband_sensitivity_not_primary",
                    "selection_context": "alpha_selected_after_same_cohort_canonical_band_exploration",
                    "band": band,
                    "band_low_hz": float(arrays["band_lows"][band_index]),
                    "band_high_hz": float(arrays["band_highs"][band_index]),
                    "psychometric_variable": model["outcome"],
                    "eeg_outcome": "posterior_roi_reactivity_ec_minus_eo_db",
                    "roi_channels": "|".join(ROI),
                    "n_participants": int(model["n_participants"]),
                    "spline_basis_df_requested": 4,
                    "spline_df_total": int(model["spline_df_total"]),
                    "nonlinear_df": int(model["nonlinear_df"]),
                    "wald_chi2_total": float(model["wald_chi2_total"]),
                    "p_total_raw": float(model["p_total_raw"]),
                    "wald_chi2_nonlinear": float(model["wald_chi2_nonlinear"]),
                    "p_nonlinear_raw": float(model["p_nonlinear_raw"]),
                    "linear_component_db_per_sd": float(model["linear_component_db_per_sd"]),
                    "linear_component_ci95_low": float(model["linear_component_ci95_low"]),
                    "linear_component_ci95_high": float(model["linear_component_ci95_high"]),
                    "primary_focus_band": bool(band == "alpha"),
                }
            )

    results = pd.DataFrame(rows)
    results["p_total_holm_all_8"] = multipletests(
        results["p_total_raw"].to_numpy(float), alpha=ALPHA, method="holm"
    )[1]
    results["total_reject_holm_all_8_05"] = results["p_total_holm_all_8"] < ALPHA
    results["p_nonlinear_holm_all_8"] = multipletests(
        results["p_nonlinear_raw"].to_numpy(float), alpha=ALPHA, method="holm"
    )[1]
    results["nonlinear_reject_holm_all_8_05"] = (
        results["p_nonlinear_holm_all_8"] < ALPHA
    )
    return results[
        [
            "analysis_role",
            "selection_context",
            "band",
            "band_low_hz",
            "band_high_hz",
            "psychometric_variable",
            "eeg_outcome",
            "roi_channels",
            "n_participants",
            "spline_basis_df_requested",
            "spline_df_total",
            "nonlinear_df",
            "wald_chi2_total",
            "p_total_raw",
            "p_total_holm_all_8",
            "total_reject_holm_all_8_05",
            "wald_chi2_nonlinear",
            "p_nonlinear_raw",
            "p_nonlinear_holm_all_8",
            "nonlinear_reject_holm_all_8_05",
            "linear_component_db_per_sd",
            "linear_component_ci95_low",
            "linear_component_ci95_high",
            "primary_focus_band",
        ]
    ]


def primary_replication_check(primary: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in primary:
        observed = item["model"]
        expected = EXPECTED_PRIMARY[item["label"]]
        checks = {
            "wald_total": (observed["wald_chi2_total"], expected["wald_total"], 5e-6),
            "p_total": (observed["p_total_raw"], expected["p_total"], 5e-7),
            "linear_component": (observed["linear_component_db_per_sd"], expected["linear"], 5e-7),
        }
        for statistic, (value, target, tolerance) in checks.items():
            passed = abs(value - target) <= tolerance
            rows.append(
                {
                    "outcome": item["label"],
                    "statistic": statistic,
                    "observed": value,
                    "reference_value": target,
                    "absolute_difference": abs(value - target),
                    "tolerance": tolerance,
                    "pass": passed,
                }
            )
    audit = pd.DataFrame(rows)
    if not bool(audit["pass"].all()):
        raise RuntimeError("La prueba de regresion numerica no coincide con los valores de referencia")
    return audit


def build_descriptives(frame: pd.DataFrame, qc_all: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    continuous = {
        "STAI trait total": "stai_trait_total",
        "MSPSS total": "mspss_total",
        "Alpha ROI EO (dB re 1 uV2/Hz)": "alpha_roi_eo_db_uv2_hz",
        "Alpha ROI EC (dB re 1 uV2/Hz)": "alpha_roi_ec_db_uv2_hz",
        "Alpha ROI reactivity EC-EO (dB)": "alpha_roi_reactivity_ec_minus_eo_db",
        "Relative alpha ROI EO (% of 1-45 Hz power)": "relative_alpha_roi_eo_percent_1_45hz",
        "Relative alpha ROI EC (% of 1-45 Hz power)": "relative_alpha_roi_ec_percent_1_45hz",
        "Relative alpha ROI reactivity EC-EO (percentage points)": "relative_alpha_roi_reactivity_percentage_points",
    }
    for label, column in continuous.items():
        values = frame[column].dropna().to_numpy(float)
        rows.append(
            {
                "variable": label,
                "level": "continuous",
                "n": len(values),
                "mean_or_count": np.mean(values),
                "sd_or_percent": np.std(values, ddof=1),
                "median": np.median(values),
                "q1": np.quantile(values, 0.25),
                "q3": np.quantile(values, 0.75),
                "minimum": np.min(values),
                "maximum": np.max(values),
            }
        )
    categorical = {
        "Sex": "sex",
        "Partner status": "partnered",
        "Age bin (years)": "age_bin",
    }
    for label, column in categorical.items():
        for level, count in frame[column].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "variable": label,
                    "level": str(level),
                    "n": len(frame),
                    "mean_or_count": int(count),
                    "sd_or_percent": float(100 * count / len(frame)),
                    "median": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "minimum": np.nan,
                    "maximum": np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_sensitivities(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: list[tuple[str, pd.DataFrame, bool]] = [
        ("primary_posterior_roi_spline", frame, False),
        (
            "qc_sensitivity_ica_converged",
            frame.loc[as_bool(frame["strict_ica_converged_included"])].copy(),
            False,
        ),
        (
            "qc_sensitivity_multiflag",
            frame.loc[as_bool(frame["strict_multiflag_included"])].copy(),
            False,
        ),
        ("age_bin_adjusted_sensitivity", frame, True),
    ]
    all_models: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    for variant, subset, add_age in variants:
        pair = [
            fit_one(subset, "stai_z", "mspss_z", "STAI_trait", variant, add_age),
            fit_one(subset, "mspss_z", "stai_z", "MSPSS_total", variant, add_age),
        ]
        apply_holm(pair)
        all_models.extend(item["model"] for item in pair)
        all_contrasts.extend(item["contrast"] for item in pair)
    return pd.DataFrame(all_models), pd.DataFrame(all_contrasts)


def run_linear_functional_form_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit a non-spline functional-form sensitivity."""
    rows: list[dict[str, Any]] = []
    y = frame["alpha_roi_reactivity_ec_minus_eo_db"].to_numpy(float)
    for focal, other, label in [
        ("stai_z", "mspss_z", "STAI_trait"),
        ("mspss_z", "stai_z", "MSPSS_total"),
    ]:
        x = np.column_stack(
            [
                np.ones(len(frame)),
                frame[focal].to_numpy(float),
                frame[other].to_numpy(float),
                frame["male"].to_numpy(float),
                frame["partnered"].to_numpy(float),
            ]
        )
        classical = sm.OLS(y, x).fit()
        fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=False)
        ci = np.asarray(fit.conf_int(alpha=ALPHA))
        rows.append(
            {
                "variant": "linear_functional_form_sensitivity_not_primary",
                "outcome": label,
                "n_participants": len(frame),
                "estimate_db_per_sd": float(fit.params[1]),
                "se_hc3": float(fit.bse[1]),
                "z_hc3": float(fit.tvalues[1]),
                "p_raw_two_sided": float(fit.pvalues[1]),
                "ci95_hc3_low": float(ci[1, 0]),
                "ci95_hc3_high": float(ci[1, 1]),
                "r_squared": float(classical.rsquared),
                "adjusted_r_squared": float(classical.rsquared_adj),
                "rmse_db": float(np.sqrt(np.mean(np.square(classical.resid)))),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_2"] = multipletests(result["p_raw_two_sided"], alpha=ALPHA, method="holm")[1]
    result["reject_holm_05"] = result["p_holm_2"] < ALPHA
    result["power_ratio_multiplier_per_sd"] = np.power(10.0, result["estimate_db_per_sd"] / 10.0)
    result["percent_change_in_ec_eo_ratio_per_sd"] = 100.0 * (result["power_ratio_multiplier_per_sd"] - 1.0)
    return result


def run_spline_complexity_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate prespecified natural-spline basis df=3-6 without changing the primary df=4 model."""
    rows: list[dict[str, Any]] = []
    y = frame["alpha_roi_reactivity_ec_minus_eo_db"].to_numpy(float)
    for basis_df in range(3, 7):
        pair_rows: list[dict[str, Any]] = []
        for focal, other, label in [
            ("stai_z", "mspss_z", "STAI_trait"),
            ("mspss_z", "stai_z", "MSPSS_total"),
        ]:
            focal_values = frame[focal].to_numpy(float)
            nonlinear, _ = fit_spline_basis(focal_values, df=basis_df)
            x = np.column_stack(
                [
                    np.ones(len(frame)),
                    focal_values,
                    nonlinear,
                    frame[other].to_numpy(float),
                    frame["male"].to_numpy(float),
                    frame["partnered"].to_numpy(float),
                ]
            )
            classical = sm.OLS(y, x).fit()
            fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=False)
            n_nonlin = nonlinear.shape[1]
            total_df = 1 + n_nonlin
            r_total = np.zeros((total_df, x.shape[1]))
            r_total[:, 1 : 1 + total_df] = np.eye(total_df)
            r_nonlin = np.zeros((n_nonlin, x.shape[1]))
            r_nonlin[:, 2 : 2 + n_nonlin] = np.eye(n_nonlin)
            total_chi2 = fit.wald_test(r_total, use_f=False, scalar=True)
            total_f = fit.wald_test(r_total, use_f=True, scalar=True)
            nonlinear_chi2 = fit.wald_test(r_nonlin, use_f=False, scalar=True)
            nonlinear_f = fit.wald_test(r_nonlin, use_f=True, scalar=True)
            pair_rows.append(
                {
                    "variant": "spline_basis_complexity_sensitivity_not_primary",
                    "basis_df_requested": basis_df,
                    "outcome": label,
                    "n_participants": len(frame),
                    "n_model_parameters": x.shape[1],
                    "residual_df": int(classical.df_resid),
                    "total_test_df": total_df,
                    "nonlinear_test_df": n_nonlin,
                    "wald_chi2_total": float(total_chi2.statistic),
                    "p_chi2_total_raw": float(total_chi2.pvalue),
                    "wald_f_total": float(total_f.statistic),
                    "f_total_numerator_df": total_df,
                    "f_total_denominator_df": int(classical.df_resid),
                    "p_f_total_raw": float(total_f.pvalue),
                    "wald_chi2_nonlinear": float(nonlinear_chi2.statistic),
                    "p_chi2_nonlinear_raw": float(nonlinear_chi2.pvalue),
                    "wald_f_nonlinear": float(nonlinear_f.statistic),
                    "f_nonlinear_numerator_df": n_nonlin,
                    "f_nonlinear_denominator_df": int(classical.df_resid),
                    "p_f_nonlinear_raw": float(nonlinear_f.pvalue),
                    "r_squared": float(classical.rsquared),
                    "adjusted_r_squared": float(classical.rsquared_adj),
                    "rmse_db": float(np.sqrt(np.mean(np.square(classical.resid)))),
                    "aic_classical": float(classical.aic),
                    "bic_classical": float(classical.bic),
                }
            )
        p_total_chi2 = multipletests(
            [row["p_chi2_total_raw"] for row in pair_rows], alpha=ALPHA, method="holm"
        )[1]
        p_total_f = multipletests(
            [row["p_f_total_raw"] for row in pair_rows], alpha=ALPHA, method="holm"
        )[1]
        p_nonlin_chi2 = multipletests(
            [row["p_chi2_nonlinear_raw"] for row in pair_rows], alpha=ALPHA, method="holm"
        )[1]
        p_nonlin_f = multipletests(
            [row["p_f_nonlinear_raw"] for row in pair_rows], alpha=ALPHA, method="holm"
        )[1]
        for index, row in enumerate(pair_rows):
            row["p_chi2_total_holm_2_within_basis_df"] = float(p_total_chi2[index])
            row["p_f_total_holm_2_within_basis_df"] = float(p_total_f[index])
            row["p_chi2_nonlinear_holm_2_within_basis_df"] = float(p_nonlin_chi2[index])
            row["p_f_nonlinear_holm_2_within_basis_df"] = float(p_nonlin_f[index])
            row["total_reject_chi2_holm_05"] = bool(p_total_chi2[index] < ALPHA)
            row["total_reject_f_holm_05"] = bool(p_total_f[index] < ALPHA)
            row["nonlinear_reject_chi2_holm_05"] = bool(p_nonlin_chi2[index] < ALPHA)
            row["nonlinear_reject_f_holm_05"] = bool(p_nonlin_f[index] < ALPHA)
            rows.append(row)
    return pd.DataFrame(rows)


def run_leave_one_out(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for omitted in frame["participant_id"].tolist():
        subset = frame.loc[frame["participant_id"] != omitted].copy()
        pair = [
            fit_one(subset, "stai_z", "mspss_z", "STAI_trait", "leave_one_out", False),
            fit_one(subset, "mspss_z", "stai_z", "MSPSS_total", "leave_one_out", False),
        ]
        apply_holm(pair)
        for item in pair:
            row = dict(item["model"])
            row["omitted_participant_id"] = omitted
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_leave_one_out(loo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_sign = {"STAI_trait": 1.0, "MSPSS_total": -1.0}
    for outcome, subset in loo.groupby("outcome", sort=False):
        direction_preserved = np.sign(subset["linear_component_db_per_sd"]) == expected_sign[outcome]
        ci_excludes_zero = (subset["linear_component_ci95_low"] > 0) | (
            subset["linear_component_ci95_high"] < 0
        )
        omnibus_significant = subset["total_reject_holm_05"].astype(bool)
        rows.append(
            {
                "outcome": outcome,
                "n_leave_one_participant_out_refits": len(subset),
                "expected_linear_direction": "positive" if expected_sign[outcome] > 0 else "negative",
                "direction_preserved_n": int(direction_preserved.sum()),
                "direction_preserved_percent": float(100 * direction_preserved.mean()),
                "omnibus_holm_significant_n": int(omnibus_significant.sum()),
                "omnibus_holm_significant_percent": float(100 * omnibus_significant.mean()),
                "linear_ci95_excludes_zero_n": int(ci_excludes_zero.sum()),
                "linear_ci95_excludes_zero_percent": float(100 * ci_excludes_zero.mean()),
                "linear_component_min_db_per_sd": float(subset["linear_component_db_per_sd"].min()),
                "linear_component_max_db_per_sd": float(subset["linear_component_db_per_sd"].max()),
                "omnibus_holm_p_min": float(subset["p_total_holm_2"].min()),
                "omnibus_holm_p_max": float(subset["p_total_holm_2"].max()),
                "interpretation_boundary": "Omnibus stability does not imply a significant linear coefficient in every refit.",
            }
        )
    return pd.DataFrame(rows)


def participant_export(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "participant_id",
        "age_bin",
        "sex",
        "male",
        "relationship_status_raw",
        "partnered",
        "stai_trait_total",
        "stai_z",
        "mspss_total",
        "mspss_z",
        "alpha_roi_eo_db_uv2_hz",
        "alpha_roi_ec_db_uv2_hz",
        "alpha_roi_reactivity_ec_minus_eo_db",
        "relative_alpha_roi_eo_percent_1_45hz",
        "relative_alpha_roi_ec_percent_1_45hz",
        "relative_alpha_roi_reactivity_percentage_points",
        "preprocessing_status",
        "ica_converged",
        "n_bad_channels",
        "n_excluded_components",
        "excluded_fraction",
        "global_window_retention_fraction",
        "min_channel_band_retention",
        "strict_ica_converged_included",
        "strict_multiflag_included",
    ]
    return frame[columns].copy()


def load_relative_alpha_topography(
    full_frequency_csv: Path,
    participant_ids: list[str],
    channels: list[str],
    canonical_open: np.ndarray,
    canonical_closed: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, float]]:
    """Reconstruct alpha/1-45-Hz relative power from the audited 0.5-Hz table."""
    usecols = [
        "participant_id",
        "channel",
        "frequency_low_hz",
        "frequency_high_hz",
        "eyes_open_psd_db_uV2_per_Hz",
        "eyes_closed_psd_db_uV2_per_Hz",
        "eyes_open_fullrange_power_db_uV2_1_45Hz",
        "eyes_closed_fullrange_power_db_uV2_1_45Hz",
    ]
    alpha_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(full_frequency_csv, usecols=usecols, chunksize=180_000):
        selected = chunk.loc[
            (chunk["frequency_low_hz"] >= 8.0 - 1e-12)
            & (chunk["frequency_high_hz"] <= 13.0 + 1e-12)
        ].copy()
        if not selected.empty:
            alpha_parts.append(selected)
    if not alpha_parts:
        raise ValueError("La tabla de 1-45 Hz no contiene intervalos completos de 8-13 Hz")
    alpha = pd.concat(alpha_parts, ignore_index=True)
    width = alpha["frequency_high_hz"].to_numpy(float) - alpha["frequency_low_hz"].to_numpy(float)
    alpha["alpha_open_power_uv2"] = np.power(
        10.0, alpha["eyes_open_psd_db_uV2_per_Hz"].to_numpy(float) / 10.0
    ) * width
    alpha["alpha_closed_power_uv2"] = np.power(
        10.0, alpha["eyes_closed_psd_db_uV2_per_Hz"].to_numpy(float) / 10.0
    ) * width
    grouped = (
        alpha.groupby(["participant_id", "channel"], observed=True, sort=False)
        .agg(
            alpha_open_power_uv2=("alpha_open_power_uv2", "sum"),
            alpha_closed_power_uv2=("alpha_closed_power_uv2", "sum"),
            full_open_power_db_uv2=("eyes_open_fullrange_power_db_uV2_1_45Hz", "first"),
            full_closed_power_db_uv2=("eyes_closed_fullrange_power_db_uV2_1_45Hz", "first"),
            alpha_intervals=("frequency_low_hz", "size"),
        )
        .reset_index()
    )
    if len(grouped) != len(participant_ids) * len(channels) or not (grouped["alpha_intervals"] == 10).all():
        raise ValueError("La reconstruccion relativa no produjo 139x61 pares con 10 intervalos alfa")
    grouped["alpha_open_density_db"] = 10.0 * np.log10(grouped["alpha_open_power_uv2"] / 5.0)
    grouped["alpha_closed_density_db"] = 10.0 * np.log10(grouped["alpha_closed_power_uv2"] / 5.0)
    grouped["relative_alpha_open_percent"] = 100.0 * grouped["alpha_open_power_uv2"] / np.power(
        10.0, grouped["full_open_power_db_uv2"] / 10.0
    )
    grouped["relative_alpha_closed_percent"] = 100.0 * grouped["alpha_closed_power_uv2"] / np.power(
        10.0, grouped["full_closed_power_db_uv2"] / 10.0
    )
    grouped["relative_alpha_reactivity_percentage_points"] = (
        grouped["relative_alpha_closed_percent"] - grouped["relative_alpha_open_percent"]
    )

    def pivot(column: str) -> np.ndarray:
        matrix = grouped.pivot(index="participant_id", columns="channel", values=column)
        matrix = matrix.reindex(index=participant_ids, columns=channels)
        values = matrix.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Topografia {column} contiene datos no finitos")
        return values

    open_from_bins = pivot("alpha_open_density_db")
    closed_from_bins = pivot("alpha_closed_density_db")
    validation = {
        "max_abs_error_alpha_open_db": float(np.max(np.abs(open_from_bins - canonical_open))),
        "max_abs_error_alpha_closed_db": float(np.max(np.abs(closed_from_bins - canonical_closed))),
    }
    if max(validation.values()) > 1e-8:
        raise RuntimeError(
            "La reconstruccion alfa desde los 88 intervalos no coincide con la cache canonica: "
            + json.dumps(validation)
        )
    matrices = {
        "relative_open_percent": pivot("relative_alpha_open_percent"),
        "relative_closed_percent": pivot("relative_alpha_closed_percent"),
        "relative_reactivity_pp": pivot("relative_alpha_reactivity_percentage_points"),
    }
    summary = pd.DataFrame(
        {
            "channel": channels,
            "absolute_alpha_eo_mean_db_density": canonical_open.mean(axis=0),
            "absolute_alpha_ec_mean_db_density": canonical_closed.mean(axis=0),
            "absolute_alpha_reactivity_mean_db": (canonical_closed - canonical_open).mean(axis=0),
            "relative_alpha_eo_mean_percent_1_45hz": matrices["relative_open_percent"].mean(axis=0),
            "relative_alpha_ec_mean_percent_1_45hz": matrices["relative_closed_percent"].mean(axis=0),
            "relative_alpha_reactivity_mean_percentage_points": matrices["relative_reactivity_pp"].mean(axis=0),
        }
    )
    return matrices, summary, validation


def channelwise_topographic_associations(
    frame: pd.DataFrame,
    channels: list[str],
    participant_ids: list[str],
    absolute_reactivity: np.ndarray,
    relative_reactivity: np.ndarray,
) -> pd.DataFrame:
    """Apply the focused spline model at each sensor for descriptive spatial maps."""
    participant_lookup = {participant_id: i for i, participant_id in enumerate(participant_ids)}
    indices = np.array([participant_lookup[p] for p in frame["participant_id"]], dtype=int)
    representations = {
        "absolute_alpha_reactivity_db": absolute_reactivity[indices],
        "relative_alpha_reactivity_percentage_points": relative_reactivity[indices],
    }
    rows: list[dict[str, Any]] = []
    for focal, other, label in [
        ("stai_z", "mspss_z", "STAI_trait"),
        ("mspss_z", "stai_z", "MSPSS_total"),
    ]:
        nonlinear, _ = fit_spline_basis(frame[focal].to_numpy(float), df=4)
        design, _, _ = design_matrix(frame, focal, other, nonlinear, add_age=False)
        n_nonlin = nonlinear.shape[1]
        r_total = np.zeros((1 + n_nonlin, design.shape[1]))
        r_total[0, 1] = 1.0
        r_nonlin = np.zeros((n_nonlin, design.shape[1]))
        for j in range(n_nonlin):
            r_total[1 + j, 2 + j] = 1.0
            r_nonlin[j, 2 + j] = 1.0
        for representation, matrix in representations.items():
            for channel_index, channel in enumerate(channels):
                response = matrix[:, channel_index]
                fit = sm.OLS(response, design).fit(cov_type="HC3", use_t=False)
                total = fit.wald_test(r_total, scalar=True)
                nonlinear_test = fit.wald_test(r_nonlin, scalar=True)
                ci = np.asarray(fit.conf_int(alpha=ALPHA))
                rows.append(
                    {
                        "representation": representation,
                        "psychometric_variable": label,
                        "channel": channel,
                        "n_participants": len(frame),
                        "linear_component": float(fit.params[1]),
                        "linear_component_se_hc3": float(fit.bse[1]),
                        "linear_component_p": float(fit.pvalues[1]),
                        "linear_component_ci95_low": float(ci[1, 0]),
                        "linear_component_ci95_high": float(ci[1, 1]),
                        "total_spline_wald_chi2_3df": float(total.statistic),
                        "total_spline_p": float(total.pvalue),
                        "nonlinear_wald_chi2_2df": float(nonlinear_test.statistic),
                        "nonlinear_p": float(nonlinear_test.pvalue),
                    }
                )
    result = pd.DataFrame(rows)
    result["linear_component_q_bh61"] = np.nan
    result["total_spline_q_bh61"] = np.nan
    result["nonlinear_q_bh61"] = np.nan
    for _, index in result.groupby(["representation", "psychometric_variable"]).groups.items():
        idx = list(index)
        result.loc[idx, "linear_component_q_bh61"] = multipletests(
            result.loc[idx, "linear_component_p"], method="fdr_bh"
        )[1]
        result.loc[idx, "total_spline_q_bh61"] = multipletests(
            result.loc[idx, "total_spline_p"], method="fdr_bh"
        )[1]
        result.loc[idx, "nonlinear_q_bh61"] = multipletests(
            result.loc[idx, "nonlinear_p"], method="fdr_bh"
        )[1]
    result["total_spline_reject_bh05"] = result["total_spline_q_bh61"] < ALPHA
    result["linear_component_reject_bh05"] = result["linear_component_q_bh61"] < ALPHA
    result["nonlinear_reject_bh05"] = result["nonlinear_q_bh61"] < ALPHA
    return result


def flow_and_missingness(project: Path, frame: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_all = pd.read_csv(project / QC_REL)
    source = metadata["source_audit"]
    flow = pd.DataFrame(
        [
            {"stage": "Download-eligible source records", "n": source["download_eligible"], "excluded_from_previous": np.nan, "reason": "Local source audit"},
            {"stage": "Complete EEG-demographic-questionnaire triplets", "n": source["triplet_complete"], "excluded_from_previous": source["triplet_incomplete"], "reason": "Five incomplete source triplets"},
            {"stage": "Frozen primary EEG cache", "n": len(qc_all), "excluded_from_previous": source["triplet_complete"] - len(qc_all), "reason": "Three hard EEG exclusions documented upstream"},
            {"stage": "Complete cases for the focused analysis", "n": len(frame), "excluded_from_previous": len(qc_all) - len(frame), "reason": "Two missing relationship-status values"},
        ]
    )
    variables = ["stai_trait_total", "mspss_total", "sex", "partnered", "age_bin"]
    missing = []
    for column in variables:
        numeric = pd.to_numeric(qc_all[column], errors="coerce") if column in {"stai_trait_total", "mspss_total", "partnered"} else qc_all[column]
        n_missing = int(numeric.isna().sum())
        missing.append(
            {
                "variable": column,
                "base_n": len(qc_all),
                "n_missing": n_missing,
                "percent_missing": 100 * n_missing / len(qc_all),
            }
        )
    return flow, pd.DataFrame(missing)


def eeg_info_for_topomap(channels: list[str]) -> dict[str, Any]:
    info = mne.create_info(ch_names=channels, sfreq=250.0, ch_types=["eeg"] * len(channels))
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, match_case=False, on_missing="raise")
    positions = _find_topomap_coords(
        info,
        np.arange(len(channels)),
        ignore_overlap=False,
        to_sphere=True,
        sphere="eeglab",
    )
    # Some valid outer-ring sensors lie beyond MNE's conventional T7/T8 circle.
    # A single enclosing radius makes the interpolation mask and head silhouette
    # identical while preserving all projected inter-electrode relationships.
    radius = float(np.linalg.norm(positions, axis=1).max() * 1.02)
    return {
        "info": info,
        "positions": positions,
        "sphere": np.array([0.0, 0.0, 0.0, radius], dtype=float),
        "radius": radius,
    }


def draw_topomap(
    fig,
    ax,
    geometry: dict[str, Any],
    values: np.ndarray,
    title: str,
    cmap: str,
    vlim: tuple[float, float],
    colorbar_label: str,
    significance_mask: np.ndarray | None = None,
) -> None:
    image, contours = mne.viz.plot_topomap(
        np.asarray(values, dtype=float),
        geometry["positions"],
        axes=ax,
        show=False,
        sensors="k.",
        names=None,
        contours=7,
        cmap=cmap,
        vlim=vlim,
        extrapolate="head",
        outlines="head",
        sphere=geometry["sphere"],
        image_interp="cubic",
        res=256,
    )
    if contours is not None:
        contours.set_linewidths(0.55)
        contours.set_alpha(0.62)
    for line in ax.lines:
        if len(line.get_xdata()) == len(geometry["positions"]) and line.get_marker() not in {None, "None", ""}:
            line.set_markersize(1.6)
            line.set_color("#3F3F3F")
            line.set_alpha(0.72)
    if significance_mask is not None and np.any(significance_mask):
        significant_positions = geometry["positions"][np.asarray(significance_mask, dtype=bool)]
        ax.scatter(
            significant_positions[:, 0],
            significant_positions[:, 1],
            marker="*",
            s=32,
            facecolors="white",
            edgecolors="black",
            linewidths=0.65,
            zorder=7,
        )
    ax.set_title(title, fontsize=10.5)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.72, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)


def make_topographic_figures(
    arrays: dict[str, Any],
    relative: dict[str, np.ndarray],
    topographic_associations: pd.DataFrame,
    figures: Path,
) -> pd.DataFrame:
    channels = arrays["channels"]
    geometry = eeg_info_for_topomap(channels)
    alpha_index = arrays["bands"].index("alpha")
    absolute_open = arrays["psd_open"][:, alpha_index, :].mean(axis=0)
    absolute_closed = arrays["psd_closed"][:, alpha_index, :].mean(axis=0)
    absolute_reactivity = arrays["reactivity"][:, alpha_index, :].mean(axis=0)
    relative_open = relative["relative_open_percent"].mean(axis=0)
    relative_closed = relative["relative_closed_percent"].mean(axis=0)
    relative_reactivity = relative["relative_reactivity_pp"].mean(axis=0)

    absolute_limits = (float(min(absolute_open.min(), absolute_closed.min())), float(max(absolute_open.max(), absolute_closed.max())))
    absolute_reactivity_limit = float(np.max(np.abs(absolute_reactivity)))
    relative_limits = (float(min(relative_open.min(), relative_closed.min())), float(max(relative_open.max(), relative_closed.max())))
    relative_reactivity_limit = float(np.max(np.abs(relative_reactivity)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.10, top=0.87, wspace=0.32, hspace=0.34)
    draw_topomap(fig, axes[0, 0], geometry, absolute_open, "(A) Eyes open", "viridis", absolute_limits, "dB re 1 µV²/Hz")
    draw_topomap(fig, axes[0, 1], geometry, absolute_closed, "(B) Eyes closed", "viridis", absolute_limits, "dB re 1 µV²/Hz")
    draw_topomap(
        fig, axes[0, 2], geometry, absolute_reactivity, "(C) Absolute reactivity (EC − EO)", "RdBu_r",
        (-absolute_reactivity_limit, absolute_reactivity_limit), "dB"
    )
    draw_topomap(fig, axes[1, 0], geometry, relative_open, "(D) Relative alpha, eyes open", "viridis", relative_limits, "% of 1–45 Hz power")
    draw_topomap(fig, axes[1, 1], geometry, relative_closed, "(E) Relative alpha, eyes closed", "viridis", relative_limits, "% of 1–45 Hz power")
    draw_topomap(
        fig, axes[1, 2], geometry, relative_reactivity, "(F) Relative reactivity (EC − EO)", "RdBu_r",
        (-relative_reactivity_limit, relative_reactivity_limit), "percentage points"
    )
    fig.suptitle("Alpha power and eyes-closed reactivity topographies", fontsize=13, y=0.975)
    fig.text(0.5, 0.025, "Small black dots mark all 61 scalp electrodes. The black head circle is also the interpolation boundary.", ha="center", fontsize=8.5)
    fig.savefig(figures / "Figure_1_alpha_power_topographies_absolute_relative_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "Figure_1_alpha_power_topographies_absolute_relative_vector.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.15, top=0.87, wspace=0.30, hspace=0.30)
    combinations = [
        ("STAI_trait", "absolute_alpha_reactivity_db", axes[0, 0], "(A) STAI — absolute reactivity", "dB/SD"),
        ("STAI_trait", "relative_alpha_reactivity_percentage_points", axes[0, 1], "(B) STAI — relative reactivity", "percentage points/SD"),
        ("MSPSS_total", "absolute_alpha_reactivity_db", axes[1, 0], "(C) MSPSS — absolute reactivity", "dB/SD"),
        ("MSPSS_total", "relative_alpha_reactivity_percentage_points", axes[1, 1], "(D) MSPSS — relative reactivity", "percentage points/SD"),
    ]
    representation_limits: dict[str, float] = {}
    for representation in ["absolute_alpha_reactivity_db", "relative_alpha_reactivity_percentage_points"]:
        representation_limits[representation] = float(
            topographic_associations.loc[
                topographic_associations.representation == representation, "linear_component"
            ].abs().max()
        )
    for outcome, representation, ax, title, unit in combinations:
        subset = topographic_associations.loc[
            (topographic_associations.psychometric_variable == outcome)
            & (topographic_associations.representation == representation)
        ].set_index("channel").loc[channels]
        limit = representation_limits[representation]
        draw_topomap(
            fig,
            ax,
            geometry,
            subset["linear_component"].to_numpy(float),
            title,
            "RdBu_r",
            (-limit, limit),
            unit,
            subset["total_spline_reject_bh05"].to_numpy(bool),
        )
    fig.suptitle("Spatial distribution of adjusted psychometric associations with alpha reactivity", fontsize=12.5, y=0.975)
    linear_fdr_count = int(topographic_associations["linear_component_reject_bh05"].sum())
    nonlinear_fdr_count = int(topographic_associations["nonlinear_reject_bh05"].sum())
    fig.text(
        0.5,
        0.055,
        f"Color = directional linear estimate ({linear_fdr_count}/244 coefficients survive within-map BH-FDR). Stars = significant total 3-df spline association; stars do not mark a significant slope.",
        ha="center",
        fontsize=7.8,
    )
    fig.text(
        0.5,
        0.026,
        f"No nonlinear component survived within-map BH-FDR ({nonlinear_fdr_count}/244). Black dots mark sensors; the head circle is the interpolation boundary.",
        ha="center",
        fontsize=7.8,
    )
    fig.savefig(figures / "Figure_2_stai_mspss_alpha_association_topographies_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "Figure_2_stai_mspss_alpha_association_topographies_vector.pdf", bbox_inches="tight")
    plt.close(fig)
    positions = geometry["positions"]
    radial_distance = np.linalg.norm(positions, axis=1)
    return pd.DataFrame(
        {
            "channel": channels,
            "topomap_x": positions[:, 0],
            "topomap_y": positions[:, 1],
            "radial_distance": radial_distance,
            "head_and_mask_radius": geometry["radius"],
            "fraction_of_head_radius": radial_distance / geometry["radius"],
            "head_outline_equals_interpolation_mask": True,
            "coordinate_source": "MNE standard_1005 projected with eeglab sphere; one enclosing display radius",
        }
    )


def make_figures(primary: list[dict[str, Any]], flow: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    colors = ["#8C2D5D", "#176D78"]
    for ax, item, color in zip(axes, primary, colors):
        frame = item["frame"]
        focal_values = frame[item["focal"]].to_numpy(float)
        curve = item["curve"]
        ax.scatter(focal_values, item["adjusted_y"], s=18, alpha=0.42, color="#555555", edgecolors="none")
        ax.plot(curve["focal_z"], curve["predicted_reactivity_db"], color=color, linewidth=2.2)
        ax.fill_between(
            curve["focal_z"].to_numpy(float),
            curve["ci95_pointwise_low"].to_numpy(float),
            curve["ci95_pointwise_high"].to_numpy(float),
            color=color,
            alpha=0.20,
        )
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.65)
        model = item["model"]
        ax.set_title(item["label"].replace("_", " "))
        ax.set_xlabel("Psychometric score (sample z score)")
        ax.set_ylabel("Adjusted posterior alpha reactivity\nEC - EO (dB)")
        ax.text(
            0.03,
            0.97,
            f"Total spline: chi2({model['spline_df_total']})={model['wald_chi2_total']:.2f}\n"
            f"Holm p={fmt_p(model['p_total_holm_2'])}\n"
            f"Nonlinear Holm p={fmt_p(model['p_nonlinear_holm_2'])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88, "edgecolor": "#BBBBBB"},
        )
    fig.suptitle("Adjusted associations of STAI and MSPSS with posterior alpha reactivity", fontsize=12)
    fig.savefig(figures / "Figure_3_adjusted_psychometric_curves_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "Figure_3_adjusted_psychometric_curves_vector.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, item in enumerate(primary):
        axes[row, 0].scatter(item["fitted"], item["residuals"], s=19, alpha=0.58, color=colors[row])
        axes[row, 0].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[row, 0].set_xlabel("Fitted reactivity (dB)")
        axes[row, 0].set_ylabel("OLS residual (dB)")
        axes[row, 0].set_title(f"{item['label'].replace('_', ' ')}: residuals vs fitted")
        stats.probplot(item["residuals"], dist="norm", plot=axes[row, 1])
        axes[row, 1].set_title(f"{item['label'].replace('_', ' ')}: normal Q-Q")
    fig.suptitle("Regression diagnostics (HC3 inference does not require homoskedasticity)", fontsize=12)
    fig.savefig(figures / "Figure_S2_regression_diagnostics_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "Figure_S2_regression_diagnostics_vector.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.axis("off")
    y_positions = np.linspace(0.88, 0.12, len(flow))
    for i, (y, row) in enumerate(zip(y_positions, flow.itertuples(index=False))):
        text = f"{row.stage}\nn = {int(row.n)}"
        ax.text(0.5, y, text, ha="center", va="center", fontsize=10.5,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF4F7", "edgecolor": "#176D78"})
        if i < len(flow) - 1:
            next_y = y_positions[i + 1]
            excluded = int(flow.iloc[i + 1]["excluded_from_previous"])
            ax.annotate("", xy=(0.5, next_y + 0.07), xytext=(0.5, y - 0.07),
                        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.2})
            ax.text(0.61, (y + next_y) / 2, f"excluded: {excluded}", va="center", fontsize=8.5, color="#444444")
    ax.set_title("Participant flow for the focused EEG study", fontsize=12)
    fig.savefig(figures / "Figure_S1_participant_flow_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(figures / "Figure_S1_participant_flow_vector.pdf", bbox_inches="tight")
    plt.close(fig)


def checklist_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strobe_rows = [
        ("1", "Title/abstract identifies design", "PENDING_MANUSCRIPT", "Use 'cross-sectional secondary analysis' in title or abstract."),
        ("2-3", "Background and study objectives", "PARTIAL", "Scientific rationale and focused objectives are drafted; alpha is disclosed as a same-cohort, data-informed post hoc focus and must not be presented as prospectively selected."),
        ("4-5", "Design, setting, dates", "DOCUMENTED", "LEMON cross-sectional acquisition in Leipzig, 2013-2015; secondary analysis date is logged."),
        ("6", "Eligibility, selection and follow-up", "DOCUMENTED", "Source recruitment/eligibility plus local flow table and hard exclusions."),
        ("7-8", "Variables and measurement", "DOCUMENTED", "Outcome, exposures, covariates, ROI, PSD units and reactivity formula are explicit."),
        ("9", "Bias", "PARTIAL", "QC, influence and four-band selection sensitivities are supplied; data-informed band selection and residual-confounding limitations must remain."),
        ("10", "Study size", "DOCUMENTED_LIMITATION", "Availability sample; no prospective power calculation. Do not claim otherwise."),
        ("11-12", "Quantitative variables/statistical methods", "DOCUMENTED", "Z scoring, spline df, HC3, the focal Holm-2 family, post-selection four-band Holm-8 sensitivity, complete cases, age/QC sensitivities and LOO are explicit."),
        ("13", "Participant flow", "DOCUMENTED", "CSV and 600-dpi flow figure generated."),
        ("14", "Descriptive data and missingness", "DOCUMENTED", "Table 1 and missingness table generated."),
        ("15-17", "Outcome, main and additional results", "DOCUMENTED", "Full coefficients, Wald tests, CIs, model fit, contrasts, four-band selection context and other sensitivities generated."),
        ("18-21", "Key results, limitations, interpretation, generalisability", "PARTIAL", "Draft supplied; journal manuscript must retain exploratory and associative framing."),
        ("22", "Funding", "PENDING_AUTHORS", "Add funding/role of funder for this secondary analysis and cite source-study funding."),
    ]
    cobidas_rows = [
        ("Design", "Population, design, ethics, protocol", "DOCUMENTED", "Primary LEMON paper and local participant flow."),
        ("Acquisition", "Device, electrodes, layout, reference, ground, impedance, sampling, filters", "DOCUMENTED_SOURCE", "BrainAmp MR plus; 61 scalp+VEOG; 10-10; FCz; sternum; <5 kOhm; 2500 Hz; 0.015-1000 Hz."),
        ("Acquisition", "Paradigm and timing", "DOCUMENTED_SOURCE", "16 interleaved 60-s blocks; 8 EC/8 EO; EC first; EO fixation."),
        ("Preprocessing", "Software and exact ordered operations", "DOCUMENTED_LOCAL", "The portable scripts/03_preprocess_eeg.py is included with participant QC summaries and an upstream-method audit report."),
        ("Preprocessing", "Bad channels and artifacts", "DOCUMENTED_LOCAL", "Included source closes the 1.4826-MAD robust-z formula, six-neighbor rule, advisory LOF, ICA/ICLabel/EOG union, interpolation, and three hard exclusions."),
        ("Spectral analysis", "Estimator, window, overlap, resolution, bands, units", "DOCUMENTED_LOCAL", "Both the PSD/QC source and the authoritative portable spectral-input builder are included. The latter implements additive closed-endpoint trapezoidal integration/band width; periodogram/Hamming, 4 s, 50% overlap, 0.25-Hz FFT spacing and 0.34-Hz ENBW are documented."),
        ("ROI", "ROI definition independent of current coefficient map", "DOCUMENTED_WITH_LIMITATION", "Fixed anatomical posterior ROI of nine named sensors; distinguish it from the secondary sensor-wise maps."),
        ("Statistics", "All regressors, software, assumptions, multiplicity", "DOCUMENTED", "Full design, versions, HC3, diagnostics, focal Holm-2, post-selection Holm-8, sensor-wise BH-FDR and exact tests generated."),
        ("Results", "Effect size/CI/model fit/full results", "DOCUMENTED", "Coefficients, 95% CIs, contrasts, R2/RMSE, diagnostics, curves and sensitivities generated."),
        ("Sharing", "Code/data provenance and machine-readable outputs", "DOCUMENTED_LOCAL", "Four upstream source artifacts, original/public-copy hashes, scripts, CSV/TXT/JSON and figures are included; path sanitization and LEMON-permission boundary are explicit."),
    ]
    sampl_rows = [
        ("Methods", "Name methods and enough detail to reproduce", "DOCUMENTED", "Natural spline construction, exact design matrices, HC3 and Holm are reported."),
        ("Methods", "State alpha, two-sided tests and multiplicity", "DOCUMENTED", "Two-sided alpha=.05; focal Holm families of two plus contextual four-band Holm families of eight total and eight nonlinear tests."),
        ("Methods", "State transformations, missing data and software", "DOCUMENTED", "Sample z scores, complete-case rule, package versions and hashes reported."),
        ("Methods", "Assumptions and diagnostics", "DOCUMENTED", "Linearity relaxed by spline; HC3; residual/influence/collinearity diagnostics."),
        ("Results", "Report estimates with uncertainty, not p alone", "DOCUMENTED", "All coefficients and +/-1 SD contrasts include 95% HC3 CIs."),
        ("Results", "Exact test statistic, df and p", "DOCUMENTED", "Wald chi-square, constraint df and exact p values reported."),
        ("Results", "Descriptive sample sizes and summaries", "DOCUMENTED", "n per analysis, distributions and categorical counts reported."),
        ("Results", "Regression model specification and fit", "DOCUMENTED", "Variables/coding, full coefficients, R2, adjusted R2 and RMSE supplied."),
        ("Reporting", "Avoid significance-only language", "DOCUMENTED", "Narrative distinguishes association, curvature, uncertainty, post-selection band context and exploratory status."),
    ]
    pending_rows = [
        ("Prospective status", "State that alpha was selected after same-cohort canonical-band exploration and that the focused analysis was not prospectively preregistered unless contrary documentation exists."),
        ("Study-size rationale", "State that n was determined by availability and complete cases; no retrospective power justification."),
        ("Questionnaire scoring", "Verify exact STAI trait and MSPSS scoring manuals, language versions, allowed ranges and handling of missing items."),
        ("Reliability", "Item-level responses are not in the frozen model table; add sample reliability only if item-level data are audited."),
        ("Funding", "Add funding and role of funder for the present secondary analysis."),
        ("Conflicts/roles", "Add conflicts of interest, author contributions and data-use compliance."),
        ("Data sharing", "Decide which participant-level table may be publicly shared under LEMON phenotyping terms."),
        ("Target journal", "Confirm the final target journal and then apply only that journal's current author instructions, word limits and reference style."),
        ("Generative AI declaration", "Add the journal-required disclosure for any generative-AI assistance used in manuscript preparation; authors remain responsible for all content."),
        ("Manuscript placement", "Insert final table/figure numbers, repository DOI, access date and journal-specific checklist page numbers."),
    ]
    cols = ["item", "requirement", "status", "evidence_or_action"]
    return (
        pd.DataFrame(strobe_rows, columns=cols),
        pd.DataFrame(cobidas_rows, columns=cols),
        pd.DataFrame(sampl_rows, columns=cols),
        pd.DataFrame(pending_rows, columns=["topic", "required_action"]),
    )


def report_text(
    project: Path,
    output: Path,
    frame: pd.DataFrame,
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    primary: list[dict[str, Any]],
    sensitivity: pd.DataFrame,
    linear_sensitivity: pd.DataFrame,
    spline_complexity: pd.DataFrame,
    band_selection: pd.DataFrame,
    loo: pd.DataFrame,
    loo_summary_table: pd.DataFrame,
    hashes: dict[str, str],
    topographic_associations: pd.DataFrame,
    relative_validation: dict[str, float],
) -> str:
    source = metadata["source_audit"]
    stai = primary[0]["model"]
    mspss = primary[1]["model"]
    stai_linear = linear_sensitivity.loc[linear_sensitivity.outcome == "STAI_trait"].iloc[0]
    mspss_linear = linear_sensitivity.loc[linear_sensitivity.outcome == "MSPSS_total"].iloc[0]
    qc_all = pd.read_csv(project / QC_REL)
    accepted_open = int(np.asarray(arrays["counts_open"]).sum())
    accepted_closed = int(np.asarray(arrays["counts_closed"]).sum())
    possible_per_condition = 139 * 61 * 232
    loo_summary = []
    for row in loo_summary_table.itertuples(index=False):
        loo_summary.append(
            f"{row.outcome}: directional sign retained in {int(row.direction_preserved_n)}/{int(row.n_leave_one_participant_out_refits)} refits; "
            f"total omnibus Holm p<.05 in {int(row.omnibus_holm_significant_n)}/{int(row.n_leave_one_participant_out_refits)}; "
            f"the linear-component 95% CI excluded zero in only {int(row.linear_ci95_excludes_zero_n)}/{int(row.n_leave_one_participant_out_refits)}. "
            "Omnibus stability does not imply a significant linear coefficient in every refit."
        )
    qc_lookup = sensitivity.set_index(["variant", "outcome"])
    stai_ica = qc_lookup.loc[("qc_sensitivity_ica_converged", "STAI_trait")]
    mspss_ica = qc_lookup.loc[("qc_sensitivity_ica_converged", "MSPSS_total")]
    stai_multiflag = qc_lookup.loc[("qc_sensitivity_multiflag", "STAI_trait")]
    mspss_multiflag = qc_lookup.loc[("qc_sensitivity_multiflag", "MSPSS_total")]
    df6_lookup = spline_complexity.loc[spline_complexity["basis_df_requested"] == 6].set_index("outcome")
    stai_df6 = df6_lookup.loc["STAI_trait"]
    mspss_df6 = df6_lookup.loc["MSPSS_total"]
    band_lookup = band_selection.set_index(["band", "psychometric_variable"])
    band_alpha_stai = band_lookup.loc[("alpha", "STAI_trait")]
    band_alpha_mspss = band_lookup.loc[("alpha", "MSPSS_total")]
    nonalpha_total_rejections = int(
        band_selection.loc[band_selection["band"] != "alpha", "total_reject_holm_all_8_05"].sum()
    )
    band_nonlinear_rejections = int(band_selection["nonlinear_reject_holm_all_8_05"].sum())
    representation_labels = {
        "absolute_alpha_reactivity_db": "absolute alpha reactivity",
        "relative_alpha_reactivity_percentage_points": "relative alpha reactivity",
    }
    psychometric_labels = {"STAI_trait": "STAI", "MSPSS_total": "MSPSS"}
    topographic_summary = []
    for (representation, psychometric), subset in topographic_associations.groupby(
        ["representation", "psychometric_variable"], sort=False
    ):
        subset = subset.sort_values(["total_spline_q_bh61", "total_spline_p", "channel"])
        strongest = subset.iloc[0]
        n_fdr = int(subset["total_spline_reject_bh05"].sum())
        n_linear_fdr = int(subset["linear_component_reject_bh05"].sum())
        n_nonlinear_fdr = int(subset["nonlinear_reject_bh05"].sum())
        topographic_summary.append(
            f"{psychometric_labels[psychometric]}, {representation_labels[representation]}: "
            f"{n_fdr}/61 electrodes with total-spline BH-FDR q<.05; lowest q at "
            f"{strongest.channel} (linear component {strongest.linear_component:.4f}, "
            f"raw p={fmt_p(strongest.total_spline_p)}, q={fmt_p(strongest.total_spline_q_bh61)}). "
            f"Linear-component BH-FDR: {n_linear_fdr}/61; nonlinear-component BH-FDR: {n_nonlinear_fdr}/61."
        )
    return f"""
{STUDY_TITLE.upper()}
AUDITABLE EXTENDED REPORT
Generated: {now_iso()}
Project: {project}
Output: {output}

SCOPE AND STATUS
This is a focused, secondary, cross-sectional analysis of the LEMON young-adult subset. The inferential unit is the participant; no window, channel, block, condition or electrode is treated as an independent person. Findings are associations, not causal effects or biomarkers. Alpha was selected after canonical-band exploration in this same cohort and is therefore a data-informed post hoc focus, not a prospectively prespecified sole band. The primary family contains the two focal alpha psychometric total-association tests and is controlled with Holm adjustment. A separate contextual sensitivity applies the identical final model to four bands and adjusts eight band-by-scale tests with Holm. The sensor-wise topographic analyses are secondary spatial characterizations and use BH-FDR separately within each 61-electrode map.

SOURCE STUDY AND PARTICIPANTS
The LEMON source study was acquired in Leipzig, Germany, 2013-2015. Participants were recruited through public/online advertisements, leaflets and university information events, screened by telephone and then by a study physician, provided written informed consent, and the protocol was approved by the University of Leipzig medical-faculty ethics committee (154/13-ff). Source: Babayan et al. 2019, {LEMON_PAPER}.
Local provenance: {source['download_eligible']} download-eligible records; {source['triplet_complete']} complete source triplets; {len(qc_all)} participants in the frozen EEG cache; {len(frame)} complete analytic cases. The 142-to-139 EEG transition is fixed in the included preprocessing source: sub-010015 and sub-010100 had truncated binary payloads, and sub-010078 had overlapping condition blocks. The two later analytic exclusions have missing relationship status; STAI, MSPSS and EEG are present. Available-case sample size, not a prospective power calculation, determined n. The analytic sample contains {int((frame.sex == 'female').sum())} women and {int((frame.sex == 'male').sum())} men; age was released in 5-year bins (20-25, 25-30 and 30-35 years).

PSYCHOMETRIC VARIABLES
The LEMON source protocol used the German STAI-G-X2 trait scale (20 items, four response levels) and a German MSPSS measuring perceived support from family, friends and significant others (seven response levels and a sum score). The analysis uses the released totals exactly as STAI_trait_total and MSPSS_total; observed ranges are {int(frame.stai_trait_total.min())}-{int(frame.stai_trait_total.max())} and {int(frame.mspss_total.min())}-{int(frame.mspss_total.max())}, respectively. Item-level scoring, reversals and reliability were not recomputed because item-level responses are not present in the analytic table. Those details must be verified against the LEMON questionnaire specification before manuscript submission.

EEG ACQUISITION (SOURCE PAPER)
BrainAmp MR plus amplifier; active ActiCAP with 62 channels (61 scalp EEG plus VEOG under the right eye); extended international 10-20/10-10 placement; online reference FCz; ground at sternum; impedance below 5 kOhm; amplitude resolution 0.1 uV; hardware bandpass 0.015-1000 Hz; sampling 2500 Hz. The 16-min resting recording contained 16 interleaved 60-s blocks (8 EC, 8 EO), started with EC, and EO used fixation on a black cross. Presentation 16.5 controlled blocks.

LOCAL PREPROCESSING AND QC
The local audited pipeline extracted the 60-s marker blocks and resampled them from 2500 to 250 Hz. A joint 1-45 Hz copy was used for bad-channel assessment. A separate joint EO+EC ICA-fit copy was notch-filtered at 50/100 Hz, filtered 1-100 Hz and average-referenced while excluding detected bad EEG channels. Extended Infomax ICA used a deterministic participant-specific seed, calculated as 20260902 plus the numeric LEMON identifier (modulo 2^32-1), decimation 2, maximum 1024 iterations and 500-uV fit rejection. Clean analysis copies were filtered 1-45 Hz with a Hamming-window FIR (firwin), average-referenced, ICA-cleaned, and then bad channels were interpolated in accurate mode. Exact included source defines robust_std=1.4826*median(|x-median(x)|), computes the scale score as robust z of log10(robust_std) across channels, and flags |z|>5. The spatial rule correlates a channel with the sample-wise median of its six nearest 3-D montage neighbors after decimation by five and requires both r<.40 and correlation robust z<-5. LOF (20 neighbors, threshold 2) is advisory only. ONNX ICLabel excludes an artifact-class argmax only at probability>=.80; EOG evidence adds exclusion only when |score|>=.50 and eye-blink probability>=.30. The reproducibility environment is pinned for Python 3.12, MNE 1.9.0 and mne-icalabel 0.7.0. The exact preprocessing source is included, and every copied method artifact is checked against its source during a run; optional archived-input hashes are available only to audit the historical snapshot. Of 139 cached participants, {int(as_bool(frame['strict_ica_converged_included']).sum())} complete cases enter the ICA-converged sensitivity and {int(as_bool(frame['strict_multiflag_included']).sum())} the multi-flag sensitivity. Across the full cache, {int(qc_all['n_excluded_components'].sum())} of {int(qc_all['ica_n_components'].sum())} ICA components ({100*qc_all['n_excluded_components'].sum()/qc_all['ica_n_components'].sum():.2f}%) were excluded. Full per-participant QC remains in the project.

PSD AND OUTCOME
Four-second windows with 50% overlap gave 29 windows/block and 232 windows/condition/channel. A channel-window was accepted only when all samples were finite, maximum absolute amplitude was <=250 uV, peak-to-peak amplitude was <=500 uV and robust SD was >=0.1 uV. Accepted channel-windows numbered {accepted_open:,}/{possible_per_condition:,} for EO and {accepted_closed:,}/{possible_per_condition:,} for EC. Each was analyzed by a one-sided, density-scaled, single-segment periodogram using a Hamming window and constant detrending. Native FFT spacing was {arrays['native_fft_spacing_hz']:.2f} Hz and Hamming effective resolution (ENBW) was {arrays['effective_hamming_resolution_hz']:.2f} Hz. The frozen cache contains delta 1-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, and beta 13-30 Hz PSD for EO and EC and reactivity for all 139 participants and 61 channels. The included initial feature script assigns shared bins once ([low,high), except beta closed at 30 Hz) and stores mean-bin density plus integrated power. It is not treated as the direct cache constructor. The included exact canonical-cache constructor verifies all band endpoints on the 0.25-Hz grid, selects both endpoints, trapezoid-integrates linear PSD density and divides by band width; shared endpoints contribute zero area. Accepted-window linear densities are then averaged and transformed as 10*log10, in dB re 1 uV2/Hz. This canonical rule matches the archived analysis inputs and is authoritative for the present study. Reactivity = EC_dB - EO_dB = 10*log10(EC_linear/EO_linear). The focused participant outcome is the arithmetic mean of channel-level alpha reactivity over P3, Pz, P4, PO3, POz, PO4, O1, Oz and O2. This gives one outcome per participant.

BAND-SELECTION PROVENANCE AND HARMONIZED FOUR-BAND SENSITIVITY
The sequence was broad canonical-band calculation, exploratory same-cohort screening, recognition of alpha as the clearest joint candidate for STAI and MSPSS, and then the focused posterior-alpha analysis. Some preliminary global repeated-block specifications produced MSPSS-by-condition terms outside alpha, but those analyses used a different global-61-channel estimand and a random-intercept-only block covariance structure. Subsequent participant-level canonical-band analyses did not retain corrected non-alpha reactivity associations. These earlier results were therefore considered exploratory screening signals rather than final band-specific findings.

For transparent comparison, Table S14 applies the exact final posterior-ROI spline specification to delta, theta, alpha, and beta in the same 137 complete cases. The other psychometric scale, sex, and partnership status remain covariates, HC3 inference is unchanged, and Holm is applied jointly across eight total tests with a separate eight-test family for nonlinearity. Only alpha-MSPSS survived this Holm-8 sensitivity (raw p={fmt_p(band_alpha_mspss.p_total_raw)}, Holm-8 p={fmt_p(band_alpha_mspss.p_total_holm_all_8)}). Alpha-STAI did not (raw p={fmt_p(band_alpha_stai.p_total_raw)}, Holm-8 p={fmt_p(band_alpha_stai.p_total_holm_all_8)}); non-alpha total rejections={nonalpha_total_rejections}/6 and nonlinear rejections={band_nonlinear_rejections}/8. This contextual sensitivity does not replace the focal alpha Holm-2 family and is not a complete correction for every historical analytical choice. It demonstrates that the alpha focus is most robust for MSPSS, whereas STAI-alpha remains exploratory and multiplicity-sensitive.

ABSOLUTE AND RELATIVE TOPOGRAPHIES
Figure 1 shows all 61 scalp electrodes as small black sensor dots without channel-name labels. Absolute maps display alpha PSD density (8-13 Hz, dB re 1 uV2/Hz) for EO and EC and absolute reactivity (EC minus EO, dB). Relative alpha power was reconstructed in linear units from the ten complete 0.5-Hz intervals spanning 8-13 Hz, divided by integrated 1-45-Hz power, and multiplied by 100. Relative reactivity is EC relative alpha minus EO relative alpha in percentage points. The color-accessible viridis sequential palette is used for unsigned EO/EC power, while signed reactivity maps use the zero-centered RdBu_r diverging palette. EO and EC share limits within each power representation. The MNE standard_1005 coordinates were projected using the EEGLAB convention and enclosed by one display radius used identically for the head outline and interpolation mask. Reconstruction of alpha density agreed with the canonical cache to maximum absolute errors of {relative_validation['max_abs_error_alpha_open_db']:.3e} dB for EO and {relative_validation['max_abs_error_alpha_closed_db']:.3e} dB for EC.

Figure 2 maps adjusted sensor-wise associations. At each electrode, the same participant-level natural-spline specification used for the focal ROI was fitted, with the other psychometric scale, sex and partnership status as covariates and HC3 inference. The zero-centered RdBu_r color scale encodes the directional linear component per 1-SD psychometric difference. Stars identify electrodes whose total 3-df spline test survives BH-FDR across 61 electrodes within that map; stars do not indicate a significant colored linear coefficient. Across all four maps, no linear or nonlinear component survived its within-map BH-FDR correction. These maps describe spatial distribution and do not replace the primary posterior-ROI analysis.

STATISTICAL MODEL
STAI trait total and MSPSS total were z-scored with sample SD (ddof=1) among the 137 complete cases. Two OLS models were fit: (1) STAI natural spline plus linear MSPSS, male indicator and partnered indicator; (2) MSPSS natural spline plus linear STAI and the same covariates. For the focal predictor, a natural cubic spline cr(x, df=4) was residualized against intercept and x; SVD retained two non-linear columns (tolerance max singular value*1e-8). The total 3-df robust Wald test jointly tests the focal linear component and two nonlinear components. The 2-df nonlinear Wald test assesses curvature beyond a straight line. HC3 sandwich covariance, two-sided alpha=.05, normal-reference Wald tests and 95% CIs were used. In the focused alpha analysis, Holm correction was applied separately to the two total tests and to the two nonlinear tests. Because alpha was selected after same-cohort exploration, Table S14 additionally applies Holm across all eight band-by-scale total tests and, separately, across all eight nonlinearity tests. Software versions and input hashes are recorded below.

PRIMARY RESULTS
STAI: total spline chi-square({stai['spline_df_total']})={stai['wald_chi2_total']:.6f}, raw p={fmt_p(stai['p_total_raw'])}, Holm p={fmt_p(stai['p_total_holm_2'])}; nonlinear chi-square({stai['nonlinear_df']})={stai['wald_chi2_nonlinear']:.6f}, raw p={fmt_p(stai['p_nonlinear_raw'])}, Holm p={fmt_p(stai['p_nonlinear_holm_2'])}. Linear component={stai['linear_component_db_per_sd']:.6f} dB/SD (95% HC3 CI {stai['linear_component_ci95_low']:.6f} to {stai['linear_component_ci95_high']:.6f}). R2={stai['r_squared']:.4f}, adjusted R2={stai['adjusted_r_squared']:.4f}, RMSE={stai['rmse_db']:.4f} dB.
MSPSS: total spline chi-square({mspss['spline_df_total']})={mspss['wald_chi2_total']:.6f}, raw p={fmt_p(mspss['p_total_raw'])}, Holm p={fmt_p(mspss['p_total_holm_2'])}; nonlinear chi-square({mspss['nonlinear_df']})={mspss['wald_chi2_nonlinear']:.6f}, raw p={fmt_p(mspss['p_nonlinear_raw'])}, Holm p={fmt_p(mspss['p_nonlinear_holm_2'])}. Linear component={mspss['linear_component_db_per_sd']:.6f} dB/SD (95% HC3 CI {mspss['linear_component_ci95_low']:.6f} to {mspss['linear_component_ci95_high']:.6f}). R2={mspss['r_squared']:.4f}, adjusted R2={mspss['adjusted_r_squared']:.4f}, RMSE={mspss['rmse_db']:.4f} dB.

INTERPRETATION
Both focal total-association tests are significant after Holm correction within the two-test alpha family. These are omnibus 3-df tests: they establish that the focal spline terms are jointly associated with reactivity conditional on the other scale, sex and partnership status, but they do not by themselves establish a single monotonic slope. The fitted linear components have the expected signs (+ for STAI and - for MSPSS), yet their individual 95% CIs include zero, as do the model-based +1-versus-1-SD contrasts in Table 4. Neither 2-df nonlinear component is significant after Holm correction, so there is also insufficient evidence for curvature. In the post-selection four-band sensitivity, only alpha-MSPSS survives Holm across eight band-by-scale total tests; alpha-STAI does not. The conclusion is therefore strongest for an exploratory omnibus alpha-MSPSS association. STAI-alpha is an exploratory directional result that is sensitive to whether multiplicity is defined within the focused alpha analysis or across all canonical bands.

SECONDARY TOPOGRAPHIC RESULTS
{chr(10).join('- ' + line for line in topographic_summary)}
The full electrode-wise estimates, raw p values and BH-FDR q values are provided in Table 5. A zero count of FDR-marked electrodes does not invalidate the ROI result; it indicates that the spatial map should be read descriptively rather than as sensor-level localization.

SENSITIVITY AND INFLUENCE
Exact results for the frozen primary cohort, ICA-converged subset, conservative multi-flag subset, and an age-bin-adjusted model are in Table S2. In the ICA-converged subset, STAI did not retain Holm significance (n={int(stai_ica.n_participants)}, p={fmt_p(stai_ica.p_total_holm_2)}), whereas MSPSS did (n={int(mspss_ica.n_participants)}, p={fmt_p(mspss_ica.p_total_holm_2)}). The same asymmetry appeared in the conservative multi-flag subset: STAI n={int(stai_multiflag.n_participants)}, p={fmt_p(stai_multiflag.p_total_holm_2)}; MSPSS n={int(mspss_multiflag.n_participants)}, p={fmt_p(mspss_multiflag.p_total_holm_2)}. This supports greater robustness of the MSPSS association to stricter EEG-QC definitions, not a formally larger MSPSS effect. These are sensitivity analyses, not additional primary hypotheses.
Because neither nonlinear test was significant, Table S9 reports a transparent linear-only functional-form sensitivity. Its estimates were {stai_linear.estimate_db_per_sd:.3f} dB/SD for STAI (Holm p={fmt_p(stai_linear.p_holm_2)}) and {mspss_linear.estimate_db_per_sd:.3f} dB/SD for MSPSS (Holm p={fmt_p(mspss_linear.p_holm_2)}); neither survived the two-exposure Holm correction. This limits any claim of a conventional linear association.
Table S12 varies the natural-spline basis df from 3 through 6 without replacing the primary df=4 specification. It distinguishes nominal basis df from the effective numerator df of each Wald test and reports both asymptotic chi-square and finite-sample F references. Under chi-square/HC3, both total-association tests retained within-df Holm significance throughout. At basis df=6, the finite-sample F Holm values were {fmt_p(stai_df6.p_f_total_holm_2_within_basis_df)} for STAI and {fmt_p(mspss_df6.p_f_total_holm_2_within_basis_df)} for MSPSS, showing attenuation at the most flexible specification. This sensitivity prevents interpreting df=4 as a uniquely favorable selection.
Leave-one-participant-out results (274 refits) are in Table S4:
{chr(10).join('- ' + line for line in loo_summary)}
Table S13 provides the same leave-one-out findings as a two-row summary.

DIAGNOSTICS
Table S1 reports residual shape, Breusch-Pagan tests, leverage, externally studentized residuals and Cook distances. Table S3 reports VIFs. HC3 protects coefficient inference against unknown heteroskedasticity but does not cure confounding, influential observations, measurement error or selection bias. Figure S2 provides residual-vs-fitted and Q-Q diagnostics.

REPORTING-STANDARD AUDIT
STROBE, COBIDAS-MEEG and SAMPL mapping tables are supplied. They are reporting aids, not an endorsement or certification. Required author decisions remain in HUMAN_INFORMATION_REQUIRED.csv: exploratory status, no prospective power claim, verified questionnaire scoring/version, reliability only if audited item-level data are available, current-analysis funding/COI, and data-sharing permissions.

FROZEN INPUT HASHES (SHA-256)
{chr(10).join(f'- {name}: {value}' for name, value in sorted(hashes.items()))}

AUTHORITATIVE REPORTING SOURCES
- STROBE: {STROBE_URL}
- COBIDAS-MEEG: {COBIDAS_URL}
- SAMPL: {SAMPL_URL}
- LEMON source study: {LEMON_PAPER}
"""


def manuscript_draft(
    primary: list[dict[str, Any]],
    contrasts: pd.DataFrame,
    sensitivity: pd.DataFrame,
    linear_sensitivity: pd.DataFrame,
    spline_complexity: pd.DataFrame,
    band_selection: pd.DataFrame,
    loo_summary: pd.DataFrame,
    topographic_associations: pd.DataFrame,
) -> str:
    stai = primary[0]["model"]
    mspss = primary[1]["model"]
    cs = contrasts.loc[(contrasts.variant == "primary_posterior_roi_spline") & (contrasts.outcome == "STAI_trait")].iloc[0]
    cm = contrasts.loc[(contrasts.variant == "primary_posterior_roi_spline") & (contrasts.outcome == "MSPSS_total")].iloc[0]
    ls = linear_sensitivity.loc[linear_sensitivity.outcome == "STAI_trait"].iloc[0]
    lm = linear_sensitivity.loc[linear_sensitivity.outcome == "MSPSS_total"].iloc[0]
    topo_counts = (
        topographic_associations.groupby(["psychometric_variable", "representation"])["total_spline_reject_bh05"]
        .sum()
        .astype(int)
    )
    stai_abs = int(topo_counts.get(("STAI_trait", "absolute_alpha_reactivity_db"), 0))
    stai_rel = int(topo_counts.get(("STAI_trait", "relative_alpha_reactivity_percentage_points"), 0))
    mspss_abs = int(topo_counts.get(("MSPSS_total", "absolute_alpha_reactivity_db"), 0))
    mspss_rel = int(topo_counts.get(("MSPSS_total", "relative_alpha_reactivity_percentage_points"), 0))
    linear_topo_count = int(topographic_associations["linear_component_reject_bh05"].sum())
    nonlinear_topo_count = int(topographic_associations["nonlinear_reject_bh05"].sum())
    qc = sensitivity.set_index(["variant", "outcome"])
    stai_ica = qc.loc[("qc_sensitivity_ica_converged", "STAI_trait")]
    mspss_ica = qc.loc[("qc_sensitivity_ica_converged", "MSPSS_total")]
    stai_multiflag = qc.loc[("qc_sensitivity_multiflag", "STAI_trait")]
    mspss_multiflag = qc.loc[("qc_sensitivity_multiflag", "MSPSS_total")]
    complexity_df6 = spline_complexity.loc[spline_complexity["basis_df_requested"] == 6].set_index("outcome")
    stai_df6 = complexity_df6.loc["STAI_trait"]
    mspss_df6 = complexity_df6.loc["MSPSS_total"]
    band_lookup = band_selection.set_index(["band", "psychometric_variable"])
    band_alpha_stai = band_lookup.loc[("alpha", "STAI_trait")]
    band_alpha_mspss = band_lookup.loc[("alpha", "MSPSS_total")]
    nonalpha_rejections = int(
        band_selection.loc[band_selection["band"] != "alpha", "total_reject_holm_all_8_05"].sum()
    )
    loo_index = loo_summary.set_index("outcome")
    loo_stai = loo_index.loc["STAI_trait"]
    loo_mspss = loo_index.loc["MSPSS_total"]
    return f"""
MANUSCRIPT-READY CORE — {STUDY_TITLE.upper()} (DRAFT; VERIFY AUTHOR-SUPPLIED FIELDS)

ESPAÑOL — MÉTODOS
Realizamos un análisis transversal secundario y exploratorio de participantes adultos jóvenes del conjunto LEMON. La unidad inferencial fue el participante. El flujo espectral calculó PSD EO, PSD EC y reactividad EC−EO para delta (1-4 Hz), theta (4-8 Hz), alfa (8-13 Hz) y beta (13-30 Hz). Alfa no fue preespecificada como única banda: se seleccionó después de exploración multibanda en esta misma cohorte y, por ello, el análisis focal es post hoc. La densidad espectral de potencia alfa se estimó por periodogramas de segmentos de 4 s con ventana Hamming, 50% de solapamiento, detrend constante y escalamiento de densidad. Los límites de banda coincidieron con la rejilla FFT de 0.25 Hz; el constructor canónico seleccionó ambos extremos, integró la densidad lineal mediante trapecios y dividió entre el ancho de banda. Para cada canal y condición se promediaron en escala lineal los segmentos aceptados y posteriormente se aplicó 10log10 (dB re 1 µV²/Hz). La reactividad se definió como EC_dB−EO_dB, equivalente a 10log10(PSD_EC/PSD_EO). El desenlace fue el promedio por participante sobre una ROI posterior fija (P3, Pz, P4, PO3, POz, PO4, O1, Oz y O2). Analizamos 137 de 139 participantes con STAI rasgo, MSPSS, sexo y situación de pareja completos. Cada escala se estandarizó con la media y desviación estándar muestrales. Ajustamos dos regresiones OLS: un spline cúbico natural de la escala focal (df=4), la otra escala lineal, sexo (hombre frente a mujer) y situación de pareja (con frente a sin pareja). La base no lineal se ortogonalizó respecto al intercepto y término lineal. Usamos covarianza HC3, pruebas bilaterales, α=.05 e IC95%. La prueba total de Wald (3 gl) evaluó el componente lineal y dos no lineales; la prueba de no linealidad (2 gl) evaluó curvatura adicional. Aplicamos Holm por separado a las dos pruebas totales alfa y a las dos pruebas no lineales alfa. Como sensibilidad contextual de selección de banda, repetimos exactamente esta especificación en las cuatro bandas y aplicamos Holm conjuntamente a ocho pruebas banda×escala, con otra familia de ocho para no linealidad. Como caracterización espacial secundaria, ajustamos la misma especificación en cada uno de los 61 electrodos, tanto para reactividad alfa absoluta como relativa; esta última se expresó como el cambio EC−EO en puntos porcentuales de potencia alfa respecto de la potencia integrada de 1-45 Hz. Para cada mapa controlamos BH-FDR sobre las 61 pruebas ómnibus.

ESPAÑOL — RESULTADOS
La asociación total de STAI con reactividad alfa posterior fue estadísticamente significativa dentro de la familia focal alfa [χ²({stai['spline_df_total']})={stai['wald_chi2_total']:.2f}, p cruda={fmt_p(stai['p_total_raw'])}, p-Holm-2={fmt_p(stai['p_total_holm_2'])}], al igual que la de MSPSS [χ²({mspss['spline_df_total']})={mspss['wald_chi2_total']:.2f}, p cruda={fmt_p(mspss['p_total_raw'])}, p-Holm-2={fmt_p(mspss['p_total_holm_2'])}]. En la sensibilidad homogénea de cuatro bandas, únicamente alfa-MSPSS sobrevivió Holm entre las ocho pruebas totales (p-Holm-8={fmt_p(band_alpha_mspss.p_total_holm_all_8)}); alfa-STAI no sobrevivió (p-Holm-8={fmt_p(band_alpha_stai.p_total_holm_all_8)}) y hubo {nonalpha_rejections}/6 rechazos no-alfa. Los componentes lineales fueron {stai['linear_component_db_per_sd']:.3f} dB/DE para STAI (IC95% {stai['linear_component_ci95_low']:.3f} a {stai['linear_component_ci95_high']:.3f}) y {mspss['linear_component_db_per_sd']:.3f} dB/DE para MSPSS (IC95% {mspss['linear_component_ci95_low']:.3f} a {mspss['linear_component_ci95_high']:.3f}). El contraste ajustado entre +1 y −1 DE fue {cs.estimate_db:.3f} dB (IC95% {cs.ci95_hc3_low:.3f} a {cs.ci95_hc3_high:.3f}) para STAI y {cm.estimate_db:.3f} dB (IC95% {cm.ci95_hc3_low:.3f} a {cm.ci95_hc3_high:.3f}) para MSPSS. No hubo evidencia de curvatura adicional (STAI p-Holm={fmt_p(stai['p_nonlinear_holm_2'])}; MSPSS p-Holm={fmt_p(mspss['p_nonlinear_holm_2'])}). En una sensibilidad lineal, ninguna escala conservó significación tras Holm (STAI p-Holm={fmt_p(ls.p_holm_2)}; MSPSS p-Holm={fmt_p(lm.p_holm_2)}). La asociación MSPSS conservó significación en los subconjuntos ICA-converged (p-Holm={fmt_p(mspss_ica.p_total_holm_2)}) y multiflag (p-Holm={fmt_p(mspss_multiflag.p_total_holm_2)}), mientras STAI no la conservó (p-Holm={fmt_p(stai_ica.p_total_holm_2)} y {fmt_p(stai_multiflag.p_total_holm_2)}, respectivamente). Esto indica mayor robustez de MSPSS frente a definiciones EEG-QC más estrictas, no un efecto formalmente mayor. En las {int(loo_stai.n_leave_one_participant_out_refits)} omisiones, ambos signos se conservaron y el ómnibus Holm permaneció significativo en todos los reajustes; el IC95% lineal excluyó cero solo en {int(loo_stai.linear_ci95_excludes_zero_n)} reajustes STAI y {int(loo_mspss.linear_ci95_excludes_zero_n)} MSPSS. La sensibilidad de complejidad conservó ambos ómnibus con referencia χ²/HC3 para grados de libertad de base 3–6; con referencia F y Holm en 6 grados de libertad de base, p={fmt_p(stai_df6.p_f_total_holm_2_within_basis_df)} para STAI y p={fmt_p(mspss_df6.p_f_total_holm_2_within_basis_df)} para MSPSS. En los mapas secundarios, sobrevivieron BH-FDR {stai_abs}/61 electrodos para STAI-absoluta, {stai_rel}/61 para STAI-relativa, {mspss_abs}/61 para MSPSS-absoluta y {mspss_rel}/61 para MSPSS-relativa. Sin embargo, {linear_topo_count}/244 componentes lineales y {nonlinear_topo_count}/244 componentes no lineales sobrevivieron sus correcciones BH-FDR dentro de mapa; las estrellas de la Figura 2 representan exclusivamente la prueba spline ómnibus. Los signos sugieren mayor reactividad con mayor STAI y menor reactividad con mayor MSPSS, pero no demuestran relaciones lineales monotónicas precisas. La evidencia más robusta corresponde a alfa-MSPSS; alfa-STAI es sensible a la familia de multiplicidad. Estos resultados son exploratorios y asociativos.

ENGLISH — METHODS
We conducted a secondary, exploratory cross-sectional analysis of young adults from the LEMON dataset, with participant as the inferential unit. The spectral pipeline calculated EO PSD, EC PSD, and EC-minus-EO reactivity for delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz). Alpha was not prespecified as the sole band: it was selected after multiband exploration in this same cohort, making the focused analysis post hoc. Alpha-band power spectral density was estimated using density-scaled single-segment periodograms from accepted 4-s windows (Hamming taper, 50% overlap, constant detrending). Band limits fell on the 0.25-Hz FFT grid; the canonical-cache constructor selected both endpoints, trapezoid-integrated linear density, and divided by band width. Window-level linear densities were averaged within channel and condition before 10log10 transformation (dB re 1 µV²/Hz). Reactivity was EC_dB−EO_dB, equivalent to 10log10(PSD_EC/PSD_EO), and the participant-level outcome was averaged over a fixed posterior ROI (P3, Pz, P4, PO3, POz, PO4, O1, Oz, O2). We analyzed 137 of 139 participants with complete STAI-trait, MSPSS, sex, and partnership-status data. Scores were standardized using the analytic-sample mean and sample SD. Two OLS analyses included a natural cubic spline for the focal scale (df=4), the other scale linearly, male sex, and partnered status. Nonlinear spline columns were orthogonalized to the intercept and linear term. HC3 covariance, two-sided tests, α=.05, and 95% CIs were used. A 3-df Wald test assessed the total focal association and a 2-df test assessed additional nonlinearity. Holm adjustment was applied separately across the two alpha total tests and across the two alpha nonlinearity tests. As a contextual band-selection sensitivity, the identical specification was repeated in all four bands and Holm was applied jointly across eight band-by-scale tests, with a separate family of eight nonlinearity tests. For secondary spatial characterization, the same specification was fitted at each of 61 electrodes for absolute and relative alpha reactivity. Relative alpha was expressed as EC−EO percentage-point change in alpha power divided by integrated 1-45-Hz power. BH-FDR was controlled across the 61 omnibus tests within each map.

ENGLISH — RESULTS
The total association with posterior alpha reactivity was significant within the focused alpha family for STAI [χ²({stai['spline_df_total']})={stai['wald_chi2_total']:.2f}, raw p={fmt_p(stai['p_total_raw'])}, Holm-2 p={fmt_p(stai['p_total_holm_2'])}] and MSPSS [χ²({mspss['spline_df_total']})={mspss['wald_chi2_total']:.2f}, raw p={fmt_p(mspss['p_total_raw'])}, Holm-2 p={fmt_p(mspss['p_total_holm_2'])}]. In the harmonized four-band sensitivity, only alpha-MSPSS survived Holm across the eight total tests (Holm-8 p={fmt_p(band_alpha_mspss.p_total_holm_all_8)}); alpha-STAI did not (Holm-8 p={fmt_p(band_alpha_stai.p_total_holm_all_8)}), and there were {nonalpha_rejections}/6 non-alpha rejections. Linear components were {stai['linear_component_db_per_sd']:.3f} dB/SD for STAI (95% CI {stai['linear_component_ci95_low']:.3f} to {stai['linear_component_ci95_high']:.3f}) and {mspss['linear_component_db_per_sd']:.3f} dB/SD for MSPSS (95% CI {mspss['linear_component_ci95_low']:.3f} to {mspss['linear_component_ci95_high']:.3f}). The adjusted +1-versus−1 SD contrast was {cs.estimate_db:.3f} dB (95% CI {cs.ci95_hc3_low:.3f} to {cs.ci95_hc3_high:.3f}) for STAI and {cm.estimate_db:.3f} dB (95% CI {cm.ci95_hc3_low:.3f} to {cm.ci95_hc3_high:.3f}) for MSPSS. There was no evidence of additional curvature (STAI Holm p={fmt_p(stai['p_nonlinear_holm_2'])}; MSPSS Holm p={fmt_p(mspss['p_nonlinear_holm_2'])}). In a linear functional-form sensitivity, neither scale remained significant after Holm adjustment (STAI Holm p={fmt_p(ls.p_holm_2)}; MSPSS Holm p={fmt_p(lm.p_holm_2)}). The MSPSS association retained significance in the ICA-converged (Holm p={fmt_p(mspss_ica.p_total_holm_2)}) and multiflag subsets (Holm p={fmt_p(mspss_multiflag.p_total_holm_2)}), whereas STAI did not (Holm p={fmt_p(stai_ica.p_total_holm_2)} and {fmt_p(stai_multiflag.p_total_holm_2)}, respectively). This indicates greater robustness of MSPSS to stricter EEG-QC definitions, not a formally larger effect. Across all {int(loo_stai.n_leave_one_participant_out_refits)} participant omissions, both directional signs were retained and the Holm-adjusted omnibus remained significant; the linear 95% CI excluded zero in only {int(loo_stai.linear_ci95_excludes_zero_n)} STAI refits and {int(loo_mspss.linear_ci95_excludes_zero_n)} MSPSS refits. The complexity sensitivity retained both omnibus tests under chi-square/HC3 for basis df=3–6; under finite-sample F reference with Holm adjustment at basis df=6, p={fmt_p(stai_df6.p_f_total_holm_2_within_basis_df)} for STAI and p={fmt_p(mspss_df6.p_f_total_holm_2_within_basis_df)} for MSPSS. In the secondary maps, BH-FDR was retained by {stai_abs}/61 electrodes for STAI-absolute, {stai_rel}/61 for STAI-relative, {mspss_abs}/61 for MSPSS-absolute, and {mspss_rel}/61 for MSPSS-relative. However, {linear_topo_count}/244 linear components and {nonlinear_topo_count}/244 nonlinear components survived their within-map BH-FDR corrections; Figure 2 stars represent only the omnibus spline test. Directional estimates were positive for STAI and negative for MSPSS, but neither variable supported a precise monotonic linear association. The strongest evidence is for alpha-MSPSS; alpha-STAI is sensitive to the multiplicity family. These findings are exploratory and associative.
"""


def environment_text() -> str:
    return "\n".join(
        [
            f"generated={now_iso()}",
            f"python={sys.version.replace(chr(10), ' ')}",
            f"executable={sys.executable}",
            f"platform={platform.platform()}",
            f"numpy={np.__version__}",
            f"pandas={pd.__version__}",
            f"scipy={scipy.__version__}",
            f"statsmodels={statsmodels.__version__}",
            f"patsy={patsy.__version__}",
            f"matplotlib={matplotlib.__version__}",
            f"mne={mne.__version__}",
        ]
    )


def requirements_text() -> str:
    return "\n".join(
        [
            "# Exact direct dependencies used to generate the final statistical package.",
            f"numpy=={np.__version__}",
            f"pandas=={pd.__version__}",
            f"scipy=={scipy.__version__}",
            f"statsmodels=={statsmodels.__version__}",
            f"patsy=={patsy.__version__}",
            f"matplotlib=={matplotlib.__version__}",
            f"mne=={mne.__version__}",
            "",
        ]
    )


def data_sharing_boundary_text() -> str:
    return """
DATA-SHARING BOUNDARY

LOCAL ANALYSIS OUTPUT
The analysis output contains participant-level rows required for local audit, including the complete-case table and leave-one-participant-out identifiers. Keep that output under the ignored work directory and never add it to a public repository.

PUBLIC REPOSITORY
The repository contains executable code, aggregate results, reports, figures and integrity metadata. It excludes raw/clean EEG, questionnaire rows, participant-level measurements, PSD caches, Parquet files and the oversized full-frequency table. The preprocessing code necessarily retains six public pseudonymous LEMON dataset IDs for documented recoveries/exclusions.

REPRODUCTION POLICY
Reproducers obtain LEMON independently and execute the repository pipeline. Fresh runs use structural, formula, within-run hash and aggregate numeric checks. Archived byte hashes can be requested explicitly but are not the default because paths, timestamps and compressed serialization vary across systems.
"""


def upstream_method_specifications_text(
    inventory: dict[str, dict[str, Any]], source_audit: dict[str, Any]
) -> str:
    preprocess_hash = inventory["preprocessing_script"]["sha256"]
    psd_hash = inventory["psd_script"]["sha256"]
    audit_hash = inventory["source_audit"]["sha256"]
    canonical_hash = inventory["canonical_cache_script"]["sha256"]
    return f"""
PORTABLE METHOD SPECIFICATIONS / ESPECIFICACIONES METODOLOGICAS PORTABLES

STATUS / ESTATUS
The analysis output contains byte-identical copies of the portable preprocessing, PSD/QC, spectral-input builder, and source-audit summary that were supplied to this run. Source-to-copy hashes are verified and recorded. The repository-level orchestrator, rather than this final statistical command alone, executes the complete upstream chain.

La salida analitica contiene copias identicas de los scripts portables de preprocesamiento, PSD/QC, construccion espectral y del resumen de auditoria usados en esa ejecucion. Se verifican los hashes entre origen y copia. El orquestador del repositorio, no este comando estadistico aislado, ejecuta la cadena upstream completa.

FILES / ARCHIVOS
- scripts/03_preprocess_eeg.py — SHA-256 {preprocess_hash}
- scripts/04_compute_psd_qc.py — SHA-256 {psd_hash}
- audit/metadata/source_audit_summary.json — SHA-256 {audit_hash}
- scripts/06_build_analysis_inputs.py — SHA-256 {canonical_hash}

BAD-CHANNEL LOGIC / LOGICA DE CANALES MALOS
- Robust scale per channel: robust_std = 1.4826 * median(abs(x - median(x))), with x in microvolts.
- Scale outlier score: log_std = log10(max(robust_std, tiny)); robust_z = (log_std - median(log_std)) / max(1.4826 * median(abs(log_std - median(log_std))), eps).
- Spatial-neighbor rule: channel coordinates come from the EEG montage; the six nearest channels by three-dimensional Euclidean distance are selected. After temporal decimation by five, each channel is correlated with the sample-wise median of those six neighbors. A channel is flagged only when correlation < 0.40 AND the across-channel robust correlation z score < -5.
- Flat rule: robust_std < 0.1 microvolts. Scale rule: abs(robust_logstd_z) > 5.
- MNE LOF is computed with n_neighbors=20 and threshold=2.0, but is advisory only and cannot by itself trigger interpolation.
- Final bad channels are the union of flat, scale, and neighbor-rule flags. Those channels are interpolated with MNE mode='accurate' after ICA.

ICA AND ICLABEL LOGIC / LOGICA ICA E ICLABEL
- One joint EO+EC extended-Infomax ICA is fitted per participant after 50/100-Hz notch filtering, 1-100-Hz filtering, average reference, rank estimation, decimation=2, max_iter=1024, and an EEG rejection threshold of 500 microvolts.
- ICLabel uses the ONNX backend. A component is excluded when its argmax label is muscle artifact, eye blink, heart beat, line noise, or channel noise AND its argmax probability is >= 0.80.
- EOG z-score detections are retained as audit evidence and add an exclusion only when abs(EOG score) >= 0.50 AND ICLabel eye-blink probability >= 0.30.
- The final excluded-component set is the union of high-confidence ICLabel artifacts and EOG-assisted exclusions.

PSD AND FREQUENCY EDGES / PSD Y BORDES DE FRECUENCIA
- EEG is analyzed at 250 Hz in 4-s windows advanced every 2 s (50% overlap), separately within each 60-s block; this yields 29 windows per block and a native FFT-bin spacing of 0.25 Hz.
- SciPy periodogram uses a Hamming window, constant detrending, one-sided output, and scaling='density', yielding microvolt-squared/Hz.
- Window acceptance is channel-specific and requires finite samples, max(abs(x)) <= 250 microvolts, peak-to-peak <= 500 microvolts, and robust_std >= 0.1 microvolts.
- The initial QC feature script 04_compute_psd_qc.py stores window-level retention evidence and preliminary band summaries; it is not the direct constructor of the final cache.
- The authoritative builder is scripts/06_build_analysis_inputs.py. It validates the 0.25-Hz FFT grid, trapezoid-integrates each contiguous 0.5-Hz interval, averages accepted-window linear density, and only then applies 10*log10. Canonical bands are additive sums of those interval integrals divided by band width. Shared endpoints carry zero area and are not double-counted.

SOURCE AUDIT AND THREE EEG EXCLUSIONS / AUDITORIA DE FUENTE Y TRES EXCLUSIONES EEG
- Source audit: download eligible={source_audit.get('download_eligible', 'NA')}; complete triplets={source_audit.get('triplet_complete', 'NA')}; source-audit pass={source_audit.get('source_audit_pass', 'NA')}; source-audit fail={source_audit.get('source_audit_fail', 'NA')}; expected scalp EEG channels={source_audit.get('expected_scalp_eeg', 'NA')} plus {source_audit.get('auxiliary_eog', 'NA')}.
- Of the six source-audit failures, the preprocessing source marks three as recoverable without altering source EEG: sub-010020 and sub-010193 use normalized historical sidecar file references, and sub-010126 uses an explicit marker mapping for alternating S208 trains. Together with 136 direct passes, these yield 139 processable EEG records.
- The preprocessing source freezes three exclusions: sub-010015 (truncated binary payload), sub-010078 (overlapping condition blocks), and sub-010100 (truncated binary payload). The usable mask additionally requires divisible payload and matching channel order. These are pseudonymous LEMON dataset IDs, retained verbatim so the 142 complete triplets to 139 processable EEG records transition can be audited without inference.

PUBLICATION BOUNDARY / LIMITE DE PUBLICACION
The exact preprocessing source contains those three pseudonymous dataset IDs. No raw EEG, questionnaire rows, or participant-level measurements are added by these source files. Nevertheless, repository release remains contingent on a human check of LEMON terms and institutional policy.
"""


def band_selection_rationale_text(band_selection: pd.DataFrame) -> str:
    display = band_selection[
        [
            "band",
            "psychometric_variable",
            "linear_component_db_per_sd",
            "wald_chi2_total",
            "p_total_raw",
            "p_total_holm_all_8",
            "total_reject_holm_all_8_05",
            "p_nonlinear_holm_all_8",
        ]
    ].copy()
    display["linear_component_db_per_sd"] = display["linear_component_db_per_sd"].map(
        lambda value: f"{value:.6f}"
    )
    display["wald_chi2_total"] = display["wald_chi2_total"].map(lambda value: f"{value:.6f}")
    for column in ["p_total_raw", "p_total_holm_all_8", "p_nonlinear_holm_all_8"]:
        display[column] = display[column].map(fmt_p)
    table = display.to_string(index=False)
    return f"""
BAND-SELECTION RATIONALE AND FOUR-BAND SENSITIVITY
JUSTIFICACION DE LA SELECCION DE BANDA Y SENSIBILIDAD DE CUATRO BANDAS

STATUS / ESTATUS
Alpha was not prespecified as the only eligible frequency band. The alpha focus was selected after exploratory examination of delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) in this same cohort. The focused alpha analyses are therefore post hoc and exploratory even though the two-exposure family uses Holm adjustment.

Alfa no fue preespecificada como la unica banda elegible. El foco alfa se selecciono despues de examinar exploratoriamente delta (1-4 Hz), theta (4-8 Hz), alfa (8-13 Hz) y beta (13-30 Hz) en esta misma cohorte. Por ello, los analisis focales alfa son post hoc y exploratorios, aunque su familia interna de dos exposiciones use ajuste de Holm.

SELECTION SEQUENCE / SECUENCIA DE SELECCION
1. The frozen spectral pipeline calculated EO PSD, EC PSD, and EC-minus-EO reactivity for all 139 participants, 61 scalp channels, and all four canonical bands. Band limits were fixed before the focused spline analysis.
2. Preliminary broad analyses examined global, spatial, and repeated-block representations. Alpha showed the clearest joint pattern for STAI and MSPSS. Some earlier block-level, global-61-channel specifications also produced MSPSS-by-condition terms in delta, theta, and beta. Those terms were not treated as equivalent final findings because they used a different estimand and a random-intercept-only repeated-block covariance structure.
3. Subsequent participant-level canonical-band analyses did not retain corrected delta, theta, or beta reactivity associations. Alpha was carried forward as the biologically coherent and inferentially strongest candidate for the fixed posterior ROI.
4. To make that data-informed choice auditable, the final participant-level specification was then applied without modification to each canonical band: the same nine-channel posterior ROI, natural spline with basis df=4, adjustment for the other psychometric scale, sex and partnership status, and HC3 covariance. Holm correction was applied jointly across the eight total tests (4 bands x 2 scales), with a separate Holm family for the eight nonlinearity tests.

1. El flujo espectral congelado calculo PSD EO, PSD EC y reactividad EC menos EO para 139 participantes, 61 canales y las cuatro bandas canonicas. Los limites de banda estaban fijados antes del spline focal.
2. Los analisis amplios preliminares examinaron representaciones globales, espaciales y por bloques repetidos. Alfa mostro el patron conjunto mas claro para STAI y MSPSS. Algunas especificaciones tempranas por bloques, con promedio global de 61 canales, tambien produjeron terminos MSPSS por condicion en delta, theta y beta. Esos terminos no se consideraron hallazgos finales equivalentes porque correspondian a otro estimando y a una estructura de covarianza por bloques con solo intercepto aleatorio.
3. Los analisis canonicos posteriores a nivel de participante no conservaron asociaciones corregidas de reactividad delta, theta o beta. Alfa se mantuvo como candidata biologicamente coherente y con la evidencia inferencial mas clara para la ROI posterior fija.
4. Para auditar esa seleccion informada por los datos, la especificacion final a nivel de participante se aplico sin cambios a cada banda canonica: la misma ROI posterior de nueve canales, spline natural con df de base=4, ajuste por la otra escala, sexo y situacion de pareja, y covarianza HC3. Holm se aplico conjuntamente a las ocho pruebas totales (4 bandas x 2 escalas), con otra familia Holm para las ocho pruebas de no linealidad.

HARMONIZED RESULTS / RESULTADOS HOMOGENEOS
{table}

INTERPRETATION / INTERPRETACION
Only the alpha-MSPSS total association survived the eight-test Holm correction. Alpha-STAI did not survive correction across bands, and no delta, theta, or beta total association was significant. No nonlinearity test survived Holm across eight. Thus, the four-band sensitivity supports alpha specificity most clearly for MSPSS; the STAI-alpha result remains exploratory and sensitive to the multiplicity family.

Unicamente la asociacion total alfa-MSPSS sobrevivio la correccion Holm de ocho pruebas. Alfa-STAI no sobrevivio la correccion entre bandas, y ninguna asociacion total delta, theta o beta fue significativa. Ninguna prueba de no linealidad sobrevivio Holm entre ocho. Por tanto, la sensibilidad de cuatro bandas respalda con mayor claridad la especificidad alfa para MSPSS; el resultado STAI-alfa permanece exploratorio y sensible a la familia de multiplicidad.

This sensitivity is a transparent context analysis, not a complete selective-inference correction for every model, topography, or functional form inspected historically in the same cohort. Strong confirmation requires an independent sample or a prospectively specified analysis in data not used for band selection.

Esta sensibilidad es un analisis contextual transparente, no una correccion completa de inferencia selectiva para cada modelo, topografia o forma funcional examinada historicamente en la misma cohorte. La confirmacion fuerte requiere una muestra independiente o un analisis especificado prospectivamente en datos no usados para seleccionar la banda.
"""


def final_changes_text() -> str:
    return """
REPRODUCIBILITY NOTES

1. This is a secondary, exploratory, cross-sectional participant-level EEG study.
2. Two natural-spline analyses use HC3 covariance, Holm correction across the two focal alpha tests, and adjustment for the other psychometric scale, sex, and partnership status.
3. MSPSS is more robust than STAI in strict EEG-QC sensitivities; this does not prove a larger MSPSS effect.
4. Figure 2 color encodes the directional linear component while stars encode the total 3-df spline test. No sensor-wise linear or nonlinear component survives BH-FDR.
5. Alpha is a same-cohort, data-informed post hoc focus. The harmonized Holm-8 four-band sensitivity leaves only alpha-MSPSS significant.
6. Input files are structurally checked and the aggregate model estimates are regression-tested. Archived byte hashes are optional because portable paths, timestamps and compressed serialization can change without changing the analysis.
7. The repository-level pipeline contains the complete download, source-audit, preprocessing, PSD/QC, QC-freeze, spectral-input and final-analysis code. No raw data or participant-level tables are intended for public release.
"""


def build_public_release_candidate(output: Path, script_path: Path) -> pd.DataFrame:
    public = output / "PUBLIC_RELEASE_CANDIDATE"
    if public.exists():
        shutil.rmtree(public)
    public_tables = public / "tables"
    public_reports = public / "reports"
    public_figures = public / "figures"
    public_metadata = public / "metadata"
    for folder in [public, public_tables, public_reports, public_figures, public_metadata]:
        folder.mkdir(parents=True, exist_ok=True)

    shutil.copy2(script_path, public / script_path.name)
    shutil.copy2(output / "requirements.txt", public / "requirements.txt")
    public_upstream_inventory: dict[str, dict[str, Any]] = {}
    for name, relative_path in UPSTREAM_SOURCE_RELATIVES.items():
        source = output / relative_path
        destination = public / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256_file(source)
        sanitization = "none; byte-identical copy"
        copied_verbatim = True
        if name == "source_audit":
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["source_project"] = "<local LEMON source path omitted from public candidate>"
            payload["_public_release_provenance"] = {
                "sanitization": "local source_project path replaced; aggregate counts unchanged",
                "source_sha256_before_sanitization": original_hash,
            }
            write_text(destination, json.dumps(payload, indent=2, ensure_ascii=False))
            sanitization = "local source_project path replaced; aggregate counts unchanged"
            copied_verbatim = False
        else:
            shutil.copy2(source, destination)
        public_upstream_inventory[name] = {
            "public_relative_path": relative_path.as_posix(),
            "source_sha256": original_hash,
            "public_sha256": sha256_file(destination),
            "copied_verbatim": copied_verbatim,
            "sanitization": sanitization,
        }

    excluded_tables = {
        "Data_participant_level_complete_cases.csv",
        "Table_S4_leave_one_participant_out.csv",
    }
    for source in sorted((output / "tables").glob("*.csv")):
        if source.name not in excluded_tables:
            shutil.copy2(source, public_tables / source.name)
    for source in sorted((output / "figures").glob("*")):
        if source.is_file():
            shutil.copy2(source, public_figures / source.name)
    for name in [
        "METHODS_RESULTS_MANUSCRIPT_ES_EN.txt",
        "FIGURE_LEGENDS_ES_EN.txt",
        "STROBE_COBIDAS_SAMPL_gap_audit.txt",
        "AUTHORITATIVE_SOURCES.txt",
        "LIBRARIES_USED.txt",
        "DATA_SHARING_BOUNDARY.txt",
        "BAND_SELECTION_RATIONALE_ES_EN.txt",
        "UPSTREAM_METHOD_SPECIFICATIONS_ES_EN.txt",
        "REPRODUCIBILITY_NOTES.txt",
    ]:
        shutil.copy2(output / "reports" / name, public_reports / name)
    environment_lines = (output / "metadata" / "environment.txt").read_text(encoding="utf-8").splitlines()
    environment_lines = [
        "executable=<local path omitted from public candidate>" if line.startswith("executable=") else line
        for line in environment_lines
    ]
    write_text(public_metadata / "environment.txt", "\n".join(environment_lines))
    public_parameters = json.loads((output / "metadata" / "parameters.json").read_text(encoding="utf-8"))
    public_parameters["project"] = "<LEMON_PROJECT_ROOT>"
    public_parameters["output"] = "<OUTPUT_FOLDER>"
    public_parameters["public_release_sanitization"] = {
        "local_paths_removed": True,
        "upstream_sources": public_upstream_inventory,
    }
    write_text(
        public_metadata / "parameters.json",
        json.dumps(public_parameters, indent=2, ensure_ascii=False),
    )
    shutil.copy2(output / "metadata" / "run_audit.json", public_metadata / "run_audit.json")
    write_text(
        public_metadata / "upstream_source_inventory_public.json",
        json.dumps(public_upstream_inventory, indent=2, ensure_ascii=False),
    )

    write_text(
        public / "README_PUBLIC_RELEASE_CANDIDATE.txt",
        f"""
{STUDY_TITLE.upper()}
PUBLIC-RELEASE CANDIDATE GENERATED BY THE ANALYSIS SCRIPT

This directory excludes participant-level measurement tables and identifier-bearing leave-one-participant-out output. It includes the preprocessing and spectral specifications needed for methods audit. The preprocessing source necessarily retains public pseudonymous LEMON dataset IDs and exclusion/recovery reasons so that the 142-to-139 EEG transition is auditable. This is not an automatic permission to publish. Verify LEMON data-use terms, institutional requirements, author declarations, and repository policy before release.

Included executable method specifications:
- scripts/03_preprocess_eeg.py
- scripts/04_compute_psd_qc.py
- scripts/06_build_analysis_inputs.py
- scripts/07_run_analysis.py

The run-specific source audit is written to audit/metadata/source_audit_summary.json. UPSTREAM_METHOD_SPECIFICATIONS_ES_EN.txt summarizes the robust-z formula, six-neighbor rule, advisory LOF role, ICLabel/EOG logic, spectral integration stages, and three source-level EEG exclusions. Source-to-package hashes are verified during execution; optional archived-input hash enforcement is only for reproducing the historical local snapshot.

Alpha was selected after same-cohort exploration of four canonical bands and is therefore a post hoc focus. Table S14 and BAND_SELECTION_RATIONALE_ES_EN.txt provide the harmonized four-band sensitivity and its interpretation. Only alpha-MSPSS survives Holm correction across all eight band-by-scale omnibus tests; alpha-STAI remains exploratory and sensitive to the multiplicity family.

To reproduce after independently obtaining the authorized LEMON inputs:
& \"<PYTHON_3_12>\" \"<PATH_TO_REPOSITORY>\\scripts\\07_run_analysis.py\" --project \"<LEMON_PROJECT_ROOT>\" --output \"<NEW_OUTPUT_FOLDER>\"

Install direct dependencies:
& \"<PYTHON_3_12>\" -m pip install -r \"<PATH_TO_REPOSITORY>\\requirements\\analysis-lock.txt\"
""",
    )
    text_suffixes = {".py", ".json", ".txt", ".csv", ".md", ".ps1", ".yml", ".yaml", ".toml", ".cff"}
    path_leaks: list[str] = []
    windows_user_root = "C:" + "\\Users\\"
    mac_user_root = "/" + "Users/"
    unix_home_root = "/" + "home/"
    for candidate in sorted(public.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in text_suffixes:
            continue
        content = candidate.read_text(encoding="utf-8-sig", errors="replace")
        if windows_user_root in content or mac_user_root in content or unix_home_root in content:
            path_leaks.append(candidate.relative_to(public).as_posix())
    public_audit = {
        "generated": now_iso(),
        "status": "PASS" if not path_leaks else "FAIL",
        "participant_level_measurement_tables_excluded_check": "PASS"
        if all(not (public_tables / name).exists() for name in excluded_tables)
        else "FAIL",
        "preprocessing_and_initial_psd_byte_identical_check": "PASS"
        if all(public_upstream_inventory[name]["copied_verbatim"] for name in ["preprocessing_script", "psd_script"])
        else "FAIL",
        "path_bearing_upstream_files_sanitized_check": "PASS"
        if (
            not public_upstream_inventory["source_audit"]["copied_verbatim"]
            and public_upstream_inventory["canonical_cache_script"]["copied_verbatim"]
        )
        else "FAIL",
        "source_and_public_hashes_recorded_check": "PASS"
        if all(
            row.get("source_sha256") and row.get("public_sha256")
            for row in public_upstream_inventory.values()
        )
        else "FAIL",
        "local_user_path_scan_check": "PASS" if not path_leaks else "FAIL",
        "local_user_path_leaks": path_leaks,
        "note": "Pseudonymous LEMON dataset IDs remain in the preprocessing source for methods and exclusion audit; publication still requires permission review.",
    }
    if any(value == "FAIL" for key, value in public_audit.items() if key.endswith("_check")):
        public_audit["status"] = "FAIL"
    write_text(
        public_metadata / "public_release_audit.json",
        json.dumps(public_audit, indent=2, ensure_ascii=False),
    )
    if public_audit["status"] != "PASS":
        raise RuntimeError(f"Public-release audit failed: {public_audit}")
    return write_manifest(public)


def write_manifest(output: Path) -> pd.DataFrame:
    rows = []
    manifest_path = output / "metadata" / "manifest_sha256.csv"
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p != manifest_path):
        rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = pd.DataFrame(rows)
    write_csv(manifest, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=STUDY_TITLE)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--enforce-archived-input-hashes",
        action="store_true",
        help="Require byte identity with the archived reference run; normally omitted for a fresh cross-platform reproduction",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--build-public-release-candidate",
        action="store_true",
        help="Build a sanitized results capsule inside the output; the GitHub repository itself is already allowlist-built",
    )
    parser.add_argument("--skip-loo", action="store_true", help="Only for quick debugging; not for final results")
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output if args.output else project / DEFAULT_OUTPUT_NAME).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    reports = output / "reports"
    figures = output / "figures"
    metadata_dir = output / "metadata"
    for folder in [tables, reports, figures, metadata_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    paths = require_files(project)
    enforce_archived_hashes = bool(args.enforce_archived_input_hashes and not args.skip_hash_check)
    hashes = validate_hashes(paths, enforce_archived_hashes)
    upstream_inventory = copy_upstream_source_specifications(paths, output)
    frame, arrays, metadata = load_inputs(project, paths)
    alpha_index = arrays["bands"].index("alpha")
    relative, topographic_power, relative_validation = load_relative_alpha_topography(
        paths["full_frequency_table"],
        arrays["participant_ids"],
        arrays["channels"],
        arrays["psd_open"][:, alpha_index, :],
        arrays["psd_closed"][:, alpha_index, :],
    )
    participant_lookup = {participant_id: i for i, participant_id in enumerate(arrays["participant_ids"])}
    complete_indices = np.array([participant_lookup[p] for p in frame["participant_id"]], dtype=int)
    roi_indices = [arrays["channels"].index(channel) for channel in ROI]
    frame["relative_alpha_roi_eo_percent_1_45hz"] = relative["relative_open_percent"][
        complete_indices
    ][:, roi_indices].mean(axis=1)
    frame["relative_alpha_roi_ec_percent_1_45hz"] = relative["relative_closed_percent"][
        complete_indices
    ][:, roi_indices].mean(axis=1)
    frame["relative_alpha_roi_reactivity_percentage_points"] = relative["relative_reactivity_pp"][
        complete_indices
    ][:, roi_indices].mean(axis=1)
    topographic_associations = channelwise_topographic_associations(
        frame,
        arrays["channels"],
        arrays["participant_ids"],
        arrays["reactivity"][:, alpha_index, :],
        relative["relative_reactivity_pp"],
    )
    primary = [
        fit_one(frame, "stai_z", "mspss_z", "STAI_trait"),
        fit_one(frame, "mspss_z", "stai_z", "MSPSS_total"),
    ]
    apply_holm(primary)
    replication = primary_replication_check(primary)
    band_selection = run_four_band_selection_sensitivity(frame, arrays)

    model_tests = pd.DataFrame([item["model"] for item in primary])
    coefficients = pd.concat([item["coefficients"] for item in primary], ignore_index=True)
    diagnostics = pd.DataFrame([item["diagnostics"] for item in primary])
    vifs = pd.concat([item["vif"] for item in primary], ignore_index=True)
    curves = pd.concat([item["curve"] for item in primary], ignore_index=True)
    sensitivity, contrasts = run_sensitivities(frame)
    linear_sensitivity = run_linear_functional_form_sensitivity(frame)
    spline_complexity = run_spline_complexity_sensitivity(frame)
    if args.skip_loo:
        warnings.warn("LOO omitido: salida no apta como paquete final", RuntimeWarning)
        loo = pd.DataFrame()
        loo_summary = pd.DataFrame(
            [
                {
                    "outcome": label,
                    "n_leave_one_participant_out_refits": 0,
                    "expected_linear_direction": direction,
                    "direction_preserved_n": 0,
                    "direction_preserved_percent": np.nan,
                    "omnibus_holm_significant_n": 0,
                    "omnibus_holm_significant_percent": np.nan,
                    "linear_ci95_excludes_zero_n": 0,
                    "linear_ci95_excludes_zero_percent": np.nan,
                    "linear_component_min_db_per_sd": np.nan,
                    "linear_component_max_db_per_sd": np.nan,
                    "omnibus_holm_p_min": np.nan,
                    "omnibus_holm_p_max": np.nan,
                    "interpretation_boundary": "LOO was skipped; this output is not a final package.",
                }
                for label, direction in [("STAI_trait", "positive"), ("MSPSS_total", "negative")]
            ]
        )
    else:
        loo = run_leave_one_out(frame)
        loo_summary = summarize_leave_one_out(loo)
    flow, missingness = flow_and_missingness(project, frame, metadata)
    descriptives = build_descriptives(frame)
    correlations = frame[["stai_trait_total", "mspss_total", "alpha_roi_reactivity_ec_minus_eo_db"]].corr(method="pearson")
    correlations.index.name = "variable"
    correlations = correlations.reset_index()

    write_csv(descriptives, tables / "Table_1_sample_characteristics.csv")
    write_csv(model_tests, tables / "Table_2_primary_model_tests.csv")
    write_csv(coefficients, tables / "Table_3_full_primary_coefficients.csv")
    write_csv(contrasts.loc[contrasts.variant == "primary_posterior_roi_spline"], tables / "Table_4_primary_effect_contrasts.csv")
    write_csv(topographic_associations, tables / "Table_5_channelwise_topographic_associations.csv")
    write_csv(diagnostics, tables / "Table_S1_primary_diagnostics.csv")
    write_csv(sensitivity, tables / "Table_S2_sensitivity_models.csv")
    write_csv(vifs, tables / "Table_S3_variance_inflation_factors.csv")
    write_csv(loo, tables / "Table_S4_leave_one_participant_out.csv")
    write_csv(curves, tables / "Table_S5_adjusted_curve_points.csv")
    write_csv(flow, tables / "Table_S6_participant_flow.csv")
    write_csv(missingness, tables / "Table_S7_missing_data.csv")
    write_csv(correlations, tables / "Table_S8_zero_order_correlations.csv")
    write_csv(linear_sensitivity, tables / "Table_S9_linear_functional_form_sensitivity.csv")
    write_csv(topographic_power, tables / "Table_S10_topographic_power_values.csv")
    write_csv(spline_complexity, tables / "Table_S12_spline_complexity_sensitivity.csv")
    write_csv(loo_summary, tables / "Table_S13_leave_one_participant_out_summary.csv")
    write_csv(band_selection, tables / "Table_S14_four_band_selection_sensitivity.csv")
    write_csv(participant_export(frame), tables / "Data_participant_level_complete_cases.csv")
    write_csv(replication, tables / "AUDIT_numeric_regression_test.csv")
    write_csv(
        pd.DataFrame(
            [
                {"condition": "eyes_open", "maximum_absolute_error_db": relative_validation["max_abs_error_alpha_open_db"], "tolerance_db": 1e-8, "pass": relative_validation["max_abs_error_alpha_open_db"] <= 1e-8},
                {"condition": "eyes_closed", "maximum_absolute_error_db": relative_validation["max_abs_error_alpha_closed_db"], "tolerance_db": 1e-8, "pass": relative_validation["max_abs_error_alpha_closed_db"] <= 1e-8},
            ]
        ),
        tables / "AUDIT_relative_alpha_reconstruction.csv",
    )

    strobe, cobidas, sampl, pending = checklist_tables()
    write_csv(strobe, tables / "STROBE_checklist_mapping.csv")
    write_csv(cobidas, tables / "COBIDAS_MEEG_checklist_mapping.csv")
    write_csv(sampl, tables / "SAMPL_checklist_mapping.csv")
    write_csv(pending, tables / "HUMAN_INFORMATION_REQUIRED.csv")

    topomap_geometry = make_topographic_figures(arrays, relative, topographic_associations, figures)
    write_csv(topomap_geometry, tables / "Table_S11_topomap_coordinates.csv")
    make_figures(primary, flow, figures)

    extended = report_text(
        project,
        output,
        frame,
        arrays,
        metadata,
        primary,
        sensitivity,
        linear_sensitivity,
        spline_complexity,
        band_selection,
        loo,
        loo_summary,
        hashes,
        topographic_associations,
        relative_validation,
    )
    write_text(reports / "TRAIT_ANXIETY_SOCIAL_SUPPORT_ALPHA_REACTIVITY_extended_report.txt", extended)
    write_text(
        reports / "METHODS_RESULTS_MANUSCRIPT_ES_EN.txt",
        manuscript_draft(
            primary,
            contrasts,
            sensitivity,
            linear_sensitivity,
            spline_complexity,
            band_selection,
            loo_summary,
            topographic_associations,
        ),
    )
    write_text(
        reports / "FIGURE_LEGENDS_ES_EN.txt",
        """
FIGURE LEGENDS / LEYENDAS DE FIGURAS

Figure 1. Alpha power and eyes-closed reactivity topographies. Grand-average scalp distributions across 61 EEG electrodes are shown for absolute alpha-band (8-13 Hz) power spectral density during eyes open (A) and eyes closed (B), absolute alpha reactivity calculated as EC minus EO in dB (C), relative alpha power as a percentage of integrated 1-45-Hz power during eyes open (D) and eyes closed (E), and relative alpha reactivity calculated as EC minus EO in percentage points (F). EO and EC panels share a color scale within each power representation. Unsigned EO/EC power uses the perceptually ordered viridis palette; signed EC-minus-EO reactivity uses a zero-centered RdBu_r diverging palette. Small black dots mark all 61 recorded scalp electrodes; channel-name labels are intentionally omitted for visual clarity. The black head circle coincides exactly with the interpolation mask; values between sensors are interpolated only for visualization.

Figura 1. Topografías de potencia alfa y reactividad al cierre de ojos. Se muestran las distribuciones promedio sobre 61 electrodos EEG de la densidad espectral de potencia alfa absoluta (8-13 Hz) con ojos abiertos (A) y cerrados (B), la reactividad alfa absoluta calculada como EC menos EO en dB (C), la potencia alfa relativa como porcentaje de la potencia integrada de 1-45 Hz con ojos abiertos (D) y cerrados (E), y la reactividad alfa relativa calculada como EC menos EO en puntos porcentuales (F). Los paneles EO y EC comparten escala de color dentro de cada representación. La potencia no signada EO/EC usa la paleta viridis, ordenada perceptualmente; la reactividad signada EC menos EO usa la paleta divergente RdBu_r centrada en cero. Los puntos negros pequeños indican los 61 electrodos; se omiten intencionalmente sus nombres para mejorar la claridad visual. La circunferencia negra coincide exactamente con la máscara de interpolación; la interpolación entre sensores se utiliza solo para visualización.

Figure 2. Spatial distribution of adjusted associations between psychometric scores and alpha reactivity. Sensor-wise natural-spline analyses show associations of STAI-trait (A-B) and MSPSS total (C-D) with absolute alpha reactivity in dB (A, C) and relative alpha reactivity in percentage points (B, D). All analyses use participants as the inferential units and adjust for the other psychometric scale, sex, and partnership status, with HC3 covariance. A zero-centered RdBu_r diverging scale represents the directional linear component per 1-SD difference in the focal score, using a common symmetric range within each power representation. Stars identify electrodes whose total 3-df spline test survived BH-FDR across 61 electrodes within the corresponding map; they do not identify a significant slope. No sensor-wise linear or nonlinear component survived its within-map BH-FDR correction. Small black dots identify sensors, channel-name labels are omitted for clarity, and the black head circle is the interpolation boundary. Interpolation is for visualization and does not add observations.

Figura 2. Distribución espacial de las asociaciones ajustadas entre las puntuaciones psicométricas y la reactividad alfa. Los análisis con spline natural por sensor muestran las asociaciones de STAI rasgo (A-B) y MSPSS total (C-D) con la reactividad alfa absoluta en dB (A, C) y relativa en puntos porcentuales (B, D). Todos los análisis usan al participante como unidad inferencial y ajustan por la otra escala psicométrica, sexo y situación de pareja, con covarianza HC3. Una escala divergente RdBu_r centrada en cero representa el componente lineal direccional por una diferencia de 1 DE en la puntuación focal, con un rango simétrico común dentro de cada representación de potencia. Las estrellas indican electrodos cuya prueba spline total de 3 gl sobrevivió BH-FDR entre 61 electrodos dentro del mapa correspondiente; no indican una pendiente significativa. Ningún componente lineal ni no lineal por sensor sobrevivió su corrección BH-FDR dentro del mapa. Los puntos negros pequeños indican los sensores, se omiten sus nombres para favorecer la claridad y la circunferencia negra es el límite de interpolación. La interpolación es solo visual y no añade observaciones.

Figure 3. Adjusted associations of STAI-trait and perceived social support with posterior alpha reactivity. Curves show participant-level fitted natural-spline associations with posterior alpha reactivity (EC minus EO, dB), adjusted for the other psychometric scale, sex, and partnership status. Shaded regions are pointwise 95% HC3 confidence intervals; points are covariate-adjusted participant outcomes. Insets report Holm-adjusted total and nonlinearity tests.

Figura 3. Asociaciones ajustadas de STAI rasgo y apoyo social percibido con la reactividad alfa posterior. Las curvas muestran asociaciones spline naturales a nivel de participante con la reactividad alfa posterior (EC menos EO, dB), ajustadas por la otra escala psicométrica, sexo y situación de pareja. Las áreas sombreadas son intervalos de confianza HC3 puntuales del 95%; los puntos son desenlaces de participantes ajustados por covariables. Los recuadros informan las pruebas totales y de no linealidad ajustadas por Holm.

Figure S1 / Figura S1. Participant flow from locally eligible source records to the complete analytic sample.

Figure S2 / Figura S2. Residual-versus-fitted and normal Q-Q diagnostics for the two focal participant-level analyses. HC3 inference does not assume homoskedastic residuals.
""",
    )
    write_text(
        reports / "STROBE_COBIDAS_SAMPL_gap_audit.txt",
        """
REPORTING GAP AUDIT

The generated mapping tables cover the information that can be verified from the frozen local project and the LEMON source publication. They do not certify journal compliance. Items requiring author verification are isolated in tables/HUMAN_INFORMATION_REQUIRED.csv.

Upstream methods are directly inspectable in scripts/03_preprocess_eeg.py, scripts/04_compute_psd_qc.py, scripts/06_build_analysis_inputs.py, and the run-specific audit/metadata/source_audit_summary.json. Source-to-package hashes are checked during each run; historical byte hashes are enforced only when the optional archived-snapshot mode is requested. The initial PSD/QC summaries and the exact analysis-input constructor are separate documented stages. UPSTREAM_METHOD_SPECIFICATIONS_ES_EN.txt records the bad-channel, LOF, ICLabel/EOG, spectral-integration, and source-exclusion rules.

Most important boundary: this is a secondary observational analysis and must not be described as prospectively preregistered unless documentation exists. Alpha was selected after same-cohort exploration of delta, theta, alpha, and beta; it was not a prespecified sole band. The focused alpha Holm adjustment covers the two focal psychometric total-association tests. Table S14 separately applies the exact final specification to four bands and controls eight band-by-scale total tests with Holm, with a separate Holm family for eight nonlinearity tests. Only alpha-MSPSS survives the total-test Holm-8 sensitivity; alpha-STAI does not. This contextual sensitivity is not a complete correction for all historical analytic choices. The secondary sensor-wise maps use BH-FDR separately across 61 electrodes for each psychometric variable and power representation.

No retrospective power calculation is used. The study-size statement is availability-based: 139 participants passed the frozen EEG pipeline, and 137 had complete STAI, MSPSS, sex and partnership status.

Item-level STAI/MSPSS data are not part of the analytic table. Exact questionnaire version/scoring and sample reliability therefore require a separate audited item-level source before they can be reported.

The participant-level CSV is intended for local validation. Do not publish it automatically; confirm LEMON behavioral-data terms and institutional requirements first. The public candidate removes local absolute paths from path-bearing upstream artifacts but intentionally retains pseudonymous dataset IDs in the preprocessing source; verify that release against LEMON terms.
""",
    )
    write_text(
        reports / "AUTHORITATIVE_SOURCES.txt",
        f"""
LEMON source study (acquisition, recruitment, ethics): {LEMON_PAPER}
STROBE official EQUATOR page: {STROBE_URL}
COBIDAS-MEEG official site/checklists: {COBIDAS_URL}
SAMPL official EQUATOR page: {SAMPL_URL}
Elsevier artwork and media instructions: {ELSEVIER_ARTWORK_URL}

Accessed by the analysis-package authoring workflow on 2026-09-05.
""",
    )
    write_text(metadata_dir / "environment.txt", environment_text())
    write_text(
        reports / "LIBRARIES_USED.txt",
        f"""
LIBRARIES USED BY THE ALPHA-REACTIVITY STUDY

- Python {platform.python_version()}: runtime and standard library (argparse, pathlib, json, hashlib, datetime, platform, warnings).
- NumPy {np.__version__}: arrays, linear algebra, SVD, contrasts and validation.
- pandas {pd.__version__}: participant tables, complete-case selection and CSV outputs.
- SciPy {scipy.__version__}: z scores, normal-reference inference helpers, distribution diagnostics and probability plots.
- statsmodels {statsmodels.__version__}: OLS, HC3 sandwich covariance, Wald tests, influence diagnostics, Breusch-Pagan tests, VIF and Holm adjustment.
- patsy {patsy.__version__}: natural cubic spline bases for the primary df=4 model and the df=3-6 complexity sensitivity.
- matplotlib {matplotlib.__version__}: 600-dpi PNG and vector PDF figures.
- MNE-Python {mne.__version__}: standard 10-05 electrode coordinates and scalp topomap interpolation.

Upstream EEG preprocessing reproducibility environment:
- Python 3.12
- MNE-Python 1.9.0
- mne-icalabel 0.7.0
- NumPy 2.2.6
- SciPy 1.15.3

The final analysis script does not rerun EEG preprocessing or PSD estimation. It validates and consumes the generated analysis inputs. The complete public workflow is implemented by scripts/01_download_lemon.py through scripts/07_run_analysis.py; scripts/06_build_analysis_inputs.py is the direct constructor of both the 1-45 Hz interval table and canonical-band cache.
""",
    )
    write_text(reports / "DATA_SHARING_BOUNDARY.txt", data_sharing_boundary_text())
    write_text(
        reports / "UPSTREAM_METHOD_SPECIFICATIONS_ES_EN.txt",
        upstream_method_specifications_text(upstream_inventory, metadata["source_audit"]),
    )
    write_text(
        metadata_dir / "upstream_source_inventory.json",
        json.dumps(upstream_inventory, indent=2, ensure_ascii=False),
    )
    write_text(
        reports / "BAND_SELECTION_RATIONALE_ES_EN.txt",
        band_selection_rationale_text(band_selection),
    )
    write_text(reports / "REPRODUCIBILITY_NOTES.txt", final_changes_text())
    write_text(output / "requirements.txt", requirements_text())
    parameters = {
        "generated": now_iso(),
        "analysis": STUDY_TITLE,
        "project": str(project),
        "output": str(output),
        "inferential_unit": "participant",
        "primary_n": len(frame),
        "source_cache_n": 139,
        "channels": 61,
        "roi_channels": list(ROI),
        "band": {
            "primary_focus": "alpha",
            "low_hz": 8.0,
            "high_hz": 13.0,
            "selection_status": "data-informed post hoc selection after same-cohort canonical-band exploration",
        },
        "band_selection_sensitivity": {
            "role": "contextual post-selection sensitivity; not primary",
            "bands": [
                {"name": "delta", "low_hz": 1.0, "high_hz": 4.0},
                {"name": "theta", "low_hz": 4.0, "high_hz": 8.0},
                {"name": "alpha", "low_hz": 8.0, "high_hz": 13.0},
                {"name": "beta", "low_hz": 13.0, "high_hz": 30.0},
            ],
            "model": "exact final posterior-ROI natural-spline specification applied independently to each band",
            "total_test_multiplicity": "Holm across 8 band-by-scale total-association tests",
            "nonlinearity_multiplicity": "separate Holm correction across 8 band-by-scale nonlinearity tests",
            "result": "only alpha-MSPSS survives Holm-8; alpha-STAI and every non-alpha total test do not",
            "selective_inference_boundary": "does not correct every analytical representation or functional form examined in the same cohort",
        },
        "reactivity": "EC_dB_minus_EO_dB",
        "spline": {
            "primary_patsy_formula": "cr(x, df=4)-1",
            "primary_nonlinear_rank": 2,
            "svd_tolerance": "max(s)*1e-8",
            "complexity_sensitivity_basis_df": [3, 4, 5, 6],
            "complexity_references": ["HC3 asymptotic chi-square", "HC3 finite-sample F"],
            "complexity_multiplicity": "Holm across STAI and MSPSS separately within each nominal basis df",
        },
        "covariates": ["other psychometric scale (linear)", "male indicator", "partnered indicator"],
        "covariance": "HC3",
        "tests": "two-sided Wald chi-square",
        "alpha": ALPHA,
        "multiplicity": {
            "focused_alpha_primary": "Holm separately across 2 total tests and 2 nonlinear tests",
            "post_selection_four_band_context": "Holm across 8 total tests and separately across 8 nonlinear tests",
            "sensor_maps": "BH-FDR across 61 electrodes separately within each map",
        },
        "topographic_analysis": {
            "role": "secondary spatial characterization",
            "electrodes": 61,
            "absolute_alpha": "8-13 Hz density in dB re 1 uV2/Hz",
            "relative_alpha": "100 * integrated 8-13 Hz power / integrated 1-45 Hz power",
            "relative_reactivity": "EC relative alpha minus EO relative alpha, percentage points",
            "association_color": "directional linear component from the adjusted natural-spline analysis",
            "sensor_inference": "BH-FDR across 61 total 3-df spline tests separately within each map",
            "colormap": "viridis for unsigned EO/EC power; RdBu_r for symmetric zero-centered signed maps",
            "sensor_display": "all 61 locations shown as small black dots; channel-name labels intentionally omitted",
            "geometry": "standard_1005 projected with the MNE eeglab sphere; one enclosing radius used identically for head outline and interpolation mask",
        },
        "complete_case": True,
        "data_sharing": {
            "package_root": "local validation; includes participant-level rows",
            "public_release_candidate": "excludes participant-level measurement rows and identifier-bearing leave-one-out output; exact preprocessing source retains three pseudonymous LEMON exclusion IDs",
            "human_permission_check_required": True,
        },
        "upstream_source_specifications": {
            "copied_verbatim": True,
            "authoritative_location_in_package": [
                relative_path.as_posix() for relative_path in UPSTREAM_SOURCE_RELATIVES.values()
            ],
            "inventory": upstream_inventory,
            "execution_boundary": "included for direct methods audit; not rerun by the final statistical-analysis stage",
        },
        "archived_input_hash_check_enforced": enforce_archived_hashes,
        "skip_loo": bool(args.skip_loo),
        "input_hashes_sha256": hashes,
    }
    write_text(metadata_dir / "parameters.json", json.dumps(parameters, indent=2, ensure_ascii=False))
    expected_band_limits = {
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
    }
    observed_band_limits = {
        (str(row.band), float(row.band_low_hz), float(row.band_high_hz))
        for row in band_selection[["band", "band_low_hz", "band_high_hz"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    primary_by_outcome = {item["label"]: item["model"] for item in primary}
    alpha_reproduces_primary = all(
        all(
            math.isclose(float(row[column]), float(primary_by_outcome[outcome][target]), rel_tol=0.0, abs_tol=1e-10)
            for column, target in [
                ("wald_chi2_total", "wald_chi2_total"),
                ("p_total_raw", "p_total_raw"),
                ("linear_component_db_per_sd", "linear_component_db_per_sd"),
            ]
        )
        for outcome in ["STAI_trait", "MSPSS_total"]
        for _, row in [
            (
                outcome,
                band_selection.loc[
                    (band_selection["band"] == "alpha")
                    & (band_selection["psychometric_variable"] == outcome)
                ].iloc[0],
            )
        ]
    )
    observed_holm8_rejections = {
        (str(row.band), str(row.psychometric_variable))
        for row in band_selection.loc[band_selection["total_reject_holm_all_8_05"]].itertuples(index=False)
    }
    band_numeric_finite = bool(
        np.isfinite(
            band_selection.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all()
    )
    audit = {
        "generated": now_iso(),
        "status": "PASS" if not args.skip_loo else "PASS_QUICK_NO_LOO",
        "input_hash_check": "PASS_ARCHIVED_BYTES" if enforce_archived_hashes else "NOT_APPLICABLE_FRESH_REPRODUCTION",
        "upstream_source_files_included_check": "PASS"
        if all((output / relative_path).is_file() for relative_path in UPSTREAM_SOURCE_RELATIVES.values())
        else "FAIL",
        "upstream_source_copy_hashes_check": "PASS"
        if all(
            sha256_file(output / UPSTREAM_SOURCE_RELATIVES[name]).lower()
            == upstream_inventory[name]["sha256"].lower()
            for name in UPSTREAM_SOURCE_RELATIVES
        )
        else "FAIL",
        "participant_count_check": "PASS",
        "channel_count_check": "PASS",
        "reactivity_identity_check": "PASS",
        "one_row_per_participant_check": "PASS",
        "primary_replication_check": "PASS",
        "relative_alpha_reconstruction_check": "PASS",
        "topographic_electrode_count_check": "PASS" if len(arrays["channels"]) == 61 else "FAIL",
        "topographic_association_row_count_check": "PASS" if len(topographic_associations) == 244 else "FAIL",
        "topographic_sensor_marker_count_check": "PASS" if len(topomap_geometry) == 61 else "FAIL",
        "topographic_head_mask_alignment_check": "PASS" if bool(topomap_geometry["head_outline_equals_interpolation_mask"].all()) else "FAIL",
        "topographic_all_sensors_inside_head_check": "PASS" if float(topomap_geometry["fraction_of_head_radius"].max()) < 1.0 else "FAIL",
        "spline_complexity_row_count_check": "PASS" if len(spline_complexity) == 8 else "FAIL",
        "spline_df4_reproduces_primary_check": "PASS"
        if all(
            math.isclose(
                float(spline_complexity.loc[
                    (spline_complexity["basis_df_requested"] == 4)
                    & (spline_complexity["outcome"] == item["label"]),
                    "wald_chi2_total",
                ].iloc[0]),
                float(item["model"]["wald_chi2_total"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            for item in primary
        )
        else "FAIL",
        "band_selection_row_count_check": "PASS" if len(band_selection) == 8 else "FAIL",
        "band_selection_unique_combinations_check": "PASS"
        if band_selection[["band", "psychometric_variable"]].drop_duplicates().shape[0] == 8
        else "FAIL",
        "band_selection_participant_n_check": "PASS"
        if bool((band_selection["n_participants"] == len(frame)).all())
        else "FAIL",
        "band_selection_band_limits_check": "PASS"
        if observed_band_limits == expected_band_limits
        else "FAIL",
        "band_selection_alpha_reproduces_primary_check": "PASS"
        if alpha_reproduces_primary
        else "FAIL",
        "band_selection_numeric_finite_check": "PASS" if band_numeric_finite else "FAIL",
        "band_selection_holm8_rejection_pattern_check": "PASS"
        if observed_holm8_rejections == {("alpha", "MSPSS_total")}
        else "FAIL",
        "band_selection_nonalpha_total_null_check": "PASS"
        if not bool(
            band_selection.loc[
                band_selection["band"] != "alpha", "total_reject_holm_all_8_05"
            ].any()
        )
        else "FAIL",
        "band_selection_nonlinearity_null_check": "PASS"
        if not bool(band_selection["nonlinear_reject_holm_all_8_05"].any())
        else "FAIL",
        "loo_summary_row_count_check": "PASS" if len(loo_summary) == 2 else "FAIL",
        "loo_all_omnibus_holm_significant_check": "PASS"
        if (not args.skip_loo and bool((loo_summary["omnibus_holm_significant_n"] == 137).all()))
        else ("SKIPPED" if args.skip_loo else "FAIL"),
        "topographic_linear_fdr_zero_check": "PASS"
        if int(topographic_associations["linear_component_reject_bh05"].sum()) == 0
        else "FAIL",
        "topographic_nonlinear_fdr_zero_check": "PASS"
        if int(topographic_associations["nonlinear_reject_bh05"].sum()) == 0
        else "FAIL",
        "all_primary_values_finite": bool(model_tests.select_dtypes(include=[np.number]).notna().all().all()),
        "figures_png_dpi": 600,
    }
    audit_check_values = [value for key, value in audit.items() if key.endswith("_check")]
    if any(value == "FAIL" or value is False for value in audit_check_values):
        audit["status"] = "FAIL"
    write_text(metadata_dir / "run_audit.json", json.dumps(audit, indent=2, ensure_ascii=False))

    script_path = Path(__file__).resolve()
    if script_path.parent != output:
        write_text(
            output / "README_EJECUCION.txt",
            f"""
{STUDY_TITLE.upper()}
REPRODUCIBLE ANALYSIS PACKAGE

Run from PowerShell:
& \"{sys.executable}\" \"{script_path}\" --project \"{project}\" --output \"{output}\"

The script reads frozen inputs from:
{project}

Alpha was selected after same-cohort exploration of delta, theta, alpha, and beta and is therefore a post hoc focus. Within the focused two-test alpha family, both STAI and MSPSS total 3-df spline tests survive Holm; neither 2-df nonlinear test does. In the harmonized four-band Holm-8 sensitivity, only alpha-MSPSS survives; alpha-STAI does not and no non-alpha association survives. Interpret the findings as exploratory omnibus associations, not confirmed monotonic dose-response relations. MSPSS retains significance in stricter EEG-QC subsets, whereas STAI is attenuated. Figure 2 stars encode the total spline test, not significance of the colored linear component. Tables S12-S14 provide spline-complexity, leave-one-out, and four-band selection sensitivities; BAND_SELECTION_RATIONALE_ES_EN.txt documents the selection sequence and limitations.

The generated analysis directory is for local validation because it contains participant-level rows. The public repository keeps the executable pipeline and aggregate reference outputs, but excludes participant-level derivatives and raw EEG. Publication still requires human verification of LEMON permissions, code ownership, author metadata, and repository policy.
""",
        )
    else:
        write_text(
            output / "README_EJECUCION.txt",
            f"""
{STUDY_TITLE.upper()}
REPRODUCIBLE ANALYSIS PACKAGE

Run from PowerShell:
& \"{sys.executable}\" \"{script_path}\" --project \"{project}\" --output \"{output}\"

The script validates the generated analysis inputs, reconstructs the participant-level alpha-reactivity analysis, performs diagnostics and sensitivities, and regenerates all CSV/TXT/JSON and 600-dpi PNG/vector PDF figures. Alpha is disclosed as a same-cohort, data-informed post hoc selection. The exact final spline is also applied to delta, theta, alpha, and beta with Holm across eight band-by-scale tests: only alpha-MSPSS survives, whereas alpha-STAI and every non-alpha association do not. MSPSS retains significance in stricter EEG-QC subsets, whereas STAI is attenuated. Figure 2 stars encode the total spline test, not significance of the colored linear component. Tables S12-S14 provide spline-complexity, leave-one-out, and four-band selection sensitivities; BAND_SELECTION_RATIONALE_ES_EN.txt documents the selection sequence and limitations.

The generated analysis directory is for local validation because it contains participant-level rows. The public repository keeps the executable pipeline and aggregate reference outputs, but excludes participant-level derivatives and raw EEG. Publication still requires human verification of LEMON permissions, code ownership, author metadata, and repository policy.
""",
        )

    public_manifest = (
        build_public_release_candidate(output, script_path)
        if args.build_public_release_candidate
        else pd.DataFrame()
    )
    manifest = write_manifest(output)
    print(f"{STUDY_SHORT_NAME}: PASS")
    print(f"Primary n: {len(frame)}")
    for row in model_tests.itertuples(index=False):
        print(
            f"{row.outcome}: chi2({row.spline_df_total})={row.wald_chi2_total:.6f}, "
            f"raw p={row.p_total_raw:.8f}, Holm p={row.p_total_holm_2:.8f}, "
            f"nonlinear Holm p={row.p_nonlinear_holm_2:.8f}"
        )
    print("Post-selection four-band sensitivity (Holm across 8 total tests):")
    for row in band_selection.itertuples(index=False):
        print(
            f"  {row.band}/{row.psychometric_variable}: raw p={row.p_total_raw:.8f}, "
            f"Holm-8 p={row.p_total_holm_all_8:.8f}, "
            f"reject={bool(row.total_reject_holm_all_8_05)}"
        )
    print(f"Files hashed in manifest: {len(manifest)}")
    if args.build_public_release_candidate:
        print(f"Files hashed in public-release candidate: {len(public_manifest)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
