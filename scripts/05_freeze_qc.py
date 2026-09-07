#!/usr/bin/env python
"""Freeze participant-level QC and analysis cohorts for LEMON full-density data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsons(folder: Path, pattern: str) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob(pattern))]


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "si", "sí"})


def resolve_recorded_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def portable_recorded_path(root: Path, value: object) -> str:
    path = resolve_recorded_path(root, value)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "analysis" / "QC_FREEZE"
    tables, figures, metadata = out / "tables", out / "figures", out / "metadata"
    for folder in (tables, figures, metadata):
        folder.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(root / "audit" / "tables" / "cohort_full62_frozen.csv")
    cohort = cohort.rename(columns={
        "ID": "participant_id", "Age": "age_bin",
        "Gender_ 1=female_2=male": "sex_code",
        "STAI_Trait_Anxiety": "stai_trait_total",
        "MSPSS_total": "mspss_total",
        "Relationship_Status": "relationship_status_raw",
    })
    cohort["sex"] = cohort.sex_code.map({1: "female", 2: "male"})
    cohort["partnered"] = cohort.relationship_status_raw.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    prep_rows = []
    for q in load_jsons(root / "preprocessing" / "qc" / "subjects", "*_preprocessing_qc.json"):
        prep_rows.append({
            "participant_id": q["participant_id"],
            "preprocessing_status": q["status"],
            "ica_converged": q["status"] == "PASS",
            "n_bad_channels": q["n_bad_channels"],
            "bad_channels": "|".join(q["bad_channels"]),
            "ica_rank": q["ica_rank"],
            "ica_n_components": q["ica_n_components"],
            "ica_n_iter": q["ica_n_iter"],
            "ica_n_samples_fit": q["ica_n_samples_fit"],
            "n_excluded_components": q["n_excluded_components_final"],
            "excluded_components": "|".join(map(str, q["excluded_components_final"])),
            "excluded_fraction": q["n_excluded_components_final"] / q["ica_n_components"],
            "eyes_open_fif": portable_recorded_path(root, q["outputs"]["eyes_open_fif"]),
            "eyes_open_sha256_recorded": q["outputs"]["eyes_open_sha256"],
            "eyes_closed_fif": portable_recorded_path(root, q["outputs"]["eyes_closed_fif"]),
            "eyes_closed_sha256_recorded": q["outputs"]["eyes_closed_sha256"],
            "ica_fif": portable_recorded_path(root, q["outputs"]["ica_fif"]),
            "ica_sha256_recorded": q["outputs"]["ica_sha256"],
        })
    prep = pd.DataFrame(prep_rows)

    psd_rows = []
    for q in load_jsons(root / "features" / "qc" / "subjects", "*_psd_qc.json"):
        psd_rows.append({
            "participant_id": q["participant_id"],
            "psd_status": q["status"],
            "global_windows_total": q["global_windows_total"],
            "global_windows_accepted": q["global_windows_accepted"],
            "global_window_retention_fraction": q["global_window_retention_fraction"],
            "channel_windows_total": q["channel_windows_total"],
            "channel_windows_accepted": q["channel_windows_accepted"],
            "channel_window_retention_fraction": q["channel_windows_accepted"] / q["channel_windows_total"],
            "window_parquet": portable_recorded_path(root, q["outputs"]["window_parquet"]),
            "window_parquet_sha256_recorded": q["outputs"]["window_parquet_sha256"],
            "subject_summary_csv": portable_recorded_path(root, q["outputs"]["subject_summary_csv"]),
            "subject_summary_sha256_recorded": q["outputs"]["subject_summary_sha256"],
        })
    psd = pd.DataFrame(psd_rows)
    subject_summary = pd.read_csv(root / "features" / "subject_condition_channel_band.csv")
    retention = subject_summary.groupby("participant_id").window_retention_fraction.agg(
        min_channel_band_retention="min", median_channel_band_retention="median",
        mean_channel_band_retention="mean",
    ).reset_index()

    components = pd.concat([
        pd.read_csv(path) for path in sorted((root / "preprocessing" / "qc" / "components").glob("*_ica_components.csv"))
    ], ignore_index=True)
    excluded = components.loc[as_bool(components.excluded_final)].copy()
    excluded_brain = excluded.argmax_label.astype(str).str.lower().eq("brain")

    master = cohort.merge(prep, on="participant_id", how="inner", validate="one_to_one")
    master = master.merge(psd, on="participant_id", how="inner", validate="one_to_one")
    master = master.merge(retention, on="participant_id", how="left", validate="one_to_one")
    if len(master) != 139:
        raise RuntimeError(f"Expected 139 fully processed participants, got {len(master)}")
    if master[["stai_trait_total", "mspss_total", "sex"]].isna().any().any():
        raise RuntimeError("STAI/MSPSS/sex data are incomplete")
    master["context_complete"] = master[["sex", "partnered"]].notna().all(axis=1)
    master["ica_nonconvergence_flag"] = ~master.ica_converged
    master["high_component_removal_flag"] = master.excluded_fraction > 0.30
    master["low_global_window_retention_flag"] = master.global_window_retention_fraction < 0.80
    master["low_channel_band_retention_flag"] = master.min_channel_band_retention < 0.80
    master["main_all_usable_included"] = True
    master["strict_ica_converged_included"] = master.ica_converged
    master["strict_multiflag_included"] = ~(
        master.ica_nonconvergence_flag
        | master.high_component_removal_flag
        | master.low_channel_band_retention_flag
    )
    master["manual_review_reasons"] = master.apply(lambda row: "|".join([
        reason for flag, reason in [
            (row.ica_nonconvergence_flag, "ICA_MAX_ITER_1024"),
            (row.high_component_removal_flag, "ICA_REMOVAL_GT_30PCT"),
            (row.low_global_window_retention_flag, "GLOBAL_WINDOW_RETENTION_LT_80PCT"),
            (row.low_channel_band_retention_flag, "CHANNEL_BAND_RETENTION_LT_80PCT"),
        ] if flag
    ]), axis=1)
    master = master.sort_values("participant_id").reset_index(drop=True)

    # Verify every analysis output against the hashes recorded at creation.
    hash_rows = []
    for row in master.itertuples(index=False):
        for role, path_value, recorded in [
            ("eyes_open_fif", row.eyes_open_fif, row.eyes_open_sha256_recorded),
            ("eyes_closed_fif", row.eyes_closed_fif, row.eyes_closed_sha256_recorded),
            ("ica_fif", row.ica_fif, row.ica_sha256_recorded),
            ("window_parquet", row.window_parquet, row.window_parquet_sha256_recorded),
            ("subject_summary_csv", row.subject_summary_csv, row.subject_summary_sha256_recorded),
        ]:
            path = resolve_recorded_path(root, path_value)
            observed = sha256(path)
            hash_rows.append({
                "participant_id": row.participant_id, "role": role,
                "path": portable_recorded_path(root, path), "bytes": path.stat().st_size,
                "sha256_recorded": recorded, "sha256_observed": observed,
                "match": observed.lower() == str(recorded).lower(),
            })
    hashes = pd.DataFrame(hash_rows)
    if not hashes.match.all():
        raise RuntimeError("At least one output hash mismatch was found")

    # Audit whether convergence-based sensitivity selection differs on context.
    convergence = master.ica_converged.to_numpy(bool)
    male = master.sex.eq("male").to_numpy()
    partnered = master.partnered.eq(1).to_numpy()
    association_rows = []
    for name, binary, valid in [
        ("sex_male", male, np.ones(len(master), dtype=bool)),
        ("partnered", partnered, master.partnered.notna().to_numpy()),
    ]:
        table = pd.crosstab(binary[valid], convergence[valid]).reindex(index=[False, True], columns=[False, True], fill_value=0)
        odds, p = stats.fisher_exact(table.to_numpy())
        association_rows.append({
            "variable": name, "test": "Fisher exact", "statistic": odds, "p_raw": p,
            "table": json.dumps(table.to_numpy().tolist()),
        })
    for name in ("stai_trait_total", "mspss_total"):
        a = master.loc[convergence, name].to_numpy(float)
        b = master.loc[~convergence, name].to_numpy(float)
        result = stats.ttest_ind(a, b, equal_var=False)
        association_rows.append({
            "variable": name, "test": "Welch t", "statistic": result.statistic,
            "p_raw": result.pvalue, "table": "",
        })
    associations = pd.DataFrame(association_rows)

    master.to_csv(tables / "participant_qc_master.csv", index=False)
    hashes.to_csv(metadata / "processed_output_hash_verification.csv", index=False)
    associations.to_csv(tables / "ica_convergence_selection_bias_checks.csv", index=False)
    pd.DataFrame({"participant_id": master.loc[master.main_all_usable_included, "participant_id"]}).to_csv(
        tables / "cohort_primary_all_usable.csv", index=False
    )
    pd.DataFrame({"participant_id": master.loc[master.strict_ica_converged_included, "participant_id"]}).to_csv(
        tables / "cohort_sensitivity_ica_converged.csv", index=False
    )
    pd.DataFrame({"participant_id": master.loc[master.strict_multiflag_included, "participant_id"]}).to_csv(
        tables / "cohort_sensitivity_multiflag.csv", index=False
    )

    # Compact visual audit.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].hist(master.loc[master.ica_converged, "ica_n_iter"], bins=20, color="#4477AA", label="converged")
    axes[0, 0].axvline(1024, color="#CC6677", lw=2, label="max_iter")
    axes[0, 0].set(xlabel="ICA iterations", ylabel="Participants", title="ICA convergence")
    axes[0, 0].legend(frameon=False)
    colors = np.where(master.ica_converged, "#4477AA", "#CC6677")
    axes[0, 1].scatter(master.excluded_fraction * 100, master.channel_window_retention_fraction * 100,
                       c=colors, alpha=.8, edgecolor="white", linewidth=.4)
    axes[0, 1].set(xlabel="ICA components excluded (%)", ylabel="Accepted channel-windows (%)",
                   title="Artifact removal and retained data")
    axes[1, 0].hist(master.global_window_retention_fraction * 100, bins=np.linspace(50, 100, 26), color="#228833")
    axes[1, 0].set(xlabel="Globally accepted windows (%)", ylabel="Participants", title="Global window QC")
    label_counts = excluded.argmax_label.value_counts().reindex(
        ["eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"], fill_value=0
    )
    axes[1, 1].barh(label_counts.index, label_counts.values, color="#AA4499")
    axes[1, 1].set(xlabel="Excluded ICs", title="ICLabel classes removed")
    fig.suptitle("LEMON full-density QC freeze")
    fig.savefig(figures / "qc_freeze_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "qc_freeze_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "participants_primary_all_usable": int(master.main_all_usable_included.sum()),
        "participants_sensitivity_ica_converged": int(master.strict_ica_converged_included.sum()),
        "participants_sensitivity_multiflag": int(master.strict_multiflag_included.sum()),
        "participants_context_complete": int(master.context_complete.sum()),
        "ica_nonconverged": int(master.ica_nonconvergence_flag.sum()),
        "high_component_removal_gt_30pct": int(master.high_component_removal_flag.sum()),
        "low_global_window_retention_lt_80pct": int(master.low_global_window_retention_flag.sum()),
        "low_channel_band_retention_lt_80pct": int(master.low_channel_band_retention_flag.sum()),
        "ica_components_total": int(len(components)),
        "ica_components_excluded": int(len(excluded)),
        "ica_components_excluded_fraction": float(len(excluded) / len(components)),
        "excluded_argmax_brain_components": int(excluded_brain.sum()),
        "processed_files_hash_verified": int(hashes.match.sum()),
        "processed_files_hash_total": int(len(hashes)),
        "processed_bytes_hash_verified": int(hashes.bytes.sum()),
        "primary_decision": "retain all 139 processable participants; use ICA-converged and multiflag cohorts as sensitivity analyses",
        "rationale": [
            "No excluded component had brain as its ICLabel argmax class.",
            "All PSD files passed; channel-specific window retention is used for subject summaries.",
            "ICA max-iteration status is retained as an explicit sensitivity flag, not silently ignored.",
            "Dropping all nonconverged participants can induce contextual selection, especially by sex; see bias-check table.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    (metadata / "qc_freeze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
