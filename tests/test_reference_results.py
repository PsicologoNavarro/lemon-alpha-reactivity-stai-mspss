from __future__ import annotations

import json
import hashlib
import math

import pandas as pd

from conftest import REPOSITORY


def test_primary_reference_matches_checked_in_table() -> None:
    expected = json.loads((REPOSITORY / "reference" / "expected_results.json").read_text(encoding="utf-8"))
    table = pd.read_csv(REPOSITORY / "results" / "tables" / "Table_2_primary_model_tests.csv")
    assert table.shape[0] == 2
    for row in table.itertuples(index=False):
        target = expected["primary_alpha_models"][row.outcome]
        assert int(row.n_participants) == target["n"]
        assert math.isclose(row.wald_chi2_total, target["wald_chi2_total_df3"], abs_tol=1e-8)
        assert math.isclose(row.p_total_raw, target["p_total_raw"], abs_tol=1e-10)
        assert math.isclose(row.p_total_holm_2, target["p_total_holm_2"], abs_tol=1e-10)
        assert math.isclose(row.linear_component_db_per_sd, target["linear_component_db_per_sd"], abs_tol=1e-9)


def test_four_band_selection_pattern() -> None:
    table = pd.read_csv(REPOSITORY / "results" / "tables" / "Table_S14_four_band_selection_sensitivity.csv")
    assert table[["band", "psychometric_variable"]].drop_duplicates().shape[0] == 8
    rejected = table.loc[table["total_reject_holm_all_8_05"].astype(bool), ["band", "psychometric_variable"]]
    assert rejected.to_records(index=False).tolist() == [("alpha", "MSPSS_total")]
    assert not table["nonlinear_reject_holm_all_8_05"].astype(bool).any()


def test_aggregate_tables_exclude_participant_rows() -> None:
    forbidden = {"participant_id", "subject_id", "omitted_participant", "relationship_status_raw"}
    for path in (REPOSITORY / "results" / "tables").glob("*.csv"):
        columns = set(pd.read_csv(path, nrows=0).columns)
        assert not (forbidden & columns), (path.name, sorted(forbidden & columns))

    correlation = pd.read_csv(REPOSITORY / "results" / "tables" / "Table_S8_zero_order_correlations.csv")
    assert correlation.shape == (3, 4)
    assert set(correlation["variable"]) == {
        "stai_trait_total", "mspss_total", "alpha_roi_reactivity_ec_minus_eo_db"
    }


def test_reference_run_audit_and_method_source_hashes() -> None:
    audit = json.loads(
        (REPOSITORY / "results" / "metadata" / "reference_analysis_run_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "PASS"
    assert audit["figures_png_dpi"] == 600
    assert all(
        value in {"PASS", "NOT_APPLICABLE_FRESH_REPRODUCTION"}
        for key, value in audit.items()
        if key.endswith("_check")
    )

    inventory = json.loads(
        (REPOSITORY / "results" / "metadata" / "reference_upstream_source_inventory.json").read_text(encoding="utf-8")
    )
    for name in ("preprocessing_script", "psd_script", "canonical_cache_script"):
        row = inventory[name]
        observed = hashlib.sha256((REPOSITORY / row["public_relative_path"]).read_bytes()).hexdigest()
        assert observed == row["public_sha256"]
