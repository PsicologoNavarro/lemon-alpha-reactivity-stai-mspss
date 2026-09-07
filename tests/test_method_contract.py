from __future__ import annotations

import json

from conftest import REPOSITORY, load_script


preprocess = load_script("preprocess_contract", "03_preprocess_eeg.py")
psd = load_script("psd_contract", "04_compute_psd_qc.py")
download = load_script("download_contract", "01_download_lemon.py")
source_audit = load_script("source_audit_contract", "02_audit_sources.py")
qc_freeze = load_script("qc_freeze_contract", "05_freeze_qc.py")


def test_preprocessing_contract() -> None:
    assert len(preprocess.SCALP_CHANNELS) == 61
    assert len(preprocess.EXPECTED_CHANNELS) == 62
    assert preprocess.ICLABEL_THRESHOLD == 0.80
    assert preprocess.BASE_ICA_SEED == 20260902
    assert len(preprocess.SPECIAL_MARKER_MAP) == 1
    assert len(preprocess.HARD_EXCLUSIONS) == 3
    assert preprocess.as_bool(preprocess.pd.Series(["False", "True", "0", "1"])).tolist() == [False, True, False, True]
    assert qc_freeze.as_bool(qc_freeze.pd.Series(["False", "True", "0", "1"])).tolist() == [False, True, False, True]


def test_window_contract() -> None:
    assert psd.WINDOW_SECONDS == 4.0
    assert psd.STEP_SECONDS == 2.0
    assert psd.BLOCKS_PER_CONDITION == 8
    assert int((psd.BLOCK_SECONDS - psd.WINDOW_SECONDS) / psd.STEP_SECONDS) + 1 == 29
    assert psd.MAX_ABS_UV == 250.0
    assert psd.MAX_PTP_UV == 500.0
    assert psd.MIN_ROBUST_STD_UV == 0.1


def test_metadata_reference_manifest_is_complete() -> None:
    assert len(download.METADATA_FILES) == 9
    assert set(download.REFERENCE_METADATA_SHA256) == {
        item.rsplit("/", 1)[-1] for item in download.METADATA_FILES
    }
    assert all(len(value) == 64 for value in download.REFERENCE_METADATA_SHA256.values())


def test_download_batch_resume_does_not_recount_complete_subject(tmp_path) -> None:
    paths = download.eeg_target_paths(tmp_path, "sub-test")
    assert download.subject_requires_transfer(paths)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    assert not download.subject_requires_transfer(paths)
    next_paths = download.eeg_target_paths(tmp_path, "sub-next")
    assert download.subject_requires_transfer(next_paths)


def test_preprocessing_batch_completion_requires_every_contract_output(tmp_path) -> None:
    paths = preprocess.preprocessing_output_paths(tmp_path, "sub-test")
    assert not preprocess.subject_preprocessing_complete(tmp_path, "sub-test")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"complete")
    assert preprocess.subject_preprocessing_complete(tmp_path, "sub-test")
    paths["component_csv"].unlink()
    assert not preprocess.subject_preprocessing_complete(tmp_path, "sub-test")


def test_source_flow_contract() -> None:
    payload = json.loads((REPOSITORY / "reference" / "known_source_variants.json").read_text(encoding="utf-8"))
    assert len(payload["recoverable"]) == 3
    assert len(payload["excluded"]) == 3
    assert payload["expected_flow"] == {
        "eligible": 147,
        "complete_brainvision_triplets": 142,
        "direct_source_audit_pass": 136,
        "recoverable": 3,
        "excluded": 3,
        "processable_eeg": 139,
        "complete_case_primary_models": 137,
    }
    assert set(source_audit.RECOVERABLE_SOURCE_VARIANTS) == set(payload["recoverable"])
    assert set(source_audit.EXPECTED_HARD_EXCLUSIONS) == set(payload["excluded"])
    assert sum("marker" in value for value in source_audit.RECOVERABLE_SOURCE_VARIANTS.values()) == 1
    assert sorted(source_audit.EXPECTED_HARD_EXCLUSIONS.values()) == [
        "overlapping_condition_blocks",
        "truncated_binary_payload",
        "truncated_binary_payload",
    ]
