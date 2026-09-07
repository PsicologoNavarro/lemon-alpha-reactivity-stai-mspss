from __future__ import annotations

import json

from conftest import REPOSITORY, load_script


verifier = load_script("repository_verifier", "verify_repository.py")


def test_pre_release_repository_audit_passes() -> None:
    result = verifier.audit(REPOSITORY, release=False)
    assert result["status"] == "PASS", result["errors"]


def test_cff_top_level_scalar_parser() -> None:
    content = 'title: "Example software"\nversion: "1.0.0"\nlicense: MIT\n'
    assert verifier.cff_scalar(content, "title") == "Example software"
    assert verifier.cff_scalar(content, "version") == "1.0.0"
    assert verifier.cff_scalar(content, "license") == "MIT"


def test_live_zenodo_metadata_contract() -> None:
    metadata = json.loads((REPOSITORY / ".zenodo.json").read_text(encoding="utf-8"))
    assert metadata["upload_type"] == "software"
    assert metadata["access_right"] == "open"
    assert metadata["license"] == "MIT"
    assert metadata["version"] == "1.0.0"
    assert metadata["related_identifiers"] == [
        {
            "identifier": "10.1038/sdata.2018.308",
            "relation": "isDerivedFrom",
            "scheme": "doi",
        }
    ]


def test_cff_and_zenodo_authors_match_submission_order() -> None:
    citation = (REPOSITORY / "CITATION.cff").read_text(encoding="utf-8")
    metadata = json.loads((REPOSITORY / ".zenodo.json").read_text(encoding="utf-8"))
    authors = verifier.cff_authors(citation)
    expected_names = [
        "Navarro Nolasco, Diego Armando",
        "Beltrán-Parrazal, Luis",
        "Martínez Chacón, Armando Jesús",
        "López-Meraz, María Leonor",
        "Morgado-Valle, Consuelo",
    ]
    assert [f"{item['family-names']}, {item['given-names']}" for item in authors] == expected_names
    assert [item["name"] for item in metadata["creators"]] == expected_names
    assert all(verifier.valid_orcid(item["orcid"]) for item in authors)
    assert all(verifier.valid_orcid(item["orcid"]) for item in metadata["creators"])


def test_completed_templates_match_active_metadata() -> None:
    assert (REPOSITORY / "CITATION.cff.template").read_bytes() == (
        REPOSITORY / "CITATION.cff"
    ).read_bytes()
    assert (REPOSITORY / ".zenodo.json.template").read_bytes() == (
        REPOSITORY / ".zenodo.json"
    ).read_bytes()
