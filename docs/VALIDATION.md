# Validation record

This repository candidate was audited on 2026-09-07 under Python 3.12 on Windows before handoff.

## Checks executed during repository construction

- The nine official LEMON metadata files already present locally matched every embedded SHA-256 reference. Cohort reconstruction found 227 IDs and reproduced 147 eligible records.
- The exact spectral-input constructor was rerun from clean FIF files for two participants. It generated 10,736 rows (`2 x 61 x 88`), with no nonfinite measurements. Maximum discrepancies against the archived canonical cache were `3.55e-15` dB for EO, `3.55e-15` dB for EC, and `6.22e-15` dB for reactivity. All compared full-frequency columns agreed exactly; the serialized EC-minus-EO identity error was at most `1.01e-10` dB.
- The complete statistical stage was rerun from the 139-participant archived spectral/QC inputs, including diagnostics, spline-complexity sensitivity, leave-one-participant-out analysis, four-band sensitivity, topographies, aggregate tables, and 600-dpi/vector figures. Every run-audit check passed and the primary analysis reproduced `n=137`.
- The primary omnibus results reproduced STAI `chi-square(3)=10.471698, p=.01495412` and MSPSS `chi-square(3)=14.898489, p=.00190548`. In the four-band Holm-8 sensitivity, only alpha-MSPSS was retained (`adjusted p=.01524387`).
- Eighteen automated tests passed. They cover spectral integration, the 88-interval grid, canonical-band additivity, source/cohort contracts, resumable download and preprocessing behavior, aggregate numeric regression, public-data boundaries, reference-source hashes, synchronized citation/Zenodo metadata, and byte identity between active metadata and completed templates.
- The active `CITATION.cff` and matching `.zenodo.json` list seven creators in the current citation order. Six verified ORCID identifiers are included; Carol Alejandra Olmos-Pastoresa is intentionally listed without an ORCID because none was verified. The source submission archive had SHA-256 `07CF6BF28F9B4E5B43C1C389EF30A859CD192E5049A1CF2EB02116AF02C0E5B3`; no manuscript file was copied into this public repository. Automated tests and the release auditor check creator order and consistency.
- The static repository audit found no raw EEG, FIF, NPZ, Parquet, participant-level table, archive, file above 50 MiB, local absolute path, local username, email outside live citation metadata, or secret-like string.
- A manifest-driven temporary checkout with all 88 public files was initialized and staged in Git. The release-mode audit passed with metadata, privacy, tracked-file completeness, and SHA-256 integrity checks enabled.
- Commit `1402ce8066e1152694e80a5c32b92ba8b969de0a` was published as GitHub tag and Release `v1.0.0`. Both the release audit and repository test workflows passed for the tag. Zenodo published version DOI `10.5281/zenodo.22647023` under concept DOI `10.5281/zenodo.22647022`.
- The Zenodo snapshot was downloaded independently after publication. Its 7,729,686-byte ZIP matched Zenodo MD5 `0f97c286b8148b7694917769467435b5`; all 88 archived files were present and all 87 entries in the archived `SHA256SUMS` matched their file contents.

## Creator-metadata correction

On 2026-09-08, the active repository metadata were corrected to the seven-creator citation order and the public Zenodo record metadata were targeted for the same correction. This metadata-only correction does not change DOI `10.5281/zenodo.22647023`, concept DOI `10.5281/zenodo.22647022`, or the immutable GitHub/Zenodo `v1.0.0` archive. Consequently, the archived `v1.0.0` copies of `CITATION.cff` and `.zenodo.json` retain the original five-creator list; the files on the current default branch are authoritative for future releases.

## Deliberate validation boundary

The full 42.66-GiB raw-source audit and approximately 3.77-hour preprocessing stage were not rerun from a new clean download while assembling this public repository. Their executable specifications, expected source dispositions, archived aggregate provenance, dependency lock, and downstream numeric outputs are included. A fully independent replication must execute stages `01` through `07` from the official LEMON distribution as described in `REPRODUCIBILITY.md`.

Release mode was exercised successfully against the staged Git index before publication. Future releases must repeat the metadata, privacy, test, and archive-verification gates.
