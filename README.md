# Trait anxiety, social support, and posterior alpha reactivity in LEMON

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22647022.svg)](https://doi.org/10.5281/zenodo.22647022)

End-to-end, auditable code for an exploratory secondary analysis of the public MPI Leipzig Mind-Brain-Body (LEMON) resting-state EEG dataset. The workflow reconstructs the eligible cohort, audits the raw BrainVision sources, preprocesses 61-channel eyes-open/eyes-closed EEG, computes PSD and EC-minus-EO reactivity, freezes participant-level QC, and regenerates the statistical tables, reports, and publication figures.

No LEMON data or participant-level measurements are distributed in this repository. Checked-in results are aggregate only.

## Scientific scope

- Inferential unit: participant, never windows or electrodes.
- Flow: 147 eligible, 142 complete BrainVision triplets, 139 processable EEG records, and 137 complete adjusted-model records.
- EEG quantity: eyes-closed minus eyes-open PSD reactivity.
- Spectral coverage: 1–45 Hz in 88 contiguous 0.5-Hz intervals, plus delta 1–4, theta 4–8, alpha 8–13, and beta 13–30 Hz.
- Manuscript focus: posterior alpha reactivity, explicitly selected after same-cohort canonical-band exploration.
- Statistical claim: omnibus spline associations, not causal effects, biomarkers, or confirmed monotonic dose-response relations.

The focused alpha models detect omnibus associations for STAI and MSPSS after Holm correction across the two focal tests. In the stricter eight-test, four-band sensitivity, only alpha–MSPSS survives. Directional linear-component confidence intervals cross zero. See [`docs/METHODS_PROVENANCE.md`](docs/METHODS_PROVENANCE.md) and the aggregate reference tables under [`results/tables`](results/tables).

## Repository layout

```text
scripts/       download through final analysis, plus PowerShell orchestration
requirements/ two frozen Python 3.12 environments
tests/         synthetic method-contract and public-release tests
reference/     expected counts, estimates, hashes, and source variants
results/       aggregate reference tables, reports, and figures only
docs/          data boundary, methods, reproduction, and release instructions
```

## Verify this checkout

The static verifier uses only Python's standard library:

```powershell
py -3.12 ".\scripts\verify_repository.py" --repository "."
```

The construction-time checks and their explicit boundary are recorded in [`docs/VALIDATION.md`](docs/VALIDATION.md). To install the exact environments:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage setup
```

Then run tests:

```powershell
& ".\.venv-analysis\Scripts\python.exe" -m pytest -q
```

## Reproduce from LEMON

Read [`docs/DATA_ACCESS_AND_PRIVACY.md`](docs/DATA_ACCESS_AND_PRIVACY.md) first. The complete download exceeds 40 GiB and requires an explicit acknowledgement flag:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage all -AcceptDataResponsibility -NoInstall
```

For a staged/resumable execution, use `download`, `audit`, `preprocess`, `features`, and `analysis` in that order. Full commands, storage expectations, validation policy, and output paths are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The long participant-level table and all EEG derivatives are generated under `workdir/` and blocked by `.gitignore` and the repository auditor. Do not use `git add -f` on them.

## GitHub and Zenodo

The live [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json), together with their completed `.template` copies, contain the current ordered creator list and affiliations. Verified ORCID identifiers are provided for six creators; Carol Alejandra Olmos-Pastoresa is listed without an ORCID. Automated tests require each template to remain byte-identical to its active counterpart. The public repository is `PsicologoNavarro/lemon-alpha-reactivity-stai-mspss`.

GitHub Release `v1.0.0` is archived in Zenodo. Cite the version DOI [`10.5281/zenodo.22647023`](https://doi.org/10.5281/zenodo.22647023) for the exact software used by the manuscript. The stable concept DOI [`10.5281/zenodo.22647022`](https://doi.org/10.5281/zenodo.22647022) resolves to the newest archived version. Release and verification procedures are documented in [`docs/GITHUB_ZENODO_RELEASE.md`](docs/GITHUB_ZENODO_RELEASE.md).

## License and citation

Original repository code is released under the [MIT License](LICENSE). This does not relicense LEMON, STAI, MSPSS, or dependencies. Cite the LEMON source descriptor: Babayan et al. (2019), [doi:10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308).

Release metadata status and future additions are listed in [`docs/HUMAN_METADATA_REQUIRED.md`](docs/HUMAN_METADATA_REQUIRED.md).
