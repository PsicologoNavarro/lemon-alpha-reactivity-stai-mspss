# Reproducibility guide

## Scope

The workflow starts with the official LEMON distribution and ends with participant-level models, aggregate CSV tables, extended TXT reports, 600-dpi PNG figures, vector PDFs, and audit manifests. The inferential unit is always one participant. Windows and channels estimate participant-level EEG quantities and never inflate the statistical sample size.

The expected flow is `147 eligible -> 142 complete BrainVision triplets -> 139 processable EEG records -> 137 complete primary-model records`.

## Computing requirements

- Windows PowerShell 5.1 or PowerShell 7; individual Python commands are portable to other operating systems.
- Python 3.12.
- At least 65–70 GiB free disk space is recommended. The archived run used approximately 42.66 GiB for audited sources, 7.71 GiB for clean FIF files, and 0.80 GiB for per-participant PSD Parquet files, plus caches and temporary space.
- The archived sequential run took approximately 3.77 hours for preprocessing and 0.17 hours for PSD/QC on one machine. Hardware, network, and library behavior will change runtime.

Two environments are intentional. The EEG/ICLabel stage and final statistical stage were generated under different frozen dependency sets; silently forcing one mixed environment would be less faithful.

## Commands

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage setup
```

Read `DATA_ACCESS_AND_PRIVACY.md`, then download the public data. The operation is resumable and may download more than 40 GiB:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage download -AcceptDataResponsibility -NoInstall
```

Run each auditable stage:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage audit -NoInstall
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage preprocess -NoInstall
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage features -NoInstall
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage analysis -NoInstall
```

After setup, a complete run can instead be requested with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage all -AcceptDataResponsibility -NoInstall
```

Every long stage is resumable. Existing complete outputs are skipped unless `-Force` is supplied. Do not use `-SkipLoo` for a final reproduction.

For deliberately small download batches, add for example `-MaxSubjects 10` and rerun the same download command. Already complete triplets are verified but do not consume the next batch quota. Omit `-MaxSubjects` for the final audit and subsequent stages.

## Stage contract

1. `01_download_lemon.py`: downloads and hashes exact metadata; constructs the 20–35-year cohort using released STAI/MSPSS totals; downloads the BrainVision triplets.
2. `02_audit_sources.py`: hashes and audits payload length, channels, sidecar references, markers, recoveries, and exclusions.
3. `03_preprocess_eeg.py`: reconstructs EO/EC blocks, detects channels, runs joint EO+EC ICA/ICLabel, interpolates flagged channels, and writes separate clean FIF files.
4. `04_compute_psd_qc.py`: computes per-window/channel QC and PSD summaries used to freeze retention flags. Its optional combined CSV export is unnecessary and is not used by the final model.
5. `05_freeze_qc.py`: verifies 695 processed-file hashes and freezes primary and QC-sensitivity cohorts.
6. `06_build_analysis_inputs.py`: independently builds the 88-bin 1–45 Hz spectra, the four canonical bands, and the local long table used for relative-power reconstruction.
7. `07_run_analysis.py`: generates all final statistical tables, diagnostics, sensitivity analyses, topographies, reports, and figures.

## Validation policy

Byte hashes in `reference/expected_results.json` identify the archived Windows run. They are not imposed on a fresh cross-platform reconstruction because absolute paths, timestamps, compressed-container metadata, MNE serialization, and numerical libraries can legitimately change bytes. Fresh runs instead require:

- exact shapes, counts, frequency grids, edge rules, and EC-minus-EO identities;
- hashes of every locally generated upstream file within that run;
- numeric regression against the aggregate primary estimates with strict tolerances;
- repository privacy and integrity checks.

`07_run_analysis.py --enforce-archived-input-hashes` is only for the original archived byte set. It is not the normal reproduction mode.

## Expected local artifacts

Generated files are under `workdir/`, which is deliberately ignored by Git. Important inputs include:

- `workdir/analysis/QC_FREEZE/tables/participant_qc_master.csv`
- `workdir/features/canonical_band_psd_EO_EC_reactivity_delta_theta_alpha_beta.npz`
- `workdir/features/full_spectrum_EO_EC_reactivity_1.00_45.00Hz_bin0.50Hz.npz`
- `workdir/LEMON_139_participantes_61_canales_88_intervalos_EO_EC_reactividad_STAI_MSPSS_JMP.csv`
- `workdir/analysis/final/`

Do not add these to Git. The checked-in `results/` directory contains only the aggregate reference outputs safe for the scientific release.
