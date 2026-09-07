# Data access, licensing, and privacy boundary

This repository contains code, aggregate result tables, figures, and audit metadata. It does not contain LEMON EEG, questionnaire rows, participant-level analysis tables, FIF/ICA files, PSD caches, Parquet files, or the 180.47 MiB full-frequency table.

## Exact data source

The frozen analysis used the public MPI Leipzig Mind-Brain-Body (LEMON) files from the GWDG distribution endpoint:

- `https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/`
- persistent identifier: `21.11101/0000-0007-C379-5`
- source access used for the archived analysis: 2026-09-02
- source descriptor: Babayan et al., *Scientific Data* (2019), [doi:10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308)
- dataset page and stated PDDL status: [INDI MPI-LEMON](https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/MPI_LEMON.html)

Public availability is not treated as permission to repackage every source file. Reusers must read the source terms and cite the source study. The MIT license in this repository applies only to original repository code; it does not relicense LEMON, STAI, MSPSS, or third-party software.

The downloader requires the explicit `--accept-data-responsibility` flag. It verifies the nine metadata files against the hashes used in the archived run. Do not silently substitute OpenNeuro, NEMAR, or another conversion: a different distribution must first be validated against the frozen source and documented.

## Psychometric boundary

The cohort code consumes the released `STAI_Trait_Anxiety` and `MSPSS_total` columns. It does not reconstruct scores from item responses and does not redistribute questionnaire items, manuals, or source behavioral CSV files. Relationship status is obtained from the released LEMON metadata and is used only in locally generated complete-case analyses.

## Dataset IDs and anonymization language

LEMON provides pseudonymous dataset IDs. Six IDs must remain in narrowly scoped source-processing code, generated methods provenance, and `reference/known_source_variants.json` because they define three transparent recovery rules and three EEG exclusions. Call them “public pseudonymous LEMON dataset IDs,” not newly anonymized participants.

Before a public release, `scripts/verify_repository.py --release` rejects:

- raw or processed electrophysiology formats;
- participant-level and oversized analytical inputs;
- local absolute paths, local usernames, emails outside live citation metadata, or secret-like strings;
- any file over 50 MiB;
- symlinks and untracked public files.

Run the audit again after `git add`; deleting a sensitive file from the working tree does not remove it from Git history. If a prohibited file was ever committed, create a new clean repository or use an appropriate history-removal procedure before publication.
