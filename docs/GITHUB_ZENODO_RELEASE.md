# GitHub and Zenodo release instructions

The repository contains an ordered creator list, affiliations, and every verified ORCID identifier. Creators must verify and authorize these metadata before each release; absence of an ORCID is valid when no identifier has been verified.

## 1. Verify human metadata

1. Review every author, affiliation, and ORCID in `CITATION.cff`, `.zenodo.json`, and their completed `.template` copies against the manuscript-author-approved record. Keep each template byte-identical to its active counterpart.
2. Confirm that the repository contributors own the original code and authorize the MIT license. The license does not cover LEMON or the questionnaires.
3. Confirm the selected public repository, `PsicologoNavarro/lemon-alpha-reactivity-stai-mspss`. Its URL is already recorded as `repository-code` in `CITATION.cff` and its completed template.
4. If a verified grant identifier or Zenodo community applies, add it to `.zenodo.json`; do not infer one from the scholarship acknowledgement.
5. Do not add a manuscript DOI until that DOI exists.

Zenodo uses `.zenodo.json` instead of `CITATION.cff` when both exist, while GitHub displays `CITATION.cff`. Keep title, authors, affiliations, ORCID identifiers, version, and license synchronized. The release auditor checks these fields across both files, but author approval remains a human responsibility. Validate the CFF with:

```powershell
& ".\.venv-analysis\Scripts\cffconvert.exe" --validate
```

## 2. Audit and create the GitHub repository

Run before `git init`:

```powershell
& ".\.venv-analysis\Scripts\python.exe" ".\scripts\verify_repository.py" --repository "." --write-manifest
```

Then initialize, inspect every staged path, commit, and create a public GitHub repository:

```powershell
git init -b main
git add .
git status --short
& ".\.venv-analysis\Scripts\python.exe" ".\scripts\verify_repository.py" --repository "." --release
git commit -m "Initial reproducible release"
gh auth login
gh repo create PsicologoNavarro/lemon-alpha-reactivity-stai-mspss --public --source . --remote origin --push
```

The commands follow GitHub's documented flow for [adding locally hosted code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github). Do not upload prior local archives or the local `workdir/`.

## 3. Enable Zenodo before the first release

Zenodo archives public repositories. In Zenodo, open the GitHub integration, choose `Sync now`, and enable this repository before publishing the first GitHub Release. See Zenodo's [enable-repository instructions](https://help.zenodo.org/docs/github/enable-repository/).

The GitHub–Zenodo integration cannot prereserve a DOI. Do not place a fictitious DOI in the metadata. [Zenodo documents this limitation](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release).

## 4. Tag and publish release v1.0.0

After the release audit passes and Zenodo is enabled:

```powershell
git tag -a v1.0.0 -m "Reproducible analysis release v1.0.0"
git push origin v1.0.0
```

Wait for the tag-triggered `release-audit` GitHub Action to pass. It validates `CITATION.cff`, privacy, tracked-file completeness, and `SHA256SUMS`. Only then publish the release:

```powershell
gh release create v1.0.0 --verify-tag --title "v1.0.0" --notes-file "RELEASE_NOTES_v1.0.0.md"
```

A tag alone is insufficient; publish the GitHub Release. See [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) and [`gh release create`](https://cli.github.com/manual/gh_release_create).

## 5. Verify the Zenodo deposit

Check the Zenodo GitHub panel and any `Errors` entry. The first archived release creates:

- a version DOI for exactly v1.0.0; cite this DOI in the reproducibility statement;
- a concept DOI that resolves to the newest version.

Future GitHub Releases receive new version DOIs under the same concept. See [Zenodo DOI versioning](https://zenodo.org/help/versioning).

Download Zenodo's archived snapshot, confirm that it corresponds to tag v1.0.0, and verify `SHA256SUMS`. Add the concept DOI badge and citation metadata after Zenodo creates the DOI; do not create an artificial release merely for that metadata-only update.

The first release was verified on 2026-09-07:

- version DOI: [`10.5281/zenodo.22647023`](https://doi.org/10.5281/zenodo.22647023);
- concept DOI: [`10.5281/zenodo.22647022`](https://doi.org/10.5281/zenodo.22647022);
- archived ZIP: 7,729,686 bytes, MD5 `0f97c286b8148b7694917769467435b5`;
- archive audit: all 88 files were present and all 87 entries in `SHA256SUMS` matched.

## Large-file warning

GitHub blocks ordinary Git objects above 100 MiB and warns above 50 MiB. This repository needs no Git LFS because all large data and derivatives are generated locally. LFS pointers may be archived instead of their target objects unless repository settings are changed, so adding LFS would make Zenodo verification harder. See [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
