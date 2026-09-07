# Ansiedad rasgo, apoyo social y reactividad alfa posterior en LEMON

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22647022.svg)](https://doi.org/10.5281/zenodo.22647022)

Repositorio reproducible y auditable para un análisis secundario exploratorio del EEG de reposo público LEMON. Incluye toda la cadena: descarga oficial, construcción de cohorte, auditoría BrainVision, preprocesamiento EO/EC de 61 canales, PSD, control de calidad, reactividad EC−EO, cuatro bandas canónicas, espectro 1–45 Hz, modelos estadísticos, tablas, informes y figuras de publicación.

No contiene EEG LEMON ni mediciones individuales. Los resultados incluidos son agregados.

## Qué reproduce

- 147 registros elegibles → 142 tripletes completos → 139 EEG procesables → 137 casos completos para los modelos ajustados.
- Ventanas de 4 s con 50% de solapamiento; periodograma Hamming; resolución FFT de 0.25 Hz y resolución efectiva aproximada de 0.34 Hz.
- 88 intervalos contiguos de 0.5 Hz entre 1 y 45 Hz.
- Delta 1–4, theta 4–8, alfa 8–13 y beta 13–30 Hz, con PSD EO, PSD EC y reactividad EC−EO.
- Modelos spline a nivel participante, HC3, multiplicidad Holm y mapas secundarios con BH-FDR.

Alfa es un foco post hoc informado por la exploración de las cuatro bandas. En la sensibilidad Holm de ocho pruebas, solamente alfa–MSPSS conserva significación; alfa–STAI y todas las asociaciones no alfa no la conservan.

## Comprobación rápida

El registro de comprobaciones realizadas y su límite explícito está en [`docs/VALIDATION.md`](docs/VALIDATION.md). Desde PowerShell, dentro de esta carpeta:

```powershell
py -3.12 ".\scripts\verify_repository.py" --repository "."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage setup
& ".\.venv-analysis\Scripts\python.exe" -m pytest -q
```

## Ejecución completa

Primero lea [`docs/DATA_ACCESS_AND_PRIVACY.md`](docs/DATA_ACCESS_AND_PRIVACY.md). La descarga supera 40 GiB y el procesamiento requiere aproximadamente 65–70 GiB libres. Después:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_pipeline.ps1" -Stage all -AcceptDataResponsibility -NoInstall
```

El proceso es reanudable. Para controlarlo por etapas utilice, en orden, `download`, `audit`, `preprocess`, `features` y `analysis`. Las instrucciones completas están en [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

Todo dato o derivado individual queda en `workdir/`, fuera de Git. No fuerce su inclusión.

## Subida a GitHub y DOI Zenodo

Los archivos activos [`CITATION.cff`](CITATION.cff) y [`.zenodo.json`](.zenodo.json), junto con sus copias completas `CITATION.cff.template` y `.zenodo.json.template`, contienen los autores, afiliaciones y ORCID en el orden informado en el manuscrito final de envío. Las pruebas automatizadas exigen que cada plantilla sea idéntica a su archivo activo. El repositorio público es `PsicologoNavarro/lemon-alpha-reactivity-stai-mspss`.

Siga [`docs/GITHUB_ZENODO_RELEASE.md`](docs/GITHUB_ZENODO_RELEASE.md). El orden correcto es:

1. auditar y subir el repositorio público a GitHub;
2. habilitar ese repositorio en Zenodo;
3. crear el tag anotado `v1.0.0`;
4. publicar una GitHub Release, no solamente el tag;
5. verificar la ingestión de Zenodo y descargar/comprobar su snapshot.

La release `v1.0.0` ya está archivada. Para reproducir la versión exacta del artículo debe citarse [`10.5281/zenodo.22647023`](https://doi.org/10.5281/zenodo.22647023). El DOI conceptual estable [`10.5281/zenodo.22647022`](https://doi.org/10.5281/zenodo.22647022) conduce a la versión archivada más reciente.

## Licencia

El código original usa MIT. Esta licencia no relicencia LEMON, STAI, MSPSS ni software de terceros. Debe citarse el estudio fuente de LEMON: Babayan et al. (2019), [doi:10.1038/sdata.2018.308](https://doi.org/10.1038/sdata.2018.308).
