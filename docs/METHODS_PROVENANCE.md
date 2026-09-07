# Methods and provenance contract

This document states the calculations implemented by the scripts. The executable code and generated JSON/CSV audits remain authoritative.

## Source and cohort

The pipeline reads the exact GWDG LEMON distribution. Eligibility is age band 20–25, 25–30, or 30–35 years, nonmissing released STAI trait total, nonmissing released MSPSS total, and Day-1 availability. Of 147 eligible records, 142 have complete BrainVision triplets. Source audit yields 136 direct passes, three documented recoveries, and three exclusions, leaving 139 processable EEG records. Two records lack partnership status, leaving 137 complete cases for adjusted primary models.

The documented recoveries and exclusions are public dataset IDs and appear only in the two source-processing scripts, this document, and `reference/known_source_variants.json`.

## Acquisition inherited from LEMON

Resting EEG uses 61 scalp electrodes plus VEOG, FCz reference, 2500-Hz source sampling, and alternating EO/EC blocks. The source paper and supplied metadata should be cited for the full acquisition protocol.

## Preprocessing

- Each of eight 60-second blocks per condition is extracted without crossing discontinuities and resampled to 250 Hz.
- Bad-channel QC uses microvolts. Robust scale is `1.4826 * median(abs(x - median(x)))`. The scale score applies the same robust-z construction to `log10(robust_scale)` across channels.
- For spatial QC, the six nearest channels in three-dimensional montage space are selected. After temporal decimation by five, each channel is correlated with the sample-wise median of its neighbors. A neighbor flag requires correlation `< 0.40` and across-channel robust correlation z `< -5`.
- Flat channels require robust SD `< 0.1 uV`; scale outliers require absolute robust log-SD z `> 5`.
- MNE LOF uses 20 neighbors and threshold 2.0 but is advisory only. LOF alone never triggers interpolation.
- The final bad-channel set is the union of flat, scale, and neighbor flags. More than ten automatically bad channels stops that record. Flagged channels are interpolated after ICA.
- One joint EO+EC extended-Infomax ICA is fitted per participant after 50/100-Hz notch filtering, 1–100-Hz filtering, average reference, rank estimation, decimation 2, maximum 1024 iterations, and 500-uV rejection during fitting. Its deterministic seed is `20260902 + numeric participant ID` modulo `2^32 - 1`.
- ICLabel uses the ONNX backend. A component is rejected when its argmax class is muscle artifact, eye blink, heart beat, line noise, or channel noise and its argmax probability is at least 0.80. An EOG flag adds rejection only when absolute EOG score is at least 0.50 and ICLabel eye-blink probability is at least 0.30.
- Clean EO and EC are filtered 1–45 Hz and average-referenced separately. Source BrainVision files are never modified. Normalized sidecars use a hard link when possible and a verified byte copy across filesystems.

## Window QC and PSD

At 250 Hz, each 4-second window contains 1000 samples. A 2-second step yields 29 windows per 60-second block and 232 per condition/channel. A channel-window is accepted only if all samples are finite, maximum absolute amplitude is at most 250 uV, peak-to-peak amplitude is at most 500 uV, and robust SD is at least 0.1 uV.

Each accepted window is analyzed with SciPy's one-sided periodogram, Hamming window, constant detrending, and `scaling="density"`. Native FFT spacing is 0.25 Hz. The Hamming equivalent-noise-bandwidth approximation is 1.36 bins, or 0.34 Hz effective resolution.

For each contiguous 0.5-Hz interval from 1 to 45 Hz, both endpoints lie on the FFT grid. The one-sided linear PSD density is trapezoid-integrated across the interval and divided by 0.5 Hz. These interval densities are averaged across accepted windows in linear units and then transformed as `10*log10`, yielding dB re 1 uV^2/Hz. Reactivity is `EC_dB - EO_dB`, equivalently `10*log10(EC_linear / EO_linear)`.

Canonical density uses additive 0.5-Hz interval integrals divided by band width before log transformation: delta 1–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, and beta 13–30 Hz. Shared endpoints carry zero area and are not double-counted. A regression test validates equivalence to direct closed-endpoint trapezoidal integration.

Relative alpha is `100 * integrated alpha power / integrated 1–45-Hz power` in linear units. Relative reactivity is EC minus EO in percentage points.

## Statistical analysis

The outcome is the arithmetic mean alpha reactivity across P3, Pz, P4, PO3, POz, PO4, O1, Oz, and O2, one value per participant. Separate natural-spline models evaluate STAI and MSPSS, with nominal basis df=4 reduced to a 3-df estimable total association. Each adjusts for the other psychometric score, sex, and partnership status. Inference uses HC3 covariance, two-sided Wald tests, 95% confidence intervals, and Holm correction across the two focal total tests; nonlinearity uses a separate two-test Holm family.

Sensor maps are secondary spatial descriptions. BH-FDR is applied across 61 total 3-df spline tests separately per map. The color encodes the directional linear component; stars encode the total spline test and do not imply that the colored linear term is significant.

Alpha was selected after same-cohort canonical-band exploration and is explicitly post hoc. The final specification is therefore repeated across delta, theta, alpha, and beta with Holm across eight band-by-scale tests. Only alpha–MSPSS survives that Holm-8 sensitivity; alpha–STAI and all non-alpha total tests do not.
