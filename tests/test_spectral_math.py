from __future__ import annotations

import math

import numpy as np

from conftest import load_script


spectral = load_script("spectral_builder", "06_build_analysis_inputs.py")


def test_interval_density_matches_linear_analytic_integral() -> None:
    frequencies = np.arange(0.0, 2.0 + 0.25, 0.25)
    density = 1.0 + 2.0 * frequencies
    observed = float(spectral.interval_mean_density(frequencies, density, 1.0, 1.5))
    # Mean of 1 + 2f over 1.0--1.5 is 3.5.
    assert math.isclose(observed, 3.5, rel_tol=0.0, abs_tol=1e-14)


def test_canonical_bands_are_additive_linear_integrals(tmp_path) -> None:
    participants = 2
    frequencies = np.arange(1.25, 45.0, 0.5)
    channels = np.asarray([f"E{index:02d}" for index in range(61)], dtype="U16")
    open_density = 2.0
    closed_density = 4.0
    shape = (participants, 88, 61)
    full = {
        "participant_ids": np.asarray(["p1", "p2"], dtype="U32"),
        "channels": channels,
        "frequencies": frequencies,
        "psd_open": np.full(shape, 10.0 * np.log10(open_density)),
        "psd_closed": np.full(shape, 10.0 * np.log10(closed_density)),
        "reactivity": np.full(shape, 10.0 * np.log10(closed_density / open_density)),
        "counts_open": np.full((participants, 61), 232, dtype=int),
        "counts_closed": np.full((participants, 61), 232, dtype=int),
        "source_signature": np.asarray("0" * 64, dtype="U64"),
    }
    path = spectral.build_canonical_cache(tmp_path, full)
    with np.load(path, allow_pickle=False) as cached:
        assert cached["psd_open"].shape == (2, 4, 61)
        assert np.allclose(cached["psd_open"], 10.0 * np.log10(open_density), atol=1e-13)
        assert np.allclose(cached["psd_closed"], 10.0 * np.log10(closed_density), atol=1e-13)
        assert np.allclose(cached["reactivity"], 10.0 * np.log10(2.0), atol=1e-13)


def test_frequency_grid_has_88_contiguous_intervals() -> None:
    centers = np.arange(1.25, 45.0, 0.5)
    assert len(centers) == 88
    assert centers[0] - 0.25 == 1.0
    assert centers[-1] + 0.25 == 45.0
