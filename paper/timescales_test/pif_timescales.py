"""Compute and save all PIF timescale analysis data.

Plotting is intentionally kept in ``pif_timescales_all.py`` and ``test.py``.
The saved NumPy archive contains radius curves, crossings, grid
classification, and summary samples for every catalog source.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import M_AGN_MAX, M_AGN_MIN, r_isco
from disk_profiles import t_wave_closed
from align_timescale import t_align_vect
from catalog import CATALOG, select_sources
from nu_solid_v2 import nu_solid_vect_p

ALPHA = 0.01
A_VALS = np.array([-0.9, -0.5, -0.25, 0.0, 0.01, 0.25, 0.5, 0.9, 0.998])
A_FIXED = 0.5
N_M = 6
N_A_SUMMARY = 100
N_M_SUMMARY = 50
N_A_GRID = 41
N_M_GRID = 41
N_R_OUT_INNER = 100
N_R_OUT_OUTER = 100
R_OUT_GRID = np.concatenate([
    np.logspace(0, 1, N_R_OUT_INNER, endpoint=False),
    np.logspace(1, np.log10(200.0), N_R_OUT_OUTER),
])
EXPLORATORY_DEX = 1.0
DATA_FILE = os.path.join(
    os.path.dirname(__file__), 'output_catalog', 'pif_timescales_analysis.npz'
)


def resolve_mass_reference(mass_range):
    if mass_range is None:
        M_lo, M_hi = M_AGN_MIN, M_AGN_MAX
        return np.sqrt(M_lo * M_hi), M_lo, M_hi, 'generic'

    lo, hi = mass_range
    if lo == hi:
        return lo, lo / 10**EXPLORATORY_DEX, lo * 10**EXPLORATORY_DEX, 'exploratory'
    return np.sqrt(lo * hi), lo, hi, 'band'


def _crossing_radii(radii, values, target):
    valid = np.isfinite(values) & (values > 0)
    radii = radii[valid]
    values = values[valid]
    if radii.size < 2:
        return []

    log_delta = np.log(values) - np.log(target)
    crossing_indices = np.flatnonzero(log_delta[:-1] * log_delta[1:] <= 0)
    crossings = []
    for index in crossing_indices:
        left_delta = log_delta[index]
        right_delta = log_delta[index + 1]
        if left_delta == 0:
            radius = radii[index]
        elif right_delta == 0:
            radius = radii[index + 1]
        else:
            fraction = -left_delta / (right_delta - left_delta)
            log_radius = np.log(radii[index]) + fraction * (
                np.log(radii[index + 1]) - np.log(radii[index])
            )
            radius = np.exp(log_radius)
        crossings.append(float(radius))
    return crossings


def _match_is_ok(match):
    r_precession = match['r_precession']
    timescale_matches = [
        radii[0] for radii in (match['r_wave'], match['r_align']) if radii
    ]
    if not timescale_matches:
        return True
    return bool(r_precession) and r_precession[0] < min(timescale_matches)


def _match_for_parameters(nu_target, spin, mass):
    r_in = float(r_isco(spin))
    r_out = R_OUT_GRID[R_OUT_GRID > r_in * 1.001]
    if r_out.size == 0:
        return None

    r_in_arr = np.full_like(r_out, r_in)
    spin_arr = np.full_like(r_out, spin)
    mass_arr = np.full_like(r_out, mass)
    t_wave = t_wave_closed(r_in_arr, r_out, spin_arr, mass_arr)
    t_align = t_align_vect(
        spin_arr, r_in_arr, r_out, mass_arr,
        np.full(r_out.size, ALPHA),
    )
    r_precession = _crossing_radii(
        r_out,
        nu_solid_vect_p(spin_arr, r_in_arr, r_out, mass_arr),
        nu_target,
    )
    return {
        'r_out': r_out,
        't_wave': t_wave,
        't_align': t_align,
        'r_precession': r_precession,
        'r_wave': _crossing_radii(r_out, t_wave, 1.0 / nu_target),
        'r_align': _crossing_radii(r_out, t_align, 1.0 / nu_target),
    }


def _source_matches(source, a_values, masses, cache):
    nu0 = source['nu0']
    matches_spin = []
    for spin, mass in zip(a_values, masses):
        key = (float(spin), float(mass))
        if key not in cache:
            cache[key] = _match_for_parameters(nu0, spin, mass)
        match = cache[key]
        if match is not None:
            matches_spin.append({
                'label': fr'$a={spin:.2f}$',
                'parameter': float(spin),
                'parameter_kind': 'a',
                **match,
            })

    matches_mass = []
    for mass in masses:
        key = (float(A_FIXED), float(mass))
        if key not in cache:
            cache[key] = _match_for_parameters(nu0, A_FIXED, mass)
        match = cache[key]
        if match is not None:
            matches_mass.append({
                'label': fr'$M={mass:.1e}\,M_\odot$',
                'parameter': float(mass),
                'parameter_kind': 'M',
                **match,
            })
    return matches_spin, matches_mass


def _print_progress(name, completed, total, index, count):
    fraction = completed / total if total else 1.0
    slots = 10
    filled = int(slots * fraction)
    bar = '#' * filled + '-' * (slots - filled)
    message = f'Source {index}/{count} {name}: [{bar}] {fraction:6.1%}'
    sys.stdout.write(f'\r\033[2K{message}')
    sys.stdout.flush()


def _compute_source_analysis(source, source_index, source_count):
    nu0 = source['nu0']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])
    cache = {}

    panel1, _ = _source_matches(
        source, A_VALS, np.full(A_VALS.size, M_ref), cache
    )
    panel2_masses = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M)
    _, panel2 = _source_matches(
        source, np.array([]), panel2_masses, cache
    )

    grid_spins = np.linspace(-1.0, 1.0, N_A_GRID)
    grid_masses = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M_GRID)
    grid_ok = np.zeros((N_M_GRID, N_A_GRID), dtype=bool)
    total = N_M_GRID * N_A_GRID
    completed = 0
    next_fraction = 0.1
    _print_progress(source['name'], 0, total, source_index, source_count)
    for mass_index, mass in enumerate(grid_masses):
        for spin_index, spin in enumerate(grid_spins):
            key = (float(spin), float(mass))
            if key not in cache:
                cache[key] = _match_for_parameters(nu0, spin, mass)
            match = cache[key]
            grid_ok[mass_index, spin_index] = (
                match is not None and _match_is_ok(match)
            )
            completed += 1
            fraction = completed / total
            if fraction >= next_fraction or completed == total:
                _print_progress(
                    source['name'], completed, total,
                    source_index, source_count,
                )
                next_fraction += 0.1

    summary_spins = np.linspace(-1.0, 1.0, N_A_SUMMARY + 2)[1:-1]
    summary_masses = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M_SUMMARY)
    summary1, _ = _source_matches(
        source, summary_spins, np.full(summary_spins.size, M_ref), cache
    )
    _, summary2 = _source_matches(
        source, np.array([]), summary_masses, cache
    )

    return {
        'source': source,
        'M_ref': float(M_ref),
        'M_lo': float(M_lo),
        'M_hi': float(M_hi),
        'kind': kind,
        'panel1_matches': panel1,
        'panel2_matches': panel2,
        'grid_spins': grid_spins,
        'grid_masses': grid_masses,
        'grid_ok': grid_ok,
        'summary_spin_matches': summary1,
        'summary_mass_matches': summary2,
    }


def compute_and_save(sources=None, indices=None, output_file=DATA_FILE):
    if sources is not None and indices is not None:
        raise ValueError('Pass either sources or indices, not both')
    if sources is None:
        sources = select_sources(indices) if indices is not None else CATALOG

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    analyses = []
    for index, source in enumerate(sources, start=1):
        analyses.append(_compute_source_analysis(source, index, len(sources)))
        sys.stdout.write('\n')
    np.savez_compressed(
        output_file,
        analyses=np.array(analyses, dtype=object),
        r_out_grid=R_OUT_GRID,
        a_vals=A_VALS,
        a_fixed=A_FIXED,
        alpha=ALPHA,
    )
    print(f'Salvato: {output_file}')
    return output_file


if __name__ == '__main__':
    compute_and_save()#indices=[0, 9, 10, 11])
