"""
Compute and save all PIF timescale analysis data.

The saved NumPy archive contains radius curves, crossings and grid
classification for every catalog source.

Efficiency: the geometric curves (g, tau_w, tau_a) depend only on the spin
(and on the r_out grid), not on M or the source. They are computed ONCE per
spin by pif_core.pif_geometric (alpha and HR_isco are fixed in pif_core);
every (source, mass) combination is then a rescaling plus a cheap crossing
search (see pif_core for the exact scaling laws, which depend on the source
only through P = M * nu_0).

Validity of a configuration (a, M): there is a precession radius r_bar with
nu_solid(r_bar) = nu_0 such that t_wave(r_bar) < 1/nu_0 < t_align(r_bar).
Since this depends on (M, nu_0) only through P = M * nu_0, the (mass, spin)
validity grid of a source is obtained with ONE vectorised call per spin
(pif_mass_freq_map.valid_on_P) over all masses at once.
"""

import os
import numpy as np

from setup import M_AGN_MAX, M_AGN_MIN
from pif_core import ALPHA, HR_ISCO_DEFAULT
from catalog import CATALOG, select_sources
from pif_mass_freq_map import build_spin_curves, R_OUT_GRID

A_VALS = np.array([-0.9, -0.5, -0.25, 0.0, 0.01, 0.25, 0.5, 0.9, 0.998])
A_FIXED = 0.5
N_M = 6
N_A_GRID = 401
N_M_GRID = 300
GRID_SPINS = np.linspace(-1.0, 1.0, N_A_GRID)
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


def _loglog_at(radii, values, radius):
    """Log-log interpolation of `values(radii)` at `radius`; NaN if `radius`
    is outside the range where `values` is finite and positive."""
    valid = np.isfinite(values) & (values > 0)
    if valid.sum() < 2:
        return np.nan
    x = np.log(radii[valid])
    if not (x[0] <= np.log(radius) <= x[-1]):
        return np.nan
    return float(np.exp(np.interp(np.log(radius), x, np.log(values[valid]))))


def _precession_radius_flags(r_out, t_wave, t_align, r_precession, nu_target):
    """
    Validity of each precession radius r_bar (nu_solid(r_bar) = nu_0): rigid
    precession requires the wave-propagation time to be shorter and the
    alignment time longer than the oscillation period,

        t_wave(r_bar) < 1/nu_0 < t_align(r_bar).

    The condition is evaluated pointwise at r_bar, so it does not rely on the
    monotonicity of t_align(r_out), which does not hold in general. A match
    without any precession radius has no flags and is therefore not valid.
    """
    period = 1.0 / nu_target
    flags = []
    for radius in r_precession:
        t_w = _loglog_at(r_out, t_wave, radius)
        t_a = _loglog_at(r_out, t_align, radius)
        flags.append(bool(np.isfinite(t_w) and np.isfinite(t_a)
                          and t_w < period < t_a))
    return flags


def match_is_ok(match):
    """A match is valid if at least one precession radius satisfies
    t_wave < 1/nu_0 < t_align (single definition used by all scripts)."""
    return any(match['precession_ok'])


def _match_from_curves(spin_curves, nu_target, mass, alpha=None):
    """
    Rescale the M-independent curves to `mass` (and `alpha`, default: the
    value fixed in pif_core) and find the crossings with the observed
    frequency nu_target: only cheap operations.
    """
    r_out = spin_curves.r_out
    nu_solid, t_wave, t_align = spin_curves.geo.physical(mass, alpha)
    r_precession = _crossing_radii(r_out, nu_solid, nu_target)
    return {
        'r_out': r_out,
        't_wave': t_wave,
        't_align': t_align,
        'r_precession': r_precession,
        'precession_ok': _precession_radius_flags(
            r_out, t_wave, t_align, r_precession, nu_target),
        'r_wave': _crossing_radii(r_out, t_wave, 1.0 / nu_target),
        'r_align': _crossing_radii(r_out, t_align, 1.0 / nu_target),
    }


def _spin_scan(curves, nu_target, spins, mass):
    """Matches for several spins at fixed mass (kind 'a')."""
    matches = []
    for spin in spins:
        spin_curves = curves.get(float(spin))
        if spin_curves is None:
            continue
        matches.append({
            'label': fr'$a={spin:.2f}$',
            'parameter': float(spin),
            'parameter_kind': 'a',
            **_match_from_curves(spin_curves, nu_target, mass),
        })
    return matches


def _mass_scan(curves, nu_target, spin, masses):
    """Matches for several masses at fixed spin (kind 'M')."""
    spin_curves = curves.get(float(spin))
    if spin_curves is None:
        return []
    return [{
        'label': fr'$M={mass:.1e}\,M_\odot$',
        'parameter': float(mass),
        'parameter_kind': 'M',
        **_match_from_curves(spin_curves, nu_target, mass),
    } for mass in masses]


def _grid_validity(curves, nu0, grid_masses):
    """
    Boolean (n_masses, n_spins) validity grid of a source with frequency nu0.
    Validity depends on (M, nu0) only through P = M * nu0, so each spin needs
    a single vectorised call over all masses.
    """
    # Lazy import: pif_mass_freq_map imports build_spin_curves from this
    # module, so a top-level import here would be circular.
    from pif_mass_freq_map import valid_on_P

    P = grid_masses * nu0
    grid_ok = np.zeros((grid_masses.size, GRID_SPINS.size), dtype=bool)
    for spin_index, spin in enumerate(GRID_SPINS):
        spin_curves = curves.get(float(spin))
        if spin_curves is not None:
            grid_ok[:, spin_index] = valid_on_P(spin_curves, P)
    return grid_ok


def _compute_source_analysis(source, curves):
    nu0 = source['nu0']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])

    panel1 = _spin_scan(curves, nu0, A_VALS, M_ref)
    panel2 = _mass_scan(
        curves, nu0, A_FIXED, np.logspace(np.log10(M_lo), np.log10(M_hi), N_M))

    grid_masses = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M_GRID)
    grid_ok = _grid_validity(curves, nu0, grid_masses)

    return {
        'source': source,
        'M_ref': float(M_ref),
        'M_lo': float(M_lo),
        'M_hi': float(M_hi),
        'kind': kind,
        'panel1_matches': panel1,
        'panel2_matches': panel2,
        'grid_spins': GRID_SPINS,
        'grid_masses': grid_masses,
        'grid_ok': grid_ok,
    }


def compute_and_save(sources=None, indices=None, output_file=DATA_FILE):
    if sources is not None and indices is not None:
        raise ValueError('Pass either sources or indices, not both')
    if sources is None:
        sources = select_sources(indices) if indices is not None else CATALOG

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Geometric curves: once per spin, shared by all sources and masses.
    all_spins = np.concatenate([A_VALS, [A_FIXED], GRID_SPINS])
    curves = build_spin_curves(all_spins)
    print(f'Geometric curves computed for {len(curves)} spins')

    analyses = []
    for index, source in enumerate(sources, start=1):
        print(f'Source {index}/{len(sources)} {source["name"]}')
        analyses.append(_compute_source_analysis(source, curves))
    np.savez_compressed(
        output_file,
        analyses=np.array(analyses, dtype=object),
        r_out_grid=R_OUT_GRID,
        a_vals=A_VALS,
        a_fixed=A_FIXED,
        alpha=ALPHA,
        hr_isco=HR_ISCO_DEFAULT,
        criterion='t_wave(r_bar) < 1/nu_0 < t_align(r_bar)',
    )
    print(f'Saved: {output_file}')
    return output_file


if __name__ == '__main__':
    compute_and_save()#indices=[0, 9, 10, 11])