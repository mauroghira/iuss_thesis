"""
Mass-frequency map of the PIF validity condition.

For every point (M, nu_0) of a grid and every spin a in [-1, 1] the precession
radius r_bar is defined by nu_solid(r_bar) = nu_0, and the configuration is
valid if  t_wave(r_bar) < 1/nu_0 < t_align(r_bar)  (same criterion as
pif_timescales.match_is_ok). The script stores, for every (M, nu_0):

    a_min_pos : smallest valid spin with a > 0   (NaN if none)
    a_max_neg : largest  valid spin with a < 0   (NaN if none)  [closest to 0]
    a_max_pos : largest  valid spin with a > 0   (NaN if none)
    a_min_neg : smallest valid spin with a < 0   (NaN if none)  [farthest from 0]

and, where no spin of a given sign is valid, WHY (failure category, below).

Efficiency: from pif_core, with g = nu_solid*M, tau_w = t_wave/M,
tau_a = t_align/M (functions of (a, r_out) only),

    nu_solid = nu_0   <=>  g(r_out) = P,        P = M * nu_0
    t_wave  < 1/nu_0  <=>  tau_w(r_out) < 1/P
    t_align > 1/nu_0  <=>  tau_a(r_out) > 1/P

so validity depends on (M, nu_0) ONLY through P = M*nu_0. The validity is
therefore computed on a 1D grid of P (per spin, fully vectorised) and then
mapped onto the 2D (M, nu_0) grid: the map is constant along M*nu_0 = const.

Failure codes of one spin at one P (classify_on_P):
    0 VALID        some r_bar satisfies t_wave < 1/nu_0 < t_align
    1 NOCROSS_HI   g(r_out) < P for every r_out: nu_0 too high, i.e. P above
                   the maximum of g (reached for r_out -> r_in)
    2 NOCROSS_LO   g(r_out) > P for every r_out: nu_0 too low, i.e. the
                   required r_out lies beyond the last node of R_OUT_GRID
    3 T_WAVE       r_bar exists, t_wave < 1/nu_0 fails (t_align holds)
    4 T_ALIGN      r_bar exists, 1/nu_0 < t_align fails (t_wave holds)
    5 BOTH         r_bar exists, neither inequality is satisfied (or each by
                   a different crossing, never jointly)

Category of a point (M, nu_0) for ONE sign of the spin (panel_category):
    - VALID if at least one spin of that sign is valid;
    - otherwise, if at least one spin has an r_bar (codes 3-5), the most
      frequent of the codes 3, 4, 5 among the spins of that sign
      (timescale failure takes precedence);
    - otherwise (no spin of that sign has an r_bar) the most frequent of
      the codes 1, 2: no r_out reproduces nu_0 for any spin.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import NamedTuple

from setup import r_isco, set_style, fix_spines
from pif_core import ALPHA, HR_ISCO_DEFAULT, pif_geometric, PIFGeometric
from catalog import CATALOG

M_MIN, M_MAX = 1e0, 1e10
NU_MIN, NU_MAX = 1e-7, 1e4

# Resolution: a SINGLE parameter. M and nu_0 use the same log-step
# h = 1/N_PER_DECADE dex, so that log P = log M + log nu_0 takes exactly
# N_M + N_NU - 1 distinct values, all on a regular lattice of step h. The 1D
# grid in P has one node on each of them: every pixel (i, j) is assigned
# EXACTLY the node i + j (no nearest-node quantisation).
# Requires the M and nu_0 ranges to span a multiple of 1/N_PER_DECADE dex.
N_PER_DECADE = 50


def _n_nodes(lo, hi, per_decade=N_PER_DECADE):
    n_steps = np.log10(hi / lo) * per_decade
    if not np.isclose(n_steps, round(n_steps), rtol=0.0, atol=1e-9):
        raise ValueError(
            f'range [{lo:g}, {hi:g}] is not a multiple of 1/{per_decade} dex')
    return int(round(n_steps)) + 1


N_M = _n_nodes(M_MIN, M_MAX)          # pixels along M
N_NU = _n_nodes(NU_MIN, NU_MAX)       # pixels along nu_0
N_P = N_M + N_NU - 1                  # nodes of the 1D grid in P = M * nu_0

# Spin grid: |a| log-spaced in [A_FINE_MIN, A_STEP) (fine scan close to 0),
# then uniform with step A_STEP up to EXACTLY A_MAX = 1; mirrored for a < 0,
# plus a = 0.
A_MAX = 1.0
A_STEP = 0.01
A_FINE_MIN = 1e-3
N_A_FINE = 10

N_R_OUT_INNER = 100
N_R_OUT_OUTER = 100
R_OUT_GRID = np.concatenate([
    np.logspace(0, 1, N_R_OUT_INNER, endpoint=False),
    np.logspace(1, np.log10(200.0), N_R_OUT_OUTER),
])
R_IN_MARGIN = 1.001      # only r_out > R_IN_MARGIN * r_in is used

VALID, NOCROSS_HI, NOCROSS_LO, T_WAVE, T_ALIGN, BOTH = range(6)
FAIL_COLORS = {
    NOCROSS_HI: '0.75',      # light grey
    NOCROSS_LO: '0.25',      # dark grey (same family: both are "no r_out")
    T_WAVE: '#56B4E9',
    T_ALIGN: '#E69F00',
    BOTH: '#CC79A7',
}
FAIL_LABELS = {
    NOCROSS_HI: r'no $r_\mathrm{out}$ ($\nu_0$ too high)',
    NOCROSS_LO: (r'no $r_\mathrm{out}\leq$ ' + f'{R_OUT_GRID[-1]:.0f}'
                 + r'$\,r_g$ ($\nu_0$ too low)'),
    T_WAVE: r'$t_\mathrm{wave}<1/\nu_0$ fails',
    T_ALIGN: r'$1/\nu_0<t_\mathrm{align}$ fails',
    BOTH: 'both fail',
}

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_catalog')
DATA_FILE = os.path.join(OUT_DIR, 'pif_mass_frequency_map.npz')
FIG_FILE = os.path.join(OUT_DIR, 'pif_mass_frequency_map.pdf')


class SpinCurves(NamedTuple):
    """M-independent curves of one spin on its valid r_out values."""
    r_out: np.ndarray
    geo: PIFGeometric


def make_spins():
    """Ascending spin grid in [-A_MAX, A_MAX] (contains 0 and +/-A_MAX)."""
    n_coarse = int(round(A_MAX / A_STEP))
    pos = np.concatenate([
        #np.logspace(np.log10(A_FINE_MIN), np.log10(A_STEP), N_A_FINE,
        #            endpoint=False),
        np.linspace(A_STEP, A_MAX, n_coarse),
    ])
    return np.concatenate([-pos[::-1], [0.0], pos])


def build_spin_curves(spins, hr_isco=HR_ISCO_DEFAULT):
    """
    Compute the geometric curves once for every unique spin in `spins`.
    Returns {float(spin): SpinCurves}; spins with no valid r_out are omitted.
    """
    spins = np.unique(np.asarray(spins, dtype=float))
    r_in = r_isco(spins)
    geo = pif_geometric(spins[:, None], R_OUT_GRID[None, :],
                        r_in=r_in[:, None], HR_ref=hr_isco)
    curves = {}
    for i, spin in enumerate(spins):
        keep = R_OUT_GRID > R_IN_MARGIN * r_in[i]
        if not keep.any():
            continue
        curves[float(spin)] = SpinCurves(
            R_OUT_GRID[keep],
            PIFGeometric(geo.g[i, keep], geo.tau_w[i, keep],
                         geo.tau_a[i, keep], geo.HR_ref, geo.alpha_ref),
        )
    return curves


def classify_on_P(curve, P, k_w=1.0, k_a=1.0):
    """
    int8 array (len(P),) of failure codes (see module docstring) for one spin
    at every P = M*nu_0.
    Same logic as _crossing_radii + _precession_radius_flags, vectorised:
    for each segment [r_i, r_{i+1}] of the r_out grid where log g - log P
    changes sign, r_bar is located by linear interpolation in (log r, log g)
    with fraction f; log tau_w, log tau_a are interpolated at the same f
    (log-log interpolation between the same two nodes).
    k_w, k_a rescale tau_w, tau_a for alpha, HR_isco different from the
    reference (see PIFGeometric.physical).
    """
    r, geo = curve.r_out, curve.geo
    g, tw, ta = geo.g, geo.tau_w * k_w, geo.tau_a * k_a
    ok = (np.isfinite(g) & (g > 0) & np.isfinite(tw) & (tw > 0)
          & np.isfinite(ta) & (ta > 0))
    codes = np.full(P.size, NOCROSS_HI, dtype=np.int8)
    if ok.sum() < 2:                      # e.g. a = 0: g = 0 identically
        return codes
    lg, ltw, lta = np.log(g[ok]), np.log(tw[ok]), np.log(ta[ok])
    lP = np.log(P)

    d = lg[None, :] - lP[:, None]         # (N_P, n_r)
    dl, dr = d[:, :-1], d[:, 1:]
    crossing = dl * dr <= 0
    den = dr - dl
    f = np.divide(-dl, den, out=np.zeros_like(dl), where=den != 0)
    f = np.clip(f, 0.0, 1.0)
    ltw_bar = ltw[:-1] + f * (ltw[1:] - ltw[:-1])
    lta_bar = lta[:-1] + f * (lta[1:] - lta[:-1])
    l_inv_P = -lP[:, None]                # log(1/P)
    w_cond = ltw_bar < l_inv_P            # t_wave  < 1/nu_0
    a_cond = l_inv_P < lta_bar            # 1/nu_0 < t_align

    has = crossing.any(axis=1)
    w_ok = (crossing & w_cond).any(axis=1)
    a_ok = (crossing & a_cond).any(axis=1)
    valid = (crossing & w_cond & a_cond).any(axis=1)

    # no crossing: d has constant sign; d > 0 means g > P for all r_out
    codes[~has & (d[:, 0] > 0)] = NOCROSS_LO
    codes[has] = BOTH
    codes[has & ~w_ok & a_ok] = T_WAVE
    codes[has & w_ok & ~a_ok] = T_ALIGN
    codes[valid] = VALID
    return codes


def valid_on_P(curve, P, k_w=1.0, k_a=1.0):
    """Boolean array (len(P),): validity for one spin at every P = M*nu_0
    (also imported by pif_timescales)."""
    return classify_on_P(curve, P, k_w, k_a) == VALID


def extreme_spins(spins, valid):
    """
    spins ascending (n_a,), valid (n_a, N_P).
    Returns (a_min_pos, a_max_neg, a_max_pos, a_min_neg), each (N_P,)
    with NaN where no valid spin of that sign exists.
    """
    extremes = []
    for sel, pick_first in (
        (spins > 0, True),
        (spins < 0, False),
        (spins > 0, False),
        (spins < 0, True),
    ):
        s = spins[sel]
        v = valid[sel]
        has = v.any(axis=0)
        idx = v.argmax(axis=0) if pick_first else v.shape[0] - 1 - v[::-1].argmax(axis=0)
        a = np.full(valid.shape[1], np.nan)
        a[has] = s[idx[has]]
        extremes.append(a)
    return tuple(extremes)


def panel_category(codes):
    """
    codes (n_spins_of_one_sign, N_P) -> category (N_P,) int8, see the module
    docstring for the rule.
    """
    cnt = np.stack([(codes == k).sum(axis=0) for k in range(1, 6)])
    timescale = cnt[2:]                   # codes 3, 4, 5
    cat = np.where(timescale.sum(axis=0) > 0,
                   T_WAVE + timescale.argmax(axis=0),
                   NOCROSS_HI + cnt[:2].argmax(axis=0))
    cat[(codes == VALID).any(axis=0)] = VALID
    return cat.astype(np.int8)


def compute_map(alpha=None, hr_isco=None):
    hr = HR_ISCO_DEFAULT if hr_isco is None else hr_isco
    al = ALPHA if alpha is None else alpha
    k_w = HR_ISCO_DEFAULT / hr
    k_a = (hr / HR_ISCO_DEFAULT) ** 2 * (ALPHA / al)

    spins = make_spins()
    curves = build_spin_curves(spins)
    print(f'{spins.size} spins, geometric curves computed for {len(curves)}')

    logP_lo, logP_hi = np.log10(M_MIN * NU_MIN), np.log10(M_MAX * NU_MAX)
    P_grid = np.logspace(logP_lo, logP_hi, N_P)
    codes = np.full((spins.size, N_P), NOCROSS_HI, dtype=np.int8)
    for i, spin in enumerate(spins):
        c = curves.get(float(spin))
        if c is not None:
            codes[i] = classify_on_P(c, P_grid, k_w, k_a)
    valid = codes == VALID
    a_min_pos_P, a_max_neg_P, a_max_pos_P, a_min_neg_P = extreme_spins(
        spins, valid
    )
    cat_pos_P = panel_category(codes[spins > 0])
    cat_neg_P = panel_category(codes[spins < 0])

    # map on the (nu, M) grid: pixel (j, i) -> P-node i + j (exact lattice)
    M = np.logspace(np.log10(M_MIN), np.log10(M_MAX), N_M)
    nu = np.logspace(np.log10(NU_MIN), np.log10(NU_MAX), N_NU)
    idx = np.add.outer(np.arange(N_NU), np.arange(N_M))   # exact node i + j

    np.savez_compressed(
        DATA_FILE, spins=spins, P_grid=P_grid, codes_P=codes, valid_P=valid,
        a_min_pos_P=a_min_pos_P, a_max_neg_P=a_max_neg_P,
        a_max_pos_P=a_max_pos_P, a_min_neg_P=a_min_neg_P,
        cat_pos_P=cat_pos_P, cat_neg_P=cat_neg_P,
        M=M, nu=nu,
        a_min_pos=a_min_pos_P[idx], a_max_neg=a_max_neg_P[idx],
        a_max_pos=a_max_pos_P[idx], a_min_neg=a_min_neg_P[idx],
        cat_pos=cat_pos_P[idx], cat_neg=cat_neg_P[idx],
        r_out_max=R_OUT_GRID[-1], alpha=al, hr_isco=hr)
    print(f'Saved: {DATA_FILE}')
    return (
        M, nu, a_min_pos_P[idx], a_max_neg_P[idx],
        a_max_pos_P[idx], a_min_neg_P[idx], cat_pos_P[idx], cat_neg_P[idx],
    )


def plot_map(M, nu, a_min_pos, a_max_neg, cat_pos=None, cat_neg=None,
             overlay_sources=False, fig_file=FIG_FILE,
             a_max_pos=None, a_min_neg=None):
    set_style()
    if (a_max_pos is None) != (a_min_neg is None):
        raise ValueError('Pass both a_max_pos and a_min_neg, or neither')

    has_new_extrema = a_max_pos is not None
    n_rows = 2 if has_new_extrema else 1
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(7.0, 3.2 * n_rows),
        sharex=True, sharey=True, squeeze=False, constrained_layout=True,
    )
    axes = axes.ravel()
    if overlay_sources:
        source_cmap = plt.get_cmap('rainbow', max(1, len(CATALOG)))
        source_colors = [source_cmap(i) for i in range(len(CATALOG))]
        source_handles = [
            Line2D(
                [], [], marker='o', linestyle='None', color=color,
                markerfacecolor=color, markersize=2, label=source['name'],
            )
            for source, color in zip(CATALOG, source_colors)
        ]
    fail_cmap = ListedColormap([FAIL_COLORS[k] for k in range(1, 6)])
    fail_norm = BoundaryNorm(np.arange(0.5, 6.5, 1.0), fail_cmap.N)
    present = set()
    panels = [
        (axes[0], a_min_pos, cat_pos, (0.0, 1.0),
         'Minimum positive'),
    ]
    if has_new_extrema:
        panels.extend([
            (axes[1], a_max_pos, cat_pos, (0.0, 1.0),
             'Maximum positive'),
            (axes[2], a_max_neg, cat_neg, (-1.0, 0.0),
             'Maximum negative'),
            (axes[3], a_min_neg, cat_neg, (-1.0, 0.0),
             'Minimum negative'),
        ])
    else:
        panels.append(
            (axes[1], a_max_neg, cat_neg, (-1.0, 0.0),
             'Maximum negative')
        )
    row_mappables = [None] * n_rows
    panel_mappables = []
    for panel_index, (ax, Z, cat, (vmin, vmax), title) in enumerate(panels):
        row = panel_index // 2
        cmap_name = 'viridis_r' if row == 0 else 'viridis'
        cmap = plt.get_cmap(cmap_name).copy()
        if cat is None:
            cmap.set_bad('0.85')                   # no valid spin
        else:                                      # failure layer below
            ax.pcolormesh(M, nu, np.ma.masked_where(cat < 1, cat),
                          cmap=fail_cmap, norm=fail_norm, shading='nearest',
                          rasterized=True)
            present.update(int(k) for k in np.unique(cat) if k > 0)
        im = ax.pcolormesh(M, nu, np.ma.masked_invalid(Z), cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='nearest',
                           rasterized=True)
        panel_mappables.append(im)
        if row_mappables[row] is None:
            row_mappables[row] = im
        ax.set_xscale('log'); ax.set_yscale('log')
        if row == n_rows - 1:
            ax.set_xlabel(r'$M\;[M_\odot]$')
        ax.text(
            0.98, 0.98, title, transform=ax.transAxes,
            ha='right', va='top', fontsize=8,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.75},
        )
        fix_spines(ax)
        if panel_index % 2 == 0:
            ax.set_ylabel(r'$\nu_0\;[\mathrm{Hz}]$')
        if overlay_sources:
            for s, color in zip(CATALOG, source_colors):
                lo, hi = s['mass_range']
                ax.plot([lo, hi], [s['nu0']] * 2, color=color, lw=0.8)
                ax.plot(
                    [np.sqrt(lo * hi)], [s['nu0']], marker='o',
                    color=color, markersize=2, linestyle='None',
                )

    if has_new_extrema:
        for row, mappable in enumerate(row_mappables):
            fig.colorbar(
                mappable, ax=axes[row * 2:(row + 1) * 2], label=r'spin $a$', pad=0.01
            )
    else:
        for ax, mappable in zip(axes, panel_mappables):
            fig.colorbar(mappable, ax=ax, label=r'spin $a$')

    # legends stacked below the figure (failure types, then sources)
    y = 0.0
    legends = []
    if present:
        legends.append(dict(
            handles=[Patch(facecolor=FAIL_COLORS[k], edgecolor='k',
                           linewidth=0.3, label=FAIL_LABELS[k])
                     for k in sorted(present)],
            ncol=min(3, len(present))))
    if overlay_sources and source_handles:
        legends.append(dict(handles=source_handles,
                            ncol=min(6, len(source_handles))))
    renderer = fig.canvas.get_renderer()
    for kw in legends:
        leg = fig.legend(loc='upper center', bbox_to_anchor=(0.5, y),
                         fontsize=6, frameon=True, **kw)
        bb = leg.get_window_extent(renderer).transformed(
            fig.transFigure.inverted())
        y = bb.y0 - 0.01
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_file}')
    return fig


def plot_saved_map(data_file=DATA_FILE, overlay_sources=False, fig_file=FIG_FILE):
    """Plot a previously computed map from its saved NPZ archive."""
    with np.load(data_file) as data:
        M = data['M']
        nu = data['nu']
        a_min_pos = data['a_min_pos']
        a_max_neg = data['a_max_neg']
        missing_extrema = {'a_max_pos', 'a_min_neg'} - set(data.files)
        if missing_extrema:
            missing = ', '.join(sorted(missing_extrema))
            raise ValueError(
                f'{data_file} is missing {missing}; recompute the map first'
            )
        a_max_pos = data['a_max_pos']
        a_min_neg = data['a_min_neg']
        cat_pos = data['cat_pos'] if 'cat_pos' in data.files else None
        cat_neg = data['cat_neg'] if 'cat_neg' in data.files else None
    return plot_map(
        M, nu, a_min_pos, a_max_neg, cat_pos, cat_neg,
        overlay_sources, fig_file, a_max_pos=a_max_pos, a_min_neg=a_min_neg,
    )


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    #"""
    M, nu, a_min_pos, a_max_neg, a_max_pos, a_min_neg, cat_pos, cat_neg = (
        compute_map()
    )
    plot_map(
        M, nu, a_min_pos, a_max_neg, cat_pos, cat_neg, overlay_sources=True,
        a_max_pos=a_max_pos, a_min_neg=a_min_neg,
    )
    #"""
    #plot_saved_map(overlay_sources=True)