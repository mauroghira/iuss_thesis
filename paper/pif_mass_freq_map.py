"""
Mass-frequency map of the PIF validity condition.

For every point (M, nu_0) of a grid and every spin a in [-1, 1] the precession
radius r_bar is defined by nu_solid(r_bar) = nu_0, and the configuration is
valid if  t_wave(r_bar) < 1/nu_0 < t_align(r_bar)  (same criterion as
pif_timescales.match_is_ok). The script stores, for every (M, nu_0):

    a_min_pos : smallest valid spin with a > 0   (NaN if none)
    a_max_neg : largest  valid spin with a < 0   (NaN if none)  [closest to 0]

Efficiency: from pif_core, with g = nu_solid*M, tau_w = t_wave/M,
tau_a = t_align/M (functions of (a, r_out) only),

    nu_solid = nu_0   <=>  g(r_out) = P,        P = M * nu_0
    t_wave  < 1/nu_0  <=>  tau_w(r_out) < 1/P
    t_align > 1/nu_0  <=>  tau_a(r_out) > 1/P

so validity depends on (M, nu_0) ONLY through P = M*nu_0. The validity is
therefore computed on a 1D grid of P (per spin, fully vectorised) and then
mapped onto the 2D (M, nu_0) grid: the map is constant along M*nu_0 = const.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import NamedTuple

from setup import r_isco, set_style, fix_spines
from pif_core import ALPHA, HR_ISCO_DEFAULT, pif_geometric, PIFGeometric
from catalog import CATALOG

M_MIN, M_MAX = 1e4, 1e10
NU_MIN, NU_MAX = 1e-7, 1e-2
N_M, N_NU = 400, 350          # resolution of the plotted grid
N_A = 201                     # spins in [-1, 1]; odd -> contains a = 0
N_P = 2000                    # nodes of the 1D grid in P = M * nu_0

N_R_OUT_INNER = 100
N_R_OUT_OUTER = 100
R_OUT_GRID = np.concatenate([
    np.logspace(0, 1, N_R_OUT_INNER, endpoint=False),
    np.logspace(1, np.log10(200.0), N_R_OUT_OUTER),
])
R_IN_MARGIN = 1.001      # only r_out > R_IN_MARGIN * r_in is used

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_catalog')
DATA_FILE = os.path.join(OUT_DIR, 'pif_mass_frequency_map.npz')
FIG_FILE = os.path.join(OUT_DIR, 'pif_mass_frequency_map.pdf')


class SpinCurves(NamedTuple):
    """M-independent curves of one spin on its valid r_out values."""
    r_out: np.ndarray
    geo: PIFGeometric

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


def valid_on_P(curve, P, k_w=1.0, k_a=1.0):
    """
    Boolean array (len(P),): validity for one spin at every P = M*nu_0.
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
    if ok.sum() < 2:                      # e.g. a = 0: g = 0 identically
        return np.zeros(P.size, dtype=bool)
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
    return (crossing & (ltw_bar < l_inv_P) & (l_inv_P < lta_bar)).any(axis=1)


def extreme_spins(spins, valid):
    """
    spins ascending (n_a,), valid (n_a, N_P).
    Returns (a_min_pos, a_max_neg), each (N_P,) with NaN where none valid.
    """
    out = []
    for sel, pick_first in ((spins > 0, True), (spins < 0, False)):
        s, v = spins[sel], valid[sel]
        has = v.any(axis=0)
        idx = v.argmax(axis=0) if pick_first else v.shape[0] - 1 - v[::-1].argmax(axis=0)
        a = np.full(valid.shape[1], np.nan)
        a[has] = s[idx[has]]
        out.append(a)
    return out[0], out[1]


def compute_map(alpha=None, hr_isco=None):
    hr = HR_ISCO_DEFAULT if hr_isco is None else hr_isco
    al = ALPHA if alpha is None else alpha
    k_w = HR_ISCO_DEFAULT / hr
    k_a = (hr / HR_ISCO_DEFAULT) ** 2 * (ALPHA / al)

    spins = np.linspace(-1.0, 1.0, N_A)
    curves = build_spin_curves(spins)
    print(f'Geometric curves computed for {len(curves)} spins')

    logP_lo, logP_hi = np.log10(M_MIN * NU_MIN), np.log10(M_MAX * NU_MAX)
    P_grid = np.logspace(logP_lo, logP_hi, N_P)
    valid = np.zeros((N_A, N_P), dtype=bool)
    for i, spin in enumerate(spins):
        c = curves.get(float(spin))
        if c is not None:
            valid[i] = valid_on_P(c, P_grid, k_w, k_a)
    a_pos_P, a_neg_P = extreme_spins(spins, valid)

    # map on the (nu, M) grid: nearest node in log P
    M = np.logspace(np.log10(M_MIN), np.log10(M_MAX), N_M)
    nu = np.logspace(np.log10(NU_MIN), np.log10(NU_MAX), N_NU)
    logP = np.log10(M)[None, :] + np.log10(nu)[:, None]
    dlogP = (logP_hi - logP_lo) / (N_P - 1)
    idx = np.clip(np.rint((logP - logP_lo) / dlogP).astype(int), 0, N_P - 1)

    np.savez_compressed(
        DATA_FILE, spins=spins, P_grid=P_grid, valid_P=valid,
        a_min_pos_P=a_pos_P, a_max_neg_P=a_neg_P,
        M=M, nu=nu, a_min_pos=a_pos_P[idx], a_max_neg=a_neg_P[idx],
        alpha=al, hr_isco=hr)
    print(f'Saved: {DATA_FILE}')
    return M, nu, a_pos_P[idx], a_neg_P[idx]


def plot_map(M, nu, a_pos, a_neg, overlay_sources=False, fig_file=FIG_FILE):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True,
                             constrained_layout=True)
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
    panels = [
        (axes[0], a_pos, (0.0, 1.0), r'minimum valid spin $a>0$'),
        (axes[1], a_neg, (-1.0, 0.0), r'maximum valid spin $a<0$'),
    ]
    for ax, Z, (vmin, vmax), title in panels:
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('0.85')                       # no valid spin
        im = ax.pcolormesh(M, nu, np.ma.masked_invalid(Z), cmap=cmap,
                           vmin=vmin, vmax=vmax, shading='nearest',
                           rasterized=True)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$M\;[M_\odot]$')
        ax.set_title(title)
        fix_spines(ax)
        fig.colorbar(im, ax=ax, label=r'$a$')
        if overlay_sources:
            for s, color in zip(CATALOG, source_colors):
                lo, hi = s['mass_range']
                ax.plot([lo, hi], [s['nu0']] * 2, color=color, lw=0.8)
                ax.plot(
                    [np.sqrt(lo * hi)], [s['nu0']], marker='o',
                    color=color, markersize=2, linestyle='None',
                )
    if overlay_sources and source_handles:
        fig.legend(
            handles=source_handles, loc='lower center',
            bbox_to_anchor=(0.5, -0.04), ncol=min(6, len(source_handles)),
            fontsize=6, frameon=True,
        )
    axes[0].set_ylabel(r'$\nu_0\;[\mathrm{Hz}]$')
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f'Saved: {fig_file}')
    return fig


def plot_saved_map(data_file=DATA_FILE, overlay_sources=False, fig_file=FIG_FILE):
    """Plot a previously computed map from its saved NPZ archive."""
    with np.load(data_file) as data:
        M = data['M']
        nu = data['nu']
        a_pos = data['a_min_pos']
        a_neg = data['a_max_neg']
    return plot_map(M, nu, a_pos, a_neg, overlay_sources, fig_file)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_map(*compute_map(), overlay_sources=True)
    #plot_saved_map(overlay_sources=True)