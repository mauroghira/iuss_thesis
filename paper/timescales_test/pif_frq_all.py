"""
pif_solid_aR.py

Pannelli (a, R_out) per il canale nu_solid, stessa logica Regione A/B
di projection_plots.py (area accessibile AGN generico vs massa nota,
intersezione = area colorata), con split a max MAX_PER_PANEL sorgenti
per pannello.

File completamente autonomo: nessuna dipendenza da rpm_v2.py (ne'
import, ne' side effect sul suo output). L'unica logica condivisa con
rpm_v2.py (calcolo area, ordinamento sorgenti per alpha decrescente) e'
reimplementata qui perche' specifica al canale nu_solid (area_aR di
rpm_v2.py e' scritta per canali locali g_LT/g_per/g_phi, non per il
funzionale nu_solid).
"""

import os
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import r_isco, fix_spines, set_style, A_THORNE, R_MAX, M_AGN_MIN, M_AGN_MAX
from catalog import CATALOG, select_sources
from nu_solid_v2 import nu_solid_vect_p
from disk_profiles import P_TORQUE
from projection_plots import _area_from_mask_2d

set_style()

MAX_PER_PANEL = 4
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), 'output_catalog')


def build_solid_channel_grid(a_vals, r_out_vals, p=P_TORQUE, n_rad=300):
    """
    Grid container for the nu_solid channel. It is independent of the
    sources and is computed once before drawing their mass-range boundaries.
    """
    A, R_out = np.meshgrid(a_vals, r_out_vals, indexing="ij")
    R_in = r_isco(A)
    valid = R_out > R_in
    # Do not evaluate the zero-width interval at R_out=r_ISCO: nu_solid is
    # an integral ratio and is undefined when numerator and denominator vanish.
    R_out_eff = np.where(valid, R_out, np.nextafter(R_in, np.inf))

    with np.errstate(invalid="ignore", divide="ignore"):
        G = nu_solid_vect_p(A, R_in, R_out_eff, M=np.ones_like(A),
                             p=p, n_rad=n_rad)
    G = np.where(valid, G, np.nan)

    return SimpleNamespace(a_vals=a_vals, r_vals=r_out_vals, A=A, R=R_out, G=G)


def _area_aR_solid(chgrid, nu0, mass_range):
    """Area enclosed by the source mass-range boundary in (a, R_out)."""
    G, A, R = chgrid.G, chgrid.A, chgrid.R
    if mass_range is None:
        levels = [M_AGN_MIN * nu0, M_AGN_MAX * nu0]
    else:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            return 0.0
        levels = [M_lo * nu0, M_hi * nu0]

    mask_fill = (G >= levels[0]) & (G <= levels[-1])

    if not mask_fill.any():
        return 0.0
    return _area_from_mask_2d(A, R, mask_fill, log_x=False, log_y=True)


def _plot_mass_range_boundary(ax, chgrid, source, color, alpha=0.35):
    """Draw only the accessible region for the source mass prescription."""
    G, A, R = chgrid.G, chgrid.A, chgrid.R
    nu0 = source['nu0']
    mass_range = source['mass_range']

    if mass_range is None:
        levels = [M_AGN_MIN * nu0, M_AGN_MAX * nu0]
        mask_fill = (G >= levels[0]) & (G <= levels[1])
    else:
        M_lo, M_hi = mass_range
        levels = [M_lo * nu0] if M_lo == M_hi else [M_lo * nu0, M_hi * nu0]
        mask_fill = (
            np.zeros_like(G, dtype=bool) if M_lo == M_hi
            else (G >= levels[0]) & (G <= levels[1])
        )

    ax.contour(A, R, G, levels=levels, colors=[color], linewidths=1)
    if mask_fill.any():
        ax.contourf(
            A, R, mask_fill.astype(float), levels=[0.5, 1.5],
            colors=[color], alpha=alpha,
        )
    ax.plot([], [], color=color, lw=4, label=source['name'])


def _order_sources_by_area(sources, colors, chgrid, alpha_min=0.2, alpha_max=0.5):
    """Area maggiore -> alpha piccolo (disegnata sotto); aree minori -> alpha crescente."""
    items = [
        (_area_aR_solid(chgrid, src["nu0"], src["mass_range"]), src, color)
        for src, color in zip(sources, colors)
    ]
    items.sort(key=lambda item: item[0], reverse=True)
    n = len(items)
    alphas = np.full(n, alpha_min) if n == 1 else np.linspace(alpha_min, alpha_max, n)
    return [(src, color, float(alpha), area)
            for alpha, (area, src, color) in zip(alphas, items)]


def plot_solid_aR(sources=None, indices=None, outdir=DEFAULT_OUTDIR,
                   max_per_panel=MAX_PER_PANEL,
                   n_a=80, n_rout=150, n_rad=300):
    if sources is not None and indices is not None:
        raise ValueError('Pass either sources or indices, not both')
    if sources is None:
        sources = select_sources(indices) if indices is not None else CATALOG

    os.makedirs(outdir, exist_ok=True)

    a_vals = np.linspace(-A_THORNE, A_THORNE, n_a)
    r_out_vals = np.logspace(0, np.log10(R_MAX), n_rout)
    chgrid = build_solid_channel_grid(a_vals, r_out_vals, p=P_TORQUE, n_rad=n_rad)

    n_panels = max(1, int(np.ceil(len(sources) / max_per_panel)))
    cmap = plt.cm.tab20
    fig, axes = plt.subplots(
        1, n_panels, squeeze=False,
        figsize=(4.5 * n_panels, 4.2),
    )
    axes = axes.ravel()

    for panel_idx in range(n_panels):
        chunk = sources[panel_idx * max_per_panel:(panel_idx + 1) * max_per_panel]
        colors = [cmap(i) for i in range(len(chunk))]
        ax = axes[panel_idx]
        fix_spines(ax)

        ordered = _order_sources_by_area(chunk, colors, chgrid,
                                          alpha_min=0.2, alpha_max=0.5)
        for src, color, alpha, area in ordered:
            _plot_mass_range_boundary(ax, chgrid, src, color, alpha)

        ax.set_ylabel(r"$\nu_{\rm solid}$: $R_{\rm out}$ [$R_g$]")
        ax.set_yscale('log')
        ax.legend(loc="best", fontsize=7, framealpha=0.85)
        ax.set_title(
            f'Sources {panel_idx * max_per_panel + 1}-'
            f'{min((panel_idx + 1) * max_per_panel, len(sources))}',
            fontsize=9,
        )

    fig.tight_layout()
    output = os.path.join(outdir, 'pif_solid_aR.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return [output]


if __name__ == '__main__':
    plot_solid_aR()#(indices=[0, 9, 10, 11])