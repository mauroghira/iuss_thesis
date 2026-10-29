"""
pif_frq_all.py

(a, R_out) panels for the nu_solid channel, same Region A/B logic as
projection_plots.py (area accessible to a generic AGN vs area allowed by the
known mass; their intersection is the coloured area), with at most
MAX_PER_PANEL sources per panel.

Fully self-contained with respect to rpm_v2.py (no import, no side effect on
its output). The area computation and the ordering of sources by decreasing
area are re-implemented here because they are specific to the nu_solid
functional (area_aR of rpm_v2.py is written for the local channels
g_LT/g_per/g_phi).

nu_solid * M = g(a, R_in, R_out) is computed by pif_core.g_solid, which is
independent of the sources and of the mass: a source with frequency nu0 and
mass M only selects the level g = M * nu0.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from setup import (
    fix_spines, set_style, A_THORNE, R_MAX, M_AGN_MIN, M_AGN_MAX, r_isco,
)
from catalog import CATALOG, select_sources
from pif_core import P_TORQUE, g_solid

set_style()

MAX_PER_PANEL = 6
DEFAULT_OUTDIR = os.path.join(os.path.dirname(__file__), 'output_catalog')


def _area_from_mask_2d(A, R, mask, log_x=False, log_y=False):
    """
    Approximate the area of a 2-D Boolean region in plotted coordinates.
    Used for (a,R), where contourf operates directly on the mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return 0.0

    xp = np.log(A[:, 0]) if log_x else A[:, 0]
    yp = np.log(R[0, :]) if log_y else R[0, :]

    dx = np.abs(np.diff(xp))
    dy = np.abs(np.diff(yp))

    cells = (
        mask[:-1, :-1]
        & mask[1:, :-1]
        & mask[:-1, 1:]
        & mask[1:, 1:]
    )

    return float(np.sum(cells * dx[:, None] * dy[None, :]))


def build_solid_channel_grid(a_vals, r_out_vals, p=P_TORQUE, n_rad=300):
    """
    Grid container for the nu_solid channel. It is independent of the
    sources and is computed once, before drawing their mass-range boundaries.
    G = nu_solid * M [Hz Msun]; NaN where R_out <= r_ISCO(a), where the
    integral ratio is undefined (zero-width interval).
    """
    A, R_out = np.meshgrid(a_vals, r_out_vals, indexing="ij")
    G = g_solid(A, R_out, p=p, n_rad=n_rad)
    return SimpleNamespace(a_vals=a_vals, r_vals=r_out_vals, A=A, R=R_out, G=G)


def _mass_levels(source):
    """Levels of G = M * nu0 delimiting the source's accessible region.
    Returns (levels, is_band): a single level for a point-like mass estimate."""
    nu0 = source['nu0']
    mass_range = source['mass_range']
    if mass_range is None:
        return [M_AGN_MIN * nu0, M_AGN_MAX * nu0], True
    M_lo, M_hi = mass_range
    if M_lo == M_hi:
        return [M_lo * nu0], False
    return [M_lo * nu0, M_hi * nu0], True


def _area_aR_solid(chgrid, source):
    """Area enclosed by the source mass-range boundary in (a, R_out)."""
    levels, is_band = _mass_levels(source)
    if not is_band:
        return 0.0     # point-like estimate: a single contour, no area
    G, A, R = chgrid.G, chgrid.A, chgrid.R
    mask_fill = (G >= levels[0]) & (G <= levels[-1])
    if not mask_fill.any():
        return 0.0
    return _area_from_mask_2d(A, R, mask_fill, log_x=False, log_y=True)


def _plot_mass_range_boundary(ax, chgrid, source, color, alpha=0.35):
    """Draw only the accessible region for the source mass prescription."""
    G, A, R = chgrid.G, chgrid.A, chgrid.R
    levels, is_band = _mass_levels(source)

    ax.contour(A, R, G, levels=levels, colors=[color], linewidths=1)
    if is_band:
        mask_fill = (G >= levels[0]) & (G <= levels[-1])
        if mask_fill.any():
            ax.contourf(
                A, R, mask_fill.astype(float), levels=[0.5, 1.5],
                colors=[color], alpha=alpha,
            )


def _order_sources_by_area(sources, colors, chgrid, alpha_min=0.2, alpha_max=0.5):
    """Largest area -> smallest alpha (drawn below); smaller areas -> increasing alpha."""
    items = [(_area_aR_solid(chgrid, src), src, color)
             for src, color in zip(sources, colors)]
    items.sort(key=lambda item: item[0], reverse=True)
    n = len(items)
    alphas = np.full(n, alpha_min) if n == 1 else np.linspace(alpha_min, alpha_max, n)
    return [(src, color, float(alpha), area)
            for alpha, (area, src, color) in zip(alphas, items)]


def plot_solid_aR(sources=None, indices=None, outdir=DEFAULT_OUTDIR,
                  max_per_panel=MAX_PER_PANEL,
                  n_a=201, n_rout=200, n_rad=300):
    if sources is not None and indices is not None:
        raise ValueError('Pass either sources or indices, not both')
    if sources is None:
        sources = select_sources(indices) if indices is not None else CATALOG

    os.makedirs(outdir, exist_ok=True)

    a_vals = np.linspace(-A_THORNE, A_THORNE, n_a)
    r_out_vals = np.logspace(0, np.log10(R_MAX), n_rout)
    chgrid = build_solid_channel_grid(a_vals, r_out_vals, p=P_TORQUE, n_rad=n_rad)

    n_panels = max(1, int(np.ceil(len(sources) / max_per_panel)))
    width = 4 if n_panels == 1 else 7
    cmap = plt.get_cmap('rainbow', max(1, len(sources)))
    source_colors = [cmap(i) for i in range(len(sources))]
    fig, axes = plt.subplots(
        1, n_panels, squeeze=False, sharey=True,
        figsize=(width, 4),
    )
    axes = axes.ravel()
    for panel_idx in range(n_panels):
        start = panel_idx * max_per_panel
        end = min(start + max_per_panel, len(sources))
        chunk = sources[start:end]
        colors = source_colors[start:end]
        ax = axes[panel_idx]
        fix_spines(ax)

        ordered = _order_sources_by_area(chunk, colors, chgrid,
                                         alpha_min=0.2, alpha_max=0.5)
        for src, color, alpha, area in ordered:
            _plot_mass_range_boundary(ax, chgrid, src, color, alpha)

        ax.plot(
            a_vals, r_isco(a_vals), color='purple', linestyle='--',
            linewidth=1.5,
        )
        ax.annotate(
            'ISCO', xy=(0, r_isco(0)), xytext=(0, -8),
            textcoords='offset points', ha='center', va='top',
            color='purple', fontsize=8,
        )
        ax.set_xlabel(r"Spin $a$")
        if panel_idx == 0:
            ax.set_ylabel(r"$R_{\rm out}$ [$R_g$]")
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.set_yscale('log')

    if len(sources) > 0:
        legend_handles = [
            Line2D([0], [0], color=color, lw=4)
            for color in source_colors
        ]
        fig.legend(
            legend_handles, [source['name'] for source in sources],
            loc='upper center', bbox_to_anchor=(0.5, 1.1),
            ncol=min(4, len(sources)), fontsize=7, framealpha=0.85,
        )
        fig.tight_layout()
    else:
        fig.tight_layout()
    output = os.path.join(outdir, 'pif_solid_aR.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return [output]


if __name__ == '__main__':
    plot_solid_aR()#indices=[0, 9, 10, 11])