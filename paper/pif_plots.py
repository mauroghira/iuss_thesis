"""Plot radius-timescale figures from the saved PIF analysis archive."""

import os
import sys

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap

from setup import fix_spines, set_style
from pif_timescales import A_FIXED, DATA_FILE

set_style()


def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def load_analysis_data(data_file=DATA_FILE):
    with np.load(data_file, allow_pickle=True) as data:
        analyses = data['analyses'].tolist()
        r_out_grid = data['r_out_grid']
        a_fixed = float(data['a_fixed'])
    return analyses, r_out_grid, a_fixed


def _plot_precession_markers(ax, match, nu_target, color):
    """One marker per precession radius r_bar: filled if
    t_wave(r_bar) < 1/nu_0 < t_align(r_bar) (flags computed in
    pif_timescales), open otherwise."""
    for radius, is_ok in zip(match['r_precession'], match['precession_ok']):
        ax.plot(
            radius, 1.0 / nu_target, marker='o', ms=4,
            markerfacecolor=color if is_ok else 'none',
            markeredgecolor=color,
            markeredgewidth=0.7, linestyle='None', zorder=5,
        )


def _draw_timescale_panel(ax, matches, nu_target, colors, r_out_grid,
                          parameter_kind):
    for match, color in zip(matches, colors):
        ax.plot(match['r_out'], match['t_wave'], color=color, lw=1, ls=':')
        ax.plot(
            match['r_out'], match['t_align'], color=color, lw=1, ls='--',
            label=match['label'],
        )
        _plot_precession_markers(ax, match, nu_target, color)

    ax.axhline(1.0 / nu_target, color='black',  linestyle=(0, (20, 10)), lw=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, r_out_grid[-1])
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    if parameter_kind == 'a':
        ax.set_ylabel('Time [s]')
    ax.legend(loc='upper right', frameon=True, fontsize=6.5, ncol=2)


def plot_source_timescales(analysis, outdir='.'):
    source = analysis['source']
    nu0 = source['nu0']
    name = source['name']
    M_ref = analysis['M_ref']
    M_lo = analysis['M_lo']
    M_hi = analysis['M_hi']
    kind = analysis['kind']
    r_out_grid = analysis['panel1_matches'][0]['r_out']

    kind_label = {
        'generic': 'generic AGN mass (no independent estimate)',
        'point': 'point-like mass estimate',
        'band': 'mass range reported in the thesis',
        'exploratory': r'point-like mass estimate $\pm1$ dex (exploratory)',
    }[kind]

    fig = plt.figure(figsize=(10.5, 8.5))
    ax1 = fig.add_axes([0.08, 0.39, 0.40, 0.54])
    ax2 = fig.add_axes([0.56, 0.39, 0.40, 0.54], sharey=ax1)
    fix_spines(ax1)
    fix_spines(ax2)

    matches1 = analysis['panel1_matches']
    colors_a = plt.cm.viridis(np.linspace(0, 1, len(matches1)))
    _draw_timescale_panel(ax1, matches1, nu0, colors_a, r_out_grid, 'a')
    ax1.set_title(
        fr"$M_{{\rm ref}} = {M_ref:.2e}\,M_\odot$ ",
        fontsize=8,
    )

    matches2 = analysis['panel2_matches']
    colors_m = plt.cm.plasma(np.linspace(0, 1, len(matches2)))
    _draw_timescale_panel(ax2, matches2, nu0, colors_m, r_out_grid, 'M')
    ax2.set_title(
        fr"$a={A_FIXED:.2f}$, "
        fr"$M \in [{M_lo:.1e}, {M_hi:.1e}]\,M_\odot$",
        fontsize=8,
    )

    style_handles = [
        Line2D([], [], color='gray', lw=1, ls=':', label=r'$t_{\rm wave}$'),
        Line2D([], [], color='gray', lw=1, ls='--', label=r'$t_{\rm align}$'),
        Line2D([], [], color='black', lw=0.5, linestyle=(0, (20, 10)), label=r'$P_{\rm osc}=1/\nu_0$'),
        Line2D([], [], color='black', marker='o', ms=4,
               markerfacecolor='black', markeredgecolor='black',
               linestyle='None',
               label=r'$\nu_{\rm solid}(\bar{r}) = \nu_0$, $t_{\rm wave}(\bar{r}) < 1/\nu_0 < t_{\rm align}(\bar{r})$'),
        Line2D([], [], color='black', marker='o', ms=4,
               markerfacecolor='none', markeredgecolor='black',
               linestyle='None',
               label=r'$\nu_{\rm solid}(\bar{r}) = \nu_0$, condition not satisfied'),
    ]
    fig.suptitle(fr"{name}  —  $\nu_0 = {nu0:.2e}$ Hz", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=2,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, 0.3))

    output = os.path.join(outdir, f'pif_timescales_{safe_filename(name)}.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


############################################################

def _cell_edges(values, logarithmic=False):
    edges = np.empty(values.size + 1)
    if logarithmic:
        edges[1:-1] = np.sqrt(values[:-1] * values[1:])
        edges[0] = values[0]**2 / edges[1]
        edges[-1] = values[-1]**2 / edges[-2]
    else:
        edges[1:-1] = 0.5 * (values[:-1] + values[1:])
        edges[0] = values[0] - 0.5 * (values[1] - values[0])
        edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges

def plot_catalog_parameter_grids(analyses, outdir='.'):
    if not analyses:
        return None

    n_columns = 4
    n_rows = int(np.ceil(len(analyses) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(4.0 * n_columns, 3.8 * n_rows),
        squeeze=False, sharex=True
    )
    axes = axes.ravel()
    cmap = ListedColormap(['green'])
    cmap.set_bad(alpha=0)

    for index, (axis, analysis) in enumerate(zip(axes, analyses)):
        row, column = divmod(index, n_columns)
        spins = analysis['grid_spins']
        masses = analysis['grid_masses']
        data = analysis['grid_ok'].astype(float)
        data[data == 0] = float('nan')
        axis.pcolormesh(
            _cell_edges(spins), _cell_edges(masses, logarithmic=True),
            data, cmap=cmap,
            shading='flat', rasterized=True,
        )
        axis.set_xlim(-1.02, 1.02)
        axis.set_yscale('log')
        if row == n_rows - 1:
            axis.set_xlabel(r'Spin $a$')
        if column == 0:
            axis.set_ylabel(r'Mass [$M_\odot$]')
        """
        if column in :
            axis.tick_params(axis='y', direction='in', pad=-28)
        elif column == n_columns - 1:
            axis.yaxis.tick_right()
            axis.tick_params(axis='y', labelleft=False, labelright=True)
        """
        axis.text(
            0.03, 0.97, analysis['source']['name'],
            transform=axis.transAxes, ha='left', va='top', fontsize=9,
            bbox={
                'facecolor': 'white', 'edgecolor': 'none',
                'alpha': 0.75, 'pad': 1.5,
            },
        )
        fix_spines(axis)

    for axis in axes[len(analyses):]:
        axis.set_visible(False)

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    output = os.path.join(outdir, 'pif_timescales_parameter_grids.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output

###################################################à


if __name__ == '__main__':
    analyses, _, _ = load_analysis_data(DATA_FILE)
    outdir = "output_catalog"
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for analysis in analyses:
        output = plot_source_timescales(analysis, outdir)
        print(f'Saved: {output}')
        saved.append(output)
    grids = plot_catalog_parameter_grids(analyses, 'output_catalog')
    print(f'Saved: {grids}')