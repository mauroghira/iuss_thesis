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

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


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

    fig, axs = plt.subplots(1,2, figsize=(7, 3.4), sharey=True, constrained_layout=True)
    ax1 = axs[0]
    ax2 = axs[1]
    fix_spines(ax1)
    fix_spines(ax2)

    matches1 = analysis['panel1_matches']
    colors_a = plt.cm.viridis(np.linspace(0, 1, len(matches1)))
    _draw_timescale_panel(ax1, matches1, nu0, colors_a, r_out_grid, 'a')
    ax1.text(
        0.98, 0.02, fr"$M_{{\rm ref}} = {M_ref:.2e}\,M_\odot$",
        transform=ax1.transAxes, ha='right', va='bottom', fontsize=8,
    )

    matches2 = analysis['panel2_matches']
    colors_m = plt.cm.plasma(np.linspace(0, 1, len(matches2)))
    _draw_timescale_panel(ax2, matches2, nu0, colors_m, r_out_grid, 'M')
    ax2.text(
        0.98, 0.02, fr"$a={A_FIXED:.2f}$",
        transform=ax2.transAxes, ha='right', va='bottom', fontsize=8,
    )

    ax1.tick_params(
        axis='y', which='both', left=True, labelleft=True,
        right=True, labelright=False,
    )
    ax2.tick_params(
        axis='y', which='both', left=True, labelleft=False,
        right=True, labelright=False,
    )
    seconds_limits = ax1.get_ylim()
    year_axes = (ax1.twinx(), ax2.twinx())
    for year_axis, show_labels in zip(year_axes, (False, True)):
        year_axis.set_yscale('log')
        year_axis.set_ylim(
            seconds_limits[0] / SECONDS_PER_YEAR,
            seconds_limits[1] / SECONDS_PER_YEAR,
        )
        year_axis.tick_params(
            axis='y', which='both', right=True, labelright=show_labels,
            left=False, labelleft=False,
        )
        year_axis.spines['left'].set_visible(False)
        year_axis.spines['top'].set_visible(False)
        year_axis.spines['bottom'].set_visible(False)
    year_axes[1].set_ylabel('Time [yr]')

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
    #fig.suptitle(fr"{name}  —  $\nu_0 = {nu0:.2e}$ Hz", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=5,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.1))

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
        figsize=(7, 1.7 * n_rows),
        squeeze=False, sharex=True, sharey=True
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
        if column > 0:
            axis.tick_params(axis='y', labelleft=False)
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

    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.07, top=0.96,
        wspace=0.02, hspace=0.02,
    )
    output = os.path.join(outdir, 'pif_timescales_parameter_grids.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output

###################################################à


if __name__ == '__main__':
    analyses, _, _ = load_analysis_data(DATA_FILE)
    outdir = "output_catalog"
    os.makedirs(outdir, exist_ok=True)
    #"""
    saved = []
    for analysis in analyses:
        output = plot_source_timescales(analysis, outdir)
        print(f'Saved: {output}')
        saved.append(output)
    #"""
    #grids = plot_catalog_parameter_grids(analyses, 'output_catalog')
    #print(f'Saved: {grids}')