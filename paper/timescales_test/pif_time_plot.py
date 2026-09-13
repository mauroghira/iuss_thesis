"""Plot radius-timescale figures from the saved PIF analysis archive."""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    r_precession = match['r_precession']
    timescale_matches = [
        radii[0] for radii in (match['r_wave'], match['r_align']) if radii
    ]
    is_ok = (
        not timescale_matches or
        (bool(r_precession) and r_precession[0] < min(timescale_matches))
    )
    marker_face = color if is_ok else 'none'
    for radius in r_precession:
        ax.plot(
            radius, 1.0 / nu_target, marker='o', ms=4,
            markerfacecolor=marker_face, markeredgecolor=color,
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
        ax.set_ylabel('Tempo [s]')
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
        'generic': 'M generica AGN (nessuna stima indipendente)',
        'point': 'M stima puntuale',
        'band': 'M da banda riportata in tesi',
        'exploratory': r'M stima puntuale $\pm1$ dex (esplorativo)',
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
               linestyle='None', label=r'$\nu_{\rm solid}(\bar{r}) = \nu_0$, $\bar{r} < r_{\rm wave}, r_{\rm align}$'),
        Line2D([], [], color='black', marker='o', ms=4,
               markerfacecolor='none', markeredgecolor='black',
               linestyle='None', label=r'$\nu_{\rm solid}(\bar{r}) = \nu_0$, $\bar{r} > r_{\rm wave}, r_{\rm align}$'),
    ]
    fig.suptitle(fr"{name}  —  $\nu_0 = {nu0:.2e}$ Hz", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=2,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, 0.3))

    output = os.path.join(outdir, f'pif_timescales_{safe_filename(name)}.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


def plot_catalog(data_file=DATA_FILE, outdir='.'):
    analyses, _, _ = load_analysis_data(data_file)
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for analysis in analyses:
        output = plot_source_timescales(analysis, outdir)
        print(f'Salvato: {output}')
        saved.append(output)
    return saved


if __name__ == '__main__':
    plot_catalog(outdir='output_catalog')
