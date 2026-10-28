"""Create summary table and spin-mass heatmap from saved PIF data."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from pif_timescales import A_FIXED, DATA_FILE, match_is_ok
from paper.pif_plots import load_analysis_data
from setup import fix_spines, set_style

set_style()


def _format_radii(radii):
    if not radii:
        return '--'
    return ', '.join(f'{radius:.2f}' for radius in radii)


def _format_ok_ranges(matches):
    if not matches:
        return '--'

    values = np.array([match['parameter'] for match in matches])
    ok = np.array([match_is_ok(match) for match in matches])
    if not np.any(ok):
        return '--'

    transitions = np.diff(ok.astype(int))
    starts = np.flatnonzero(np.r_[ok[0], transitions == 1])
    ends = np.flatnonzero(np.r_[transitions == -1, ok[-1]])
    ranges = []
    for start, end in zip(starts, ends):
        first, last = values[start], values[end]
        if matches[0]['parameter_kind'] == 'a':
            ranges.append(
                f'a={first:.3f}' if start == end
                else f'a in [{first:.3f}, {last:.3f}]'
            )
        else:
            ranges.append(
                f'M={first:.2e}' if start == end
                else f'M in [{first:.2e}, {last:.2e}]'
            )
    return '; '.join(ranges)


def plot_catalog_summary(analyses, outdir='.'):
    rows = []
    for analysis in analyses:
        source = analysis['source']
        matches1 = analysis['summary_spin_matches']
        matches2 = analysis['summary_mass_matches']
        matches = matches1 + matches2
        M_ref = analysis['M_ref']
        ok_matches = [match for match in matches if match_is_ok(match)]
        spin_ranges = _format_ok_ranges(matches1)
        mass_ranges = _format_ok_ranges(matches2)
        spin_ranges = f'{spin_ranges} (M_fixed={M_ref:.2e} Msun)'
        rows.append([
            source['name'],
            'YES' if ok_matches else 'NO',
            f'{len(ok_matches)}/{len(matches)}',
            spin_ranges,
            mass_ranges,
        ])
        print(
            f"{source['name']}: spins={spin_ranges}; "
            f"masses={mass_ranges}"
        )

    fig = plt.figure(figsize=(14, max(4.0, 0.42 * (len(rows) + 2))))
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.90])
    ax.axis('off')
    ax.set_title('Catalog summary: conditions for solid precession', pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=[
            'source', 'any OK', 'valid matches',
            'OK spin ranges (M_fixed)',
            fr'OK mass ranges (a_fixed={A_FIXED:.2f})',
        ],
        cellLoc='center', colLoc='center', bbox=[0, 0, 1, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(col=list(range(5)))
    output = os.path.join(outdir, 'pif_timescales_catalog_summary.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


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

    """
    fig.suptitle(
        'Condition for solid-body precession across the spin-mass grid '
        r'(green = OK: $t_{\rm wave} < 1/\nu_0 < t_{\rm align}$ at $\bar{r}_{\rm out}$)',
        fontsize=12,
    )
    """
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    output = os.path.join(outdir, 'pif_timescales_parameter_grids.pdf')
    plt.savefig(output, bbox_inches='tight')
    plt.close(fig)
    return output


if __name__ == '__main__':
    os.makedirs('output_catalog', exist_ok=True)
    analyses, _, _ = load_analysis_data(DATA_FILE)
    #summary = plot_catalog_summary(analyses, 'output_catalog')
    grids = plot_catalog_parameter_grids(analyses, 'output_catalog')
    #print(f'Saved: {summary}')
    print(f'Saved: {grids}')