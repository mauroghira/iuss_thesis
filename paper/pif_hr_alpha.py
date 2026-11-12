"""
Dependence of the PIF validity on the disc parameters (H/R_isco, alpha).

Why this is cheap: with p, q fixed (see pif_core)

    nu_solid = g / M                                (independent of HR, alpha)
    t_wave   = M tau_w * k_w,   k_w = HR_ref / HR
    t_align  = M tau_a * k_a,   k_a = (HR / HR_ref)^2 * (ALPHA / alpha)

so the geometric curves are computed ONCE (one call to build_spin_curves) and
every (HR, alpha) is an exact rescaling by (k_w, k_a). Moreover validity
depends on (M, nu_0) only through P = M nu_0 (pif_mass_freq_map), so

  * the whole mass-frequency map of one (HR, alpha) is a 1D function of P for
    each spin: it is shown here in the (P, a) plane, one panel per (HR, alpha)
    (Fig. 1, replaces a full 2D (M, nu_0) map per combination);
  * a source occupies the interval [M_lo nu_0, M_hi nu_0] of P, so its
    (M, a) validity grid is a slice of the same plane. Fig. 2 compresses it
    to, for every spin, the fraction of the (log) mass range that is valid,
    with one row per (HR, alpha) and one panel per source;
  * Fig. 3 is the timescale plot of single sources (as pif_plots) for all
    (HR, alpha), at the reference mass.

Mass range of a source: pif_timescales.resolve_mass_reference (generic AGN
range if unknown, +/- EXPLORATORY_DEX around a point estimate).

Usage: python pif_hr_alpha.py           (recomputes if the cache is missing)
       python pif_hr_alpha.py --recompute
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from setup import set_style, fix_spines
from catalog import CATALOG, select_sources
from pif_core import ALPHA, HR_ISCO_DEFAULT
from pif_mass_freq_map import (
    make_spins, build_spin_curves, classify_on_P, VALID, FAIL_COLORS,
    FAIL_LABELS, R_OUT_GRID,
)
from pif_timescales import A_VALS, resolve_mass_reference, _match_from_curves
from pif_plots import _plot_precession_markers, safe_filename

# ---------------------------------------------------------------- settings
HR_VALS = (0.03, 0.1, 0.3)          # columns
ALPHA_VALS = (0.001, 0.01, 0.1)     # rows
COMBOS = [(hr, al) for al in ALPHA_VALS for hr in HR_VALS]

SOURCE_INDICES = [c['idx'] for c in CATALOG]  # Fig. 2: any subset/order
TIMESCALE_SOURCES = [10]                      # Fig. 3: one figure per source

P_MIN, P_MAX, P_PER_DECADE = 1e-4, 1e5, 50    # covers the whole valid window
N_M_SOURCE = 120                              # masses per source range

VALID_COLOR = '#009E73'
OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_hr_alpha')
DATA_FILE = os.path.join(OUT_DIR, 'pif_hr_alpha.npz')


def rescaling(hr, al):
    """(k_w, k_a) multiplying tau_w, tau_a (see the module docstring)."""
    return HR_ISCO_DEFAULT / hr, (hr / HR_ISCO_DEFAULT) ** 2 * (ALPHA / al)


def _codes(curves, spins, P, hr, al):
    """int8 (n_spins, n_P) failure codes (pif_mass_freq_map) at (hr, al)."""
    k_w, k_a = rescaling(hr, al)
    out = np.full((spins.size, P.size), -1, dtype=np.int8)   # -1: no curve
    for i, s in enumerate(spins):
        c = curves.get(float(s))
        if c is not None:
            out[i] = classify_on_P(c, P, k_w, k_a)
    return out


# ------------------------------------------------------------- computation
def compute(output_file=DATA_FILE):
    spins = make_spins()
    curves = build_spin_curves(np.concatenate([spins, A_VALS]))
    print(f'Geometric curves computed for {len(curves)} spins')

    n_P = int(round(np.log10(P_MAX / P_MIN) * P_PER_DECADE)) + 1
    P = np.logspace(np.log10(P_MIN), np.log10(P_MAX), n_P)
    sources = CATALOG
    codes_P = np.empty((len(COMBOS), spins.size, P.size), dtype=np.int8)
    # frac[c, s, a]: fraction of the log-mass range of source s where spin a
    # is valid for combination c
    frac = np.zeros((len(COMBOS), len(sources), spins.size))
    m_ranges = np.zeros((len(sources), 2))
    for j, src in enumerate(sources):
        _, lo, hi, _ = resolve_mass_reference(src['mass_range'])
        m_ranges[j] = lo, hi

    for c, (hr, al) in enumerate(COMBOS):
        print(f'HR={hr:g} alpha={al:g}')
        codes_P[c] = _codes(curves, spins, P, hr, al)
        for j, src in enumerate(sources):
            masses = np.logspace(*np.log10(m_ranges[j]), N_M_SOURCE)
            frac[c, j] = (_codes(curves, spins, masses * src['nu0'], hr, al)
                          == VALID).mean(axis=1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file, combos=np.array(COMBOS), hr_vals=HR_VALS,
        alpha_vals=ALPHA_VALS, spins=spins, P=P, codes_P=codes_P,
        frac=frac, source_idx=[s['idx'] for s in sources],
        m_ranges=m_ranges, hr_ref=HR_ISCO_DEFAULT, alpha_ref=ALPHA)
    print(f'Saved: {output_file}')
    return curves


def load(data_file=DATA_FILE):
    with np.load(data_file) as d:
        return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ helpers
def _combo_label(hr, al):
    return fr'$H/R_{{\rm isco}}={hr:g}$, $\alpha={al:g}$'


def _is_ref(hr, al):
    return np.isclose(hr, HR_ISCO_DEFAULT) and np.isclose(al, ALPHA)


def _mark_ref(ax):
    """Thicker frame for the reference (HR, alpha) of the other scripts."""
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)


# ----------------------------------------------------- Fig. 1: (P, a) map
def plot_P_map(data, outfile=None):
    set_style()
    P, spins, codes_P = data['P'], data['spins'], data['codes_P']
    hr_vals, al_vals = list(data['hr_vals']), list(data['alpha_vals'])
    combos = [tuple(c) for c in data['combos']]
    nr, nc = len(al_vals), len(hr_vals)
    srcs = select_sources(SOURCE_INDICES)
    src_colors = plt.get_cmap('rainbow', max(1, len(srcs)))(range(len(srcs)))

    fig = plt.figure(figsize=(7.0, 1.0 + 1.9 * nr), constrained_layout=True)
    gs = fig.add_gridspec(nr + 1, nc, height_ratios=[0.55] + [1] * nr)
    cmap = ListedColormap([VALID_COLOR] + [FAIL_COLORS[k] for k in range(1, 6)])
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N)
    axes = np.empty((nr, nc), dtype=object)
    for col in range(nc):
        # strip with the P interval of every source (same for all columns)
        sx = fig.add_subplot(gs[0, col])
        for k, (s, color) in enumerate(zip(srcs, src_colors)):
            _, lo, hi, kind = resolve_mass_reference(s['mass_range'])
            ls = '-' if kind == 'band' else (0, (2, 1))
            sx.plot([lo * s['nu0'], hi * s['nu0']], [k, k], color=color,
                    lw=1.2, ls=ls, solid_capstyle='butt')
        sx.set_xscale('log'); sx.set_xlim(P[0], P[-1])
        sx.set_ylim(len(srcs) - 0.3, -0.7)
        sx.set_yticks([]); sx.tick_params(axis='x', labelbottom=False)
        sx.set_title(fr'$H/R_{{\rm isco}}={hr_vals[col]:g}$')
        fix_spines(sx)
        for row in range(nr):
            ax = fig.add_subplot(gs[row + 1, col], sharex=sx,
                                 sharey=axes[0, 0] if row or col else None)
            axes[row, col] = ax
            c = combos.index((hr_vals[col], al_vals[row]))
            z = np.ma.masked_less(codes_P[c], 0)
            ax.pcolormesh(P, spins, z, cmap=cmap, norm=norm,
                          shading='nearest', rasterized=True)
            ax.set_xscale('log')
            ax.set_ylim(-1, 1)
            if _is_ref(hr_vals[col], al_vals[row]):
                _mark_ref(ax)
            if col == 0:
                ax.set_ylabel(r'spin $a$')
            else:
                ax.tick_params(axis='y', labelleft=False)
            if col == nc - 1:
                ax.text(1.03, 0.5, fr'$\alpha={al_vals[row]:g}$',
                        transform=ax.transAxes, rotation=270,
                        ha='left', va='center')
            if row == nr - 1:
                ax.set_xlabel(r'$P = M\nu_0\;[M_\odot\,\mathrm{Hz}]$')
            else:
                ax.tick_params(axis='x', labelbottom=False)
            fix_spines(ax)

    present = sorted({int(k) for k in np.unique(codes_P) if k >= 0})
    handles = [Patch(facecolor=VALID_COLOR, label='valid')] + [
        Patch(facecolor=FAIL_COLORS[k], edgecolor='k', linewidth=0.3,
              label=FAIL_LABELS[k]) for k in present if k > 0]
    leg = fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=6,
                     bbox_to_anchor=(0.5, 0.0))
    src_handles = [Line2D([], [], color=col, lw=1.2, label=s['name'])
                   for s, col in zip(srcs, src_colors)]
    fig.canvas.draw()
    y0 = leg.get_window_extent().transformed(fig.transFigure.inverted()).y0
    fig.legend(handles=src_handles, loc='upper center', ncol=5, fontsize=5.5,
               bbox_to_anchor=(0.5, y0 - 0.005))
    outfile = outfile or os.path.join(OUT_DIR, 'pif_hr_alpha_map.pdf')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', outfile)


# --------------------------------------------- Fig. 2: sources x (HR, alpha)
def plot_sources(data, indices=SOURCE_INDICES, n_cols=4, outfile=None):
    set_style()
    spins, frac = data['spins'], data['frac']
    combos = [tuple(c) for c in data['combos']]
    pos = {int(i): k for k, i in enumerate(data['source_idx'])}
    srcs = select_sources(indices)
    # rows ordered by alpha groups, HR inside each group (top to bottom)
    order = [combos.index((hr, al)) for al in data['alpha_vals']
             for hr in data['hr_vals']]
    n_hr = len(data['hr_vals'])
    ylabels = [fr'{combos[c][0]:g}, {combos[c][1]:g}' for c in order]

    n_rows = int(np.ceil(len(srcs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 1.25 * n_rows + 0.4),
                             sharex=True, sharey=True, squeeze=False,
                             constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_under('white')
    im = None
    for k, (ax, s) in enumerate(zip(axes.ravel(), srcs)):
        z = frac[order, pos[s['idx']], :]
        im = ax.pcolormesh(spins, np.arange(len(order)), z, cmap=cmap,
                           vmin=1e-9, vmax=1, shading='nearest',
                           rasterized=True)
        for b in range(n_hr, len(order), n_hr):          # alpha groups
            ax.axhline(b - 0.5, color='k', lw=0.4)
        ref = [combos[c] for c in order].index((HR_ISCO_DEFAULT, ALPHA))
        ax.add_patch(plt.Rectangle((-1, ref - 0.5), 2, 1, fill=False,
                                   ec='red', lw=0.6, zorder=5))
        ax.set_xlim(-1, 1); ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)), ylabels, fontsize=4.5)
        ax.text(0.03, 0.97, s['name'], transform=ax.transAxes, ha='left',
                va='top', fontsize=6,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8,
                          pad=1))
        ax.tick_params(axis='y', length=0)
        fix_spines(ax)
    fig.supxlabel(r'spin $a$', fontsize=8)
    fig.supylabel(r'$H/R_{\rm isco},\ \alpha$', fontsize=8)
    spare = axes.ravel()[len(srcs):]
    for ax in spare:
        ax.set_visible(False)
    cb_kw = dict(label='valid fraction of the mass range (white: none)',
                 ticks=[0.25, 0.5, 0.75, 1.0])
    if len(spare):      # colorbar inside the first empty slot
        cax = spare[0].inset_axes([0.05, 0.45, 0.9, 0.12])
        cax.set_visible(True)
        spare[0].set_visible(True); spare[0].axis('off')
        fig.colorbar(im, cax=cax, orientation='horizontal', **cb_kw)
    else:
        fig.colorbar(im, ax=axes, location='right', shrink=0.6, pad=0.01,
                     **cb_kw)
    outfile = outfile or os.path.join(OUT_DIR, 'pif_hr_alpha_sources.pdf')
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', outfile)


# ------------------------------------- Fig. 3: timescales of one source
def plot_source_timescales(curves, source_idx, data, outfile=None):
    set_style()
    src = select_sources([source_idx])[0]
    nu0 = src['nu0']
    M_ref, _, _, _ = resolve_mass_reference(src['mass_range'])
    hr_vals, al_vals = list(data['hr_vals']), list(data['alpha_vals'])
    nr, nc = len(al_vals), len(hr_vals)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(A_VALS)))
    fig, axs = plt.subplots(nr, nc, figsize=(7.0, 2.1 * nr), sharex=True,
                            sharey=True, squeeze=False,
                            constrained_layout=True)
    for i, al in enumerate(al_vals):
        for j, hr in enumerate(hr_vals):
            ax = axs[i, j]
            for spin, col in zip(A_VALS, colors):
                c = curves.get(float(spin))
                if c is None:
                    continue
                m = _match_from_curves(c, nu0, M_ref, alpha=al, hr_isco=hr)
                ax.plot(m['r_out'], m['t_wave'], color=col, lw=0.8, ls=':')
                ax.plot(m['r_out'], m['t_align'], color=col, lw=0.8, ls='--',
                        label=f'$a={spin:.3g}$')
                _plot_precession_markers(ax, m, nu0, col)
            ax.axhline(1.0 / nu0, color='k', lw=0.5, ls=(0, (20, 10)))
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, R_OUT_GRID[-1])
            if _is_ref(hr, al):
                _mark_ref(ax)
            if i == 0:
                ax.set_title(fr'$H/R_{{\rm isco}}={hr:g}$')
            if j == nc - 1:
                ax.text(1.03, 0.5, fr'$\alpha={al:g}$', transform=ax.transAxes,
                        rotation=270, ha='left', va='center')
            if i == nr - 1:
                ax.set_xlabel(r'$R_{\rm out}$ [$R_g$]')
            if j == 0:
                ax.set_ylabel('Time [s]')
            fix_spines(ax)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    style = [
        Line2D([], [], color='gray', lw=1, ls=':', label=r'$t_{\rm wave}$'),
        Line2D([], [], color='gray', lw=1, ls='--', label=r'$t_{\rm align}$'),
        Line2D([], [], color='k', lw=0.5, ls=(0, (20, 10)),
               label=r'$1/\nu_0$'),
    ]
    fig.legend(handles=handles + style, loc='upper center', ncol=7,
               fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(fr"{src['name']}, $\nu_0={nu0:.2e}$ Hz, "
                 fr"$M={M_ref:.1e}\,M_\odot$", fontsize=8)
    outfile = outfile or os.path.join(
        OUT_DIR, f"pif_timescales_{safe_filename(src['name'])}_hr_alpha.pdf")
    fig.savefig(outfile, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', outfile)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    curves = None
    if '--recompute' in sys.argv or not os.path.exists(DATA_FILE):
        curves = compute()
    data = load()
    plot_P_map(data)
    plot_sources(data)
    if TIMESCALE_SOURCES:
        if curves is None:
            curves = build_spin_curves(A_VALS)
        for idx in TIMESCALE_SOURCES:
            plot_source_timescales(curves, idx, data)
