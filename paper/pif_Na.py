"""
Number of precession cycles before alignment, N_a = nu_0 t_align = g tau_a
(Eq. eq5:Na), at the fiducial parameters: figure replacing Table tab:Na.

N_a depends only on (a, r_bar), so for a set of spins the precession radius
and P = g(a, r_bar) are shown as functions of N_a:

    top    : r_bar / r_ISCO
    bottom : log [g_max(a) / P]

Each curve runs from the upper edge of the valid region, r_max(a)
(pif_constraints.window), inwards. Where alignment sets r_max the curve
starts at N_a = 1; for a >~ 0.93 waves set it, and the curve starts at
N_a(r_max) > 1 (open circle).

The curves are computed on a radial grid uniform in log(r/r_ISCO - 1), much
finer close to the ISCO than the shared R_OUT_GRID, so that large N_a are
resolved.

Usage: python pif_Na.py          fiducial (q = 3/2, h_ISCO = 0.1, alpha = 0.01)
       python pif_Na.py --q05    constant aspect ratio (q = 1/2), same h, alpha
       python pif_Na.py -o file  other output file (e.g. a .png preview)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from setup import r_isco, set_style, fix_spines
from pif_core import Q_SOUND, pif_geometric
from pif_mass_freq_map import SpinCurves, radial_nodes
import pif_constraints as pc
from pif_q import g_max, Q_ALT

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots_paper')
SPINS = (-1.0, -0.7, -0.3, -0.1, 0.2, 0.6, 0.9, 0.95, pc.A_THORNE)
N_LIM = (1.0, 1e5)
# r / r_isco - 1 from X_MIN to X_MAX (log-spaced), N_X nodes
X_MIN, N_X = 1e-3, 600
X_MAX = {Q_SOUND: 15.0, Q_ALT: 40.0}
N_TABLE = (2, 5, 10, 100, 1e5)        # printed, to check the text


def spin_curve(a, q=Q_SOUND):
    """Fine curve of one spin, cut at r_max. Returns dict with N_a, r/r_isco,
    log(g_max/P) (ascending r, i.e. descending N_a) and the limit of r_max."""
    ri = r_isco(a)
    r_out = ri * (1.0 + np.logspace(np.log10(X_MIN), np.log10(X_MAX[q]), N_X))
    geo = pif_geometric(a, r_out, q=q, n_rad=int(radial_nodes(r_out[-1])))
    rm, lim, _, _ = pc.window(SpinCurves(r_out, geo))

    lr, lN, lg = np.log(r_out), np.log(geo.g * geo.tau_a), np.log(geo.g)
    keep = r_out < rm
    # close the curve exactly at r_max (log-log interpolation)
    lr_e = np.log(rm)
    lN_e, lg_e = np.interp(lr_e, lr, lN), np.interp(lr_e, lr, lg)
    if lim == 'align':
        lN_e = 0.0                     # N_a = 1 by definition of r_max
    lr = np.r_[lr[keep], lr_e]
    N = np.exp(np.r_[lN[keep], lN_e])
    g = np.exp(np.r_[lg[keep], lg_e])
    return dict(a=a, limit=lim, N=N, x=np.exp(lr) / ri,
                d=np.log10(g_max(a) / g))


def at_cycles(c, n):
    """(r/r_isco, log(g_max/P)) at N_a = n, NaN outside the curve."""
    lN = np.log(c['N'][::-1])                    # ascending
    if not lN[0] <= np.log(n) <= lN[-1]:
        return np.nan, np.nan
    return (float(np.exp(np.interp(np.log(n), lN, np.log(c['x'][::-1])))),
            float(np.interp(np.log(n), lN, c['d'][::-1])))


def spin_colors(spins):
    """Rainbow map, evenly spaced in the order of the spins (purple for the
    most retrograde, red for the most prograde)."""
    spins = sorted(spins)
    return dict(zip(spins, plt.cm.rainbow(np.linspace(0.0, 1.0, len(spins)))))


def _spin_label(a):
    return f'${a:+.3f}$' if np.isclose(a, pc.A_THORNE) else f'${a:+g}$'


def plot(curves, output, x_lim=N_LIM):
    set_style()
    cols = spin_colors([c['a'] for c in curves])

    fig, (ax_r, ax_d) = plt.subplots(
        2, 1, sharex=True, figsize=(3.4, 3.6), gridspec_kw=dict(hspace=0.0))

    for c in curves:
        st = dict(color=cols[c['a']], lw=1.0,
                  ls=(0, (4, 1.5)) if c['a'] < 0 else '-')
        for ax, y in ((ax_r, c['x']), (ax_d, c['d'])):
            ax.plot(c['N'], y, **st)
            if c['limit'] != 'align':          # r_max set by waves
                ax.plot(c['N'][-1], y[-1], 'o', ms=2.5, mfc='white',
                        mec=cols[c['a']], mew=0.8, zorder=3)

    for ax in (ax_r, ax_d):
        fix_spines(ax)
        ax.set_xscale('log')
        ax.grid(True, which='major', axis='x', color='0.9', lw=0.4, zorder=0)
    ax_d.set_xlim(*x_lim)
    ax_d.set_xlabel(r'$N_a=\nu_0\,t_\mathrm{align}$')

    ax_r.set_yscale('log')
    ax_r.set_ylim(1.0, 12.0)
    ticks = (1, 1.5, 2, 3, 5, 10)
    ax_r.set_yticks(ticks, [f'{t:g}' for t in ticks])
    ax_r.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_r.set_ylabel(r'$\bar r/r_\mathrm{ISCO}$')

    ax_d.set_ylim(0.0, 2.0)
    ax_d.set_yticks(np.arange(0.0, 1.9, 0.5))     # free the panel junction
    ax_d.set_ylabel(r'$\log\,[g_\mathrm{max}(a)/P]$')

    handles = [Line2D([], [], color=cols[c['a']], lw=1.0,
                      ls=(0, (4, 1.5)) if c['a'] < 0 else '-',
                      label=_spin_label(c['a'])) for c in curves]
    ax_d.legend(handles=handles, loc='upper right',
                ncol=3, fontsize=6, frameon=False, handlelength=1.8,
                columnspacing=0.9, title=r'Spin $a$', title_fontsize=6)

    fig.savefig(output, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return output


def print_table(curves):
    print('r/r_isco and log(g_max/P) at N_a =', N_TABLE)
    for c in curves:
        vals = [at_cycles(c, n) for n in N_TABLE]
        fr = ' '.join('   --' if not np.isfinite(r) else f'{r:5.2f}' for r, _ in vals)
        fd = ' '.join('   --' if not np.isfinite(d) else f'{d:5.2f}' for _, d in vals)
        print(f"  a={c['a']:+.3f} {fr} | {fd} | N_a(r_max)={c['N'][-1]:.3g} "
              f"({c['limit']})")


if __name__ == '__main__':
    q = Q_ALT if '--q05' in sys.argv else Q_SOUND
    default = 'pif_Na_q05.pdf' if q == Q_ALT else 'pif_Na.pdf'
    out = (sys.argv[sys.argv.index('-o') + 1] if '-o' in sys.argv
           else os.path.join(PLOT_DIR, default))
    curves = [spin_curve(a, q) for a in SPINS]
    print_table(curves)
    print('Saved:', plot(curves, out))
