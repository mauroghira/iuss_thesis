"""
Limits on (h_ISCO, alpha) for rapidly spinning holes at the P of type-C
QPOs (Appendix B, Eq. eqA:bounds): figure replacing Table tab:xrb.

At fixed (a, P) the precession radius r_bar(a, P) does not depend on the disc
parameters, and with N_w^fid = g tau_w, N_a^fid = g tau_a at h_ISCO = 0.1,
alpha = 0.01 the conditions N_w < 1 < N_a become

    h_ISCO > h_min = 0.1 N_w^fid
    alpha  < alpha_max = 0.01 (h_ISCO / 0.1)^2 N_a^fid

Since every outer radius gives one P = g(a, r_out), the curves are drawn
parametrically in r_out (no interpolation in P):

    top    : h_min                                  vs P
    bottom : alpha_max (0.1 / h_ISCO)^2 = 0.01 N_a^fid  vs P

for a = 0.9, 0.95, 0.998 and the two thickness profiles q = 3/2 (fiducial,
solid) and q = 1/2 (dashed). The top axis gives nu_0 for M = 10 Msun.

Usage: python pif_xrb.py [-o file]
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from setup import set_style, fix_spines
from pif_core import ALPHA, HR_ISCO_DEFAULT, Q_SOUND
from pif_mass_freq_map import build_spin_curves
import pif_constraints as pc
from pif_q import Q_ALT, XRB_P, loglog_at

PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots_paper')
SPINS = (0.9, 0.95, pc.A_THORNE)
P_LIM = (1.0, 100.0)                     # Hz Msun: 0.1-10 Hz at 10 Msun
M_XRB = 10.0                             # Msun, for the top axis
# outer radii covering P_LIM for all SPINS (r_bar = 14-195 R_g in the table)
R_OUT = np.logspace(np.log10(8.0), np.log10(400.0), 300)
Q_STYLES = {Q_SOUND: dict(ls='-', label=r'$q=3/2$'),
            Q_ALT: dict(ls=(0, (4, 1.5)), label=r'$q=1/2$')}


def compute():
    """{q: {a: (P, h_min, alpha_max (0.1/h)^2, r_out)}}, ascending P."""
    out = {}
    for q in Q_STYLES:
        curves = build_spin_curves(SPINS, q=q, r_out_grid=R_OUT)
        out[q] = {}
        for a in SPINS:
            c = pc.nearest(curves, a)
            g = c.geo.g
            h_min = HR_ISCO_DEFAULT * g * c.geo.tau_w
            a_max = ALPHA * g * c.geo.tau_a
            out[q][a] = tuple(v[::-1] for v in (g, h_min, a_max, c.r_out))
    return out


def plot(res, output):
    set_style()
    cols = dict(zip(SPINS, plt.cm.rainbow(np.linspace(0.0, 1.0, len(SPINS)))))

    fig, (ax_h, ax_a) = plt.subplots(
        2, 1, sharex=True, figsize=(3.4, 3.6), gridspec_kw=dict(hspace=0.0))

    for q, st in Q_STYLES.items():
        for a in SPINS:
            P, h_min, a_max, _ = res[q][a]
            kw = dict(color=cols[a], lw=1.0, ls=st['ls'])
            ax_h.plot(P, h_min, **kw)
            ax_a.plot(P, a_max, **kw)

    for ax in (ax_h, ax_a):
        fix_spines(ax)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax_a.set_xlim(*P_LIM)
    ax_a.set_xlabel(r'$P=M\nu_0$ [Hz $M_\odot$]')

    ax_h.set_ylabel(r'$h_\mathrm{min}$')
    ax_a.set_ylabel(r'$\alpha_\mathrm{max}\,(0.1/h_\mathrm{ISCO})^2$')

    # top axis: frequency of a 10 Msun hole (a unit conversion of P)
    top = ax_h.secondary_xaxis('top', functions=(lambda P: P / M_XRB,
                                                 lambda nu: nu * M_XRB))
    top.set_xlabel(rf'$\nu_0$ [Hz] for $M={M_XRB:g}\,M_\odot$')
    top.tick_params(direction='in', which='both')
    ax_h.tick_params(axis='x', which='both', top=False)

    spin_h = [Line2D([], [], color=cols[a], lw=1.0,
                     label=(rf'$a={a:.3f}$' if np.isclose(a, pc.A_THORNE)
                            else rf'$a={a:g}$')) for a in SPINS]
    q_h = [Line2D([], [], color='0.3', lw=1.0, ls=st['ls'], label=st['label'])
           for st in Q_STYLES.values()]
    # both legends in the empty band between the q = 3/2 and q = 1/2 curves
    ax_h.legend(handles=q_h, loc='center left', bbox_to_anchor=(0.0, 0.45),
                fontsize=6, frameon=False, handlelength=1.8)
    ax_a.legend(handles=spin_h, loc='lower right', bbox_to_anchor=(1.0, 0.0),
                fontsize=6, frameon=False, handlelength=1.8)

    fig.savefig(output, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return output


def print_table(res):
    """Values at the P of Table tab:xrb, to check against it."""
    print('a      P    r_bar | q=3/2: h_min  a_max(0.3) | q=1/2: h_min  '
          'a_max(0.2) a_max(0.3)')
    for a in SPINS:
        for P0 in XRB_P:
            row = []
            for q, hs in ((Q_SOUND, (0.3,)), (Q_ALT, (0.2, 0.3))):
                P, h_min, a_max, r = res[q][a]
                c = type('C', (), dict(r_out=P))         # log-log in P
                hm, am = loglog_at(c, h_min, P0), loglog_at(c, a_max, P0)
                row.append(f'{hm:6.3f} ' + ' '.join(
                    f'{am * (h / HR_ISCO_DEFAULT) ** 2 if h > hm else np.nan:9.2g}'
                    for h in hs))
                rb = loglog_at(c, r, P0)
            print(f'{a:.3f} {P0:5g} {rb:6.1f} | ' + ' | '.join(row))


if __name__ == '__main__':
    out = (sys.argv[sys.argv.index('-o') + 1] if '-o' in sys.argv
           else os.path.join(PLOT_DIR, 'pif_xrb_limits.pdf'))
    res = compute()
    print_table(res)
    print('Saved:', plot(res, out))
