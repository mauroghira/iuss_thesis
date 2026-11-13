"""
Timescale plots for J1257 for several (H/R_isco, alpha) combinations.
Rows: method (eig; motta + lt_exact). Columns: (HR, alpha) combos.
Mass fixed (M_REF), curves for different spins, same style as pif_plots.

Cost: the eigenmode is recomputed for EVERY combo (not rescalable), about
n_spin * 200 r_out * ~50 ms per combo; use WORKERS > 1. Curves are cached
in a pickle so that re-plotting is instant.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"          # forced, not setdefault

import pickle

import numpy as np
import matplotlib.pyplot as plt

from setup import set_style, fix_spines
from catalog import select_sources
from pif_mass_freq_map import build_spin_curves
from pif_timescales import A_VALS, _match_from_curves
from pif_plots import _plot_precession_markers
from pif_core import VGRAV_DEFAULT

COMBOS = [(0.03, 0.01), (0.1, 0.01), (0.3, 0.01), (0.1, 0.001), (0.1, 0.1)]
METHODS = [('eig', dict(method='eig', rate='lt', n_eig=120)),
           ('motta, lt_exact', dict(method='motta', closure='lt_exact'))]
SOURCE_IDX = 10                 # J1257
M_REF = 2.0e6                   # Msun, J1257 (catalog)
WORKERS = 8
VGRAV = VGRAV_DEFAULT           # sound speed: 'kerr' (H Omega_z) or 'kepler'
OUT_DIR = 'hr_alpha_test'
# the cache name carries VGRAV, so curves of another c_s are never reused
# (J1257_hr_alpha_curves.pkl, without suffix, was computed with 'kepler')
CACHE = os.path.join(OUT_DIR, f'J1257_hr_alpha_curves_{VGRAV}.pkl')
SEC_PER_YR = 365.25 * 24 * 3600


def compute_all():
    cache = {}
    for hr, al in COMBOS:
        for label, kw in METHODS:
            print(f'HR={hr} alpha={al} {label}', flush=True)
            extra = dict(workers=WORKERS) if kw['method'] == 'eig' else {}
            cache[(hr, al, label)] = build_spin_curves(
                A_VALS, hr_isco=hr, alpha=al, vgrav=VGRAV, **kw, **extra)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(cache, f)
    return cache


def plot(cache):
    set_style()
    src = select_sources([SOURCE_IDX])[0]
    nu0 = src['nu0']
    colors = plt.cm.viridis(np.linspace(0, 1, len(A_VALS)))
    nr, nc = len(METHODS), len(COMBOS)
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.6 * nr),
                            sharex=True, sharey=True, squeeze=False,
                            constrained_layout=True)
    for i, (label, _) in enumerate(METHODS):
        for j, (hr, al) in enumerate(COMBOS):
            ax = axs[i, j]
            curves = cache[(hr, al, label)]
            for spin, col in zip(A_VALS, colors):
                c = curves.get(float(spin))
                if c is None:
                    continue
                m = _match_from_curves(c, nu0, M_REF)
                ax.plot(m['r_out'], m['t_wave'], color=col, lw=0.8, ls=':')
                ax.plot(m['r_out'], m['t_align'], color=col, lw=0.8, ls='--',
                        label=f'$a={spin:.3g}$')
                _plot_precession_markers(ax, m, nu0, col)
            ax.axhline(1.0 / nu0, color='k', lw=0.5, ls=(0, (20, 10)))
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlim(1, 200); ax.set_ylim(1e0, 1e16)
            fix_spines(ax)
            ax.text(0.03, 0.97,
                    fr'{label}' + '\n' +
                    fr'$H/R_{{\rm isco}}={hr:g},\ \alpha={al:g}$',
                    transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75))
            if i == nr - 1:
                ax.set_xlabel(r'$R_{\rm out}$ [$R_g$]')
            if j == 0:
                ax.set_ylabel('Time [s]')
    axs[0, 0].legend(loc='lower right', fontsize=5, ncol=2, frameon=True)
    fig.suptitle(fr"{src['name']}, $\nu_0={nu0:.2e}$ Hz, "
                 fr"$M={M_REF:.1e}\,M_\odot$", fontsize=8)
    out = os.path.join(OUT_DIR, 'pif_timescales_J1257_hr_alpha.pdf')
    fig.savefig(out, bbox_inches='tight')
    print('Saved:', out)


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else compute_all()
    plot(cache)