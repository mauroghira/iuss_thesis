"""
disk_mass_vs_radius.py
======================
Cumulative disk mass M(<r) as a function of radius for the Simple
and Novikov-Thorne disk models, with several parameter combinations.

Two side-by-side panels:
  left  — Simple model  (power-law B and Sigma, vary Sigma0 / alpha_S)
  right — NT model      (vary mdot and alpha)

Run from the project root:
    python disk_mass_vs_radius.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import r_isco, Rg_SUN, M_BH
from AEI_setups.simple_disc import disk_model_simple
from AEI_setups.nt_disc     import disk_model_NT

M_SUN_G = 1.989e33   # g
RG      = Rg_SUN * M_BH   # cm per r_g

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

A_SPIN  = 0.0
M       = M_BH
N_R     = 2000
R_MAX   = 1e6    # [r_g]  outer boundary for both models

# Simple model: parameter combos  (B00, Sigma0, alpha_B, alpha_S, label)
SIMPLE_CONFIGS = [
    (1.0, 1e3,  1.5, 3/5, r'$\Sigma_0=10^3$, $\alpha_S=-3/5$'),
    (1.0, 1e4,  3/5, 3/5, r'$\Sigma_0=10^4$, $\alpha_S=-3/5$'),
    (1.0, 1e5,  3/5, 3/5, r'$\Sigma_0=10^5$, $\alpha_S=-3/5$'),
    (1.0, 1e4,  1.5, 1.0, r'$\Sigma_0=10^4$, $\alpha_S=-1$'),
    (1.0, 1e4,  1.5, 1.5, r'$\Sigma_0=10^4$, $\alpha_S=-3/2$'),
]

# NT model: parameter combos  (mdot, alpha, label)
NT_CONFIGS = [
    (1e-4, 0.1,  r'$\dot{m}=10^{-4}$, $\alpha=0.1$'),
    (1e-3, 0.1,  r'$\dot{m}=10^{-3}$, $\alpha=0.1$'),
    (1e-2, 0.1,  r'$\dot{m}=10^{-2}$,     $\alpha=0.1$'),
    (1e-1, 0.1, r'$\dot{m}=10^{-1}$, $\alpha=0.1$'),
    (1e-3, 0.001,  r'$\dot{m}=10^{-3}$, $\alpha=0.001$'),
]

OUTFILE = 'disk_mass_vs_radius'

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def cumulative_mass(disk_model, params, r_max=R_MAX, M=M_BH, n_r=N_R):
    """
    Returns (r_arr [r_g], M_cum [M_sun]).

    M_cum[i] = integral from r_ISCO to r_arr[i] of 2π R Σ dR.
    """
    a     = float(params['a'])
    rISCO = float(r_isco(a))
    Rg    = Rg_SUN * M

    r_arr     = np.geomspace(rISCO, r_max, n_r)
    result    = disk_model(r_arr, **params)
    Sigma_arr = np.asarray(result[1], dtype=float)

    R_cm      = r_arr * Rg
    integrand = 2.0 * np.pi * R_cm * Sigma_arr   # g/cm

    dR        = np.diff(R_cm)
    trapz     = 0.5 * (integrand[:-1] + integrand[1:]) * dR
    M_cum_g   = np.concatenate([[0.0], np.cumsum(trapz)])

    return r_arr, M_cum_g / M_SUN_G


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_disk_mass():
    rISCO = float(r_isco(A_SPIN))

    cmap   = plt.cm.tab10
    colors = [cmap(i) for i in range(10)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    """
    fig.suptitle(
        f'Cumulative disk mass  $M(<r)$  —  '
        f'a = {A_SPIN},  M = {M:.2e} M$_\\odot$,  '
        f'$r_{{\\rm max}}$ = {R_MAX:.0f} r$_g$',
        fontsize=11, y=1.01,
    )
    """

    # ── LEFT: Simple model ────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title('Simple model', fontsize=11)

    for idx, (B00, S0, aB, aS, lbl) in enumerate(SIMPLE_CONFIGS):
        params = dict(a=A_SPIN, B00=B00, Sigma0=S0,
                      alpha_B=aB, alpha_S=aS)
        try:
            r_arr, M_cum = cumulative_mass(disk_model_simple, params, M=M)
            M_tot = M_cum[-1]
            M_tot_bh = M_tot / M_BH
            lbl_str = (f'{lbl}  '
                       f'($M_{{\\rm tot}}={M_tot:.2e}\\,'
                       f'M_\\odot = {M_tot_bh:.2e}\\,M_{{\\rm BH}}$)')
            ax.plot(r_arr / rISCO, M_cum,
                    color=colors[idx], lw=1.8,
                    label=lbl_str)
        except Exception as e:
            print(f'  [Simple] skip {lbl}: {e}')

    ax.axvline(1.0, color='black', ls='--', lw=1.2, alpha=0.6,
               label=r'$r_{\rm ISCO}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\,/\,r_{\rm ISCO}$', fontsize=11)
    ax.set_ylabel(r'$M(<r)$  [M$_\odot$]', fontsize=11)
    ax.grid(True, which='both', alpha=0.15)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    # ── RIGHT: NT model ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title('Novikov-Thorne model', fontsize=11)

    for idx, (mdot, alpha, lbl) in enumerate(NT_CONFIGS):
        params = dict(a=A_SPIN, mdot=mdot, alpha_visc=alpha)

        def _model_nt(r_rg, a, mdot, alpha_visc):
            return disk_model_NT(r_rg, a=a, mdot=mdot,
                                 alpha_visc=alpha_visc, hr=None, M=M)

        try:
            r_arr, M_cum = cumulative_mass(
                lambda r, **p: disk_model_NT(
                    r, a=p['a'], mdot=p['mdot'],
                    alpha_visc=p['alpha_visc'], hr=None, M=M),
                params, M=M,
            )
            M_tot = M_cum[-1]
            M_tot_bh = M_tot / M_BH
            lbl_str = (f'{lbl}  '
                       f'($M_{{\\rm tot}}={M_tot:.2e}\\,'
                       f'M_\\odot = {M_tot_bh:.2e}\\,M_{{\\rm BH}}$)')
            ax.plot(r_arr / rISCO, M_cum,
                    color=colors[idx], lw=1.8,
                    label=lbl_str)
        except Exception as e:
            print(f'  [NT] skip {lbl}: {e}')

    ax.axvline(1.0, color='black', ls='--', lw=1.2, alpha=0.6,
               label=r'$r_{\rm ISCO}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r\,/\,r_{\rm ISCO}$', fontsize=11)
    ax.set_ylabel(r'$M(<r)$  [M$_\odot$]', fontsize=11)
    ax.grid(True, which='both', alpha=0.15)
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTFILE+".png", dpi=150, bbox_inches='tight')
    plt.savefig(OUTFILE+".pdf", dpi=150, bbox_inches='tight')
    print(f'Saved → {OUTFILE}')
    plt.show()


if __name__ == '__main__':
    plot_disk_mass()
