"""
t_align / M vs r_out for two choices of Omega_loc in the stress G(R):
  solid  : Omega_loc = Omega_LT          (exact, G(r_out) = 0 by construction)
  dashed : Omega_loc = Z Omega_phi       (Foucart & Lai 2014, Motta+17)
Two panels, same H / R = HR_ISCO_DEFAULT at the ISCO (H = const for q = 3/2):
  top    : c_s = H Omega_K   (Newtonian vertical gravity, Motta+17)
  bottom : c_s = H Omega_z   (Kerr, as in pif_core)
M = 1, alpha = ALPHA (see pif_core).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

from setup import C, Rg_SUN, r_isco, g_theta, set_style, fix_spines
from pif_core import (_radial_setup, _trapz, sound_speed, P_TORQUE, Q_SOUND,
                      HR_ISCO_DEFAULT, ALPHA, N_RAD)

SPINS = (1.0, 0.8, 0.2, -0.2, -0.8, -1.0)
R_OUT_MAX = 500.0                              # [R_g], same for all spins


def tau_align(a, r_out, use_Z, kerr):
    """t_align / M [s/Msun] on the r_out grid, for one spin.
    kerr: c_s = H Omega_z (True) or H Omega_K (False)."""
    a_f = np.full(r_out.size, a)
    r_in = r_isco(a_f)
    R, a2, gphi, Sigma, w = _radial_setup(a_f, r_in, r_out, P_TORQUE, N_RAD)
    gth = g_theta(R, a2)
    g_lt = gphi - gth                                   # signed nu_LT * M
    g_p = _trapz(g_lt * w, R, axis=-1) / _trapz(w, R, axis=-1)
    loc = 0.5 * (1 - (gth / gphi)**2) * gphi if use_Z else g_lt

    G = cumulative_trapezoid((2*np.pi)**2 * w * (g_p[:, None] - loc), R,
                             axis=-1, initial=0.0)
    if kerr:
        cs2 = sound_speed(R, a2, gth=gth)**2
    else:
        cs2 = (C * HR_ISCO_DEFAULT * r_isco(a)**(Q_SOUND - 0.5)
               * R**(-Q_SOUND))**2
    num = _trapz(4 * ALPHA * G**2 / (Sigma * cs2 * R**3), R, axis=-1)
    den = _trapz(2*np.pi * w, R, axis=-1)
    return 1.0 / (num / den * Rg_SUN**2)


set_style()
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True, sharey=True,
                         gridspec_kw=dict(hspace=0.0))
colors = plt.cm.rainbow(np.linspace(0, 1, len(SPINS)))
PANELS = ((axes[0], False, r'$c_s=H\Omega_{\rm K}$'),
          (axes[1], True, r'$c_s=H\Omega_z$'))

for ax, kerr, title in PANELS:
    for a, col in zip(SPINS, colors):
        r_out = np.geomspace(1.02 * r_isco(a), R_OUT_MAX, 200)
        ax.plot(r_out, tau_align(a, r_out, False, kerr), color=col, ls='-', lw=1)
        ax.plot(r_out, tau_align(a, r_out, True, kerr), color=col, ls='--', lw=1)
    ax.text(0.04, 0.95, title, transform=ax.transAxes, ha='left', va='top')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$t_{\rm align}/M\ [{\rm s}\,M_\odot^{-1}]$')
    fix_spines(ax)

spin_handles = [Line2D([], [], color=c, lw=1, label=f'$a={a:1.1f}$')
                for a, c in zip(SPINS, colors)]
handles = [Line2D([], [], color='k', ls='-', lw=1, label=r'$\Omega_{\rm LT}$'),
           *spin_handles[:3],
           Line2D([], [], color='k', ls='--', lw=1,
                  label=r'$\mathcal{Z}\Omega_\varphi$'),
           *spin_handles[3:]]
axes[0].legend(handles=handles, loc='upper right', ncol=2, frameon=False)

axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1e3)   # room for the legend
axes[1].set_xlabel(r'$r_{\rm out}$')
fig.savefig('plots_paper/t_align_oloc.pdf', bbox_inches='tight')
plt.show()
