"""
t_align / M vs r_out for two choices of Omega_loc in the stress G(R):
  solid  : Omega_loc = Omega_LT          (exact, G(r_out) = 0 by construction)
  dashed : Omega_loc = Z Omega_phi       (Foucart & Lai 2014, Motta+17)
M = 1, alpha = ALPHA, HR_isco = HR_ISCO_DEFAULT (see pif_core).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

from setup import C, Rg_SUN, r_isco, g_theta, set_style, fix_spines
from pif_core import (_radial_setup, _trapz, P_TORQUE, Q_SOUND,
                      HR_ISCO_DEFAULT, ALPHA, N_RAD)

SPINS = (1.0, 0.8, 0.2, -0.2, -0.8, -1.0)
R_OUT_MAX = 500.0                              # [R_g], same for all spins


def tau_align(a, r_out, use_Z):
    """t_align / M [s/Msun] on the r_out grid, for one spin."""
    a_f = np.full(r_out.size, a)
    r_in = r_isco(a_f)
    R, a2, gphi, Sigma, w = _radial_setup(a_f, r_in, r_out, P_TORQUE, N_RAD)
    gth = g_theta(R, a2)
    g_lt = gphi - gth                                   # signed nu_LT * M
    g_p = _trapz(g_lt * w, R, axis=-1) / _trapz(w, R, axis=-1)
    loc = 0.5 * (1 - (gth / gphi)**2) * gphi if use_Z else g_lt

    G = cumulative_trapezoid((2*np.pi)**2 * w * (g_p[:, None] - loc), R,
                             axis=-1, initial=0.0)
    HR0 = HR_ISCO_DEFAULT * r_isco(a)**(Q_SOUND - 0.5)
    cs2 = (C * HR0)**2 * R**(-2 * Q_SOUND)
    num = _trapz(4 * ALPHA * G**2 / (Sigma * cs2 * R**3), R, axis=-1)
    den = _trapz(2*np.pi * w, R, axis=-1)
    return 1.0 / (num / den * Rg_SUN**2)


set_style()
fig, ax = plt.subplots(figsize=(3.4, 2.6))
colors = plt.cm.rainbow(np.linspace(0, 1, len(SPINS)))

for a, col in zip(SPINS, colors):
    r_out = np.geomspace(1.02 * r_isco(a), R_OUT_MAX, 200)
    ax.plot(r_out, tau_align(a, r_out, use_Z=False), color=col, ls='-', lw=1)
    ax.plot(r_out, tau_align(a, r_out, use_Z=True), color=col, ls='--', lw=1)

spin_handles = [Line2D([], [], color=c, lw=1, label=f'$a={a:1.1f}$')
                for a, c in zip(SPINS, colors)]
handles = [Line2D([], [], color='k', ls='-', lw=1, label=r'$\Omega_{\rm LT}$'),
           *spin_handles[:3],
           Line2D([], [], color='k', ls='--', lw=1,
                  label=r'$\mathcal{Z}\Omega_\varphi$'),
           *spin_handles[3:]]
ax.legend(handles=handles, loc='upper right', ncol=2, frameon=False)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(top=ax.get_ylim()[1] * 1e3)       # room for the legend
ax.set_xlabel(r'$r_{\rm out}$')
ax.set_ylabel(r'$t_{\rm align}/M\ [{\rm s}\,M_\odot^{-1}]$')
fix_spines(ax)
fig.savefig('plots_paper/t_align_oloc.pdf', bbox_inches='tight')
plt.show()
