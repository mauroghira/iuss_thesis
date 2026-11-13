"""
t_wave / M and t_align / M vs r_out for two choices of the vertical
frequency in the sound-speed normalisation c_s = H Omega_vert, with
H = HR_isco * R_isco independent of R (radiation-pressure dominated flow):
  solid  : Omega_vert = Omega_z   (Kerr vertical epicyclic frequency)
  dashed : Omega_vert = Omega_K   (Newtonian, c_s ~ R^-3/2, Motta+17)
i.e. c_s^z = c_s^K * Omega_z / Omega_K, with Omega_K = c / (R_g r^{3/2}).

t_wave = int 2 dR / c_s (trapezoid; = Motta+17 Eq. 6 for Omega_K),
t_align from Eq. 7 of Motta+17 with the 'lt_exact' closure.
M = 1, alpha = ALPHA, HR_isco = HR_ISCO_DEFAULT, q = 3/2 (see pif_core).

NB: for a = 1, Omega_z = 0 at r_isco = 1, so c_s^z -> 0 there and t_wave^z
diverges logarithmically: that curve is not drawn. t_align^z stays finite
(the integrand vanishes like (r - r_in)^p).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid

from setup import C, Rg_SUN, r_isco, g_theta, set_style, fix_spines
from pif_core import (_radial_setup, _trapz, P_TORQUE, Q_SOUND,
                      HR_ISCO_DEFAULT, ALPHA, N_RAD)

SPINS = (0.998, 0.8, 0.2, -0.2, -0.8, -1.0)
R_OUT_MAX = 500.0                              # [R_g], same for all spins
N_ROUT = 200


def timescales(a, r_out, vgrav):
    """(t_wave / M, t_align / M) [s/Msun] on the r_out grid, for one spin.
    vgrav = 'kepler' (Omega_K) or 'kerr' (Omega_z) in c_s = H Omega_vert."""
    a_f = np.full(r_out.size, a)
    r_in = r_isco(a_f)
    R, a2, gphi, Sigma, w = _radial_setup(a_f, r_in, r_out, P_TORQUE, N_RAD)
    gth = g_theta(R, a2)

    # --- sound speed -----------------------------------------------------
    HR0 = HR_ISCO_DEFAULT * r_isco(a)**(Q_SOUND - 0.5)
    cs = C * HR0 * R**(-Q_SOUND)                      # H Omega_K (Motta+17)
    if vgrav == 'kerr':
        # Omega_z / Omega_K = 2 pi g_theta R_g r^{3/2} / c
        cs = cs * gth * (2.0 * np.pi * Rg_SUN / C) * R**1.5
    elif vgrav != 'kepler':
        raise ValueError(f"vgrav must be 'kepler' or 'kerr', got {vgrav!r}")

    # --- t_wave / M = int 2 dR / c_s ------------------------------------
    tau_w = _trapz(2.0 * Rg_SUN / cs, R, axis=-1)

    # --- t_align / M, closure 'lt_exact' ---------------------------------
    g_lt = gphi - gth                                 # signed nu_LT * M
    g_p = _trapz(g_lt * w, R, axis=-1) / _trapz(w, R, axis=-1)
    G = cumulative_trapezoid((2*np.pi)**2 * w * (g_p[:, None] - g_lt), R,
                             axis=-1, initial=0.0)
    num = _trapz(4 * ALPHA * G**2 / (Sigma * cs**2 * R**3), R, axis=-1)
    den = _trapz(2*np.pi * w, R, axis=-1)
    tau_a = 1.0 / (num / den * Rg_SUN**2)
    return tau_w, tau_a


set_style()
fig, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True,
                                 gridspec_kw=dict(hspace=0.0))
colors = plt.cm.rainbow(np.linspace(0, 1, len(SPINS)))

for a, col in zip(SPINS, colors):
    r_out = np.geomspace(1.02 * r_isco(a), R_OUT_MAX, N_ROUT)
    tw_z, ta_z = timescales(a, r_out, 'kerr')
    tw_k, ta_k = timescales(a, r_out, 'kepler')
    if a < 1.0:                                       # t_wave^z diverges at a = 1
        ax_w.plot(r_out, tw_z, color=col, ls='-', lw=1)
    ax_w.plot(r_out, tw_k, color=col, ls='--', lw=1)
    ax_a.plot(r_out, ta_z, color=col, ls='-', lw=1)
    ax_a.plot(r_out, ta_k, color=col, ls='--', lw=1)
    i3 = np.argmin(np.abs(r_out - 3 * r_isco(a)))
    rw = f"{tw_z[i3] / tw_k[i3]:.3g}" if a < 1.0 else "inf"
    print(f"a = {a:+.1f}:  r_out = 3 r_isco  ->  t_wave^z/t_wave^K = {rw},"
          f"  t_align^z/t_align^K = {ta_z[i3] / ta_k[i3]:.3g}")

spin_handles = [Line2D([], [], color=c, lw=1, label=f'$a={a:1.1f}$')
                for a, c in zip(SPINS, colors)]
handles = [Line2D([], [], color='k', ls='-', lw=1, label=r'$\Omega_z$'),
           *spin_handles[:3],
           Line2D([], [], color='k', ls='--', lw=1, label=r'$\Omega_{\rm K}$'),
           *spin_handles[3:]]
ax_a.legend(handles=handles, loc='upper right', ncol=2, frameon=False)

for ax in (ax_w, ax_a):
    ax.set_xscale('log')
    ax.set_yscale('log')
    fix_spines(ax)
ax_a.set_ylim(top=ax_a.get_ylim()[1] * 1e3)       # room for the legend
ax_w.set_ylabel(r'$t_{\rm wave}/M\ [{\rm s}\,M_\odot^{-1}]$')
ax_a.set_ylabel(r'$t_{\rm align}/M\ [{\rm s}\,M_\odot^{-1}]$')
ax_a.set_xlabel(r'$r_{\rm out}$')
fig.savefig('plots_paper/t_cs_vgrav.pdf', bbox_inches='tight')
plt.show()
