"""
t_wave / M and t_align / M vs r_out for two choices of the vertical
frequency in the sound-speed normalisation c_s = H Omega_vert, with
H = HR_isco * R_isco independent of R (radiation-pressure dominated flow):
  solid  : Omega_vert = Omega_z   (Kerr vertical epicyclic frequency)
  dashed : Omega_vert = Omega_K   (Newtonian, c_s ~ R^-3/2, Motta+17)
i.e. c_s^z = c_s^K * Omega_z / Omega_K, with Omega_K = c / (R_g r^{3/2}).

t_wave = int 2 dR / c_s (= Motta+17 Eq. 6 for Omega_K), t_align from Eq. 7
of Motta+17 with the 'lt_exact' closure, both from pif_core.pif_geometric
(parameter vgrav). M = 1, alpha = ALPHA, HR_isco = HR_ISCO_DEFAULT, q = 3/2.

NB: for a = 1, Omega_z = 0 at r_isco = 1, so c_s^z -> 0 there and t_wave^z
diverges logarithmically (pif_core returns inf): that curve is not drawn.
t_align^z stays finite (the integrand vanishes like (r - r_in)^p).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from setup import r_isco, set_style, fix_spines
from pif_core import pif_geometric

SPINS = (0.998, 0.8, 0.2, -0.2, -0.8, -1.0)
R_OUT_MAX = 500.0                              # [R_g], same for all spins
N_ROUT = 200


def timescales(a, r_out, vgrav):
    """(t_wave / M, t_align / M) [s/Msun] on the r_out grid, for one spin.
    vgrav = 'kepler' (Omega_K) or 'kerr' (Omega_z) in c_s = H Omega_vert."""
    geo = pif_geometric(a, r_out, closure='lt_exact', vgrav=vgrav)
    return geo.tau_w, geo.tau_a


set_style()
fig, (ax_w, ax_a) = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True,
                                 gridspec_kw=dict(hspace=0.0))
colors = plt.cm.rainbow(np.linspace(0, 1, len(SPINS)))

for a, col in zip(SPINS, colors):
    r_out = np.geomspace(1.02 * r_isco(a), R_OUT_MAX, N_ROUT)
    tw_z, ta_z = timescales(a, r_out, 'kerr')
    tw_k, ta_k = timescales(a, r_out, 'kepler')
    ax_w.plot(r_out, tw_z, color=col, ls='-', lw=1)   # inf (not drawn) at a = 1
    ax_w.plot(r_out, tw_k, color=col, ls='--', lw=1)
    ax_a.plot(r_out, ta_z, color=col, ls='-', lw=1)
    ax_a.plot(r_out, ta_k, color=col, ls='--', lw=1)
    i3 = np.argmin(np.abs(r_out - 3 * r_isco(a)))
    rw = f"{tw_z[i3] / tw_k[i3]:.3g}"
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
