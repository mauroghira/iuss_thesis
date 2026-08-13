"""
rpm_projections.py

Griglia 3x3: righe = canale (nu_LT, nu_per, nu_phi), colonne = proiezione
pairwise ((a,R), (a,M), (R,M)), con le sorgenti selezionate sovrapposte
per colore. Si veda projection_plots.py per la derivazione del metodo.

Per scegliere quali sorgenti mostrare (e in quale ordine di plotting),
modificare SOURCE_INDICES qui sotto: gli indici si riferiscono
all'ordine numerato in catalog.CATALOG.
"""

import numpy as np
import matplotlib.pyplot as plt
from setup import g_LT, g_per, g_phi, fix_spines, set_style, A_THORNE, R_MAX
from catalog import select_sources
from projection_plots import ChannelGrid, plot_projection_aR, plot_projection_aM, plot_projection_RM

set_style()

# ---- scegli qui le sorgenti da mostrare, nell'ordine desiderato -------
SOURCE_INDICES = [0, 9, 10, 11]   # es: J1257, RE J1034+396, NGC 4945, 1ES 1927+654
# -------------------------------------------------------------------

sources = select_sources(SOURCE_INDICES)

N_A, N_R = 500, 1500
a_vals = np.linspace(-A_THORNE, A_THORNE, N_A)
r_vals = np.logspace(0, np.log10(R_MAX), N_R)

channels = [
    (r"$\nu_{\rm LT}$", g_LT),
    (r"$\nu_{\rm per}$", g_per),
    (r"$\nu_{\varphi}$", g_phi),
]

cmap = plt.cm.tab20
colors = [cmap(i) for i in range(len(sources))]

fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5))

for row, (cname, g_func) in enumerate(channels):
    chgrid = ChannelGrid(g_func, a_vals, r_vals)   # calcolato una volta per riga

    ax_aR, ax_aM, ax_RM = axes[row]
    for ax in (ax_aR, ax_aM, ax_RM):
        fix_spines(ax)

    for src, color in zip(sources, colors):
        label = src["name"] if row == 0 else None
        plot_projection_aR(ax_aR, chgrid, src["nu0"], src["mass_range"],
                            color=color, label=label)
        plot_projection_aM(ax_aM, chgrid, src["nu0"], color=color)
        plot_projection_RM(ax_RM, chgrid, src["nu0"], color=color)

    ax_aR.set_ylabel(f"{cname}\nR [$R_g$]")
    ax_aM.set_ylim(1e5, 1e10)
    ax_RM.set_ylim(1e5, 1e10)

axes[0, 0].legend(loc="upper left", fontsize=6, framealpha=0.85, ncol=1)
for ax in axes[:-1].flatten():
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("images/rpm_projections_3x3.pdf", bbox_inches="tight")
plt.show()