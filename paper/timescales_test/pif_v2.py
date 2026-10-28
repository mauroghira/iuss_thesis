import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from setup import *
from plts_funcs import *
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
set_style()

from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

# --- 1: definisci parametri (senza rin)
params = {
    "a":   (-1, 1, 201),
    "rout": (1, 400, 200),
    "zeta": (-0.5, 0.5, 3),
    "M":   (10**6.3, 10**6.4, 1),
}
labels = list(params.keys())

param_vectors, mesh_arrays = create_param_grid(params)
A, ROUT, ZETA, M = mesh_arrays

# --- 2: costruisci r_in da a
RIN = r_isco(A)        # stesso shape della griglia

# --- 3: vincolo di consistenza su r_out
ROUT = np.maximum(ROUT, RIN)

# --- 4: chiama la funzione
freq = nu_solid_vect(A, RIN, ROUT, ZETA, M, n_rad=500)

fig, axes = plt.subplots(2, 3, figsize=(7, 4), sharey=True, sharex=True,
                         gridspec_kw={"wspace": 0.05, "hspace": 0.05})
axes = axes.flatten()
for ax in axes:
    fix_spines(ax)

colors = ["orange", "lightgreen", "blue"]

# Loop over zeta values
for i, ia in enumerate(np.linspace(0, 200, 6).round().astype(int)):
    for iz in [0,1,2]:
        rout_vals = ROUT[ia, :, iz, 0]      # vector shape (No,)
        freq_vals = freq[ia, :, iz, 0]    # vector shape (No,)

        # plot only valid positive frequencies
        mask2 = np.isfinite(freq_vals) & (freq_vals > 0)
        
        z = ZETA[ia, 0, iz, 0] 

        label = fr"$\zeta$ = {z}"
        axes[i].plot(rout_vals[mask2], freq_vals[mask2], lw=1, label=label, color = colors[iz])


    r_grid = np.linspace(1, 400, 100)
    # Target frequency band
    #axes[i].fill_between(r_grid, TARGET_MIN, TARGET_MAX, color='gray', alpha=0.2,
    #                label="Target range")
    axes[i].axhline(NU0, color="black", ls=':', lw=1)
    axes[i].axvline(r_isco(A[ia, 0, 0, 0]), color="purple", ls='--', lw=1)

    axes[i].text(
        0.95, 0.95,                      # posizione (x, y) in coordinate "axes" (0–1)
        f"a = {A[ia, 0, 0, 0]:.1f}",
        transform=axes[i].transAxes,     # usa coordinate relative agli assi
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)  # sfondo opzionale
    )

    if i>2:
        axes[i].set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    if i%3 == 0:
        axes[i].set_ylabel("Frequency [Hz]")
    axes[i].set_xscale("log")
    axes[i].set_yscale("log")

axes[-1].legend(loc="lower left", bbox_to_anchor=(0.1, 0.05))

plt.tight_layout()
plt.savefig('rpm_iomg/pif_6_spin.pdf', bbox_inches='tight')
plt.show()

##########################################

fig, ax = plt.subplots(figsize=(3.4, 3.1))
fix_spines(ax)

colors = ["orange", "lightgreen", "blue"]
styles = ["-", "--", ":", "-."]

legend_handles = []
# Loop over zeta values
for i, ia in enumerate(np.linspace(0, 200, 4).round().astype(int)):
    legend_handles.append(
        Line2D([], [], color="black", lw=1, ls=styles[i],
                        label=fr"a = {A[ia, 0, 0, 0]:.1f}")
    )
    
    for iz in [0,1,2]:
        rout_vals = ROUT[ia, :, iz, 0]      # vector shape (No,)
        freq_vals = freq[ia, :, iz, 0]    # vector shape (No,)

        # plot only valid positive frequencies
        mask2 = np.isfinite(freq_vals) & (freq_vals > 0)
        
        z = ZETA[ia, 0, iz, 0] 

        ax.plot(rout_vals[mask2], freq_vals[mask2], lw=1, ls=styles[i], color = colors[iz])

        if i == 3:
            legend_handles.append(
                Line2D([], [], color=colors[iz], lw=1, ls="solid",
                        label=fr"$\zeta$ = {z}")
            )
    

ax.axhline(NU0, color="cyan", ls='-', lw=1)
ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
ax.set_ylabel("Frequency [Hz]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(handles=legend_handles, loc="lower left", ncol=2,
          frameon=True, framealpha=0.8)

plt.tight_layout()
plt.savefig('rpm_iomg/PIF4spin.pdf', bbox_inches='tight')
plt.show()

###############################################
# SECOND
###############################################


# --- 1: definisci parametri (senza rin)
params3 = {
    "a":   (-0.999, 0.999, 201),
    "rin": np.linspace(3, 11, 5),
    "rout": (1, 50, 50),
    "zeta": (0, 0, 1),
    "M":   (10**6.3, 10**7, 1),
}
labels3 = list(params3.keys())

param_vectors3, mesh_arrays3 = create_param_grid(params3)
A3, RIN3_grid, ROUT3_grid, ZETA3, M3 = mesh_arrays3

# r_in effettivo = max(r_in_grid, ISCO(a))
RIN3 = np.maximum(RIN3_grid, r_isco(A3))

# r_out consistente
ROUT3 = np.maximum(ROUT3_grid, RIN3)

# maschera fisica: dove rout_originale < r_in effettivo, metti nan
invalid = ROUT3_grid < RIN3
RIN3  = np.where(invalid, np.nan, RIN3)
ROUT3 = np.where(invalid, np.nan, ROUT3)

############################################

invalid = ROUT3_grid < RIN3
valid   = ~invalid

# array 1D solo dei punti validi
a_v    = A3[valid]
rin_v  = RIN3[valid]
rout_v = ROUT3[valid]
zeta_v = ZETA3[valid]
m_v    = M3[valid]

# calcola solo su questi (molto più veloce se invalid è grande)
freq_valid = nu_solid_vect(a_v, rin_v, rout_v, zeta_v, m_v, n_rad=500)

# ricostruisci array pieno con nan dove invalid
freq3 = np.full(A3.shape, np.nan)
freq3[valid] = freq_valid

#############################################

fig = plt.figure(figsize=(7, 1.5))
# Crea un GridSpec con 6 colonne: 5 per i pannelli + 1 stretta per la colorbar
gs = gridspec.GridSpec(1, 6, figure=fig,
                       width_ratios=[1, 1, 1, 1, 1, 0.08],
                       wspace=0.05)

axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
cax  = fig.add_subplot(gs[0, 5])   # asse dedicato alla colorbar

valid = freq3[np.isfinite(freq3) & (freq3 > 0)]
vmin_global = valid.min()
vmax_global = valid.max()
norm = LogNorm(vmin=vmin_global, vmax=vmax_global)
cmap = plt.cm.inferno

a_vals    = param_vectors3["a"]
rout_vals = param_vectors3["rout"]

for ax, idx in zip(axes, range(10)):
    fix_spines(ax)
    F = freq3[:, idx, :, 0, 0]
    ROUT2D, A2D = np.meshgrid(rout_vals, a_vals)
    pcm = ax.pcolormesh(ROUT2D, A2D, F, shading="auto", cmap=cmap, norm=norm, rasterized=True)

    isco_vals = r_isco(a_vals)
    ax.plot(isco_vals, a_vals, "--", color="purple", lw=1)
    rin_eff = np.maximum(param_vectors3["rin"][idx], isco_vals)
    ax.plot(rin_eff, a_vals, "--", color="magenta", lw=1)

    ax.set_xscale("log")
    ax.set_xlim(rout_vals.min(), rout_vals.max())
    ax.set_title(fr"$r_{{\rm in}}$ = {param_vectors3['rin'][idx]:.0f} $R_g$",)
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")

    ax.contour(ROUT2D, A2D, F, levels=[NU0], colors=["cyan"], linewidths=2)

    # Nascondi le etichette y dei pannelli interni
    if idx > 0:
        ax.tick_params(labelleft=False)
        ax.sharey(axes[0])

axes[0].set_ylabel("a")

# Colorbar sull'asse dedicato → non tocca la larghezza dei pannelli
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cbar.set_label("Frequency [Hz]")

plt.savefig('rpm_iomg/pif_rin.pdf', bbox_inches='tight')
plt.show()