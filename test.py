import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from setup import *
from plts_funcs import *

#functions for plots

def select(a_match, n_pick=10):
    a_match = np.array(a_match)
    # Sample a few spin values for clarity
    idx = np.linspace(0, len(a_match)-1, n_pick).astype(int)
    a_sampled = a_match[idx]
    return a_sampled

def plot_nu_vs_r(a_sampled, M, label, title, model):
    plt.figure(figsize=(9,6))

    for a in a_sampled:
        r_grid = np.linspace(r_isco(a), 1e4, 800)
        freq = np.array([model(r, a, M) for r in r_grid])
        plt.plot(r_grid, freq, label=f"a = {a:.5f}")
        
    r_gr0 = np.linspace(r_isco(1), 1e4, 800)
    # Target frequency band
    plt.fill_between(r_gr0, TARGET_MIN, TARGET_MAX, color='gray', alpha=0.2,
                    label="Target range")
    plt.plot(r_gr0, np.ones_like(r_grid)*NU0, label="Target frequency")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("r  [GM/c²]")
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# --- 1: definisci parametri (senza rin)
params3 = {
    "a":   np.linspace(-0.999, 0.999, 111),
    "rin": np.linspace(1, 12, 50),
    "rout": np.linspace(1, 30, 30),
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

from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Costruzione mappa: r_out del match per ogni (r_in, a)
a_vals    = param_vectors3["a"]
rout_vals = param_vectors3["rout"]
rin_vals  = param_vectors3["rin"]  

match_map = np.full((len(rin_vals), len(a_vals)), np.nan)

for i, r_in in enumerate(rin_vals):
    F_row = freq3[:, i, :, 0, 0]   # shape (n_a, n_rout)

    for j, a in enumerate(a_vals):
        r_isco_val = r_isco(a)
        r_min = max(r_in, r_isco_val)

        freq_1d = F_row[j, :]       # frequenza in funzione di r_out
        valid   = rout_vals >= r_min
        f_v     = freq_1d[valid]
        r_v     = rout_vals[valid]

        if len(f_v) < 2:
            continue

        diff    = f_v - NU0
        sign_ch = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_ch) == 0:
            continue

        j0 = sign_ch[0]
        try:
            interp = interp1d(f_v[j0:j0+2], r_v[j0:j0+2])
            match_map[i, j] = float(interp(NU0))
        except Exception:
            continue

# Plot
fig, ax = plt.subplots(figsize=(5, 4))

RIN2D, A2D = np.meshgrid(rin_vals, a_vals, indexing="ij")

valid_vals = match_map[np.isfinite(match_map) & (match_map > 0)]
norm_map   = Normalize(vmin=1, vmax=30)

pcm = ax.pcolormesh(RIN2D, A2D, match_map,
                    cmap="plasma",
                    norm=norm_map,
                    shading="auto")

ax.set_xlabel(r"$r_{in}$ [$R_g$]", fontsize=12)
ax.set_ylabel("a", fontsize=12)
#ax.set_title(fr"$r_{{out}}$ del match $\nu = \nu_0 = {NU0:.1e}$ Hz", fontsize=12)
ax.grid(True, which="both", ls=":", alpha=0.3)

cbar = fig.colorbar(cm.ScalarMappable(norm=norm_map, cmap="plasma"),
                    ax=ax,
                    label=r"$r_{out}$ match [$R_g$]")

plt.savefig("pif_rin_2.pdf", dpi=150, bbox_inches="tight")