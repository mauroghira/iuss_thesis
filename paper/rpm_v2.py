"""
rpm_projections.py

Griglia 3x3: righe = canale (nu_LT, nu_per, nu_phi), colonne = proiezione
pairwise ((a,R), (a,M), (R,M)), con le sorgenti selezionate sovrapposte
per colore. Si veda projection_plots.py per la derivazione del metodo e
per la distinzione Regione A (prior AGN generico, linea continua) /
Regione B (massa nota di letteratura, linea tratteggiata) / intersezione
(riempimento colorato).

Per scegliere quali sorgenti mostrare (e in quale ordine di plotting),
modificare SOURCE_INDICES qui sotto: gli indici si riferiscono
all'ordine numerato in catalog.CATALOG.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from setup import (g_LT, g_per, g_phi, fix_spines, set_style,
                   A_THORNE, R_MAX, M_AGN_MIN, M_AGN_MAX)
from catalog import select_sources
from projection_plots import ChannelGrid, plot_projection_aR, plot_projection_aM, plot_projection_RM


def area_aR(chgrid, nu0, mass_range):
    G, A, R = chgrid.G, chgrid.A, chgrid.R
    levels_A = [M_AGN_MIN * nu0, M_AGN_MAX * nu0]
    mask_A = (G >= levels_A[0]) & (G <= levels_A[1])

    mask_fill = mask_A
    if mass_range is not None:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            mask_fill = np.zeros_like(mask_A)
        else:
            levels_B = [M_lo * nu0, M_hi * nu0]
            mask_B = (G >= levels_B[0]) & (G <= levels_B[1])
            mask_fill = mask_A & mask_B

    if not mask_fill.any():
        return 0.0

    # Area nelle coordinate effettivamente visualizzate: (a, log R).
    x = chgrid.a_vals
    y = np.log(chgrid.r_vals)
    dx = np.diff(x)
    dy = np.diff(y)
    cell_area = dx[:, None] * dy[None, :]
    cell_mask = (mask_fill[:-1, :-1] & mask_fill[1:, :-1] &
                 mask_fill[:-1, 1:] & mask_fill[1:, 1:])
    return float(np.sum(cell_area[cell_mask]))


def _area_band(x, lo, hi, valid, log_x=False, log_y=True):
    if lo is None or not np.any(valid):
        return 0.0
    xx = x[valid]
    yy_lo = lo[valid]
    yy_hi = hi[valid]
    if log_x:
        xx = np.log(xx)
    if log_y:
        yy_lo = np.log(yy_lo)
        yy_hi = np.log(yy_hi)
    return float(np.trapezoid(np.maximum(yy_hi - yy_lo, 0.0), xx))


def area_aM(chgrid, nu0, mass_range):
    M_lo_raw = chgrid.Gmin_a / nu0
    M_hi_raw = chgrid.Gmax_a / nu0
    valid_raw = np.isfinite(M_lo_raw) & np.isfinite(M_hi_raw)

    A_lo = np.maximum(M_lo_raw, M_AGN_MIN)
    A_hi = np.minimum(M_hi_raw, M_AGN_MAX)
    valid_A = valid_raw & (A_lo <= A_hi)

    if mass_range is None:
        fill_lo, fill_hi, valid_fill = A_lo, A_hi, valid_A
    else:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            return 0.0
        B_lo = np.maximum(M_lo_raw, M_lo)
        B_hi = np.minimum(M_hi_raw, M_hi)
        valid_B = valid_raw & (B_lo <= B_hi)
        fill_lo = np.maximum(A_lo, B_lo)
        fill_hi = np.minimum(A_hi, B_hi)
        valid_fill = valid_A & valid_B & (fill_lo <= fill_hi)

    return _area_band(chgrid.a_vals, fill_lo, fill_hi, valid_fill)


def area_RM(chgrid, nu0, mass_range):
    M_lo_raw = chgrid.Gmin_R / nu0
    M_hi_raw = chgrid.Gmax_R / nu0
    valid_raw = np.isfinite(M_lo_raw) & np.isfinite(M_hi_raw)

    A_lo = np.maximum(M_lo_raw, M_AGN_MIN)
    A_hi = np.minimum(M_hi_raw, M_AGN_MAX)
    valid_A = valid_raw & (A_lo <= A_hi)

    if mass_range is None:
        fill_lo, fill_hi, valid_fill = A_lo, A_hi, valid_A
    else:
        M_lo, M_hi = mass_range
        if M_lo == M_hi:
            return 0.0
        B_lo = np.maximum(M_lo_raw, M_lo)
        B_hi = np.minimum(M_hi_raw, M_hi)
        valid_B = valid_raw & (B_lo <= B_hi)
        fill_lo = np.maximum(A_lo, B_lo)
        fill_hi = np.minimum(A_hi, B_hi)
        valid_fill = valid_A & valid_B & (fill_lo <= fill_hi)

    return _area_band(chgrid.r_vals, fill_lo, fill_hi, valid_fill, log_x=True)


def ordered_sources_by_area(sources, colors, chgrid, projection, alpha_min=0.2, alpha_max=0.5):
    if projection == "aR":
        area_func = area_aR
    elif projection == "aM":
        area_func = area_aM
    elif projection == "RM":
        area_func = area_RM
    else:
        raise ValueError(f"Proiezione sconosciuta: {projection}")

    items = []
    for src, color in zip(sources, colors):
        area = area_func(chgrid, src["nu0"], src["mass_range"])
        items.append((area, src, color))

    # Area maggiore prima: alpha piccolo. Aree minori dopo: alpha crescente.
    items.sort(key=lambda item: item[0], reverse=True)
    n = len(items)
    alphas = np.full(n, alpha_min) if n == 1 else np.linspace(alpha_min, alpha_max, n)
    return [(src, color, float(alpha), area)
            for alpha, (area, src, color) in zip(alphas, items)]


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

    # L'ordine viene ricalcolato per ogni pannello: le aree dipendono infatti
    # dal canale (riga), dalla proiezione (colonna) e dalla sorgente.
    for projection, ax in (("aR", ax_aR), ("aM", ax_aM), ("RM", ax_RM)):
        ordered = ordered_sources_by_area(
            sources, colors, chgrid, projection, alpha_min=0.2, alpha_max=0.5
        )

        for src, color, alpha, area in ordered:
            label = src["name"] if row == 0 else None
            if projection == "aR":
                plot_projection_aR(
                    ax, chgrid, src["nu0"], src["mass_range"],
                    color=color, alpha=alpha, label=label
                )
            elif projection == "aM":
                plot_projection_aM(
                    ax, chgrid, src["nu0"], src["mass_range"],
                    color=color, alpha=alpha
                )
            else:
                plot_projection_RM(
                    ax, chgrid, src["nu0"], src["mass_range"],
                    color=color, alpha=alpha
                )

    ax_aR.set_ylabel(f"{cname}\nR [$R_g$]")
    ax_aM.set_ylim(1e4, 1e10)
    ax_RM.set_ylim(1e4, 1e10)


# --- legenda -------------------------------------------------------
# Per riga (colore = sorgente) si aggiungono gli handle di riga 0 (nomi
# delle sorgenti); qui si aggiungono in coda gli elementi che spiegano
# il SIGNIFICATO di stile-linea / riempimento, comune a tutte le righe
# e a tutte le sorgenti (non e' una proprieta' della sorgente, quindi va
# tenuta separata dai colori).
legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
legend_handles.extend([
    Line2D([], [], color="black", lw=1.2, ls="-"),
    Line2D([], [], color="black", lw=1, ls=":"),
    Patch(facecolor="black", alpha=0.25, edgecolor="none"),
])
legend_labels.extend([
    r"A: accessible parameter space",
    r"B: known mass range",
    r"valid parameter space (A ∩ B)",
])

# Legenda unica, fuori dai pannelli, in alto; ncol = numero di sorgenti
# (le righe di sorgente occupano cosi' una singola riga di legenda, gli
# elementi di stile A/B/intersezione vanno a capo automaticamente).
fig.legend(legend_handles, legend_labels, loc="lower center",
           bbox_to_anchor=(0.5, 1.0), ncol=len(sources),
           fontsize=7, framealpha=0.85)

for ax in axes[:-1].flatten():
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("images/rpm_projections_3x3.pdf", bbox_inches="tight")
#plt.show()