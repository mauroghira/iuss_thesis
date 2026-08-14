"""
pif_timescales_universal.py

Pannello "universale" (Soluzione A) per l'intero catalogo di sorgenti QPO,
in sostituzione dei 12 grafici a due pannelli di pif_timescales.py.

Scaling esatta sfruttata (verificata dall'utente su setup.py/nu_solid_v2.py)
------------------------------------------------------------------------
    t_wave(a, r_out, M)                = M         * tau_hat_wave(a, r_out)
    t_align(a, r_in, r_out, M, alpha)  = (M/alpha) * tau_hat_align(a, r_in, r_out)

con tau_hat_wave, tau_hat_align funzioni puramente geometriche (indipendenti
da M, alpha e dalla sorgente). Si ottengono chiamando le funzioni ESISTENTI
t_wave_closed / t_align_vect con M=1 (e alpha=1 per t_align_vect): nessuna
riscrittura della fisica, solo lettura della scaling gia' stabilita.

Asse universale
----------------
Definendo y = t * nu0 (adimensionale) e x = M * nu0 (adimensionale, un solo
numero per sorgente):

    y_wave(r_out)  = x            * tau_hat_wave(a, r_out)
    y_align(r_out) = (x / ALPHA)  * tau_hat_align(a, r_in, r_out)

La condizione P_osc = 1/nu0 diventa y = 1: STESSA soglia per tutte le
sorgenti, quindi un'unica linea orizzontale su un unico pannello, al posto
di un axhline diverso per ognuna delle 12 figure originali.

Costo computazionale
---------------------
tau_hat_wave e tau_hat_align vengono calcolate UNA sola volta su una
griglia (a, r_out) comune a tutte le sorgenti (A_VALS x R_OUT_GRID, stessa
griglia del pannello 1 di pif_timescales.py). Il contributo di ciascuna
sorgente e' poi una singola moltiplicazione per lo scalare x = M_ref*nu0
(o x/ALPHA): O(N_sorgenti * N_rout), niente integrazioni ripetute.

M_ref e' risolto con la STESSA logica gia' usata nel pannello 1 di
pif_timescales.py (resolve_mass_reference): media geometrica della banda
riportata in tesi, stima puntuale, o media geometrica del range AGN
generico se non c'e' stima indipendente.
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import r_isco, set_style, fix_spines
from disk_profiles import t_wave_closed
from align_timescale import t_align_vect
from catalog import CATALOG, select_sources
from pif_timescales import resolve_mass_reference, ALPHA, A_VALS, A_FIXED, R_OUT_GRID

set_style()

A_REPR = A_FIXED  # spin rappresentativo per la curva principale (coerente col pannello 2 originale)

# Margine di esclusione ESATTAMENTE come nello script originale
# (pif_timescales.py, mask = R_OUT_GRID > r_in*1.001): serve solo a
# evitare il punto puntualmente singolare R_out=r_in, non a tagliare un
# intorno piu' ampio.
#
# NOTA IMPORTANTE (verificata numericamente, vedi log della sessione):
# t_align diverge per R_out -> r_in^+ come una VERA legge di potenza,
# t_align ~ (R_out - r_in)^-n con n~1.5-2 (pendenza log-log misurata:
# da -1.94 a -1.49 su un intervallo di (R_out-r_in) da 0.01 a 0.16 Rg,
# a=0.5). Non e' un artefatto numerico isolato ma un comportamento
# fisico genuino del modello (un anulus radiale che collassa a
# larghezza nulla non fornisce leva al torque di allineamento viscoso).
# Di conseguenza NON va escluso un intorno ampio (farlo scarterebbe
# informazione fisica reale sul tempo di allineamento vicino all'ISCO,
# proprio la regione piu' rilevante per il quesito "perche' poche QPO
# coerenti"); si preferisce invece limitare la scala VISIBILE dell'asse
# y (Y_CLIP sotto), dichiarandolo esplicitamente in figura, senza
# rimuovere le curve dai dati.
R_OUT_MARGIN = 1.001
Y_CLIP = (1e-4, 1e6)  # range visibile dell'asse y (clipping, non esclusione)


def _tau_hat_grid(a_vals, r_out_grid):
    """
    tau_hat_wave[i, k] = t_wave_closed(r_in(a_i), r_out_k, a_i, M=1)
    tau_hat_align[i, k] = t_align_vect(a_i, r_in(a_i), r_out_k, M=1, alpha=1)

    Indipendenti dalla sorgente per costruzione (vedi scaling nel docstring
    del modulo). Punti con r_out <= r_isco(a)*R_OUT_MARGIN restano NaN:
    esclude il regime realmente singolare di t_align vicino a r_in (vedi
    commento su R_OUT_MARGIN), non solo il punto esattamente singolare.

    Ritorna (tau_wave, tau_align), shape (len(a_vals), len(r_out_grid)).
    """
    n_a, n_r = len(a_vals), len(r_out_grid)
    tau_wave = np.full((n_a, n_r), np.nan)
    tau_align = np.full((n_a, n_r), np.nan)

    for i, a in enumerate(a_vals):
        r_in = r_isco(a)
        mask = r_out_grid > r_in * R_OUT_MARGIN
        r_out_v = r_out_grid[mask]
        if r_out_v.size == 0:
            continue

        r_in_arr = np.full_like(r_out_v, r_in)
        a_arr = np.full_like(r_out_v, a)
        M_arr = np.ones_like(r_out_v)
        alpha_arr = np.ones_like(r_out_v)

        tau_wave[i, mask] = t_wave_closed(r_in_arr, r_out_v, a_arr, M_arr)
        tau_align[i, mask] = t_align_vect(a_arr, r_in_arr, r_out_v, M_arr, alpha_arr)

    return tau_wave, tau_align


def plot_universal_panel(indices=None, outdir='.', fname='pif_timescales_universal.pdf'):
    """
    Un solo pannello log-log, x = r_out [Rg], y = t*nu0 (adimensionale).

    Per ogni sorgente:
      - curva a spin rappresentativo A_REPR: tratto puntinato per
        t_wave*nu0, tratteggiato per t_align*nu0 (stessa convenzione
        stilistica dello script originale);
      - banda ombreggiata = inviluppo min/max su tutti gli spin in
        A_VALS (sensitivita' allo spin, sostituisce le 6 curve a colore
        del pannello 1 originale).

    Linea nera orizzontale y=1: condizione P_osc = 1/nu0, identica per
    tutte le sorgenti.
    """
    sources = select_sources(indices) if indices is not None else CATALOG
    os.makedirs(outdir, exist_ok=True)

    a_grid_vals = np.union1d(A_VALS, [A_REPR])
    tau_wave_grid, tau_align_grid = _tau_hat_grid(a_grid_vals, R_OUT_GRID)
    idx_repr = int(np.where(a_grid_vals == A_REPR)[0][0])

    fig, ax = plt.subplots(figsize=(7, 5))
    fix_spines(ax)

    colors = plt.cm.tab20(np.linspace(0, 1, len(sources)))

    for source, col in zip(sources, colors):
        nu0 = source['nu0']
        M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])
        x_ref = M_ref * nu0  # adimensionale

        y_wave_all = x_ref * tau_wave_grid
        y_align_all = (x_ref / ALPHA) * tau_align_grid

        y_wave_repr = y_wave_all[idx_repr]
        y_align_repr = y_align_all[idx_repr]

        """
        with warnings.catch_warnings():
            # "All-NaN slice": atteso per r_out sotto r_isco(a) per
            # qualche spin della griglia; il NaN risultante e' corretto
            # e va escluso dal plot, non e' un errore di calcolo (stessa
            # logica di param_space.extremal_over_axis).
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            y_wave_min = np.nanmin(y_wave_all, axis=0)
            y_wave_max = np.nanmax(y_wave_all, axis=0)
            y_align_min = np.nanmin(y_align_all, axis=0)
            y_align_max = np.nanmax(y_align_all, axis=0)

        ax.fill_between(R_OUT_GRID, y_wave_min, y_wave_max, color=col, alpha=0.08, lw=0)
        ax.fill_between(R_OUT_GRID, y_align_min, y_align_max, color=col, alpha=0.08, lw=0)
        """

        ax.plot(R_OUT_GRID, y_wave_repr, color=col, lw=1, ls=':')
        ax.plot(R_OUT_GRID, y_align_repr, color=col, lw=1.2, ls='--', label=source['name'])

    ax.axhline(1.0, color='black', lw=1.5, ls='-')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, R_OUT_GRID[-1])
    #ax.set_ylim(*Y_CLIP)
    ax.set_xlabel(r"$R_{\rm out}$  [$R_g$]")
    ax.set_ylabel(r"$t \cdot \nu_0$  (adimensionale)")
    ax.set_title(
        fr"Confronto universale — curva a $a={A_REPR:.2f}$, banda $a\in[{A_VALS.min():.2f},{A_VALS.max():.2f}]$, "
        fr"$\alpha={ALPHA}$",
        fontsize=7.5)

    style_handles = [
        Line2D([], [], color='gray', lw=1, ls=':', label=r'$t_{\rm wave}\cdot\nu_0$'),
        Line2D([], [], color='gray', lw=1.2, ls='--', label=r'$t_{\rm align}\cdot\nu_0$'),
        Line2D([], [], color='black', lw=1.5, ls='-', label=r'$P_{\rm osc}\cdot\nu_0=1$'),
    ]
    leg1 = ax.legend(handles=style_handles, loc='lower left', frameon=True, fontsize=7)
    ax.add_artist(leg1)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True,
              fontsize=6.5, title='Sorgente', ncol=1)

    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, bbox_inches='tight')
    plt.close(fig)
    return fpath


if __name__ == "__main__":
    out = plot_universal_panel(indices=[0,9,10,11], outdir="output_catalog")
    print(f"Salvato: {out}")