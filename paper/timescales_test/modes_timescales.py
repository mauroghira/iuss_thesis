"""
modes_timescales.py

Generalizzazione a QUALUNQUE modo diskoseismico (p, g, c; qualunque
(m,n)) dei due grafici gia' introdotti per il solo p-mode fondamentale:

  Pannello 1 : M = M_ref fissata (dal catalogo), scan sullo spin (A_VALS)
  Pannello 2 : a = A_FIXED fissato, scan sulla massa (M_lo..M_hi)

In entrambi: per ciascun valore scansionato, si disegna il tasso di
smorzamento/crescita convertito in tempo (curva r-dipendente per il
p-mode, retta costante per il c-mode -- l'interfaccia generica di
mode_registry.ModeRequest nasconde la differenza) e il segmento
orizzontale t_wave sulla stessa finestra radiale; P_osc=1/nu0 e' l'unica
retta nera, identica per tutti i modi di una sorgente (dipende solo da
nu0).

Selezione dei modi da plottare
--------------------------------
Si passa una lista di mode_registry.ModeRequest a plot_catalog/
plot_source_all_modes, es.:

    from mode_registry import ModeRequest
    modes = [ModeRequest('p', 0, 0), ModeRequest('c', 1, 1), ModeRequest('c', 2, 1)]

Layout: combine=True impila i pannelli di TUTTI i modi richiesti (che
ammettono almeno una finestra) uno sotto l'altro in UN'UNICA figura per
sorgente (N_modi righe x 2 colonne, sharey per riga: scale diverse tra
modi restano leggibili). combine=False produce un file separato per
ogni (sorgente, modo), come nello script originale per il solo p-mode.

Modi senza alcuna finestra di trapping (per nessuno spin/massa
scansionati) vengono omessi riga per riga (non l'intera sorgente), e
riportati esplicitamente in console. Se NESSUN modo richiesto ammette
una finestra per una sorgente, la sorgente e' saltata (nessun file
vuoto/fuorviante), analogamente alla gestione di casi nulli in
pif_forest.py.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import set_style, fix_spines
from disk_profiles import t_wave_closed
from catalog import CATALOG, select_sources
from pif_timescales import resolve_mass_reference, A_VALS, A_FIXED, safe_filename
from modes_registry import ModeRequest

set_style()

N_M = 6   # n. curve di massa nel pannello 2, come pif_timescales.py
N_RGRID_DEFAULT = 300


# ======================================================================
# CALCOLO (separato dal disegno: permette di decidere il layout della
# figura -- quante righe, quali modi validi -- PRIMA di creare gli assi)
# ======================================================================
def _compute_spin_scan_data(mode, M_ref, nu0, n_rgrid=N_RGRID_DEFAULT):
    """Per ciascuno spin in A_VALS (a M=M_ref fissata): finestra, curva
    del tasso convertito in tempo, t_wave. Ritorna (data, excluded)."""
    data, excluded = [], []
    for a in A_VALS:
        window = mode.window(a, M_ref, nu0)
        if not window['valid']:
            excluded.append(a)
            continue
        r_in, r_out = window['r_in'], window['r_out']
        r_grid = np.linspace(r_in * 1.001, r_out * 0.999, n_rgrid)
        t_curve = mode.timescale(r_grid, a, M_ref, nu0, window)
        t_wave = t_wave_closed(r_in, r_out, a, M_ref)
        data.append(dict(a=a, r_grid=r_grid, t_curve=t_curve, t_wave=t_wave,
                          r_in=r_in, r_out=r_out))
    return data, excluded


def _compute_mass_scan_data(mode, M_lo, M_hi, nu0, n_rgrid=N_RGRID_DEFAULT):
    """Per N_M valori di M in [M_lo,M_hi] (ad a=A_FIXED fissato): finestra,
    curva del tasso, t_wave. Ritorna (data, excluded, M_vals) -- M_vals
    serve a valle per costruire una colormap coerente anche in presenza
    di masse escluse."""
    M_vals = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M)
    data, excluded = [], []
    for M in M_vals:
        window = mode.window(A_FIXED, M, nu0)
        if not window['valid']:
            excluded.append(M)
            continue
        r_in, r_out = window['r_in'], window['r_out']
        r_grid = np.linspace(r_in * 1.001, r_out * 0.999, n_rgrid)
        t_curve = mode.timescale(r_grid, A_FIXED, M, nu0, window)
        t_wave = t_wave_closed(r_in, r_out, A_FIXED, M)
        data.append(dict(M=M, r_grid=r_grid, t_curve=t_curve, t_wave=t_wave,
                          r_in=r_in, r_out=r_out))
    return data, excluded, M_vals


# ======================================================================
# DISEGNO (consuma i dati gia' calcolati, nessuna fisica qui dentro)
# ======================================================================
def _draw_spin_panel(ax, data, mode):
    colors = plt.cm.viridis(np.linspace(0, 1, len(A_VALS)))
    color_of = dict(zip(A_VALS, colors))
    for d in data:
        col = color_of[d['a']]
        ax.plot(d['r_grid'], d['t_curve'], color=col, lw=1.3, ls='--',
                label=fr"$a={d['a']:.2f}$")
        ax.plot([d['r_in'], d['r_out']], [d['t_wave'], d['t_wave']],
                color=col, ls=':', lw=1.3)
        ax.axvline(d['r_in'], color=col, ls=':', lw=0.5, alpha=0.35)
    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylabel("Tempo [s]")
    if data:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True,
                  fontsize=6, title='Spin', ncol=1)
    if mode.family['timescale_label']:
        ax.annotate(mode.family['timescale_label'], xy=(0.02, 0.96),
                    xycoords='axes fraction', fontsize=6.5, color='0.3', va='top')


def _draw_mass_panel(ax, data, mode, M_vals):
    colors = plt.cm.plasma(np.linspace(0, 1, len(M_vals)))
    color_of = dict(zip(M_vals, colors))
    for d in data:
        col = color_of[d['M']]
        ax.plot(d['r_grid'], d['t_curve'], color=col, lw=1.3, ls='--',
                label=fr"$M={d['M']:.1e}\,M_\odot$")
        ax.plot([d['r_in'], d['r_out']], [d['t_wave'], d['t_wave']],
                color=col, ls=':', lw=1.3)
    if data:
        ax.axvline(data[0]['r_in'], color='0.4', ls=':', lw=0.5, alpha=0.5)
        # r_in = r_isco(A_FIXED) e' lo stesso per tutte le curve (a fissato)
    ax.set_xlabel(r"$r$ [$R_g$]")
    if data:
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True,
                  fontsize=6, title='Massa', ncol=1)


# ======================================================================
# ORCHESTRAZIONE PER SORGENTE
# ======================================================================
def _prepare_modes_for_source(source, modes, n_rgrid=N_RGRID_DEFAULT):
    """
    Calcola (senza disegnare) i dati di entrambi i pannelli per ciascun
    modo richiesto. Salta silenziosamente (con messaggio in console) i
    modi non implementati (NotImplementedError, es. 'g') e quelli senza
    alcuna finestra di trapping in nessuno dei due pannelli.

    Ritorna: M_ref, M_lo, M_hi, kind, prepared
      prepared = lista di (mode, data1, excl1, data2, excl2, M_vals)
    """
    nu0 = source['nu0']
    name = source['name']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])

    prepared = []
    for mode in modes:
        try:
            data1, excl1 = _compute_spin_scan_data(mode, M_ref, nu0, n_rgrid)
            data2, excl2, M_vals = _compute_mass_scan_data(mode, M_lo, M_hi, nu0, n_rgrid)
        except NotImplementedError as e:
            print(f"[{name}] modo {mode.id} saltato: {e}")
            continue

        print(f"[{name}] modo {mode.id}: pannello spin {len(data1)}/{len(A_VALS)} validi, "
              f"pannello massa {len(data2)}/{N_M} validi")
        for a in excl1:
            print(f"    spin escluso a={a:+.3f} (nessuna finestra di trapping)")
        for Mv in excl2:
            print(f"    massa esclusa M={Mv:.3e} Msun (nessuna finestra di trapping)")

        if not data1 and not data2:
            print(f"    -> nessuna finestra per {mode.id}, riga omessa")
            continue

        prepared.append((mode, data1, excl1, data2, excl2, M_vals))

    return M_ref, M_lo, M_hi, kind, prepared


_KIND_LABEL = {
    'generic':     "M generica AGN (nessuna stima indipendente)",
    'point':       "M stima puntuale",
    'band':        "M da banda riportata in tesi",
    'exploratory': "M stima puntuale +-1 dex esplorativo",
}

_STYLE_HANDLES_BASE = [
    Line2D([], [], color='gray', lw=1.3, ls='--',
           label=r"tasso conv. in tempo (v. etichetta per riga)"),
    Line2D([], [], color='gray', lw=1.3, ls=':', label=r"$t_{\rm wave}$"),
    Line2D([], [], color='black', lw=1.5, ls='-', label=r"$P_{\rm osc}=1/\nu_0$"),
]


def plot_source_all_modes(source, modes, outdir='.', combine=True,
                           n_rgrid=N_RGRID_DEFAULT):
    """
    Genera, per una sorgente, i grafici t_wave/tasso per tutti i `modes`
    richiesti che ammettono almeno una finestra di trapping.

    combine=True  -> UN'UNICA figura, un modo per riga (N_modi x 2 pannelli)
    combine=False -> un file separato per ciascun modo

    Ritorna la lista dei path salvati (vuota se nessun modo era valido).
    """
    nu0 = source['nu0']
    name = source['name']

    M_ref, M_lo, M_hi, kind, prepared = _prepare_modes_for_source(source, modes, n_rgrid)
    kind_label = _KIND_LABEL[kind]

    if not prepared:
        print(f"[{name}] ATTENZIONE: nessun modo richiesto ammette una finestra "
              f"di trapping -- sorgente saltata, nessun file prodotto.")
        return []

    P_osc = 1.0 / nu0
    saved = []

    if combine:
        n_modes = len(prepared)
        fig, axes = plt.subplots(n_modes, 2, figsize=(9.5, 3.6 * n_modes),
                                  sharey='row', squeeze=False)
        for row, (mode, data1, excl1, data2, excl2, M_vals) in enumerate(prepared):
            ax1, ax2 = axes[row]
            fix_spines(ax1); fix_spines(ax2)

            _draw_spin_panel(ax1, data1, mode)
            _draw_mass_panel(ax2, data2, mode, M_vals)

            ax1.axhline(P_osc, color='black', lw=1.5, ls='-')
            ax2.axhline(P_osc, color='black', lw=1.5, ls='-')
            ax1.set_yscale('log')
            ax1.set_ylabel(mode.label + "\nTempo [s]", fontsize=7)

            if row == 0:
                ax1.set_title(fr"$M_{{\rm ref}}={M_ref:.2e}\,M_\odot$ fissata"
                              "\n" + kind_label, fontsize=8)
                ax2.set_title(fr"$a={A_FIXED:.2f}$ fissato, "
                              fr"$M\in[{M_lo:.1e},{M_hi:.1e}]\,M_\odot$", fontsize=8)

        fig.suptitle(fr"{name}, $\nu_0={nu0:.2e}$ Hz", fontsize=10)
        fig.legend(handles=_STYLE_HANDLES_BASE, loc='lower center', ncol=3,
                   frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout(rect=[0, 0.015, 1, 0.95])
        fname = os.path.join(outdir, f"modes_timescales_{safe_filename(name)}_ALL.pdf")
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"Salvato: {fname}")
        saved.append(fname)

    else:
        for mode, data1, excl1, data2, excl2, M_vals in prepared:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
            fix_spines(ax1); fix_spines(ax2)

            _draw_spin_panel(ax1, data1, mode)
            _draw_mass_panel(ax2, data2, mode, M_vals)

            ax1.axhline(P_osc, color='black', lw=1.5, ls='-')
            ax2.axhline(P_osc, color='black', lw=1.5, ls='-')
            ax1.set_yscale('log')
            ax1.set_title(fr"$M_{{\rm ref}}={M_ref:.2e}\,M_\odot$ fissata"
                          "\n" + kind_label, fontsize=8)
            ax2.set_title(fr"$a={A_FIXED:.2f}$ fissato, "
                          fr"$M\in[{M_lo:.1e},{M_hi:.1e}]\,M_\odot$", fontsize=8)

            style_handles = [
                Line2D([], [], color='gray', lw=1.3, ls='--',
                       label=mode.family['timescale_label']),
                Line2D([], [], color='gray', lw=1.3, ls=':', label=r"$t_{\rm wave}$"),
                Line2D([], [], color='black', lw=1.5, ls='-', label=r"$P_{\rm osc}=1/\nu_0$"),
            ]
            fig.suptitle(fr"{mode.label} --- {name}, $\nu_0={nu0:.2e}$ Hz", fontsize=10)
            fig.legend(handles=style_handles, loc='lower center', ncol=3,
                       frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.03))

            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            fname = os.path.join(outdir, f"modes_timescales_{safe_filename(name)}_{mode.id}.pdf")
            plt.savefig(fname, bbox_inches='tight')
            plt.close(fig)
            print(f"Salvato: {fname}")
            saved.append(fname)

    return saved


def plot_catalog(indices=None, modes=None, outdir='.', combine=True):
    """
    Genera le figure per un sottoinsieme del catalogo (o per tutto il
    catalogo se indices=None), nell'ordine dato da indices se fornito.

    modes=None -> default [ModeRequest('p',0,0), ModeRequest('c',1,1)].
    """
    if modes is None:
        modes = [ModeRequest('p', 0, 0), ModeRequest('c', 1, 1)]

    sources = select_sources(indices) if indices is not None else CATALOG
    saved_all = []
    for source in sources:
        saved = plot_source_all_modes(source, modes, outdir=outdir, combine=combine)
        saved_all.extend(saved)
    return saved_all


if __name__ == "__main__":
    os.makedirs("output_modes", exist_ok=True)
    MODES = [ModeRequest('p', 0, 0), ModeRequest('p', 1, 0), ModeRequest('c', 1, 1), ModeRequest('c', 2, 1)]
    plot_catalog(indices=[10], outdir="output_modes", modes=MODES, combine=True)