"""
pif_timescales_catalog.py

  Pannello 1: M = M_ref fissata (dal mass_range del catalogo, o valore
      generico AGN se assente), spin variabile -> analogo di Fig. 7
      di Motta et al. 2018.
  Pannello 2: a = A_FIXED fissato, M variabile nel range appropriato
      alla sorgente (banda riportata, range esplorativo attorno a una
      stima puntuale, o range AGN generico).

In ogni caso NU_TARGET = nu0 della sorgente 

Gestione di mass_range secondo il docstring di catalog.py:
  - None                -> nessuna stima indipendente. Pannello 1: M_ref
                            = media geometrica di (M_AGN_MIN, M_AGN_MAX).
                            Pannello 2: scan sull'intero range generico.
  - (lo, hi) con lo==hi  -> stima puntuale. Pannello 1: M_ref = lo.
                            Pannello 2: scan ESPLORATIVO di +-1 dex
                            attorno al valore puntuale (assunzione
                            dichiarata: NON e' un intervallo di
                            confidenza sulla stima, va letto solo come
                            sensitivity check).
  - (lo, hi) con lo<hi   -> banda riportata in tesi. Pannello 1: M_ref
                            = media geometrica di (lo, hi). Pannello 2:
                            scan esattamente su (lo, hi).
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import r_isco, set_style, fix_spines, M_AGN_MIN, M_AGN_MAX
from disk_profiles import t_wave_closed
from align_timescale import t_align_vect
from catalog import CATALOG, select_sources

set_style()

# --- parametri fissi (assunzioni dichiarate, come nello script originale) ---
ALPHA = 0.01
A_VALS = np.array([-0.9, -0.5, 0.0, 0.5, 0.9, 0.998])   # pannello 1
A_FIXED = 0.5                                              # pannello 2
N_M = 6                                                     # n. curve M nel pannello 2
R_OUT_GRID = np.logspace(0, np.log10(200.0), 100)
EXPLORATORY_DEX = 1.0   # +-1 ordine di grandezza attorno a stima puntuale


def resolve_mass_reference(mass_range):
    """
    Da mass_range del catalogo ricava:
      M_ref  : massa singola per il pannello 1
      M_lo, M_hi : estremi dello scan per il pannello 2
      kind   : 'point' | 'band' | 'generic' | 'exploratory'
                (per annotare correttamente il pannello 2)
    """
    if mass_range is None:
        M_lo, M_hi = M_AGN_MIN, M_AGN_MAX
        M_ref = np.sqrt(M_lo * M_hi)
        return M_ref, M_lo, M_hi, 'generic'

    lo, hi = mass_range
    if lo == hi:
        M_ref = lo
        M_lo = lo / 10**EXPLORATORY_DEX
        M_hi = lo * 10**EXPLORATORY_DEX
        return M_ref, M_lo, M_hi, 'exploratory'

    M_ref = np.sqrt(lo * hi)
    return M_ref, lo, hi, 'band'


def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def _panel1_fixed_mass(ax, M_ref, nu_target):
    colors_a = plt.cm.viridis(np.linspace(0, 1, len(A_VALS)))
    for a, col in zip(A_VALS, colors_a):
        r_in = r_isco(a)
        mask = R_OUT_GRID > r_in * 1.001
        r_out_v = R_OUT_GRID[mask]
        if r_out_v.size == 0:
            continue
        r_in_arr = np.full_like(r_out_v, r_in)
        a_arr = np.full_like(r_out_v, a)
        M_arr = np.full_like(r_out_v, M_ref)

        t_wave = t_wave_closed(r_in_arr, r_out_v, a_arr, M_arr)
        t_align = t_align_vect(a_arr, r_in_arr, r_out_v, M_arr,
                                np.full(r_out_v.size, ALPHA))

        ax.plot(r_out_v, t_wave, color=col, lw=1, ls=':')
        ax.plot(r_out_v, t_align, color=col, lw=1, ls='--',
                 label=fr"$a={a:.2f}$")

    ax.axhline(1.0 / nu_target, color='black', ls='-', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, R_OUT_GRID[-1])
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    ax.set_ylabel("Tempo [s]")
    ax.legend(loc='upper right', frameon=True, fontsize=6.5, ncol=2)


def _panel2_fixed_spin(ax, M_lo, M_hi, nu_target):
    r_in_fixed = r_isco(A_FIXED)
    mask2 = R_OUT_GRID > r_in_fixed * 1.001
    r_out_v2 = R_OUT_GRID[mask2]

    M_vals = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M)
    colors_M = plt.cm.plasma(np.linspace(0, 1, len(M_vals)))

    for M, col in zip(M_vals, colors_M):
        r_in_arr = np.full_like(r_out_v2, r_in_fixed)
        a_arr = np.full_like(r_out_v2, A_FIXED)
        M_arr = np.full_like(r_out_v2, M)

        t_wave = t_wave_closed(r_in_arr, r_out_v2, a_arr, M_arr)
        t_align = t_align_vect(a_arr, r_in_arr, r_out_v2, M_arr,
                                np.full(r_out_v2.size, ALPHA))

        ax.plot(r_out_v2, t_wave, color=col, lw=1, ls=':')
        ax.plot(r_out_v2, t_align, color=col, lw=1, ls='--',
                 label=fr"$M={M:.1e}\,M_\odot$")

    ax.axhline(1.0 / nu_target, color='black', ls='-', lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, R_OUT_GRID[-1])
    ax.set_xlabel(r"$R_{\rm out}$ [$R_g$]")
    ax.legend(loc='upper right', frameon=True, fontsize=6.5, ncol=2)


def plot_source_timescales(source, outdir='.'):
    """
    Genera e salva la figura a due pannelli per una singola sorgente del
    catalogo. Restituisce il path del file salvato.
    """
    nu0 = source['nu0']
    name = source['name']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])

    kind_label = {
        'generic':     "M generica AGN (nessuna stima indipendente)",
        'point':       "M stima puntuale",
        'band':        "M da banda riportata in tesi",
        'exploratory': fr"M stima puntuale $\pm${EXPLORATORY_DEX:.0f} dex (esplorativo)",
    }[kind]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    fix_spines(ax1)
    fix_spines(ax2)

    _panel1_fixed_mass(ax1, M_ref, nu0)
    ax1.set_title(fr"$M_{{\rm ref}} = {M_ref:.2e}\,M_\odot$ fissata"
                   "\n" + kind_label, fontsize=8)

    _panel2_fixed_spin(ax2, M_lo, M_hi, nu0)
    ax2.set_title(fr"$a={A_FIXED:.2f}$ fissato, $M \in [{M_lo:.1e}, {M_hi:.1e}]\,M_\odot$",
                   fontsize=8)

    style_handles = [
        Line2D([], [], color='gray', lw=1, ls=':', label=r'$t_{\rm wave}$'),
        Line2D([], [], color='gray', lw=1, ls='--', label=r'$t_{\rm align}$'),
        Line2D([], [], color='black', lw=1.5, ls='-', label=r'$P_{\rm osc}=1/\nu_0$'),
    ]
    fig.suptitle(fr"{name}  —  $\nu_0 = {nu0:.2e}$ Hz", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=3,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fname = os.path.join(outdir, f"pif_timescales_{safe_filename(name)}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_catalog(indices=None, outdir='.'):
    """
    Genera le figure per un sottoinsieme del catalogo (o per tutto il
    catalogo se indices=None), NELL'ORDINE dato da indices (se fornito),
    altrimenti nell'ordine canonico del catalogo.
    """
    sources = select_sources(indices) if indices is not None else CATALOG
    saved = []
    for source in sources:
        fname = plot_source_timescales(source, outdir=outdir)
        print(f"Salvato: {fname}")
        saved.append(fname)
    return saved


if __name__ == "__main__":
    os.makedirs("output_catalog", exist_ok=True)
    plot_catalog(indices=[0,9,10,11], outdir="output_catalog")