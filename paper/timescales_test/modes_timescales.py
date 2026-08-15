"""
modes_timescales_all.py

Generalizzazione di modes_timescales.py all'intero catalogo QPO.

Lo script originale confrontava, per UNA sorgente (J1257) a spin e massa
fissati, t_visc_Kato(r) (Eq. 73 semplificata) con t_wave e P_osc=1/nu0
sulla finestra di trapping del p-mode fondamentale (m=0, n=0), la cui
frontiera esterna r1 e' definita da kappa(r1) = nu0 (Kato, Sez. 5.1,
Fig. 6: al di fuori della co-risonanza kappa=omega il modo non e' piu'
intrappolato).

Generalizzazioni introdotte
----------------------------
1. Massa (pannello 1): M_BH_TEST=2e6 (valore singolo, arbitrario) ->
   M_ref risolta da resolve_mass_reference(mass_range) (stessa
   funzione, stessa logica generic/point/band/exploratory gia' usata in
   pif_timescales.py e pif_timescales_all.py).

   CORREZIONE rispetto a una versione precedente di questo docstring:
   qui viene inizialmente affermato che uno scan esplicito in M fosse
   ridondante perche' kappa~1/M (kappa=g_r(r,a)/M, g_r puramente
   geometrico, vedi setup.py) sposterebbe solo la SOGLIA kappa_max
   proporzionalmente. Questo e' vero per kappa_max, ma NON per il
   confronto con P_osc=1/nu0: il bordo r1 e' definito da
   g_r(r1,a) = M*nu0 (dipende esplicitamente da M attraverso la soglia
   M*nu0, non solo da a), e t_visc_Kato(r) tramite growth_rate_p_mode
   eredita a sua volta una dipendenza da M attraverso il fattore
   dimensionale Rg(M) (stessa struttura di t_align in
   align_timescale.py, dove compare un fattore Rg^2 ~ M^2). Quindi
   t_visc(r)/P_osc NON e' invariante sotto scaling di M a spin fissato:
   uno scan in M e' fisicamente informativo, non ridondante. Aggiunto
   qui sotto come pannello 2, analogo a _panel2_fixed_spin di
   pif_timescales.py.

2. Spin (pannello 1): A_SPIN=0.5 (valore singolo) -> scan su A_VALS
   (stessa griglia di spin del pannello 1 di pif_timescales.py:
   [-0.9,-0.5,0,0.5,0.9,0.998]), colorata con viridis, ESATTAMENTE come
   gia' fatto in _panel1_fixed_mass di pif_timescales.py, a M=M_ref
   fissata. Per ogni spin la finestra di trapping [r_isco(a), r1(a)] e
   r1(a) stesso sono, in generale, DIVERSI (r_isco(a) e kappa(r,a)
   dipendono entrambi da a), quindi non e' possibile sovrapporre le
   curve su un asse r comune senza prima calcolare la finestra
   spin-per-spin: e' esattamente cio' che questo script fa nel loop.

2bis. Massa (pannello 2, NUOVO): scan su N_M valori di M in
   [M_lo, M_hi] (stessi estremi di resolve_mass_reference, stessa
   logica gia' usata in _panel2_fixed_spin di pif_timescales.py), a
   spin FISSATO A_FIXED=0.5 (stesso valore rappresentativo usato in
   pif_timescales.py, per coerenza tra le figure del catalogo:
   r_isco(A_FIXED) e quindi r_in e' lo STESSO per tutte le curve di
   questo pannello, a differenza del pannello 1 dove r_in varia con lo
   spin). Colormap plasma, come in _panel2_fixed_spin.

3. r_scan_max: 200.0 (arbitrario, specifico del vecchio script) -> R_MAX
   (=500, da setup.py), il cutoff radiale GIA' usato come limite fisico
   in param_space.py e projection_plots.py. Elimina un secondo cutoff
   scollegato dal resto del codice senza introdurne uno nuovo.

4. ALPHA: rinominata ALPHA_PMODE=0.1 e mantenuta DISTINTA da ALPHA=0.01
   di pif_timescales.py. Sono due parametri fisicamente diversi (la
   viscosita' turbolenta di Kato che regola il growth-rate radiativo
   del p-mode, vs. la viscosita' di Foucart-Lai che regola t_align nel
   PIF): un import diretto di ALPHA da pif_timescales.py qui avrebbe
   silenziosamente sostituito il valore 0.1 di Kato con 0.01, alterando
   il growth-rate calcolato senza alcun avviso. Tenerle distinte non e'
   opzionale.

5. Caso "nessuna finestra di trapping" (nu0 >= kappa_max(a, M_ref)):
   nello script originale terminava l'esecuzione con un print e nessun
   grafico. Qui lo spin viene escluso singolarmente (non l'intera
   sorgente) e riportato esplicitamente in console; se NESSUNO spin in
   A_VALS ammette una finestra, la sorgente viene saltata (nessun file
   vuoto/fuorviante prodotto), analogamente alla gestione di casi nulli
   in pif_forest.py.

Dipendenza non verificabile in questa sessione
-------------------------------------------------
growth_rate_p_mode e find_p_mode_outer_boundary sono importate da
growth_rate.py, che NON e' tra i file del progetto disponibili qui: non
ne conosco l'implementazione interna. La generalizzazione sotto riusa
ESCLUSIVAMENTE la firma gia' dimostrata in modes_timescales.py:

    find_p_mode_outer_boundary(a, M, nu_target, r_scan_max=..., n_scan=...) -> r1 (float)
    growth_rate_p_mode(r_grid, a, M, nu_target, alpha) -> array, stesso shape di r_grid

Se la firma reale differisse, e' necessario segnalarlo: qui non viene
fatta alcuna assunzione ulteriore sul loro contenuto fisico.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import r_isco, nu_r, set_style, fix_spines, R_MAX
from disk_profiles import t_wave_closed
from growth_rate import growth_rate_p_mode, find_p_mode_outer_boundary
from catalog import CATALOG, select_sources
from pif_timescales import resolve_mass_reference, A_VALS, A_FIXED, safe_filename

set_style()

# --- parametri fissi per lo scan spin-per-sorgente -------------------
ALPHA_PMODE = 0.1     # viscosita' turbolenta di Kato (Fig. 12 Kato): DISTINTA
                       # da ALPHA=0.01 di pif_timescales.py (t_align, Foucart-Lai)
R_SCAN_MAX = R_MAX    # =500, cutoff radiale gia' usato in param_space.py
N_SCAN = 20000         # stessa risoluzione dello scan dello script originale
N_M = 6                # n. curve di massa nel pannello 2, come pif_timescales.py


def _trapping_window(a, M, nu0, r_scan_max=R_SCAN_MAX, n_scan=N_SCAN):
    """
    Per (a, M) fissati, restituisce:
        (r_in, r1, kappa_max) se la finestra di trapping esiste
        (r_in, None, kappa_max) se non esiste (nu0 >= kappa_max)

    kappa_max = max_r nu_r(r, a, M) sullo scan radiale [r_in*1.001,
    r_scan_max]: e' il valore di soglia oltre cui, per definizione della
    co-risonanza kappa(r1)=nu0 (Kato), nessun r1 puo' esistere se
    nu0 >= kappa_max (la curva kappa(r) non raggiunge mai nu0 dall'alto).

    Usata da entrambi i pannelli: nel pannello 1 M e' fissato (=M_ref) e
    a varia; nel pannello 2 a e' fissato (=A_FIXED) e M varia. La
    funzione e' identica nei due casi, cambia solo quale dei due
    argomenti viene tenuto costante dal chiamante.
    """
    r_in = r_isco(a)
    r_scan = np.geomspace(r_in * 1.001, r_scan_max, n_scan)
    kappa_max = nu_r(r_scan, a, M).max()
    if nu0 >= kappa_max:
        return r_in, None, kappa_max
    r1 = find_p_mode_outer_boundary(a, M, nu0,
                                     r_scan_max=r_scan_max, n_scan=n_scan)
    return r_in, r1, kappa_max


def _panel_spin_scan(ax, M_ref, nu0, n_rgrid=300):
    """
    Pannello 1: M=M_ref fissata, scan sullo spin su A_VALS (viridis).
    Ogni curva t_visc_Kato(r) e ogni segmento t_wave sono limitati al
    proprio dominio [r_in(a), r1(a)] (diverso per ogni spin, vedi
    _trapping_window). Ritorna (any_valid, excluded) per il logging.
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(A_VALS)))
    excluded = []
    any_valid = False

    for a, col in zip(A_VALS, colors):
        r_in, r1, kappa_max = _trapping_window(a, M_ref, nu0)
        if r1 is None:
            excluded.append((a, kappa_max))
            continue

        r_grid = np.linspace(r_in * 1.001, r1 * 0.999, n_rgrid)
        G = growth_rate_p_mode(r_grid, a, M_ref, nu0, ALPHA_PMODE)
        t_visc = 1.0 / np.abs(G)
        t_wave = t_wave_closed(r_in, r1, a, M_ref)

        ax.plot(r_grid, t_visc, color=col, lw=1.3, ls='--',
                label=fr"$a={a:.2f}$")
        ax.plot([r_in, r1], [t_wave, t_wave], color=col, ls=':', lw=1.3)
        ax.axvline(r_in, color=col, ls=':', lw=0.5, alpha=0.35)
        any_valid = True

    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylabel("Tempo [s]")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True,
              fontsize=6, title='Spin', ncol=1)
    return any_valid, excluded


def _panel_mass_scan(ax, M_lo, M_hi, nu0, n_rgrid=300):
    """
    Pannello 2: a=A_FIXED fissato, scan su N_M valori di M in
    [M_lo, M_hi] (plasma). r_in = r_isco(A_FIXED) e' lo STESSO per
    tutte le curve (non dipende da M); r1(M) invece varia con M perche'
    la soglia della co-risonanza e' g_r(r1,a) = M*nu0 (vedi docstring
    del modulo). Ritorna (any_valid, excluded) per il logging.
    """
    M_vals = np.logspace(np.log10(M_lo), np.log10(M_hi), N_M)
    colors = plt.cm.plasma(np.linspace(0, 1, len(M_vals)))
    excluded = []
    any_valid = False

    for M, col in zip(M_vals, colors):
        r_in, r1, kappa_max = _trapping_window(A_FIXED, M, nu0)
        if r1 is None:
            excluded.append((M, kappa_max))
            continue

        r_grid = np.linspace(r_in * 1.001, r1 * 0.999, n_rgrid)
        G = growth_rate_p_mode(r_grid, A_FIXED, M, nu0, ALPHA_PMODE)
        t_visc = 1.0 / np.abs(G)
        t_wave = t_wave_closed(r_in, r1, A_FIXED, M)

        ax.plot(r_grid, t_visc, color=col, lw=1.3, ls='--',
                label=fr"$M={M:.1e}\,M_\odot$")
        ax.plot([r_in, r1], [t_wave, t_wave], color=col, ls=':', lw=1.3)
        any_valid = True

    if any_valid:
        # r_in e' comune a tutte le curve (a fissato): una sola volta
        ax.axvline(r_isco(A_FIXED), color='0.4', ls=':', lw=0.5, alpha=0.5)

    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True,
              fontsize=6, title='Massa', ncol=1)
    return any_valid, excluded


def plot_source_p_mode(source, outdir='.'):
    """
    Genera, per una singola sorgente del catalogo, una figura a due
    pannelli affiancati (sharey, scala log):
      pannello 1: M=M_ref fissata, scan sullo spin (A_VALS);
      pannello 2: a=A_FIXED fissato, scan sulla massa (M_lo..M_hi).
    In entrambi, P_osc=1/nu0 e' l'unica retta nera orizzontale (dipende
    solo da nu0, fissata per la sorgente, non da a o M).

    Restituisce il path del file salvato, o None se NESSUno dei due
    pannelli ammette almeno una finestra di trapping (nessun file
    prodotto).
    """
    nu0 = source['nu0']
    name = source['name']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])

    kind_label = {
        'generic':     "M generica AGN (nessuna stima indipendente)",
        'point':       "M stima puntuale",
        'band':        "M da banda riportata in tesi",
        'exploratory': "M stima puntuale +-1 dex esplorativo",
    }[kind]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    fix_spines(ax1)
    fix_spines(ax2)

    valid1, excl1 = _panel_spin_scan(ax1, M_ref, nu0)
    valid2, excl2 = _panel_mass_scan(ax2, M_lo, M_hi, nu0)

    print(f"[{name}] M_ref={M_ref:.3e} Msun ({kind}), nu0={nu0:.3e} Hz")
    for a, kmax in excl1:
        print(f"  pannello 1: spin a={a:+.3f} escluso "
              f"(kappa_max={kmax:.3e} Hz <= nu0)")
    for M, kmax in excl2:
        print(f"  pannello 2: M={M:.3e} Msun esclusa "
              f"(kappa_max={kmax:.3e} Hz <= nu0)")

    if not (valid1 or valid2):
        print(f"  ATTENZIONE: nessuna finestra di trapping del p-mode "
              f"fondamentale per {name} in nessuno dei due scan "
              f"-- sorgente saltata, nessun file prodotto.")
        plt.close(fig)
        return None

    P_osc = 1.0 / nu0
    ax1.axhline(P_osc, color='black', lw=1.5, ls='-')
    ax2.axhline(P_osc, color='black', lw=1.5, ls='-')

    ax1.set_yscale('log')  # sharey=True: basta impostarla su un asse
    ax1.set_title(fr"$M_{{\rm ref}}={M_ref:.2e}\,M_\odot$ fissata"
                  "\n" + kind_label, fontsize=8)
    ax2.set_title(fr"$a={A_FIXED:.2f}$ fissato, "
                  fr"$M\in[{M_lo:.1e},{M_hi:.1e}]\,M_\odot$", fontsize=8)

    style_handles = [
        Line2D([], [], color='gray', lw=1.3, ls='--',
               label=r"$t_{\rm visc}^{\rm Kato}(r)$"),
        Line2D([], [], color='gray', lw=1.3, ls=':',
               label=r"$t_{\rm wave}$ (su $[r_{\rm in},r_1]$)"),
        Line2D([], [], color='black', lw=1.5, ls='-',
               label=r"$P_{\rm osc}=1/\nu_0$"),
    ]
    fig.suptitle(fr"p-mode fondamentale — {name}, $\nu_0={nu0:.2e}$ Hz, "
                 fr"$\alpha_{{\rm Kato}}={ALPHA_PMODE}$", fontsize=10)
    fig.legend(handles=style_handles, loc='lower center', ncol=3,
               frameon=True, fontsize=7, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    fname = os.path.join(outdir, f"modes_timescales_{safe_filename(name)}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname


def plot_catalog(indices=None, outdir='.'):
    """
    Genera le figure per un sottoinsieme del catalogo (o per tutto il
    catalogo se indices=None), nell'ordine dato da indices se fornito.
    Sorgenti senza alcuna finestra di trapping in A_VALS sono saltate
    (vedi plot_source_p_mode) e riportate in console, non silenziate.
    """
    sources = select_sources(indices) if indices is not None else CATALOG
    saved = []
    for source in sources:
        fname = plot_source_p_mode(source, outdir=outdir)
        if fname is not None:
            print(f"Salvato: {fname}")
            saved.append(fname)
    return saved


if __name__ == "__main__":
    os.makedirs("output_modes", exist_ok=True)
    plot_catalog(outdir="output_modes")