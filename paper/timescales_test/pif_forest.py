"""
pif_forest_plot.py

Forest plot sull'intero catalogo QPO: per ciascuna sorgente,
un segmento orizzontale che mostra l'intervallo di R_out in cui e'
soddisfatta la condizione di precessione rigida coerente del PIF

    t_align(a, r_in, R_out, M, alpha) >= P_osc = 1/nu0 >= t_wave(a, r_in, R_out, M)

("necessaria" secondo Motta et al. 2017: il flusso interno deve
disallinearsi piu' velocemente di quanto il warp si propaghi, E la
precessione osservata deve rientrare nel tempo di comunicazione radiale
del disco).

Riuso della scaling esatta (stessa di pif_timescales_universal.py)
--------------------------------------------------------------------
Definendo y = t*nu0 (adimensionale) e x = M*nu0:
    y_wave(R_out)  = x           * tau_hat_wave(a, R_out)
    y_align(R_out) = (x / alpha) * tau_hat_align(a, r_in, R_out)
con tau_hat_* geometriche, indipendenti da sorgente/M (vedi
pif_timescales_universal._tau_hat_grid, riusata qui senza duplicazione).
La condizione diventa: y_align(R_out) >= 1 >= y_wave(R_out).

Monotonia in R_out (giustificazione, non assunta)
---------------------------------------------------
- y_wave(R_out) e' crescente in R_out: t_wave_closed contiene il termine
  (r_out^(1+q) - r_in^(1+q))/(1+q), derivata rispetto a r_out uguale a
  r_out^q > 0 per r_out>0; il resto del prefattore non dipende da r_out.
  Quindi y_wave e' strettamente crescente, e y_wave(r_in^+) ~ 0 < 1.
- y_align(R_out) diverge per R_out -> r_in^+ (verificato numericamente
  nella sessione precedente, vedi pif_timescales_all.py) e in pratica
  decresce con R_out: y_align(r_in^+) = +infinito > 1.
Di conseguenza la finestra [r_start, r_end] parte SEMPRE da r_in (per
qualunque spin, entrambe le condizioni sono banalmente soddisfatte li'),
e il bordo destro r_end e' fissato dalla PRIMA delle due soglie che
viene attraversata muovendosi verso R_out crescenti:
    R_w = primo R_out per cui y_wave supera 1 (soglia t_wave)
    R_a = primo R_out per cui y_align scende sotto 1 (soglia t_align)
    r_end = min(R_w, R_a)
"t_wave fallisce per primo" <=> R_w < R_a; "t_align fallisce per primo"
<=> R_a < R_w. Non e' un'assunzione: e' una conseguenza diretta della
monotonia di y_wave e (empiricamente) di y_align, verificata comunque
punto per punto sulla griglia (nessuna estrapolazione).

Inviluppo di spin e regime dominante (nuovo)
-----------------------------------------------
La barra grigia (inviluppo) e l'identificazione di quale condizione
"fallisce per prima" sono ora calcolate su una griglia FITTA di spin
A_DENSE in [-A_THORNE, +A_THORNE] (limite di Thorne 1974, gia' usato
altrove nel codice come limite fisico di spin per accrescimento;
sostituisce l'intervallo matematico [-1,1] richiesto perche' a=+-1 e'
un limite estremale non fisicamente raggiungibile e la formula di
r_isco(a) degenera li'). Per ciascuno spin si classifica il regime
('wave' se R_w<R_a, 'align' se R_a<R_w, 'open' se nessuna delle due
soglie e' raggiunta entro R_OUT_MAX) e si individuano TUTTI i cambi di
regime lungo la griglia (non si assume un singolo spin separatore: lo
si verifica). Il confine tra un blocco 'wave' e uno 'align' e'
raffinato per interpolazione lineare dell'indicatore R_w(a)-R_a(a)
(stesso schema di setup._resonance_radii._find_zero); il confine con
un blocco 'open' resta al punto medio di griglia (l'indicatore non e'
definito in quel regime).
"""

import os
import sys
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import set_style, fix_spines, A_THORNE
from catalog import CATALOG, select_sources
from pif_timescales import resolve_mass_reference, ALPHA, A_FIXED, R_OUT_GRID
from pif_timescales_all import _tau_hat_grid

set_style()

R_OUT_MAX = R_OUT_GRID[-1]  # troncamento arbitrario del disco, non un limite fisico

N_SPIN_DENSE = 401   # risoluzione della griglia di spin (~0.005 in a); ridurre se il runtime e' un problema
A_DENSE = np.linspace(-A_THORNE, A_THORNE, N_SPIN_DENSE)


def _find_window(r_out_grid, y_align, y_wave):
    """
    Blocco contiguo piu' esterno (che tocca l'ultimo punto valido della
    griglia, se esiste) di indici in cui y_align>=1 e y_wave<=1
    contemporaneamente. Nessuna assunzione di monotonia: maschera
    diretta + ricerca dei blocchi contigui via np.diff.

    Ritorna (r_start, r_end, open_ended) o None se la condizione non e'
    mai soddisfatta sulla griglia data. open_ended=True se il blocco
    arriva fino all'ultimo punto della griglia (R_OUT_MAX): la finestra
    reale potrebbe estendersi oltre R_OUT_MAX, che e' un troncamento
    arbitrario, non un confine fisico.
    """
    valid = np.isfinite(y_align) & np.isfinite(y_wave)
    mask_ok = valid & (y_align >= 1.0) & (y_wave <= 1.0)
    if not mask_ok.any():
        return None

    idx = np.where(mask_ok)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    # scelgo il blocco con indice finale minimo (quello piu' vicino a,
    # o che tocca, il bordo destro della griglia): e' il piu'
    # significativo fisicamente, dato che t_wave cresce e t_align cala
    # verso R_out grandi, quindi la finestra "vera" e' quella che si
    # trova all'interno del disco.
    run = min(runs, key=lambda r: r[-1])

    r_start = float(r_out_grid[run[0]])
    r_end = float(r_out_grid[run[-1]])
    open_ended = (run[-1] == len(r_out_grid) - 1)
    return r_start, r_end, open_ended


def _find_threshold(r_out_grid, y, kind):
    """
    Prima soglia y=1 lungo r_out_grid, per interpolazione lineare tra
    campioni consecutivi VALIDI e ADIACENTI in griglia (nessuna
    interpolazione attraverso un buco di invalidita', es. r_out<r_isco).

    kind='rising'  -> cerca il primo attraversamento dal basso verso
                       l'alto (usato per t_wave, che cresce con r_out)
    kind='falling' -> cerca il primo attraversamento dall'alto verso
                       il basso (usato per t_align, che cala con r_out)

    Ritorna NaN se la soglia non viene mai raggiunta entro la griglia
    (condizione soddisfatta, o mai soddisfatta, su tutto il dominio
    esplorato: il caso va distinto a valle dal segno di y agli estremi,
    qui ci si limita a segnalare "nessun attraversamento").
    """
    z = y - 1.0
    valid = np.isfinite(z)
    idx = np.where(valid)[0]
    for k in range(len(idx) - 1):
        i, j = idx[k], idx[k + 1]
        if j != i + 1:
            continue  # salto un buco di invalidita': non interpolo attraverso
        z0, z1 = z[i], z[j]
        if kind == 'rising' and z0 <= 0 < z1:
            r0, r1 = r_out_grid[i], r_out_grid[j]
            return r0 - z0 * (r1 - r0) / (z1 - z0)
        if kind == 'falling' and z0 >= 0 > z1:
            r0, r1 = r_out_grid[i], r_out_grid[j]
            return r0 - z0 * (r1 - r0) / (z1 - z0)
    return np.nan


def _window_and_thresholds_for_source_spin(source, a, a_grid_vals, tau_wave_grid, tau_align_grid):
    """
    Per una sorgente e uno spin fissato, calcola in un solo passaggio:
      window : (r_start, r_end, open_ended) oppure None
      R_w    : soglia di t_wave (NaN se mai raggiunta)
      R_a    : soglia di t_align (NaN se mai raggiunta)
    """
    nu0 = source['nu0']
    M_ref, M_lo, M_hi, kind = resolve_mass_reference(source['mass_range'])
    x_ref = M_ref * nu0

    idx_a = int(np.where(a_grid_vals == a)[0][0])
    y_wave = x_ref * tau_wave_grid[idx_a]
    y_align = (x_ref / ALPHA) * tau_align_grid[idx_a]

    window = _find_window(R_OUT_GRID, y_align, y_wave)
    R_w = _find_threshold(R_OUT_GRID, y_wave, 'rising')
    R_a = _find_threshold(R_OUT_GRID, y_align, 'falling')
    return window, R_w, R_a


def _classify_regime(R_w, R_a):
    """'wave' se la soglia di t_wave e' la piu' vicina (fallisce prima),
    'align' se lo e' quella di t_align, 'open' se nessuna delle due e'
    raggiunta entro R_OUT_MAX (la finestra resta valida fino al
    troncamento arbitrario del disco)."""
    if np.isnan(R_w) and np.isnan(R_a):
        return 'open'
    if np.isnan(R_a):
        return 'wave'
    if np.isnan(R_w):
        return 'align'
    return 'wave' if R_w < R_a else 'align'


def _regime_runs_and_boundaries(a_grid, R_w_arr, R_a_arr):
    """
    Sequenza di blocchi contigui a regime costante lungo a_grid, e
    confini tra blocchi consecutivi (raffinati per interpolazione
    lineare quando il confine e' tra 'wave' e 'align'; altrimenti al
    punto medio di griglia, dato che l'indicatore R_w-R_a non e'
    definito in un blocco 'open').

    Ritorna (runs, boundaries) con runs = lista di (label, i0, i1)
    (indici di inizio/fine blocco in a_grid) e boundaries = lista di
    len(runs)-1 valori di spin.
    """
    cat = np.array([_classify_regime(Rw, Ra) for Rw, Ra in zip(R_w_arr, R_a_arr)])

    runs = []
    start_i = 0
    for i in range(1, len(cat)):
        if cat[i] != cat[i - 1]:
            runs.append((cat[i - 1], start_i, i - 1))
            start_i = i
    runs.append((cat[-1], start_i, len(cat) - 1))

    boundaries = []
    for k in range(len(runs) - 1):
        label_L, _, iL1 = runs[k]
        label_R, iR0, _ = runs[k + 1]
        i, j = iL1, iR0
        if {label_L, label_R} == {'wave', 'align'}:
            g0 = R_w_arr[i] - R_a_arr[i]
            g1 = R_w_arr[j] - R_a_arr[j]
            if np.isfinite(g0) and np.isfinite(g1) and g0 != g1:
                a_bound = a_grid[i] - g0 * (a_grid[j] - a_grid[i]) / (g1 - g0)
            else:
                a_bound = 0.5 * (a_grid[i] + a_grid[j])
        else:
            a_bound = 0.5 * (a_grid[i] + a_grid[j])
        boundaries.append(a_bound)

    return runs, boundaries


MAX_STABLE_RUNS = 5   # oltre questa soglia, i cambi di regime rilevati non sono
                       # attendibili come confini fisici (vedi nota sotto)


def _regime_annotation(runs, boundaries):
    """
    Etichetta compatta su una riga, es. 'align -> wave (a~-0.31)'.
    Se non ci sono cambi di regime sull'intera griglia esplorata, non
    compare alcun valore di spin (non c'e' nulla da separare).

    SALVAGUARDIA (verificata numericamente): se una delle due curve
    (tipicamente t_wave*nu0) resta molto vicina a 1 su un ampio
    intervallo di spin (quasi tangente a P_osc), il numero di
    attraversamenti rilevati NON converge in modo monotono con la
    risoluzione di R_OUT_GRID (verificato: 100 punti -> 58 run, 300 ->
    98, 500 -> 95, 800 -> 70; solo a 2000 punti si stabilizza su un
    singolo regime, a un costo computazionale ~40x superiore). In
    questo caso elencare tutti i confini darebbe un falso senso di
    precisione: si segnala invece esplicitamente l'instabilita', senza
    fingere un singolo spin separatore ne' elencarne decine spuri.
    """
    if len(runs) > MAX_STABLE_RUNS:
        return (fr"instabile ({len(runs)} cambi rilevati): una curva sfiora "
                r"$P_{\rm osc}$ su un ampio $\Delta a$ — confine non affidabile "
                "a questa risoluzione di $R_{\\rm out}$")

    short = {'wave': r'$t_{\rm wave}$', 'align': r'$t_{\rm align}$', 'open': 'open'}
    seq = " → ".join(short[r[0]] for r in runs)
    if boundaries:
        at = ", ".join(f"{b:.2f}" for b in boundaries)
        return f"{seq}  (a≈{at})"
    return seq


def _wrap_regime_label(runs, boundaries, max_chars_per_line=16):
    """
    Come _regime_annotation, ma avvolta su piu' righe SENZA mai
    spezzare un singolo token (es. '$t_{\\rm wave}$') a meta': si
    accumulano token interi fino al budget di caratteri per riga, poi
    si va a capo. Necessario per stare dentro il bordo del grafico
    invece che a destra fuori dagli assi.
    """
    if len(runs) > MAX_STABLE_RUNS:
        tokens = ["instabile", f"({len(runs)} cambi):", "confine non", "affidabile"]
    else:
        short = {'wave': r'$t_{\rm wave}$', 'align': r'$t_{\rm align}$', 'open': 'open'}
        tokens = []
        for i, r in enumerate(runs):
            tokens.append(short[r[0]])
            if i < len(runs) - 1:
                tokens.append('→')
        if boundaries:
            at = ", ".join(f"{b:.2f}" for b in boundaries)
            tokens.append(f"(a≈{at})")

    lines, cur = [], ""
    for tok in tokens:
        candidate = (cur + " " + tok).strip()
        if len(candidate) > max_chars_per_line and cur:
            lines.append(cur)
            cur = tok
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _envelope_and_regime_for_source(source, a_grid_vals, tau_wave_grid, tau_align_grid):
    """
    Calcola, sulla griglia fitta A_DENSE:
      - inviluppo (min r_start, max r_end) su tutti gli spin per cui
        esiste una finestra;
      - la sequenza di regimi 'wave'/'align'/'open' con relativi confini.
    """
    starts, ends = [], []
    R_w_arr = np.full(len(A_DENSE), np.nan)
    R_a_arr = np.full(len(A_DENSE), np.nan)

    for k, a in enumerate(A_DENSE):
        window, R_w, R_a = _window_and_thresholds_for_source_spin(
            source, a, a_grid_vals, tau_wave_grid, tau_align_grid)
        R_w_arr[k] = R_w
        R_a_arr[k] = R_a
        if window is not None:
            starts.append(window[0])
            ends.append(window[1])

    envelope = (min(starts), max(ends)) if starts else None
    runs, boundaries = _regime_runs_and_boundaries(A_DENSE, R_w_arr, R_a_arr)
    return envelope, runs, boundaries


def _window_for_source_spin(source, a, a_grid_vals, tau_wave_grid, tau_align_grid):
    """Finestra R_out per una sorgente a uno spin a fissato (M_ref della sorgente)."""
    window, _, _ = _window_and_thresholds_for_source_spin(
        source, a, a_grid_vals, tau_wave_grid, tau_align_grid)
    return window


def _compact_annotation_inline(runs, boundaries):
    """Genera un'etichetta compatta su singola riga per l'inserimento inter-barra."""
    if len(runs) > MAX_STABLE_RUNS:
        return r"$\ast$ regime instabile / quasi-tangente"

    short = {'wave': r'$t_{\rm wave}$', 'align': r'$t_{\rm align}$', 'open': 'open'}
    seq = " → ".join(short[r[0]] for r in runs)
    
    if boundaries:
        at = ", ".join(f"{b:.2f}" for b in boundaries)
        return f"{seq}  (limite a $a \\approx {at}$)"
    return seq


def plot_forest(indices=None, outdir='.', fname='pif_forest_plot.pdf'):
    sources = select_sources(indices) if indices is not None else CATALOG
    os.makedirs(outdir, exist_ok=True)

    a_grid_vals = np.union1d(A_DENSE, [A_FIXED])
    tau_wave_grid, tau_align_grid = _tau_hat_grid(a_grid_vals, R_OUT_GRID)

    rows = []
    for source in sources:
        w_main = _window_for_source_spin(source, A_FIXED, a_grid_vals, tau_wave_grid, tau_align_grid)
        envelope, runs, boundaries = _envelope_and_regime_for_source(
            source, a_grid_vals, tau_wave_grid, tau_align_grid)
        rows.append((source, w_main, envelope, runs, boundaries))

    n = len(rows)
    
    # Aumentiamo il fattore d'altezza (0.65 invece di 0.42) per dare respiro al testo sopra le barre
    fig, ax = plt.subplots(figsize=(8.5, 0.65 * n + 1.2))
    fix_spines(ax)

    for i, (source, w_main, envelope, runs, boundaries) in enumerate(rows):
        y = n - 1 - i  # Posizione della barra

        # 1. Barra grigia dell'inviluppo
        if envelope is not None:
            ax.plot([envelope[0], envelope[1]], [y, y], color='0.85', lw=7,
                    solid_capstyle='round', zorder=1)

        # 2. Barra blu dello spin di riferimento
        if w_main is not None:
            r_start, r_end, open_ended = w_main
            ax.plot([r_start, r_end], [y, y], color='#1f77b4', lw=3.2,
                    solid_capstyle='butt', zorder=2)
            ax.plot(r_start, y, marker='|', color='#1f77b4', markersize=8, markeredgewidth=1.5, zorder=3)
            if open_ended:
                ax.annotate('', xy=(R_OUT_MAX, y), xytext=(r_end * 0.85, y),
                            arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.3', 
                                            color='#1f77b4', lw=1.5),
                            zorder=3)
            else:
                ax.plot(r_end, y, marker='|', color='#1f77b4', markersize=8, markeredgewidth=1.5, zorder=3)
        else:
            ax.plot(R_OUT_GRID[0] * 1.2, y, marker='x', color='firebrick',
                    markersize=7, markeredgewidth=1.8, zorder=3)

        # 3. Annotazione POSIZIONATA SOPRA LA BARRA (y + 0.28)
        label_text = _compact_annotation_inline(runs, boundaries)
        # Scegliamo un punto x iniziale sicuro per il testo (es. subito dopo l'inizio della barra o a x=1.3)
        x_text = max(envelope[0] if envelope else 1.2, 1.25)
        
        ax.text(x_text, y - 0.4, label_text, va='bottom', ha='left',
                fontsize=8, color='0.3', zorder=4)

    # Assi e Griglia
    ax.set_yticks(range(n))
    ax.set_yticklabels([s['name'] for s, _, _, _, _ in reversed(rows)], fontsize=8.5, fontweight='medium')
    ax.set_xscale('log')
    ax.set_xlim(1, R_OUT_MAX)
    
    # Limiti verticali aggiustati per dare respiro al testo sopra la prima riga
    ax.set_ylim(-0.6, n - 0.7)
    ax.set_xlabel(r"$R_{\rm out} \quad [R_g]$", fontsize=9.5)

    # Titolo
    ax.set_title(r"Finestra di precessione rigida coerente ($t_{\rm wave} \leq P_{\rm osc} \leq t_{\rm align}$)",
                 fontsize=10, pad=12, weight='bold')

    # Legenda sotto il grafico
    legend_elements = [
        Line2D([0], [0], color='#1f77b4', lw=3.2, label=fr'Spin rif. ($a={A_FIXED:.2f}$)'),
        Line2D([0], [0], color='0.85', lw=7, label=fr'Inviluppo spin ($a \in [\pm {A_THORNE:.3f}]$)'),
        Line2D([0], [0], marker='x', color='firebrick', lw=0, markersize=7, markeredgewidth=1.8, label='Nessuna finestra'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=8)

    plt.tight_layout()
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return fpath

if __name__ == "__main__":
    out = plot_forest(outdir="output_catalog")
    print(f"Salvato: {out}")
