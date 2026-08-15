"""
modes_scan.py

Scan (a, M) per il p-mode fondamentale (m=0, n=0) a nu0 fissata:
per ciascun punto della griglia calcola se esiste una finestra di
trapping [r_isco, r1] (condizione kappa(r)=omega, Kato Fig. 6), t_wave
(Eq. 6 Motta) sull'intera finestra, e la FRAZIONE della finestra in cui
t_visc_Kato(r) > P_osc (cioe' dove l'ipotesi quasi-non-viscosa di Kato,
Sez. 6.1, e' rispettata).

Perche' una frazione e non un valore a un singolo raggio
-----------------------------------------------------------
G(r) e' monotona decrescente entro la finestra (verificato numericamente
su piu' punti della griglia: G(r_isco) e' il valore massimo, G->0 per
r->r1 per costruzione, dato che r1 e' definito da kappa(r1)=omega e
G ~ -(omega^2-kappa(r)^2)). Valutare G a un singolo raggio scelto
arbitrariamente (es. il centro) non e' giustificato: la grandezza
rilevante e' DOVE, dentro la finestra, la condizione t_visc>P_osc
(cioe' G(r)<nu0) e' soddisfatta. Data la monotonia, esiste al piu' un
unico raggio r_star con G(r_star)=nu0, che separa la finestra in una
parte "invalida" [r_isco, r_star] (crescita piu' rapida di un periodo
di oscillazione, ipotesi quasi-non-viscosa violata) e una parte
"valida" [r_star, r1]. Si riporta la frazione (in raggio) di finestra
valida: se G(r_isco)<nu0 allora TUTTA la finestra e' valida (frazione
=1); se G(r_isco)>=nu0 esiste un r_star con frazione =
(r1-r_star)/(r1-r_isco).

Design computazionale
----------------------
Sia il root-finding di r1 (kappa(r)=omega) sia quello di r_star
(G(r)=nu0) richiedono una ricerca di zero non vettorizzabile in forma
chiusa: sono le uniche parti del calcolo fatte con un doppio loop
Python su (a, M), risolte con bisezione (scipy.optimize.brentq) con
un bracket che si RESTRINGE proporzionalmente alla larghezza della
finestra (r1-r_isco), non a un offset assoluto fisso -- stessa
correzione gia' applicata alla ricerca di r1 stesso, necessaria perche'
il crossing puo' trovarsi arbitrariamente vicino a uno dei due bordi
della finestra (verificato numericamente: per alcuni (a,M) il crossing
e' oltre la risoluzione di un campionamento a passo fisso). Tutto il
resto (t_wave) e' vettorizzato dopo che le griglie di r1, r_star sono
note.
"""

import numpy as np
from scipy.optimize import brentq
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from setup import r_isco, nu_r
from disk_profiles import t_wave_closed
from growth_rate import growth_rate_p_mode


def _r1_and_kappamax_scalar(a, M, nu0, r_scan_max, n_scan):
    """
    Root-finding scalare: r1 (bordo trapping) e kappa_max.

    Scan log-spaziato nella DISTANZA delta=r-r_in (non in r stesso), da
    un offset minimo molto piccolo fino a r_scan_max: necessario perche'
    kappa puo' crescere ripidissimamente vicino all'ISCO (M piccola),
    rendendo la vera r1 piu' vicina all'ISCO di quanto un offset fisso
    (es. r_in*1.0001) riesca a risolvere (verificato numericamente).
    """
    r_in = r_isco(a)
    delta_min = 1e-9 * max(r_in, 1.0)
    delta_max = r_scan_max - r_in
    delta_scan = np.geomspace(delta_min, delta_max, n_scan)
    r_scan = r_in + delta_scan

    kappa_scan = nu_r(r_scan, a, M)
    kappa_max = kappa_scan.max()

    if nu0 >= kappa_max:
        return np.nan, kappa_max  # nessun trapping

    diff = nu0 - kappa_scan
    sign_changes = np.where(np.diff(np.sign(diff)) < 0)[0]
    if len(sign_changes) == 0:
        return np.nan, kappa_max  # vera r1 oltre la risoluzione di delta_min

    i = sign_changes[0]
    denom = diff[i + 1] - diff[i]
    if denom == 0:
        r1 = r_scan[i]
    else:
        r1 = r_scan[i] - diff[i] * (r_scan[i + 1] - r_scan[i]) / denom
    return r1, kappa_max


def _safe_eval_G(r_target, r_in, r1, a, M, nu0, alpha, direction, max_tries=12):
    """
    Valuta G(r_target). Se NaN (r_target troppo vicino a un bordo,
    oltre la precisione residua con cui r1 stesso e' noto -- verificato
    numericamente: l'interpolazione lineare usata per trovare r1 lascia
    un errore residuo che puo' far si' che kappa(r) superi leggermente
    omega proprio vicinissimo a r1), ci si allontana dal bordo di un
    fattore 10 alla volta finche' non si ottiene un valore finito.
    direction=+1 sposta r_target verso l'interno (usato per il bordo r1);
    direction=-1 non e' necessario qui (bordo r_in e' esatto, kappa=0).
    """
    width = r1 - r_in
    r = r_target
    for _ in range(max_tries):
        G = growth_rate_p_mode(np.array([r]), a, M, nu0, alpha)[0]
        if np.isfinite(G):
            return r, G
        r = r1 - direction * (r1 - r) * 10.0 if direction > 0 else r
    return r, np.nan


def _valid_fraction_scalar(a, M, r_in, r1, nu0, alpha):
    """
    Frazione (in raggio) della finestra [r_in, r1] dove G(r) < nu0
    (cioe' t_visc(r) > P_osc = 1/nu0), sfruttando la monotonia
    decrescente di G(r) dentro la finestra.

    Bracket per brentq scalato sulla larghezza della finestra stessa
    (non un offset assoluto), per non perdere crossing molto vicini a
    r_in o a r1. Il bordo vicino a r1 usa un retry adattivo (vedi
    _safe_eval_G): la precisione finita con cui r1 e' noto puo' rendere
    NaN una valutazione troppo vicina ad esso.
    """
    width = r1 - r_in
    if width <= 0 or not np.isfinite(width):
        return np.nan, np.nan

    r_lo = r_in + 1e-9 * width
    G_lo = growth_rate_p_mode(np.array([r_lo]), a, M, nu0, alpha)[0]

    r_hi_target = r1 - 1e-9 * width
    r_hi, G_hi = _safe_eval_G(r_hi_target, r_in, r1, a, M, nu0, alpha, direction=1)

    if not np.isfinite(G_lo) or not np.isfinite(G_hi):
        return np.nan, np.nan  # non risolvibile in modo affidabile

    if G_lo < nu0:
        return r_in, 1.0
    if G_hi > nu0:
        return np.nan, np.nan

    def f(r):
        return growth_rate_p_mode(np.array([r]), a, M, nu0, alpha)[0] - nu0

    r_star = brentq(f, r_lo, r_hi, xtol=1e-10 * width, rtol=1e-10)
    valid_fraction = (r1 - r_star) / width
    return r_star, valid_fraction


def scan_p_mode_grid(a_vals, M_vals, nu0, alpha, r_scan_max=5000.0, n_scan=3000):
    """
    Griglia (a, M) [meshgrid indexing='ij'] per il p-mode fondamentale a
    nu0 fissata.

    Ritorna un dict con, ciascuno shape (len(a_vals), len(M_vals)):
      r_isco_grid, r1_grid, kappa_max_grid : geometria/trapping (NaN se
          non esiste trapping)
      t_wave_grid : tempo di attraversamento sonoro sull'intera finestra
      r_star_grid, valid_fraction_grid : raggio di crossing G(r)=nu0 e
          frazione di finestra con t_visc>P_osc (NaN dove non risolvibile,
          si veda _valid_fraction_scalar)
      trapped_mask : bool, True dove esiste una finestra di trapping
    """
    NA, NM = len(a_vals), len(M_vals)
    r1_grid = np.full((NA, NM), np.nan)
    kappa_max_grid = np.full((NA, NM), np.nan)
    r_star_grid = np.full((NA, NM), np.nan)
    valid_fraction_grid = np.full((NA, NM), np.nan)

    # --- unica parte non vettorizzabile: root-finding di r1 e r_star ---
    for i, a in enumerate(a_vals):
        for j, M in enumerate(M_vals):
            r1, kmax = _r1_and_kappamax_scalar(a, M, nu0, r_scan_max, n_scan)
            r1_grid[i, j] = r1
            kappa_max_grid[i, j] = kmax
            if np.isfinite(r1):
                r_in = r_isco(a)
                r_star, vf = _valid_fraction_scalar(a, M, r_in, r1, nu0, alpha)
                r_star_grid[i, j] = r_star
                valid_fraction_grid[i, j] = vf

    trapped_mask = np.isfinite(r1_grid)

    # --- resto vettorizzato sull'intera griglia ---
    A_grid, M_grid = np.meshgrid(a_vals, M_vals, indexing='ij')
    r_isco_grid = r_isco(A_grid)

    t_wave_grid = t_wave_closed(r_isco_grid, r1_grid, A_grid, M_grid)

    return dict(
        r_isco_grid=r_isco_grid, r1_grid=r1_grid, kappa_max_grid=kappa_max_grid,
        t_wave_grid=t_wave_grid, r_star_grid=r_star_grid,
        valid_fraction_grid=valid_fraction_grid, trapped_mask=trapped_mask,
    )


"""
modes_scan_catalog.py

Generalizzazione di modes_scan_plot.py all'intero catalogo (catalog.py):
per ciascuna sorgente selezionata, scan (a, M) del p-mode fondamentale a
nu0 = nu0 della sorgente, con lo stesso design a tre pannelli gia'
validato per J1257 (larghezza relativa della finestra di trapping,
t_wave/P_osc, frazione di finestra "valida" per l'ipotesi
quasi-non-viscosa di Kato -- si veda la motivazione completa in
modes_scan.py sul perche' una frazione e non un valore a singolo raggio).

Se la sorgente ha una stima di massa indipendente (mass_range non
None, secondo la convenzione di catalog.py), viene sovrapposta come
riferimento orizzontale sui tre pannelli: una linea se stima puntuale
(lo==hi), una banda ombreggiata se e' un range riportato in tesi.
Questo permette di leggere a colpo d'occhio se la regione (a,M)
fisicamente rilevante per la sorgente (secondo stime indipendenti)
cade in una zona favorevole o sfavorevole dello scan.
"""

import re
import matplotlib.pyplot as plt
from setup import set_style, fix_spines, M_AGN_MIN, M_AGN_MAX
from catalog import CATALOG, select_sources

set_style()

# --- parametri fissi (stessi di modes_scan_plot.py per J1257) ---
ALPHA = 0.1
N_A, N_M = 80, 80
A_VALS = np.linspace(-0.998, 0.998, N_A)
M_VALS = np.logspace(5, 10, N_M)


def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def _overlay_mass_range(ax, mass_range):
    """
    Sovrappone la stima di massa indipendente della sorgente (se
    disponibile), secondo la stessa convenzione di catalog.py:
      None       -> nessuna stima, nessun overlay
      (lo,lo)    -> stima puntuale, linea nera continua
      (lo,hi)    -> banda riportata in tesi, area ombreggiata + bordi
                    tratteggiati
    """
    if mass_range is None:
        return
    lo, hi = mass_range
    if lo == hi:
        ax.axhline(lo, color='black', ls='-', lw=1.1, alpha=0.8, zorder=5)
    else:
        ax.axhspan(lo, hi, color='black', alpha=0.12, zorder=0)
        ax.axhline(lo, color='black', ls='--', lw=0.7, alpha=0.6, zorder=5)
        ax.axhline(hi, color='black', ls='--', lw=0.7, alpha=0.6, zorder=5)


def plot_source_modes_scan(source, outdir='.'):
    """
    Genera e salva la figura a tre pannelli per una singola sorgente.
    Ritorna (path_file, summary_dict) con statistiche riassuntive utili
    per un confronto rapido tra sorgenti (vedi plot_catalog).
    """
    nu0 = source['nu0']
    name = source['name']
    mass_range = source['mass_range']

    res = scan_p_mode_grid(A_VALS, M_VALS, nu0, ALPHA, n_scan=3000)

    A_grid, M_grid = np.meshgrid(A_VALS, M_VALS, indexing='ij')
    Posc = 1.0 / nu0
    trapped = res['trapped_mask']

    ratio_wave = np.where(trapped, res['t_wave_grid'] / Posc, np.nan)
    valid_fraction = res['valid_fraction_grid']
    width_rel = np.where(trapped,
                          (res['r1_grid'] - res['r_isco_grid']) / res['r_isco_grid'],
                          np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax in axes:
        fix_spines(ax)
        ax.set_yscale('log')
        ax.set_xlabel(r"$a$")
        ax.set_xlim(-1, 1)
        ax.set_ylim(M_VALS[0], M_VALS[-1])
        _overlay_mass_range(ax, mass_range)

    ax = axes[0]
    pcm = ax.pcolormesh(A_grid, M_grid, np.log10(width_rel), shading='auto', cmap='viridis')
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$\log_{10}[(r_1-r_{\rm isco})/r_{\rm isco}]$")
    ax.set_title("Larghezza relativa finestra")
    ax.set_ylabel(r"$M\ [M_\odot]$")

    ax = axes[1]
    pcm = ax.pcolormesh(A_grid, M_grid, np.log10(ratio_wave), shading='auto', cmap='RdBu_r',
                         vmin=-3, vmax=3)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"$\log_{10}(t_{\rm wave}/P_{\rm osc})$")
    ax.set_title(r"$t_{\rm wave}/P_{\rm osc}$")

    ax = axes[2]
    pcm = ax.pcolormesh(A_grid, M_grid, valid_fraction, shading='auto', cmap='viridis',
                         vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04).set_label(
        r"frazione finestra con $t_{\rm visc}^{\rm Kato}>P_{\rm osc}$")
    ax.set_title("Frazione finestra 'valida'")

    fig.suptitle(fr"{name}  —  p-mode fondamentale, $\nu_0={nu0:.2e}$ Hz, $\alpha={ALPHA}$",
                 fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(outdir, f"modes_scan_{safe_filename(name)}.pdf")
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)

    n_trap = trapped.sum()
    summary = dict(
        name=name, nu0=nu0, mass_range=mass_range,
        frac_trapped=trapped.mean(),
        frac_wave_ok=(float(np.nansum(ratio_wave < 1.0)) / n_trap if n_trap > 0 else np.nan),
        mean_valid_fraction=(float(np.nanmean(valid_fraction)) if n_trap > 0 else np.nan),
        median_valid_fraction=(float(np.nanmedian(valid_fraction)) if n_trap > 0 else np.nan),
        n_trap=int(n_trap),
    )
    return fname, summary


def plot_catalog(indices=None, outdir='.'):
    """
    Genera le figure per un sottoinsieme del catalogo (o per tutto il
    catalogo se indices=None), NELL'ORDINE dato da indices se fornito,
    altrimenti nell'ordine canonico del catalogo (stessa convenzione di
    pif_timescales.plot_catalog).
    """
    sources = select_sources(indices) if indices is not None else CATALOG
    os.makedirs(outdir, exist_ok=True)
    summaries = []
    for source in sources:
        fname, summary = plot_source_modes_scan(source, outdir=outdir)
        print(f"Salvato: {fname}")
        if summary['n_trap'] > 0:
            print(f"  trapping: {100*summary['frac_trapped']:.1f}% griglia | "
                  f"t_wave<P_osc: {100*summary['frac_wave_ok']:.1f}% dei trapped | "
                  f"frazione valida media: {summary['mean_valid_fraction']:.4f} | "
                  f"mediana: {summary['median_valid_fraction']:.4f}")
        else:
            print("  NESSUN trapping in tutta la griglia (a,M) esplorata")
        summaries.append(summary)
    return summaries


if __name__ == "__main__":
    plot_catalog(indices=[0,9,10,11],outdir="output_modes")