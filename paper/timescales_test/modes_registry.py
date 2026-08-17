"""
mode_registry.py

Registro generico dei modi diskoseismici (p, g, c) per la generazione
automatizzata, per ciascuna sorgente del catalogo, dei due grafici
t_wave / (t_visc o t_damp) gia' introdotti in modes_timescales.py per
il solo p-mode fondamentale -- qui estesi a QUALUNQUE modo (tipo +
numeri quantici m,n) tramite un'unica interfaccia comune.

Interfaccia di una "famiglia" di modo (p, g, c)
--------------------------------------------------
Ogni famiglia espone due funzioni con firma fissa:

  find_window(a, M, nu0, m, n) -> dict(r_in, r_out, valid, extra)
      Trova la finestra radiale [r_in, r_out] su cui viene integrato
      t_wave (disk_profiles.t_wave_closed). r_out E' UN PARAMETRO
      DICHIARATO del modo, non un dettaglio nascosto: per il p-mode e'
      r1 (kappa(r1)=omega, Kato Sez. 5.1); per il c-mode e' scelto
      esplicitamente da CMODE_TWAVE_OUTER_BOUNDARY sotto (r_IVR o
      r_ILR -- vedi discussione li', NON ancora confermata dall'utente).
      'extra' porta eventuale struttura aggiuntiva utile a valle (es.
      r_IVR, r_ILR per il c-mode, per diagnostica). 'valid'=False se la
      finestra non esiste per questi (a, M, nu0, m, n).

  timescale(r_grid, a, M, nu0, m, n, window, alpha) -> array, stessa
      shape di r_grid.
      Tasso di smorzamento/crescita convertito in tempo. Per modi con
      tasso r-dipendente (p-mode, G(r) di Kato) e' una VERA curva; per
      modi con tasso "globale" (c-mode: omega_i e' un autovalore
      dell'intero modo, Eq. 39 Tsang & Lai, non funzione di r) e' un
      array COSTANTE (stesso valore ripetuto su r_grid). La funzione di
      plotting generica non deve sapere quale dei due casi si applica:
      vede sempre un array su r_grid.

Una famiglia e' registrata in MODE_FAMILIES sotto una chiave kind in
{'p','g','c'}; (m,n) sono parametri della RICHIESTA (ModeRequest), non
della registrazione: find_window/timescale del c-mode sono gia'
generiche in m,n (growth_rate.py le accetta come argomenti); quelle del
p-mode ignorano m,n perche' in growth_rate.py e' implementato solo il
fondamentale (m,n)=(0,0).

NOTA SUL COSTO COMPUTAZIONALE (dichiarata, non ottimizzata qui)
--------------------------------------------------------------------
Per il c-mode, _cmode_window calcola r_IVR/r_ILR per il pannello
finestra, e _cmode_timescale chiama t_damp_cmode (quindi
omega_i_cmode), che RICALCOLA internamente r_IVR/r_ILR da zero: stessa
radice trovata due volte per ogni punto (a o M) scansionato. E'
ridondante ma corretto (deterministico, stesso risultato); ottimizzarlo
richiederebbe esporre in growth_rate.py una variante di omega_i_cmode
che accetti r_IVR/r_ILR gia' noti, il che e' fuori dallo scope di
questa generalizzazione architetturale.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from setup import r_isco, R_MAX, r_ilr, r_olr, r_vertical_resonance
from growth_rate import (growth_rate_p_mode, find_p_mode_outer_boundary,
                          find_r_IVR, find_r_ILR, t_damp_cmode)

# --- dominio di ricerca condiviso da TUTTI i modi ----------------------
# R_MAX=500 e' gia' il troncamento arbitrario del disco dichiarato e
# usato altrove nel progetto (es. pif_forest.py, R_OUT_MAX): riusato qui
# per coerenza, al posto del default 5000.0 interno a growth_rate.py.
R_SCAN_MAX = R_MAX
N_SCAN = 20000

# Confine esterno usato per t_wave nel c-mode: 'IVR' (sola Regione I,
# la zona di trapping/propagazione libera, analoga concettualmente alla
# finestra [r_isco, r1] del p-mode) oppure 'ILR' (intera estensione del
# modo, trapping+evanescente). DEFAULT='IVR' -- scelta dichiarata qui
# esplicitamente, NON ancora confermata dall'utente: per cambiarla
# basta questa riga, nessun'altra parte del codice va toccata.
CMODE_TWAVE_OUTER_BOUNDARY = 'IVR'

ALPHA_PMODE_DEFAULT = 0.1  # viscosita' turbolenta di Kato (Fig. 12), DISTINTA
                            # da ALPHA=0.01 di pif_timescales.py (Foucart-Lai)


# ======================================================================
# P-MODE (fondamentale, m=n=0 -- unico caso implementato in growth_rate.py)
# ======================================================================
def _pmode_window(a, M, nu0, m, n):
    r_in = float(r_isco(a))
    r1 = find_p_mode_outer_boundary(a, M, nu0, m, n, r_scan_max=R_SCAN_MAX, n_scan=N_SCAN)
    valid = bool(np.isfinite(r1))
    return dict(r_in=r_in, r_out=r1, valid=valid, extra={})


def _pmode_timescale(r_grid, a, M, nu0, m, n, window, alpha):
    G = growth_rate_p_mode(r_grid, a, M, nu0, alpha)
    return 1.0 / np.abs(G)
    

# ======================================================================
# C-MODE (Tsang & Lai 2008, m>=1, n>=1 generico)
# ======================================================================
def _cmode_window(a, M, nu0, m, n):
    r_in = float(r_isco(a))
    r_IVR = r_vertical_resonance(a, m, n, nu_obs=nu0, M=M)
    r_ILR = r_ilr(a, m, nu_obs=nu0, M=M)
    valid = bool(np.isfinite(r_IVR) and np.isfinite(r_ILR))
    r_out = r_IVR
    return dict(r_in=r_in, r_out=r_out, valid=valid,
                extra=dict(r_IVR=r_IVR, r_ILR=r_ILR))


def _cmode_timescale(r_grid, a, M, nu0, m, n, window, alpha):
    """alpha ignorato: il damping di Tsang & Lai non dipende da una
    viscosita' turbolenta locale (e' assorbimento alla corotazione via
    tunneling), resta in firma solo per uniformita' con l'interfaccia
    generica. omega_i (quindi t_damp) e' un autovalore dell'INTERO modo,
    non funzione di r: si restituisce un array costante su r_grid."""
    t_damp = t_damp_cmode(a, M, m, n, nu0, r_scan_max=R_SCAN_MAX, n_scan=N_SCAN)
    r_grid = np.asarray(r_grid, dtype=float)
    return np.full_like(r_grid, t_damp)


# ======================================================================
# G-MODE -- NON IMPLEMENTATO
# ======================================================================
def _gmode_window(a, M, nu0, m, n):
    r_ILR = r_ilr(a, m, nu_obs=nu0, M=M)
    r_OLR = r_olr(a, m, nu_obs=nu0, M=M)

    valid = (
        np.isfinite(r_ILR)
        and np.isfinite(r_OLR)
        and r_ILR < r_OLR
    )

    return dict(r_in=r_ILR, r_out=r_OLR, valid=valid, extra={})


MODE_FAMILIES = {
    'p': dict(
        find_window=_pmode_window,
        timescale=_pmode_timescale,
        needs_alpha=True,
        alpha_default=ALPHA_PMODE_DEFAULT,
        timescale_label=r"$t_{\rm visc}^{\rm Kato}(r)$",
        timescale_is_curve=True,
        label_fmt=lambda m, n: r"$p$-mode fondamentale $(m,n)=(0,0)$",
    ),
    'c': dict(
        find_window=_cmode_window,
        timescale=_cmode_timescale,
        needs_alpha=False,
        alpha_default=None,
        timescale_label=r"$t_{\rm damp}$ (Tsang & Lai)",
        timescale_is_curve=False,
        label_fmt=lambda m, n: fr"$c$-mode $(m,n)=({m},{n})$",
    ),
    'g': dict(
        find_window=_gmode_window,
        timescale=None,
        needs_alpha=None,
        alpha_default=None,
        timescale_label=None,
        timescale_is_curve=None,
        label_fmt=lambda m, n: fr"$g$-mode $(m,n)=({m},{n})$ [NON IMPLEMENTATO]",
    ),
}


class ModeRequest:
    """
    Una singola richiesta di modo da plottare: (kind, m, n, alpha).
    alpha=None usa il default della famiglia (ALPHA_PMODE_DEFAULT per il
    p-mode; ignorato per il c-mode/g-mode).
    """

    def __init__(self, kind, m=0, n=0, alpha=None):
        if kind not in MODE_FAMILIES:
            raise ValueError(f"Tipo di modo sconosciuto: {kind!r} "
                              f"(attesi: {list(MODE_FAMILIES)})")
        self.kind = kind
        self.m = m
        self.n = n
        self.family = MODE_FAMILIES[kind]
        self.alpha = alpha if alpha is not None else self.family['alpha_default']
        self.label = self.family['label_fmt'](m, n)
        self.id = f"{kind}{m}{n}"

    def window(self, a, M, nu0):
        return self.family['find_window'](a, M, nu0, self.m, self.n)

    def timescale(self, r_grid, a, M, nu0, window):
        return self.family['timescale'](r_grid, a, M, nu0, self.m, self.n,
                                         window, self.alpha)

    def __repr__(self):
        return f"ModeRequest({self.id!r})"