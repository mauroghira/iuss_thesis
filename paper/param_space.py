# param_space.py
#
# Infrastruttura generica e scalabile per:
#   1) invertire M dato un target di frequenza, sfruttando (quando disponibile)
#      la scaling esatta nu(params) = g(params_geometrici) / M;
#   2) applicare vincoli fisici (ISCO, limite di Thorne, range di massa AGN);
#   3) estrarre le combinazioni estremali (inviluppo) dello spazio dei
#      parametri ammesso, per una singola sorgente o per uno span di
#      frequenze continuo.
#
# Pensato per essere riusato da qualunque modello le cui frequenze siano
# costruite algebricamente a partire dalle frequenze di Kerr (RPM, PIF,
# diskoseismologia in approssimazione WKB, ...). Per modelli che NON
# godono della scaling 1/M esatta (es. AEI, dove il campo magnetico non
# scala come le frequenze geodetiche) e' fornito un fallback numerico
# generico basato su bisezione vettorizzata.

import warnings
import numpy as np
from setup import r_isco, A_THORNE


##########################################################
# --------------------------------------------------------
##########################################################
# GRIGLIA DI FREQUENZE DI SCANSIONE
def freq_scan_grid(nu_min, nu_max, n, log=True):
    """
    Griglia di n frequenze di scansione tra nu_min e nu_max [Hz].

    log=True (default): spaziatura logaritmica, appropriata dato che lo
    span copre ~4 ordini di grandezza (1e-7 - 1e-3 Hz); una spaziatura
    lineare sovracampionerebbe le frequenze alte e sottocampionerebbe
    quelle basse, dove si concentrano le sorgenti LF-QPO piu' informative
    per i vincoli sullo spin.
    """
    if log:
        return np.logspace(np.log10(nu_min), np.log10(nu_max), n)
    return np.linspace(nu_min, nu_max, n)


##########################################################
# --------------------------------------------------------
##########################################################
# PERCORSO ANALITICO (modelli con scaling esatta nu = g(...)/M)
def invert_mass_analytic(g_grid, nu_targets):
    """
    Inversione esatta di M dato un fattore geometrico g_grid (output di
    una funzione tipo g_LT(r,a), g_per(r,a), ... valutata su una griglia
    N-dimensionale qualsiasi di parametri geometrici) e un array di
    frequenze target nu_targets, shape (n_freq,).

    Restituisce M_grid di shape g_grid.shape + (n_freq,), con
        M_grid[..., k] = g_grid[...] / nu_targets[k]

    Vettorizzato: nessun ciclo Python, nessuna soglia di tolleranza.
    Costo: O(N_geom * n_freq) in tempo e memoria. Se serve solo
    l'inviluppo (max/min di M sullo span), usare mass_envelope_analytic,
    che costa O(N_geom) e non richiede materializzare l'asse delle
    frequenze.
    """
    g_grid = np.asarray(g_grid)
    nu_targets = np.asarray(nu_targets)
    return g_grid[..., None] / nu_targets


def mass_envelope_analytic(g_grid, nu_min, nu_max):
    """
    Inviluppo esatto di M compatibile con un CONTINUO di frequenze in
    [nu_min, nu_max], dato un fattore geometrico g_grid = g_X(params).

    Giustificazione: per g_grid fissato, M(nu) = g/nu e' strettamente
    monotona decrescente in nu (derivata -g/nu^2 < 0 per g>0). Quindi il
    massimo di M su [nu_min,nu_max] si ottiene esattamente in nu_min e il
    minimo esattamente in nu_max — non serve campionare nu_targets punto
    per punto, ne' assumere alcuna proprieta' di g rispetto ai parametri
    geometrici (r, a, ...): la monotonia e' in nu, non in essi.

    Restituisce (M_min, M_max), stessa shape di g_grid.
    """
    g_grid = np.asarray(g_grid)
    M_max = g_grid / nu_min
    M_min = g_grid / nu_max
    return M_min, M_max


##########################################################
# --------------------------------------------------------
##########################################################
# PERCORSO NUMERICO GENERICO 
def invert_mass_numeric(freq_func, geom_kwargs, nu_target,
                         M_bounds=(1e5, 1e10), n_iter=60):
    """
    Trova M tale che freq_func(M=M, **geom_kwargs) == nu_target, per
    ciascun punto della griglia geometrica, tramite bisezione vettoriale.

    freq_func    : callable, firma freq_func(M=..., **geom_kwargs) -> ndarray
                   (stessa shape dei valori in geom_kwargs)
    geom_kwargs  : dict di ndarray (es. {"r": R, "a": A}), parametri
                   geometrici gia' su griglia, con M NON incluso
    nu_target    : float, frequenza target singola
    M_bounds     : (M_lo, M_hi) intervallo di ricerca in masse solari
    n_iter       : iterazioni di bisezione; 60 iterazioni dimezzano
                   l'intervallo di un fattore 2^-60, largamente
                   sufficiente per la precisione doppia

    Assume freq_func monotona DECRESCENTE in M (vero per tutte le
    frequenze considerate in questo lavoro, poiche' nu ~ 1/Rg ~ 1/M).
    Se il modello futuro non rispettasse questa monotonia, va sostituita
    con una scansione esplicita in M seguita da ricerca di zero locale.
    """
    shape = next(iter(geom_kwargs.values())).shape
    M_lo = np.full(shape, M_bounds[0], dtype=float)
    M_hi = np.full(shape, M_bounds[1], dtype=float)

    # verifica di consistenza dei bound: freq(M_lo) deve essere >= target
    # e freq(M_hi) <= target (funzione monotona decrescente in M)
    f_lo = freq_func(M=M_lo, **geom_kwargs)
    f_hi = freq_func(M=M_hi, **geom_kwargs)
    valid = (f_lo >= nu_target) & (f_hi <= nu_target)

    for _ in range(n_iter):
        M_mid = 0.5 * (M_lo + M_hi)
        f_mid = freq_func(M=M_mid, **geom_kwargs)
        take_upper = f_mid > nu_target   # freq ancora troppo alta -> serve M piu' grande
        M_lo = np.where(take_upper, M_mid, M_lo)
        M_hi = np.where(take_upper, M_hi, M_mid)

    M_sol = 0.5 * (M_lo + M_hi)
    return np.where(valid, M_sol, np.nan)


##########################################################
# --------------------------------------------------------
##########################################################
# VINCOLI FISICI
def physical_mask(a_grid, r_grid, M_grid,
                   a_limit=A_THORNE,
                   M_range=None):
    """
    Maschera booleana composita con i vincoli fisici standard:
      - r >= r_isco(a)               (orbita al di fuori dell'ISCO)
      - |a| <= a_limit                (limite di Thorne, default 0.998)
      - M in M_range = (M_min, M_max) se fornito (range AGN o stima
        indipendente della singola sorgente)

    a_grid, r_grid, M_grid devono essere broadcastabili tra loro (tipico:
    output di create_param_grid con mesh=True, oppure il risultato di
    invert_mass_analytic/mass_envelope_analytic).
    """
    isco = r_isco(a_grid)
    mask = (r_grid >= isco) & (np.abs(a_grid) <= a_limit)
    if M_range is not None:
        M_lo, M_hi = M_range
        mask &= (M_grid >= M_lo) & (M_grid <= M_hi)
    return mask


##########################################################
# --------------------------------------------------------
##########################################################
# ESTRAZIONE DELLE COMBINAZIONI ESTREMALI
def extremal_over_axis(M_grid, mask, axis):
    """
    Estrae l'inviluppo (max e min) di M_grid lungo l'asse `axis` (tipico:
    l'asse del raggio r, o piu' in generale un parametro "libero" del
    modello, es. r_out per PIF), tenendo conto della maschera fisica.

    Non assume alcuna monotonia di M_grid lungo l'asse ridotto: usa
    nanmax/nanmin numerici, quindi e' corretto anche per canali come
    nu_LT dove g_LT(r,a) puo' avere un massimo locale non all'ISCO
    (si veda discussione su nu_LT: la non-monotonia va sempre verificata
    numericamente, mai assunta a priori).

    Restituisce (M_min_env, M_max_env), shape = M_grid.shape senza
    l'asse `axis`. Punti dove nessun valore e' ammesso -> NaN.
    """
    M_masked = np.where(mask, M_grid, np.nan)
    with warnings.catch_warnings():
        # "All-NaN slice": atteso quando, per un dato valore dell'asse
        # ridotto (es. uno spin a), nessun punto soddisfa i vincoli
        # fisici; il risultato NaN e' corretto e va gestito a valle
        # (es. escluso dal plot), non e' un errore di calcolo.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        M_max_env = np.nanmax(M_masked, axis=axis)
        M_min_env = np.nanmin(M_masked, axis=axis)
    return M_min_env, M_max_env