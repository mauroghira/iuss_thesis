# file for basic functions and parameters for various models
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.constants import G, c, M_sun, sigma_T

import inspect

#let's set the parameters
M_BH = 10**6.3
NU0 = 3.3*10**(-5)

# Physical constants
C = c.cgs.value         # cm/s
MSUN = M_sun.cgs.value     # g
GG = G.cgs.value       # cm^3 g^-1 s^-2
AU = (1 * u.AU).to(u.m).value  # AU

SigTOM = sigma_T.cgs.value  # cm^2

Rg_SUN = GG * MSUN / C**2  # in cm
L_Edd_SUN = 1.26e38  # erg/s

#let's set the tolerance for the target frequency
TOL = 0.01 * NU0  # 1% tolerance
TARGET_MIN = 1e-5
TARGET_MAX = 1e-4

# --------------------------------------------------------
NU_OBS_MIN = 2.76e-7   # NGC 4945, Smith et al. 2020
NU_OBS_MAX = 2.5e-3    # Masterson
NU_SCAN_MIN = 1e-7
NU_SCAN_MAX = 1e-3

# Range di massa fisico per AGN limite
# di spin di Thorn  e (1974), oltre il quale l'accrescimento auto-limita
# lo spin del BH.
M_AGN_MIN = 1e5
M_AGN_MAX = 1e10
A_THORNE = 0.998
R_MAX = 500


##########################################################
# --------------------------------------------------------
##########################################################
# GENERAL FUNCTION TO CREATE GRIDS
def create_param_grid(param_dict, mesh=True, flatten=False):
    """
    Create an N-dimensional grid from a dictionary of parameters.
    
    param_dict : dict
        Keys = parameter names (str)
        Values = tuple/list defining the grid for each parameter.
                 Accepted formats:
                    - (min, max, n_points)
                    - array-like explicit list of values
                    
    mesh : bool
        If True, returns a meshgrid (N arrays).
        If False, only returns the 1D vectors for each parameter.

    flatten : bool
        If True, also returns a 2D array shape (N_points_total, N_params)
        useful for vectorized evaluation.

    Returns:
        param_vectors: dict of 1D arrays for each parameter
        mesh_arrays  : list of meshgrid arrays (if mesh=True)
        flat_matrix  : 2D array of flattened grid points (if flatten=True)
    """

    labels = list(param_dict.keys())
    values = []

    # Build 1D arrays for each parameter
    for key, val in param_dict.items():
        if len(val) == 3 and all(isinstance(x, (int, float)) for x in val):
            vmin, vmax, n = val
            if "r" in key or "B00" in key or "Sigma0" in key:
                # logarithmic spacing for radii
                values.append(np.logspace(np.log10(vmin), np.log10(vmax), n))
            else:
                values.append(np.linspace(vmin, vmax, n))
        else:
            # explicit array provided
            values.append(np.array(val))

    # Return only vectors
    param_vectors = {lab: vec for lab, vec in zip(labels, values)}

    if not mesh and not flatten:
        return param_vectors

    # Build meshgrid
    mesh_arrays = np.meshgrid(*values, indexing="ij")

    if not flatten:
        return param_vectors, mesh_arrays

    # Flatten meshgrid to shape (N_total, N_params)
    stacked = np.stack(mesh_arrays, axis=-1)   # shape (..., N_params)
    flat_matrix = stacked.reshape(-1, len(labels))

    return param_vectors, mesh_arrays, flat_matrix


##########################################################
# --------------------------------------------------------
##########################################################
# GENERAL FUNCTION TO FIND MATCHES
def find_matches(mesh_arrays, labels, param_vectors, frq_fun):
    """
    N-dimensional match finder with automatic rISCO constraints.
    
    mesh_arrays : list of ndarrays
        Meshgrid arrays from create_param_grid()
    labels : list of str
        Parameter names (same order as mesh_arrays)
    param_vectors : dict
        1D parameter vectors (output of create_param_grid)
    r_isco : callable
        Function r_isco(a)
    freq_func : callable
        freq = freq_func(param_dict)
    """

    # Parametri N-D (meshgrid)
    param_dict = {lab: arr for lab, arr in zip(labels, mesh_arrays)}

    # Frequenze sul reticolo
    freq_func = frq_wrap(frq_fun)
    freq = freq_func(param_dict)

    # Maschera del match in frequenza
    mask_freq = np.abs(freq - NU0) < TOL

    # ------- rISCO positive & negative -------
    a_vec = param_vectors["a"]          # 1D array
    isco = r_isco(a_vec)

    # broadcasting
    r_isco_nd = isco.reshape(-1, *[1]*(freq.ndim - 1))

    # maschere complete
    mask = np.ones_like(freq, bool)

    # applica il vincolo solo ai parametri che contengono "r"
    for lab, arr in param_dict.items():
        if "r" in lab:
            mask &= (arr >= r_isco_nd)

    # maschere finali
    mask_match = mask_freq & mask

    # ------- raccogli risultati -------
    rows = []

    idxs_pos = np.argwhere(mask_match)
    for idx in idxs_pos:
        row = {lab: arr[tuple(idx)] for lab, arr in param_dict.items()}
        row["freq"] = freq[tuple(idx)]
        rows.append(row)

    # ---- DataFrame finale ----
    df = pd.DataFrame(rows)
    return df


##########################################################
# --------------------------------------------------------
##########################################################
# GENERAL WRAPPER TO MAKE FREQUENCY FUNCTIONS
def frq_wrap(freq_callable):
    """
    Ritorna una funzione che accetta un dizionario di parametri
    e passa alla freq_callable solo quelli che essa richiede.
    """
    sig = inspect.signature(freq_callable)
    param_names = list(sig.parameters.keys())

    def wrapper(param_dict):
        # Estrai solo i parametri richiesti
        args = [param_dict[name] for name in param_names]
        return freq_callable(*args)

    return wrapper


##########################################################
# --------------------------------------------------------
##########################################################
# KERR METRIC BASIC FUNCTIONS
# Kerr frequencies in Hz
#
# SCALING ESATTA IN M
# --------------------
# R_g(M) = Rg_SUN * M  e' lineare in M, quindi
#     nu_phi(r,a,M) = C / (2 pi Rg_SUN M (r^1.5+a)) = g_phi(r,a) / M
# dove g_phi(r,a) e' un fattore puramente geometrico (indipendente da M).
# nu_theta e nu_r sono nu_phi moltiplicata per fattori anch'essi
# indipendenti da M, quindi ereditano la stessa scaling 1/M:
#     nu_theta(r,a,M) = g_theta(r,a)/M ,  nu_r(r,a,M) = g_r(r,a)/M
# Qualunque combinazione lineare/omogenea di grado 1 di queste frequenze
# (differenze, valori assoluti, medie pesate come in nu_solid_vect)
# conserva la stessa scaling 1/M. Questo e' il motivo per cui i fattori
# g_X vengono esposti separatamente: permettono di invertire M in forma
# chiusa (M = g_X(r,a)/nu_oss) senza dover campionare M su una griglia.
def g_phi(r, a):
    """Fattore geometrico (indipendente da M) di nu_phi. nu_phi = g_phi/M."""
    r = np.asarray(r)
    a = np.asarray(a)
    return C / (2*np.pi * Rg_SUN * (r**1.5 + a))

def g_theta(r, a):
    """Fattore geometrico di nu_theta. nu_theta = g_theta/M.
    L'argomento della radice puo' risultare negativo per r < r_isco(a)
    (regione non fisica, comunque esclusa a valle dalla maschera ISCO):
    si clampa a 0 per evitare NaN/warning spuri, in analogia a g_r."""
    arg = 1 - (4*a)/r**1.5 + (3*a**2)/r**2
    factor = np.sqrt(np.maximum(arg, 0))
    return g_phi(r, a) * factor

def g_r(r, a):
    """Fattore geometrico di nu_r. nu_r = g_r/M. Clampato a 0 sotto ISCO."""
    arg = 1 - 6/r + 8*a/r**1.5 - 3*a**2/r**2
    factor = np.sqrt(np.maximum(arg, 0))
    return g_phi(r, a) * factor

def g_LT(r, a):
    """Fattore geometrico di nu_LT = |nu_phi - nu_theta|. nu_LT = g_LT/M."""
    return np.abs(g_phi(r, a) - g_theta(r, a))

def g_per(r, a):
    """Fattore geometrico di nu_per = |nu_phi - nu_r|. nu_per = g_per/M."""
    return np.abs(g_phi(r, a) - g_r(r, a))


# Wrapper 
def nu_phi(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)
    return g_phi(r, a) / M

def nu_theta(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)
    return g_theta(r, a) / M

def nu_r(r, a, M=M_BH):
    r = np.asarray(r)
    a = np.asarray(a)
    M = np.asarray(M)
    return g_r(r, a) / M

# Kerr ISCO radius
def r_isco(a):
    a = np.asarray(a)
    # sign(a) but safe for vectorization (returns 0 if a=0)
    sgn = np.sign(a)

    Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    Z2 = np.sqrt(3*a**2 + Z1**2)
    return 3 + Z2 - sgn*np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))

# Kerr Event Horizon radius
def r_horizon(a):
    a = np.asarray(a)
    return 1 + np.sqrt(1 - a**2)


def _find_zero_crossing(r, f, sign_condition=None):
    """
    Trova il primo zero di f(r) (array valutato sulla griglia r) per
    interpolazione lineare tra campioni consecutivi.

    sign_condition: 'pos_to_neg', 'neg_to_pos', o None (qualsiasi cambio).

    Estratta a livello di modulo (prima era una closure interna di
    _resonance_radii) per poter essere riusata da QUALUNQUE routine di
    root-finding su una risonanza del tipo omega_tilde ± (frequenza
    secondaria) = 0, senza duplicare la logica: sia per r_CR/r_ILR/r_OLR
    qui sotto, sia per r_vertical_resonance (risonanza verticale
    generalizzata, usata dal c-mode di growth_rate.py).
    """
    signs = np.sign(f)
    diffs = np.diff(signs)

    if sign_condition == 'pos_to_neg':
        idx = np.where(diffs < 0)[0]
    elif sign_condition == 'neg_to_pos':
        idx = np.where(diffs > 0)[0]
    else:
        idx = np.where(diffs != 0)[0]

    if len(idx) == 0:
        return np.nan

    i = idx[0]
    denom = f[i + 1] - f[i]
    if denom == 0:
        return float(r[i])
    return float(r[i] - f[i] * (r[i + 1] - r[i]) / denom)


def _resonance_radii(a, m, nu_obs=NU0, M=M_BH, n_scan=8000, r_scan_max=5000.0):
    """
    Calcola tutti e tre i raggi di risonanza in un'unica passata,
    evitando ricalcoli ridondanti di nu_phi e nu_r.

    r_scan_max : raggio massimo dello scan radiale (default 5000.0,
        IDENTICO al valore hardcoded nella versione precedente: nessun
        cambiamento di comportamento per i chiamanti che non lo
        specificano esplicitamente). Esposto come parametro per essere
        riusato da r_vertical_resonance con lo stesso dominio di ricerca.

    Restituisce
    -----------
    r_CR, r_ILR, r_OLR : float  raggi in r_g (NaN se non trovati)
    """
    a     = float(a)
    isco  = float(r_isco(a))
    r     = np.geomspace(isco * 1.001, r_scan_max, n_scan)

    # calcolo unico delle frequenze
    om_phi   = 2 * np.pi * nu_phi(r, a, M)   # Ω_φ(r)
    kappa    = 2 * np.pi * nu_r(r, a, M)      # κ(r)
    om_obs   = 2 * np.pi * nu_obs
    om_tilde = om_obs - m * om_phi             # ω̃(r)

    # corotation: ω̃ = 0, da positivo a negativo verso l'interno
    # (siccome andiamo da r grande a r piccolo, lo scan è inverso:
    #  usiamo il primo cambio qualsiasi e lasciamo la fisica guidare)
    r_cr  = _find_zero_crossing(r, om_tilde)

    # ILR: ω̃ + κ = 0
    r_ilr = _find_zero_crossing(r, om_tilde + kappa, sign_condition='neg_to_pos')

    # OLR: ω̃ − κ = 0
    r_olr = _find_zero_crossing(r, om_tilde - kappa)

    return r_cr, r_ilr, r_olr


# wrapper
def r_corotation(a, m,  nu_obs=NU0, M=M_BH, n_scan=8000, r_scan_max=5000.0):
    return _resonance_radii(a, m, nu_obs, M, n_scan, r_scan_max)[0]

def r_ilr(a, m,  nu_obs=NU0, M=M_BH, n_scan=8000, r_scan_max=5000.0):
    return _resonance_radii(a, m, nu_obs, M, n_scan, r_scan_max)[1]

def r_olr(a, m, nu_obs=NU0, M=M_BH, n_scan=8000, r_scan_max=5000.0):
    return _resonance_radii(a, m, nu_obs, M, n_scan, r_scan_max)[2]


def r_vertical_resonance(a, m, n, nu_obs=NU0, M=M_BH, n_scan=8000, r_scan_max=5000.0):
    """
    Raggio di risonanza verticale generalizzata: primo r, scandendo da
    r_isco(a) verso l'esterno, dove

        |omega_tilde(r)| = sqrt(n) * Omega_perp(r)
        <=>  omega_tilde(r) + sqrt(n)*Omega_perp(r) = 0

    Giustificazione della condizione (stessa struttura di r_ilr sopra,
    con Omega_perp=2*pi*nu_theta al posto di kappa=2*pi*nu_r):
    per r in [r_isco, r_corotation), omega_tilde(r) = omega_obs -
    m*Omega_phi(r) e' NEGATIVO, perche' vicino a r_isco l'orbita
    kepleriana e' molto piu' rapida della frequenza osservata
    (Omega_phi(r_isco) >> omega_obs/m), e Omega_phi(r) e' strettamente
    decrescente (g_phi(r,a)=cost/(r^1.5+a)), quindi omega_tilde cresce
    monotonicamente verso omega_obs (finito) per r->r_corotation^-.
    Percio' |omega_tilde|=-omega_tilde su tutto il dominio, e la
    condizione diventa omega_tilde+sqrt(n)*Omega_perp=0 con il primo
    attraversamento da negativo a positivo (sign_condition='neg_to_pos')
    che individua il primo raggio in cui sqrt(n)*Omega_perp supera
    |omega_tilde| in modulo.

    n=1: risonanza verticale ordinaria. n generico: usata dal c-mode di
    Tsang & Lai 2008 (growth_rate.py, r_IVR: confine Regione I/Regione
    II), dove la condizione di trapping coinvolge n*Omega_perp^2.

    Riusa _find_zero_crossing (stessa routine di r_ilr/r_olr/r_corotation):
    nessuna duplicazione della logica di root-finding.
    """
    a = float(a); M = float(M); nu_obs = float(nu_obs); n = float(n)
    isco = float(r_isco(a))
    r = np.geomspace(isco * 1.001, r_scan_max, n_scan)

    om_phi     = 2 * np.pi * nu_phi(r, a, M)
    Omega_perp = 2 * np.pi * nu_theta(r, a, M)
    om_obs     = 2 * np.pi * nu_obs
    om_tilde   = om_obs - m * om_phi

    f = om_tilde + np.sqrt(n) * Omega_perp
    return _find_zero_crossing(r, f, sign_condition='neg_to_pos')



##########################################################
# --------------------------------------------------------
##########################################################
# RELATIVISTIC PRECESSION MODEL FREQUENCIES
# lense-thirring precession frequency
def nu_LT(r, a, M=M_BH):
    r = np.asarray(r); a = np.asarray(a); M = np.asarray(M)
    return g_LT(r, a) / M

#periastron precession frequency
def nu_per(r, a, M=M_BH):
    r = np.asarray(r); a = np.asarray(a); M = np.asarray(M)
    return g_per(r, a) / M


##########################################################
# --------------------------------------------------------
##########################################################
# PRECESSING INNER FLOW MODEL FREQUENCY
def nu_solid_vect(a, rin, rout, zeta, M=M_BH, n_rad=2000):
    a    = np.atleast_1d(a)
    rin  = np.atleast_1d(rin)
    rout = np.atleast_1d(rout)
    zeta = np.atleast_1d(zeta)
    M    = np.atleast_1d(M)

    x = np.linspace(0, 1, n_rad)
    x = x.reshape((1,) * a.ndim + (n_rad,))

    R = rin[..., None] + x * (rout - rin)[..., None]

    Sigma  = R**(-zeta[..., None])
    weight = Sigma * R**3 * nu_phi(R, a[..., None], M[..., None])

    num = np.trapezoid(nu_LT(R, a[..., None], M[..., None]) * weight, x, axis=-1)
    den = np.trapezoid(weight, x, axis=-1)

    return num / den


import matplotlib.pyplot as plt

def set_style_beamer():
    # Beamer 16:9: textwidth ≈ 12.8 cm = 5.04 in
    # figura intera = ~5 in, mezza figura = ~2.4 in
    # font size base beamer = 11pt → label ~10pt nei grafici
    
    plt.rcParams.update({
        # "figure.figsize": (5.0, 3.1),   # figura intera (aspect 16:10 circa)
        # "figure.figsize": (2.4, 2.4),   # figura quadrata mezza colonna
        # "figure.figsize": (5.0, 2.4),   # figura wide intera larghezza
        
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,      # linee più visibili su schermo
        "lines.markersize": 5,
    })

def set_style():
    plt.rcParams.update({
        #"figure.figsize": (3.4, 2.4),
        # 3.4 per 1 col, 7 per 2 col
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "axes.linewidth": 0.5,
    })
def fix_spines(ax):
    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)