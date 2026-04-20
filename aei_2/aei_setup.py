
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys, os
# path setup PRIMA di qualsiasi import locale
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simple_disc import disk_model_simple
from setup import (
    r_isco, nu_phi, nu_r, r_horizon,
    Rg_SUN, M_BH, NU0,
)

HOR = 0.001
mm = 1
ALPHA_VISC = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

ALP_B = 5 / 4    # Simple-v1 magnetic field exponent
ALP_S = 3 / 5    # Simple-v1 surface density exponent

CONFIGS = [
    dict(mm=1, hor=0.05,  color='#3b82f6', ls='-',  label=r'$m=1,\ H/r=0.05$'),
    dict(mm=1, hor=0.001, color='#f97316', ls='-',  label=r'$m=1,\ H/r=10^{-3}$'),
    dict(mm=2, hor=0.05,  color='#22c55e', ls='--', label=r'$m=2,\ H/r=0.05$'),
    dict(mm=2, hor=0.001, color='#ef4444', ls='--', label=r'$m=2,\ H/r=10^{-3}$'),
]

A_GRID      = np.linspace(-1, 1, 100)   # spin grid
B00_GRID    = np.logspace(1,  8, 48)          # B00 [G]
SIGMA0_GRID = np.logspace(2,  7, 36)          # Sigma0 [g/cm²]

N_R        = 300   # radial points for full profile (ISCO → OLR)
N_R_CAV    = 200   # radial points for cavity (ISCO → ILR)

# Reference values for dQ/dr plots (only sign matters, not amplitude)
B00_REF    = 1e4
SIGMA0_REF = 1e4

# Spins to show in the multi-spin dQ/dr panel
A_SUBPLOT = [-1, -0.5, 0.0, 0.5, 0.90, 1]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SOLVER  (identico per tutti i modelli)
# ═══════════════════════════════════════════════════════════════════════════

def solve_k_aei(r_rg, a, B0, Sigma, c_s, m=mm, M=M_BH):
    """
    Risolve la relazione di dispersione AEI (Tagger & Pellat 1999, Eq. 17):

        ω̃² = κ² + (2B₀²/Σ)(k/r) + (k²/r²) c_s²

    con  ω̃ = ω_obs − m Ω_φ,  k adimensionale (numero d'onda in s = ln r).

    La quadratica in k è:
        A k² + B k + C = 0
        A = c_s² / r_cm²
        B = 2 B₀² / (Σ r_cm)
        C = κ² − ω̃²

    Parametri
    ----------
    r_rg  : array_like   raggio in unità di r_g  (adimensionale)
    a     : float        spin del BH  [adim, −1…1]
    B0    : array_like   campo magnetico verticale  [G]
    Sigma : array_like   densità superficiale  [g/cm²]
    c_s   : array_like   velocità del suono  [cm/s]
    m     : int          numero d'onda azimutale  (default 1)
    M     : float        massa BH  [M_sun]

    Restituisce
    -----------
    k : ndarray
        Numero d'onda adimensionale  (NaN dove non esiste soluzione reale > 0).
        Relazioni utili:
          k_fisico [1/cm]  =  k / (r_rg * Rg)
          k_adimensionale    [adim]  =  k                  (≡ k_fis × r_cm)
    """
    r_rg  = np.asarray(r_rg,  dtype=float)
    B0    = np.asarray(B0,    dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    c_s   = np.asarray(c_s,   dtype=float)

    Rg   = Rg_SUN * M
    r_cm = r_rg * Rg                              # cm — richiesto dalla dim. analysis

    kappa_sq  = (2 * np.pi * nu_r(r_rg,  a, M))**2
    Omega_phi = 2 * np.pi * nu_phi(r_rg, a, M)
    om_tilde  = 2 * np.pi * NU0 - m * Omega_phi  # ω_obs − m Ω_φ

    # Coefficienti  A k² + B k + C = 0  (tutti in s⁻²)
    A  = c_s**2 / r_cm**2
    B  = 2*B0**2  / (Sigma*r_cm)
    C  = kappa_sq - om_tilde**2

    Delta = B**2 - 4*A*C

    k = np.full_like(r_rg, np.nan)
    ok = Delta >= 0
    if np.any(ok):
        sqD      = np.sqrt(Delta[ok])
        kp       = (-B[ok] + sqD) / (2*A[ok])
        km       = (-B[ok] - sqD) / (2*A[ok])
        # preferisci il ramo k_plus (più grande); fallback su k_minus
        k[ok] = np.where(kp > 0, kp, np.where(km > 0, km, np.nan))

    return k   # adimensionale


# ==========================================================
# checks
# =====================================================
def compute_beta(B0, Sigma, c_s, r_rg, hr=HOR, M=M_BH):
    """
    Parametro plasma beta:

        β = 8π Σ c_s² / (2 H B₀²)    con H = hr · r · Rg
        (altezza totale del disco = 2H)
    Parametri
    ----------
    B0, Sigma, c_s : array_like   profili fisici  [G, g/cm², cm/s]
    r_rg  : array_like            raggio  [r_g]
    hr    : float                 aspect ratio H/r
    M     : float                 massa BH  [M_sun]

    Restituisce
    -----------
    beta : ndarray
    """
    Rg = Rg_SUN * M
    H  = hr * np.asarray(r_rg) * Rg
    return 8*np.pi * np.asarray(Sigma) * np.asarray(c_s)**2 / (2*H * np.asarray(B0)**2)

def compute_dQdr(r_rg, a, B0_func, Sigma_func, M=M_BH, dr_frac=0.01):
    """
    Derivata radiale di  Q = Ω_φ · Σ / B₀²,  con differenze finite.

    Condizione AEI: dQ/dr > 0.

    Parametri
    ----------
    r_rg      : array_like   raggi  [r_g]
    a         : float        spin
    B0_func   : callable     B0_func(r_rg)  → B0  [G]
    Sigma_func: callable     Sigma_func(r_rg) → Sigma  [g/cm²]
    M         : float        massa  [M_sun]
    dr_frac   : float        passo relativo per la diff. finita

    Restituisce
    -----------
    dQdr : ndarray   [unità di 1/r_g, relative a Ω·Σ/B²]
    """
    r_rg = np.asarray(r_rg, dtype=float)
    dr   = dr_frac * r_rg
    rp   = r_rg + dr

    def Q(r):
        return 2*np.pi * nu_phi(r, a, M) * Sigma_func(r) / B0_func(r)**2

    return (Q(rp) - Q(r_rg)) / dr


# ═══════════════════════════════════════════════════════════════════════════
# 2b.  RISONANZE LINDBLAD E COROTAZIONE
# ═══════════════════════════════════════════════════════════════════════════

def r_corotation(a, nu_obs=NU0, m=mm, M=M_BH, n_scan=8000):
    """
    Raggio di corotazione: il raggio dove ω̃ = ω_obs − m Ω_φ = 0,
    ovvero dove Ω_φ(r_CR) = ω_obs / m.

    Al raggio di corotazione l'onda spirale AEI ruota solidalmente con
    il disco — è il centro geometrico dell'instabilità di Rossby.

    Parametri
    ----------
    a      : float   spin adimensionale  [−1, 1]
    nu_obs : float   frequenza osservata [Hz]  (default NU0)
    m      : int     numero d'onda azimutale
    M      : float   massa BH [M_sun]
    n_scan : int     punti di scansione radiale

    Restituisce
    -----------
    r_CR : float   raggio di corotazione in r_g  (NaN se non trovato)
    """
    a = float(a)
    isco = float(r_isco(a))
    r = np.geomspace(isco * 1.001, 5000.0, n_scan)
    om_tilde = 2*np.pi*nu_obs - m * 2*np.pi * nu_phi(r, a, M)
    # corotation: om_tilde cambia segno (da positivo a negativo procedendo verso l'interno)
    idx = np.where(np.diff(np.sign(om_tilde)) != 0)[0]
    if len(idx) == 0:
        return np.nan
    # interpolazione lineare al punto di zero
    i = idx[0]
    r_cr = r[i] - om_tilde[i] * (r[i+1] - r[i]) / (om_tilde[i+1] - om_tilde[i])
    return float(r_cr)


def r_ilr(a, nu_obs=NU0, m=mm, M=M_BH, n_scan=8000):
    """
    Inner Lindblad Resonance (ILR): confine esterno della cavity AEI.

    Condizione: ω̃ + κ = 0  (dove ω̃ = ν_obs - m Ω_φ < 0 vicino all ISCO).

    Significato fisico
    ------------------
    Vicino all ISCO, Ω_φ >> ν_obs/m → ω̃ < 0.
    La cavity di propagazione dell onda AEI è r_ISCO < r < r_ILR
    dove ω̃² > κ², ovvero ω̃ < -κ (lato interno della risonanza).
    Alla ILR (ω̃ = -κ) le onde vengono riflesse e formano standing waves
    con autofequenze discrete — le uniche configurazioni che producono QPO.

    Parametri
    ----------
    a      : float   spin adimensionale
    nu_obs : float   frequenza osservata [Hz]
    m      : int     numero d onda azimutale
    M      : float   massa BH [M_sun]
    n_scan : int     punti di scansione radiale

    Restituisce
    -----------
    r_ILR : float   raggio ILR in r_g  (NaN se non trovato)
    """
    a = float(a)
    isco = float(r_isco(a))
    r = np.geomspace(isco * 1.001, 5000.0, n_scan)

    kappa    = 2*np.pi * nu_r(r, a, M)
    om_tilde = 2*np.pi*nu_obs - m * 2*np.pi * nu_phi(r, a, M)

    # ILR: omega_tilde + kappa = 0  (transizione da negativo a positivo)
    diff = om_tilde + kappa
    sign_changes = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(sign_changes) == 0:
        return np.nan

    i = sign_changes[0]
    denom = diff[i+1] - diff[i]
    if denom == 0:
        return float(r[i])
    return float(r[i] - diff[i] * (r[i+1] - r[i]) / denom)

def r_olr(a, nu_obs=NU0, m=mm, M=M_BH, n_scan=8000):
    """
    Outer Lindblad Resonance (OLR): ω̃ − κ = 0.

    The OLR marks the outer boundary of the propagation region for
    outward-propagating density waves. It satisfies:

        ω_obs − m·Ω_φ(r) = +κ(r)

    Parameters
    ----------
    a      : float   dimensionless spin  [−1, 1]
    nu_obs : float   observed frequency  [Hz]  (default NU0)
    m      : int     azimuthal wavenumber
    M      : float   BH mass [M_sun]
    n_scan : int     radial scan points

    Returns
    -------
    r_OLR : float   OLR radius in r_g  (NaN if not found)
    """
    a = float(a)
    isco = float(r_isco(a))
    r = np.geomspace(isco * 1.001, 5000.0, n_scan)

    kappa    = 2*np.pi * nu_r(r, a, M)
    om_tilde = 2*np.pi*nu_obs - m * 2*np.pi * nu_phi(r, a, M)

    # OLR: om_tilde − kappa = 0
    diff = om_tilde - kappa
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        return np.nan

    i = sign_changes[0]
    denom = diff[i+1] - diff[i]
    if denom == 0:
        return float(r[i])
    return float(r[i] - diff[i] * (r[i+1] - r[i]) / denom)



# ═══════════════════════════════════════════════════════════════════════════
# 4.  FULL DISC INFOS PER ANALISI RADIALE
# ═══════════════════════════════════════════════════════════════════════════
def compute_disk_profile(
    disk_model, params,
    mm=mm, hr=HOR, M=M_BH,
    r_min=None, r_max=None, n_points=500,
    r_max_aei=1000.0, n_points_ext=80,
):
    """
    Profilo radiale completo per qualsiasi modello di disco.
 
    Funziona con la stessa `disk_model` passata a `find_rossby` —
 
    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, zone)
        Stessa firma usata in find_rossby.
        Può restituire un quinto elemento `info` (dict con r_ISCO, r_AB,
        r_BC, mdot, ...) che viene incluso in meta automaticamente.
 
    params : dict
        Parametri scalari del disco, es.:
          {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5}
        Vengono passati a disk_model come **kwargs.
 
    mm : int
        Modo azimutale m della perturbazione AEI.
 
    hr : float
        Aspect ratio H/r di fallback (usato se disk_model non restituisce
        hr per punto). Entra nel vincolo WKB superiore: k_max = r/H = 1/hr.
 
    M : float
        Massa BH [M_sun].
 
    r_min : float o None
        Raggio interno della griglia [r_g].
        Se None: r_isco(params['a'])
 
    r_max : float o None
        Raggio esterno della griglia [r_g].
        Se None: richiede che disk_model restituisca info['r_BC'],
                 allora r_max = 3 × r_BC.
                 Se info non è disponibile, solleva ValueError.
 
    n_points : int
        Numero di punti radiali log-spaziati.
 
    Vincolo WKB
    -----------
    Il check sul numero d'onda usa ora i limiti fisicamente corretti:

        k_min = 1          →  perturbazione sub-orbitale  (λ < 2πr)
        k_max(r) = 1/hr(r) →  perturbazione sopra-disco   (λ > 2πH)

    k_max dipende dal raggio quando hr(r) non è costante (modelli SS/NT):
    il valore hr(r) viene estratto direttamente dall'output di disk_model
    (quarto elemento della tupla restituita), garantendo coerenza con la
    struttura termica calcolata dal modello.

    Restituisce
    -----------
    df : pd.DataFrame
        Colonne: r, zone, B0, Sigma, c_s, hr, k, k_max,
                 beta, dQdr, k_valid, beta_valid, shear_valid,
                 aei_valid, ilr_valid, aei_ilr_valid.
        k_max è il limite WKB superiore per ogni punto: r/H(r) = 1/hr(r).
 
    meta : dict
        Contiene almeno: r_H, r_ISCO, a, mm, hr, M, + tutto ciò che
        disk_model restituisce in info (r_AB, r_BC, mdot, norms, ...).
 
    Esempi
    ------
    # modello SS
    df, meta = compute_disk_profile(
        disk_model = lambda r, **p: disk_model_SS(r, **p, alpha_visc=0.1, hr=0.05),
        params     = {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5},
        mm=1, hr=0.05,
    )
 
    # modello NT — identico, solo disk_model cambia
    df, meta = compute_disk_profile(
        disk_model = lambda r, **p: disk_model_NT(r, **p, alpha_visc=0.1, hr=0.05),
        params     = {'a': 0.5, 'B00': 1e6, 'Sigma0': 1e5},
        mm=1, hr=0.05,
    )
 
    # con r_min/r_max espliciti
    df, meta = compute_disk_profile(
        disk_model = my_model,
        params     = {'a': 0.5, 'alpha': 0.3},
        mm=1, hr=0.05,
        r_min=5.0, r_max=500.0,
    )
    """
    a = float(params['a'])
 
    # ── griglia radiale ───────────────────────────────────────────────────
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
 
    if r_min is None:
        r_min = rISCO
 
    # Chiamata di prova su un singolo punto per estrarre info se disponibile
    _r_probe = np.array([r_min])
    _result  = disk_model(_r_probe, **params)
    # Nuova firma (6 valori): B0, Sigma, c_s, hr, zone, info
    # Vecchia firma (5 valori): B0, Sigma, c_s, zone, info
    _n_probe = len(_result)
    if _n_probe >= 6:
        _info = _result[5]
    elif _n_probe == 5:
        _info = _result[4]
    else:
        _info = {}
 
    if r_max is None:
        if 'r_max_hint' in _info:
            r_max = 3.0 * _info['r_max_hint']
        elif 'r_BC' in _info:
            r_max = 3.0 * _info['r_BC']
        else:
            raise ValueError(
                "r_max non specificato e disk_model non restituisce info['r_BC']. "
                "Passare r_max esplicitamente oppure aggiornare disk_model per "
                "restituire un quinto elemento info={'r_BC': ...}."
            )
 
    # ── griglia composita: densa nella zona AEI, rada all'esterno ──────────
    if r_max_aei is not None and r_max > r_max_aei * 1.01:
        r_arr = np.concatenate([
            np.geomspace(r_min,     r_max_aei, n_points),
            np.geomspace(r_max_aei, r_max,     n_points_ext)[1:],  # evita duplicato
        ])
    else:
        r_arr = np.geomspace(r_min, r_max, n_points)
 
    # ── profili fisici ────────────────────────────────────────────────────
    result    = disk_model(r_arr, **params)
    n_ret     = len(result)
    B0_arr    = np.asarray(result[0], dtype=float)
    Sigma_arr = np.asarray(result[1], dtype=float)
    cs_arr    = np.asarray(result[2], dtype=float)
    # Nuova firma (6 valori): B0, Sigma, c_s, hr, zone, info
    # Vecchia firma (5 valori): B0, Sigma, c_s, zone, info
    if n_ret >= 6:
        zone_arr = np.asarray(result[4])
        info     = result[5]
    elif n_ret == 5:
        zone_arr = np.asarray(result[3])
        info     = result[4]
    elif n_ret == 4:
        zone_arr = np.asarray(result[3])
        info     = {}
    else:
        zone_arr = np.full(len(r_arr), 'N/A', dtype=object)
        info     = {}
 
    # ── profili hr ───────────────────────────────────────────────────────────
    # Recupera hr per ogni punto radiale PRIMA del solver AEI, così
    # compute_beta usa hr(r) reale invece del valore costante del parametro.
    # - SS/NT: il 4° elemento del return è hr array dal modello
    # - Simple / vecchi modelli: hr costante dal parametro della funzione
    if n_ret >= 6:
        hr_ret = result[3]   # nuova firma: B0, Sigma, c_s, hr, zone, info
        if np.isscalar(hr_ret):
            hr_arr = np.full(len(r_arr), float(hr_ret))
        else:
            hr_arr = np.asarray(hr_ret, dtype=float)
    else:
        hr_arr = np.full(len(r_arr), float(hr))
 
    # ── solver AEI — solo nella zona interna (r <= r_max_aei) ───────────
    aei_zone = (r_arr <= r_max_aei) if r_max_aei is not None else np.ones(len(r_arr), bool)
    r_aei    = r_arr[aei_zone]
 
    k_arr    = np.full(len(r_arr), np.nan)
    beta_arr = np.full(len(r_arr), np.nan)
    dQdr_arr = np.full(len(r_arr), np.nan)
 
    if aei_zone.any():
        k_arr[aei_zone]    = solve_k_aei(r_aei, a, B0_arr[aei_zone],
                                          Sigma_arr[aei_zone], cs_arr[aei_zone],
                                          m=mm, M=M)
        beta_arr[aei_zone] = compute_beta(B0_arr[aei_zone], Sigma_arr[aei_zone],
                                           cs_arr[aei_zone], r_aei,
                                           hr_arr[aei_zone], M)   # hr(r) dal modello
        dQdr_arr[aei_zone] = compute_dQdr(r_aei, a,
                                           _make_interp(r_aei, B0_arr[aei_zone]),
                                           _make_interp(r_aei, Sigma_arr[aei_zone]), M)
 
    # ── maschere di validità ──────────────────────────────────────────────
    # k_max(r) = r/H(r) = 1/hr(r)  — dipende dal modello e dal raggio.
    #
    # Guard: se hr(r) ≈ 0 (es. NT vicino all'ISCO dove Σ→0 e c_s→0),
    # il modello non è fisicamente significativo in quel punto per l'analisi
    # WKB; usiamo il valore di fallback `hr` passato a compute_disk_profile
    # come pavimento, per evitare k_max=∞ che non filtra nulla.
    #
    # Fisicamente: quando hr_modello < hr_fallback il disco è geometricamente
    # più sottile del caso isotermo di riferimento; in quel limite l'approssimazione
    # WKB è ancora più stringente, non più permissiva.
    k_max_arr = np.full(len(r_arr), np.inf)
    if aei_zone.any():
        hr_aei    = hr_arr[aei_zone]
        #hr_floor  = max(float(hr), 1e-4)          # mai sotto 10^-4 per stabilità
        #hr_eff    = np.maximum(hr_aei, hr_floor)   # prende il massimo tra i due
        k_max_arr[aei_zone] = 1.0 / hr_aei

    k_valid     = np.where(aei_zone,
                           ~np.isnan(k_arr) & check_k_wkb(
                               np.nan_to_num(k_arr),
                               k_min=1.0,
                               k_max=k_max_arr),
                           False)
    beta_valid  = np.where(aei_zone, beta_arr <= 1.0, False)
    shear_valid = np.where(aei_zone, dQdr_arr > 0,    False)
    aei_valid   = k_valid & beta_valid & shear_valid
 
    # ── risonanze Lindblad e corotazione ─────────────────────────────────
    rILR = r_ilr(a, NU0, mm, M)
    rCR  = r_corotation(a, NU0, mm, M)
    ilr_valid     = (r_arr < rILR) if np.isfinite(rILR) else np.zeros(len(r_arr), dtype=bool)
    aei_ilr_valid = aei_valid & ilr_valid   # soluzioni AEI dentro la cavity QPO
 
    df = pd.DataFrame({
        'r':             r_arr,
        'zone':          zone_arr,
        'B0':            B0_arr,
        'Sigma':         Sigma_arr,
        'c_s':           cs_arr,
        'hr':            hr_arr,
        'k':             k_arr,
        'k_max':         k_max_arr,   # r/H(r) — limite WKB superiore per punto
        'beta':          beta_arr,
        'dQdr':          dQdr_arr,
        'k_valid':       k_valid,
        'beta_valid':    beta_valid,
        'shear_valid':   shear_valid,
        'aei_valid':     aei_valid,
        'ilr_valid':     ilr_valid,
        'aei_ilr_valid': aei_ilr_valid,
    })
 
    _merged = {**params, **info}
    meta = dict(r_H=rH, r_ISCO=rISCO, r_ILR=rILR, r_CR=rCR,
                mm=mm, hr=hr, M=M, **_merged)
    return df, meta


#  ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def make_disk(r, a, B00, Sigma0, hor):
    """Wrapper for Simple-v1 disk model."""
    return disk_model_simple(r, a, B00, Sigma0,
                             alpha_B=ALP_B, alpha_S=ALP_S, hr=hor, M=M_BH)

def get_resonances(a, mm):
    """Compute (r_ILR, r_OLR, r_CR) for given spin and azimuthal mode."""
    return (r_ilr(a, NU0, mm, M_BH),
            r_olr(a, NU0, mm, M_BH),
            r_corotation(a, NU0, mm, M_BH))

def _make_interp(r_nodes, y_nodes):
    """
    Restituisce una funzione  f(r) → y  tramite interpolazione lineare
    in spazio log-log (power-law locale tra i nodi).
    Usato internamente da find_rossby per calcolare dQ/dr senza
    richiamare disk_model una seconda volta.
    """
    log_r = np.log(r_nodes)
    log_y = np.log(np.maximum(y_nodes, 1e-300))

    def interp(r_query):
        r_query = np.asarray(r_query, dtype=float)
        return np.exp(np.interp(np.log(r_query), log_r, log_y))

    return interp