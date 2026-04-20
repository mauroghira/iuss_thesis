"""
aei_common.py
=============
Funzioni comuni a tutti i modelli AEI (simple_disc, full_disk_SS, NT).

Contiene:
  - solve_k_aei        : solver della relazione di dispersione (Tagger & Pellat 1999, Eq.17)
  - check_k_wkb        : vincolo WKB su k adimensionale
  - check_beta_aei     : β ≤ 1 (pressione magnetica dominante)
  - check_shear_aei    : dQ/dr > 0 (condizione di shear)
  - compute_beta       : calcolo di β
  - compute_dQdr       : calcolo di dQ/dr con differenze finite
  - r_corotation       : raggio di corotazione ω̃ = 0
  - r_ilr              : Inner Lindblad Resonance (confine della cavity AEI)
  - check_ilr_aei      : constraint r < r_ILR (QPO fisicamente possibile)
  - find_rossby        : finder vettorizzato, funziona con qualsiasi modello di disco
  - compute_disk_profile : profilo radiale completo con tutti i constraint

──────────────────────────────────────────────────────────────────────────────
NOTE SULLE UNITÀ (da Tagger & Pellat 1999)
──────────────────────────────────────────────────────────────────────────────
Il paper lavora nella variabile  s = ln(r),  quindi il numero d'onda k
è definito come:
    ∂/∂s Φ_M ≈ ik Φ_M   →   k adimensionale

Il numero d'onda fisico radiale è  k_fis = k / r_cm  [1/cm].

Analisi dimensionale dell'Eq. (17):

  Termine magnetico:  2B₀²/Σ · (k/r)
    [G²/(g/cm²)] · [1/cm]  =  [cm·s⁻²] · [1/cm]  =  s⁻²  ✓   (r in cm)

  Termine pressione:  k²/r² · c_s²
    [adim²/cm²] · [cm²/s²]  =  s⁻²  ✓                          (r in cm)

Quindi i coefficienti A, B della quadratica devono usare r in cm.
Il k restituito dal solver è adimensionale.

Conseguenze sui check fisici:
  Il WKB richiede  k_fis ≫ 1/r, ovvero  k_fis · r ≫ 1.
  Poiché  k_fis · r_cm = k  (adimensionale), il bound WKB si traduce in:
      k_min < k < k_max    (tutto adimensionale)
  con k_min = 0.1, k_max = 10  (come da notebook, sezione AEI setup).

──────────────────────────────────────────────────────────────────────────────
STRUTTURA DEL DISCO — interfaccia per find_rossby
──────────────────────────────────────────────────────────────────────────────
find_rossby accetta un callable  `disk_model`  con firma:

    B0, Sigma, c_s = disk_model(r_rg, **extra_params)

dove r_rg è un array 1D di raggi in r_g e **extra_params è qualsiasi
set di parametri scalari (a, B00, Sigma0, alpha_visc, …).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
sys.path.append("..")
from setup import (
    r_isco, nu_phi, nu_r, r_horizon,
    Rg_SUN, M_BH, NU0,
)

HOR = 0.001
mm = 1
ALPHA_VISC = 0.1

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


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CHECK FISICI  (tutti operano su k adimensionale)
# ═══════════════════════════════════════════════════════════════════════════

def check_k_wkb(k, k_min=1.0, k_max=None, hr=None):
    """
    Vincolo WKB sul numero d'onda adimensionale k = k_fis · r_cm.

    Il vincolo WKB corretto richiede che la lunghezza d'onda della
    perturbazione sia sub-orbitale ma sopra-disco:

        1/r  ≪  k_fis  ≪  1/H   →   1  ≪  k  ≪  r/H = 1/(H/r)

    Limiti:
      k_min = 1          : perturbazione sub-orbitale  (k_fis · r > 1)
      k_max = r/H = 1/hr : perturbazione sopra-disco   (k_fis · H < 1)

    k_max può essere un array (quando hr(r) non è costante, es. modelli SS/NT),
    oppure uno scalare, oppure None (nel qual caso si usa hr per calcolarlo).

    Parametri
    ----------
    k     : array_like         numero d'onda adimensionale
    k_min : float              limite inferiore (default 1.0)
    k_max : float | array | None
        Se fornito esplicitamente, viene usato direttamente.
        Se None, viene derivato da hr come k_max = 1/hr.
        Può essere un array della stessa forma di k.
    hr    : float | array | None
        Aspect ratio H/r. Usato solo se k_max is None.
        Se None e k_max is None, si usa il valore globale HOR.

    Restituisce
    -----------
    mask : ndarray bool
    """
    k = np.asarray(k, dtype=float)

    # ── calcola k_max ─────────────────────────────────────────────────────
    if k_max is None:
        if hr is None:
            hr = HOR          # fallback al valore globale del modulo
        hr_arr = np.asarray(hr, dtype=float)
        k_max_arr = 1.0 / np.where(hr_arr > 0, hr_arr, np.inf)
    else:
        k_max_arr = np.asarray(k_max, dtype=float)

    return np.isfinite(k) & (k >= k_min) & (k <= k_max_arr)


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


def check_beta_aei(B0, Sigma, c_s, r_rg, hr=HOR, M=M_BH, beta_max=1.0):
    """
    Condizione AEI: β ≤ beta_max  (pressione magnetica dominante).

    Restituisce
    -----------
    mask : ndarray bool
    """
    return compute_beta(B0, Sigma, c_s, r_rg, hr, M) <= beta_max


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


def check_shear_aei(r_rg, a, B0_func, Sigma_func, M=M_BH, dr_frac=0.01):
    """
    Condizione di shear AEI: dQ/dr > 0.

    Parametri: stessi di compute_dQdr.

    Restituisce
    -----------
    mask : ndarray bool
    """
    return compute_dQdr(r_rg, a, B0_func, Sigma_func, M, dr_frac) > 0


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

def check_ilr_aei(r_rg, a, nu_obs=NU0, m=mm, M=M_BH):
    """
    Constraint ILR: seleziona solo le soluzioni dentro la cavity di risonanza
    (r < r_ILR), ovvero le uniche che possono generare un QPO coerente.

    Soluzioni con r > r_ILR soddisfano la relazione di dispersione ma corrispondono
    a onde esterne alla cavity: si propagano verso l'esterno e si dissipano senza
    produrre modi normali quantizzati.

    Parametri
    ----------
    r_rg   : array_like   raggi in r_g
    a      : float        spin
    nu_obs : float        frequenza osservata [Hz]
    m      : int          modo azimutale
    M      : float        massa BH [M_sun]

    Restituisce
    -----------
    mask : ndarray bool   True dove r < r_ILR
    """
    r_rg  = np.asarray(r_rg, dtype=float)
    r_ILR = r_ilr(a, nu_obs, m, M)
    if np.isnan(r_ILR):
        return np.zeros(r_rg.shape, dtype=bool)
    return r_rg < r_ILR


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
# 3.  FINDER VETTORIZZATO
# ═══════════════════════════════════════════════════════════════════════════

def find_rossby(
    r_vec, param_grid, disk_model,
    m=mm, hr=HOR, M=M_BH,
    check_k=True, check_beta=True, check_shear=True, check_ilr=False,
    k_min=1.0, k_max=None, beta_max=1.0, dr_frac=0.01,
):
    """
    Finder vettorizzato della relazione di dispersione AEI.
 
    Funziona con qualsiasi modello di disco purché venga fornito
    l'adapter `disk_model` (vedi sotto).
 
    ──────────────────────────────────────────────────────────────────
    Strategia di vettorizzazione
    ──────────────────────────────────────────────────────────────────
    L'ostacolo alla piena vettorizzazione è che per i modelli SS e NT
    le frontiere  r_AB, r_BC  dipendono da (a, Sigma0) e non da r:
    non si possono precalcolare in un unico meshgrid.
 
    Soluzione adottata:
      - si itera sul prodotto cartesiano dei parametri *fissi per disco*
        (tutto tranne r),  che è in genere N_a × N_B00 × N_Sigma0 ≪ N_tot
      - per ogni combinazione, si chiama disk_model(r_vec, **params)
        che restituisce (B0, Sigma, c_s) sull'intero vettore r — questa
        chiamata è completamente vettorizzata in r
      - i check fisici e il solver sono anch'essi vettorizzati in r
 
    Costo: O(N_a × N_B00 × N_Sigma0 × N_r)  con solo  N_a × N_B00 × N_Sigma0
    chiamate Python (contro  N_a × N_B00 × N_Sigma0 × N_r  nel vecchio codice).
 
    ──────────────────────────────────────────────────────────────────
    Parametri
    ──────────────────────────────────────────────────────────────────
    r_vec      : array_like
        Raggi in r_g su cui calcolare i profili.
 
    param_grid : dict  {nome: array_1d}
        Griglia dei parametri *che non dipendono da r*.
        Deve contenere almeno 'a'.
        Esempio per simple_disc:   {'a': ..., 'B00': ..., 'Sigma0': ...}
        Esempio per full_disk_SS:  {'a': ..., 'B00': ..., 'Sigma0': ...}
        (i parametri 'r' vanno in r_vec, non qui)
 
    disk_model : callable
        disk_model(r_rg, **row) → (B0, Sigma, c_s)
        dove row è un dict con i parametri scalari della riga corrente.
        Vedi sezione adapter in fondo al file.
 
    m, hr, M   : int, float, float
        Modo azimutale, aspect ratio di fallback, massa BH.
 
    check_k, check_beta, check_shear : bool
        Attiva/disattiva i tre constraint fisici standard.
 
    check_ilr : bool  (default False)
        Se True, applica il constraint r < r_ILR: mantiene solo le soluzioni
        dentro la cavity di risonanza AEI tra il bordo interno del disco e
        la Inner Lindblad Resonance.  Queste sono le uniche fisicamente
        capaci di produrre QPO coerenti (standing waves quantizzate).
        r_ILR dipende solo da (a, ν₀, m, M) — viene pre-calcolata una volta
        per ogni valore unico di spin per efficienza.
 
    k_min : float (default 1.0)
        Limite inferiore WKB: k > 1 garantisce che la perturbazione
        sia sub-orbitale (λ < 2πr).
 
    k_max : float | None (default None)
        Limite superiore WKB.
        Se None (default): k_max(r) = 1/hr(r) — dipende dal punto radiale
        e dal modello (corretto fisicamente: λ > 2πH).
        Se fornito come scalare, viene usato come limite fisso per tutti i
        punti (comportamento legacy; utile solo per test o confronti).
 
    beta_max : float
        Soglia per il check β.
 
    dr_frac : float
        Passo relativo per dQ/dr.
 
    ──────────────────────────────────────────────────────────────────
    Restituisce
    ──────────────────────────────────────────────────────────────────
    df : DataFrame
        Una riga per ogni (r, **params) che supera tutti i check attivi.
        Colonne garantite:
          r, a, k, beta, dQdr, k_max
          + tutte le chiavi di param_grid
          + 'zone' (se disk_model la restituisce, altrimenti assente)
    """
    r_vec = np.asarray(r_vec, dtype=float)
 
    # ── prodotto cartesiano dei parametri non-r ──────────────────────────────
    keys   = list(param_grid.keys())
    arrays = [np.asarray(param_grid[k]) for k in keys]
 
    # meshgrid N-D sui parametri non-r
    grids = np.meshgrid(*arrays, indexing='ij')
    flat  = {k: g.ravel() for k, g in zip(keys, grids)}
    N_combos = flat[keys[0]].size
 
    # ── vincolo ISCO (dipende solo da a) ────────────────────────────────────
    a_flat    = flat['a']
    isco_flat = r_isco(a_flat)          # (N_combos,)
 
    # ── pre-calcolo r_ILR per spin unici (O(N_a) chiamate, non O(N_combos)) ──
    # r_ILR dipende solo da (a, ν₀, m, M), indipendente da B00 e Sigma0
    if check_ilr:
        _a_unique  = np.unique(a_flat)
        _ilr_cache = {float(av): r_ilr(float(av), NU0, m, M) for av in _a_unique}
    else:
        _ilr_cache = {}
 
    rows = []
 
    for i in range(N_combos):
        # parametri scalari per questa combinazione
        row_params = {k: float(flat[k][i]) for k in keys}
        a_val      = row_params['a']
        isco_val   = float(isco_flat[i])
 
        # maschera ISCO su r
        r_ok = r_vec >= isco_val
        r_i  = r_vec[r_ok]
        if r_i.size == 0:
            continue
 
        # ── profili fisici (vettorizzati in r) ──────────────────────────────
        result = disk_model(r_i, **row_params)
        n_ret  = len(result)
        B0_i, Sigma_i, cs_i = result[0], result[1], result[2]
        # Nuova firma (6 valori): B0, Sigma, c_s, hr, zone, info
        # Vecchia firma (5 valori): B0, Sigma, c_s, zone, info
        if n_ret >= 6:
            hr_ret = result[3]
            hr_i   = (np.full_like(r_i, float(hr_ret))
                      if np.isscalar(hr_ret) else np.asarray(hr_ret, dtype=float))
            zone_i = result[4]
            info_i = result[5]
        elif n_ret == 5:
            hr_i   = np.full_like(r_i, float(hr))
            zone_i = result[3]
            info_i = result[4]
        else:
            hr_i   = np.full_like(r_i, float(hr))
            zone_i = result[3] if n_ret == 4 else None
            info_i = {}
 
        # ── solver k ────────────────────────────────────────────────────────
        k_i = solve_k_aei(r_i, a_val, B0_i, Sigma_i, cs_i, m=m, M=M)
 
        # ── check fisici ────────────────────────────────────────────────────
        mask = np.isfinite(k_i) & (k_i > 0)
 
        if check_k:
            # k_max fisicamente corretto: 1/hr(r).
            # Se k_max è passato esplicitamente come scalare (uso legacy),
            # si usa quello; altrimenti si calcola da hr_i punto per punto,
            # con lo stesso guard (pavimento = hr di fallback) usato in
            # compute_disk_profile.
            if k_max is not None:
                _k_max = k_max                        # scalare fisso (legacy)
            else:
                hr_floor = max(float(hr), 1e-4)
                hr_eff   = np.maximum(hr_i, hr_floor)
                _k_max   = 1.0 / hr_eff               # array radiale
            mask &= check_k_wkb(k_i, k_min=k_min, k_max=_k_max)
 
        # beta e dQdr calcolati solo dove k è già valido (risparmio CPU)
        beta_i = np.full_like(r_i, np.nan)
        dQdr_i = np.full_like(r_i, np.nan)
 
        if np.any(mask):
            beta_i[mask] = compute_beta(
                B0_i[mask], Sigma_i[mask], cs_i[mask], r_i[mask], hr_i[mask], M
            )
 
            # per dQdr serve B0_func e Sigma_func come callable in r
            # le costruiamo interpolando i profili già calcolati sull'intero r_i
            # (evita di richiamare disk_model due volte)
            _B0_interp    = _make_interp(r_i, B0_i)
            _Sigma_interp = _make_interp(r_i, Sigma_i)
            dQdr_i[mask]  = compute_dQdr(
                r_i[mask], a_val, _B0_interp, _Sigma_interp, M, dr_frac
            )
 
        if check_beta:
            mask &= (beta_i <= beta_max)
        if check_shear:
            mask &= (dQdr_i > 0)
        if check_ilr:
            r_ILR_val = _ilr_cache.get(a_val, np.nan)
            if np.isnan(r_ILR_val):
                mask[:] = False          # nessuna cavity valida per questo spin
            else:
                mask &= (r_i < r_ILR_val)
 
        # ── raccolta risultati ───────────────────────────────────────────────
        if not np.any(mask):
            continue
 
        r_ILR_entry = _ilr_cache.get(a_val, np.nan) if check_ilr else np.nan
 
        # k_max per output (scalare o array a seconda della modalità)
        if k_max is not None:
            _k_max_out = np.full_like(r_i, float(k_max))
        else:
            hr_floor = max(float(hr), 1e-4)
            _k_max_out = 1.0 / np.maximum(hr_i, hr_floor)
 
        idx_ok = np.where(mask)[0]
        for j in idx_ok:
            entry = dict(row_params)
            entry['r']     = r_i[j]
            entry['k']     = k_i[j]
            entry['k_max'] = _k_max_out[j]
            entry['beta']  = beta_i[j]
            entry['dQdr']  = dQdr_i[j]
            entry['m']     = m
            entry['hr']    = float(hr_i[j])
            entry['r_ILR'] = r_ILR_entry
            if zone_i is not None:
                entry['zone'] = zone_i[j]
            rows.append(entry)
 
    return pd.DataFrame(rows)


# ── helper interno: interpolazione lineare log-log per ricostruire profili ──

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

# ═══════════════════════════════════════════════════════════════════════════
# 5.  MASSA TOTALE DEL DISCO
# ═══════════════════════════════════════════════════════════════════════════

def disk_mass(disk_model, params, r_max, M=M_BH, n_points=2000):
    """
    Calcola la massa totale del disco da r_ISCO a r_max tramite integrazione
    numerica di  dM = 2π r_cm Σ(r) dr_cm.

    Utilizza la stessa firma di disk_model usata ovunque nel progetto,
    quindi funziona con qualsiasi modello (SS, NT, Simple).

    Formula:
        M_disk = ∫_{r_ISCO}^{r_max} 2π R Σ(R) dR
    con R = r · R_g (in cm), dr = R_g · d(r).

    Parametri
    ----------
    disk_model : callable
        disk_model(r_rg, **params) → (B0, Sigma, c_s, [hr,] zone, info)
        Stessa firma usata in compute_disk_profile e find_rossby.

    params : dict
        Parametri scalari del disco. Deve contenere almeno 'a'.

    r_max : float
        Raggio esterno dell'integrazione [r_g].
        Può essere None: in tal caso r_max = info['r_BC'] restituito dal modello
        (se disponibile), altrimenti viene sollevato ValueError.

    M : float
        Massa BH [M_sun].

    n_points : int
        Numero di punti per l'integrazione (default 2000, log-spaziati).

    Restituisce
    -----------
    M_disk : float   massa totale del disco [g]
    M_disk_msun : float   massa totale del disco [M_sun]
    meta : dict
        r_ISCO, r_max, M_BH, n_points, mdot (se in params o info)
    """
    a = float(params['a'])
    Rg = Rg_SUN * M            # cm per unità r_g

    rISCO = float(r_isco(a))

    # ── determina r_max ──────────────────────────────────────────────────
    if r_max is None:
        _r_probe = np.array([rISCO])
        _result  = disk_model(_r_probe, **params)
        _info    = _result[5] if len(_result) >= 6 else (_result[4] if len(_result) == 5 else {})
        if 'r_BC' in _info:
            r_max = float(_info['r_BC'])
        else:
            raise ValueError(
                "r_max non specificato e disk_model non restituisce info['r_BC']. "
                "Specificare r_max esplicitamente."
            )

    # ── griglia radiale log-spaziata ──────────────────────────────────────
    r_arr  = np.geomspace(rISCO, r_max, n_points)   # [r_g]

    # ── profili fisici ────────────────────────────────────────────────────
    result    = disk_model(r_arr, **params)
    Sigma_arr = np.asarray(result[1], dtype=float)  # [g/cm²]

    # ── integrazione  ∫ 2π R Σ dR  con R in cm ───────────────────────────
    R_cm  = r_arr * Rg                     # [cm]
    integrand = 2.0 * np.pi * R_cm * Sigma_arr   # [g/cm]
    M_disk_g  = np.trapz(integrand, R_cm)  # [g]

    M_SUN_G   = 1.989e33                   # g
    M_disk_msun = M_disk_g / M_SUN_G

    # ── info da disk_model (se disponibile) ───────────────────────────────
    n_ret = len(result)
    _info = result[5] if n_ret >= 6 else (result[4] if n_ret == 5 else {})

    meta = dict(
        r_ISCO    = rISCO,
        r_max     = r_max,
        M_BH_msun = M,
        n_points  = n_points,
        **{k: _info[k] for k in ('r_AB', 'r_BC', 'mdot', 'alpha') if k in _info},
    )
    if 'mdot' in params:
        meta.setdefault('mdot', params['mdot'])

    return M_disk_g, M_disk_msun, meta