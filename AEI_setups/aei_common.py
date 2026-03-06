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
  - find_rossby        : finder vettorizzato, funziona con qualsiasi modello di disco

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

import sys
sys.path.append("..")
from setup import (
    r_isco, nu_phi, nu_r,
    Rg_SUN, M_BH, NU0,
)

HOR = 0.05
mm = 1
ALPHA_VISC = 0.01

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
    A  = c_s**2   / r_cm**2
    B  = 2*B0**2  / (Sigma * r_cm)
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

def check_k_wkb(k, k_min=0.1, k_max=10.0):
    """
    Vincolo WKB sul numero d'onda adimensionale k = k_fis · r_cm.

    Il limite WKB richiede  k_fis ≫ 1/r,  cioè  k ≫ 1.
    In pratica si usa il range conservativo  [0.1, 10]

    Parametri
    ----------
    k     : array_like   numero d'onda adimensionale
    k_min : float        limite inferiore (default 0.1)
    k_max : float        limite superiore (default 10.0)

    Restituisce
    -----------
    mask : ndarray bool
    """
    k = np.asarray(k)
    return np.isfinite(k) & (k >= k_min) & (k <= k_max)


def compute_beta(B0, Sigma, c_s, r_rg, hr=HOR, M=M_BH):
    """
    Parametro plasma beta:

        β = 8π Σ c_s² / (H B₀²)    con H = hr · r · Rg

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
    return 8*np.pi * np.asarray(Sigma) * np.asarray(c_s)**2 / (H * np.asarray(B0)**2)


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
# 3.  FINDER VETTORIZZATO
# ═══════════════════════════════════════════════════════════════════════════

def find_rossby(
    r_vec,
    param_grid,
    disk_model,
    m=mm,
    hr=HOR,
    M=M_BH,
    check_k=True,
    check_beta=True,
    check_shear=True,
    k_min=0.1,
    k_max=10.0,
    beta_max=1.0,
    dr_frac=0.01,
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
        Modo azimutale, aspect ratio, massa BH.

    check_k, check_beta, check_shear : bool
        Attiva/disattiva i check fisici.

    k_min, k_max : float
        Range WKB per k adimensionale.

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
          r, a, k, beta, dQdr
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
        if len(result) == 3:
            B0_i, Sigma_i, cs_i = result
            zone_i = None
        else:
            B0_i, Sigma_i, cs_i, zone_i = result   # modello SS/NT può restituire zona

        # ── solver k ────────────────────────────────────────────────────────
        k_i = solve_k_aei(r_i, a_val, B0_i, Sigma_i, cs_i, m=m, M=M)

        # ── check fisici ────────────────────────────────────────────────────
        mask = np.isfinite(k_i) & (k_i > 0)

        if check_k:
            mask &= check_k_wkb(k_i, k_min, k_max)

        # beta e dQdr calcolati solo dove k è già valido (risparmio CPU)
        beta_i = np.full_like(r_i, np.nan)
        dQdr_i = np.full_like(r_i, np.nan)

        if np.any(mask):
            beta_i[mask] = compute_beta(
                B0_i[mask], Sigma_i[mask], cs_i[mask], r_i[mask], hr, M
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

        # ── raccolta risultati ───────────────────────────────────────────────
        if not np.any(mask):
            continue

        idx_ok = np.where(mask)[0]
        for j in idx_ok:
            entry = dict(row_params)
            entry['r']     = r_i[j]
            entry['k']     = k_i[j]
            entry['beta']  = beta_i[j]
            entry['dQdr']  = dQdr_i[j]
            entry['m']     = m
            entry['hr']    = hr
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
# 4.  ADAPTER UFFICIALI PER OGNI MODELLO
#     (da copiare nelle celle di setup del notebook o nei file modello)
# ═══════════════════════════════════════════════════════════════════════════

# ── 4a. simple_disc (power-law omogeneo) ────────────────────────────────────
#
# Nel notebook / simple_disc.py:
#
#   from aei_common import find_rossby
#   from simple_disc import B0_profile, Sigma_profile, sound_speed_thin
#
#   def disk_model_simple(r_rg, a, B00, Sigma0, alpha_B=alp_B, alpha_S=alp_S,
#                          hr=hor, M=M_BH):
#       B0    = B0_profile(r_rg, a, B00, alpha_B)
#       Sigma = Sigma_profile(r_rg, a, Sigma0, alpha_S)
#       c_s   = sound_speed_thin(r_rg, a, hr, M)
#       return B0, Sigma, c_s
#
#   param_grid = {
#       'a':      np.linspace(-0.9, 0.9, 19),
#       'B00':    np.logspace(-3, 8, 24),
#       'Sigma0': np.logspace(3, 7, 20),
#   }
#   r_vec = np.geomspace(1, 1000, 100)
#
#   df = find_rossby(
#       r_vec, param_grid,
#       disk_model=lambda r, **p: disk_model_simple(r, **p, hr=0.05, M=M_BH),
#       m=1, hr=0.05, M=M_BH,
#       check_k=True, check_beta=True, check_shear=True,
#   )


# ── 4b. full_disk_SS (tre zone Shakura-Sunyaev) ─────────────────────────────
#
# Nel notebook / full_disk_SS.py:
#
#   from aei_common import find_rossby
#   from full_disk_SS import (
#       ss_boundaries, compute_norms,
#       B0_disk, Sigma_disk, sound_speed_disk,
#       zone_index, ZONE_NAMES,
#   )
#
#   def disk_model_SS(r_rg, a, B00, Sigma0, alpha_visc=0.1, M=M_BH, hr=0.05):
#       r_AB, r_BC, _ = ss_boundaries(a, Sigma0, alpha=alpha_visc, M=M)
#       norms         = compute_norms(a, B00, Sigma0, r_AB, r_BC)
#       B0    = B0_disk(r_rg, norms, r_AB, r_BC)
#       Sigma = Sigma_disk(r_rg, norms, r_AB, r_BC)
#       c_s   = sound_speed_disk(r_rg, a, hr, M)
#       zi    = zone_index(r_rg, r_AB, r_BC)
#       zone  = np.array([ZONE_NAMES[i] for i in zi])
#       return B0, Sigma, c_s, zone
#
#   param_grid = {
#       'a':      np.linspace(-0.9, 0.9, 19),
#       'B00':    np.logspace(-3, 8, 24),
#       'Sigma0': np.logspace(3, 7, 20),
#   }
#   r_vec = np.geomspace(2, 1000, 150)
#
#   df = find_rossby(
#       r_vec, param_grid,
#       disk_model=lambda r, **p: disk_model_SS(r, **p, alpha_visc=0.1, M=M_BH, hr=0.05),
#       m=1, hr=0.05, M=M_BH,
#       check_k=True, check_beta=True, check_shear=True,
#   )


# ── 4c. Novikov-Thorne (profili NT diretti) ──────────────────────────────────
#
# Se si hanno direttamente i profili B0(r), Sigma(r), c_s(r) calcolati con
# ss_nt_boundaries, basta un adapter minimale:
#
#   from aei_common import find_rossby
#   from ss_nt_boundaries import nt_profiles  # funzione che restituisce profili NT
#
#   def disk_model_NT(r_rg, a, mdot, alpha_visc=0.1, M=M_BH, hr=0.05):
#       B0, Sigma, c_s = nt_profiles(r_rg, a, mdot, alpha=alpha_visc, M=M)
#       return B0, Sigma, c_s
#
#   param_grid = {
#       'a':    np.linspace(-0.9, 0.9, 19),
#       'mdot': np.logspace(-2, 0, 15),
#   }
#   r_vec = np.geomspace(2, 500, 150)
#
#   df = find_rossby(
#       r_vec, param_grid,
#       disk_model=lambda r, **p: disk_model_NT(r, **p, alpha_visc=0.1, M=M_BH, hr=0.05),
#       m=1, hr=0.05, M=M_BH,
#   )
