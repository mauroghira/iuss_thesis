"""
nt_disc.py
==========
Modello di disco Novikov-Thorne (1973) per l'analisi AEI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUTTURA RADIALE — tre zone con fattori relativistici
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Zona A  [r_ISCO, r_AB]:  P_rad >> P_gas,  opacità e-scattering
  Zona B  [r_AB,   r_BC]:  P_gas >> P_rad,  opacità e-scattering
  Zona C  [r_BC,   ∞   ):  P_gas >> P_rad,  opacità free-free

  Frontiere (Abramowicz review, eqs. 100-101):
    r_AB: β/(1−β) = 1  →  P_rad = P_gas
          f_AB(r) = 4×10⁻⁶ · α⁻¹/⁴ · m⁻¹/⁴ · ṁ⁻² · r^{21/8} · [NT_AB] = 1
    r_BC: τ_ff/τ_es = 1  →  cambio regime opacità
          f_BC(r) = 2×10⁻⁶ · ṁ⁻¹ · r^{3/2} · [NT_BC] = 1

  Zone degeneri:
    Se la condizione r_AB > r_ISCO non è soddisfatta → zona A assente.
    Se r_BC ≤ r_AB  → zona B assente.
    In entrambi i casi le formule sono usate pure, senza fattori di raccordo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FATTORI RELATIVISTICI NT  A, B, C, D, E, Q
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Con  y = r^{1/2}  (coordinata radiale compatta):

    A(r) = 1 + a²/r² + 2a²/r³
    B(r) = 1 + a/r^{3/2}
    C(r) = 1 − 3/r + 2a/r^{3/2}       ≥ 0 per r ≥ r_ISCO
    D(r) = 1 − 2/r + a²/r²
    E(r) = 1 + 4a²/r² − 4a²/r³ + 3a⁴/r⁴

    Q(r) = fattore di flusso NT, calcolato con integral analytic di Page & Thorne:
           Q = B/(y sqrt(C)) · [y−y₀ − (3a/2) ln(y/y₀) − Σᵢ cᵢ ln((y−yᵢ)/(y₀−yᵢ))]
           Q(r_ISCO) = 0  (zero-torque boundary condition)

  Nel limite r → ∞: tutti i fattori → 1 (recupero newtoniano).
  Nel limite a → 0: recupero del caso Schwarzschild.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DENSITÀ SUPERFICIALE Σ(r)  —  formule NT (Abramowicz review eqs. 97-99)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Zona A: Σ_A = 5 · α⁻¹ · ṁ⁻¹ · r^{3/2} · A⁻²B³C^{1/2}E Q⁻¹              (99)
  Zona B: Σ_B = 9×10⁴ · α⁻⁴/⁵ · ṁ^{3/5} · r^{-3/5} · B⁻⁴/⁵C^{1/2}D⁻⁴/⁵Q^{3/5} (98)
  Zona C: Σ_C = 4×10⁵ · α⁻⁴/⁵ · m^{1/5} · ṁ^{7/10} · r^{-3/4}
                · A^{1/10}B⁻⁴/⁵C^{1/2}D⁻¹⁷/²⁰E⁻¹/²⁰Q^{7/10}              (97)

  Nota: Σ_A → ∞ per r → r_ISCO (Q → 0). Questo è fisicamente corretto:
  vicino all'ISCO il flusso NT tende a zero (zero-torque) e il gas si accumula.
  L'inversione mdot ↔ Σ_A usa il minimo di Σ_A nella zona A (non il valore
  all'ISCO), come descritto in nt_mdot_from_Sigma0().

  NON si applicano fattori di raccordo o scala.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMISPESSORE H(r) — da SS 1973 + fattori NT di Kerr per la frequenza kepleriana
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Le formule per H nelle zone B e C vengono da SS 1973 ma con la frequenza
  kepleriana relativistica Ω_K^{NT} = c/R_g · B(r)/r^{3/2} invece di
  Ω_K^{SS} = c/R_g · r^{-3/2}. Per la zona A, H_A è derivata dalla
  struttura termica NT con pressione di radiazione dominante.

  In pratica si usa la stessa espressione di SS con fattori NT inclusi nei
  prefattori fisici delle equazioni di struttura (vedi Abramowicz review).
  Per semplicità e consistenza con Σ_NT, usiamo:

    H(r) = c_s^{NT}(r) / Ω_K^{NT}(r)

  dove c_s^{NT}(r) = sqrt(P_mid/ρ_mid) e ρ_mid = Σ/(2H) → H = sqrt(P_mid/ρ_mid) / Ω_K
  → P_mid = Σ · Ω_K^{NT,2} · H / 2  (analogo al caso SS ma con Ω_K di Kerr)

  Per il calcolo di B si usa direttamente questa relazione:
      P_mid = Σ_NT(r) · Ω_φ^{NT}(r)² · H_SS(r) / 2

  dove Ω_φ = 2π ν_φ(r, a, M) è la frequenza orbitale di Kerr e H_SS è il
  semispessore SS corretto per i fattori relativistici. Per le zone B e C
  si usa H_SS(r) moltiplicato per B(r)·C(r)^{-1/2} (fattore NT verticale).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAMPO MAGNETICO B₀(r)  —  equipartizione dalla P_mid fisica NT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Dall'equilibrio idrostatico verticale in metrica di Kerr:

      P_mid = Σ(r) · Ω_φ^{NT}(r)² · H_NT(r) / 2

  dove H_NT è il semispessore fisico NT derivato dalle equazioni di struttura.
  Condizione di equipartizione:

      B_eq(r) = sqrt(4π · Σ(r) · Ω_φ^{NT}(r)² · H_NT(r))

  La differenza rispetto al modello SS è nell'uso di Ω_φ di Kerr (non
  newtoniana) e di H_NT che incorpora i fattori relativistici A,B,C,D.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETRI LIBERI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  a      — spin adimensionale del BH              [−1, 1]
  mdot   — tasso di accrescimento ṁ = Ṁc^2/L_Edd  [> 0]   ← PRIMARIO
            OSS: è diverso rispeto a SS, fattore 13.6 più grande, devi 
            convertire quando lo usi
  alpha  — parametro di viscosità α               [adim]
  M      — massa BH                               [M_sun]
  hr     — aspect ratio per c_s AEI               [adim] (non entra in Σ né B)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API PUBBLICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  nt_factors(r, a)                               → dict {A,B,C,D,E,Q0,Q}
  nt_boundaries(a, mdot, alpha, M)               → r_AB, r_BC, zone_present
  disk_model_NT(r_rg, a, mdot, ...)              → B0, Sigma, c_s, zone, info
  disk_inner_values_NT(a, mdot, ...)             → dict {r_ISCO, r_H, Sigma_ISCO, ...}
  check_continuity_NT(a, mdot, ...)              → dict + stampa diagnostica

  Funzioni di profilo per zona (accesso diretto):
  Sigma_A_NT(r, a, mdot, alpha)
  Sigma_B_NT(r, a, mdot, alpha)
  Sigma_C_NT(r, a, mdot, alpha, M)
  H_NT(r, a, mdot, alpha, M, r_AB, r_BC)

Dipendenze: numpy, functools, setup.py, aei_common.py
"""

import numpy as np
from functools import lru_cache

from .aei_common import ALPHA_VISC, HOR

import sys
sys.path.append("..")
from setup import r_isco, r_horizon, nu_phi, Rg_SUN, M_BH, SigTOM

ZONE_NAMES = ['A', 'B', 'C']


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  COSTANTI DI SPIN  (cache — calcolate una volta per valore di a)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=128)
def _nt_spin_constants(a):
    """
    Costanti y₀, y₁, y₂, y₃, c₁, c₂, c₃ per il fattore Q di NT.

    y_i sono le tre radici reali di  y³ − 3y + 2a = 0  (orbite circolari),
    con  y = r^{1/2}  e  y₀ = sqrt(r_ISCO).

    c_i sono i coefficienti dei logaritmi nell'integrale analitico del
    flusso di energia NT (Page & Thorne 1974, eq. 15n):
        c_i = 3(y_i − a)² / [y_i · ∏_{j≠i}(y_i − y_j)]

    Parametro: a float (hashable) — spin adimensionale
    """
    as_   = float(a)
    rISCO = float(r_isco(as_))
    y0    = np.sqrt(rISCO)

    _acos = np.arccos(np.clip(as_, -1 + 1e-10, 1 - 1e-10))
    y1    =  2 * np.cos((_acos - np.pi) / 3)
    y2    =  2 * np.cos((_acos + np.pi) / 3)
    y3    = -2 * np.cos( _acos          / 3)

    c1 = 3*(y1 - as_)**2 / (y1*(y1-y2)*(y1-y3))
    c2 = 3*(y2 - as_)**2 / (y2*(y2-y1)*(y2-y3))
    c3 = 3*(y3 - as_)**2 / (y3*(y3-y1)*(y3-y2))

    return y0, y1, y2, y3, c1, c2, c3


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FATTORI RELATIVISTICI NT  A, B, C, D, E, Q
# ═══════════════════════════════════════════════════════════════════════════════

def nt_factors(r, a):
    """
    Fattori relativistici di Novikov-Thorne: A, B, C, D, E, Q₀, Q.

    Tutti adimensionali, tendono a 1 per r → ∞.

    Definizioni (con y = r^{1/2}):
        A(r) = 1 + a²/r²  + 2a²/r³
        B(r) = 1 + a/r^{3/2}
        C(r) = 1 − 3/r   + 2a/r^{3/2}    (= 0 all'ISCO, ≥ 0 fuori)
        D(r) = 1 − 2/r   + a²/r²
        E(r) = 1 + 4a²/r² − 4a²/r³ + 3a⁴/r⁴

        Q₀(r) = B(r) / (y · sqrt(C(r)))

        Q(r)  = Q₀(r) · [y−y₀ − (3a/2) ln(y/y₀)
                          − c₁ ln((y−y₁)/(y₀−y₁))
                          − c₂ ln((y−y₂)/(y₀−y₂))
                          − c₃ ln((y−y₃)/(y₀−y₃))]

    Q è il fattore di flusso NT: zero all'ISCO (zero-torque), positivo fuori.
    Tutti i logaritmi sono protetti da valori assoluti per stabilità numerica
    vicino alle radici y_i (che stanno tipicamente dentro r_ISCO).

    Parametri
    ----------
    r : array_like   raggio [r_g]
    a : float        spin adimensionale

    Restituisce
    -----------
    dict con chiavi 'A', 'B', 'C', 'D', 'E', 'Q0', 'Q'
    """
    r   = np.asarray(r, dtype=float)
    as_ = float(a)
    y   = np.sqrt(r)

    A  = 1 + as_**2 * r**(-2) + 2*as_**2 * r**(-3)
    B  = 1 + as_ * r**(-1.5)
    C  = 1 - 3*r**(-1) + 2*as_ * r**(-1.5)
    D  = 1 - 2*r**(-1) + as_**2 * r**(-2)
    E  = 1 + 4*as_**2*r**(-2) - 4*as_**2*r**(-3) + 3*as_**4*r**(-4)

    C_safe = np.maximum(C, 1e-30)
    Q0 = B / (y * np.sqrt(C_safe))

    y0, y1, y2, y3, c1, c2, c3 = _nt_spin_constants(as_)
    sl = lambda x: np.log(np.maximum(np.abs(x), 1e-300))

    term0 = y - y0 - 1.5*as_*sl(y / y0)
    Q_raw = Q0 * (term0
                  - c1*sl((y-y1)/(y0-y1))
                  - c2*sl((y-y2)/(y0-y2))
                  - c3*sl((y-y3)/(y0-y3)))
    Q = np.where(y > y0*1.001, np.maximum(Q_raw, 1e-10), 1e-10)

    return dict(A=A, B=B, C=C, D=D, E=E, Q0=Q0, Q=Q)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FRONTIERE  r_AB  e  r_BC
# ═══════════════════════════════════════════════════════════════════════════════

def _bisect_vec(func, r_lo, r_hi, n_scan=400, n_bisect=60):
    """Bisezione vettorizzata: trova r dove func(r) = 1."""
    r_arr = np.geomspace(r_lo, r_hi, n_scan)
    f_arr = func(r_arr) - 1.0
    sc    = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sc) == 0:
        return None
    a_r, b_r = r_arr[sc[0]], r_arr[sc[0]+1]
    for _ in range(n_bisect):
        mid = (a_r + b_r) / 2
        if func(np.array([mid]))[0] - 1.0 < 0:
            a_r = mid
        else:
            b_r = mid
    return float((a_r + b_r) / 2)


def nt_boundaries(a, mdot, alpha=ALPHA_VISC, M=M_BH):
    """
    Calcola le frontiere radiali r_AB, r_BC e le zone presenti.

    Frontiere (Abramowicz review, eqs. 100-101):

      r_AB: P_rad = P_gas  →  β/(1−β) = 1
            f_AB(r) = 4×10⁻⁶ · α⁻¹/⁴ · m⁻¹/⁴ · ṁ⁻² · r^{21/8}
                      · A(r)^{-5/2} · B(r)^{9/2} · D(r) · E(r)^{5/4} / Q(r)² = 1

      r_BC: τ_ff/τ_es = 1  →  cambio regime opacità
            f_BC(r) = 2×10⁻⁶ · ṁ⁻¹ · r^{3/2}
                      · A(r)^{-1} · B(r)² · sqrt(D(r)) · sqrt(E(r)) / Q(r) = 1

    I limiti superiori per la bisezione sono stimati nel limite r >> r_ISCO
    (fattori NT → 1):
      r_AB_est ~ [α^{1/4} m^{1/4} ṁ² / 4×10⁻⁶]^{8/21}
      r_BC_est ~ [ṁ / 2×10⁻⁶]^{2/3}

    Zone degeneri:
      Se f_AB(r_ISCO) ≥ 1 → zona A assente (r_AB = r_ISCO)
      Se f_BC(r_AB) ≥ 1   → zona B assente (r_BC = r_AB)

    Parametri
    ----------
    a     : float   spin adimensionale
    mdot  : float   ṁ = Ṁ/Ṁ_Edd > 0
    alpha : float   parametro di viscosità α
    M     : float   massa BH [M_sun]

    Restituisce
    -----------
    r_AB        : float   frontiera A-B [r_g]
    r_BC        : float   frontiera B-C [r_g]
    zone_present: dict    {'A': bool, 'B': bool, 'C': bool}
    """
    rISCO = float(r_isco(a))
    m     = float(M)

    # ── r_AB ────────────────────────────────────────────────────────────────
    def f_AB(r_arr):
        f   = nt_factors(r_arr, a)
        rel = (np.maximum(f['A'], 1e-30)**(-5.0/2) * f['B']**(9.0/2)
               * np.maximum(f['D'], 1e-30) * np.maximum(f['E'], 1e-30)**(5.0/4)
               / f['Q']**2)
        return 4e-6 * alpha**(-0.25) * m**(-0.25) * mdot**(-2) \
               * np.asarray(r_arr)**(21.0/8) * rel

    _r_scan_AB = np.geomspace(rISCO * 1.05, rISCO * 200, 300)
    _f_scan_AB = f_AB(_r_scan_AB)
    _imin_AB   = int(np.argmin(_f_scan_AB))
    _r_min_AB  = float(_r_scan_AB[_imin_AB])
    _fmin_AB   = float(_f_scan_AB[_imin_AB])
 
    if _fmin_AB >= 1.0:
        r_AB   = rISCO
        zone_A = False
    else:
        r_AB_est = (alpha**(0.25) * m**(0.25) * mdot**2 / 4e-6)**(8.0/21)
        r_AB_hi  = float(max(r_AB_est * 3.0, _r_min_AB * 10.0))
        r_AB = _bisect_vec(f_AB, _r_min_AB, r_AB_hi)
        if r_AB is None:
            r_AB = r_AB_hi
        r_AB   = float(max(r_AB, rISCO))
        zone_A = True

    # ── r_BC ────────────────────────────────────────────────────────────────
    def f_BC(r_arr):
        f   = nt_factors(r_arr, a)
        rel = (np.maximum(f['A'], 1e-30)**(-1)
               * f['B']**2
               * np.sqrt(np.maximum(f['D'], 1e-30))
               * np.sqrt(np.maximum(f['E'], 1e-30))
               / f['Q'])
        return 2e-6 * mdot**(-1) * np.asarray(r_arr)**(1.5) * rel

    _r_start_BC  = max(r_AB, rISCO) * 1.05
    _r_scan_BC   = np.geomspace(_r_start_BC, _r_start_BC * 200, 300)
    _f_scan_BC   = f_BC(_r_scan_BC)
    _imin_BC     = int(np.argmin(_f_scan_BC))
    _r_min_BC    = float(_r_scan_BC[_imin_BC])
    _fmin_BC     = float(_f_scan_BC[_imin_BC])
 
    if _fmin_BC >= 1.0:
        r_BC   = r_AB
        zone_B = False
    else:
        r_BC_est = (mdot / 2e-6)**(2.0/3)
        r_BC_hi  = float(max(r_BC_est * 3.0, _r_min_BC * 10.0))
        r_BC = _bisect_vec(f_BC, _r_min_BC, r_BC_hi)
        if r_BC is None:
            r_BC = r_BC_hi
        r_BC   = float(max(r_BC, r_AB))
        zone_B = True

    return r_AB, r_BC, {'A': zone_A, 'B': zone_B, 'C': True}


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DENSITÀ SUPERFICIALE Σ(r)  —  formule NT raw
# ═══════════════════════════════════════════════════════════════════════════════

def Sigma_A_NT(r, a, mdot, alpha):
    """
    Σ zona A  [g/cm²]  —  NT eq. 99 (Abramowicz review)

        Σ_A = 5 · α⁻¹ · ṁ⁻¹ · r^{3/2} · A⁻² B³ C^{1/2} E Q⁻¹

    P_rad dominante, opacità e-scattering. I fattori NT A,B,C,E,Q modificano
    l'andamento radiale rispetto alla zona A SS (prefattore 4.6 vs 5, stessa
    dipendenza da α e ṁ).

    NOTA: Σ_A → ∞ per r → r_ISCO perché Q → 0 (zero-torque NT).
    """
    r   = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['A']**(-2) * f['B']**3
           * np.sqrt(np.maximum(f['C'], 0))
           * f['E'] / np.maximum(f['Q'], 1e-10))
    return 5.0 * alpha**(-1.0) * mdot**(-1.0) * r**(1.5) * rel


def Sigma_B_NT(r, a, mdot, alpha, m=M_BH):
    """
    Σ zona B  [g/cm²]  —  NT eq. 98 (Abramowicz review)

    P_gas dominante, opacità e-scattering. Confronto con SS zona B:
    prefattore NT 9×10⁴ vs SS 1.7×10⁵ (fattori NT riducono l'ampiezza).
    """
    r   = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['B']**(-4.0/5)
           * np.sqrt(np.maximum(f['C'], 0))
           * np.maximum(f['D'], 1e-30)**(-4.0/5)
           * f['Q']**(3.0/5))
    return 9e4 * alpha**(-4.0/5) * mdot**(3.0/5) * m**(1/5) * r**(-3.0/5) * rel


def Sigma_C_NT(r, a, mdot, alpha, M=M_BH):
    """
    Σ zona C  [g/cm²]  —  NT eq. 97 (Abramowicz review)

        Σ_C = 4×10⁵ · α⁻⁴/⁵ · m^{1/5} · ṁ^{7/10} · r^{-3/4}
              · A^{1/10} B^{-4/5} C^{1/2} D^{-17/20} E^{-1/20} Q^{7/10}

    P_gas dominante, opacità free-free. L'esponente m^{1/5} (diverso da m^{1/2}
    del modello SS) emerge dalla diversa combinazione di fattori relativistici.
    """
    r   = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['A']**(1.0/10) * f['B']**(-4.0/5)
           * np.sqrt(np.maximum(f['C'], 0))
           * np.maximum(f['D'], 1e-30)**(-17.0/20)
           * np.maximum(f['E'], 1e-30)**(-1.0/20)
           * f['Q']**(7.0/10))
    return 4e5 * alpha**(-4.0/5) * float(M)**(1.0/5) * mdot**(7.0/10) \
           * r**(-3/4) * rel



# ═══════════════════════════════════════════════════════════════════════════════
# 3b.  RACCORDO REGIONE DI PLUNGE  r_ISCO < r < r_match
# ═══════════════════════════════════════════════════════════════════════════════

def _r_match(a, r_scan_profiles=None, Q_threshold=0.1):
    """
    r_scan_profiles : tuple (r, Sigma, B0, Omega) già calcolati
                      su una griglia fine vicino all'ISCO.
                      Se None: fallback su Q_NT = Q_threshold.
    """
    rISCO = float(r_isco(a))

    if r_scan_profiles is not None:
        r_s, S_s, B0_s, Om_s = r_scan_profiles
        Q_aei = Om_s * S_s / np.maximum(B0_s**2, 1e-300)
        i_min = int(np.argmin(Q_aei))
        return float(r_s[i_min])
    else:
        r_scan = np.geomspace(rISCO * 1.001, rISCO * 6, 2000)
        Q_scan = nt_factors(r_scan, a)['Q']
        idx    = np.searchsorted(Q_scan, Q_threshold)
        if idx >= len(r_scan):
            return rISCO * 1.7
        return float(r_scan[idx])


def _hermite_interp(t, f0, f1, d0, d1):
    """
    Cubic Hermite interpolant on t in [0, 1].
 
    Boundary conditions
    -------------------
    f(0) = f0,  f(1) = f1
    f'(0) = d0, f'(1) = d1   (derivatives w.r.t. t)
 
    The four Hermite basis polynomials:
      h00 = 2t³ − 3t² + 1
      h10 = t³  − 2t² + t
      h01 = −2t³ + 3t²
      h11 = t³  − t²
    """
    t2 = t * t
    t3 = t2 * t
    h00 =  2*t3 - 3*t2 + 1
    h10 =    t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 =    t3 -   t2
    return h00*f0 + h10*d0 + h01*f1 + h11*d1
 
 
def _numerical_deriv(r, f, i_ref, n_pts=4):
    """
    Estimate df/dr at index i_ref using a one-sided finite-difference
    stencil on log(f) vs r (power-law assumption), falling back to linear
    finite differences if fewer than n_pts points are available.
 
    Returns the derivative in *linear* space: df/dr.
    """
    # use the n_pts points immediately to the right of i_ref (physical side)
    i0  = i_ref
    i1  = min(i_ref + n_pts, len(r) - 1)
    sub_r = r[i0:i1 + 1]
    sub_f = f[i0:i1 + 1]
 
    if len(sub_r) < 2:
        return 0.0
 
    pos = sub_f > 0
    if pos.sum() >= 2:
        # fit in log–log space: log f = slope * log r + const
        log_r = np.log(sub_r[pos])
        log_f = np.log(sub_f[pos])
        if log_r.max() - log_r.min() > 0:
            slope = np.polyfit(log_r, log_f, 1)[0]   # power-law index
            # df/dr = slope * f / r  evaluated at r_match
            return slope * sub_f[0] / sub_r[0]
 
    # fallback: simple linear finite difference
    return (sub_f[-1] - sub_f[0]) / (sub_r[-1] - sub_r[0])
 
 
def _apply_plunge_raccordo(r, a, Sigma, H, B,
                            mdot=None, alpha=None,
                            r_AB=None, zone_present=None,
                            r_scan_profiles=None,
                            Q_threshold=0.1):
    r     = np.asarray(r, float)
    rISCO = float(r_isco(a))
    rm    = _r_match(a, r_scan_profiles=r_scan_profiles,
                     Q_threshold=Q_threshold)

    plunge  = r < rm
    if not np.any(plunge):
        return Sigma.copy(), H.copy(), B.copy(), rm

    outside = r >= rm
    i_ref   = int(np.argmax(outside)) if np.any(outside) else len(r) - 1

    Sigma_out = Sigma.copy()
    H_out     = H.copy()
    B_out     = B.copy()

    delta_r  = rm - rISCO
    t        = np.clip((r[plunge] - rISCO) / delta_r, 0.0, 1.0)
    t2, t3   = t**2, t**3

    for qty_arr, out_arr in [(Sigma, Sigma_out), (H, H_out)]:
        f_m   = float(qty_arr[i_ref])
        df_dr = _numerical_deriv(r, qty_arr, i_ref, n_pts=4)

        # df/dt at t=1 (derivative w.r.t. t, not r)
        d_match = df_dr * delta_r

        # Hermite basis with f(0)=0, f'(0)=0, f(1)=f_m, f'(1)=d_match
        # h01 = 3t² - 2t³  (rises from 0 to 1 with zero slope at both ends)
        # h11 = t³ - t²    (derivative shape)
        vals = f_m * (3*t2 - 2*t3) + d_match * (t3 - t2)
        vals = np.clip(vals, 0.0, None)

        out_arr[plunge] = vals

    # B: linear extrapolation from r_match inward (C¹ at r_match)
    B_m   = float(B[i_ref])
    dB_dr = _numerical_deriv(r, B, i_ref, n_pts=4)

    B_linear             = B_m + dB_dr * (r[plunge] - rm)
    B_out[plunge]        = np.clip(B_linear, 0.0, None)

    return Sigma_out, H_out, B_out, rm


def _apply_plunge_raccordo_old(r, a, Sigma, H, B,
                            mdot=None, alpha=None,
                            r_AB=None, zone_present=None,
                            r_scan_profiles=None,
                            Q_threshold=0.1):
    """
    C¹-continuous plunge raccordo for r_ISCO < r < r_match.
 
    Strategy
    --------
    For each quantity X in {Sigma, H}:
 
    1.  At r_match: read the physical value X_m and estimate the derivative
        dX/dr|_match from the NT profile (log-space power-law fit on the
        4 nearest points on the physical side).
 
    2.  Extrapolate linearly inward to find the value at r_ISCO:
            X_ISCO = X_m + (dX/dr)|_match * (r_ISCO - r_match)
        This is the value the linear tangent reaches at the inner boundary.
 
    3.  Build a **cubic Hermite** spline on [r_ISCO, r_match] with
        boundary conditions:
            X(r_match) = X_m,        X'(r_match) = dX/dr|_match
            X(r_ISCO)  = X_ISCO,     X'(r_ISCO)  = dX/dr|_match
        The identical derivative at both ends makes the plunge region
        a smooth, monotone tangent-continuation — the curve enters r_ISCO
        with the same slope it had at r_match (i.e. a "linear-like" trend
        that curves gently to avoid overshoot).
 
    4.  Clip X to be non-negative everywhere (numerical safety).
 
    5.  B is recomputed from equipartition rather than interpolated:
            B(r) = sqrt(4π · Sigma(r) · Omega_phi(r)² · H(r))
        This guarantees B is always physically consistent with the
        raccorded Sigma and H.
 
    Parameters
    ----------
    r, a, Sigma, H, B : as in the original function
    mdot, alpha, r_AB, zone_present, Q_threshold : forwarded to _r_match
 
    Returns
    -------
    Sigma_out, H_out, B_out : ndarray  (same shape as input)
    rm                      : float    r_match [r_g]
    """
    r     = np.asarray(r, float)
    rISCO = float(r_isco(a))
    rm    = _r_match(a, r_scan_profiles=r_scan_profiles, Q_threshold=Q_threshold)
 
    plunge  = r < rm
    if not np.any(plunge):
        return Sigma.copy(), H.copy(), B.copy(), rm

    outside = r >= rm
    # ── reference index: first point on the physical (outside) side ─────────
    i_ref = int(np.argmax(outside)) if np.any(outside) else len(r) - 1
 
    Sigma_out = Sigma.copy()
    H_out     = H.copy()
    B_out     = B.copy()
    
    #""" v1
    Sigma_m = float(Sigma[i_ref])
    H_m     = float(H[i_ref])
    B_m     = float(B[i_ref])

    # peso sqrt: 0 all'ISCO, 1 a r_match
    xi = np.clip((r - rISCO) / (rm - rISCO), 0.0, 1.0)
    w  = np.sqrt(xi)

    Sigma_out          = Sigma.copy()
    H_out              = H.copy()
    B_out              = B.copy()
    Sigma_out[plunge]  = Sigma_m * w[plunge] #se decommento va a 0
    H_out[plunge]      = H_m     * w[plunge]
    B_out[plunge]      = B_m     # plateau

    """#version 2
    r_plunge = r[plunge]
 
    # ── parametric coordinate t ∈ [0, 1]:  t=0 → r_ISCO,  t=1 → r_match ──
    delta_r = rm - rISCO
    t       = np.clip((r_plunge - rISCO) / delta_r, 0.0, 1.0)
 
    for qty_arr, out_arr in [(Sigma, Sigma_out), (H, H_out)]:
 
        # value and derivative at r_match (physical side)
        f_m   = float(qty_arr[i_ref])
        df_dr = _numerical_deriv(r, qty_arr, i_ref, n_pts=4)
 
        # derivative in t-space: df/dt = df/dr * delta_r
        d_match = df_dr * delta_r   # df/dt at t = 1
 
        # linear extrapolation to r_ISCO → value at t = 0
        f_isco = f_m + df_dr * (rISCO - rm)   # = f_m - df_dr * delta_r
        f_isco = max(f_isco, 0.0)              # physical floor
 
        # derivative at r_ISCO: keep the same slope (linear tangent)
        d_isco = d_match                       # df/dt at t = 0
 
        # cubic Hermite: t=0 is r_ISCO, t=1 is r_match
        vals = _hermite_interp(t, f_isco, f_m, d_isco, d_match)
        vals = np.clip(vals, 0.0, None)        # non-negative safety
 
        out_arr[plunge] = vals
 
    # ── recompute B from equipartition with the raccorded Sigma and H ───────
    Omega_phi_sq = (2.0 * np.pi * nu_phi(r[plunge], a, M_BH)) ** 2
    B_recomp     = np.sqrt(np.maximum(
        4.0 * np.pi * Sigma_out[plunge] * Omega_phi_sq * H_out[plunge], 0.0
    ))
    B_out[plunge] = B_recomp
    #"""
 
    return Sigma_out, H_out, B_out, rm



def _Sigma_disk(r, a, mdot, alpha, M, r_AB, r_BC):
    """
    Σ(r) sull'intero disco — formule NT raw, nessun raccordo.

    Assegna Σ_A, Σ_B, Σ_C in base alla zona, gestendo le zone degeneri.
    """
    r  = np.asarray(r, float)
    SA = Sigma_A_NT(r, a, mdot, alpha)
    SB = Sigma_B_NT(r, a, mdot, alpha)
    SC = Sigma_C_NT(r, a, mdot, alpha, M)

    result = SA.copy()
    result[r > r_AB] = SB[r > r_AB]
    result[r > r_BC] = SC[r > r_BC]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SEMISPESSORE H_NT(r)  —  con frequenza orbitale di Kerr
# ═══════════════════════════════════════════════════════════════════════════════
def H_A(r, a, mdot, m=M_BH):
    """
    r : array_like   raggio [3r_g, adimensionale]
    m : float        massa BH [M_sun]
    """
    r = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['A']**2 * f['B']**(-3)
           * np.sqrt(np.maximum(f['C'], 0))
           * np.maximum(f['D'], 1e-30)**(-1)
           * np.maximum(f['E'], 1e-30)**(-1)
           * f['Q'])
    return 1e5 * mdot * m * rel


def H_B(r, a, mdot, alpha, M):
    """
    Semispessore fisico zona B  [cm]  —  SS 1973 eq. 2.16
    Zona gas-dominated, opacità e-scattering.

    r     : array_like   raggio [r_g]
    mdot  : float        ṁ = Ṁ/Ṁ_Edd
    alpha : float        parametro viscosità α
    M     : float        massa BH [M_sun]
    """
    r = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['A'] * f['B']**(-6/5)
           * np.sqrt(np.maximum(f['C'], 0))
           * np.maximum(f['D'], 1e-30)**(-3/5)
           * np.maximum(f['E'], 1e-30)**(-1/2)
           * f['Q']**(1/5))
    return 1e3 * alpha**(-1/10) * mdot**(1/5) * float(M)**(9/10) \
           * r**(21/20) * rel


def H_C(r, a, mdot, alpha, M):
    """
    Semispessore fisico zona C  [cm]  —  SS 1973 eq. 2.19
    Zona gas-dominated, opacità free-free (Kramers: κ_ff ∝ ρ T^{-7/2}).

    r     : array_like   raggio [r_g]
    mdot  : float        ṁ = Ṁ/Ṁ_Edd
    alpha : float        parametro viscosità α
    M     : float        massa BH [M_sun]
    """
    r = np.asarray(r, float)
    f   = nt_factors(r, a)
    rel = (f['A']**(19/20) * f['B']**(-11/10)
           * np.sqrt(np.maximum(f['C'], 0))
           * np.maximum(f['D'], 1e-30)**(-23/40)
           * np.maximum(f['E'], 1e-30)**(-19/40)
           * f['Q']**(3/20))
    return 4e2 * alpha**(-1/10) * mdot**(3/20) * float(M)**(9/10) \
           * r**(9/8) * rel

def H_NT(r, a, mdot, alpha, M, r_AB, r_BC):
    """
    Semispessore fisico NT  H(r)  [cm] per zona.

    Derivazione: dall'equilibrio idrostatico verticale in metrica di Kerr,
    il semispessore è:
        H(r) = c_s^{NT}(r) / Ω_φ^{NT}(r)

    dove c_s^{NT} è la velocità del suono che include P_tot zona per zona.

    Per mantener consistenza con le formule di Σ_NT e la struttura termica,
    usiamo:
        H(r) = H_SS_zona(r) × fattore_NT_verticale(r)

    I fattori SS per H sono (da Tab. 1 SS 1973):
        H_A^{SS} = 1.0e8 · m^{-1/2} · r^{-3/4}
        H_B^{SS} = 1.5e9 · α^{1/20} · ṁ^{2/5}   · m^{-9/20} · r^{-51/40} · f^{2/5}
        H_C^{SS} = 2.1e9 · α^{1/20} · ṁ^{17/40} · m^{-9/20} · r^{-21/16} · f^{17/40}

    Il fattore NT verticale è  B(r) / sqrt(C(r)), che corregge la frequenza
    kepleriana e la frequenza epicliclica verticale in metrica di Kerr.

    Per zona A usiamo H_A^{SS} × B/sqrt(C) (struttura verticale rad-dominated NT).
    Per zone B e C usiamo H_B,C^{SS} × B/sqrt(C) (gas-dominated NT).

    Parametri
    ----------
    r     : array_like   raggio [r_g]
    a     : float        spin
    mdot  : float        ṁ
    alpha : float        α
    M     : float        massa BH [M_sun]
    r_AB, r_BC : float   frontiere [r_g]
    """
    r   = np.asarray(r, float)
    HA  = H_A(r, a, mdot, M)
    HB  = H_B(r, a, mdot, alpha, M)
    HC  = H_C(r, a, mdot, alpha, M)

    result = HA.copy()
    result[r > r_AB] = HB[r > r_AB]
    result[r > r_BC] = HC[r > r_BC]
    return result



# ═══════════════════════════════════════════════════════════════════════════════
# 5.  campo magnetico e velocità suono
# ═══════════════════════════════════════════════════════════════════════════════

def _B0_disk(r, a, mdot, alpha, M, r_AB, r_BC):
    """
    B₀(r) in equipartizione — usa Σ_NT e H_NT raccordati nella regione di plunge.

        B_eq(r) = sqrt(4π · Σ_NT(r) · Ω_φ^{NT}(r)² · H_NT(r))

    Ω_φ^{NT} = 2π ν_φ(r, a, M) è la frequenza orbitale di Kerr.
    Σ e H vengono raccordati a zero con profilo √ per r < r_match (plunge).
    B → 0 all'ISCO di conseguenza (Σ → 0 e H → 0 con la stessa legge).
    """
    r  = np.asarray(r, float)

    S         = _Sigma_disk(r, a, mdot, alpha, M, r_AB, r_BC)
    Hv        = H_NT(r, a, mdot, alpha, M, r_AB, r_BC)
    Omega_phi = 2.0 * np.pi * nu_phi(r, a, M)
    B_raw     = np.sqrt(np.maximum(4.0 * np.pi * S * Omega_phi**2 * Hv, 0.0))
    _, _, B_out, _ = _apply_plunge_raccordo(r, a, S, Hv, B_raw)
    return B_out


def _sound_speed(r, a, Hv=None, hr=None, M=M_BH):
    """
    Velocità del suono effettiva per la relazione di dispersione AEI  [cm/s].

        c_s(r) = hr · r · R_g · Ω_φ(r)

    hr è l'aspect ratio fenomenologico per il solver AEI.
    Non entra in Σ né in B (che usano H_NT dalla fisica di ogni zona).

    r  : array_like   raggio [r_g]
    a  : float        spin del BH
    hr : float        aspect ratio H/r per la dispersione AEI
    M  : float        massa BH [M_sun]
    """
    r  = np.asarray(r, dtype=float)
    if Hv is not None:
        cs = Hv * np.sqrt(nu_phi(r, a, M))
    elif hr is not None:
        cs = hr * r * Rg_SUN * M * nu_phi(r, a, M)
    return cs


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  INDICE DI ZONA
# ═══════════════════════════════════════════════════════════════════════════════

def _zone_array(r, r_AB, r_BC):
    """Restituisce array di etichette zona ('A','B','C') per ogni r."""
    r  = np.asarray(r)
    zi = np.zeros(r.shape, dtype=int)
    zi[r > r_AB] = 1
    zi[r > r_BC] = 2
    return np.array([ZONE_NAMES[i] for i in zi])


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ADAPTER PRINCIPALE  —  disk_model_NT
# ═══════════════════════════════════════════════════════════════════════════════

def disk_model_NT(r_rg, a, mdot, alpha_visc=ALPHA_VISC, hr=None, M=M_BH):
    """
    Profili fisici del disco NT 1973 — firma standard per find_rossby.

    Calcola su un array radiale:
      - Σ(r): formule NT raw per zona, nessun raccordo
      - B(r): equipartizione da P_mid = Σ_NT · Ω_φ^{NT,2} · H_NT / 2
      - c_s(r): thin-disc fenomenologica c_s = hr·r·R_g·Ω_φ (per solver AEI)

    Parametri liberi: a, mdot, alpha_visc, M.
    hr entra solo in c_s.

    Parametri
    ----------
    r_rg      : array_like   raggi [r_g]
    a         : float        spin adimensionale [−1, 1]
    mdot      : float        ṁ = Ṁ/Ṁ_Edd > 0
    alpha_visc: float        parametro α di viscosità
    hr        : float        aspect ratio H/r per c_s AEI
    M         : float        massa BH [M_sun]

    Restituisce
    -----------
    B0    : ndarray   campo magnetico in equipartizione [G]
    Sigma : ndarray   densità superficiale [g/cm²]
    c_s   : ndarray   velocità del suono AEI [cm/s]
    zone  : ndarray   etichette zona ('A', 'B', 'C')
    info  : dict      r_AB, r_BC, zone_present, mdot, alpha
    """
    r_rg = np.asarray(r_rg, float)
    r_AB, r_BC, zone_present = nt_boundaries(a, mdot, alpha=alpha_visc, M=M)

    Sigma = _Sigma_disk(r_rg, a, mdot, alpha_visc, M, r_AB, r_BC)
    Hv    = H_NT(r_rg, a, mdot, alpha_visc, M, r_AB, r_BC)
    Omega = 2.0 * np.pi * nu_phi(r_rg, a, M)
    B0    = np.sqrt(np.maximum(4.0 * np.pi * Sigma * Omega**2 * Hv, 0.0))

    # trova r_match dal minimo di Q_AEI sui profili raw
    # usa una griglia fine vicino all'ISCO, non r_rg (troppo sparsa)
    rISCO = float(r_isco(a))
    r_fine  = np.geomspace(rISCO * 1.001, rISCO * 6, 2000)
    S_fine  = _Sigma_disk(r_fine, a, mdot, alpha_visc, M, r_AB, r_BC)
    H_fine  = H_NT(r_fine, a, mdot, alpha_visc, M, r_AB, r_BC)
    Om_fine = 2.0 * np.pi * nu_phi(r_fine, a, M)
    B_fine  = np.sqrt(np.maximum(4.0 * np.pi * S_fine * Om_fine**2 * H_fine, 0.0))

    Sigma, Hv, B0, r_match = _apply_plunge_raccordo(
        r_rg, a, Sigma, Hv, B0,
        #r_scan_profiles=(r_fine, S_fine, B_fine, Om_fine),
        Q_threshold=0.1
    )
    
    if hr is None:
        hr = Hv / np.maximum(r_rg * Rg_SUN * M, 1e-30)
    else:
        Hv = hr * r_rg * Rg_SUN * M
    c_s  = _sound_speed(r_rg, a, hr=hr, M=M)
    zone = _zone_array(r_rg, r_AB, r_BC)

    # Se la zona B è assente r_BC coincide con r_ISCO e non rappresenta
    # l'estensione radiale del disco (il disco è tutto zona C, illimitato).
    # Aggiungiamo r_max_hint come stima per compute_disk_profile, senza
    # sovrascrivere r_BC che rimane la frontiera fisica reale.
    if not zone_present['B']:
        r_max_hint = float((mdot / 2e-6) ** (2.0 / 3.0))
        r_max_hint = max(r_max_hint, r_AB * 10.0)
    else:
        r_max_hint = r_BC

    info = {
        'r_AB':         r_AB,
        'r_BC':         r_BC,        # frontiera fisica reale (= r_ISCO se zona B assente)
        'r_max_hint':   r_max_hint,  # usato da compute_disk_profile per r_max
        'zone_present': zone_present,
        'mdot':         mdot,
        'alpha':        alpha_visc,
    }
    return B0, Sigma, c_s, hr, zone, info


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  VALORI AI BORDI INTERNI
# ═══════════════════════════════════════════════════════════════════════════════

def disk_inner_values_NT(a, mdot, alpha_visc=ALPHA_VISC, hr=HOR, M=M_BH):
    """
    Restituisce Σ, H_NT e B ai bordi interni fisicamente significativi.

    Bordi:
      r_ISCO — Inner Stable Circular Orbit (bordo del disco)
      r_H    — orizzonte degli eventi

    Restituisce
    -----------
    dict con:
      r_ISCO      [r_g]     raggio ISCO
      r_H         [r_g]     raggio orizzonte
      r_AB        [r_g]     frontiera A-B
      r_BC        [r_g]     frontiera B-C
      zone_present          {'A': bool, 'B': bool, 'C': bool}
      Sigma_ISCO  [g/cm²]   Σ all'ISCO
      B_ISCO      [G]       B_eq all'ISCO (bordo interno del disco)
      B_rH        [G]       B_eq all'orizzonte (sempre 0 per NT perché r_H < dominio disco)

    Nota: per NT r_H cade fuori dal dominio del disco (r_H < r_ISCO sempre),
    quindi B_rH è strutturalmente zero. Usare B_ISCO come parametro di normalizzazione.
    """
def disk_inner_values_NT(a, mdot, alpha_visc=ALPHA_VISC, hr=HOR, M=M_BH):
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
    r_AB, r_BC, zone_present = nt_boundaries(a, mdot, alpha=alpha_visc, M=M)

    # griglia fine interna
    r_fine  = np.geomspace(rISCO * 1.001, rISCO * 6, 2000)
    S_fine  = _Sigma_disk(r_fine, a, mdot, alpha_visc, M, r_AB, r_BC)
    H_fine  = H_NT(r_fine, a, mdot, alpha_visc, M, r_AB, r_BC)
    Om_fine = 2.0 * np.pi * nu_phi(r_fine, a, M)
    B_fine  = np.sqrt(np.maximum(4.0 * np.pi * S_fine * Om_fine**2 * H_fine, 0.0))

    r_match = _r_match(a, r_scan_profiles=(r_fine, S_fine, B_fine, Om_fine))

    # applica il raccordo agli stessi profili fini
    S_racc, H_racc, B_racc, _ = _apply_plunge_raccordo(
        r_fine, a, S_fine, H_fine, B_fine,
        r_scan_profiles=(r_fine, S_fine, B_fine, Om_fine),
    )

    # Sigma_ref: massimo di Σ raccordata nel tratto [r_ISCO, r_match]
    plunge_mask = r_fine <= r_match
    if plunge_mask.any():
        Sigma_ref = float(np.max(S_racc[plunge_mask]))
    else:
        Sigma_ref = float(S_racc[0])

    # B_ISCO: valore di B raccordata al primo punto della griglia (≈ r_ISCO)
    B_ISCO = float(B_racc[0])

    return {
        'r_ISCO':       rISCO,
        'r_H':          rH,
        'r_AB':         r_AB,
        'r_BC':         r_BC,
        'r_match':      r_match,
        'zone_present': zone_present,
        'Sigma_ISCO':   Sigma_ref,
        'B_ISCO':       B_ISCO,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  DIAGNOSTICA  —  check_continuity_NT
# ═══════════════════════════════════════════════════════════════════════════════

def check_continuity_NT(a, mdot, alpha_visc=ALPHA_VISC, hr=HOR, M=M_BH,
                        tol=0.5, verbose=True, only_multizone=True):
    """
    Misura le discontinuità reali di Σ, H_NT e B alle frontiere r_AB e r_BC.

    Calcola il salto relativo  |f(r⁺) − f(r⁻)| / |f(r⁻)|  a distanza
    ε = 10⁻⁴ · r₀ da ogni frontiera.

    I salti riflettono la coerenza fisica delle frontiere NT: in un modello
    perfettamente coerente dovrebbero essere piccoli (le formule sono derivate
    proprio imponendo la continuità delle condizioni termodinamiche alle frontiere).
    Salti grandi indicano problemi nei prefattori numerici delle formule NT.

    tol : float   soglia per il flag 'ok' (default 0.5 = 50%)

    Restituisce
    -----------
    results : dict
        Per ogni frontiera ('r_AB', 'r_BC') e variabile ('Sigma', 'H', 'B'):
            {'value_in', 'value_out', 'jump_rel', 'ok'}
    """
    r_AB, r_BC, zone_present = nt_boundaries(a, mdot, alpha=alpha_visc, M=M)
    rISCO = float(r_isco(a))
    eps   = 1e-4

    # filtro dischi monotoni
    if only_multizone and not (zone_present['A'] or zone_present['B']):
        if verbose:
            print("Disco NT monotona (solo zona C) — check saltato.")
        return {}, zone_present

    results = {}

    if verbose:
        print(f"┌─ Salti NT raw  (a={a:.3f}, mdot={mdot:.3e}, α={alpha_visc}, M={M:.2e} Msun)")
        print(f"│  r_ISCO = {rISCO:.4g} rg")
        print(f"│  r_AB   = {r_AB:.4g} rg  [zona A {'presente' if zone_present['A'] else 'ASSENTE'}]")
        print(f"│  r_BC   = {r_BC:.4g} rg  [zona B {'presente' if zone_present['B'] else 'ASSENTE'}]")
        print(f"├{'─'*68}")

    # selezione frontiere fisicamente presenti
    boundaries = []
    if zone_present['A']:
        boundaries.append(('r_AB', r_AB))
    if zone_present['B']:
        boundaries.append(('r_BC', r_BC))

    for label, r0 in boundaries:
        r_in  = np.array([r0 * (1.0 - eps)])
        r_out = np.array([r0 * (1.0 + eps)])

        S_in  = float(_Sigma_disk(r_in,  a, mdot, alpha_visc, M, r_AB, r_BC)[0])
        S_out = float(_Sigma_disk(r_out, a, mdot, alpha_visc, M, r_AB, r_BC)[0])
        Hi_in = float(H_NT(r_in,  a, mdot, alpha_visc, M, r_AB, r_BC)[0])
        Hi_out= float(H_NT(r_out, a, mdot, alpha_visc, M, r_AB, r_BC)[0])
        B_in  = float(_B0_disk(r_in,  a, mdot, alpha_visc, M, r_AB, r_BC)[0])
        B_out = float(_B0_disk(r_out, a, mdot, alpha_visc, M, r_AB, r_BC)[0])

        dS = abs(S_in  - S_out)  / max(abs(S_in),  1e-300)
        dH = abs(Hi_in - Hi_out) / max(abs(Hi_in), 1e-300)
        dB = abs(B_in  - B_out)  / max(abs(B_in),  1e-300)

        results[label] = {
            'Sigma': {'value_in': S_in,  'value_out': S_out,  'jump_rel': dS, 'ok': dS < tol},
            'H':     {'value_in': Hi_in, 'value_out': Hi_out, 'jump_rel': dH, 'ok': dH < tol},
            'B':     {'value_in': B_in,  'value_out': B_out,  'jump_rel': dB, 'ok': dB < tol},
        }

        if verbose:
            def fmtrow(v_in, v_out, dv, ok):
                factor = max(v_in, v_out) / max(min(v_in, v_out), 1e-300)
                flag   = '✓' if ok else f'✗  (fattore {factor:.1f}×)'
                return f"{v_in:.3e} → {v_out:.3e}  |  Δ/val = {dv:.2e}  {flag}"
            print(f"│  {label} = {r0:.4g} rg")
            print(f"│    Σ:  {fmtrow(S_in, S_out, dS, dS < tol)}")
            print(f"│    H:  {fmtrow(Hi_in, Hi_out, dH, dH < tol)}")
            print(f"│    B:  {fmtrow(B_in, B_out, dB, dB < tol)}")

    if verbose:
        print(f"└{'─'*68}")

    return results, zone_present
