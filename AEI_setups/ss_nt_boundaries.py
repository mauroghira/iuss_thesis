"""
ss_nt_boundaries.py  
==========================================
Modello di disco Novikov-Thorne per l'analisi AEI.

Parametri liberi del modello:  a,  B00,  Sigma0,  alpha_visc
                                (mdot è sempre derivato internamente da Sigma0)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FISICA: COME VENGONO CALCOLATI Σ(r) e B₀(r)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Il disco è diviso in tre zone da due frontiere radiali:

  r_AB  dove β/(1-β) = 1        →  P_rad = P_gas          (bordo interno zona A)
  r_BC  dove τ_ff / τ_es = 1    →  cambio regime opacità   (bordo esterno zona B)

Le frontiere dipendono solo da (a, Sigma0, alpha) — non da B00.

──────────────────────────────────────────────────────────────────────────
Σ(r) — modello ibrido SS / NT
──────────────────────────────────────────────────────────────────────────
  Zona A  [r_ISCO, r_AB]:  power-law identica a full_disk_SS
      Σ_A(r) = Sigma0 · (r / r_ISCO)^{+3/2}   (esponente S&S α_Σ = -3/2)
      Sigma0 è il valore di Σ all'ISCO — parametro libero assoluto.
      Continua per costruzione, nessuno smoothing necessario.

  Zone B e C  [r_AB, ∞):  formule NT analitiche esatte (review Abramowicz)
      Σ_B(r) = Σ_B_NT(r) × [Σ_A(r_AB) / Σ_B_NT(r_AB)]   ← raccordo a r_AB
      Σ_C(r) = Σ_C_NT(r) × [Σ_B(r_BC)  / Σ_C_NT(r_BC)]  ← raccordo a r_BC

  I fattori di riscalatura garantiscono continuità esatta alle frontiere.
  Le formule NT contengono i fattori relativistici A,B,C,D,E,Q (sezione 2)
  e i prefattori fisici corretti per ogni zona — non semplici power-law.

  Il parametro ṁ è derivato internamente da Sigma0 e non è mai esposto.

──────────────────────────────────────────────────────────────────────────
B₀(r) — modello ibrido SS / NT
──────────────────────────────────────────────────────────────────────────
  Zona A  [r_H, r_AB]:  power-law identica a full_disk_SS
      B₀_A(r) = B00 · (r / r_H)^{-3/4}
      B00 è il campo in Gauss all'orizzonte — parametro libero assoluto.
      Identico alla zona A di full_disk_SS: stessa fisica, stesso parametro.

  Zone B e C  [r_AB, ∞):  formule S&S 1973 (eq. 2.16 e 2.19)
      B₀_B(r) = ss_B0_B(r) × [B₀_A(r_AB) / ss_B0_B(r_AB)]  ← raccordo a r_AB
      B₀_C(r) = ss_B0_C(r) × [B₀_B(r_BC)  / ss_B0_C(r_BC)] ← raccordo a r_BC

  Le formule S&S danno l'andamento radiale fisico di ogni zona
  (esponenti -51/40 e -21/16, vicini a -5/4).
  Il fattore di riscalatura aggancia l'ampiezza assoluta a B00 via zona A.

  DIFFERENZA rispetto a full_disk_SS: le zone B e C hanno l'andamento S&S esatto (leggermente
  diverso da -5/4) con la stessa normalizzazione assoluta B00.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API PUBBLICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # diagnostica
  nt_boundaries(a, Sigma0, alpha, M)        → mdot, r_AB, r_BC
  nt_print_boundaries(a, Sigma0, alpha, M)  → stampa riepilogo
  nt_scan_grid(Sigma0_vals, a_vals, ...)    → DataFrame diagnostico

  # profili (basso livello, raramente necessari)
  nt_ABCDEQ(r, a)                           → dict fattori NT
  ss_B0_B(r, mdot, alpha, M)               → B₀ zona B [G]
  ss_B0_C(r, mdot, alpha, M)               → B₀ zona C [G]

  # adapter per find_rossby
  disk_model_NT(r, a, B00, Sigma0, alpha_visc, hr, M)
      → B0, Sigma, c_s, zone               usare con find_rossby

  # profilo radiale completo (drop-in di compute_full_disk_profile)
  compute_nt_disk_profile(a, B00, Sigma0, mm, hr, alpha, M, ...)
      → df, meta                           usare con radial_scan_grid

Dipendenze: numpy, pandas, setup.py, aei_common.py
"""

import numpy as np
import pandas as pd
from functools import lru_cache

from .aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    check_k_wkb, _make_interp,
    ALPHA_VISC, HOR
)

import sys
sys.path.append("..")
from setup import r_isco, r_horizon, nu_phi, Rg_SUN, M_BH

ZONE_NAMES    = ['A', 'B', 'C']


# ═══════════════════════════════════════════════════════════════════════════
# 1.  COSTANTI DI SPIN  (cache LRU — calcolate una volta per valore di a)
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _nt_spin_constants(a):
    as_ = float(a)
    rISCO = float(r_isco(as_))
    y0 = np.sqrt(rISCO)
    _acos = np.arccos(np.clip(as_, -1 + 1e-10, 1 - 1e-10))
    y1 =  2 * np.cos((_acos - np.pi) / 3)
    y2 =  2 * np.cos((_acos + np.pi) / 3)
    y3 = -2 * np.cos( _acos          / 3)
    c1 = 3*(y1 - as_)**2 / (y1*(y1-y2)*(y1-y3))
    c2 = 3*(y2 - as_)**2 / (y2*(y2-y1)*(y2-y3))
    c3 = 3*(y3 - as_)**2 / (y3*(y3-y1)*(y3-y2))
    return y0, y1, y2, y3, c1, c2, c3


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FATTORI RELATIVISTICI NT  (vettorizzati)
# ═══════════════════════════════════════════════════════════════════════════

def nt_ABCDEQ(r, a):
    """
    Fattori relativistici A,B,C,D,E,Q0,Q di Novikov-Thorne.
    Tutti su array r senza loop Python.

    r : float o ndarray [r_g]
    a : float spin adimensionale
    """
    r   = np.asarray(r, dtype=float)
    as_ = float(a)
    y   = np.sqrt(r)

    A  = 1 + as_**2*y**(-4) + 2*as_**2*y**(-6)
    B  = 1 + as_*y**(-3)
    C  = 1 - 3*y**(-2) + 2*as_*y**(-3)
    D  = 1 - 2*y**(-2) + as_**2*y**(-4)
    E  = 1 + 4*as_**2*y**(-4) - 4*as_**2*y**(-6) + 3*as_**4*y**(-8)
    Q0 = (1 + as_*y**(-3)) / (y * np.sqrt(np.maximum(C, 1e-30)))

    y0, y1, y2, y3, c1, c2, c3 = _nt_spin_constants(as_)
    sl = lambda x: np.log(np.maximum(np.abs(x), 1e-300))

    term0 = y - y0 - 1.5*as_*sl(y / y0)
    Q_raw = Q0 * (term0
                  - c1*sl((y-y1)/(y0-y1))
                  - c2*sl((y-y2)/(y0-y2))
                  - c3*sl((y-y3)/(y0-y3)))
    Q = np.where(y > y0*1.001, np.maximum(Q_raw, 1e-10), 1e-10)
    if r.ndim == 0:
        Q = float(Q)
    return dict(A=A, B=B, C=C, D=D, E=E, Q0=Q0, Q=Q)


# ═══════════════════════════════════════════════════════════════════════════
# Funzione per carrdo profili
# ═══════════════════════════════════════════════════════════════════════════

def _log_blend(r, r_c, f_lo, f_hi, width=0.05):
    """
    Raccordo log-lineare tra f_lo e f_hi attorno alla frontiera r_c.

    Il blending avviene nella finestra [r_c·(1-w), r_c·(1+w)], con w = width.
    Il peso t è lineare in log(r) — scelta naturale per grandezze power-law:

        t(r) = [log(r) - log(r_lo)] / [log(r_hi) - log(r_lo)]  ∈ [0, 1]

    Il raccordo è un'interpolazione lineare nei logaritmi delle funzioni:

        log f_blend = (1-t)·log f_lo + t·log f_hi

    ovvero  f_blend = f_lo^(1-t) · f_hi^t.

    Proprietà:
      - Continuo esatto agli estremi della finestra (t=0 → f_lo, t=1 → f_hi)
      - Monotono se entrambe le funzioni lo sono in quella regione
      - Nessun overshoot — rimane sempre tra f_lo e f_hi in log-spazio
      - Rispecchia la geometria naturale del problema (tutto è power-law)

    Parametri
    ----------
    r      : array   raggi in r_g
    r_c    : float   raggio di frontiera (centro della finestra)
    f_lo   : array   valori zona a sinistra  (già calcolati su tutto r)
    f_hi   : array   valori zona a destra    (già calcolati su tutto r)
    width  : float   semi-ampiezza relativa  (default 0.05 = ±5%)
    """
    r_lo = r_c * (1.0 - width)
    r_hi = r_c * (1.0 + width)
    t = np.clip(
        (np.log(r) - np.log(r_lo)) / (np.log(r_hi) - np.log(r_lo)),
        0.0, 1.0
    )
    log_f = (1.0 - t) * np.log(np.maximum(f_lo, 1e-300)) \
           +        t  * np.log(np.maximum(f_hi, 1e-300))
    return np.exp(log_f)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Σ(r) — formule NT esatte per zona
# ═══════════════════════════════════════════════════════════════════════════

def _Sigma_A(r, a, mdot, alpha):
    f   = nt_ABCDEQ(r, a)
    rel = f['A']**(-2) * f['B']**3 * np.sqrt(np.maximum(f['C'], 0)) * f['E'] / f['Q']
    return 5.0 * alpha**(-1) * mdot**(-1) * np.asarray(r, float)**(3/2) * rel


def _Sigma_B(r, a, mdot, alpha):
    f   = nt_ABCDEQ(r, a)
    rel = f['B']**(-4/5) * np.sqrt(np.maximum(f['C'], 0)) * f['D']**(-4/5) * f['Q']**(3/5)
    return 9e4 * alpha**(-4/5) * mdot**(3/5) * np.asarray(r, float)**(-3/5) * rel


def _Sigma_C(r, a, mdot, alpha, m):
    f   = nt_ABCDEQ(r, a)
    rel = (f['A']**(1/10) * f['B']**(-4/5) * np.sqrt(np.maximum(f['C'], 0))
           * f['D']**(-17/20) * f['E']**(-1/20) * f['Q']**(7/10))
    return 4e5 * alpha**(-4/5) * float(m)**(1/5) * mdot**(7/10) * np.asarray(r, float)**(-3/4) * rel


def _Sigma_disk(r, a, Sigma0, mdot, alpha, M, r_AB, r_BC, blend_width=0.05):
    """
    Σ(r) sull'intero disco — formule NT/SS esatte con raccordo log-lineare.

    Zona A  [r_ISCO, r_AB]:  formula NT inner esatta (review Abramowicz, eq. 99)
        Σ_A(r) = 5 α⁻¹ ṁ⁻¹ r*^{3/2} · A⁻²B³C^{1/2}EQ⁻¹
        Normalizzata a Sigma0 per costruzione: nt_mdot_from_Sigma0 inverte
        Σ_A(r_ISCO) = Sigma0, quindi nessun fattore di scala aggiuntivo.

    Zona B  [r_AB, r_BC]:  formula NT middle esatta (review Abramowicz, eq. 98)
        Σ_B(r) = 9×10⁴ α⁻⁴/⁵ ṁ^{3/5} r*^{-3/5} · B⁻⁴/⁵C^{1/2}D⁻⁴/⁵Q^{3/5}

    Zona C  [r_BC, ∞):  formula NT outer esatta (review Abramowicz, eq. 97)
        Σ_C(r) = 4×10⁵ α⁻⁴/⁵ m^{1/5} ṁ^{7/10} r*^{-3/4} ·
                 A^{1/10}B⁻⁴/⁵C^{1/2}D⁻¹⁷/²⁰E⁻¹/²⁰Q^{7/10}

    Raccordo alle frontiere:
        Nella regione [r_c·(1-w), r_c·(1+w)] (w = blend_width = 5% di default)
        le due formule adiacenti sono interpolate log-linearmente in log(r):
            Σ_blend = Σ_lo^(1-t) · Σ_hi^t,   t ∈ [0,1] lineare in log(r)
        Fuori dalla finestra ogni zona usa esclusivamente la propria formula.

    Parametri
    ----------
    blend_width : float   semi-ampiezza relativa della finestra di raccordo
                          (default 0.05 → ±5% di r_AB e r_BC)
    """
    r = np.asarray(r, float)
    m = float(M)

    # valori grezzi di ogni zona calcolati su tutto il dominio
    S_A = _Sigma_A(r, a, mdot, alpha)
    S_B = _Sigma_B(r, a, mdot, alpha)
    S_C = _Sigma_C(r, a, mdot, alpha, m)

    # blend A↔B attorno a r_AB  e  B↔C attorno a r_BC
    S_AB = _log_blend(r, r_AB, S_A, S_B, width=blend_width)
    S_BC = _log_blend(r, r_BC, S_B, S_C, width=blend_width)

    # maschere di zona (le regioni di blend si sovrappongono ai bordi)
    w = blend_width
    in_blend_AB = (r >= r_AB * (1.0 - w)) & (r <  r_AB * (1.0 + w))
    in_pure_B   = (r >= r_AB * (1.0 + w)) & (r <  r_BC * (1.0 - w))
    in_blend_BC = (r >= r_BC * (1.0 - w)) & (r <  r_BC * (1.0 + w))
    in_pure_C   =  r >= r_BC * (1.0 + w)

    result = S_A.copy()                          # default: pura zona A
    result[in_blend_AB] = S_AB[in_blend_AB]
    result[in_pure_B]   = S_B[in_pure_B]
    result[in_blend_BC] = S_BC[in_blend_BC]
    result[in_pure_C]   = S_C[in_pure_C]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4.  B₀(r) — formule S&S 1973 con raccordo log-lineare
# ═══════════════════════════════════════════════════════════════════════════

def _ss_f(r):
    """Fattore non-relativista f = max(1 - r^{-1/2}, ε)."""
    return np.maximum(1.0 - np.asarray(r, float)**(-0.5), 1e-10)


def ss_B0_B(r, mdot, alpha, M=M_BH):
    """
    B₀ zona B [G] — S&S 1973 eq. 2.16.
        H_B = 1.5×10⁹ · α^{1/20} · ṁ^{2/5} · m^{-9/20} · r^{-51/40} · f^{2/5}
    Esponente esatto -51/40 = -1.275  (vs approssimazione -5/4 = -1.25).
    Prefattore fisico assoluto in Gauss.
    """
    r = np.asarray(r, float)
    return 1.5e9 * alpha**(1/20) * mdot**(2/5) * float(M)**(-9/20) * r**(-51/40) * _ss_f(r)**(2/5)


def ss_B0_C(r, mdot, alpha, M=M_BH):
    """
    B₀ zona C [G] — S&S 1973 eq. 2.19.
        H_C = 2.1×10⁹ · α^{1/20} · ṁ^{17/40} · m^{-9/20} · r^{-21/16} · f^{17/40}
    Esponente esatto -21/16 = -1.3125  (vs approssimazione -5/4 = -1.25).
    Prefattore fisico assoluto in Gauss.
    """
    r = np.asarray(r, float)
    return 2.1e9 * alpha**(1/20) * mdot**(17/40) * float(M)**(-9/20) * r**(-21/16) * _ss_f(r)**(17/40)


def _B0_disk(r, a, B00, mdot, alpha, M, r_AB, r_BC, blend_width=0.05):
    """
    B₀(r) sull'intero disco — formule S&S 1973 esatte con raccordo log-lineare.

    Zona A  [r_H, r_AB]:  power-law con parametro libero B00
        B₀_A(r) = B00 · (r / r_H)^{-3/4}
        B00 è il valore di B₀ in Gauss all'orizzonte degli eventi r_H.

    Zona B  [r_AB, r_BC]:  formula S&S 1973 eq. 2.16
        B₀_B(r) = 1.5×10⁹ · α^{1/20} · ṁ^{2/5} · m^{-9/20} · r^{-51/40} · f^{2/5}

    Zona C  [r_BC, ∞):  formula S&S 1973 eq. 2.19
        B₀_C(r) = 2.1×10⁹ · α^{1/20} · ṁ^{17/40} · m^{-9/20} · r^{-21/16} · f^{17/40}

    Raccordo alle frontiere:
        Come per Σ, si usa un'interpolazione log-lineare in log(r):
            B_blend = B_lo^(1-t) · B_hi^t,   t ∈ [0,1] lineare in log(r)
        nella finestra [r_c·(1-w), r_c·(1+w)] con w = blend_width.

    Parametri
    ----------
    blend_width : float   semi-ampiezza relativa della finestra di raccordo
                          (default 0.05 → ±5% di r_AB e r_BC)
    """
    r  = np.asarray(r, float)
    rH = float(r_horizon(a))

    B0_A = B00 * (r / rH)**(-3/4)
    B0_B = ss_B0_B(r, mdot, alpha, M)
    B0_C = ss_B0_C(r, mdot, alpha, M)

    B_AB = _log_blend(r, r_AB, B0_A, B0_B, width=blend_width)
    B_BC = _log_blend(r, r_BC, B0_B, B0_C, width=blend_width)

    w = blend_width
    in_blend_AB = (r >= r_AB * (1.0 - w)) & (r <  r_AB * (1.0 + w))
    in_pure_B   = (r >= r_AB * (1.0 + w)) & (r <  r_BC * (1.0 - w))
    in_blend_BC = (r >= r_BC * (1.0 - w)) & (r <  r_BC * (1.0 + w))
    in_pure_C   =  r >= r_BC * (1.0 + w)

    result = B0_A.copy()                         # default: pura zona A
    result[in_blend_AB] = B_AB[in_blend_AB]
    result[in_pure_B]   = B0_B[in_pure_B]
    result[in_blend_BC] = B_BC[in_blend_BC]
    result[in_pure_C]   = B0_C[in_pure_C]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Σ₀ → ṁ  e  frontiere r_AB, r_BC
# ═══════════════════════════════════════════════════════════════════════════

def nt_mdot_from_Sigma0(Sigma0, a, alpha=ALPHA_VISC):
    """
    Inverte Σ_A(r_ISCO) = Sigma0 per ricavare ṁ.
    Clippato in [1e-4, 50] per evitare regimi fisicamente assurdi.
    """
    rISCO = float(r_isco(a))
    return float(5.0 * rISCO**(3/2) / (alpha * float(Sigma0)))


def _bisect_vec(func_vec, r_lo, r_hi, n_scan=200, n_bisect=40):
    """Trova r dove func(r)=1 tramite bisezione vettorizzata."""
    r_arr  = np.geomspace(r_lo, r_hi, n_scan)
    f_arr  = func_vec(r_arr) - 1.0
    sc     = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sc) == 0:
        return None
    a_r, b_r = r_arr[sc[0]], r_arr[sc[0]+1]
    for _ in range(n_bisect):
        mid = (a_r + b_r) / 2
        if func_vec(np.array([mid]))[0] - 1.0 < 0:
            a_r = mid
        else:
            b_r = mid
    return (a_r + b_r) / 2


def _r_AB_search_limit(mdot, alpha, M):
    """
    Stima analitica del limite superiore per la ricerca di r_AB.

    Per r >> r_ISCO i fattori NT → 1 e f_AB ~ r^(21/8), quindi il crossing vale:
        r_AB ≈ (α^{1/4} M^{1/4} ṁ² / 4×10⁻⁶)^{8/21}
    Restituisce 5× questa stima (margine di sicurezza), limitato a [500, 1e5].
    """
    r_est = (alpha**(1/4) * float(M)**(1/4) * mdot**2 / 4e-6)**(8/21)
    return float(np.clip(r_est * 5.0, 500.0, 1e5))


def nt_boundaries(a, Sigma0, alpha=ALPHA_VISC, M=M_BH):
    """
    Calcola mdot, r_AB, r_BC dai parametri liberi (a, Sigma0, alpha).

    Casi speciali gestiti esplicitamente:

    * f_AB(r_ISCO) ≥ 1  →  il disco è già dominato dalla pressione di gas
      all'ISCO: la zona A è assente, r_AB = r_ISCO  (frontiera collassata).

    * f_AB cresce sempre come r^{21/8}: se f_AB(r_ISCO) < 1 il crossing
      esiste certamente; il limite superiore di ricerca viene stimato
      analiticamente in modo da coprire tutti i valori di ṁ fisicamente
      raggiunti dalla griglia dei parametri.

    Returns
    -------
    mdot  : float   ṁ = Ṁc²/L_Edd  (derivato da Sigma0)
    r_AB  : float   frontiera A-B  [r_g]
    r_BC  : float   frontiera B-C  [r_g]
    """
    rISCO = float(r_isco(a))
    m     = float(M)
    mdot  = nt_mdot_from_Sigma0(Sigma0, a, alpha)

    # ── r_AB: β/(1-β) = 1  (condizione P_rad = P_gas) ───────────────────
    def f_AB(r_arr):
        f   = nt_ABCDEQ(r_arr, a)
        rel = f['A']**(-5/2) * f['B']**(9/2) * f['D'] * f['E']**(5/4) / f['Q']**2
        return 4e-6 * alpha**(-1/4) * m**(-1/4) * mdot**(-2) * np.asarray(r_arr)**(21/8) * rel

    f_at_ISCO = float(f_AB(np.array([rISCO * 1.01]))[0])

    if f_at_ISCO >= 1.0:
        # Tutto il disco è dominato dal gas già all'ISCO: zona A assente.
        r_AB = rISCO
    else:
        # Il crossing esiste certamente (f_AB ~ r^{21/8} → ∞).
        # Il limite superiore è stimato analiticamente per coprire mdot grandi.
        r_AB_hi = _r_AB_search_limit(mdot, alpha, m)
        r_AB = _bisect_vec(f_AB, rISCO * 1.01, r_AB_hi)
        if r_AB is None:
            # Fallback di sicurezza: non dovrebbe accadere con il limite dinamico,
            # ma se succede significa che r_AB > r_AB_hi → usiamo il limite stesso.
            r_AB = r_AB_hi
        r_AB = float(np.clip(r_AB, rISCO, 1e5))

    # ── r_BC: τ_ff/τ_es = 1  (cambio regime opacità) ────────────────────
    def f_BC(r_arr):
        f   = nt_ABCDEQ(r_arr, a)
        rel = (f['A']**(-1) * f['B']**2
               * np.sqrt(np.maximum(f['D'], 0)) * np.sqrt(np.maximum(f['E'], 0)) / f['Q'])
        return 2e-6 * mdot**(-1) * np.asarray(r_arr)**(3/2) * rel

    r_BC = _bisect_vec(f_BC, r_AB * 1.01, 1e5)
    if r_BC is None:
        # f_BC(r_AB) ≥ 1: zona B assente (disco tutto in zona C da r_AB in poi)
        # oppure crossing oltre 1e5: usiamo il confine del dominio.
        f_at_rAB = float(f_BC(np.array([r_AB * 1.01]))[0])
        r_BC = r_AB * 1.01 if f_at_rAB >= 1.0 else 1e5
    r_BC = float(np.clip(r_BC, r_AB, 1e5))

    return mdot, r_AB, r_BC


# ═══════════════════════════════════════════════════════════════════════════
# 6.  ADAPTER PER find_rossby  e  compute_disk_profile
# ═══════════════════════════════════════════════════════════════════════════

def _zone_array(r, r_AB, r_BC):
    zi = np.zeros(np.asarray(r).shape, dtype=int)
    zi[np.asarray(r) > r_AB] = 1
    zi[np.asarray(r) > r_BC] = 2
    return np.array([ZONE_NAMES[i] for i in zi])


def _sound_speed(r, a, hr, M):
    return hr * 2*np.pi * nu_phi(np.asarray(r, float), a, M) * np.asarray(r, float) * Rg_SUN * M


def disk_model_NT(r_rg, a, B00, Sigma0, alpha_visc=ALPHA_VISC, hr=HOR, M=M_BH):
    """
    Adapter per aei_common.find_rossby — firma standard:

        B0, Sigma, c_s, zone = disk_model_NT(r_rg, **row)

    Parametri liberi: a, B00, Sigma0
    Parametri fissi (passare con lambda o partial): alpha_visc, hr, M

    Esempio d'uso con find_rossby:
        df = find_rossby(
            r_vec, param_grid,
            disk_model = lambda r, **p: disk_model_NT(
                             r, **p, alpha_visc=alpha_visc, hr=hr, M=M_BH),
            m=m, hr=hr, M=M_BH, ...
        )
    """
    r   = np.asarray(r_rg, float)
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha_visc, M)
    B0    = _B0_disk(r, a, B00, mdot, alpha_visc, M, r_AB, r_BC)
    Sigma = _Sigma_disk(r, a, Sigma0, mdot, alpha_visc, M, r_AB, r_BC)
    c_s   = _sound_speed(r, a, hr, M)
    zone  = _zone_array(r, r_AB, r_BC)
    info  = {'r_AB': r_AB, 'r_BC': r_BC, 'mdot': mdot, 'alpha': alpha_visc}
    return B0, Sigma, c_s, zone, info


# ═══════════════════════════════════════════════════════════════════════════
# 7.  DIAGNOSTICA
# ═══════════════════════════════════════════════════════════════════════════

def nt_print_boundaries(a, Sigma0, alpha=ALPHA_VISC, M=M_BH):
    """
    Stampa riepilogo di frontiere per il modello NT.

    Wrapper di ``aei_common.print_disk_boundaries`` con firma compatibile
    alle versioni precedenti. Per usare con altri modelli:

        from AEI_setups.aei_common import print_disk_boundaries
        print_disk_boundaries(disk_model, params)
    """
    _model = lambda r, **p: disk_model_NT(r, **p, alpha_visc=alpha, M=M)
    print_disk_boundaries(_model, {'a': a, 'B00': 1.0, 'Sigma0': Sigma0}, M=M)


def nt_scan_grid(Sigma0_vals, a_vals, alpha=ALPHA_VISC, M=M_BH):
    """
    Tabella diagnostica su griglia (Σ₀, a) per il modello NT.

    Wrapper di ``aei_common.scan_disk_grid`` con firma compatibile alle
    versioni precedenti. Per usare con altri modelli:

        from AEI_setups.aei_common import scan_disk_grid
        scan_disk_grid(disk_model, Sigma0_vals, a_vals, extra_params=...)
    """
    _model = lambda r, **p: disk_model_NT(r, **p, alpha_visc=alpha, M=M)
    return scan_disk_grid(_model, Sigma0_vals, a_vals,
                          extra_params={'B00': 1.0}, M=M)