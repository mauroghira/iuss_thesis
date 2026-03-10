"""
full_disk_SS.py
===============
Modello di disco Shakura-Sunyaev 1973 puro per l'analisi AEI.

Struttura radiale a tre zone (S&S 1973, Tab. 1 / eqs. 2.8–2.19):

  Zona A  [r_ISCO, r_AB]:  P_rad >> P_gas,  opacità e-scattering
  Zona B  [r_AB,   r_BC]:  P_gas >> P_rad,  opacità e-scattering
  Zona C  [r_BC,   ∞   ):  P_gas >> P_rad,  opacità free-free

Parametri liberi globali:
  B00    — valore di B₀ all'orizzonte degli eventi r_H   [Gauss]
  Sigma0 — valore di Σ all'ISCO r_ISCO                   [g/cm²]
  a      — spin adimensionale del BH

Profili fisici (formule S&S 1973 esatte, fattore f = 1 - r^{-1/2}):

  Σ(r):
    zona A: u₀  = 4.6  α⁻¹   ṁ⁻¹   r^{3/2}  f⁻¹           (eq. 2.9)
    zona B: u₀  = 1.7e5 α⁻⁴/⁵ ṁ^{3/5} m^{1/5} r^{-3/5} f^{3/5}  (eq. 2.16)
    zona C: u₀  = 6.1e3 α⁻⁴/⁵ ṁ^{7/10} m^{1/4} r^{-3/4} f^{7/10} (eq. 2.19)

  B₀(r)  (da H = z₀ · sqrt(4π/3 · α · ε), con ε ∝ u₀):
    zona A: H   = 1e8   m^{-1/2} r^{-3/4}                    (eq. 2.11)
    zona B: H   = 1.5e9 α^{1/20} ṁ^{2/5}  m^{-9/20} r^{-51/40} f^{2/5}  (eq. 2.16)
    zona C: H   = 2.1e9 α^{1/20} ṁ^{17/40} m^{-9/20} r^{-21/16} f^{17/40} (eq. 2.19)

  Raccordo: interpolazione log-lineare in log(r) su finestra ±5% delle frontiere.

  Normalizzazione assoluta di B: zona A è ancorata a B00 all'orizzonte.
  Zone B e C hanno un fattore di riscalatura ricavato dalla continuità a r_AB / r_BC.
  Σ zona A è ancorata a Sigma0 all'ISCO per costruzione (ṁ è invertito da lì).

Frontiere (S&S 1973, eqs. 2.17, 2.20) — NO fattori NT:
  r_AB: r / f^{16/21} = 150 (αm)^{2/21} ṁ^{16/21}
  r_BC: r / f^{2/3}   = 6.3e3 ṁ^{2/3}
  ṁ invertito da Σ_A(r_ISCO) = Sigma0  →  ṁ = 4.6 r_ISCO^{3/2} / (α Σ₀ f_ISCO)

Dipendenze: numpy, pandas, matplotlib
            setup.py  (r_isco, r_horizon, nu_phi, Rg_SUN, M_BH)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd

from .aei_common import (
    solve_k_aei, compute_beta, compute_dQdr, check_k_wkb, _make_interp,
    ALPHA_VISC, HOR, mm
)

import sys
sys.path.append("..")
from setup import (
    create_param_grid,
    r_isco, r_horizon, nu_phi, nu_r,
    Rg_SUN, M_BH, NU0, C
)

ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = {'A': '#f97316', 'B': '#3b82f6', 'C': '#22c55e'}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FRONTIERE S&S  (formule chiuse, bisezione semplice)
# ═══════════════════════════════════════════════════════════════════════════════

def _ss_f(r):
    """Fattore non-relativistico f = max(1 - r^{-1/2}, ε)."""
    return np.maximum(1.0 - np.asarray(r, float)**(-0.5), 1e-10)


def ss_mdot_from_Sigma0(Sigma0, a, alpha=0.1):
    """
    Inverte Σ_A(r_ISCO) = Sigma0  →  ṁ_SS  (formula S&S zona A, eq. 2.9).

        u₀_A = 4.6 α⁻¹ ṁ⁻¹ r^{3/2} f⁻¹
        ṁ    = 4.6 r_ISCO^{3/2} / (α · Σ₀ · f_ISCO)

    Solo il fattore non-relativistico f = (1 - r^{-1/2}), nessun termine NT.
    """
    rISCO  = float(r_isco(a))
    f_isco = float(_ss_f(np.array([rISCO]))[0])
    return float(4.6 * rISCO**(3/2) / (f_isco * alpha * float(Sigma0)))


def _bisect_ss(func, r_lo, r_hi, n_scan=300, n_bisect=50):
    """Bisezione scalare: trova r tale che func(r) = 0, con scansione iniziale."""
    r_arr = np.geomspace(r_lo, r_hi, n_scan)
    f_arr = func(r_arr)
    sc    = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sc) == 0:
        return None
    a_r, b_r = r_arr[sc[0]], r_arr[sc[0] + 1]
    for _ in range(n_bisect):
        mid = (a_r + b_r) / 2.0
        if func(np.array([mid]))[0] < 0:
            a_r = mid
        else:
            b_r = mid
    return float((a_r + b_r) / 2.0)


def ss_boundaries(a, Sigma0, alpha=0.1, M=M_BH):
    """
    Frontiere r_AB e r_BC  —  formule S&S 1973 PURE (no fattori NT).

    ṁ  ←  Σ_A(r_ISCO) = Sigma0  (eq. 2.9)
    r_AB  ←  r / f^{16/21} = 150 (αm)^{2/21} ṁ^{16/21}   (eq. 2.17)
    r_BC  ←  r / f^{2/3}   = 6.3e3 ṁ^{2/3}               (eq. 2.20)

    B00 non entra nelle frontiere.

    Returns
    -------
    r_AB, r_BC : float  [r_g]
    mdot       : float  ṁ = Ṁ / Ṁ_Edd
    """
    rISCO = float(r_isco(a))
    m     = float(M)
    mdot  = ss_mdot_from_Sigma0(Sigma0, a, alpha)

    # ── r_AB  (eq. 2.17) ────────────────────────────────────────────────────
    rhs_AB = 150.0 * (alpha * m)**(2/21) * mdot**(16/21)

    def eq_AB(r):
        r = np.asarray(r, float)
        return r / _ss_f(r)**(16/21) - rhs_AB

    if eq_AB(np.array([rISCO * 1.01]))[0] >= 0.0:
        # zona A assente: disco tutto gas-pressure dall'ISCO
        r_AB = rISCO
    else:
        r_AB_hi = float(np.clip(rhs_AB * 2.0, 500.0, 1e5))
        r_AB    = _bisect_ss(eq_AB, rISCO * 1.01, r_AB_hi) or r_AB_hi
        r_AB    = float(np.clip(r_AB, rISCO, 1e5))

    # ── r_BC  (eq. 2.20) ────────────────────────────────────────────────────
    rhs_BC = 6.3e3 * mdot**(2/3)

    def eq_BC(r):
        r = np.asarray(r, float)
        return r / _ss_f(r)**(2/3) - rhs_BC

    r_BC_raw = _bisect_ss(eq_BC, r_AB * 1.01, 1e5)
    if r_BC_raw is None:
        # crossing oltre 1e5 oppure f_BC(r_AB) ≥ 0  → zona B assente
        f_at_rAB = float(eq_BC(np.array([r_AB * 1.01]))[0])
        r_BC_raw = r_AB * 1.01 if f_at_rAB >= 0 else 1e5
    r_BC = float(np.clip(r_BC_raw, r_AB, 1e5))

    return r_AB, r_BC, mdot


def zone_index(r, r_AB, r_BC):
    """Indice di zona (0=A, 1=B, 2=C) per ogni r — scalare o array."""
    r   = np.asarray(r)
    idx = np.zeros_like(r, dtype=int)
    idx[r > r_AB] = 1
    idx[r > r_BC] = 2
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  RACCORDO LOG-LINEARE  (identico a ss_nt_boundaries)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_blend(r, r_c, f_lo, f_hi, width=0.05):
    """
    Interpolazione log-lineare in log(r) nella finestra [r_c(1-w), r_c(1+w)].

        log f_blend = (1-t) log f_lo + t log f_hi,   t ∈ [0,1] lineare in log r

    Continuo agli estremi, nessun overshoot, adatto a power-law.
    """
    r_lo = r_c * (1.0 - width)
    r_hi = r_c * (1.0 + width)
    t    = np.clip(
        (np.log(r) - np.log(r_lo)) / (np.log(r_hi) - np.log(r_lo)),
        0.0, 1.0
    )
    return np.exp(
        (1.0 - t) * np.log(np.maximum(f_lo, 1e-300))
        +       t  * np.log(np.maximum(f_hi, 1e-300))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PROFILI FISICI SS  —  Σ(r)
# ═══════════════════════════════════════════════════════════════════════════════

def _ss_Sigma_A(r, mdot, alpha):
    """
    Σ zona A  [g/cm²]  —  S&S 1973 eq. 2.9
        u₀ = 4.6 α⁻¹ ṁ⁻¹ r^{3/2} f⁻¹
    """
    r = np.asarray(r, float)
    return 4.6 * alpha**(-1) * mdot**(-1) * r**(3/2) * _ss_f(r)**(-1)


def _ss_Sigma_B(r, mdot, alpha, M):
    """
    Σ zona B  [g/cm²]  —  S&S 1973 eq. 2.16
        u₀ = 1.7e5 α⁻⁴/⁵ ṁ^{3/5} m^{1/5} r^{-3/5} f^{3/5}
    """
    r = np.asarray(r, float)
    return 1.7e5 * alpha**(-4/5) * mdot**(3/5) * float(M)**(1/5) * r**(-3/5) * _ss_f(r)**(3/5)


def _ss_Sigma_C(r, mdot, alpha, M):
    """
    Σ zona C  [g/cm²]  —  S&S 1973 eq. 2.19
        u₀ = 6.1e3 α⁻⁴/⁵ ṁ^{7/10} m^{1/4} r^{-3/4} f^{7/10}
    """
    r = np.asarray(r, float)
    return 6.1e3 * alpha**(-4/5) * mdot**(7/10) * float(M)**(1/4) * r**(-3/4) * _ss_f(r)**(7/10)


def _ss_Sigma_disk(r, mdot, alpha, M, r_AB, r_BC, blend_width=0.05):
    """
    Σ(r) sull'intero disco con raccordo log-lineare alle frontiere.

    Ogni zona usa la formula fisica SS 1973 senza alcun fattore di riscalatura.
    La continuità è garantita unicamente dalla finestra di blend ±blend_width
    attorno a r_AB e r_BC, identicamente a quanto fatto in ss_nt_boundaries.
    """
    r   = np.asarray(r, float)
    S_A = _ss_Sigma_A(r, mdot, alpha)
    S_B = _ss_Sigma_B(r, mdot, alpha, M)
    S_C = _ss_Sigma_C(r, mdot, alpha, M)

    S_AB = _log_blend(r, r_AB, S_A, S_B, width=blend_width)
    S_BC = _log_blend(r, r_BC, S_B, S_C, width=blend_width)

    w           = blend_width
    in_blend_AB = (r >= r_AB * (1.0 - w)) & (r <  r_AB * (1.0 + w))
    in_pure_B   = (r >= r_AB * (1.0 + w)) & (r <  r_BC * (1.0 - w))
    in_blend_BC = (r >= r_BC * (1.0 - w)) & (r <  r_BC * (1.0 + w))
    in_pure_C   =  r >= r_BC * (1.0 + w)

    result = S_A.copy()
    result[in_blend_AB] = S_AB[in_blend_AB]
    result[in_pure_B]   = S_B[in_pure_B]
    result[in_blend_BC] = S_BC[in_blend_BC]
    result[in_pure_C]   = S_C[in_pure_C]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PROFILI FISICI SS  —  B₀(r)
# ═══════════════════════════════════════════════════════════════════════════════

def _ss_B0_A(r, B00, a):
    """
    B₀ zona A  [G]  — power-law ancorata a B00 all'orizzonte.
        B₀_A(r) = B00 · (r / r_H)^{-3/4}

    Il prefattore fisico di H in zona A (eq. 2.11: H = 1e8 m^{-1/2} r^{-3/4})
    è indipendente da α e ṁ, quindi l'andamento radiale è una pura power-law r^{-3/4}.
    B00 è il parametro libero assoluto che fissa l'ampiezza.
    """
    r  = np.asarray(r, float)
    rH = float(r_horizon(a))
    return B00 * (r / rH)**(-3/4)


def _ss_B0_B_raw(r, mdot, alpha, M):
    """
    B₀ zona B  [G]  —  S&S 1973 eq. 2.16
        H = 1.5e9 α^{1/20} ṁ^{2/5} m^{-9/20} r^{-51/40} f^{2/5}
    """
    r = np.asarray(r, float)
    return 1.5e9 * alpha**(1/20) * mdot**(2/5) * float(M)**(-9/20) * r**(-51/40) * _ss_f(r)**(2/5)


def _ss_B0_C_raw(r, mdot, alpha, M):
    """
    B₀ zona C  [G]  —  S&S 1973 eq. 2.19
        H = 2.1e9 α^{1/20} ṁ^{17/40} m^{-9/20} r^{-21/16} f^{17/40}
    """
    r = np.asarray(r, float)
    return 2.1e9 * alpha**(1/20) * mdot**(17/40) * float(M)**(-9/20) * r**(-21/16) * _ss_f(r)**(17/40)


def _ss_B0_disk(r, a, B00, mdot, alpha, M, r_AB, r_BC, blend_width=0.05):
    """
    B₀(r) sull'intero disco con raccordo log-lineare alle frontiere.

    Ogni zona usa la formula fisica SS 1973 senza alcun fattore di riscalatura.
    La continuità è garantita unicamente dalla finestra di blend ±blend_width
    attorno a r_AB e r_BC.
    """
    r   = np.asarray(r, float)
    B_A = _ss_B0_A(r, B00, a)
    B_B = _ss_B0_B_raw(r, mdot, alpha, M)
    B_C = _ss_B0_C_raw(r, mdot, alpha, M)

    B_AB = _log_blend(r, r_AB, B_A, B_B, width=blend_width)
    B_BC = _log_blend(r, r_BC, B_B, B_C, width=blend_width)

    w           = blend_width
    in_blend_AB = (r >= r_AB * (1.0 - w)) & (r <  r_AB * (1.0 + w))
    in_pure_B   = (r >= r_AB * (1.0 + w)) & (r <  r_BC * (1.0 - w))
    in_blend_BC = (r >= r_BC * (1.0 - w)) & (r <  r_BC * (1.0 + w))
    in_pure_C   =  r >= r_BC * (1.0 + w)

    result = B_A.copy()
    result[in_blend_AB] = B_AB[in_blend_AB]
    result[in_pure_B]   = B_B[in_pure_B]
    result[in_blend_BC] = B_BC[in_blend_BC]
    result[in_pure_C]   = B_C[in_pure_C]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VELOCITÀ DEL SUONO
# ═══════════════════════════════════════════════════════════════════════════════

def sound_speed_disk(r, a, hr, M=M_BH):
    """c_s(r) = (H/r) · r · Ω_φ · r_g  — thin-disc approximation."""
    r  = np.asarray(r, dtype=float)
    Rg = Rg_SUN * M
    return hr * 2 * np.pi * nu_phi(r, a, M) * r * Rg


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  DIAGNOSTICA  —  check continuità
# ═══════════════════════════════════════════════════════════════════════════════

def check_continuity_ss(a, B00, Sigma0, alpha=ALPHA_VISC, M=M_BH, tol=1e-3):
    """
    Verifica numerica del raccordo alle frontiere r_AB e r_BC.
    Stampa ΔB/B e ΔΣ/Σ valutati appena dentro/fuori ogni frontiera.
    """
    r_AB, r_BC, mdot = ss_boundaries(a, Sigma0, alpha=alpha, M=M)
    eps = 1e-3

    for label, r0 in [('r_AB', r_AB), ('r_BC', r_BC)]:
        r_in  = np.array([r0 * (1.0 - eps)])
        r_out = np.array([r0 * (1.0 + eps)])
        B_in  = float(_ss_B0_disk(r_in,  a, B00, mdot, alpha, M, r_AB, r_BC)[0])
        B_out = float(_ss_B0_disk(r_out, a, B00, mdot, alpha, M, r_AB, r_BC)[0])
        S_in  = float(_ss_Sigma_disk(r_in,  mdot, alpha, M, r_AB, r_BC)[0])
        S_out = float(_ss_Sigma_disk(r_out, mdot, alpha, M, r_AB, r_BC)[0])
        dB = abs(B_in - B_out) / (B_in + 1e-300)
        dS = abs(S_in - S_out) / (S_in + 1e-300)
        print(f"  {label} = {r0:.2f} rg  |  "
              f"ΔB/B = {dB:.2e} {'✓' if dB < tol else '✗'}  |  "
              f"ΔΣ/Σ = {dS:.2e} {'✓' if dS < tol else '✗'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ADAPTER PER find_rossby  (firma standard del pacchetto)
# ═══════════════════════════════════════════════════════════════════════════════

def disk_model_SS(r_rg, a, B00, Sigma0, alpha_visc=ALPHA_VISC, M=M_BH, hr=HOR,
                  blend_width=0.05):
    """
    Adapter per aei_common.find_rossby — firma standard:

        B0, Sigma, c_s, zone, info = disk_model_SS(r_rg, **row)

    Usa le formule fisiche SS 1973 per Σ e B₀, con raccordo log-lineare.
    """
    r_rg = np.asarray(r_rg, float)
    r_AB, r_BC, mdot = ss_boundaries(a, Sigma0, alpha=alpha_visc, M=M)

    B0    = _ss_B0_disk(r_rg, a, B00, mdot, alpha_visc, M, r_AB, r_BC,
                        blend_width=blend_width)
    Sigma = _ss_Sigma_disk(r_rg, mdot, alpha_visc, M, r_AB, r_BC,
                           blend_width=blend_width)
    c_s   = sound_speed_disk(r_rg, a, hr, M)
    zi    = zone_index(r_rg, r_AB, r_BC)
    zone  = np.array([ZONE_NAMES[i] for i in zi])
    info  = {'r_AB': r_AB, 'r_BC': r_BC, 'mdot': mdot, 'alpha': alpha_visc}
    return B0, Sigma, c_s, zone, info