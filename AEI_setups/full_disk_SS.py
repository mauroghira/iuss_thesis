"""
full_disk_SS.py
===============
Modello di disco Shakura-Sunyaev unificato per l'analisi AEI.

Struttura radiale a tre zone con raccordo continuo di B₀(r) e Σ(r):

  Zona A  [r_H,   r_AB]:  α_B = 3/4,  α_Σ = -3/2   (radiation-pressure dominated)
  Zona B  [r_AB,  r_BC]:  α_B = 5/4,  α_Σ =  3/5   (gas-pressure + e-scattering)
  Zona C  [r_BC,  ∞   ):  α_B = 5/4,  α_Σ =  3/4   (gas-pressure + free-free)

Parametri liberi globali:
  B00    — valore di B₀ all'orizzonte degli eventi r_H
  Sigma0 — valore di Σ all'ISCO r_ISCO  (bordo interno del disco)
  a      — spin adimensionale del BH

Le normalizzazioni di zona B e C sono ricavate per continuità:
  B_A(r_AB) = B_B(r_AB)   →   B_norm_B
  B_B(r_BC) = B_C(r_BC)   →   B_norm_C
  (idem per Σ)

Le frontiere r_AB e r_BC seguono le formule S&S (1973) in unità di r_g,
con Ṁ/Ṁ_Edd ottenuto da B00 come proxy via equilibrio di pressione magnetica.

Dipendenze: numpy, pandas, matplotlib  (già usate nel notebook)
            setup.py  (r_isco, r_horizon, nu_phi, nu_r, Rg_SUN, M_BH, NU0)
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

# Esponenti delle tre zone
ZONE_ALPHA = {
    'A': dict(alpha_B=3/4,  alpha_S=-3/2),
    'B': dict(alpha_B=5/4,  alpha_S= 3/5),
    'C': dict(alpha_B=5/4,  alpha_S= 3/4),
}
ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = {'A': '#f97316', 'B': '#3b82f6', 'C': '#22c55e'}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FRONTIERE S&S
# ═══════════════════════════════════════════════════════════════════════════════

def ss_mdot_from_Sigma0(Sigma0, a, alpha=0.1):
    """
    Inverte u₀_A(r_ISCO) = Sigma0 con la formula SS pura (eq. 2.9 S&S 1973):
        u₀ = 4.6 α⁻¹ ṁ⁻¹ r^{3/2} (1 - r^{-1/2})⁻¹

    Invertendo a r_ISCO:
        ṁ_SS = 4.6 · r_ISCO^{3/2} · (1 - r_ISCO^{-1/2})⁻¹ / (α · Σ₀)

    Solo il fattore non-relativistico f = (1 - r^{-1/2}), senza A,B,C,D,E,Q NT.
    """
    rISCO = float(r_isco(a))
    f_isco = max(1.0 - rISCO**(-0.5), 1e-10)
    return float(4.6 * rISCO**(3/2) / (f_isco * alpha * float(Sigma0)))


def _bisect_ss(func, r_lo, r_hi, n_scan=300, n_bisect=50):
    """Bisezione scalare: trova r dove func(r) = 0."""
    r_arr = np.geomspace(r_lo, r_hi, n_scan)
    f_arr = func(r_arr)
    sc = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sc) == 0:
        return None
    a_r, b_r = r_arr[sc[0]], r_arr[sc[0] + 1]
    for _ in range(n_bisect):
        mid = (a_r + b_r) / 2
        if func(np.array([mid]))[0] < 0:
            a_r = mid
        else:
            b_r = mid
    return float((a_r + b_r) / 2)


def ss_boundaries(a, Sigma0, alpha=0.1, M=M_BH):
    """
    Frontiere r_AB e r_BC con le formule S&S 1973 PURE (non-relativistiche).

    ṁ ricavato da Σ₀ tramite la formula SS zona A (eq. 2.9):
        u₀ = 4.6 α⁻¹ ṁ⁻¹ r^{3/2} (1 - r^{-1/2})⁻¹
        → ṁ_SS = 4.6 r_ISCO^{3/2} (1 - r_ISCO^{-1/2})⁻¹ / (α Σ₀)

    r_AB (eq. 2.17):  r_ab / (1 - r_ab^{-1/2})^{16/21} = 150 (αm)^{2/21} ṁ^{16/21}
    r_bc (eq. 2.20):  r_bc = 6.3×10³ ṁ^{2/3} (1 - r_bc^{-1/2})^{2/3}

    Entrambe le equazioni sono implicite in r (via f = 1-r^{-1/2}) e si risolvono
    con bisezione semplice — nessun fattore relativistico NT.

    B₀₀ NON entra nelle frontiere — è un parametro libero del campo.

    Parameters
    ----------
    a      : float   spin adimensionale (usato solo per r_ISCO nel calcolo di ṁ)
    Sigma0 : float   Σ₀ in g/cm²  (determina ṁ)
    alpha  : float   viscosità α (default 0.1)
    M      : float   massa BH in M_sun

    Returns
    -------
    r_AB, r_BC : float   raggi di frontiera in r_g
    mdot       : float   ṁ derivato (formula SS pura)
    """
    rISCO = float(r_isco(a))
    m     = float(M)
    mdot  = ss_mdot_from_Sigma0(Sigma0, a, alpha)

    def _f(r):
        return np.maximum(1.0 - np.asarray(r, float)**(-0.5), 1e-10)

    # ── r_AB: eq. 2.17  →  r / f(r)^{16/21} - 150 (αm)^{2/21} ṁ^{16/21} = 0 ──
    rhs_AB = 150.0 * (alpha * m)**(2/21) * mdot**(16/21)

    def eq_AB(r):
        r = np.asarray(r, float)
        return r / _f(r)**(16/21) - rhs_AB

    # controlla se zona A esiste (se eq_AB(r_ISCO) ≥ 0 → r_AB collassato a r_ISCO)
    if eq_AB(np.array([rISCO * 1.01]))[0] >= 0:
        r_AB = rISCO
    else:
        r_AB_hi = float(np.clip(rhs_AB * 2.0, 500.0, 1e5))
        r_AB = _bisect_ss(eq_AB, rISCO * 1.01, r_AB_hi)
        if r_AB is None:
            r_AB = r_AB_hi
        r_AB = float(np.clip(r_AB, rISCO, 1e5))

    # ── r_BC: eq. 2.20  →  r / f(r)^{2/3} - 6.3e3 · ṁ^{2/3} = 0 ──────────────
    rhs_BC = 6.3e3 * mdot**(2/3)

    def eq_BC(r):
        r = np.asarray(r, float)
        return r / _f(r)**(2/3) - rhs_BC

    r_BC = _bisect_ss(eq_BC, r_AB * 1.01, 1e5)
    if r_BC is None:
        f_at_rAB = float(eq_BC(np.array([r_AB * 1.01]))[0])
        r_BC = r_AB * 1.01 if f_at_rAB >= 0 else 1e5
    r_BC = float(np.clip(r_BC, r_AB, 1e5))

    return r_AB, r_BC, mdot


def zone_index(r, r_AB, r_BC):
    """
    Restituisce l'indice di zona (0=A, 1=B, 2=C) per ogni r.
    Funziona su scalari e array.
    """
    r = np.asarray(r)
    idx = np.zeros_like(r, dtype=int)
    idx[r > r_AB] = 1
    idx[r > r_BC] = 2
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  NORMALIZZAZIONI PER CONTINUITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def compute_norms(a, B00, Sigma0, r_AB, r_BC):
    """
    Calcola le normalizzazioni di B e Σ per le zone B e C
    imponendo la continuità alle frontiere.

    Ogni zona usa:
        B_z(r)  = B_norm_z  * (r / r_ref_z)^(-alpha_B_z)
        Σ_z(r)  = S_norm_z  * (r / r_ref_z)^(-alpha_S_z)

    Zona A:  r_ref = r_H (per B),  r_ISCO (per Σ)
    Zona B:  r_ref = r_AB  (raccordo con A)
    Zona C:  r_ref = r_BC  (raccordo con B)

    Returns
    -------
    norms : dict con chiavi 'A','B','C', ciascuna con
            B_norm, S_norm, r_ref_B, r_ref_S
    """
    rH    = r_horizon(a)
    rISCO = r_isco(a)
    aA, aB, aC = ZONE_ALPHA['A'], ZONE_ALPHA['B'], ZONE_ALPHA['C']

    # Zona A — normalizzazioni dai parametri liberi
    norms = {
        'A': dict(B_norm=B00,    S_norm=Sigma0, r_ref_B=rH,    r_ref_S=rISCO),
    }

    # Valore alla frontiera AB
    B_A_rAB = B00    * (r_AB / rH   )**(-aA['alpha_B'])
    S_A_rAB = Sigma0 * (r_AB / rISCO)**(-aA['alpha_S'])

    # Zona B — raccordo a r_AB
    norms['B'] = dict(B_norm=B_A_rAB, S_norm=S_A_rAB, r_ref_B=r_AB, r_ref_S=r_AB)

    # Valore alle frontiera BC
    B_B_rBC = B_A_rAB * (r_BC / r_AB)**(-aB['alpha_B'])
    S_B_rBC = S_A_rAB * (r_BC / r_AB)**(-aB['alpha_S'])

    # Zona C — raccordo a r_BC
    norms['C'] = dict(B_norm=B_B_rBC, S_norm=S_B_rBC, r_ref_B=r_BC, r_ref_S=r_BC)

    return norms


def check_continuity(norms, r_AB, r_BC, tol=1e-3):
    """
    Verifica numerica del raccordo alle frontiere.
    Stampa ΔB/B e ΔΣ/Σ a r_AB e r_BC.
    """
    dr = 1e-4
    for boundary, zF, zT, r0 in [('r_AB', 'A', 'B', r_AB), ('r_BC', 'B', 'C', r_BC)]:
        BF = B0_profile_full(r0 - dr, zF, norms)
        BT = B0_profile_full(r0 + dr, zT, norms)
        SF = Sigma_profile_full(r0 - dr, zF, norms)
        ST = Sigma_profile_full(r0 + dr, zT, norms)
        dB = abs(BF - BT) / (BF + 1e-300)
        dS = abs(SF - ST) / (SF + 1e-300)
        ok_B = "✓" if dB < tol else "✗"
        ok_S = "✓" if dS < tol else "✗"
        print(f"  {boundary} = {r0:.2f} rg  |  "
              f"ΔB/B = {dB:.2e} {ok_B}  |  ΔΣ/Σ = {dS:.2e} {ok_S}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PROFILI FISICI
# ═══════════════════════════════════════════════════════════════════════════════

def B0_profile_full(r, zone, norms):
    """
    B₀(r) per una zona specificata (scalare o array della stessa zona).
    """
    n  = norms[zone]
    aB = ZONE_ALPHA[zone]['alpha_B']
    return n['B_norm'] * (r / n['r_ref_B'])**(-aB)


def Sigma_profile_full(r, zone, norms):
    """Σ(r) per una zona specificata."""
    n  = norms[zone]
    aS = ZONE_ALPHA[zone]['alpha_S']
    return n['S_norm'] * (r / n['r_ref_S'])**(-aS)


def B0_disk(r, norms, r_AB, r_BC):
    """B₀(r) sull'intero disco (array-safe)."""
    r   = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    idx = zone_index(r, r_AB, r_BC)
    for i, zn in enumerate(ZONE_NAMES):
        mask = idx == i
        if np.any(mask):
            out[mask] = B0_profile_full(r[mask], zn, norms)
    return out


def Sigma_disk(r, norms, r_AB, r_BC):
    """Σ(r) sull'intero disco (array-safe)."""
    r   = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    idx = zone_index(r, r_AB, r_BC)
    for i, zn in enumerate(ZONE_NAMES):
        mask = idx == i
        if np.any(mask):
            out[mask] = Sigma_profile_full(r[mask], zn, norms)
    return out


def sound_speed_disk(r, a, hr, M=M_BH):
    """c_s(r) = (H/r) * r * Ω_φ * r_g — identico a sound_speed_thin nel notebook."""
    r   = np.asarray(r, dtype=float)
    Rg  = Rg_SUN * M
    return hr * 2 * np.pi * nu_phi(r, a, M) * r * Rg


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ADAPTER PER FZ UNIVERSASLI
# ═══════════════════════════════════════════════════════════════════════════════
def disk_model_SS(r_rg, a, B00, Sigma0, alpha_visc=ALPHA_VISC, M=M_BH, hr=HOR):
    r_AB, r_BC, mdot = ss_boundaries(a, Sigma0, alpha=alpha_visc, M=M)
    norms            = compute_norms(a, B00, Sigma0, r_AB, r_BC)
    B0    = B0_disk(r_rg, norms, r_AB, r_BC)
    Sigma = Sigma_disk(r_rg, norms, r_AB, r_BC)
    c_s   = sound_speed_disk(r_rg, a, hr, M)
    zi    = zone_index(r_rg, r_AB, r_BC)
    zone  = np.array([ZONE_NAMES[i] for i in zi])
    info  = {'r_AB': r_AB, 'r_BC': r_BC, 'mdot': mdot, 'alpha': alpha_visc}
    return B0, Sigma, c_s, zone, info