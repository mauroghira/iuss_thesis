"""
ss_nt_boundaries.py
===================
Calcolo delle frontiere S&S con correzioni relativistiche di Novikov-Thorne,
usando le formule ESATTE della review di Abramowicz (sezioni 5.3, pagine 28-29).

FISICA:
  - ṁ ricavato da Σ₀ tramite la formula ESATTA della zona A (inner region)
  - B₀₀ è un parametro libero indipendente (non entra nelle frontiere)
  - α è parametro libero (default 0.1)
  - r_AB: dove β/(1-β) = 1  →  P_rad = P_gas  (da zona A, eq. 99)
  - r_BC: dove τ_ff/τ_es = 1  →  cambio opacità  (da zona C, eq. 97)

CONVENZIONI (dal paper):
  r* = r·c²/GM = r  [in unità di r_g = GM/c²]
  m  = M/M_sun
  ṁ  = Ṁc²/L_Edd  (tasso di accrescimento adimensionale)
  y  = (r/M)^{1/2} = sqrt(r*)  [con M=1 in unità r_g]
  a* = a  (spin adimensionale, = a/M con M=1)
  y0 = r_ms^{1/2} = sqrt(r_ISCO)

FATTORI RELATIVISTICI (eq. 237 del paper):
  A  = 1 + a*² y⁻⁴ + 2a*² y⁻⁶
  B  = 1 + a* y⁻³
  C  = 1 - 3y⁻² + 2a* y⁻³
  D  = 1 - 2y⁻² + a*² y⁻⁴
  E  = 1 + 4a*² y⁻⁴ - 4a*² y⁻⁶ + 3a*⁴ y⁻⁸
  Q0 = (1 + a* y⁻³) / [y (1 - 3y⁻² + 2a* y⁻³)^{1/2}]
  Q  = integrale di Novikov-Thorne (formula esplicita dal paper)

FORMULE USATE:
  Zona A (inner, P_rad, κ_es):
    Σ_A = 5        [g/cm²]  · α⁻¹  ṁ⁻¹  r*^{3/2}  A⁻²B³C^{1/2}EQ⁻¹
    β/(1-β)_A = 4e-6        · α⁻¹/⁴ m⁻¹/⁴ ṁ⁻²  r*^{21/8}  A⁻⁵/²B^{9/2}DE^{5/4}Q⁻²
    (τ_ff τ_es)_A = 1e-4    · α⁻¹⁷/¹⁶ m⁻¹/¹⁶ ṁ⁻² r*^{93/32} A⁻²⁵/⁸B^{41/8}C^{1/2}D^{1/2}E^{25/16}Q⁻²

  Zona B (middle, P_gas, κ_es):
    Σ_B = 9e4      [g/cm²]  · α⁻⁴/⁵ m^{1/5} ṁ^{3/5} r*^{-3/5} B⁻⁴/⁵C^{1/2}D⁻⁴/⁵Q^{3/5}  [NOTE: m^0=1 here]
    τ_ff/τ_es_B = 2e-6      · ṁ⁻¹  r*^{3/2}  A⁻¹B²D^{1/2}E^{1/2}Q⁻¹

  Zona C (outer, P_gas, κ_ff):
    Σ_C = 4e5      [g/cm²]  · α⁻⁴/⁵ m^{2/10} ṁ_0*^{7/10} r*^{-3/4} A^{1/10}B⁻⁴/⁵C^{1/2}D⁻¹⁷/²⁰E⁻¹/²⁰Q^{7/10}
    τ_ff/τ_es_C = 2e-3      · ṁ⁻¹/² r*^{3/4}  A⁻¹/²B^{2/5}D^{1/4}E^{1/4}Q⁻¹/²

  Nota: r_AB si trova con β/(1-β)_A = 1
        r_BC si trova con τ_ff/τ_es_B = 1 (zona B, eq. 98)
        In alternativa con τ_ff/τ_es_C = 1 (zona C, eq. 97) — devono coincidere

Dipendenze: numpy, setup.py
"""

import numpy as np
from functools import lru_cache

import sys
sys.path.append("..")
from setup import r_isco, r_horizon, M_BH

ALPHA_DEFAULT = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  COEFFICIENTI DELLE RADICI  (cache per spin fisso)
# ═══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _nt_spin_constants(a):
    """
    Pre-calcola tutte le costanti che dipendono solo da a:
      y0, y1, y2, y3, c1, c2, c3
    Cache LRU: con 19 valori di spin vengono calcolate una volta sola.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FATTORI RELATIVISTICI — completamente vettorizzati
# ═══════════════════════════════════════════════════════════════════════════════

def nt_ABCDEQ(r, a):
    """
    Fattori A,B,C,D,E,Q0,Q di Novikov-Thorne su array r (zero loop Python).

    r  : float o ndarray  — raggio in r_g
    a  : float            — spin adimensionale
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

    # costanti cached
    y0, y1, y2, y3, c1, c2, c3 = _nt_spin_constants(as_)

    sl = lambda x: np.log(np.maximum(np.abs(x), 1e-300))

    term0 = y - y0 - 1.5*as_*sl(y / y0)
    t1    = c1 * sl((y - y1) / (y0 - y1))
    t2    = c2 * sl((y - y2) / (y0 - y2))
    t3    = c3 * sl((y - y3) / (y0 - y3))

    Q_raw = Q0 * (term0 - t1 - t2 - t3)
    Q     = np.where(y > y0*1.001, np.maximum(Q_raw, 1e-10), 1e-10)

    if r.ndim == 0:
        Q = float(Q)

    return dict(A=A, B=B, C=C, D=D, E=E, Q0=Q0, Q=Q)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FORMULE ESPLICITE ZONE A, B, C
# ═══════════════════════════════════════════════════════════════════════════════

def nt_Sigma_A(r, a, mdot, alpha):
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-2) * f['B']**3 * np.sqrt(np.maximum(f['C'],0)) * f['E'] / f['Q']
    return 5.0 * alpha**(-1) * mdot**(-1) * np.asarray(r)**(3/2) * rel

def nt_Sigma_B(r, a, mdot, alpha):
    f = nt_ABCDEQ(r, a)
    rel = f['B']**(-4/5) * np.sqrt(np.maximum(f['C'],0)) * f['D']**(-4/5) * f['Q']**(3/5)
    return 9e4 * alpha**(-4/5) * mdot**(3/5) * np.asarray(r)**(-3/5) * rel

def nt_Sigma_C(r, a, mdot, alpha, m):
    f = nt_ABCDEQ(r, a)
    rel = (f['A']**(1/10) * f['B']**(-4/5) * np.sqrt(np.maximum(f['C'],0))
           * f['D']**(-17/20) * f['E']**(-1/20) * f['Q']**(7/10))
    return 4e5 * alpha**(-4/5) * m**(1/5) * mdot**(7/10) * np.asarray(r)**(-3/4) * rel

def nt_beta_ratio_A(r, a, mdot, alpha, m):
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-5/2) * f['B']**(9/2) * f['D'] * f['E']**(5/4) / f['Q']**2
    return 4e-6 * alpha**(-1/4) * m**(-1/4) * mdot**(-2) * np.asarray(r)**(21/8) * rel

def nt_tau_ratio_B(r, a, mdot):
    f = nt_ABCDEQ(r, a)
    rel = (f['A']**(-1) * f['B']**2
           * np.sqrt(np.maximum(f['D'],0)) * np.sqrt(np.maximum(f['E'],0)) / f['Q'])
    return 2e-6 * mdot**(-1) * np.asarray(r)**(3/2) * rel

def nt_tau_ratio_C(r, a, mdot):
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-1/2) * f['B']**(2/5) * f['D']**(1/4) * f['E']**(1/4) / f['Q']**(1/2)
    return 2e-3 * mdot**(-1/2) * np.asarray(r)**(3/4) * rel


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  INVERSIONE Σ₀ → ṁ
# ═══════════════════════════════════════════════════════════════════════════════

def nt_mdot_from_Sigma0(Sigma0, a, alpha=ALPHA_DEFAULT):
    """ṁ dalla formula zona A valutata a r_ISCO."""
    rISCO = float(r_isco(a))
    f     = nt_ABCDEQ(rISCO, a)
    rel_A = (f['A']**(-2) * f['B']**3
             * np.sqrt(max(float(np.squeeze(f['C'])), 0))
             * f['E'] / f['Q'])
    mdot  = 5.0 * rISCO**(3/2) * float(rel_A) / (alpha * Sigma0)
    return float(np.clip(mdot, 1e-4, 50.0))


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  RICERCA RADICE — vettorizzata sul grid scan
# ═══════════════════════════════════════════════════════════════════════════════

def _bisect_vec(func_vec, r_lo, r_hi, n_scan=200, n_bisect=40):
    """
    Trova r in [r_lo, r_hi] dove func(r) = 1.
    func_vec deve accettare un array numpy e restituire un array.
    Più veloce di _bisect scalare perché valuta func sull'intero array in una volta.
    """
    r_arr = np.geomspace(r_lo, r_hi, n_scan)
    f_arr = func_vec(r_arr) - 1.0

    sign_ch = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sign_ch) == 0:
        return None

    a_r = r_arr[sign_ch[0]]
    b_r = r_arr[sign_ch[0] + 1]
    for _ in range(n_bisect):
        mid = (a_r + b_r) / 2
        fm  = func_vec(np.array([mid]))[0] - 1.0
        if fm < 0:
            a_r = mid
        else:
            b_r = mid
    return (a_r + b_r) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FRONTIERE r_AB e r_BC
# ═══════════════════════════════════════════════════════════════════════════════

def nt_boundaries(a, Sigma0, alpha=ALPHA_DEFAULT, M=M_BH):
    """
    Frontiere S&S relativistiche.

    r_AB : β/(1-β) = 1  (zona A, P_rad = P_gas)
    r_BC : τ_ff/τ_es = 1  (zona B, cambio opacità)

    ṁ ricavato da Σ₀ via formula esatta zona A.
    B₀₀ non entra nelle frontiere.

    Returns: mdot, r_AB, r_BC
    """
    rISCO = float(r_isco(a))
    m     = float(M)

    mdot  = nt_mdot_from_Sigma0(Sigma0, a, alpha)

    # r_AB — valuta beta_ratio su tutto l'array in una chiamata
    f_AB  = lambda r_arr: nt_beta_ratio_A(r_arr, a, mdot, alpha, m)
    r_AB  = _bisect_vec(f_AB, rISCO*1.01, 300.0)
    if r_AB is None:
        r_AB = rISCO*1.5 if f_AB(np.array([rISCO*1.01]))[0] > 1 else 150.0
    r_AB  = float(np.clip(r_AB, rISCO*1.1, 300.0))

    # r_BC — valuta tau_ratio su tutto l'array in una chiamata
    f_BC  = lambda r_arr: nt_tau_ratio_B(r_arr, a, mdot)
    r_BC  = _bisect_vec(f_BC, r_AB*1.01, 1e5)
    if r_BC is None:
        r_BC = r_AB*2.0 if f_BC(np.array([r_AB*1.01]))[0] > 1 else 1000.0
    r_BC  = float(np.clip(r_BC, r_AB*1.2, 1e5))

    return mdot, r_AB, r_BC


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  DIAGNOSTICA
# ═══════════════════════════════════════════════════════════════════════════════

def nt_print_boundaries(a, Sigma0, alpha=ALPHA_DEFAULT, M=M_BH):
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha, M)
    tau_C = float(nt_tau_ratio_C(r_BC, a, mdot))
    print(f"\n{'='*60}")
    print(f"  a={a:.2f}  Σ₀={Sigma0:.1e}  α={alpha:.2f}  M={M:.1e} M☉")
    print(f"{'='*60}")
    print(f"  ṁ      = {mdot:.4f}")
    print(f"  r_H    = {rH:.3f}  r_g")
    print(f"  r_ISCO = {rISCO:.3f}  r_g")
    print(f"  r_AB   = {r_AB:.2f}  r_g  ({r_AB/rISCO:.1f} × r_ISCO)")
    print(f"  r_BC   = {r_BC:.1f}  r_g  ({r_BC/r_AB:.1f} × r_AB)")
    print(f"  check τ_ff/τ_es (zona C) @ r_BC = {tau_C:.3f}  [atteso ≈ 1]")
    print(f"{'='*60}")


def nt_scan_grid(Sigma0_vals, a_vals, alpha=ALPHA_DEFAULT, M=M_BH):
    import pandas as pd
    rows = []
    for S0 in Sigma0_vals:
        for a_val in a_vals:
            try:
                mdot, r_AB, r_BC = nt_boundaries(a_val, S0, alpha, M)
                rISCO = float(r_isco(a_val))
                rows.append(dict(Sigma0=S0, a=a_val, mdot=mdot,
                                 r_AB=r_AB, r_BC=r_BC,
                                 r_AB_rISCO=r_AB/rISCO, r_BC_rAB=r_BC/r_AB))
            except Exception:
                rows.append(dict(Sigma0=S0, a=a_val, mdot=np.nan,
                                 r_AB=np.nan, r_BC=np.nan,
                                 r_AB_rISCO=np.nan, r_BC_rAB=np.nan))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='{:.3g}'.format))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import time

    cases = [(a, S0)
             for a  in np.linspace(-0.9, 0.9, 19)
             for S0 in np.geomspace(1e3, 1e7, 20)]

    print(f"Benchmark su {len(cases)} chiamate (19 spin × 20 Sigma0)...")
    t0 = time.time()
    for a_v, S0 in cases:
        nt_boundaries(a_v, S0, alpha=0.1)
    t1 = time.time()
    dt = t1 - t0
    print(f"  Tempo totale:  {dt:.2f} s")
    print(f"  Per chiamata:  {dt/len(cases)*1000:.2f} ms")
    # griglia completa: 19 spin × 24 B00 × 20 Sigma0 = 9120 chiamate
    print(f"  Stima griglia completa (9120 calls): {dt/len(cases)*9120:.1f} s")

    print("\n=== Test singolo ===")
    nt_print_boundaries(a=0.5, Sigma0=1e5, alpha=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  DIAGNOSTICA
# ═══════════════════════════════════════════════════════════════════════════════

def nt_print_boundaries(a, Sigma0, alpha=ALPHA_DEFAULT, M=M_BH):
    """Stampa un riassunto delle frontiere e del ṁ derivato."""
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha, M)

    # verifica: τ_ff/τ_es dalla zona C a r_BC (deve essere ≈ 1)
    tau_C_at_rBC = nt_tau_ratio_C(r_BC, a, mdot)

    print(f"\n{'='*60}")
    print(f"  NT boundaries  |  a={a:.2f}  Σ₀={Sigma0:.1e}"
          f"  α={alpha:.2f}  M={M:.1e} M☉")
    print(f"{'='*60}")
    print(f"  ṁ  = {mdot:.4f}   (da Σ₀ zona A)")
    print(f"  r_H    = {rH:.3f}  r_g")
    print(f"  r_ISCO = {rISCO:.3f}  r_g")
    print(f"  r_AB   = {r_AB:.2f}  r_g  ({r_AB/rISCO:.1f} × r_ISCO)")
    print(f"  r_BC   = {r_BC:.1f}  r_g  ({r_BC/r_AB:.1f} × r_AB)")
    print(f"  verifica: τ_ff/τ_es (zona C) a r_BC = {tau_C_at_rBC:.3f}"
          f"  (atteso ≈ 1)")
    print(f"{'='*60}")


def nt_scan_grid(Sigma0_vals, a_vals, alpha=ALPHA_DEFAULT, M=M_BH):
    """
    Tabella di ṁ, r_AB, r_BC su una griglia di (Σ₀, a).
    Utile per verificare la sensibilità fisica delle frontiere.

    Returns
    -------
    df : pandas.DataFrame
    """
    import pandas as pd
    rows = []
    for S0 in Sigma0_vals:
        for a_val in a_vals:
            try:
                mdot, r_AB, r_BC = nt_boundaries(a_val, S0, alpha, M)
                rISCO = float(r_isco(a_val))
                rows.append(dict(
                    Sigma0=S0, a=a_val, mdot=mdot,
                    r_AB=r_AB, r_BC=r_BC,
                    r_AB_rISCO=r_AB / rISCO,
                    r_BC_rAB=r_BC / r_AB,
                ))
            except Exception as e:
                rows.append(dict(
                    Sigma0=S0, a=a_val, mdot=np.nan,
                    r_AB=np.nan, r_BC=np.nan,
                    r_AB_rISCO=np.nan, r_BC_rAB=np.nan,
                ))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='{:.3g}'.format))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  CELLE PER IL NOTEBOOK
# ═══════════════════════════════════════════════════════════════════════════════
"""
# ── import ───────────────────────────────────────────────────────────────────
from ss_nt_boundaries import (
    nt_boundaries, nt_mdot_from_Sigma0,
    nt_print_boundaries, nt_scan_grid,
    nt_Sigma_A, nt_Sigma_B, nt_Sigma_C,
    nt_beta_ratio_A, nt_tau_ratio_B, nt_tau_ratio_C,
    nt_ABCDEQ,
)

# ── test singolo punto ───────────────────────────────────────────────────────
nt_print_boundaries(a=0.5, Sigma0=1e5, alpha=0.1)
nt_print_boundaries(a=0.9, Sigma0=1e4, alpha=0.1)

# ── scan sulla griglia ───────────────────────────────────────────────────────
df_nt = nt_scan_grid(
    Sigma0_vals=[1e3, 1e4, 1e5, 1e6, 1e7],
    a_vals=[-0.9, 0.0, 0.5, 0.9],
    alpha=0.1,
)

# ── confronto con ss_boundaries_v2 (proxy vecchio) ──────────────────────────
from ss_boundaries_v2 import print_boundaries as old_print_boundaries
for S0 in [1e4, 1e5, 1e6]:
    print(f"\\n=== Sigma0 = {S0:.0e} ===")
    nt_print_boundaries(a=0.5, Sigma0=S0, alpha=0.1)
    old_print_boundaries(a=0.5, Sigma0=S0, alpha=0.1)

# ── patch di full_disk_SS per usare le nuove frontiere ───────────────────────
import full_disk_SS as fds
from ss_nt_boundaries import nt_boundaries, nt_mdot_from_Sigma0

def ss_boundaries_nt(a, B00, Sigma0=None, M=M_BH, alpha=0.1):
    # B00 non entra nelle frontiere
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha, M)
    return r_AB, r_BC

fds.ss_boundaries = ss_boundaries_nt

# ora compute_full_disk_profile usa automaticamente le frontiere NT
df_disk, meta = fds.compute_full_disk_profile(
    a=0.5, B00=1e7, Sigma0=1e5, check_norm=True
)
print(f"ṁ = {nt_mdot_from_Sigma0(1e5, 0.5):.4f}")
fds.plot_summary_table(df_disk, meta)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN — test autonomo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    print("=== Test singoli punti ===")
    for a_t in [-0.9, 0.0, 0.5, 0.9]:
        for S0 in [1e4, 1e5, 1e6]:
            nt_print_boundaries(a=a_t, Sigma0=S0, alpha=0.1)

    print("\n=== Scan griglia Σ₀ × a ===")
    nt_scan_grid(
        Sigma0_vals=[1e3, 1e4, 1e5, 1e6, 1e7],
        a_vals=[-0.9, 0.0, 0.5, 0.9],
    )
