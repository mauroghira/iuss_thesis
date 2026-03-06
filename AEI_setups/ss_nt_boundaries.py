"""
ss_nt_boundaries.py
===================
Calcolo delle frontiere S&S con correzioni relativistiche di Novikov-Thorne,
usando le formule ESATTE della review di Abramowicz (sezioni 5.3, pagine 28-29).

FISICA:
  - ṁ ricavato da Σ₀ tramite la formula ESATTA della zona A (inner region)
  - B₀₀ è un parametro libero indipendente (non entra nelle frontiere)
  - α è parametro libero (default 0.1)
  - r_AB: dove β/(1-β) = 1  →  P_rad = P_gas  (da zona A)
  - r_BC: dove τ_ff/τ_es = 1  →  cambio opacità (da zona B)

CONVENZIONI (dal paper):
  r* = r·c²/GM = r  [in unità di r_g = GM/c²]
  m  = M/M_sun
  ṁ  = Ṁc²/L_Edd  (tasso di accrescimento adimensionale)
  y  = sqrt(r*)
  a* = a  (spin adimensionale)
  y0 = sqrt(r_ISCO)

FATTORI RELATIVISTICI NT (review Abramowicz, eq. 237):
  A  = 1 + a*² y⁻⁴ + 2a*² y⁻⁶
  B  = 1 + a* y⁻³
  C  = 1 - 3y⁻² + 2a* y⁻³
  D  = 1 - 2y⁻² + a*² y⁻⁴
  E  = 1 + 4a*² y⁻⁴ - 4a*² y⁻⁶ + 3a*⁴ y⁻⁸
  Q0 = (1 + a* y⁻³) / [y sqrt(C)]
  Q  = integrale NT (formula esplicita)

FORMULE DI STRUTTURA — ZONA A (review Abramowicz, fattori NT esatti):
  Σ_A     = 5    · α⁻¹ ṁ⁻¹ r^{3/2}   · A⁻²B³C^{1/2}EQ⁻¹
  β/(1-β) = 4e-6 · α⁻¹/⁴ m⁻¹/⁴ ṁ⁻² r^{21/8}  · A⁻⁵/²B^{9/2}DE^{5/4}Q⁻²
  τ_ff τ_es = 1e-4 · α⁻¹⁷/¹⁶ m⁻¹/¹⁶ ṁ⁻² r^{93/32} · NT_A

FORMULE DI STRUTTURA — ZONE B e C (review Abramowicz, fattori NT esatti):
  Σ_B = 9e4  · α⁻⁴/⁵ m^{1/5} ṁ^{3/5} r^{-3/5}  · B⁻⁴/⁵C^{1/2}D⁻⁴/⁵Q^{3/5}
  Σ_C = 4e5  · α⁻⁴/⁵ m^{1/5} ṁ^{7/10} r^{-3/4} · A^{1/10}B⁻⁴/⁵C^{1/2}D⁻¹⁷/²⁰E⁻¹/²⁰Q^{7/10}

CAMPO MAGNETICO — ZONE B e C (Shakura & Sunyaev 1973, eq. 2.16 e 2.19):
  Nel paper S&S 1973 la lettera H indica il campo magnetico  [G]
  (non il semispessore, che è z_0).

  Zona B:  H_B = 1.5e9 · α^{1/20} · ṁ^{2/5} · m^{-9/20} · r^{-51/40}
                       · (1 - r^{-1/2})^{2/5}
           esponente radiale: -51/40 = -1.275  ≈  -5/4  ✓

  Zona C:  H_C = 2.1e9 · α^{1/20} · ṁ^{17/40} · m^{-9/20} · r^{-21/16}
                       · (1 - r^{-1/2})^{17/40}
           esponente radiale: -21/16 = -1.3125  ≈  -5/4  ✓

  Queste formule danno il campo magnetico "reale" previsto dalla struttura
  del disco S&S, con andamento r^{-5/4} approssimato.
  Possono essere usate per:
    (a) derivare B₀₀_SS come normalizzazione fisica alternativa al parametro libero
    (b) verificare la consistenza di B₀₀ scelto manualmente
    (c) costruire un profilo B₀(r) fisicamente motivato zona per zona

Dipendenze: numpy, setup.py
"""

import numpy as np
from functools import lru_cache

import sys
sys.path.append("..")
from setup import r_isco, r_horizon, M_BH

ALPHA_DEFAULT = 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 1.  COEFFICIENTI DELLE RADICI  (cache per spin fisso)
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def _nt_spin_constants(a):
    """
    Pre-calcola le costanti che dipendono solo da a:
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


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FATTORI RELATIVISTICI — completamente vettorizzati
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FORMULE DI STRUTTURA — ZONE A, B, C  (fattori NT esatti)
# ═══════════════════════════════════════════════════════════════════════════

def nt_Sigma_A(r, a, mdot, alpha):
    """Σ_A  [g/cm²] — zona A (P_rad, κ_es). Fattori NT esatti."""
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-2) * f['B']**3 * np.sqrt(np.maximum(f['C'], 0)) * f['E'] / f['Q']
    return 5.0 * alpha**(-1) * mdot**(-1) * np.asarray(r)**(3/2) * rel


def nt_Sigma_B(r, a, mdot, alpha):
    """Σ_B  [g/cm²] — zona B (P_gas, κ_es). Fattori NT esatti."""
    f = nt_ABCDEQ(r, a)
    rel = f['B']**(-4/5) * np.sqrt(np.maximum(f['C'], 0)) * f['D']**(-4/5) * f['Q']**(3/5)
    return 9e4 * alpha**(-4/5) * mdot**(3/5) * np.asarray(r)**(-3/5) * rel


def nt_Sigma_C(r, a, mdot, alpha, m):
    """Σ_C  [g/cm²] — zona C (P_gas, κ_ff). Fattori NT esatti."""
    f = nt_ABCDEQ(r, a)
    rel = (f['A']**(1/10) * f['B']**(-4/5) * np.sqrt(np.maximum(f['C'], 0))
           * f['D']**(-17/20) * f['E']**(-1/20) * f['Q']**(7/10))
    return 4e5 * alpha**(-4/5) * m**(1/5) * mdot**(7/10) * np.asarray(r)**(-3/4) * rel


def nt_beta_ratio_A(r, a, mdot, alpha, m):
    """β/(1-β) — zona A. Usata per trovare r_AB dove = 1."""
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-5/2) * f['B']**(9/2) * f['D'] * f['E']**(5/4) / f['Q']**2
    return 4e-6 * alpha**(-1/4) * m**(-1/4) * mdot**(-2) * np.asarray(r)**(21/8) * rel


def nt_tau_ratio_B(r, a, mdot):
    """τ_ff/τ_es — zona B. Usata per trovare r_BC dove = 1."""
    f = nt_ABCDEQ(r, a)
    rel = (f['A']**(-1) * f['B']**2
           * np.sqrt(np.maximum(f['D'], 0)) * np.sqrt(np.maximum(f['E'], 0)) / f['Q'])
    return 2e-6 * mdot**(-1) * np.asarray(r)**(3/2) * rel


def nt_tau_ratio_C(r, a, mdot):
    """τ_ff/τ_es — zona C. Verifica di consistenza a r_BC."""
    f = nt_ABCDEQ(r, a)
    rel = f['A']**(-1/2) * f['B']**(2/5) * f['D']**(1/4) * f['E']**(1/4) / f['Q']**(1/2)
    return 2e-3 * mdot**(-1/2) * np.asarray(r)**(3/4) * rel


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CAMPO MAGNETICO — ZONE B e C  (S&S 1973, eq. 2.16 e 2.19)
# ═══════════════════════════════════════════════════════════════════════════

def _ss_f(r):
    """Fattore non-relativista f = (1 - r^{-1/2}), clippato a > 0."""
    return np.maximum(1.0 - np.asarray(r, dtype=float)**(-0.5), 1e-10)


def ss_B0_B(r, mdot, alpha, M=M_BH):
    """
    Campo magnetico zona B  [G]  — Shakura & Sunyaev 1973, eq. 2.16.

        H_B = 1.5×10⁹ · α^{1/20} · ṁ^{2/5} · m^{-9/20}
                      · r^{-51/40} · (1 - r^{-1/2})^{2/5}

    Nota: nella notazione S&S 1973,  H  indica il campo magnetico [G],
    non il semispessore (che è z_0).
    L'esponente radiale esatto è -51/40 = -1.275, approssimato con -5/4.

    Parametri
    ----------
    r     : array_like   raggio [r_g]
    mdot  : float        ṁ = Ṁc²/L_Edd
    alpha : float        viscosità α
    M     : float        massa BH [M_sun]

    Restituisce
    -----------
    B0 : ndarray  [G]
    """
    r = np.asarray(r, dtype=float)
    m = float(M)
    f = _ss_f(r)
    return 1.5e9 * alpha**(1/20) * mdot**(2/5) * m**(-9/20) * r**(-51/40) * f**(2/5)


def ss_B0_C(r, mdot, alpha, M=M_BH):
    """
    Campo magnetico zona C  [G]  — Shakura & Sunyaev 1973, eq. 2.19.

        H_C = 2.1×10⁹ · α^{1/20} · ṁ^{17/40} · m^{-9/20}
                      · r^{-21/16} · (1 - r^{-1/2})^{17/40}

    L'esponente radiale esatto è -21/16 = -1.3125, approssimato con -5/4.

    Parametri
    ----------
    r     : array_like   raggio [r_g]
    mdot  : float        ṁ
    alpha : float        viscosità α
    M     : float        massa BH [M_sun]

    Restituisce
    -----------
    B0 : ndarray  [G]
    """
    r = np.asarray(r, dtype=float)
    m = float(M)
    f = _ss_f(r)
    return 2.1e9 * alpha**(1/20) * mdot**(17/40) * m**(-9/20) * r**(-21/16) * f**(17/40)


def nt_B0_from_mdot(r, a, mdot, alpha, M=M_BH, r_AB=None, r_BC=None,
                    approx_exp=True):
    """
    Profilo B₀(r) dalle formule S&S 1973, raccordato per zona.

    Usa le formule esatte per ogni zona e raccorda per continuità
    alle frontiere r_AB, r_BC, con esponente radiale:
      - esatto: -51/40 (zona B), -21/16 (zona C)
      - approssimato: -5/4 in entrambe (se approx_exp=True)

    Se r_AB o r_BC non sono forniti, vengono calcolati internamente.

    Parametri
    ----------
    r         : array_like   raggio [r_g]
    a         : float        spin
    mdot      : float        ṁ
    alpha     : float        viscosità α
    M         : float        massa BH [M_sun]
    r_AB      : float        frontiera A-B [r_g]  (None → calcolata)
    r_BC      : float        frontiera B-C [r_g]  (None → calcolata)
    approx_exp: bool         se True usa r^{-5/4} in entrambe le zone;
                             se False usa gli esponenti esatti di S&S

    Restituisce
    -----------
    B0   : ndarray  [G]   profilo raccordato
    B00  : float    [G]   normalizzazione all'orizzonte (per uso in full_disk_SS)
    """
    r  = np.asarray(r, dtype=float)
    rH = float(r_horizon(a))
    m  = float(M)

    if r_AB is None or r_BC is None:
        _, r_AB_calc, r_BC_calc = nt_boundaries(a, None, alpha, M)
        if r_AB is None: r_AB = r_AB_calc
        if r_BC is None: r_BC = r_BC_calc

    # Valori delle formule S&S agli raccordi
    B0_at_rAB_B = float(ss_B0_B(np.array([r_AB]), mdot, alpha, M)[0])
    B0_at_rBC_C = float(ss_B0_C(np.array([r_BC]), mdot, alpha, M)[0])
    # raccordo continuo B→C a r_BC usando zona B
    B0_at_rBC_B = float(ss_B0_B(np.array([r_BC]), mdot, alpha, M)[0])

    if approx_exp:
        # Power-law r^{-5/4} raccordate alle frontiere
        # Zona A: raccordato alla zona B a r_AB
        B0_A = B0_at_rAB_B * (r / r_AB)**(-5/4)
        # Zona B: formula esatta ma con esponente 5/4 (centrata a r_AB)
        B0_B = B0_at_rAB_B * (r / r_AB)**(-5/4)
        # Zona C: raccordato a zona B a r_BC (non a formula C, per continuità)
        B0_C = B0_at_rBC_B * (r / r_BC)**(-5/4)
    else:
        # Esponenti esatti S&S
        B0_A = B0_at_rAB_B * (r / r_AB)**(-51/40)   # usa esponente zona B anche in A
        B0_B = ss_B0_B(r, mdot, alpha, M)
        B0_C = ss_B0_C(r, mdot, alpha, M)
        # raccordo esplicito C a r_BC per continuità
        B0_C = B0_C * (B0_at_rBC_B / float(ss_B0_C(np.array([r_BC]), mdot, alpha, M)[0]))

    out = np.where(r <= r_AB, B0_A,
          np.where(r <= r_BC, B0_B, B0_C))

    # normalizzazione all'orizzonte per uso come B₀₀ in full_disk_SS
    B00 = float(B0_at_rAB_B) * (r_AB / rH)**(5/4)

    return out, B00


def nt_B00_from_mdot(a, mdot, alpha, M=M_BH, r_AB=None):
    """
    Restituisce direttamente la normalizzazione B₀₀  [G]  all'orizzonte,
    raccordata dalla formula S&S zona B valutata a r_AB.

    Conveniente per passare B₀₀ a full_disk_SS o find_rossby senza
    costruire l'intero profilo.

    Parametri
    ----------
    a     : float   spin
    mdot  : float   ṁ
    alpha : float   viscosità α
    M     : float   massa BH [M_sun]
    r_AB  : float   frontiera A-B [r_g]  (None → calcolata)
    """
    if r_AB is None:
        _, r_AB, _ = nt_boundaries(a, None, alpha, M)
    rH = float(r_horizon(a))
    B0_at_rAB = float(ss_B0_B(np.array([r_AB]), mdot, alpha, M)[0])
    return B0_at_rAB * (r_AB / rH)**(5/4)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  INVERSIONE Σ₀ → ṁ
# ═══════════════════════════════════════════════════════════════════════════

def nt_mdot_from_Sigma0(Sigma0, a, alpha=ALPHA_DEFAULT):
    """ṁ dalla formula zona A valutata a r_ISCO."""
    rISCO = float(r_isco(a))
    f     = nt_ABCDEQ(rISCO, a)
    rel_A = (f['A']**(-2) * f['B']**3
             * np.sqrt(max(float(np.squeeze(f['C'])), 0))
             * f['E'] / f['Q'])
    mdot  = 5.0 * rISCO**(3/2) * float(rel_A) / (alpha * Sigma0)
    return float(np.clip(mdot, 1e-4, 50.0))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  RICERCA RADICE — vettorizzata
# ═══════════════════════════════════════════════════════════════════════════

def _bisect_vec(func_vec, r_lo, r_hi, n_scan=200, n_bisect=40):
    """
    Trova r in [r_lo, r_hi] dove func(r) = 1.
    func_vec deve accettare un array numpy e restituire un array.
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


# ═══════════════════════════════════════════════════════════════════════════
# 7.  FRONTIERE r_AB e r_BC
# ═══════════════════════════════════════════════════════════════════════════

def nt_boundaries(a, Sigma0, alpha=ALPHA_DEFAULT, M=M_BH):
    """
    Frontiere S&S relativistiche.

    r_AB : β/(1-β) = 1  (zona A, P_rad = P_gas)
    r_BC : τ_ff/τ_es = 1  (zona B, cambio opacità)
    ṁ ricavato da Σ₀ via formula esatta zona A.
    B₀₀ non entra nelle frontiere.

    Se Sigma0 è None, mdot deve essere passato esternamente tramite
    nt_boundaries_from_mdot — questa funzione richiede Sigma0 != None.

    Returns: mdot, r_AB, r_BC
    """
    if Sigma0 is None:
        raise ValueError("Sigma0 richiesto per nt_boundaries. "
                         "Usa nt_boundaries_from_mdot se hai ṁ direttamente.")
    rISCO = float(r_isco(a))
    m     = float(M)

    mdot = nt_mdot_from_Sigma0(Sigma0, a, alpha)

    f_AB  = lambda r_arr: nt_beta_ratio_A(r_arr, a, mdot, alpha, m)
    r_AB  = _bisect_vec(f_AB, rISCO*1.01, 300.0)
    if r_AB is None:
        r_AB = rISCO*1.5 if f_AB(np.array([rISCO*1.01]))[0] > 1 else 150.0
    r_AB  = float(np.clip(r_AB, rISCO*1.1, 300.0))

    f_BC  = lambda r_arr: nt_tau_ratio_B(r_arr, a, mdot)
    r_BC  = _bisect_vec(f_BC, r_AB*1.01, 1e5)
    if r_BC is None:
        r_BC = r_AB*2.0 if f_BC(np.array([r_AB*1.01]))[0] > 1 else 1000.0
    r_BC  = float(np.clip(r_BC, r_AB*1.2, 1e5))

    return mdot, r_AB, r_BC


def nt_boundaries_from_mdot(a, mdot, alpha=ALPHA_DEFAULT, M=M_BH):
    """
    Variante di nt_boundaries che accetta ṁ direttamente invece di Σ₀.
    Utile quando si usa ss_B0_B/ss_B0_C come parametro di input.

    Returns: r_AB, r_BC   (mdot è già noto)
    """
    rISCO = float(r_isco(a))
    m     = float(M)

    f_AB  = lambda r_arr: nt_beta_ratio_A(r_arr, a, mdot, alpha, m)
    r_AB  = _bisect_vec(f_AB, rISCO*1.01, 300.0)
    if r_AB is None:
        r_AB = rISCO*1.5 if f_AB(np.array([rISCO*1.01]))[0] > 1 else 150.0
    r_AB  = float(np.clip(r_AB, rISCO*1.1, 300.0))

    f_BC  = lambda r_arr: nt_tau_ratio_B(r_arr, a, mdot)
    r_BC  = _bisect_vec(f_BC, r_AB*1.01, 1e5)
    if r_BC is None:
        r_BC = r_AB*2.0 if f_BC(np.array([r_AB*1.01]))[0] > 1 else 1000.0
    r_BC  = float(np.clip(r_BC, r_AB*1.2, 1e5))

    return r_AB, r_BC


# ═══════════════════════════════════════════════════════════════════════════
# 8.  DIAGNOSTICA
# ═══════════════════════════════════════════════════════════════════════════

def nt_print_boundaries(a, Sigma0, alpha=ALPHA_DEFAULT, M=M_BH):
    """Stampa un riassunto delle frontiere, ṁ, e B₀₀ fisicamente motivato."""
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha, M)
    tau_C = float(nt_tau_ratio_C(r_BC, a, mdot))

    # B0 fisicamente motivato dalle formule S&S
    B0_at_rAB = float(ss_B0_B(np.array([r_AB]), mdot, alpha, M)[0])
    B0_at_rBC = float(ss_B0_B(np.array([r_BC]), mdot, alpha, M)[0])
    B00_ss    = B0_at_rAB * (r_AB / rH)**(5/4)

    print(f"\n{'='*60}")
    print(f"  NT boundaries  |  a={a:.2f}  Σ₀={Sigma0:.1e}"
          f"  α={alpha:.2f}  M={M:.1e} M☉")
    print(f"{'='*60}")
    print(f"  ṁ      = {mdot:.4f}   (da Σ₀ zona A)")
    print(f"  r_H    = {rH:.3f}  r_g")
    print(f"  r_ISCO = {rISCO:.3f}  r_g")
    print(f"  r_AB   = {r_AB:.2f}  r_g  ({r_AB/rISCO:.1f} × r_ISCO)")
    print(f"  r_BC   = {r_BC:.1f}  r_g  ({r_BC/r_AB:.1f} × r_AB)")
    print(f"  verifica: τ_ff/τ_es (zona C) @ r_BC = {tau_C:.3f}  [atteso ≈ 1]")
    print(f"  --- Campo magnetico S&S ---")
    print(f"  B₀(r_AB) zona B  = {B0_at_rAB:.3e} G")
    print(f"  B₀(r_BC) zona B  = {B0_at_rBC:.3e} G")
    print(f"  B₀₀_SS (↑ r_H)  = {B00_ss:.3e} G   [usabile come B₀₀ in full_disk_SS]")
    print(f"{'='*60}")


def nt_scan_grid(Sigma0_vals, a_vals, alpha=ALPHA_DEFAULT, M=M_BH):
    """
    Tabella di ṁ, r_AB, r_BC, B₀₀_SS su una griglia (Σ₀, a).
    """
    import pandas as pd
    rows = []
    rH_vals = {av: float(r_horizon(av)) for av in a_vals}
    for S0 in Sigma0_vals:
        for a_val in a_vals:
            try:
                mdot, r_AB, r_BC = nt_boundaries(a_val, S0, alpha, M)
                rISCO = float(r_isco(a_val))
                B00_ss = float(ss_B0_B(np.array([r_AB]), mdot, alpha, M)[0]) \
                         * (r_AB / rH_vals[a_val])**(5/4)
                rows.append(dict(
                    Sigma0=S0, a=a_val, mdot=mdot,
                    r_AB=r_AB, r_BC=r_BC,
                    r_AB_rISCO=r_AB/rISCO, r_BC_rAB=r_BC/r_AB,
                    B00_SS=B00_ss,
                ))
            except Exception:
                rows.append(dict(
                    Sigma0=S0, a=a_val, mdot=np.nan,
                    r_AB=np.nan, r_BC=np.nan,
                    r_AB_rISCO=np.nan, r_BC_rAB=np.nan,
                    B00_SS=np.nan,
                ))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format='{:.3g}'.format))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MAIN — test autonomo
# ═══════════════════════════════════════════════════════════════════════════

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

    print("\n=== Test B₀ profilo con formule S&S ===")
    import matplotlib.pyplot as plt
    a_t = 0.5; S0 = 1e5; alpha = 0.1
    mdot, r_AB, r_BC = nt_boundaries(a_t, S0, alpha)
    r_test = np.geomspace(r_AB * 0.5, r_BC * 3, 200)
    B0_profile, B00 = nt_B0_from_mdot(r_test, a_t, mdot, alpha, M_BH,
                                        r_AB=r_AB, r_BC=r_BC)
    print(f"  a={a_t}, Σ₀={S0:.0e}, ṁ={mdot:.4f}")
    print(f"  r_AB={r_AB:.1f}, r_BC={r_BC:.1f}, B₀₀_SS={B00:.3e} G")

    print("\n=== Benchmark 19×20 combinazioni ===")
    import time
    cases = [(a, S0)
             for a  in np.linspace(-0.9, 0.9, 19)
             for S0 in np.geomspace(1e3, 1e7, 20)]
    t0 = time.time()
    for a_v, S0 in cases:
        nt_boundaries(a_v, S0, alpha=0.1)
    dt = time.time() - t0
    print(f"  {len(cases)} chiamate in {dt:.2f} s  ({dt/len(cases)*1000:.2f} ms/call)")
