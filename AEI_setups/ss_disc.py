"""
ss_disc.py
==========
Modello di disco Shakura-Sunyaev (1973) per l'analisi AEI.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUTTURA RADIALE — tre zone (SS 1973, Tab. 1 / eqs. 2.8–2.19)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Zona A  [r_ISCO, r_AB]:  P_rad >> P_gas,  opacità e-scattering
Zona B  [r_AB,   r_BC]:  P_gas >> P_rad,  opacità e-scattering
Zona C  [r_BC,   ∞   ):  P_gas >> P_rad,  opacità free-free (Kramers)

Fattore  f(r) = 1 − r^{−1/2}  (condizione al bordo newtoniana zero-torque)

Frontiere (SS 1973, eqs. 2.17, 2.20):
    r_AB :  r / f^{16/21} = 150 · (α m)^{2/21} · ṁ^{16/21}
    r_BC :  r / f^{2/3}   = 6.3×10³ · ṁ^{2/3} · m^{2/3}

La dipendenza da m in r_BC è resa esplicita: SS 1973 assume implicitamente
m = 1 (BH stellare), ma per AGN (m ~ 10^6) il fattore m^{2/3} è essenziale.

Zone degeneri:
    Se la condizione r_AB > r_ISCO non è soddisfatta → zona A assente.
    Se r_BC ≤ r_AB → zona B assente.
    In entrambi i casi il profilo parte dalla prima zona presente dall'ISCO,
    con le formule raw senza fattori di raccordo o di scala.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DENSITÀ SUPERFICIALE Σ(r)  —  formule SS 1973 Tab. 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    OSS: r è il raggio R/3Rg bnnelle formule di ogni zona nelpc paper

NON si applicano fattori di raccordo o scala: formule pure.
Le discontinuità alle frontiere sono fisicamente reali e diagnosticabili.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMISPESSORE H(r)  —  formule SS 1973 Tab. 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAMPO MAGNETICO B₀(r)  —  equipartizione dalla pressione fisica di ogni zona
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

L'equipartizione magnetica locale impone:
    B²/(8π) = P_mid(r)

La pressione al piano mediano si ricava dall'equilibrio idrostatico verticale:
    P_mid = ρ_mid · Ω_K² · H²  =  (Σ / 2H) · Ω_K² · H²  =  Σ · Ω_K² · H / 2

dove Ω_K = c/R_g · r^{-3/2} è la frequenza kepleriana newtoniana
e H = H_zona(r) è il semispessore fisico SS per quella zona.

Condizione di equipartizione:
    B_eq(r) = sqrt(4π · Σ(r) · Ω_K(r)² · H_zona(r))

Questa P_mid include implicitamente P_gas e P_rad attraverso il valore di H:
    - zona A: H ∝ L/c ∝ P_rad → B_eq dominato dalla pressione di radiazione
    - zone B/C: H ∝ P_gas → B_eq dominato dalla pressione del gas

Differenza rispetto al modello fenomenologico H/r = hr = cost:
    - La formula fisica usa H_zona(r) con esponenti e prefattori diversi
    - Le discontinuità di B alle frontiere riflettono il salto reale di P_mid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETRI LIBERI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

a      — spin adimensionale del BH              [−1, 1]
mdot   — tasso di accrescimento ṁ = Ṁ/Ṁ_Edd    [> 0]   ← PRIMARIO
alpha  — parametro di viscosità α               [adim]
M      — massa BH                               [M_sun]
hr     — aspect ratio H/r per c_s AEI           [adim]
            (non entra in Σ né in B, solo in c_s per il solver AEI)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API PUBBLICA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ss_boundaries(a, mdot, alpha, M)             → r_AB, r_BC, zone_present
disk_model_SS(r_rg, a, mdot, ...)            → B0, Sigma, c_s, zone, info
disk_inner_values_SS(a, mdot, ...)           → dict {r_ISCO, r_H, Sigma_ISCO, ...}
check_continuity_SS(a, mdot, ...)            → dict + stampa diagnostica

Funzioni di profilo per zona (accesso diretto):
Sigma_A(r, mdot, alpha)
Sigma_B(r, mdot, alpha, M)
Sigma_C(r, mdot, alpha, M)
H_A(r, M),  H_B(r, mdot, alpha, M),  H_C(r, mdot, alpha, M)
B_eq_zone(r, Sigma, H, M)

Dipendenze: numpy, setup.py, aei_common.py
"""

import numpy as np

from .aei_common import ALPHA_VISC, HOR, mm

import sys
sys.path.append("..")
from setup import (
    r_isco, r_horizon, nu_phi, 
    Rg_SUN, M_BH, C, SigTOM
)

ZONE_NAMES  = ['A', 'B', 'C']
ZONE_COLORS = {'A': '#f97316', 'B': '#3b82f6', 'C': '#22c55e'}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  FATTORE f  e  bisezione scalare
# ═══════════════════════════════════════════════════════════════════════════════

def _ss_f(r):
    """
    Fattore di momento angolare newtoniano:  f(r) = max(1 − r^{−1/2}, ε).

    Rappresenta la correzione alla derivata di E-Omega angolare vicino al bordo
    interno (zero-torque condition all'ISCO nella metrica piatta).
    Protetto da zero per stabilità numerica a r molto vicino a r_ISCO.
    """
    return np.maximum(1.0 - np.asarray(r, float)**(-0.5), 1e-10)


def _bisect(func, r_lo, r_hi, n_scan=400, n_bisect=60):
    """
    Bisezione scalare robusta: trova r in [r_lo, r_hi] tale che func(r) = 0.

    Scansione geometrica iniziale per localizzare il cambio di segno,
    poi raffinamento con n_bisect iterazioni di bisezione classica.
    Restituisce None se nessun cambio di segno è trovato.
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FRONTIERE  r_AB  e  r_BC
# ═══════════════════════════════════════════════════════════════════════════════

def ss_boundaries(a, mdot, alpha=ALPHA_VISC, M=M_BH):
    """
    Calcola le frontiere radiali r_AB, r_BC e indica le zone presenti.

    Frontiere (SS 1973, eqs. 2.17, 2.20):

    r_AB: P_rad = P_gas
            r / f(r)^{16/21} = 150 · (α m)^{2/21} · ṁ^{16/21}

    r_BC: τ_ff = τ_es (cambio regime opacità)
            r / f(r)^{2/3}  = 6.3×10³ · ṁ^{2/3} · m^{2/3}

    NOTA su m^{2/3} in r_BC: questo fattore è presente nelle equazioni SS 1973
    ma spesso omesso perché il paper lavora implicita con m=1 (BH stellare).
    Per AGN con m ~ 10^6 sposta r_BC da ~10^3 a ~10^5 r_g — non trascurabile.

    Zone degeneri:
    Se r_AB ≤ r_ISCO → zona A assente: il profilo inizia in zona B dall'ISCO
    Se r_BC ≤ r_AB  → zona B assente: il profilo salta direttamente a zona C

    Parametri
    ----------
    a     : float   spin adimensionale
    mdot  : float   ṁ = Ṁ/Ṁ_Edd > 0
    alpha : float   parametro di viscosità α
    M     : float   massa BH [M_sun]

    Restituisce
    -----------
    r_AB        : float   frontiera A-B [r_g]  (= r_ISCO se zona A assente)
    r_BC        : float   frontiera B-C [r_g]  (= r_AB se zona B assente)
    zone_present: dict    {'A': bool, 'B': bool, 'C': bool}
                        C è sempre True (si estende a infinito)
    """
    rISCO = float(r_isco(a))
    m     = float(M)

    # ── r_AB  (eq. 2.17) ────────────────────────────────────────────────────
    rhs_AB = 150.0 * (alpha * m)**(2.0/21) * mdot**(16.0/21)

    def eq_AB(r):
        r = np.asarray(r, float)
        return r / _ss_f(r)**(16.0/21) - rhs_AB

    if eq_AB(np.array([rISCO * 1.001]))[0] >= 0.0:
        # f_AB > 1 già all'ISCO → zona A assente, il disco è tutto gas-pressure
        r_AB   = rISCO/3
        zone_A = False
    else:
        r_AB_hi = float(max(rhs_AB * 3.0, rISCO * 10.0))
        r_AB    = _bisect(eq_AB, rISCO * 1.001/3, r_AB_hi)
        if r_AB is None:
            r_AB = r_AB_hi
        r_AB   = float(max(r_AB, rISCO/3))
        zone_A = True

    # ── r_BC  (eq. 2.20 con m^{2/3} esplicito) ──────────────────────────────
    rhs_BC = 6.3e3 * mdot**(2.0/3) * m**(2.0/3)

    def eq_BC(r):
        r = np.asarray(r, float)
        return r / _ss_f(r)**(2.0/3) - rhs_BC

    if eq_BC(np.array([r_AB * 1.001]))[0] >= 0.0:
        # τ_ff/τ_es > 1 già a r_AB → zona B assente
        r_BC   = r_AB
        zone_B = False
    else:
        r_BC_hi = float(max(rhs_BC * 3.0, r_AB * 10.0))
        r_BC    = _bisect(eq_BC, r_AB * 1.001, r_BC_hi)
        if r_BC is None:
            r_BC = r_BC_hi
        r_BC   = float(max(r_BC, r_AB))
        zone_B = True

    # moltiplico per 3 per convertire da r/3Rg a r/Rg, coerentemente con le formule di H e Σ
    return r_AB*3.0, r_BC*3.0, {'A': zone_A, 'B': zone_B, 'C': True}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  SEMISPESSORE H(r) per zona  —  SS 1973 Tab. 1
# ═══════════════════════════════════════════════════════════════════════════════

def H_A(r, mdot, m=M_BH):
    """
    r : array_like   raggio [3r_g, adimensionale]
    m : float        massa BH [M_sun]
    """
    r = np.asarray(r/3, float)
    
    return 3.2e6 * mdot * float(m) * _ss_f(r)


def H_B(r, mdot, alpha, M):
    """
    Semispessore fisico zona B  [cm]  —  SS 1973 eq. 2.16
    Zona gas-dominated, opacità e-scattering.

    r     : array_like   raggio [r_g]
    mdot  : float        ṁ = Ṁ/Ṁ_Edd
    alpha : float        parametro viscosità α
    M     : float        massa BH [M_sun]
    """
    r = np.asarray(r/3, float)
    return 1.2e4 * alpha**(-1/10) * mdot**(1/5) * float(M)**(9/10) \
        * r**(21/20) * _ss_f(r)**(1/5)


def H_C(r, mdot, alpha, M):
    """
    Semispessore fisico zona C  [cm]  —  SS 1973 eq. 2.19
    Zona gas-dominated, opacità free-free (Kramers: κ_ff ∝ ρ T^{-7/2}).

    r     : array_like   raggio [r_g]
    mdot  : float        ṁ = Ṁ/Ṁ_Edd
    alpha : float        parametro viscosità α
    M     : float        massa BH [M_sun]
    """
    r = np.asarray(r/3, float)
    return 6.1e3 * alpha**(-1/10) * mdot**(3/20) * float(M)**(9/10) \
        * r**(9/8) * _ss_f(r)**(3/20)


def _H_disk(r, mdot, alpha, r_AB, r_BC, M=M_BH):
    """
    H(r) sull'intero disco — formula di zona corretta, nessun raccordo.

    Assegna H_A, H_B, H_C in base alla zona di appartenenza,
    gestendo correttamente le zone degeneri (r_AB == r_ISCO o r_BC == r_AB).
    """
    r  = np.asarray(r, float)
    HA = H_A(r, mdot, M)
    HB = H_B(r, mdot, alpha, M)
    HC = H_C(r, mdot, alpha, M)

    result = HA.copy()
    result[r > r_AB] = HB[r > r_AB]
    result[r > r_BC] = HC[r > r_BC]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DENSITÀ SUPERFICIALE Σ(r)  —  formule SS 1973 raw
# ═══════════════════════════════════════════════════════════════════════════════

def Sigma_A(r, mdot, alpha):
    """
    Σ zona A  [g/cm²]  —  SS 1973 eq. 2.9

        Σ_A = 4.6 · α⁻¹ · ṁ⁻¹ · r^{3/2} · f(r)^{−1}

    P_rad dominante, opacità e-scattering. Profilo "invertito": Σ_A cresce
    con r a causa del fattore f → 0 per r → r_ISCO (zero-torque boundary).
    """
    r = np.asarray(r/3, float)
    return 4.6 * alpha**(-1.0) * mdot**(-1.0) * r**(1.5) * _ss_f(r)**(-1.0)


def Sigma_B(r, mdot, alpha, M):
    """
    Σ zona B  [g/cm²]  —  SS 1973 eq. 2.16

        Σ_B = 1.7×10⁵ · α⁻⁴/⁵ · ṁ^{3/5} · m^{1/5} · r^{-3/5} · f(r)^{3/5}

    P_gas dominante, opacità e-scattering. Σ decresce con r.
    """
    r = np.asarray(r/3, float)
    return 1.7e5 * alpha**(-4.0/5) * mdot**(3.0/5) * float(M)**(1.0/5) \
        * r**(-3.0/5) * _ss_f(r)**(3.0/5)


def Sigma_C(r, mdot, alpha, M):
    """
    Σ zona C  [g/cm²]  —  SS 1973 Tab. 1, eq. 2.19

    P_gas dominante, opacità free-free. Esponente m^{1/2} = 2 × m^{1/4}
    dalla combinazione di ρ ∝ m^{1/4} e h ∝ m^{1/4}.
    """
    r = np.asarray(r/3, float)
    return 6.1e5 * alpha**(-4.0/5) * mdot**(7.0/10) * float(M)**(1/5) \
        * r**(-3/4) * _ss_f(r)**(7.0/10)


def _Sigma_disk(r, mdot, alpha, M, r_AB, r_BC):
    """
    Σ(r) sull'intero disco — formule raw SS 1973, nessun raccordo.

    Assegna Σ_A, Σ_B, Σ_C in base alla zona, gestendo le zone degeneri.
    Le discontinuità alle frontiere sono fisicamente reali: le formule
    SS sono approssimazioni asintotiche valide solo all'interno di ogni zona.
    Quantificare i salti con check_continuity_SS().
    """
    r  = np.asarray(r, float)
    SA = Sigma_A(r, mdot, alpha)
    SB = Sigma_B(r, mdot, alpha, M)
    SC = Sigma_C(r, mdot, alpha, M)

    result = SA.copy()
    result[r > r_AB] = SB[r > r_AB]
    result[r > r_BC] = SC[r > r_BC]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CAMPO MAGNETICO B₀(r)  —  equipartizione dalla P_mid fisica
# ═══════════════════════════════════════════════════════════════════════════════

def B_eq_zone(r, Sigma, H, M):
    """
    Campo magnetico in equipartizione con la pressione al piano mediano  [Gauss].

    Dall'equilibrio idrostatico verticale del disco sottile:

        P_mid = ρ_mid · Ω_K² · H²  =  (Σ/2H) · Ω_K² · H²  =  Σ · Ω_K² · H / 2

    Questa pressione include P_gas e P_rad attraverso il valore fisico di H:
    - zona A: H ∝ P_rad/(ρ Ω_K²) → B_eq influenzata dalla pressione di radiazione
    - zone B/C: H ∝ P_gas/(ρ Ω_K²) → B_eq influenzata dalla pressione del gas

    Condizione di equipartizione B²/(8π) = P_mid:
        B_eq(r) = sqrt(4π · Σ(r) · Ω_K(r)² · H_zona(r))

    Passando H dalla zona fisica corretta (non da hr·r·R_g) si ottiene
    il profilo di B fisicamente motivato dalle proprietà termodinamiche
    del disco zona per zona.

    Parametri
    ----------
    r     : array_like   raggio [r_g]
    Sigma : array_like   densità superficiale [g/cm²]
    H     : array_like   semispessore fisico [cm] (dalla zona corretta)
    M     : float        massa BH [M_sun]
    """
    r     = np.asarray(r, float)
    Sigma = np.asarray(Sigma, float)
    H     = np.asarray(H, float)
    OmKsq = _Omega_K_sq(r, M)
    return np.sqrt(4.0 * np.pi * Sigma * OmKsq * H)


def _B0_disk(r, mdot, alpha, M, r_AB, r_BC):
    """
    B₀(r) in equipartizione — usa H e Σ della zona fisica corretta.

        B_eq(r) = sqrt(4π · Σ_zona(r) · Ω_K²(r) · H_zona(r))

    Le discontinuità di B alle frontiere riflettono il salto di H e Σ tra zone.
    """
    r  = np.asarray(r, float)
    S  = _Sigma_disk(r, mdot, alpha, M, r_AB, r_BC)
    Hv = _H_disk(r, mdot, alpha, M, r_AB, r_BC)
    return B_eq_zone(r, S, Hv, M)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  VELOCITÀ DEL SUONO EFFETTIVA c_s(r)
# ═══════════════════════════════════════════════════════════════════════════════

def _Omega_K_sq(r, M):
    """
    Frequenza kepleriana newtoniana al quadrato  Ω_K²(r)  [rad²/s²].

        Ω_K² = GM / r_cm³ = (c / R_g)² · r^{-3}

    con r adimensionale in r_g e R_g = GM/c² [cm].
    Usata nel modello SS (newtoniano); nel modello NT si usa nu_phi di Kerr.

    r : array_like   raggio [r_g]
    M : float        massa BH [M_sun]
    """
    return (C / (M * Rg_SUN))**2 * np.asarray(r, float)**(-3.0)


def _sound_speed(r, Hv=None, M=M_BH, hr=None):
    """
    Velocità del suono effettiva per la relazione di dispersione AEI  [cm/s].

        c_s(r) = H · Ω_φ(r)

    dove Ω_φ = 2π ν_φ è la frequenza orbitale kepleriana (SS assume disco non relativistico)
    Il parametro hr è usato qui come aspect ratio fenomenologico per il
    solver AEI, indipendente da H_zona usato per B.

    Nota: questa c_s non è usata per calcolare P_mid né B. È l'input
    per il termine di pressione nella relazione di dispersione AEI.

    r  : array_like   raggio [r_g]
    a  : float        spin del BH
    M  : float        massa BH [M_sun]
    """
    r  = np.asarray(r, dtype=float)
    if Hv is not None:
        cs = Hv * np.sqrt(_Omega_K_sq(r, M))
    elif hr is not None:
        cs = hr * r * Rg_SUN * M * _Omega_K_sq(r, M)**0.5
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
# 8.  ADAPTER PRINCIPALE  —  disk_model_SS
# ═══════════════════════════════════════════════════════════════════════════════

def disk_model_SS(r_rg, a, mdot, alpha_visc=ALPHA_VISC, hr=None, M=M_BH):
    """
    Profili fisici del disco SS 1973 — firma standard per find_rossby.

    Calcola su un array radiale:
    - Σ(r): formule SS raw per zona, nessun raccordo o fattore di scala
    - B(r): equipartizione da P_mid = Σ · Ω_K² · H_zona / 2
    - c_s(r): thin-disc fenomenologica c_s = H · Ω_φ (per solver AEI)

    Parametri
    ----------
    r_rg      : array_like   raggi [r_g]
    a         : float        spin adimensionale [−1, 1]
    mdot      : float        ṁ = Ṁ/Ṁ_Edd > 0
    alpha_visc: float        parametro α di viscosità
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
    r_AB, r_BC, zone_present = ss_boundaries(a, mdot, alpha=alpha_visc, M=M)

    Sigma = _Sigma_disk(r_rg, mdot, alpha_visc, M, r_AB, r_BC)
    if hr is None:
        Hv = _H_disk(r_rg, mdot, alpha_visc, M, r_AB, r_BC)
        hr = Hv / (r_rg * Rg_SUN * M)  # aspect ratio
    else:
        Hv = hr * r_rg * Rg_SUN * M
    B0    = B_eq_zone(r_rg, Sigma, Hv, M)
    c_s   = _sound_speed(r_rg, Hv, M)
    zone  = _zone_array(r_rg, r_AB, r_BC)

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

def disk_inner_values_SS(a, mdot, alpha_visc=ALPHA_VISC, M=M_BH):
    """
    Restituisce Σ, H e B ai bordi interni fisicamente significativi.

    Bordi:
    r_ISCO — Inner Stable Circular Orbit (bordo del disco)
    r_H    — orizzonte degli eventi (bordo della magnetosfera)

    Parametri: come disk_model_SS (hr non usato qui).

    Restituisce
    -----------
    dict con:
    r_ISCO      [r_g]     raggio ISCO
    r_H         [r_g]     raggio orizzonte
    r_AB        [r_g]     frontiera A-B
    r_BC        [r_g]     frontiera B-C
    zone_present          {'A': bool, 'B': bool, 'C': bool}
    Sigma_ISCO  [g/cm²]   Σ all'ISCO
    B_rH        [G]       B_eq all'orizzonte
    """
    rISCO = float(r_isco(a))
    rH    = float(r_horizon(a))
    r_AB, r_BC, zone_present = ss_boundaries(a, mdot, alpha=alpha_visc, M=M)

    r_pts = np.array([rISCO, rH])
    S_pts = _Sigma_disk(r_pts, mdot, alpha_visc, M, r_AB, r_BC)
    H_pts = _H_disk(r_pts, mdot, alpha_visc, M, r_AB, r_BC)
    B_pts = B_eq_zone(r_pts, S_pts, H_pts, M)

    return {
        'r_ISCO':       rISCO,
        'r_H':          rH,
        'r_AB':         r_AB,
        'r_BC':         r_BC,
        'zone_present': zone_present,
        'Sigma_ISCO':   float(S_pts[0]),
        'B_rH':         float(B_pts[1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  DIAGNOSTICA  —  check_continuity_SS
# ═══════════════════════════════════════════════════════════════════════════════

def check_continuity_SS(a, mdot, alpha_visc=ALPHA_VISC, hr=None, M=M_BH,
                        tol=0.5, verbose=True, only_multizone=True):
    """
    Misura le discontinuità reali di Σ, H e B alle frontiere r_AB e r_BC.

    Calcola il salto relativo  |f(r⁺) − f(r⁻)| / |f(r⁻)|  a distanza
    ε = 10⁻⁴ · r₀ da ogni frontiera.

    Con le formule SS raw i salti possono essere grandi (fattore 10–100
    per parametri AGN): questo è fisicamente atteso. Le formule SS sono
    approssimazioni valide solo nell'interno di ogni zona — il salto misura
    il "mismatch" tra le asintotiche di zone adiacenti alla frontiera.

    tol : float   soglia per il flag 'ok' (default 0.5 = 50%);
                un salto del 50% è considerato accettabile per formule
                asintotiche; valori > 10 (fattore ~10) indicano problemi.

    Restituisce
    -----------
    results : dict
        Per ogni frontiera ('r_AB', 'r_BC') e variabile ('Sigma', 'H', 'B'):
            {'value_in', 'value_out', 'jump_rel', 'ok'}
    """
    r_AB, r_BC, zone_present = ss_boundaries(a, mdot, alpha=alpha_visc, M=M)
    rISCO = float(r_isco(a))
    eps   = 1e-4

    # filtro dischi monotoni
    if only_multizone and not (zone_present['A'] or zone_present['B']):
        if verbose:
            print("Disco SS monotona (solo zona C) — check saltato.")
        return {}, zone_present

    results = {}

    if verbose:
        print(f"┌─ Salti SS raw  (a={a:.3f}, mdot={mdot:.3e}, α={alpha_visc}, M={M:.2e} Msun)")
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

        S_in  = float(_Sigma_disk(r_in,  mdot, alpha_visc, M, r_AB, r_BC)[0])
        S_out = float(_Sigma_disk(r_out, mdot, alpha_visc, M, r_AB, r_BC)[0])
        Hi_in = float(_H_disk(r_in,  mdot, alpha_visc, M, r_AB, r_BC)[0])
        Hi_out= float(_H_disk(r_out, mdot, alpha_visc, M, r_AB, r_BC)[0])
        B_in  = float(B_eq_zone(r_in,  np.array([S_in]),  np.array([Hi_in]),  M)[0])
        B_out = float(B_eq_zone(r_out, np.array([S_out]), np.array([Hi_out]), M)[0])

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
                flag = '✓' if ok else f'✗  (fattore {max(v_in,v_out)/max(min(v_in,v_out),1e-300):.1f}×)'
                return f"{v_in:.3e} → {v_out:.3e}  |  Δ/val = {dv:.2e}  {flag}"
            print(f"│  {label} = {r0:.4g} rg")
            print(f"│    Σ:  {fmtrow(S_in, S_out, dS, dS < tol)}")
            print(f"│    H:  {fmtrow(Hi_in, Hi_out, dH, dH < tol)}")
            print(f"│    B:  {fmtrow(B_in, B_out, dB, dB < tol)}")

    if verbose:
        print(f"└{'─'*68}")

    return results, zone_present