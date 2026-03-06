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

# import del modulo NT (stesso folder)
from .ss_nt_boundaries import nt_boundaries, nt_mdot_from_Sigma0

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

def ss_boundaries(a, Sigma0, alpha=0.1, M=M_BH):
    """
    Frontiere r_AB e r_BC con correzioni relativistiche di Novikov-Thorne.

    ṁ ricavato da Σ₀ tramite la formula ESATTA della zona A (inner region)
    del review Shakura, sez. 5.3:
        Σ_A(r_ISCO) = Σ₀  →  ṁ = 5 r_ISCO^{3/2} rel_A / (α Σ₀)

    r_AB: β/(1-β) = 1  (P_rad = P_gas, zona A)
    r_BC: τ_ff/τ_es = 1  (cambio opacità, zona B)

    B₀₀ NON entra nelle frontiere — è un parametro libero del campo.

    Parameters
    ----------
    a      : float   spin adimensionale
    Sigma0 : float   Σ₀ in g/cm²  (determina ṁ)
    alpha  : float   viscosità α (default 0.1)
    M      : float   massa BH in M_sun

    Returns
    -------
    r_AB, r_BC : float   raggi di frontiera in r_g
    mdot       : float   ṁ derivato
    """
    mdot, r_AB, r_BC = nt_boundaries(a, Sigma0, alpha=alpha, M=M)
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

    # Valore di A alle frontiere
    B_A_rAB = B00    * (r_AB / rH   )**(-aA['alpha_B'])
    S_A_rAB = Sigma0 * (r_AB / rISCO)**(-aA['alpha_S'])

    # Zona B — raccordo a r_AB
    norms['B'] = dict(B_norm=B_A_rAB, S_norm=S_A_rAB, r_ref_B=r_AB, r_ref_S=r_AB)

    # Valore di B alle frontiere
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
# 4.  SOLVER AEI  (dispersion relation)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_k_disk(r_rg, a, B0, Sigma, c_s, m, M=M_BH):
    """
    Risolve la relazione di dispersione AEI (Tagger & Pellat 1999, Eq. 17):

        ω̃² = κ² + (2B₀²/Σ)(|k|/r) + (k²/r²) c_s²

    dove k è il numero d'onda nella variabile s = ln(r)  →  k adimensionale,
    e r è il raggio fisico in cm.

    ANALISI DIMENSIONALE:
      - ω̃², κ²       : s⁻²
      - 2B₀²/Σ · k/r : [G²/(g/cm²)] · [1/cm] = s⁻²  (r in cm, k adim.)
      - k²/r² · c_s² : [1/cm²] · [cm²/s²]    = s⁻²  (r in cm, k adim.)

    Il codice converte r [r_g] → r [cm] internamente prima di costruire
    i coefficienti. k restituito è adimensionale (numero d'onda in s=ln r).
    kperr = k (già adimensionale, equivale a k_fisico × r).

    Parameters
    ----------
    r_rg  : array_like   raggio in unità di r_g
    a     : float        spin adimensionale
    B0    : array_like   campo magnetico [G]
    Sigma : array_like   densità superficiale [g/cm²]
    c_s   : array_like   velocità del suono [cm/s]
    m     : int          numero d'onda azimutale
    M     : float        massa BH [M_sun]

    Returns
    -------
    k   : ndarray   numero d'onda adimensionale (in s = ln r)
                    NaN dove non esiste soluzione reale e positiva.
                    k_fisico [1/cm] = k / r_cm
                    kperr (adim.)   = k  (identicamente)
    """
    r_rg  = np.asarray(r_rg,  dtype=float)
    B0    = np.asarray(B0,    dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    c_s   = np.asarray(c_s,   dtype=float)

    # ── conversione r in cm ──────────────────────────────────────────────────
    Rg    = Rg_SUN * M          # cm per r_g
    r_cm  = r_rg * Rg           # raggio fisico in cm

    # ── frequenze ───────────────────────────────────────────────────────────
    kappa_sq  = (2 * np.pi * nu_r(r_rg,  a, M))**2   # s⁻²
    Omega_phi = 2 * np.pi * nu_phi(r_rg, a, M)        # s⁻¹
    omega     = 2 * np.pi * NU0                        # s⁻¹  (frequenza osservata)
    om_tilde  = omega - m * Omega_phi                  # s⁻¹

    # ── coefficienti quadratica  A k² + B k + CC = 0 ────────────────────────
    # r in cm → k adimensionale → unità omogenee s⁻²
    A  = c_s**2   / r_cm**2          # s⁻²
    B  = 2*B0**2  / (Sigma * r_cm)   # s⁻²
    CC = kappa_sq - om_tilde**2       # s⁻²

    Delta = B**2 - 4*A*CC

    k = np.full_like(r_rg, np.nan)
    good = Delta >= 0
    if np.any(good):
        sqD = np.sqrt(Delta[good])
        kp  = (-B[good] + sqD) / (2*A[good])
        km  = (-B[good] - sqD) / (2*A[good])
        # preferisci k_plus (più grande); se negativo prova k_minus
        k[good] = np.where(kp > 0, kp, np.where(km > 0, km, np.nan))

    return k   # adimensionale (numero d'onda in s = ln r)


def compute_beta_disk(r, a, B0, Sigma, c_s, hr, M=M_BH):
    """β = 8π Σ c_s² / (H B₀²)  con H = hr * r * Rg."""
    Rg  = Rg_SUN * M
    H   = hr * r * Rg
    return 8 * np.pi * Sigma * c_s**2 / (H * B0**2)


def compute_dQdr_disk(r, a, norms, r_AB, r_BC, M=M_BH, dr_frac=0.01):
    """
    d/dr [Ω_φ Σ / B₀²]  calcolato con differenze finite.
    Usa i profili raccordati B0_disk / Sigma_disk.
    """
    dr   = dr_frac * r
    rp   = r + dr

    B0_r  = B0_disk(r,  norms, r_AB, r_BC)
    B0_rp = B0_disk(rp, norms, r_AB, r_BC)
    S_r   = Sigma_disk(r,  norms, r_AB, r_BC)
    S_rp  = Sigma_disk(rp, norms, r_AB, r_BC)

    Q_r  = 2 * np.pi * nu_phi(r,  a, M) * S_r  / B0_r**2
    Q_rp = 2 * np.pi * nu_phi(rp, a, M) * S_rp / B0_rp**2

    return (Q_rp - Q_r) / dr


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PROFILO RADIALE COMPLETO → DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def compute_full_disk_profile(a, B00, Sigma0, mm, hr, alpha=0.01, M=M_BH,
                               n_points=300, r_max_factor=3.0,
                               check_norm=False):
    """
    Calcola il profilo radiale completo del disco unificato.

    Le frontiere r_AB e r_BC sono calcolate con le formule relativistiche
    di Novikov-Thorne (ss_nt_boundaries.py):
      - ṁ ricavato da Σ₀ via formula esatta zona A
      - r_AB: β/(1-β) = 1  (zona A)
      - r_BC: τ_ff/τ_es = 1  (zona B)
    I profili B(r) e Σ(r) usano il raccordo per continuità (power-law
    con esponenti S&S per zona), con normalizzazioni fissate da B₀₀ e Σ₀.

    Parameters
    ----------
    a            : float   spin (−1, 1)
    B00          : float   B₀ all'orizzonte [G]
    Sigma0       : float   Σ all'ISCO [g/cm²]
    M            : float   massa BH [M_sun]
    alpha        : float   viscosità α (default 0.1)
    n_points     : int     punti sulla griglia radiale
    r_max_factor : float   r_max = r_max_factor * r_BC
    check_norm   : bool    se True stampa verifica di continuità

    Returns
    -------
    df   : DataFrame con colonne:
           r, zone, B0, Sigma, c_s, k_cm, k, kr,
           beta, dQdr, k_valid, beta_valid, shear_valid, aei_valid
    meta : dict con r_H, r_ISCO, r_AB, r_BC, mdot, alpha, norms
    """
    rH    = r_horizon(a)
    rISCO = r_isco(a)
    r_AB, r_BC, mdot = ss_boundaries(a, Sigma0, alpha=alpha, M=M)
    norms            = compute_norms(a, B00, Sigma0, r_AB, r_BC)

    if check_norm:
        print(f"\nVerifica continuità (a={a:.2f}, B00={B00:.2e}, Σ0={Sigma0:.2e}):")
        check_continuity(norms, r_AB, r_BC)

    # Griglia radiale log-spaziata dall'ISCO a r_max_factor * r_BC
    r_min = rISCO * 1.005
    r_max = min(r_max_factor * r_BC, 2000.0)
    r_arr = np.geomspace(r_min, r_max, n_points)

    # Profili fisici
    B0_arr    = B0_disk(r_arr, norms, r_AB, r_BC)
    Sigma_arr = Sigma_disk(r_arr, norms, r_AB, r_BC)
    cs_arr    = sound_speed_disk(r_arr, a, hr, M)
    zone_arr  = zone_index(r_arr, r_AB, r_BC)

    Rg = Rg_SUN * M

    # Solver k
    k_cm_arr = solve_k_disk(r_arr, a, B0_arr, Sigma_arr, cs_arr, m=mm, M=M)
    kr_arr   = k_arr * r_arr        # adimensionale

    # Beta e shear
    beta_arr  = compute_beta_disk(r_arr, a, B0_arr, Sigma_arr, cs_arr, hr, M)
    dQdr_arr  = compute_dQdr_disk(r_arr, a, norms, r_AB, r_BC, M)

    # Maschere di validità fisica
    k_valid     = (kr_arr >= 0.1) & (kr_arr <= 10) & np.isfinite(k_arr)
    beta_valid  = beta_arr <= 1.0
    shear_valid = dQdr_arr > 0
    aei_valid   = k_valid & beta_valid & shear_valid

    df = pd.DataFrame({
        'r':           r_arr,
        'zone':        [ZONE_NAMES[i] for i in zone_arr],
        'B0':          B0_arr,
        'Sigma':       Sigma_arr,
        'c_s':         cs_arr,
        'k':           k_arr,
        'kr':          kr_arr,
        'beta':        beta_arr,
        'dQdr':        dQdr_arr,
        'k_valid':     k_valid,
        'beta_valid':  beta_valid,
        'shear_valid': shear_valid,
        'aei_valid':   aei_valid,
    })

    meta = dict(r_H=rH, r_ISCO=rISCO, r_AB=r_AB, r_BC=r_BC,
                mdot=mdot, alpha=alpha,
                norms=norms, a=a, B00=B00, Sigma0=Sigma0)
    return df, meta


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def _zone_vlines(ax, meta, alpha=0.7):
    """Aggiunge linee verticali per r_ISCO, r_AB, r_BC."""
    ax.axvline(meta['r_ISCO'], color='white',   ls=':', lw=1, alpha=0.5,  label=f"r_ISCO={meta['r_ISCO']:.1f}")
    ax.axvline(meta['r_AB'],   color='#f97316', ls='--', lw=1, alpha=alpha, label=f"r_AB={meta['r_AB']:.1f}")
    ax.axvline(meta['r_BC'],   color='#3b82f6', ls='--', lw=1, alpha=alpha, label=f"r_BC={meta['r_BC']:.0f}")


def plot_full_disk_profiles(df, meta, alpha_visc, figsize=(16, 12)):
    """
    Figura con 4 pannelli:
      (1) B₀(r) e Σ(r) per zona
      (2) k·r(r)
      (3) β(r)
      (4) dQ/dr(r)  +  regioni AEI valide evidenziate
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Disco S&S unificato — a={meta['a']:.2f},  "
        f"B₀₀={meta['B00']:.1e},  Σ₀={meta['Sigma0']:.1e},  "
        f"ṁ={meta.get('mdot', float('nan')):.3f},  "
        f"α={meta.get('alpha', alpha_visc):.2f}",
        fontsize=13
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    zone_colors = ZONE_COLORS

    # ── pannello 1: B e Sigma ─────────────────────────────────────────────────
    ax = axes[0]
    ax2 = ax.twinx()
    for zn, col in zone_colors.items():
        sub = df[df['zone'] == zn]
        ax.semilogy(sub['r'], sub['B0'],    color=col, lw=2,   label=f"B₀ zona {zn}")
        ax2.semilogy(sub['r'], sub['Sigma'], color=col, lw=2, ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]');  ax.set_ylabel('B₀ [u.a.]', color='white')
    ax2.set_ylabel('Σ [u.a.]', color='gray')
    ax.set_title('Profili B₀(r)  —  Σ(r) (tratteggio)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8, loc='upper right')

    # ── pannello 2: k·r ───────────────────────────────────────────────────────
    ax = axes[1]
    for zn, col in zone_colors.items():
        sub = df[df['zone'] == zn]
        valid = sub[sub['k'].notna()]
        ax.semilogy(valid['r'], valid['kr'], color=col, lw=2, label=f"Zona {zn}")
    ax.axhline(0.1, color='gray', ls=':', lw=1, alpha=0.7)
    ax.axhline(10,  color='gray', ls=':', lw=1, alpha=0.7, label='range fisico')
    # evidenzia regioni AEI valide
    aei = df[df['aei_valid']]
    ax.scatter(aei['r'], aei['kr'], color='yellow', s=8, zorder=5, alpha=0.5, label='AEI valida')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]');  ax.set_ylabel('k·r  (adimensionale)')
    ax.set_title('Wavenumber adimensionale k·r(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    # ── pannello 3: beta ──────────────────────────────────────────────────────
    ax = axes[2]
    for zn, col in zone_colors.items():
        sub = df[df['zone'] == zn]
        ax.semilogy(sub['r'], sub['beta'], color=col, lw=2, label=f"Zona {zn}")
    ax.axhline(1.0, color='red', ls='--', lw=1.2, label='β = 1')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]');  ax.set_ylabel('β')
    ax.set_title('Magnetizzazione β(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    # ── pannello 4: dQ/dr ─────────────────────────────────────────────────────
    ax = axes[3]
    for zn, col in zone_colors.items():
        sub = df[df['zone'] == zn]
        ax.plot(sub['r'], sub['dQdr'], color=col, lw=2, label=f"Zona {zn}")
    ax.axhline(0, color='red', ls='--', lw=1.2, label='dQ/dr = 0')
    # shading per AEI valida
    ax.fill_between(df['r'], df['dQdr'],
                    where=df['aei_valid'], color='yellow', alpha=0.15,
                    label='AEI valida')
    ax.set_xscale('log')
    ax.set_xlabel('r [rg]');  ax.set_ylabel('dQ/dr  [u.a.]')
    ax.set_title('Condizione di shear dQ/dr(r)')
    _zone_vlines(ax, meta)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return fig


def plot_aei_validity_map(df, meta, figsize=(14, 4)):
    """
    Mappa a barre orizzontali che mostra, per ogni r,
    quali criteri AEI sono soddisfatti.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    criteria = [
        ('k_valid',     'k fisico  (0.1 ≤ kr ≤ 10)', '#a78bfa'),
        ('beta_valid',  'β ≤ 1',                      '#ef4444'),
        ('shear_valid', 'dQ/dr > 0',                  '#f59e0b'),
        ('aei_valid',   'Tutti e tre  →  AEI',         '#22c55e'),
    ]
    for ax, (col, label, color) in zip(axes, criteria):
        ax.fill_between(df['r'], df[col].astype(float),
                        step='mid', color=color, alpha=0.7)
        ax.set_ylim(0, 1.3);  ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9, color=color)
        ax.axvline(meta['r_AB'], color='#f97316', ls='--', lw=0.8, alpha=0.6)
        ax.axvline(meta['r_BC'], color='#3b82f6', ls='--', lw=0.8, alpha=0.6)
        ax.grid(False)

    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('r [rg]')
    fig.suptitle(
        f"Mappa di validità AEI — a={meta['a']:.2f},  "
        f"B₀₀={meta['B00']:.1e},  Σ₀={meta['Sigma0']:.1e}",
        fontsize=11
    )
    plt.tight_layout()
    return fig


def plot_summary_table(df, meta, alpha_visc=0.1):
    """Stampa una tabella riassuntiva per zona."""
    print("\n" + "="*75)
    print(f"  SINTESI DISCO SS UNIFICATO  |  a={meta['a']:.2f}  "
          f"B00={meta['B00']:.1e}  Σ0={meta['Sigma0']:.1e}")
    print(f"  ṁ={meta.get('mdot', float('nan')):.4f}  "
          f"α={meta.get('alpha', alpha_visc):.2f}  "
          f"r_ISCO={meta['r_ISCO']:.2f}  r_AB={meta['r_AB']:.1f}  "
          f"r_BC={meta['r_BC']:.0f}  [rg]")
    print("="*75)
    header = f"{'Zona':>5} {'N':>5} {'k fisico':>10} {'β≤1':>8} {'shear>0':>9} {'AEI ok':>8} "
    header += f"{'B∝r^α':>8} {'Σ∝r^α':>8} {'β∝r^α':>8}"
    print(header)
    print("-"*75)

    for zn in ZONE_NAMES:
        sub = df[df['zone'] == zn]
        N   = len(sub)
        if N == 0:
            print(f"{zn:>5} {'—':>5}")
            continue

        def pct(col): return f"{sub[col].sum()/N*100:.0f}%"

        # power-law slopes (log-log fit)
        def slope(xcol, ycol):
            d = sub[(sub[xcol]>0) & (sub[ycol]>0) & sub[ycol].notna()]
            if len(d) < 4: return "—"
            x = np.log10(d[xcol].values); y = np.log10(d[ycol].values)
            mx,my = x.mean(), y.mean()
            sl = np.dot(x-mx, y-my) / np.dot(x-mx, x-mx)
            return f"{sl:+.3f}"

        row = (f"{zn:>5} {N:>5} {pct('k_valid'):>10} {pct('beta_valid'):>8} "
               f"{pct('shear_valid'):>9} {pct('aei_valid'):>8} "
               f"{slope('r','B0'):>8} {slope('r','Sigma'):>8} {slope('r','beta'):>8}")
        print(row)
    print("="*75)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  WRAPPER PER IL NOTEBOOK  (drop-in replacement di find_rossby_matches)
# ═══════════════════════════════════════════════════════════════════════════════

def find_rossby_matches_SS(param_dict, alpha_visc, m, hr,
                           check_k=True, check_beta=True, check_shear=True, M=M_BH):
    """
    Equivalente di find_rossby_matches() del notebook,
    ma usa il modello di disco S&S unificato a tre zone.

    Il param_dict ha lo stesso formato di prima:
        {'a': [...], 'r': [...], 'B00': [...], 'Sigma0': [...]}

    Le frontiere r_AB/r_BC e le normalizzazioni di B e Σ sono calcolate
    internamente per ogni combinazione (a, B00, Sigma0).

    Returns
    -------
    df : DataFrame  (stesso schema di find_rossby_matches)
    """

    param_vectors, mesh_arrays = create_param_grid(param_dict, mesh=True)
    labels = list(param_dict.keys())
    param_mesh = {lab: arr for lab, arr in zip(labels, mesh_arrays)}

    r_mesh    = param_mesh['r']
    a_mesh    = param_mesh['a']
    B00_mesh  = param_mesh['B00']
    S0_mesh   = param_mesh['Sigma0']

    # Per ogni punto della griglia calcola zona, profili, k
    shape = r_mesh.shape
    k_arr     = np.full(shape, np.nan)
    beta_arr  = np.full(shape, np.nan)
    dQdr_arr  = np.full(shape, np.nan)
    zone_arr  = np.full(shape, '', dtype=object)

    # Itera sulle combinazioni uniche di (a, B00, Sigma0)
    # per calcolare le frontiere una sola volta per combinazione
    a_vals   = param_vectors['a']
    B00_vals = param_vectors['B00']
    S0_vals  = param_vectors['Sigma0']

    Rg = Rg_SUN * M

    for ia, a_val in enumerate(a_vals):
        rISCO = r_isco(a_val)
        for iB, B00_val in enumerate(B00_vals):
            for iS, S0_val in enumerate(S0_vals):
                # frontiere e normalizzazioni per questa combinazione
                r_AB, r_BC, mdot = ss_boundaries(a_val, S0_val, alpha=alpha_visc, M=M)
                norms      = compute_norms(a_val, B00_val, S0_val, r_AB, r_BC)

                # indice lungo r per questo (a, B00, Sigma0)
                # (dipende dall'ordine dei parametri nel meshgrid)
                # recupera la slice giusta
                idx_a  = np.where(a_mesh   == a_val  )[0] if a_mesh.ndim == 1 else None

                # più robusto: usa la griglia r per questo "stack"
                # il meshgrid ha ordinamento param_dict.keys()
                # troviamo gli indici corrispondenti
                # costruiamo una slice multi-dimensionale
                param_keys = list(param_dict.keys())
                slices = []
                for lab in param_keys:
                    if lab == 'a':      slices.append(ia)
                    elif lab == 'r':    slices.append(slice(None))
                    elif lab == 'B00':  slices.append(iB)
                    elif lab == 'Sigma0': slices.append(iS)
                    else:               slices.append(slice(None))
                sl = tuple(slices)

                r_slice = r_mesh[sl]
                # converti r_slice in 1D per il calcolo
                r_flat  = r_slice.ravel() if hasattr(r_slice,'ravel') else np.array([r_slice])

                zi_flat = zone_index(r_flat, r_AB, r_BC)
                B0_flat = B0_disk(r_flat, norms, r_AB, r_BC)
                S_flat  = Sigma_disk(r_flat, norms, r_AB, r_BC)
                cs_flat = sound_speed_disk(r_flat, a_val, hr, M)

                k_flat   = solve_k_disk(r_flat, a_val, B0_flat, S_flat, cs_flat, m=m, M=M)
                k_flat_rg = k_flat * Rg
                beta_flat = compute_beta_disk(r_flat, a_val, B0_flat, S_flat, cs_flat, hr, M)
                dQ_flat   = compute_dQdr_disk(r_flat, a_val, norms, r_AB, r_BC, M)

                # ri-scrivi nella griglia
                orig_shape = r_slice.shape if hasattr(r_slice,'shape') else ()
                k_arr  [sl] = k_flat_rg.reshape(orig_shape) if orig_shape else k_flat_rg[0]
                beta_arr[sl] = beta_flat.reshape(orig_shape) if orig_shape else beta_flat[0]
                dQdr_arr[sl] = dQ_flat.reshape(orig_shape)  if orig_shape else dQ_flat[0]
                zone_arr[sl] = np.array([ZONE_NAMES[i] for i in zi_flat]).reshape(orig_shape) \
                               if orig_shape else ZONE_NAMES[zi_flat[0]]

    kr_arr = k_arr * r_mesh

    # maschera base: k reale e positivo + sopra ISCO
    mask = np.isfinite(k_arr) & (k_arr > 0)
    a_vec   = param_vectors['a']
    isco    = r_isco(a_vec)
    r_isco_nd = isco.reshape(-1, *[1]*(r_mesh.ndim - 1))
    mask &= (r_mesh >= r_isco_nd)

    if check_k:
        mask &= (kr_arr >= 0.1) & (kr_arr <= 10)
    if check_beta:
        mask &= (beta_arr <= 1.0)
    if check_shear:
        mask &= (dQdr_arr > 0)

    rows = []
    for idx in np.argwhere(mask):
        t = tuple(idx)
        row = {lab: arr[t] for lab, arr in param_mesh.items()}
        row['k']      = k_arr[t]
        row['kperr']  = kr_arr[t]
        row['beta']   = beta_arr[t]
        row['dQ_dr']  = dQdr_arr[t]
        row['zone']   = zone_arr[t]
        row['m']      = m
        row['hr']     = hr
        rows.append(row)

    return pd.DataFrame(rows)