"""
nt_boundary_comparison.py
=========================
Confronto tra due metodi per trovare le frontiere di zona del disco NT:

  Metodo 1 (fisico)  — nt_boundaries()  già in nt_disc.py
      r_AB: f_AB(r) = 1  →  P_rad = P_gas   (zona A valutata al crossing)
      r_BC: f_BC(r) = 1  →  τ_ff  = τ_es   (zona B valutata al crossing)

  Metodo 2 (Σ-crossing)  — nt_boundaries_sigma()  definito qui
      r_AB: Σ_A(r) = Σ_B(r)   (crossing delle due formule di densità)
      r_BC: Σ_B(r) = Σ_C(r)   (crossing delle due formule di densità)

Funzioni esportate
------------------
  nt_boundaries_sigma(a, mdot, alpha, M)
      → r_AB_s, r_BC_s, zone_present

  compare_boundaries(a, mdot, alpha, M, verbose)
      → DataFrame con entrambi i metodi e differenze relative

  plot_boundary_comparison(a, mdot, alpha, M)
      → figura matplotlib con i profili Σ e i due set di frontiere

  scan_boundary_comparison(a_vals, mdot_vals, alpha, M)
      → DataFrame della scansione su griglia (a, mdot)

IMPORTANTE
----------
  Il crossing di Σ NON è fisicamente equivalente a P_rad = P_gas o τ_ff = τ_es.
  Le tre zone hanno dipendenze diverse da (α, ṁ, m) proprio perché la struttura
  termica cambia: Σ_A ∝ α⁻¹ ṁ⁻¹ vs Σ_B ∝ α⁻⁴/⁵ ṁ³/⁵.  Il crossing di Σ è
  quindi un proxy approssimato che dipende dai prefattori assoluti delle formule.
  Usare questo confronto come test di consistenza: se i due metodi danno r molto
  diversi, i prefattori NT potrebbero avere errori.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── importa dal pacchetto locale ────────────────────────────────────────────
# (adatta il path se il progetto ha una struttura diversa)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nt_disc import (
    nt_factors,
    nt_boundaries,
    Sigma_A_NT,
    Sigma_B_NT,
    Sigma_C_NT,
)
from setup import r_isco, M_BH
from aei_common import ALPHA_VISC


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  BISEZIONE GENERICA  (identica a quella in nt_disc per coerenza)
# ═══════════════════════════════════════════════════════════════════════════════

def _bisect(func, r_lo, r_hi, n_scan=600, n_bisect=60):
    """
    Trova r in [r_lo, r_hi] dove func(r) = 0  (primo zero dal basso).

    Parametri
    ----------
    func     : callable   f(array) → array  (zero-crossing cercato)
    r_lo     : float      estremo inferiore
    r_hi     : float      estremo superiore
    n_scan   : int        punti di scansione iniziale (geomspace)
    n_bisect : int        iterazioni di bisezione

    Restituisce
    -----------
    float | None   raggio del crossing, o None se non trovato
    """
    r_arr = np.geomspace(r_lo, r_hi, n_scan)
    f_arr = func(r_arr)
    sc    = np.where(np.diff(np.sign(f_arr)))[0]
    if len(sc) == 0:
        return None
    a_r, b_r = r_arr[sc[0]], r_arr[sc[0] + 1]
    for _ in range(n_bisect):
        mid = 0.5 * (a_r + b_r)
        if func(np.array([mid]))[0] < 0:
            a_r = mid
        else:
            b_r = mid
    return float(0.5 * (a_r + b_r))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BOUNDARIES DA SIGMA-CROSSING
# ═══════════════════════════════════════════════════════════════════════════════

def nt_boundaries_sigma(a, mdot, alpha=ALPHA_VISC, M=M_BH):
    """
    Frontiere r_AB e r_BC trovate imponendo la continuità di Σ tra zone adiacenti.

    Condizioni:
      r_AB :  Σ_A(r) = Σ_B(r)   →   log(Σ_A/Σ_B) = 0
      r_BC :  Σ_B(r) = Σ_C(r)   →   log(Σ_B/Σ_C) = 0

    NOTA CONCETTUALE
    ----------------
    Questa condizione NON coincide in generale con P_rad = P_gas (per r_AB)
    o τ_ff = τ_es (per r_BC).  Σ_A e Σ_B hanno dipendenze diverse da α e ṁ
    perché derivano da strutture termiche diverse.  Il crossing di Σ è un
    indicatore geometrico del punto in cui la formula "interna" smette di
    dominare, non una condizione fisica diretta.

    Parametri
    ----------
    a     : float   spin adimensionale [−1, 1]
    mdot  : float   ṁ = Ṁ/Ṁ_Edd
    alpha : float   parametro di viscosità α
    M     : float   massa BH [M_sun]

    Restituisce
    -----------
    r_AB_s      : float   frontiera A-B dal Σ-crossing [r_g]
    r_BC_s      : float   frontiera B-C dal Σ-crossing [r_g]
    zone_present: dict    {'A': bool, 'B': bool, 'C': bool}
    """
    rISCO = float(r_isco(a))

    # ── r_AB: Σ_A = Σ_B ─────────────────────────────────────────────────────
    def f_AB_sigma(r_arr):
        SA = Sigma_A_NT(r_arr, a, mdot, alpha)
        SB = Sigma_B_NT(r_arr, a, mdot, alpha, M)
        # uso log-ratio per stabilità numerica (entrambe > 0)
        return np.log(SA / np.maximum(SB, 1e-300))

    # stima limite superiore: nel limite newtoniano Σ_A = Σ_B dà
    # r_AB ~ (5 α⁻¹ ṁ⁻¹) / (9e4 α⁻⁴/⁵ ṁ³/⁵)  con esponenti r
    # => r_AB^{(3/2 + 3/5)} = cost  → r_AB^{21/10} ~ cost
    # usa lo stesso upper-bound del metodo fisico, ampliato
    r_AB_est = (alpha**0.25 * M**0.25 * mdot**2 / 4e-6)**(8.0/21)
    r_AB_hi  = max(r_AB_est * 5.0, rISCO * 20.0)

    # controlla se Σ_A > Σ_B già all'ISCO (zona A assente)
    f_at_ISCO = f_AB_sigma(np.array([rISCO * 1.001]))[0]
    if f_at_ISCO <= 0.0:
        # Σ_A ≤ Σ_B all'ISCO: zona A non domina mai → assente
        r_AB_s = rISCO
        zone_A = False
    else:
        r_AB_s = _bisect(f_AB_sigma, rISCO * 1.001, r_AB_hi)
        if r_AB_s is None:
            r_AB_s = r_AB_hi   # non trovato: usa upper bound
        r_AB_s = float(max(r_AB_s, rISCO))
        zone_A = True

    # ── r_BC: Σ_B = Σ_C ─────────────────────────────────────────────────────
    def f_BC_sigma(r_arr):
        SB = Sigma_B_NT(r_arr, a, mdot, alpha, M)
        SC = Sigma_C_NT(r_arr, a, mdot, alpha, M)
        return np.log(SB / np.maximum(SC, 1e-300))

    r_BC_est = (mdot / 2e-6)**(2.0/3)
    r_BC_hi  = max(r_BC_est * 5.0, r_AB_s * 20.0)

    f_at_rAB = f_BC_sigma(np.array([r_AB_s * 1.001]))[0]
    if f_at_rAB <= 0.0:
        r_BC_s = r_AB_s
        zone_B = False
    else:
        r_BC_s = _bisect(f_BC_sigma, r_AB_s * 1.001, r_BC_hi)
        if r_BC_s is None:
            r_BC_s = r_BC_hi
        r_BC_s = float(max(r_BC_s, r_AB_s))
        zone_B = True

    return r_AB_s, r_BC_s, {'A': zone_A, 'B': zone_B, 'C': True}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CONFRONTO SCALARE
# ═══════════════════════════════════════════════════════════════════════════════

def compare_boundaries(a, mdot, alpha=ALPHA_VISC, M=M_BH, verbose=True):
    """
    Confronta i due metodi per un singolo set di parametri.

    Restituisce
    -----------
    dict con chiavi:
      r_ISCO
      r_AB_phys, r_BC_phys   (metodo fisico, già in nt_boundaries)
      r_AB_sig,  r_BC_sig    (metodo Σ-crossing)
      delta_AB, delta_BC     differenza relativa  (r_sig - r_phys) / r_phys
      zone_phys, zone_sig    dict {'A','B','C': bool}
    """
    rISCO = float(r_isco(a))
    r_AB_p, r_BC_p, zp = nt_boundaries(a, mdot, alpha=alpha, M=M)
    r_AB_s, r_BC_s, zs = nt_boundaries_sigma(a, mdot, alpha=alpha, M=M)

    d_AB = (r_AB_s - r_AB_p) / max(r_AB_p, rISCO)
    d_BC = (r_BC_s - r_BC_p) / max(r_BC_p, rISCO)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  a={a:.3f}  ṁ={mdot:.3e}  α={alpha}  M={M:.2e} M☉")
        print(f"{'─'*60}")
        print(f"  r_ISCO  = {rISCO:.3f} rg")
        print(f"{'─'*60}")
        print(f"  {'':12s}  {'fisico':>12s}  {'Σ-crossing':>12s}  {'Δrel':>8s}")
        print(f"  {'r_AB':12s}  {r_AB_p:12.3f}  {r_AB_s:12.3f}  {d_AB:+8.3f}")
        print(f"  {'r_BC':12s}  {r_BC_p:12.1f}  {r_BC_s:12.1f}  {d_BC:+8.3f}")
        print(f"{'─'*60}")
        print(f"  Zone fisico:     A={zp['A']}  B={zp['B']}  C={zp['C']}")
        print(f"  Zone Σ-cross:    A={zs['A']}  B={zs['B']}  C={zs['C']}")
        print(f"{'═'*60}")

    return dict(
        r_ISCO  = rISCO,
        r_AB_phys=r_AB_p, r_BC_phys=r_BC_p,
        r_AB_sig =r_AB_s, r_BC_sig =r_BC_s,
        delta_AB = d_AB,  delta_BC  = d_BC,
        zone_phys= zp,    zone_sig  = zs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOT DIAGNOSTICO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_boundary_comparison(a, mdot, alpha=ALPHA_VISC, M=M_BH,
                             r_min_fac=1.01, r_max=1e4, n_pts=600,
                             figsize=(14, 9)):
    """
    Figura diagnostica con 3 pannelli:

      Pannello 1 (alto):  Σ_A, Σ_B, Σ_C vs r  in scala log-log
                          Vengono indicati entrambi i set di frontiere.

      Pannello 2 (sinistra basso):  Σ_A/Σ_B  e  f_AB_fisica vs r
                                    Il crossing fisico e quello Σ sono marcati.

      Pannello 3 (destra basso):    Σ_B/Σ_C  e  f_BC_fisica vs r
    """
    rISCO = float(r_isco(a))
    r_arr = np.geomspace(rISCO * r_min_fac, r_max, n_pts)

    # ── calcola Σ per ogni zona su tutto il range ────────────────────────────
    SA = Sigma_A_NT(r_arr, a, mdot, alpha)
    SB = Sigma_B_NT(r_arr, a, mdot, alpha, M)
    SC = Sigma_C_NT(r_arr, a, mdot, alpha, M)

    # ── funzioni fisiche di crossing (= 1 alla frontiera) ───────────────────
    def f_AB_phys(r_arr):
        f   = nt_factors(r_arr, a)
        rel = (np.maximum(f['A'], 1e-30)**(-5.0/2) * f['B']**(9.0/2)
               * np.maximum(f['D'], 1e-30) * np.maximum(f['E'], 1e-30)**(5.0/4)
               / np.maximum(f['Q'], 1e-30)**2)
        return 4e-6 * alpha**(-0.25) * M**(-0.25) * mdot**(-2) \
               * r_arr**(21.0/8) * rel

    def f_BC_phys(r_arr):
        f   = nt_factors(r_arr, a)
        rel = (np.maximum(f['A'], 1e-30)**(-1) * f['B']**2
               * np.sqrt(np.maximum(f['D'], 1e-30))
               * np.sqrt(np.maximum(f['E'], 1e-30))
               / np.maximum(f['Q'], 1e-30))
        return 2e-6 * mdot**(-1) * r_arr**(1.5) * rel

    fAB = f_AB_phys(r_arr)
    fBC = f_BC_phys(r_arr)

    # ── boundaries ──────────────────────────────────────────────────────────
    r_AB_p, r_BC_p, zp = nt_boundaries(a, mdot, alpha=alpha, M=M)
    r_AB_s, r_BC_s, zs = nt_boundaries_sigma(a, mdot, alpha=alpha, M=M)

    # ── figura ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, :])   # profilo Σ intero
    ax1 = fig.add_subplot(gs[1, 0])   # crossing A/B
    ax2 = fig.add_subplot(gs[1, 1])   # crossing B/C

    title = (f"Confronto boundaries NT  —  "
             f"a={a:.2f}, ṁ={mdot:.2e}, α={alpha}, M={M:.1e} M☉")
    fig.suptitle(title, fontsize=11, y=1.01)

    # colori per le zone
    cA, cB, cC = '#e05252', '#4c9be8', '#4ec97d'
    c_phys = 'black'
    c_sig  = 'darkorange'

    # ── Pannello 0: profili Σ ─────────────────────────────────────────────
    ax0.loglog(r_arr, SA, color=cA, lw=1.8, label=r'$\Sigma_A$')
    ax0.loglog(r_arr, SB, color=cB, lw=1.8, label=r'$\Sigma_B$')
    ax0.loglog(r_arr, SC, color=cC, lw=1.8, label=r'$\Sigma_C$')

    def _vlines(ax, r_ab, r_bc, color, ls, lbl_suffix, ymin=0.05, ymax=0.95):
        ax.axvline(r_ab, color=color, ls=ls, lw=1.4,
                   label=f'$r_{{AB}}$ {lbl_suffix} = {r_ab:.1f}')
        ax.axvline(r_bc, color=color, ls=ls, lw=1.4,
                   label=f'$r_{{BC}}$ {lbl_suffix} = {r_bc:.0f}')

    _vlines(ax0, r_AB_p, r_BC_p, c_phys, '--', '(fisico)')
    _vlines(ax0, r_AB_s, r_BC_s, c_sig,  ':',  '(Σ-cross)')
    ax0.axvline(rISCO, color='gray', lw=1.0, ls='-.', label=f'ISCO={rISCO:.2f}')

    ax0.set_xlabel(r'$r \ [r_g]$')
    ax0.set_ylabel(r'$\Sigma \ [\mathrm{g\,cm^{-2}}]$')
    ax0.set_title('Profili di densità superficiale per zona')
    ax0.legend(fontsize=8, ncol=3, loc='upper right')
    ax0.grid(True, which='both', alpha=0.25)

    # ── Pannello 1: crossing A/B ──────────────────────────────────────────
    ratio_AB = SA / np.maximum(SB, 1e-300)
    ax1.loglog(r_arr, ratio_AB, color='steelblue', lw=2,
               label=r'$\Sigma_A / \Sigma_B$')
    ax1.loglog(r_arr, np.clip(fAB, 1e-4, 1e4), color='tomato', lw=2,
               ls='--', label=r'$f_{AB}^{\rm phys}$  (= 1 alla frontiera)')
    ax1.axhline(1.0, color='k', lw=1.0, ls='-', label='= 1')
    ax1.axvline(r_AB_p, color=c_phys, lw=1.4, ls='--',
                label=f'$r_{{AB}}$ fisico = {r_AB_p:.2f}')
    ax1.axvline(r_AB_s, color=c_sig,  lw=1.4, ls=':',
                label=f'$r_{{AB}}$ Σ-cross = {r_AB_s:.2f}')
    ax1.axvline(rISCO, color='gray', lw=1.0, ls='-.')
    ax1.set_xlabel(r'$r \ [r_g]$')
    ax1.set_ylabel('rapporto')
    ax1.set_title('Frontiera A/B')
    ax1.set_ylim(1e-2, 1e2)
    ax1.legend(fontsize=7.5)
    ax1.grid(True, which='both', alpha=0.25)

    # ── Pannello 2: crossing B/C ──────────────────────────────────────────
    ratio_BC = SB / np.maximum(SC, 1e-300)
    ax2.loglog(r_arr, ratio_BC, color='steelblue', lw=2,
               label=r'$\Sigma_B / \Sigma_C$')
    ax2.loglog(r_arr, np.clip(fBC, 1e-4, 1e4), color='tomato', lw=2,
               ls='--', label=r'$f_{BC}^{\rm phys}$  (= 1 alla frontiera)')
    ax2.axhline(1.0, color='k', lw=1.0, ls='-', label='= 1')
    ax2.axvline(r_BC_p, color=c_phys, lw=1.4, ls='--',
                label=f'$r_{{BC}}$ fisico = {r_BC_p:.0f}')
    ax2.axvline(r_BC_s, color=c_sig,  lw=1.4, ls=':',
                label=f'$r_{{BC}}$ Σ-cross = {r_BC_s:.0f}')
    ax2.set_xlabel(r'$r \ [r_g]$')
    ax2.set_ylabel('rapporto')
    ax2.set_title('Frontiera B/C')
    ax2.set_ylim(1e-2, 1e2)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, which='both', alpha=0.25)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SCANSIONE SU GRIGLIA (a, mdot)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_boundary_comparison(a_vals, mdot_vals,
                              alpha=ALPHA_VISC, M=M_BH,
                              verbose=True):
    """
    Tabella di confronto su griglia (a, mdot).

    Restituisce
    -----------
    pd.DataFrame con colonne:
        a, mdot, r_ISCO,
        r_AB_phys, r_BC_phys,
        r_AB_sig,  r_BC_sig,
        delta_AB,  delta_BC,
        zone_A_phys, zone_B_phys,
        zone_A_sig,  zone_B_sig
    """
    rows = []
    for a in a_vals:
        for mdot in mdot_vals:
            try:
                res = compare_boundaries(a, mdot, alpha=alpha, M=M,
                                         verbose=False)
                rows.append(dict(
                    a          = a,
                    mdot       = mdot,
                    r_ISCO     = res['r_ISCO'],
                    r_AB_phys  = res['r_AB_phys'],
                    r_BC_phys  = res['r_BC_phys'],
                    r_AB_sig   = res['r_AB_sig'],
                    r_BC_sig   = res['r_BC_sig'],
                    delta_AB   = res['delta_AB'],
                    delta_BC   = res['delta_BC'],
                    zone_A_phys= res['zone_phys']['A'],
                    zone_B_phys= res['zone_phys']['B'],
                    zone_A_sig = res['zone_sig']['A'],
                    zone_B_sig = res['zone_sig']['B'],
                ))
            except Exception as exc:
                rows.append(dict(a=a, mdot=mdot, _error=str(exc)))

    df = pd.DataFrame(rows)

    if verbose:
        cols = ['a', 'mdot', 'r_ISCO',
                'r_AB_phys', 'r_AB_sig', 'delta_AB',
                'r_BC_phys', 'r_BC_sig', 'delta_BC']
        ok_cols = [c for c in cols if c in df.columns]
        print(df[ok_cols].to_string(
            index=False,
            float_format=lambda x: f'{x:.3g}'
        ))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ESEMPIO  (eseguibile direttamente come script)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    a_test    = 0.5
    mdot_test = 0.1
    alpha     = ALPHA_VISC
    M         = M_BH

    print("─── Confronto singolo punto ───")
    compare_boundaries(a_test, mdot_test, alpha, M, verbose=True)

    print("\n─── Scansione griglia ───")
    df = scan_boundary_comparison(
        a_vals    = np.linspace(-0.9, 0.9, 5),
        mdot_vals = np.logspace(-3, 0, 5),
        alpha=alpha, M=M, verbose=True,
    )

    fig = plot_boundary_comparison(a_test, mdot_test, alpha, M)
    fig.savefig('boundary_comparison.png', dpi=150, bbox_inches='tight')
    print("\nFigura salvata in boundary_comparison.png")