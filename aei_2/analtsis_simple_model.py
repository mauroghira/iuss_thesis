"""
aei_cavity_analysis.py
======================
AEI cavity analysis for the Simple-v1 disk model (alpha_B=5/4, alpha_S=3/5).

For each configuration (m, H/r) in {1,2} × {0.05, 0.001}:

  Step A — dQ/dr profile [ISCO → OLR]
    Q = Omega_phi(r) * Sigma/B0^2 ∝ (Sigma0/B00^2) * nu_phi(r) * r^(2αB−αS)
    The SIGN of dQ/dr is independent of B00, Sigma0, and H/r  →  computed
    once per (a, m).  Condition: dQ/dr > 0 for ALL r ∈ [ISCO, OLR].

  Step B — beta ≤ 1 in [ISCO → OLR]
    beta ∝ Sigma0/B00^2 * f(r, a, H/r)
    Depends on (B00, Sigma0, H/r).  Condition: beta(r) ≤ 1 ∀r ∈ [ISCO, OLR].

  Step C — WKB dispersion relation in cavity [ISCO → ILR]
    Solve A k² + B k + C = 0 at each r and check 1 ≤ k ≤ 1/hr(r) ∀r.

Resonance radii (already in aei_common):
    r_ILR : ω̃ + κ = 0  (inner Lindblad resonance — cavity boundary)
    r_OLR : ω̃ − κ = 0  (outer Lindblad resonance)
    r_CR  : ω̃ = 0       (corotation radius)

Plots produced:
    dQdr_profiles.pdf          dQ/dr shape, all configs at a=0.99 + multi-spin panel
    cavity_summary.pdf         pass-rate by spin, k histogram, B00/Sigma0 heatmap
    cavity_k_profiles.pdf      k(r) inside cavity for representative valid solutions
    cavity_beta_profiles.pdf   beta(r) for valid solutions
    cavity_paramspace.pdf      scatter valid solutions in (B00,Sigma0) per spin
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

from aei_setup import (
    solve_k_aei, compute_beta, compute_dQdr,
    r_ilr, r_olr, r_corotation, _make_interp,
    make_disk, get_resonances,
    ALP_B, ALP_S, CONFIGS , A_GRID, B00_GRID, 
    SIGMA0_GRID, B00_REF, SIGMA0_REF, N_R, N_R_CAV
)
from diagnosi import diagnose_wkb_failure, diagnose_beta_failure
from cavity_plots import (
    plot_paramspace_heatmap, plot_dQdr_at_CR,
    plot_wkb_heatmap_comparison
)

sys.path.append('..')
from setup import M_BH, NU0, r_isco, Rg_SUN
from aei_2.simple_disc import disk_model_simple


def dQdr_profile(r_vec, a, B00, Sigma0, hor):
    """
    Compute dQ/dr = d/dr [Omega_phi * Sigma / B0²].
    The sign is independent of B00, Sigma0 (positive multiplicative constant).
    H/r (hor) does not enter Q at all.
    """
    res = make_disk(r_vec, a, B00, Sigma0, hor)
    B0i = _make_interp(r_vec, res[0])
    Si  = _make_interp(r_vec, res[1])
    return compute_dQdr(r_vec, a, B0i, Si, M_BH)

# ═══════════════════════════════════════════════════════════════════════════════
# 2 ── MAIN SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def cavity_scan(mm, hor, verbose=False):
    """
    Three-step filter over the full (a, B00, Sigma0) grid.

    Step A  dQ/dr > 0 for ALL r ∈ [CR-eps, CR+eps]
            Computed ONCE per (a, mm) — independent of B00, Sigma0, hor.
            The B00/Sigma0 loop is entirely skipped for failing spins.

    Step B  beta(r) ≤ 1 for ALL r ∈ [ISCO, OLR]
            Depends on Sigma0/B00² and hor.

    Step C  1 ≤ k(r) ≤ 1/hr(r) for ALL r ∈ [ISCO, ILR]
            Solves the AEI dispersion relation point by point.

    Returns
    -------
    pd.DataFrame with one row per (a, B00, Sigma0).
    Columns: a, B00, Sigma0, r_ILR, r_OLR, r_CR,
             dQdr_CR, pass_shear, pass_beta, pass_wkb,
             beta_max, k_med, k_min_val, k_max_val, frac_wkb_ok
    """
    records = []
    n_spin = len(A_GRID)
    n_par  = len(B00_GRID) * len(SIGMA0_GRID)

    for i_a, a_val in enumerate(A_GRID):
        rISCO = float(r_isco(a_val))
        r_ILR, r_OLR, r_CR = get_resonances(a_val, mm)

        if verbose:
            print(f'  spin {i_a+1}/{n_spin}  a={a_val:.2f} '
                  f'| r_ILR={r_ILR:.1f}  r_OLR={r_OLR:.1f}  r_CR={r_CR:.1f}',
                  end='  ')

        # ── sanity check on resonance radii ───────────────────────────────
        if not (np.isfinite(r_OLR) and np.isfinite(r_ILR)):
            if verbose:
                print('SKIP (resonances not found)')
            # fill records with failed rows
            for B00 in B00_GRID:
                for S0 in SIGMA0_GRID:
                    records.append(dict(a=a_val, B00=B00, Sigma0=S0,
                                        r_ILR=np.nan, r_OLR=np.nan, r_CR=r_CR,
                                        dQdr_CR=np.nan,
                                        pass_shear=False, pass_beta=False,
                                        pass_wkb=False))
            continue

        if r_ILR <= rISCO * 1.001:
            if verbose:
                print('SKIP (ILR inside ISCO)')
            for B00 in B00_GRID:
                for S0 in SIGMA0_GRID:
                    records.append(dict(a=a_val, B00=B00, Sigma0=S0,
                                        r_ILR=r_ILR, r_OLR=r_OLR, r_CR=r_CR,
                                        dQdr_CR=np.nan,
                                        pass_shear=False, pass_beta=False,
                                        pass_wkb=False))
            continue

        # ── Step A: dQ/dr (independent of B00, Sigma0, hor) ───────────────
        r_olr_cap = min(r_OLR, rISCO * 500.0)
        r_full    = np.geomspace(rISCO*1.001, r_olr_cap, N_R)

        r_cr_eps = np.linspace(r_CR*0.95, r_CR*1.05, 20)
        dq_full  = dQdr_profile(r_cr_eps, a_val, B00_REF, SIGMA0_REF, hor)
        dqdr_at_CR = float(np.interp(r_CR, r_cr_eps, dq_full))
        pass_shear = bool(dqdr_at_CR > 0)

        if verbose:
            shear_str = 'SHEAR OK' if pass_shear else 'shear FAIL'
            print(shear_str)

        if not pass_shear:
            for B00 in B00_GRID:
                for S0 in SIGMA0_GRID:
                    records.append(dict(a=a_val, B00=B00, Sigma0=S0,
                                        r_ILR=r_ILR, r_OLR=r_OLR, r_CR=r_CR,
                                        dQdr_CR=dqdr_at_CR,
                                        pass_shear=False, pass_beta=False,
                                        pass_wkb=False))
            continue

        # ── radial grids for beta & cavity ─────────────────────────────────
        r_ilr_cap = min(r_ILR*0.98, r_olr_cap)
        r_cavity  = np.geomspace(rISCO*1.01, r_ilr_cap, N_R_CAV)

        # ── Steps B & C: loop over (B00, Sigma0) ──────────────────────────
        for B00 in B00_GRID:
            for S0 in SIGMA0_GRID:

                # ── Step B: beta ≤ 1 in [ISCO, OLR] ─────────────────────
                res_full = make_disk(r_full, a_val, B00, S0, hor)
                hr_full  = res_full[3]   # per-point H/r from model
                beta_full = compute_beta(res_full[0], res_full[1], res_full[2],
                                         r_full, hr_full, M_BH)
                pass_beta = bool(np.all(beta_full <= 1.0))
                beta_max  = float(np.max(beta_full))

                if not pass_beta:
                    records.append(dict(a=a_val, B00=B00, Sigma0=S0,
                                        r_ILR=r_ILR, r_OLR=r_OLR, r_CR=r_CR,
                                        dQdr_CR=dqdr_at_CR,
                                        pass_shear=True, pass_beta=False,
                                        pass_wkb=False, beta_max=beta_max))
                    continue

                # ── Step C: WKB in cavity [ISCO, ILR] ────────────────────
                res_cav  = make_disk(r_cavity, a_val, B00, S0, hor)
                B0_c, Sg_c, cs_c, hr_c = res_cav[:4]

                k_arr = solve_k_aei(r_cavity, a_val, B0_c, Sg_c, cs_c,
                                    m=mm, M=M_BH)
                k_max_arr = 1 / np.maximum(hr_c, 1e-10)  # 1/hr(r)

                wkb_ok   = (np.isfinite(k_arr)
                            & (k_arr >= mm)
                            & (k_arr <= k_max_arr))
                pass_wkb = bool(wkb_ok.mean() >= 0.95)
                frac_ok  = float(wkb_ok.sum() / len(wkb_ok))

                k_valid = k_arr[np.isfinite(k_arr)]
                records.append(dict(
                    a=a_val, B00=B00, Sigma0=S0,
                    r_ILR=r_ILR, r_OLR=r_OLR, r_CR=r_CR,
                    dQdr_CR=dqdr_at_CR,
                    pass_shear=True, pass_beta=True, pass_wkb=pass_wkb,
                    beta_max=beta_max,
                    frac_wkb_ok=frac_ok,
                    k_med=float(np.nanmedian(k_arr)),
                    k_min_val=float(np.nanmin(k_arr)) if k_valid.size else np.nan,
                    k_max_val=float(np.nanmax(k_arr)) if k_valid.size else np.nan,
                ))

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ── RUN ALL CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run cavity_scan for all 4 (m, H/r) configurations."""
    all_results = {}
    for cfg in CONFIGS:
        mm, hor = cfg['mm'], cfg['hor']
        print(f"\n{'='*65}")
        print(f"  Config: m={mm}, H/r={hor}")
        print(f"{'='*65}")
        df = cavity_scan(mm, hor, verbose=False)
        all_results[(mm, hor)] = df

        n      = len(df)
        n_sh   = df['pass_shear'].sum()
        n_be   = df.get('pass_beta',  pd.Series(dtype=bool)).sum() if 'pass_beta' in df else 0
        n_wkb  = df.get('pass_wkb',   pd.Series(dtype=bool)).sum() if 'pass_wkb'  in df else 0
        print(f'\n  Total combos : {n}')
        print(f'  Pass shear   : {n_sh:>6}  ({100*n_sh/n:.1f}%)')
        print(f'  + beta ≤ 1   : {n_be:>6}  ({100*n_be/n:.1f}%)')
        print(f'  + WKB cavity : {n_wkb:>6}  ({100*n_wkb/n:.1f}%)')

    return all_results


def identify_worst(all_results):
    """Return the CONFIGS entry with the lowest fraction of WKB-valid solutions."""
    fracs = {}
    for cfg in CONFIGS:
        key = (cfg['mm'], cfg['hor'])
        df  = all_results[key]
        fracs[key] = df['pass_wkb'].sum() / len(df) if len(df) > 0 else 1.0
    worst_key = min(fracs, key=fracs.get)
    worst_cfg = next(c for c in CONFIGS if (c['mm'], c['hor']) == worst_key)
    print(f'\n  Worst config: m={worst_cfg["mm"]}, H/r={worst_cfg["hor"]}'
          f'  (valid fraction = {fracs[worst_key]:.3f})')
    return worst_cfg

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=' * 65)
    print('  AEI Cavity Analysis — Simple-v1')
    print(f'  alpha_B = {ALP_B}  |  alpha_S = {ALP_S}')
    print(f'  Spins  : {len(A_GRID)} values in [{A_GRID[0]:.2f}, {A_GRID[-1]:.2f}]')
    print(f'  B00    : {len(B00_GRID)} values in [{B00_GRID[0]:.0e}, {B00_GRID[-1]:.0e}] G')
    print(f'  Sigma0 : {len(SIGMA0_GRID)} values in [{SIGMA0_GRID[0]:.0e}, {SIGMA0_GRID[-1]:.0e}] g/cm²')
    print(f'  Configs: {[(c["mm"], c["hor"]) for c in CONFIGS]}')
    print('=' * 65)

    """
    # quick diagnosis at median-ish parameters — run BEFORE the full scan
    print('\n> WKB failure diagnosis...')
    diagnose_wkb_failure(mm=1, hor=0.05,  a_val=0.0, B00=1e4, Sigma0=1e4)
    diagnose_wkb_failure(mm=1, hor=0.001, a_val=0.0, B00=1e4, Sigma0=1e4)
    print('\n> Beta failure diagnosis...')
    diagnose_beta_failure(mm=1, hor=0.05, a_val=0.0,
                        B00=B00_GRID[-1], Sigma0=SIGMA0_GRID[0])
    exit()
    #"""

    # ── step 1: full scan ─────────────────────────────────────────────────────
    print('\n> Running cavity scan for all configs...')
    all_results = run_all()

    #cambia configurazioni
    print('\n> Plot 1: paramspace heatmap...')
    plot_paramspace_heatmap(all_results, cfg_index=1)
    #print('\n> Plot 2: dQ/dr at corotation vs spin...')
    #plot_dQdr_at_CR(hor=0.001)

    print('\n> Plot C: WKB heatmap comparison (due H/r)...')
    plot_wkb_heatmap_comparison(all_results, mm=2)

    print('\n  Done!')