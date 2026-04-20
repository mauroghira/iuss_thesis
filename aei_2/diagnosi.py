import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import M_BH, NU0, r_isco, Rg_SUN
from AEI_setups.aei_common import (
    solve_k_aei, compute_beta, compute_dQdr,
    r_ilr, r_olr, r_corotation, _make_interp,
)
from aei_2.simple_disc import disk_model_simple


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

ALP_B = 5 / 4    # Simple-v1 magnetic field exponent
ALP_S = 3 / 5    # Simple-v1 surface density exponent

CONFIGS = [
    dict(mm=1, hor=0.05,  color='#3b82f6', ls='-',  label=r'$m=1,\ H/r=0.05$'),
    dict(mm=1, hor=0.001, color='#f97316', ls='-',  label=r'$m=1,\ H/r=10^{-3}$'),
    dict(mm=2, hor=0.05,  color='#22c55e', ls='--', label=r'$m=2,\ H/r=0.05$'),
    dict(mm=2, hor=0.001, color='#ef4444', ls='--', label=r'$m=2,\ H/r=10^{-3}$'),
]

A_GRID      = np.linspace(-1, 1, 100)   # spin grid
B00_GRID    = np.logspace(1,  8, 32)          # B00 [G]
SIGMA0_GRID = np.logspace(2,  7, 24)          # Sigma0 [g/cm²]

N_R        = 300   # radial points for full profile (ISCO → OLR)
N_R_CAV    = 200   # radial points for cavity (ISCO → ILR)

# Reference values for dQ/dr plots (only sign matters, not amplitude)
B00_REF    = 1e4
SIGMA0_REF = 1e4

# Spins to show in the multi-spin dQ/dr panel
A_SUBPLOT = [-1, -0.5, 0.0, 0.5, 0.90, 1]

def make_disk(r, a, B00, Sigma0, hor):
    """Wrapper for Simple-v1 disk model."""
    return disk_model_simple(r, a, B00, Sigma0,
                             alpha_B=ALP_B, alpha_S=ALP_S, hr=hor, M=M_BH)


def get_resonances(a, mm):
    """Compute (r_ILR, r_OLR, r_CR) for given spin and azimuthal mode."""
    return (r_ilr(a, NU0, mm, M_BH),
            r_olr(a, NU0, mm, M_BH),
            r_corotation(a, NU0, mm, M_BH))


#==============================================================
# DIGNOSI
#==============================================================
def diagnose_wkb_failure(mm=1, hor=0.05, a_val=0.0,
                          B00=1e4, Sigma0=1e4):
    """
    For a single (a, B00, Sigma0, m, H/r) combination, plot k(r) inside
    the cavity alongside the WKB bounds [mm, 1/hr(r)], and break down
    WHY each radial point fails: no real solution (Delta<0), k too low,
    or k too high.

    Call this with representative / median parameters from your grid after
    you notice pass_wkb=0 across the board.
    """
    rISCO = float(r_isco(a_val))
    r_ILR, r_OLR, r_CR = get_resonances(a_val, mm)

    if not np.isfinite(r_ILR) or r_ILR <= rISCO:
        print(f"  No valid ILR for a={a_val} m={mm}"); return

    r_cavity = np.geomspace(rISCO, r_ILR, N_R_CAV)
    res      = make_disk(r_cavity, a_val, B00, Sigma0, hor)
    B0_c, Sg_c, cs_c, hr_c = res[:4]

    k_arr     = solve_k_aei(r_cavity, a_val, B0_c, Sg_c, cs_c, m=mm, M=M_BH)
    k_max_arr = 1.0 / np.maximum(hr_c, 1e-10)
    k_min_val = float(mm)

    no_sol   = ~np.isfinite(k_arr)
    too_low  = np.isfinite(k_arr) & (k_arr < k_min_val)
    too_high = np.isfinite(k_arr) & (k_arr > k_max_arr)
    valid    = np.isfinite(k_arr) & (k_arr >= k_min_val) & (k_arr <= k_max_arr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        rf'WKB failure diagnosis — $m={mm}$, $H/r={hor}$, '
        rf'$a={a_val}$, $B_{{00}}={B00:.0e}$, $\Sigma_0={Sigma0:.0e}$',
        fontsize=11
    )
    r_n = r_cavity / rISCO   # normalised radius

    # ── panel 1: k(r) vs bounds ──────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(r_n, np.where(np.isfinite(k_arr), k_arr, np.nan),
                color='white', lw=2, label='k(r)')
    ax.semilogy(r_n, k_max_arr, color='orange', lw=1.5, ls='--',
                label=r'$k_{\max} = 1/hr(r)$')
    ax.axhline(k_min_val, color='cyan', lw=1.5, ls='--',
               label=rf'$k_{{\min}} = m = {mm}$')
    ax.fill_between(r_n, k_min_val, k_max_arr,
                    color='green', alpha=0.10, label='WKB window')
    # scatter failure modes
    if no_sol.any():
        ax.scatter(r_n[no_sol],
                   np.full(no_sol.sum(), k_min_val * 0.5),
                   color='red', s=12, zorder=5, label=f'No real k  ({no_sol.sum()})')
    if too_low.any():
        ax.scatter(r_n[too_low], k_arr[too_low],
                   color='magenta', s=12, zorder=5, label=f'k < k_min  ({too_low.sum()})')
    if too_high.any():
        ax.scatter(r_n[too_high], k_arr[too_high],
                   color='yellow', s=12, zorder=5, label=f'k > k_max  ({too_high.sum()})')
    if valid.any():
        ax.scatter(r_n[valid], k_arr[valid],
                   color='lime', s=20, zorder=6, label=f'WKB OK  ({valid.sum()})')
    ax.set_xlabel(r'$r/r_{\rm ISCO}$'); ax.set_ylabel('k')
    ax.set_title('k(r) vs WKB window')
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.15)

    # ── panel 2: failure-mode breakdown vs radius ─────────────────────────────
    ax = axes[1]
    ax.bar(r_n, no_sol.astype(float),   width=np.diff(r_n, append=r_n[-1]),
           color='red',     alpha=0.7, label=f'Delta<0  ({no_sol.sum()})',  align='edge')
    ax.bar(r_n, too_low.astype(float),  width=np.diff(r_n, append=r_n[-1]),
           color='magenta', alpha=0.7, label=f'k<k_min  ({too_low.sum()})', align='edge',
           bottom=no_sol.astype(float))
    ax.bar(r_n, too_high.astype(float), width=np.diff(r_n, append=r_n[-1]),
           color='yellow',  alpha=0.7, label=f'k>k_max  ({too_high.sum()})', align='edge',
           bottom=(no_sol | too_low).astype(float))
    ax.set_xlabel(r'$r/r_{\rm ISCO}$'); ax.set_ylabel('failure flag')
    ax.set_title('Failure mode by radius')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # ── panel 3: k / k_max ratio — shows how far k is from the window ────────
    ax = axes[2]
    ratio_lo = k_arr / k_min_val          # want > 1
    ratio_hi = k_arr / k_max_arr          # want < 1
    ax.semilogy(r_n, ratio_lo, color='cyan',   lw=2, label=r'$k / k_{\min}$ (want > 1)')
    ax.semilogy(r_n, ratio_hi, color='orange', lw=2, label=r'$k / k_{\max}$ (want < 1)')
    ax.axhline(1.0, color='white', ls='--', lw=1)
    ax.set_xlabel(r'$r/r_{\rm ISCO}$'); ax.set_ylabel('ratio')
    ax.set_title('Distance from WKB window')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.15)

    # ── text summary ─────────────────────────────────────────────────────────
    n = len(r_cavity)
    print(f"\nWKB diagnosis  m={mm}, H/r={hor}, a={a_val}, B00={B00:.1e}, S0={Sigma0:.1e}")
    print(f"  N radial pts : {n}")
    print(f"  No real k    : {no_sol.sum():>4}  ({100*no_sol.sum()/n:.1f}%)")
    print(f"  k < {k_min_val:.0f} (too low) : {too_low.sum():>4}  ({100*too_low.sum()/n:.1f}%)")
    print(f"  k > 1/hr(too high): {too_high.sum():>4}  ({100*too_high.sum()/n:.1f}%)")
    print(f"  WKB valid    : {valid.sum():>4}  ({100*valid.sum()/n:.1f}%)")
    if np.isfinite(k_arr).any():
        print(f"  k range      : [{np.nanmin(k_arr):.3e}, {np.nanmax(k_arr):.3e}]")
        print(f"  k_max range  : [{k_max_arr.min():.3e}, {k_max_arr.max():.3e}]")

    plt.tight_layout()
    plt.savefig('wkb_diagnosis.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('wkb_diagnosis.png', bbox_inches='tight', dpi=150)
    print("  -> wkb_diagnosis.pdf / .png")
    plt.show()
    return fig


def diagnose_beta_failure(mm=1, hor=0.05, a_val=0.0,
                          B00=1e4, Sigma0=1e4):
    """
    For a single combination, show:
      Panel 1 — beta(r) profile vs 1, coloured by how far above/below
      Panel 2 — beta(r) decomposed: the r-dependent factor f(r) ∝ Sigma*cs²/(H*B0²)
                 vs the Sigma0/B00² prefactor — reveals whether failure is
                 structural (f too large everywhere) or just a parameter choice
      Panel 3 — the minimum B00 needed to achieve beta≤1 at each r, i.e.
                 B00_min(r) = sqrt(8π Sigma(r) cs(r)² / (2 H(r)))
                 compared to your actual B00 → tells you how far you are
    """
    rISCO = float(r_isco(a_val))
    r_ILR, r_OLR, _ = get_resonances(a_val, mm)

    r_end   = min(r_OLR * 1.05 if np.isfinite(r_OLR) else rISCO * 500, rISCO * 500)
    r_vec   = np.geomspace(rISCO * 1.001, r_end, N_R)

    res     = make_disk(r_vec, a_val, B00, Sigma0, hor)
    B0_arr, Sg_arr, cs_arr, hr_arr = res[:4]

    Rg      = Rg_SUN * M_BH
    H_arr   = hr_arr * r_vec * Rg
    beta    = 8 * np.pi * Sg_arr * cs_arr**2 / (2 * H_arr * B0_arr**2)

    # B00 that would be needed to push beta=1 at each r
    B00_needed = np.sqrt(8 * np.pi * Sg_arr * cs_arr**2 / (2 * H_arr))

    # structural factor: beta / (Sigma0/B00²) — pure r-dependence
    ratio_param = Sigma0 / B00**2
    f_structural = beta / ratio_param   # [cm^-2 g^-1 s^2] — same shape for all param combos

    r_n = r_vec / rISCO

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        rf'$\beta$ failure diagnosis — $m={mm}$, $H/r={hor}$, '
        rf'$a={a_val}$, $B_{{00}}={B00:.0e}$, $\Sigma_0={Sigma0:.0e}$',
        fontsize=11
    )

    # ── panel 1: beta(r) ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.semilogy(r_n, beta, color='white', lw=2, label=r'$\beta(r)$')
    ax.axhline(1.0, color='red', ls='--', lw=1.5, label=r'$\beta = 1$')
    if np.isfinite(r_ILR):
        ax.axvline(r_ILR / rISCO, color='cyan',   ls=':', lw=1.2, label='ILR')
    if np.isfinite(r_OLR):
        ax.axvline(r_OLR / rISCO, color='magenta', ls=':', lw=1.2, label='OLR')
    # shade the failing region
    ax.fill_between(r_n, 1, beta,
                    where=(beta > 1), color='red', alpha=0.15, label='beta > 1')
    ax.set_xlabel(r'$r/r_{\rm ISCO}$'); ax.set_ylabel(r'$\beta$')
    ax.set_title(r'$\beta(r)$ profile')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # ── panel 2: structural factor f(r) ──────────────────────────────────────
    ax = axes[1]
    ax.semilogy(r_n, f_structural, color='orange', lw=2,
                label=r'$f(r) = \beta\,/\,(\Sigma_0/B_{00}^2)$')
    # threshold: f must be < 1/ratio_param for beta<=1
    threshold = 1.0 / ratio_param
    ax.axhline(threshold, color='red', ls='--', lw=1.5,
               label=rf'threshold $= B_{{00}}^2/\Sigma_0 = {threshold:.2e}$')
    ax.set_xlabel(r'$r/r_{\rm ISCO}$')
    ax.set_ylabel(r'$f(r) \propto \Sigma c_s^2\,/\,(H B_0^2)$')
    ax.set_title('Structural factor (independent of param scaling)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # ── panel 3: B00 needed vs actual ────────────────────────────────────────
    ax = axes[2]
    ax.semilogy(r_n, B00_needed, color='yellow', lw=2,
                label=r'$B_{00}^{\rm needed}(r)$ for $\beta=1$')
    ax.axhline(B00, color='white', ls='--', lw=1.5,
               label=rf'actual $B_{{00}} = {B00:.0e}$ G')
    # worst point (max B00_needed)
    i_worst = int(np.argmax(B00_needed))
    ax.scatter([r_n[i_worst]], [B00_needed[i_worst]],
               color='red', s=60, zorder=5,
               label=rf'max needed = {B00_needed[i_worst]:.1e} G  @ $r={r_vec[i_worst]:.1f}\,r_g$')
    ax.set_xlabel(r'$r/r_{\rm ISCO}$'); ax.set_ylabel(r'$B_{00}$ [G]')
    ax.set_title(r'$B_{00}$ needed to reach $\beta \leq 1$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.15)

    # ── text summary ─────────────────────────────────────────────────────────
    frac_fail = (beta > 1).mean()
    print(f"\nbeta diagnosis  m={mm}, H/r={hor}, a={a_val}, B00={B00:.1e}, S0={Sigma0:.1e}")
    print(f"  beta range   : [{beta.min():.3e}, {beta.max():.3e}]")
    print(f"  frac beta>1  : {100*frac_fail:.1f}%")
    print(f"  B00 needed   : [{B00_needed.min():.2e}, {B00_needed.max():.2e}] G")
    print(f"  → need B00 ≥ {B00_needed.max():.2e} G everywhere  (you have {B00:.2e})")
    print(f"  f(r) range   : [{f_structural.min():.2e}, {f_structural.max():.2e}]")
    print(f"  threshold    : {threshold:.2e}  → {'NEVER reachable with this Sigma0/B00²' if f_structural.min() > threshold else 'reachable at some r'}")

    plt.tight_layout()
    plt.savefig('beta_diagnosis.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('beta_diagnosis.png', bbox_inches='tight', dpi=150)
    print("  -> beta_diagnosis.pdf / .png")
    plt.show()
    return fig