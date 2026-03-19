"""
nt_radial_profiles_diagnostic.py
=================================
Radial profiles of B0, Sigma, H/r, beta for the NT disk model
at fixed spin, mdot, alpha.

Vertical markers:
  - r_ISCO          (black solid)
  - r_ILR           (cyan dashed)
  - r_OLR           (magenta dash-dot)
  - r_AB            (orange dashed)
  - r_BC            (green dashed)
  - Q_AEI min       (red dashed)   ← r_match
  - Q_AEI max       (orange dashed)

Run from the project root:
    python nt_radial_profiles_diagnostic.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

from setup import r_isco, M_BH, nu_phi, Rg_SUN
from AEI_setups.aei_common import (
    r_ilr, r_olr,
    compute_beta,
    ALPHA_VISC, HOR, mm,
)
from AEI_setups.nt_disc import disk_model_NT, nt_boundaries

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

A_SPIN     = 0.9
MDOT       = 0.02       # mdot = Mdot / Mdot_Edd
ALPHA      = 0.1
M          = M_BH

N_R        = 600
R_MIN_FAC  = 1.001     # r_min = r_ISCO * R_MIN_FAC
R_MAX_FAC  = 250.0      # r_max = r_ISCO * R_MAX_FAC

OUTFILE    = 'nt_radial_profiles_diagnostic'

ZONE_COLOR = {'A': '#3b82f6', 'B': '#f97316', 'C': '#22c55e'}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: find Q_AEI extrema from sign changes of dQ/dr
# ══════════════════════════════════════════════════════════════════════════════

def find_Q_aei_extrema(r, Sigma, B0, a, M=M_BH, dr_frac=0.01):
    """
    Compute Q_AEI = Omega * Sigma / B0^2 and locate its local min and max
    via sign changes of dQ_AEI/dr.

    Returns
    -------
    Q_aei   : ndarray
    r_max_Q : float or None   (r of local maximum, closer to ISCO)
    r_min_Q : float or None   (r of local minimum, the r_match candidate)
    """
    Omega = 2.0 * np.pi * nu_phi(r, a, M)
    Q_aei = Omega * Sigma / np.maximum(B0**2, 1e-300)

    # finite-difference dQ/dr
    dr      = dr_frac * r
    rp      = r + dr

    # interpolate Q_aei at r+dr using log-linear interp on the array
    log_r   = np.log(r)
    log_Q   = np.log(np.maximum(Q_aei, 1e-300))
    Q_at_rp = np.exp(np.interp(np.log(rp), log_r, log_Q))
    dQdr    = (Q_at_rp - Q_aei) / dr

    sign_ch = np.where(np.diff(np.sign(dQdr)))[0]

    # local max: first change + → -
    i_max = next((i for i in sign_ch if dQdr[i] > 0), None)
    # local min: first change - → + (after the max)
    i_min = next((i for i in sign_ch if dQdr[i] < 0), None)

    r_max_Q = float(r[i_max]) if i_max is not None else None
    r_min_Q = float(r[i_min]) if i_min is not None else None

    return Q_aei, r_max_Q, r_min_Q


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_profiles(a=A_SPIN, mdot=MDOT, alpha=ALPHA, M=M_BH):
 
    rISCO = float(r_isco(a))
    r_vec = np.geomspace(rISCO * R_MIN_FAC, rISCO * R_MAX_FAC, N_R)
 
    # ── disk profiles ────────────────────────────────────────────────────────
    result = disk_model_NT(r_vec, a, mdot, alpha_visc=alpha, hr=None, M=M)
    B0_arr, Sig_arr, cs_arr, hr_arr, zone_arr, info = result
 
    r_AB = info['r_AB']
    r_BC = info['r_BC']
 
    # physical H in cm → H/r adimensional
    Rg     = Rg_SUN * M
    H_cm   = hr_arr * r_vec * Rg
    hr_phys = hr_arr   # already H/r from model
 
    # beta
    beta_arr = compute_beta(B0_arr, Sig_arr, cs_arr, r_vec, hr_arr, M)
 
    # ── resonance radii ──────────────────────────────────────────────────────
    r_ILR = r_ilr(a, M=M)
    r_OLR = r_olr(a, M=M)
 
    # ── Q_AEI extrema ────────────────────────────────────────────────────────
    Q_aei, r_max_Q, r_min_Q = find_Q_aei_extrema(r_vec, Sig_arr, B0_arr, a, M)
 
    # ── collect vertical markers ─────────────────────────────────────────────
    markers = [
        (rISCO,   'black',   '-',  r'$r_{\rm ISCO}$',         2.0),
    ]
    if np.isfinite(r_ILR):
        markers.append((r_ILR, '#06b6d4', '--', r'$r_{\rm ILR}$',  1.8))
    if np.isfinite(r_OLR):
        markers.append((r_OLR, '#d946ef', '-.', r'$r_{\rm OLR}$',  1.8))
    if r_AB > rISCO * 1.01:
        markers.append((r_AB,  '#f97316', '--', r'$r_{AB}$',        1.5))
    if r_BC > r_AB * 1.01:
        markers.append((r_BC,  '#22c55e', '--', r'$r_{BC}$',        1.5))

    """
    if r_max_Q is not None:
        markers.append((r_max_Q, 'orange', '--',
                        r'$Q_{\rm AEI}$ max', 1.5))
    if r_min_Q is not None:
        markers.append((r_min_Q, 'red', '--',
                        r'$Q_{\rm AEI}$ min', 1.5))
    """
 
    # ── zone background ──────────────────────────────────────────────────────
    def shade_zones(ax):
        zones_def = [
            ('A', rISCO,  r_AB),
            ('B', r_AB,   r_BC),
            ('C', r_BC,   r_vec[-1]),
        ]
        for zname, zlo, zhi in zones_def:
            if zhi > zlo:
                ax.axvspan(zlo, zhi, alpha=0.06,
                           color=ZONE_COLOR.get(zname, 'gray'),
                           label=f'Zone {zname}')
 
    def add_markers(ax):
        for rv, col, ls, lbl, lw in markers:
            ax.axvline(rv, color=col, ls=ls, lw=lw, alpha=0.85, label=lbl)
 
    # ── figure: 4 stacked panels, shared x-axis ──────────────────────────────
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
 
    panels = [
        (B0_arr,   r'$B_0$  [G]',                True,  []),
        (Sig_arr,  r'$\Sigma$  [g cm$^{-2}$]',   True,  []),
        (hr_phys,  r'$H/r$',                     True,
                   [(0.1, 'steelblue', ':', 'H/r = 0.1'),
                    (0.3, 'crimson',   ':', 'H/r = 0.3')]),
        (beta_arr, r'$\beta$',                   True,
                   [(1.0, 'crimson', '--', r'$\beta = 1$')]),
    ]
 
    fig, axes = plt.subplots(
        4, 1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw=dict(hspace=0.06),
    )
    """
    fig.suptitle(
        f'NT disk — a = {a},  ṁ = {mdot:.3g},  α = {alpha},  '
        f'M = {M:.2e} M$_\\odot$',
        fontsize=10, y=1.002,
    )
    """
 
    for k, (ax, (data, ylabel, log_y, hlines)) in enumerate(zip(axes, panels)):
        shade_zones(ax)
        add_markers(ax)
 
        for zname, col in ZONE_COLOR.items():
            mask  = zone_arr == zname
            valid = mask & (data > 0) if log_y else mask
            if valid.sum() < 2:
                continue
            ax.plot(r_vec[valid], data[valid], color=col, lw=2.0, zorder=3)
 
        for yval, hcol, hls, hlbl in hlines:
            ax.axhline(yval, color=hcol, ls=hls, lw=1.4, alpha=0.85)
 
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.15, which='both')
        ax.set_xlim(r_vec[0], r_vec[-1])
 
        # x tick labels only on the bottom panel
        if k < 3:
            ax.tick_params(axis='x', labelbottom=False)
 
    axes[-1].set_xlabel(r'$r$  [$r_g$]', fontsize=11)
 
    # ── legend inside the top panel ──────────────────────────────────────────
    legend_els = []
    for rv, col, ls, lbl, lw in markers:
        legend_els.append(Line2D([0], [0], color=col, ls=ls, lw=lw, label=lbl))
    for zname, col in ZONE_COLOR.items():
        legend_els.append(Patch(facecolor=col, alpha=0.35, label=f'Zone {zname}'))
 
    axes[0].legend(
        handles=legend_els,
        fontsize=7.5, ncol=2,
        loc='upper right',
        framealpha=0.88,
        borderpad=0.5,
        labelspacing=0.25,
        handlelength=1.6,
    )
 
    #plt.tight_layout()
    plt.savefig(OUTFILE+".png", dpi=150, bbox_inches='tight')
    plt.savefig(OUTFILE+".pdf", dpi=150, bbox_inches='tight')
    print(f'Saved → {OUTFILE}')
 
    # ── numeric summary ───────────────────────────────────────────────────────
    print(f'\nKey radii  (a={a}, mdot={mdot:.3g}, alpha={alpha}):')
    print(f'  r_ISCO  = {rISCO:.3f} rg')
    print(f'  r_ILR   = {r_ILR:.3f} rg' if np.isfinite(r_ILR) else '  r_ILR  : not found')
    print(f'  r_OLR   = {r_OLR:.3f} rg' if np.isfinite(r_OLR) else '  r_OLR  : not found')
    print(f'  r_AB    = {r_AB:.3f} rg')
    print(f'  r_BC    = {r_BC:.3f} rg')
    if r_max_Q is not None:
        print(f'  Q max   = {r_max_Q:.3f} rg  ({r_max_Q/rISCO:.3f} rISCO)')
    else:
        print('  Q max   : not found in range')
    if r_min_Q is not None:
        print(f'  Q min   = {r_min_Q:.3f} rg  ({r_min_Q/rISCO:.3f} rISCO)')
    else:
        print('  Q min   : not found in range')
 
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    plot_profiles(a=A_SPIN, mdot=MDOT, alpha=ALPHA, M=M_BH)
