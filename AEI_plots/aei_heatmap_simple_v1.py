"""
aei_heatmap_simple_v1.py
========================
Heatmap of AEI solutions in the parameter space (B00, Sigma0)
for the Simple-v1 model (alpha_B=5/4, alpha_S=3/5).

Each grid cell is coloured by the median radius of the valid AEI
solutions found in that cell. Cells with no solutions are left blank.

4 panels (2x2):
  - Baseline        : only k > 0 and finite (no physical checks)
  - WKB check only  : 1 <= k < r/h  (correct WKB condition)
  - beta check only : beta <= 1
  - All checks      : WKB + beta + shear (dQ/dr > 0)

Note: the correct WKB range is 1 <= k < r/h = 1/HOR.
      k_min = 1   (WKB condition: k_phys * r >> 1, i.e. k >> 1)
      k_max = r/h = 1/HOR = 20  (for HOR = 0.05)

Usage:
    python aei_heatmap_simple_v1.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# ── project path ──────────────────────────────────────────────────────────────
# Adjust if needed
sys.path.append('..')   # project root directory

from setup import M_BH, NU0, r_isco, r_horizon, nu_phi, nu_r, Rg_SUN
from AEI_setups.aei_common import (
    solve_k_aei, check_k_wkb, compute_beta, compute_dQdr,
    HOR, mm, _make_interp
)
from aei_2.simple_disc import disk_model_simple

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

ALP_B_v1 = 5/4
ALP_S_v1 = 3/5

# Spin grid (same 19 points as in the notebook)
A_GRID = np.linspace(-0.99, 0.99, 19)

# Dense (B00, Sigma0) grid — log-spaced
N_B00    = 50   # points along B00
N_SIGMA0 = 45   # points along Sigma0

B00_RANGE    = (1e1, 1e8)    # [G]
SIGMA0_RANGE = (1e2, 1e7)    # [g/cm^2]

B00_GRID    = np.logspace(np.log10(B00_RANGE[0]),    np.log10(B00_RANGE[1]),    N_B00)
SIGMA0_GRID = np.logspace(np.log10(SIGMA0_RANGE[0]), np.log10(SIGMA0_RANGE[1]), N_SIGMA0)

# Radial grid
N_R   = 300
R_MAX = 1000.0   # r_g

# Correct WKB range: 1 <= k < r/h = 1/HOR
K_MIN_WKB = 1.0
K_MAX_WKB = 1.0 / HOR   # = 20 for HOR = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# MAIN SCAN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def disk_model_v1(r_rg, a, B00, Sigma0):
    """Simple-v1 wrapper with fixed power-law exponents."""
    return disk_model_simple(r_rg, a, B00, Sigma0,
                             alpha_B=ALP_B_v1, alpha_S=ALP_S_v1,
                             hr=HOR, M=M_BH)


def scan_grid(check_k=False, check_beta=False, check_shear=False,
              k_min=K_MIN_WKB, k_max=K_MAX_WKB):
    """
    Scan the (B00, Sigma0) grid over all spins in A_GRID.
    For each cell (B00, Sigma0) the valid radii from every spin value
    are pooled together; the median radius is computed over that union.
    Returns an (N_B00 x N_SIGMA0) matrix of median radii
    (NaN where no valid solution was found for any spin).
    """
    # Accumulate valid radii for each cell
    r_accum = [[[] for _ in range(N_SIGMA0)] for _ in range(N_B00)]

    total = len(A_GRID) * N_B00 * N_SIGMA0
    count = 0

    for a_val in A_GRID:
        # radial grid adapted to the ISCO of this spin
        r_vec = np.geomspace(r_isco(a_val) * 1.01, R_MAX, N_R)

        for i, B00 in enumerate(B00_GRID):
            for j, Sigma0 in enumerate(SIGMA0_GRID):
                count += 1
                if count % 2000 == 0:
                    print(f"  {count}/{total}  ({100*count/total:.0f}%)", end='\r')

                result = disk_model_v1(r_vec, a_val, B00, Sigma0)
                B0_arr, Sigma_arr, cs_arr, hr_arr, zone_arr, info = result

                k_arr = solve_k_aei(r_vec, a_val, B0_arr, Sigma_arr, cs_arr,
                                     m=mm, M=M_BH)

                # baseline mask: finite, positive k
                mask = np.isfinite(k_arr) & (k_arr > 0)
                if not np.any(mask):
                    continue

                # WKB check
                if check_k:
                    mask &= (k_arr >= k_min) & (k_arr <= k_max)
                    if not np.any(mask):
                        continue

                # beta check
                if check_beta:
                    beta_arr = compute_beta(B0_arr, Sigma_arr, cs_arr,
                                            r_vec, HOR, M_BH)
                    mask &= (beta_arr <= 1.0)
                    if not np.any(mask):
                        continue

                # shear check (dQ/dr > 0)
                if check_shear:
                    _B0_interp    = _make_interp(r_vec, B0_arr)
                    _Sigma_interp = _make_interp(r_vec, Sigma_arr)
                    dQdr_arr = compute_dQdr(r_vec, a_val,
                                            _B0_interp, _Sigma_interp, M_BH)
                    mask &= (dQdr_arr > 0)
                    if not np.any(mask):
                        continue

                r_accum[i][j].extend(r_vec[mask].tolist())

    print()

    # compute median radius for each cell
    r_median = np.full((N_B00, N_SIGMA0), np.nan)
    for i in range(N_B00):
        for j in range(N_SIGMA0):
            if r_accum[i][j]:
                r_median[i, j] = np.median(r_accum[i][j])

    return r_median


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_heatmap(axes_flat, data_list, titles):
    """
    Draw 4 heatmap panels.
    data_list : list of 4 matrices (N_B00 x N_SIGMA0)
    """
    # shared colour range across all panels
    all_vals = np.concatenate([d[np.isfinite(d)] for d in data_list])
    if all_vals.size == 0:
        print("WARNING: no valid solutions found in any panel.")
        return

    vmin, vmax = all_vals.min(), all_vals.max()
    norm = LogNorm(vmin=max(vmin, 1.0), vmax=vmax)
    cmap = plt.cm.plasma

    for ax, data, title in zip(axes_flat, data_list, titles):
        im = ax.pcolormesh(
            SIGMA0_GRID, B00_GRID, data,
            norm=norm, cmap=cmap,
            shading='auto',
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\Sigma_0$  [g/cm$^2$]', fontsize=11)
        ax.set_ylabel(r'$B_{00}$  [G]', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which='both', alpha=0.15, lw=0.5)

        # fraction of cells with a valid solution
        n_valid = np.sum(np.isfinite(data))
        n_total = data.size
        ax.text(0.97, 0.03,
                f'{n_valid}/{n_total} valid cells ({100*n_valid/n_total:.1f}%)',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

        cb = plt.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(r'$r_{\rm med}$ [$r_g$]', fontsize=9)
        ax.plot(1e4, 1e4, 'x', color='green', ms=10, mew=2, zorder=5, label="Reference values")
        ax.axhline(1e4, color='green', lw=1.8, ls='--',)
        ax.legend(fontsize=8, loc='lower left')

    return im, norm


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 65)
    print("  AEI Heatmap -- Simple-v1   (alpha_B=5/4, alpha_S=3/5)")
    print(f"  Spin: {len(A_GRID)} values in [{A_GRID[0]:.1f}, {A_GRID[-1]:.1f}]")
    print(f"  Grid: {N_B00} x {N_SIGMA0} = {N_B00*N_SIGMA0} cells  x  {len(A_GRID)} spins")
    print(f"  WKB check: {K_MIN_WKB:.1f} <= k < {K_MAX_WKB:.1f}  (= 1/HOR)")
    print("=" * 65)

    panels = [
        # (label, check_k, check_beta, check_shear)
        ("Baseline (k > 0, finite only)",                    False, False, False),
        (fr"WKB check only (${K_MIN_WKB:.0f} \leq k \leq r/H = {K_MAX_WKB:.0f}$)",
                                                               True,  False, False),
        (r"$\beta$ check only ($\beta$ <= 1)",                      False, True,  False),
        ("All checks",                  True,  True,  True),
    ]

    data_list = []
    for label, ck, cb, cs in panels:
        label_clean = label.replace('\n', ' ')
        print(f"\n>  {label_clean}")
        data = scan_grid(check_k=ck, check_beta=cb, check_shear=cs)
        data_list.append(data)
        n_valid = np.sum(np.isfinite(data))
        print(f"   -> {n_valid} cells with valid AEI solutions")

    print("\n>  Generating figure...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes_flat = axes.flatten()
    titles = [p[0] for p in panels]

    plot_heatmap(axes_flat, data_list, titles)

    """
    fig.suptitle(
        r"AEI solutions $-$ Simple-v1  ($\alpha_B = 5/4$, $\alpha_\Sigma = 3/5$)"
        "\n"
        f"Parameter space $(B_{{00}},\\, \\Sigma_0)$  --  "
        f"spin $a \\in [{A_GRID[0]:.2f},\\, {A_GRID[-1]:.2f}]$ --  "
        r"colour = median $r$ of valid solutions [$r_g$]",
        fontsize=12, y=1.01
    )
    """

    plt.tight_layout()
    plt.savefig('aei_heatmap_simple_v1.pdf', bbox_inches='tight', dpi=150)
    plt.savefig('aei_heatmap_simple_v1.png', bbox_inches='tight', dpi=150)
    print("   -> saved aei_heatmap_simple_v1.pdf / .png")
    plt.show()
